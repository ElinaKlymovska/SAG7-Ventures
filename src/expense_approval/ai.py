from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from expense_approval.models import AssessmentSource, ExpenseClaim

logger = logging.getLogger(__name__)

AI_INSTRUCTIONS = """You assist a human expense approver. Summarize the claim in one or two
short sentences and identify only clear inconsistencies between amount, category, and description.
Never recommend approval or rejection. Never invent policy, receipts, names, or payment details.
Keep reasons short and return an empty reasons list when the claim is internally consistent."""


class StructuredAssessment(BaseModel):
    summary: str = Field(min_length=1, max_length=350)
    is_inconsistent: bool
    reasons: list[str] = Field(default_factory=list, max_length=3)


@dataclass(frozen=True)
class AssessmentResult:
    summary: str
    is_inconsistent: bool
    reasons: list[str]
    source: AssessmentSource
    model: str
    latency_ms: int
    error_message: str | None = None


def build_ai_payload(claim: ExpenseClaim) -> dict[str, str]:
    """Return the complete and intentionally minimal payload sent to the AI provider."""
    return {
        "amount_usd": f"{claim.amount_cents / 100:.2f}",
        "category": claim.category.name,
        "description": claim.description,
        "expense_date": claim.expense_date.isoformat(),
    }


def payload_hash(payload: dict[str, str]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExpenseAnalyzer:
    def __init__(self, api_key: str | None, model: str, timeout_seconds: float = 4.0):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def openai_enabled(self) -> bool:
        return bool(self.api_key)

    def analyze(self, payload: dict[str, str]) -> AssessmentResult:
        if not self.api_key:
            return rule_based_assessment(payload, "OPENAI_API_KEY is not configured.")

        started = perf_counter()
        try:
            client = OpenAI(api_key=self.api_key, max_retries=0, timeout=self.timeout_seconds)
            response = client.responses.parse(
                model=self.model,
                instructions=AI_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=StructuredAssessment,
                max_output_tokens=250,
                store=False,
                timeout=self.timeout_seconds,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("The model returned no structured assessment.")
            latency_ms = int((perf_counter() - started) * 1000)
            reasons = [reason.strip() for reason in parsed.reasons if reason.strip()][:3]
            is_inconsistent = bool(parsed.is_inconsistent or reasons)
            return AssessmentResult(
                summary=parsed.summary.strip(),
                is_inconsistent=is_inconsistent,
                reasons=reasons,
                source=AssessmentSource.OPENAI,
                model=self.model,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            safe_error = _safe_error(exc)
            logger.warning("OpenAI assessment failed; using fallback: %s", safe_error)
            fallback = rule_based_assessment(payload, safe_error)
            return AssessmentResult(
                summary=fallback.summary,
                is_inconsistent=fallback.is_inconsistent,
                reasons=fallback.reasons,
                source=fallback.source,
                model=fallback.model,
                latency_ms=int((perf_counter() - started) * 1000),
                error_message=fallback.error_message,
            )


def rule_based_assessment(
    payload: dict[str, str], error_message: str | None = None
) -> AssessmentResult:
    category = payload["category"]
    description = payload["description"].strip()
    normalized = f" {description.lower()} "
    amount = _safe_float(payload["amount_usd"])
    reasons: list[str] = []

    keyword_conflicts: dict[str, tuple[set[str], str]] = {
        "Office": (
            {"flight", "airfare", "hotel", "train", "taxi", "uber", "restaurant", "dinner"},
            "The description appears travel- or entertainment-related, not office-related.",
        ),
        "Travel": (
            {"desk", "chair", "keyboard", "printer", "stationery", "subscription", "license"},
            "The description appears office- or software-related, not travel-related.",
        ),
        "Client Entertainment": (
            {"desk", "chair", "printer", "subscription", "license", "airfare", "hotel"},
            "The description does not appear related to client entertainment.",
        ),
        "Software/Subscriptions": (
            {"flight", "airfare", "hotel", "taxi", "dinner", "restaurant", "chair", "desk"},
            "The description appears unrelated to software or subscriptions.",
        ),
    }
    keywords, message = keyword_conflicts.get(category, (set(), ""))
    if any(_contains_term(normalized, term) for term in keywords):
        reasons.append(message)

    amount_thresholds = {
        "Office": 2_500,
        "Travel": 10_000,
        "Client Entertainment": 5_000,
        "Software/Subscriptions": 5_000,
        "Other": 2_500,
    }
    threshold = amount_thresholds.get(category)
    if threshold is not None and amount > threshold:
        reasons.append(
            f"The ${amount:,.2f} amount is unusually high for the selected {category} category."
        )

    vague_terms = {"misc", "miscellaneous", "stuff", "expense", "other things"}
    if description.lower() in vague_terms or len(description.split()) < 3:
        reasons.append("The description is too vague to confirm the selected category.")

    reasons = reasons[:3]
    clipped_description = description if len(description) <= 120 else f"{description[:117]}..."
    summary = (
        f"{category} claim for ${amount:,.2f}, dated {payload['expense_date']}, "
        f"for: {clipped_description}"
    )
    return AssessmentResult(
        summary=summary,
        is_inconsistent=bool(reasons),
        reasons=reasons,
        source=AssessmentSource.FALLBACK,
        model="rule-engine-v1",
        latency_ms=0,
        error_message=error_message,
    )


def _contains_term(text: str, term: str) -> bool:
    return f" {term} " in text or f" {term}s " in text


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_error(exc: Exception) -> str:
    name = type(exc).__name__
    message = str(exc).strip().replace("\n", " ")
    return f"{name}: {message[:160]}" if message else name
