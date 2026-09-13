from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import date
from time import perf_counter
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from expense_approval.documents import (
    DocumentExtraction,
    DocumentKind,
    ProcessedDocument,
    parse_money,
    sanitize_document_text_for_ai,
)
from expense_approval.models import AssessmentSource, ExpenseClaim

logger = logging.getLogger(__name__)

AI_INSTRUCTIONS = """You assist a human expense approver. Summarize the claim in one or two
short sentences and identify only clear inconsistencies between amount, category, and description.
Never recommend approval or rejection. Never invent policy, receipts, names, or payment details.
Keep reasons short and return an empty reasons list when the claim is internally consistent."""

PAYMENT_DETAILS_INSTRUCTIONS = """Create one concise, non-sensitive reimbursement reference for
an employee expense claim. Describe what the reimbursement is for using only the supplied claim
fields. Never invent or request bank accounts, IBANs, card numbers, routing numbers, payment
methods, invoice numbers, people, or vendors. The result is an editable reference, not a payment
instruction. Use plain English and no more than one sentence."""

DOCUMENT_EXTRACTION_INSTRUCTIONS = """Extract expense-document fields from any language or
layout. Use only evidence visible in the supplied image or sanitized OCR text. Never invent a
vendor, number, amount, currency, or date; return null when absent or unreadable. original_total
must be the final payable total using a dot decimal separator and no currency symbol. Provide a
specific, unrestricted business category such as Professional Services, Electricity, Air Travel,
or Telecommunications. Separately choose exactly one routing_category from the configured list.
Treat invoices, receipts, and genuine travel evidence as expense evidence; reject resumes,
policies, blank forms, promotional material, and unrelated documents. Keep the description and
warnings concise."""


class StructuredAssessment(BaseModel):
    summary: str = Field(min_length=1, max_length=350)
    is_inconsistent: bool
    reasons: list[str] = Field(default_factory=list, max_length=3)


class StructuredPaymentSuggestion(BaseModel):
    payment_details: str = Field(min_length=5, max_length=200)


class StructuredDocumentExtraction(BaseModel):
    document_kind: Literal["invoice", "receipt", "boarding_pass", "other"]
    is_expense_evidence: bool
    vendor: str | None
    document_number: str | None
    original_total: str | None
    currency: str | None
    expense_date: str | None
    suggested_category: str = Field(min_length=1, max_length=100)
    routing_category: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=5, max_length=500)
    confidence_percent: int = Field(ge=0, le=100)
    warnings: list[str] = Field(max_length=4)


@dataclass(frozen=True)
class AssessmentResult:
    summary: str
    is_inconsistent: bool
    reasons: list[str]
    source: AssessmentSource
    model: str
    latency_ms: int
    error_message: str | None = None


@dataclass(frozen=True)
class PaymentSuggestionResult:
    payment_details: str
    source: AssessmentSource
    model: str
    latency_ms: int
    error_message: str | None = None


@dataclass(frozen=True)
class DocumentAnalysisResult:
    extraction: DocumentExtraction
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


def build_payment_details_payload(
    *, amount_usd: str, category: str, description: str, expense_date: str
) -> dict[str, str]:
    """Return the complete privacy-limited payload for an employee-facing suggestion."""
    return {
        "amount_usd": amount_usd,
        "category": category,
        "description": description,
        "expense_date": expense_date,
    }


class ExpenseAnalyzer:
    def __init__(self, api_key: str | None, model: str, timeout_seconds: float = 10.0):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        # Reuse one HTTP connection pool for the lifetime of the cached Streamlit
        # resource. Creating a client for every assessment adds DNS/TLS overhead
        # and made the previous four-second request budget unreliable.
        self._client = None
        self._client_error: str | None = None
        if api_key:
            try:
                self._client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout_seconds)
            except Exception as exc:
                # A broken SDK install or hostile proxy settings must degrade the AI
                # panel to the rule-based fallback, not take the whole app down.
                self._client_error = f"{type(exc).__name__}: {exc}"
                logger.warning("OpenAI client unavailable; using fallback: %s", self._client_error)

    @property
    def openai_enabled(self) -> bool:
        return self._client is not None

    def analyze(self, payload: dict[str, str]) -> AssessmentResult:
        if not self.api_key:
            return rule_based_assessment(payload, "OPENAI_API_KEY is not configured.")

        started = perf_counter()
        try:
            if self._client is None:
                raise RuntimeError("OpenAI client is unavailable.")
            response = self._client.responses.parse(
                model=self.model,
                instructions=AI_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=StructuredAssessment,
                reasoning={"effort": "minimal"},
                max_output_tokens=200,
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
            # Streamlit Community Cloud reliably captures stdout in its app logs.
            # Keep the diagnostic free of credentials and request payload data.
            print(f"OpenAI assessment fallback: {safe_error}", flush=True)
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

    def suggest_payment_details(
        self, payload: dict[str, str]
    ) -> PaymentSuggestionResult:
        if not self.api_key:
            return rule_based_payment_details(
                payload, "OPENAI_API_KEY is not configured."
            )

        started = perf_counter()
        try:
            if self._client is None:
                raise RuntimeError("OpenAI client is unavailable.")
            response = self._client.responses.parse(
                model=self.model,
                instructions=PAYMENT_DETAILS_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=StructuredPaymentSuggestion,
                reasoning={"effort": "minimal"},
                max_output_tokens=100,
                store=False,
                timeout=self.timeout_seconds,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("The model returned no payment-details suggestion.")
            suggestion = parsed.payment_details.strip()
            if not _is_safe_payment_suggestion(suggestion):
                raise ValueError("The model returned an unsafe payment-details suggestion.")
            return PaymentSuggestionResult(
                payment_details=suggestion,
                source=AssessmentSource.OPENAI,
                model=self.model,
                latency_ms=int((perf_counter() - started) * 1000),
            )
        except Exception as exc:
            safe_error = _safe_error(exc)
            logger.warning("OpenAI payment suggestion failed; using fallback: %s", safe_error)
            print(f"OpenAI payment suggestion fallback: {safe_error}", flush=True)
            fallback = rule_based_payment_details(payload, safe_error)
            return PaymentSuggestionResult(
                payment_details=fallback.payment_details,
                source=fallback.source,
                model=fallback.model,
                latency_ms=int((perf_counter() - started) * 1000),
                error_message=fallback.error_message,
            )

    def analyze_document(
        self, document: ProcessedDocument, routing_categories: list[str]
    ) -> DocumentAnalysisResult:
        if not self.api_key:
            return _document_analysis_fallback(
                document, "OPENAI_API_KEY is not configured."
            )

        started = perf_counter()
        try:
            if self._client is None:
                raise RuntimeError("OpenAI client is unavailable.")
            sanitized_text = sanitize_document_text_for_ai(
                getattr(document, "extracted_text", "")
            )
            context = json.dumps(
                {
                    "configured_routing_categories": routing_categories,
                    "sanitized_local_ocr_text": sanitized_text,
                },
                ensure_ascii=False,
            )
            content: list[dict[str, str]] = [
                {"type": "input_text", "text": context}
            ]
            uses_vision = document.mime_type.startswith("image/")
            if uses_vision:
                encoded = base64.b64encode(document.content).decode("ascii")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{document.mime_type};base64,{encoded}",
                    }
                )
            response = self._client.responses.parse(
                model=self.model,
                instructions=DOCUMENT_EXTRACTION_INSTRUCTIONS,
                input=[{"role": "user", "content": content}],
                text_format=StructuredDocumentExtraction,
                reasoning={"effort": "minimal"},
                max_output_tokens=600,
                store=False,
                timeout=self.timeout_seconds,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("The model returned no document extraction.")
            extraction = _merge_document_extraction(
                document.extraction,
                parsed,
                routing_categories,
                "OpenAI vision" if uses_vision else "OpenAI sanitized OCR analysis",
            )
            return DocumentAnalysisResult(
                extraction=extraction,
                source=AssessmentSource.OPENAI,
                model=self.model,
                latency_ms=int((perf_counter() - started) * 1000),
            )
        except Exception as exc:
            safe_error = _safe_error(exc)
            logger.warning("OpenAI document extraction failed; using local result: %s", safe_error)
            print(f"OpenAI document extraction fallback: {safe_error}", flush=True)
            fallback = _document_analysis_fallback(document, safe_error)
            return DocumentAnalysisResult(
                extraction=fallback.extraction,
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


def rule_based_payment_details(
    payload: dict[str, str], error_message: str | None = None
) -> PaymentSuggestionResult:
    amount = _safe_float(payload["amount_usd"])
    amount_phrase = f" for ${amount:,.2f} USD" if amount > 0 else ""
    payment_details = (
        f"Reimbursement reference: {payload['category']} expense dated "
        f"{payload['expense_date']}{amount_phrase}."
    )
    return PaymentSuggestionResult(
        payment_details=payment_details,
        source=AssessmentSource.FALLBACK,
        model="rule-engine-v1",
        latency_ms=0,
        error_message=error_message,
    )


def _merge_document_extraction(
    local: DocumentExtraction,
    parsed: StructuredDocumentExtraction,
    routing_categories: list[str],
    processing_method: str,
) -> DocumentExtraction:
    kind = (
        DocumentKind(parsed.document_kind)
        if parsed.document_kind != "other"
        else DocumentKind.OTHER
    )
    amount = parse_money(parsed.original_total or "") or local.amount
    currency = (parsed.currency or local.currency or "").strip().upper() or None
    extracted_date = _parse_iso_date(parsed.expense_date) or local.expense_date
    route_lookup = {category.casefold(): category for category in routing_categories}
    routing_category = route_lookup.get(parsed.routing_category.strip().casefold())
    if routing_category is None:
        routing_category = route_lookup.get("other") or next(iter(routing_categories), None)
    vendor = _clean_optional(parsed.vendor) or local.vendor
    document_number = _clean_optional(parsed.document_number) or local.document_number
    warnings = tuple(warning.strip() for warning in parsed.warnings if warning.strip())
    if vendor is None:
        warnings += ("Vendor is not visible in the document; do not guess it.",)
    return DocumentExtraction(
        kind=kind,
        is_expense_evidence=parsed.is_expense_evidence,
        vendor=vendor,
        document_number=document_number,
        amount=amount,
        currency=currency,
        expense_date=extracted_date,
        category_hint=parsed.suggested_category.strip(),
        description=parsed.description.strip(),
        confidence=parsed.confidence_percent / 100,
        processing_method=processing_method,
        warnings=warnings[:5],
        routing_category=routing_category,
    )


def _document_analysis_fallback(
    document: ProcessedDocument, error_message: str
) -> DocumentAnalysisResult:
    warning = "AI extraction was unavailable; the displayed fields come from local OCR rules."
    extraction = replace(
        document.extraction,
        warnings=tuple(dict.fromkeys((*document.extraction.warnings, warning))),
    )
    return DocumentAnalysisResult(
        extraction=extraction,
        source=AssessmentSource.FALLBACK,
        model="local-document-parser-v1",
        latency_ms=0,
        error_message=error_message,
    )


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _clean_optional(value: str | None) -> str | None:
    cleaned = value.strip() if value else ""
    return cleaned or None


def _contains_term(text: str, term: str) -> bool:
    return f" {term} " in text or f" {term}s " in text


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_safe_payment_suggestion(value: str) -> bool:
    normalized = value.casefold()
    sensitive_markers = (
        "account number",
        "bank account",
        "card number",
        "routing number",
        "swift code",
        "iban",
    )
    return 5 <= len(value) <= 200 and not any(
        marker in normalized for marker in sensitive_markers
    )


def _safe_error(exc: Exception) -> str:
    name = type(exc).__name__
    message = str(exc).strip().replace("\n", " ")
    return f"{name}: {message[:160]}" if message else name
