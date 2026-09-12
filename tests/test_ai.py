from __future__ import annotations

from types import SimpleNamespace

from expense_approval.ai import (
    ExpenseAnalyzer,
    StructuredAssessment,
    build_ai_payload,
    payload_hash,
    rule_based_assessment,
)
from expense_approval.models import AssessmentSource
from expense_approval.services import ExpenseService

EMPLOYEE = "employee@expense-demo.local"
APPROVER = "approver@expense-demo.local"
MISMATCH_DESCRIPTION = "Round-trip flight to London for a partner meeting"
CONSISTENT_DESCRIPTION = "Ergonomic keyboard and mouse for the home office"


def test_ai_payload_excludes_identity_and_payment_details(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim = service.get_claim_for_approver(ids[APPROVER], ids[MISMATCH_DESCRIPTION])
    payload = build_ai_payload(claim)

    assert set(payload) == {"amount_usd", "category", "description", "expense_date"}
    assert payload["category"] == "Office"
    assert payload["description"] == MISMATCH_DESCRIPTION
    serialized = str(payload)
    assert claim.payment_details not in serialized
    assert claim.employee.email not in serialized


def test_rule_fallback_flags_mismatch_and_accepts_consistent_claim(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    mismatch = service.get_claim_for_approver(ids[APPROVER], ids[MISMATCH_DESCRIPTION])
    consistent = service.get_claim_for_approver(ids[APPROVER], ids[CONSISTENT_DESCRIPTION])

    mismatch_result = rule_based_assessment(build_ai_payload(mismatch))
    consistent_result = rule_based_assessment(build_ai_payload(consistent))

    assert mismatch_result.is_inconsistent is True
    assert "travel" in mismatch_result.reasons[0].lower()
    assert consistent_result.is_inconsistent is False
    assert consistent_result.reasons == []


def test_missing_key_uses_labeled_fallback(service: ExpenseService, ids: dict[str, int]) -> None:
    claim = service.get_claim_for_approver(ids[APPROVER], ids[MISMATCH_DESCRIPTION])
    result = ExpenseAnalyzer(None, "gpt-5-mini").analyze(build_ai_payload(claim))

    assert result.source == AssessmentSource.FALLBACK
    assert result.model == "rule-engine-v1"
    assert "OPENAI_API_KEY" in (result.error_message or "")


def test_openai_structured_response_is_parsed(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=StructuredAssessment(
                    summary="A concise summary.",
                    is_inconsistent=False,
                    reasons=[],
                )
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object):
            captured["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr("expense_approval.ai.OpenAI", FakeOpenAI)
    analyzer = ExpenseAnalyzer("test-key", "gpt-5-mini", 4.0)
    payload = {
        "amount_usd": "12.00",
        "category": "Office",
        "description": "A box of blue ballpoint pens",
        "expense_date": "2026-09-10",
    }
    result = analyzer.analyze(payload)

    assert result.source == AssessmentSource.OPENAI
    assert captured["model"] == "gpt-5-mini"
    assert captured["store"] is False
    assert captured["text_format"] is StructuredAssessment
    assert captured["reasoning"] == {"effort": "minimal"}
    assert captured["max_output_tokens"] == 200
    assert captured["client"] == {
        "api_key": "test-key",
        "max_retries": 0,
        "timeout": 4.0,
    }
    assert "payment_details" not in str(captured["input"])


def test_openai_client_is_reused_between_assessments(monkeypatch: object) -> None:
    client_creations = 0

    class FakeResponses:
        def parse(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                output_parsed=StructuredAssessment(
                    summary="A concise summary.",
                    is_inconsistent=False,
                    reasons=[],
                )
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs: object):
            nonlocal client_creations
            client_creations += 1
            self.responses = FakeResponses()

    monkeypatch.setattr("expense_approval.ai.OpenAI", FakeOpenAI)
    analyzer = ExpenseAnalyzer("test-key", "gpt-5-mini")
    payload = {
        "amount_usd": "12.00",
        "category": "Office",
        "description": "A box of blue ballpoint pens",
        "expense_date": "2026-09-10",
    }

    analyzer.analyze(payload)
    analyzer.analyze(payload)

    assert client_creations == 1


def test_api_failure_falls_back(monkeypatch: object) -> None:
    class FailingResponses:
        def parse(self, **_kwargs: object) -> None:
            raise TimeoutError("request timed out")

    class FailingOpenAI:
        def __init__(self, **_kwargs: object):
            self.responses = FailingResponses()

    monkeypatch.setattr("expense_approval.ai.OpenAI", FailingOpenAI)
    result = ExpenseAnalyzer("test-key", "gpt-5-mini").analyze(
        {
            "amount_usd": "780.00",
            "category": "Office",
            "description": MISMATCH_DESCRIPTION,
            "expense_date": "2026-09-10",
        }
    )

    assert result.source == AssessmentSource.FALLBACK
    assert result.is_inconsistent is True
    assert "TimeoutError" in (result.error_message or "")


def test_assessment_cache_is_keyed_by_claim_snapshot(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim = service.get_claim_for_approver(ids[APPROVER], ids[MISMATCH_DESCRIPTION])
    input_hash = payload_hash(build_ai_payload(claim))

    saved = service.save_assessment(
        claim_id=claim.id,
        input_hash=input_hash,
        summary="Cached summary",
        is_inconsistent=True,
        reasons=["Category mismatch"],
        source=AssessmentSource.FALLBACK,
        model="rule-engine-v1",
        latency_ms=3,
    )
    cached = service.get_cached_assessment(claim.id, input_hash)

    assert cached == saved
    assert service.get_cached_assessment(claim.id, "different") is None


def test_openai_result_upgrades_a_cached_fallback(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim = service.get_claim_for_approver(ids[APPROVER], ids[MISMATCH_DESCRIPTION])
    input_hash = payload_hash(build_ai_payload(claim))
    service.save_assessment(
        claim_id=claim.id,
        input_hash=input_hash,
        summary="Fallback summary",
        is_inconsistent=True,
        reasons=["Fallback reason"],
        source=AssessmentSource.FALLBACK,
        model="rule-engine-v1",
        latency_ms=0,
    )

    upgraded = service.save_assessment(
        claim_id=claim.id,
        input_hash=input_hash,
        summary="OpenAI summary",
        is_inconsistent=False,
        reasons=[],
        source=AssessmentSource.OPENAI,
        model="gpt-5-mini",
        latency_ms=120,
    )

    assert upgraded.source == AssessmentSource.OPENAI
    assert upgraded.summary == "OpenAI summary"
    assert upgraded.model == "gpt-5-mini"
