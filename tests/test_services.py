from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal

import pytest

from expense_approval.models import ExpenseStatus, RoleName
from expense_approval.services import (
    AccessDeniedError,
    ExpenseService,
    TransitionError,
    ValidationError,
)

EMPLOYEE = "employee@expense-demo.local"
APPROVER = "approver@expense-demo.local"
DUAL = "dual@expense-demo.local"
SECOND_EMPLOYEE = "second.employee@expense-demo.local"
MISMATCH_DESCRIPTION = "Round-trip flight to London for a partner meeting"
DUAL_DESCRIPTION = "Dinner with the Acme client team after the quarterly review"


def test_authentication_and_dual_roles(service: ExpenseService) -> None:
    employee = service.authenticate(EMPLOYEE.upper(), "Demo123!")
    dual = service.authenticate(DUAL, "Demo123!")

    assert employee is not None
    assert employee.roles == {RoleName.EMPLOYEE}
    assert dual is not None
    assert dual.roles == {RoleName.EMPLOYEE, RoleName.APPROVER}
    assert service.authenticate(EMPLOYEE, "wrong") is None


def test_create_claim_routes_to_category_approver(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim = service.create_claim(
        ids[EMPLOYEE],
        amount=Decimal("123.45"),
        category_id=ids["Office"],
        description="A monitor arm for the design workstation",
        expense_date=date.today(),
        payment_details="Demo account ending in 9912",
    )

    assert claim.amount_cents == 12_345
    assert claim.currency == "USD"
    assert claim.status == ExpenseStatus.PENDING
    assert claim.category.approver.email == APPROVER


@pytest.mark.parametrize(
    ("amount", "description", "payment_details", "expected"),
    [
        ("10.001", "A valid description", "Demo account 123", "two decimal"),
        ("0.00", "A valid description", "Demo account 123", "between"),
        ("10.00", "short", "Demo account 123", "at least 10"),
        ("10.00", "A valid description", "x", "at least 5"),
    ],
)
def test_claim_validation(
    service: ExpenseService,
    ids: dict[str, int],
    amount: str,
    description: str,
    payment_details: str,
    expected: str,
) -> None:
    with pytest.raises(ValidationError, match=expected):
        service.create_claim(
            ids[EMPLOYEE],
            amount=amount,
            category_id=ids["Office"],
            description=description,
            expense_date=date.today(),
            payment_details=payment_details,
        )


def test_employee_data_isolation(service: ExpenseService, ids: dict[str, int]) -> None:
    employee_claims = service.list_employee_claims(ids[EMPLOYEE])
    second_claims = service.list_employee_claims(ids[SECOND_EMPLOYEE])

    assert employee_claims
    assert second_claims
    assert {claim.employee_id for claim in employee_claims} == {ids[EMPLOYEE]}
    assert {claim.employee_id for claim in second_claims} == {ids[SECOND_EMPLOYEE]}
    with pytest.raises(AccessDeniedError):
        service.get_claim_for_employee(ids[SECOND_EMPLOYEE], employee_claims[0].id)


def test_approver_only_sees_assigned_categories(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    primary_queue = service.list_approval_queue(ids[APPROVER])
    dual_queue = service.list_approval_queue(ids[DUAL])

    assert primary_queue
    assert dual_queue
    assert {claim.category.name for claim in primary_queue} <= {
        "Office",
        "Travel",
        "Software/Subscriptions",
    }
    assert {claim.category.name for claim in dual_queue} <= {"Client Entertainment", "Other"}
    with pytest.raises(AccessDeniedError):
        service.get_claim_for_approver(ids[DUAL], ids[MISMATCH_DESCRIPTION])


def test_rejection_requires_comment(service: ExpenseService, ids: dict[str, int]) -> None:
    claim_id = ids[MISMATCH_DESCRIPTION]
    with pytest.raises(ValidationError, match="[Rr]ejection comment"):
        service.decide_claim(ids[APPROVER], claim_id, ExpenseStatus.REJECTED, "  ")

    rejected = service.decide_claim(
        ids[APPROVER], claim_id, ExpenseStatus.REJECTED, "Please use the Travel category."
    )
    assert rejected.status == ExpenseStatus.REJECTED
    assert rejected.decision_comment == "Please use the Travel category."
    with pytest.raises(TransitionError):
        service.decide_claim(ids[APPROVER], claim_id, ExpenseStatus.APPROVED)


def test_rejection_comment_must_be_meaningful(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    with pytest.raises(ValidationError, match="at least 10 characters"):
        service.decide_claim(
            ids[APPROVER], ids[MISMATCH_DESCRIPTION], ExpenseStatus.REJECTED, "no"
        )


def test_approval_still_allows_empty_comment(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim_id = ids["Ergonomic keyboard and mouse for the home office"]
    approved = service.decide_claim(ids[APPROVER], claim_id, ExpenseStatus.APPROVED, "")
    assert approved.status is ExpenseStatus.APPROVED
    assert not approved.decision_comment


def test_owner_can_only_withdraw_pending_claim(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim_id = ids["Ergonomic keyboard and mouse for the home office"]
    withdrawn = service.withdraw_claim(ids[EMPLOYEE], claim_id)
    assert withdrawn.status == ExpenseStatus.WITHDRAWN
    with pytest.raises(TransitionError):
        service.withdraw_claim(ids[EMPLOYEE], claim_id)
    with pytest.raises(AccessDeniedError):
        service.withdraw_claim(ids[SECOND_EMPLOYEE], ids[MISMATCH_DESCRIPTION])


def test_self_approval_is_allowed_for_dual_role_user(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    decided = service.decide_claim(
        ids[DUAL], ids[DUAL_DESCRIPTION], ExpenseStatus.APPROVED, "Within demo policy."
    )
    assert decided.status == ExpenseStatus.APPROVED
    assert decided.employee_id == ids[DUAL]
    assert decided.decided_by_id == ids[DUAL]


def test_only_one_concurrent_decision_succeeds(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    claim_id = ids[MISMATCH_DESCRIPTION]

    def decide(status: ExpenseStatus) -> str:
        try:
            service.decide_claim(
                ids[APPROVER],
                claim_id,
                status,
                "A required reason" if status == ExpenseStatus.REJECTED else None,
            )
            return "success"
        except TransitionError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(decide, [ExpenseStatus.APPROVED, ExpenseStatus.REJECTED])
        )

    assert sorted(results) == ["conflict", "success"]

