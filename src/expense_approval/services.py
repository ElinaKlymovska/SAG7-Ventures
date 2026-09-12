from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import Select, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload, selectinload

from expense_approval.db import Database
from expense_approval.models import (
    AIAssessment,
    AssessmentSource,
    Category,
    ExpenseClaim,
    ExpenseStatus,
    RoleName,
    User,
    UserRole,
    utc_now,
)


class ExpenseApprovalError(Exception):
    """Base exception safe to display in the UI."""


class ValidationError(ExpenseApprovalError):
    pass


class AccessDeniedError(ExpenseApprovalError):
    pass


class NotFoundError(ExpenseApprovalError):
    pass


class TransitionError(ExpenseApprovalError):
    pass


@dataclass(frozen=True)
class DashboardCounts:
    total: int
    pending: int
    approved: int
    rejected: int
    withdrawn: int


@dataclass(frozen=True)
class StoredAssessment:
    claim_id: int
    input_hash: str
    summary: str
    is_inconsistent: bool
    reasons: list[str]
    source: AssessmentSource
    model: str
    latency_ms: int


class ExpenseService:
    MAX_AMOUNT = Decimal("1000000.00")

    def __init__(self, database: Database, password_hasher: PasswordHasher | None = None):
        self.database = database
        self.password_hasher = password_hasher or PasswordHasher()

    @staticmethod
    def claim_query() -> Select[tuple[ExpenseClaim]]:
        return select(ExpenseClaim).options(
            joinedload(ExpenseClaim.employee),
            joinedload(ExpenseClaim.category).joinedload(Category.approver),
            joinedload(ExpenseClaim.decided_by),
            selectinload(ExpenseClaim.assessments),
        )

    def get_user(self, user_id: int) -> User:
        with self.database.session() as session:
            user = session.scalar(
                select(User)
                .where(User.id == user_id, User.is_active.is_(True))
                .options(selectinload(User.role_links), selectinload(User.assigned_categories))
            )
            if user is None:
                raise NotFoundError("User account is unavailable.")
            return user

    def authenticate(self, email: str, password: str) -> User | None:
        normalized_email = email.strip().lower()
        if not normalized_email or not password:
            return None
        with self.database.session() as session:
            user = session.scalar(
                select(User)
                .where(User.email == normalized_email, User.is_active.is_(True))
                .options(selectinload(User.role_links), selectinload(User.assigned_categories))
            )
            if user is None:
                return None
            try:
                self.password_hasher.verify(user.password_hash, password)
            except (VerifyMismatchError, InvalidHashError):
                return None
            return user

    def list_categories(self) -> list[Category]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(Category).options(joinedload(Category.approver)).order_by(Category.name)
                ).unique()
            )

    def create_claim(
        self,
        actor_id: int,
        *,
        amount: Decimal | str | float,
        category_id: int,
        description: str,
        expense_date: date,
        payment_details: str,
    ) -> ExpenseClaim:
        amount_cents = self._validate_amount(amount)
        clean_description = self._validate_text(description, "Description", 10, 500)
        clean_payment = self._validate_text(payment_details, "Payment details", 5, 500)
        if expense_date > date.today():
            raise ValidationError("Expense date cannot be in the future.")

        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.EMPLOYEE)
            category = session.get(Category, category_id)
            if category is None:
                raise ValidationError("Select a valid expense category.")
            claim = ExpenseClaim(
                employee_id=actor_id,
                category_id=category_id,
                amount_cents=amount_cents,
                currency="USD",
                description=clean_description,
                expense_date=expense_date,
                payment_details=clean_payment,
                status=ExpenseStatus.PENDING,
            )
            session.add(claim)
            session.flush()
            claim_id = claim.id

        return self.get_claim_for_employee(actor_id, claim_id)

    def list_employee_claims(self, actor_id: int) -> list[ExpenseClaim]:
        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.EMPLOYEE)
            query = (
                self.claim_query()
                .where(ExpenseClaim.employee_id == actor_id)
                .order_by(ExpenseClaim.created_at.desc(), ExpenseClaim.id.desc())
            )
            return list(session.scalars(query).unique())

    def list_approval_queue(self, actor_id: int) -> list[ExpenseClaim]:
        return self._list_approver_claims(actor_id, pending_only=True)

    def list_approver_history(self, actor_id: int) -> list[ExpenseClaim]:
        return self._list_approver_claims(actor_id, pending_only=False)

    def _list_approver_claims(self, actor_id: int, *, pending_only: bool) -> list[ExpenseClaim]:
        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.APPROVER)
            query = self.claim_query().join(Category).where(Category.approver_id == actor_id)
            if pending_only:
                query = query.where(ExpenseClaim.status == ExpenseStatus.PENDING)
            else:
                query = query.where(ExpenseClaim.status != ExpenseStatus.PENDING)
            query = query.order_by(ExpenseClaim.updated_at.desc(), ExpenseClaim.id.desc())
            return list(session.scalars(query).unique())

    def get_claim_for_employee(self, actor_id: int, claim_id: int) -> ExpenseClaim:
        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.EMPLOYEE)
            claim = session.scalar(
                self.claim_query().where(
                    ExpenseClaim.id == claim_id,
                    ExpenseClaim.employee_id == actor_id,
                )
            )
            if claim is None:
                raise AccessDeniedError("This claim is not available to you.")
            return claim

    def get_claim_for_approver(self, actor_id: int, claim_id: int) -> ExpenseClaim:
        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.APPROVER)
            claim = session.scalar(
                self.claim_query()
                .join(Category)
                .where(ExpenseClaim.id == claim_id, Category.approver_id == actor_id)
            )
            if claim is None:
                raise AccessDeniedError("This claim is not assigned to you.")
            return claim

    def withdraw_claim(self, actor_id: int, claim_id: int) -> ExpenseClaim:
        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.EMPLOYEE)
            owned = session.scalar(
                select(ExpenseClaim.id).where(
                    ExpenseClaim.id == claim_id,
                    ExpenseClaim.employee_id == actor_id,
                )
            )
            if owned is None:
                raise AccessDeniedError("This claim is not available to you.")
            timestamp = utc_now()
            result = session.execute(
                update(ExpenseClaim)
                .where(
                    ExpenseClaim.id == claim_id,
                    ExpenseClaim.employee_id == actor_id,
                    ExpenseClaim.status == ExpenseStatus.PENDING,
                )
                .values(
                    status=ExpenseStatus.WITHDRAWN,
                    withdrawn_at=timestamp,
                    updated_at=timestamp,
                )
            )
            if result.rowcount != 1:
                raise TransitionError("Only a pending claim can be withdrawn.")
        return self.get_claim_for_employee(actor_id, claim_id)

    def decide_claim(
        self,
        actor_id: int,
        claim_id: int,
        decision: ExpenseStatus,
        comment: str | None = None,
    ) -> ExpenseClaim:
        if decision not in {ExpenseStatus.APPROVED, ExpenseStatus.REJECTED}:
            raise ValidationError("Decision must be approved or rejected.")
        clean_comment = (comment or "").strip()
        if decision == ExpenseStatus.REJECTED and not clean_comment:
            raise ValidationError("A rejection comment is required.")
        if len(clean_comment) > 500:
            raise ValidationError("Decision comment must be 500 characters or fewer.")

        with self.database.session() as session:
            self._require_role(session, actor_id, RoleName.APPROVER)
            assigned = session.scalar(
                select(ExpenseClaim.id)
                .join(Category)
                .where(ExpenseClaim.id == claim_id, Category.approver_id == actor_id)
            )
            if assigned is None:
                raise AccessDeniedError("This claim is not assigned to you.")
            timestamp = utc_now()
            result = session.execute(
                update(ExpenseClaim)
                .where(
                    ExpenseClaim.id == claim_id,
                    ExpenseClaim.status == ExpenseStatus.PENDING,
                )
                .values(
                    status=decision,
                    decision_comment=clean_comment or None,
                    decided_by_id=actor_id,
                    decided_at=timestamp,
                    updated_at=timestamp,
                )
            )
            if result.rowcount != 1:
                raise TransitionError("This claim has already been resolved or withdrawn.")
        return self.get_claim_for_approver(actor_id, claim_id)

    def get_cached_assessment(self, claim_id: int, input_hash: str) -> StoredAssessment | None:
        with self.database.session() as session:
            assessment = session.scalar(
                select(AIAssessment).where(
                    AIAssessment.claim_id == claim_id,
                    AIAssessment.input_hash == input_hash,
                )
            )
            return self._stored_assessment(assessment) if assessment else None

    def save_assessment(
        self,
        *,
        claim_id: int,
        input_hash: str,
        summary: str,
        is_inconsistent: bool,
        reasons: list[str],
        source: AssessmentSource,
        model: str,
        latency_ms: int,
    ) -> StoredAssessment:
        values = {
            "summary": summary[:500],
            "is_inconsistent": is_inconsistent,
            "reasons_json": json.dumps(reasons[:3]),
            "source": source,
            "model": model[:100],
            "latency_ms": max(0, latency_ms),
        }
        try:
            with self.database.session() as session:
                existing = session.scalar(
                    select(AIAssessment).where(
                        AIAssessment.claim_id == claim_id,
                        AIAssessment.input_hash == input_hash,
                    )
                )
                if existing is not None:
                    if (
                        existing.source == AssessmentSource.FALLBACK
                        and source == AssessmentSource.OPENAI
                    ):
                        for field, value in values.items():
                            setattr(existing, field, value)
                        session.flush()
                    return self._stored_assessment(existing)

                assessment = AIAssessment(
                    claim_id=claim_id,
                    input_hash=input_hash,
                    **values,
                )
                session.add(assessment)
                session.flush()
                return self._stored_assessment(assessment)
        except IntegrityError:
            cached = self.get_cached_assessment(claim_id, input_hash)
            if cached is not None:
                return cached
            raise

    def employee_counts(self, actor_id: int) -> DashboardCounts:
        claims = self.list_employee_claims(actor_id)
        return self._counts(claims)

    @staticmethod
    def _counts(claims: list[ExpenseClaim]) -> DashboardCounts:
        by_status = {status: 0 for status in ExpenseStatus}
        for claim in claims:
            by_status[claim.status] += 1
        return DashboardCounts(
            total=len(claims),
            pending=by_status[ExpenseStatus.PENDING],
            approved=by_status[ExpenseStatus.APPROVED],
            rejected=by_status[ExpenseStatus.REJECTED],
            withdrawn=by_status[ExpenseStatus.WITHDRAWN],
        )

    @staticmethod
    def _stored_assessment(assessment: AIAssessment) -> StoredAssessment:
        try:
            reasons = json.loads(assessment.reasons_json)
        except json.JSONDecodeError:
            reasons = []
        return StoredAssessment(
            claim_id=assessment.claim_id,
            input_hash=assessment.input_hash,
            summary=assessment.summary,
            is_inconsistent=assessment.is_inconsistent,
            reasons=[str(reason) for reason in reasons],
            source=assessment.source,
            model=assessment.model,
            latency_ms=assessment.latency_ms,
        )

    @staticmethod
    def _require_role(session: object, user_id: int, role: RoleName) -> None:
        role_link = session.scalar(  # type: ignore[attr-defined]
            select(UserRole).where(UserRole.user_id == user_id, UserRole.role == role)
        )
        if role_link is None:
            raise AccessDeniedError(f"The {role.value} role is required for this action.")

    @classmethod
    def _validate_amount(cls, amount: Decimal | str | float) -> int:
        try:
            decimal_amount = Decimal(str(amount))
        except (InvalidOperation, ValueError):
            raise ValidationError("Enter a valid amount.") from None
        if not decimal_amount.is_finite():
            raise ValidationError("Enter a valid amount.")
        quantized = decimal_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if quantized != decimal_amount:
            raise ValidationError("Amount can have at most two decimal places.")
        if quantized <= 0 or quantized > cls.MAX_AMOUNT:
            raise ValidationError("Amount must be between $0.01 and $1,000,000.00.")
        return int(quantized * 100)

    @staticmethod
    def _validate_text(value: str, label: str, minimum: int, maximum: int) -> str:
        cleaned = value.strip()
        if len(cleaned) < minimum:
            raise ValidationError(f"{label} must be at least {minimum} characters.")
        if len(cleaned) > maximum:
            raise ValidationError(f"{label} must be {maximum} characters or fewer.")
        return cleaned
