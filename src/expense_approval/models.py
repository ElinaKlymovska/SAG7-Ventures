from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class RoleName(StrEnum):
    EMPLOYEE = "employee"
    APPROVER = "approver"


class ExpenseStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class AssessmentSource(StrEnum):
    OPENAI = "openai"
    FALLBACK = "fallback"


def enum_values(enum_class: type[StrEnum]) -> list[str]:
    return [item.value for item in enum_class]


class Base(DeclarativeBase):
    type_annotation_map: dict[Any, Any] = {}


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[RoleName] = mapped_column(
        Enum(RoleName, values_callable=enum_values, native_enum=False), primary_key=True
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    role_links: Mapped[list[UserRole]] = relationship(
        cascade="all, delete-orphan", lazy="selectin"
    )
    submitted_claims: Mapped[list[ExpenseClaim]] = relationship(
        back_populates="employee", foreign_keys="ExpenseClaim.employee_id"
    )
    assigned_categories: Mapped[list[Category]] = relationship(back_populates="approver")

    @property
    def roles(self) -> set[RoleName]:
        return {link.role for link in self.role_links}

    def has_role(self, role: RoleName) -> bool:
        return role in self.roles


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    approver_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))

    approver: Mapped[User] = relationship(back_populates="assigned_categories")
    claims: Mapped[list[ExpenseClaim]] = relationship(back_populates="category")


class ExpenseClaim(Base):
    __tablename__ = "expense_claims"
    __table_args__ = (
        CheckConstraint("amount_cents > 0", name="ck_expense_claim_amount_positive"),
        CheckConstraint("currency = 'USD'", name="ck_expense_claim_currency_usd"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="RESTRICT")
    )
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    description: Mapped[str] = mapped_column(String(500))
    expense_date: Mapped[date] = mapped_column(Date)
    payment_details: Mapped[str] = mapped_column(String(500))
    status: Mapped[ExpenseStatus] = mapped_column(
        Enum(ExpenseStatus, values_callable=enum_values, native_enum=False),
        default=ExpenseStatus.PENDING,
        index=True,
    )
    decision_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    employee: Mapped[User] = relationship(
        back_populates="submitted_claims", foreign_keys=[employee_id]
    )
    category: Mapped[Category] = relationship(back_populates="claims")
    decided_by: Mapped[User | None] = relationship(foreign_keys=[decided_by_id])
    assessments: Mapped[list[AIAssessment]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )
    document: Mapped[ExpenseDocument | None] = relationship(
        back_populates="claim", cascade="all, delete-orphan", uselist=False
    )

    @property
    def display_id(self) -> str:
        return f"EXP-{self.id:04d}"

    @property
    def amount_usd(self) -> str:
        return f"{self.amount_cents / 100:,.2f}"


class AIAssessment(Base):
    __tablename__ = "ai_assessments"
    __table_args__ = (
        UniqueConstraint("claim_id", "input_hash", name="uq_assessment_claim_input"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("expense_claims.id", ondelete="CASCADE"), index=True
    )
    input_hash: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(String(500))
    is_inconsistent: Mapped[bool] = mapped_column(Boolean)
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    source: Mapped[AssessmentSource] = mapped_column(
        Enum(AssessmentSource, values_callable=enum_values, native_enum=False)
    )
    model: Mapped[str] = mapped_column(String(100))
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    claim: Mapped[ExpenseClaim] = relationship(back_populates="assessments")


class ExpenseDocument(Base):
    __tablename__ = "expense_documents"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="ck_expense_document_size_positive"),
        CheckConstraint(
            "size_bytes <= 8388608", name="ck_expense_document_size_maximum"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("expense_claims.id", ondelete="CASCADE"), unique=True, index=True
    )
    original_name: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    document_kind: Mapped[str] = mapped_column(String(40))
    processing_method: Mapped[str] = mapped_column(String(80))
    vendor: Mapped[str | None] = mapped_column(String(120), nullable=True)
    document_number: Mapped[str | None] = mapped_column(String(80), nullable=True)
    detected_amount: Mapped[str | None] = mapped_column(String(40), nullable=True)
    detected_currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    detected_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    category_hint: Mapped[str] = mapped_column(String(80))
    confidence_percent: Mapped[int] = mapped_column(Integer)
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    claim: Mapped[ExpenseClaim] = relationship(back_populates="document")
