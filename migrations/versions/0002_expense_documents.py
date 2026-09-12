"""Add supporting expense documents.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "expense_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("claim_id", sa.Integer(), nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("document_kind", sa.String(length=40), nullable=False),
        sa.Column("processing_method", sa.String(length=80), nullable=False),
        sa.Column("vendor", sa.String(length=120), nullable=True),
        sa.Column("document_number", sa.String(length=80), nullable=True),
        sa.Column("detected_amount", sa.String(length=40), nullable=True),
        sa.Column("detected_currency", sa.String(length=3), nullable=True),
        sa.Column("detected_date", sa.Date(), nullable=True),
        sa.Column("category_hint", sa.String(length=80), nullable=False),
        sa.Column("confidence_percent", sa.Integer(), nullable=False),
        sa.Column("warnings_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("size_bytes > 0", name="ck_expense_document_size_positive"),
        sa.CheckConstraint("size_bytes <= 8388608", name="ck_expense_document_size_maximum"),
        sa.ForeignKeyConstraint(["claim_id"], ["expense_claims.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("claim_id"),
    )
    op.create_index(
        "ix_expense_documents_claim_id", "expense_documents", ["claim_id"], unique=True
    )
    op.create_index("ix_expense_documents_sha256", "expense_documents", ["sha256"])


def downgrade() -> None:
    op.drop_index("ix_expense_documents_sha256", table_name="expense_documents")
    op.drop_index("ix_expense_documents_claim_id", table_name="expense_documents")
    op.drop_table("expense_documents")
