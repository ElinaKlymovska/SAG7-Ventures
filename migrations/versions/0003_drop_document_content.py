"""Stop storing the original supporting document.

Only the extracted fields, the file name and its checksum are kept, so the binary
column is dropped along with the data it holds.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("expense_documents") as batch:
        batch.drop_column("content")


def downgrade() -> None:
    # The original bytes are gone for good; the column returns empty and nullable.
    with op.batch_alter_table("expense_documents") as batch:
        batch.add_column(sa.Column("content", sa.LargeBinary(), nullable=True))
