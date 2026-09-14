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
    # 0002 declared `content` NOT NULL, and the bytes it held are gone for good. A
    # nullable column would leave rows that the rolled-back code cannot read: it
    # hashes document.content on every submission. Restore the NOT NULL shape by
    # backfilling empty bytes, so the schema matches 0002 even though the original
    # attachments cannot come back.
    with op.batch_alter_table("expense_documents") as batch:
        batch.add_column(
            sa.Column(
                "content",
                sa.LargeBinary(),
                nullable=False,
                server_default=sa.text("x''"),
            )
        )
