"""Add RESOLVED and CLOSED to workorderstatus enum

Revision ID: 6d2e4f8a1c3b
Revises: 4c0c2b293399
Create Date: 2026-10-30 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '6d2e4f8a1c3b'
down_revision: Union[str, Sequence[str], None] = '4c0c2b293399'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add RESOLVED and CLOSED values to the workorderstatus enum."""
    # PostgreSQL does not support ALTER TYPE ... ADD VALUE IF NOT EXISTS,
    # so we guard with a DO block to make the migration idempotent.
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_enum
                WHERE enumlabel = 'RESOLVED'
                  AND enumtypid = 'workorderstatus'::regtype
            ) THEN
                ALTER TYPE workorderstatus ADD VALUE 'RESOLVED';
            END IF;
        END $$;
    """)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_enum
                WHERE enumlabel = 'CLOSED'
                  AND enumtypid = 'workorderstatus'::regtype
            ) THEN
                ALTER TYPE workorderstatus ADD VALUE 'CLOSED';
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """Remove RESOLVED and CLOSED from the workorderstatus enum.

    PostgreSQL does not support removing enum values directly. A full
    downgrade would require recreating the enum type, which is unsafe
    without knowing all dependent columns. Left as a no-op to avoid
    data loss.
    """
    pass
