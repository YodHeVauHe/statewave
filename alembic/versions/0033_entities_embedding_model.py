"""subject_entities.embedding_model: record which model produced each entity vector (#460).

Revision ID: 0033_entities_embedding_model
Revises: 0032_memories_embedding_model
Create Date: 2026-09-30

Migration 0032 added `memories.embedding_model` but deliberately left
`subject_entities` untouched, noting that `upsert_entity_with_link`'s
three-way fork (exact-match, semantic merge, insert-or-update) needed its
own pass. `upsert_entity_with_link`'s semantic-merge step compares a new
entity's embedding against every existing entity row for the subject via
plain cosine distance, with nothing recording which model produced either
vector: after a same-dimension embedding model swap, that distance is
computed across two unrelated vector spaces and its value carries no
defined meaning, which can either merge two unrelated entities that land
close together by coincidence in the new space, or fail to merge a real
duplicate re-embedded under the new model.

This adds the same nullable `embedding_model` column used on `memories`.
NULL means "unknown provenance" (a legacy row, or a row written before this
column existed) and, exactly as with `memories.embedding_model`, is never
treated as a mismatch: `upsert_entity_with_link` only refuses to trust a
comparison when BOTH sides name a known, different model.

No backfill here, unlike 0032: `memories.embedding_model` needed backfilling
because the read path uses it to WARN/REFUSE existing content, and an
all-NULL corpus would have made every row look unbackfilled forever.
`subject_entities.embedding_model` only gates a WRITE-time merge decision on
NEW rows going forward; a legacy row's NULL simply preserves today's
behaviour for comparisons involving it; a real model swap that happened
before this column existed is not something this migration can detect
after the fact either way.

Reversible: downgrade drops the column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0033_entities_embedding_model"
down_revision: Union[str, None] = "0032_memories_embedding_model"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "subject_entities",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subject_entities", "embedding_model")
