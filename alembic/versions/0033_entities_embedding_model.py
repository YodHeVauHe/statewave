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

from server.services.migrations import configured_embedding_model

revision: str = "0033_entities_embedding_model"
down_revision: Union[str, None] = "0032_memories_embedding_model"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "subject_entities",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )

    # Backfill existing rows with the model presently producing embeddings.
    #
    # Without this the guard only protects rows written after the upgrade:
    # every existing row keeps a NULL model, NULL is exempt by design, and
    # the exact-match path returns before it stamps anything, so nothing
    # ever heals them. `rebuild-entities` re-upserts through that same path,
    # so it does not heal them either. On any deployment that already has
    # entities the cross-model merge would stay live forever.
    #
    # Stamping the configured model is only wrong where the operator already
    # swapped models before upgrading, which is the case that is broken
    # today regardless, and it is right everywhere else. NULL stays for
    # deployments with no resolvable provider, where guessing would be worse
    # than admitting we do not know.
    #
    # Single statement rather than 0032's batching: subject_entities carries
    # no vector index (0028 indexes subject_id only), so there is no
    # index-maintenance cost per row, and the table holds one row per entity
    # rather than one per memory.
    bind = op.get_bind()
    current_model = configured_embedding_model(bind)
    if current_model is not None:
        bind.execute(
            sa.text(
                "UPDATE subject_entities SET embedding_model = :model "
                "WHERE embedding IS NOT NULL AND embedding_model IS NULL"
            ).bindparams(sa.bindparam("model", value=current_model, type_=sa.Text()))
        )


def downgrade() -> None:
    op.drop_column("subject_entities", "embedding_model")
