"""upsert_entity_with_link must not trust a cosine distance across two
different embedding models (#460).

A cosine-distance comparison between two embedding vectors is only
meaningful when both vectors were produced by the same embedding model.
`subject_entities` carried no per-row model tag, so a semantic-dedup probe
against a row whose vector predates a model swap could wrongly refuse a true
duplicate or wrongly merge two unrelated entities that happen to land close
together in the newer model's space. This mirrors the tracking
`memories.embedding_model` already has (migration 0032) and pins the merge
decision that a raw distance comparison cannot make safely.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from server.db import repositories as repo
from server.db.tables import EMBEDDING_DIMENSIONS, SubjectEntityRow

pytestmark = pytest.mark.anyio

_V = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
# Small perturbation of _V: cosine distance from _V is ~0.005, well within
# the default 0.05 (1 - 0.95) threshold, but not identical, so it sorts as
# the SECOND-nearest candidate behind an exact copy of _V.
_V_NEAR = [1.0, 0.1] + [0.0] * (EMBEDDING_DIMENSIONS - 2)


async def _rows_for(session_factory, subject_id: str):
    async with session_factory() as session:
        result = await session.execute(
            select(SubjectEntityRow).where(SubjectEntityRow.subject_id == subject_id)
        )
        return list(result.scalars())


async def _upsert(session_factory, *, subject_id, text, embedding, embedding_model, memory_id):
    async with session_factory() as session:
        row = await repo.upsert_entity_with_link(
            session,
            subject_id=subject_id,
            tenant_id=None,
            entity_text=text,
            entity_normalized=text.lower(),
            entity_kind=None,
            embedding=embedding,
            embedding_model=embedding_model,
            memory_id=memory_id,
        )
        await session.commit()
        return row


async def test_cross_model_candidate_is_not_merged(session_factory):
    """An identical vector tagged with a DIFFERENT known model must not be
    trusted for the merge decision, however close the raw distance is.

    Distinct entity text on purpose: identical text would merge at the
    step-1 exact-normalized-text match, before embeddings ever get
    compared, and this bug lives entirely in step 2's semantic probe."""
    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace Hopper",
        embedding=_V,
        embedding_model="model-a",
        memory_id=uuid.uuid4(),
    )
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="G. Hopper",
        embedding=_V,
        embedding_model="model-b",
        memory_id=uuid.uuid4(),
    )

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 2, "cross-model candidates must not be merged into one row"


async def test_far_candidate_stops_the_scan_without_a_model_check(session_factory):
    """A candidate outside dedup_cosine_threshold must stop the scan (rows
    are ordered by ascending distance, so nothing closer remains) rather
    than fall through to a fresh insert only because the model matched."""
    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"
    far_vector = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)  # orthogonal to _V
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace Hopper",
        embedding=far_vector,
        embedding_model="model-a",
        memory_id=uuid.uuid4(),
    )
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="G. Hopper",
        embedding=_V,
        embedding_model="model-a",
        memory_id=uuid.uuid4(),
    )

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 2, "a candidate outside the threshold must not be merged"


async def test_same_model_still_merges(session_factory):
    """The ordinary case (same model both sides) must keep merging exactly
    as before: this bug fix must not make dedup stricter than it needs to.

    Distinct entity text, same reason as above: this must exercise step 2's
    semantic probe, not step 1's exact-text match."""
    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"
    m1, m2 = uuid.uuid4(), uuid.uuid4()
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace Hopper",
        embedding=_V,
        embedding_model="model-a",
        memory_id=m1,
    )
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="G. Hopper",
        embedding=_V,
        embedding_model="model-a",
        memory_id=m2,
    )

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1
    assert set(rows[0].linked_memory_ids) == {m1, m2}


@pytest.mark.parametrize(
    "existing_model,new_model",
    [(None, "model-a"), ("model-a", None), (None, None)],
)
async def test_unknown_model_on_either_side_is_treated_as_comparable(
    session_factory, existing_model, new_model
):
    """NULL means unknown provenance, never a mismatch, same convention as
    MemoryRow.embedding_model. A legacy row (or a caller that never learned
    its model) must not permanently block the merge it would otherwise get.

    Distinct entity text, same reason as above: this must exercise step 2's
    semantic probe, not step 1's exact-text match."""
    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"
    m1, m2 = uuid.uuid4(), uuid.uuid4()
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace Hopper",
        embedding=_V,
        embedding_model=existing_model,
        memory_id=m1,
    )
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="G. Hopper",
        embedding=_V,
        embedding_model=new_model,
        memory_id=m2,
    )

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1
    assert set(rows[0].linked_memory_ids) == {m1, m2}


async def test_looks_past_a_cross_model_candidate_to_a_same_model_one(session_factory):
    """The nearest-by-distance row can be the wrong one to trust. A second,
    slightly farther candidate that IS the same model and still within
    threshold must be found and merged with, not skipped in favour of a
    fresh insert."""
    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"
    cross_model_row = await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace Hopper",
        embedding=_V,
        embedding_model="model-b",
        memory_id=uuid.uuid4(),
    )
    same_model_row = await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace M. Hopper",
        embedding=_V_NEAR,
        embedding_model="model-a",
        memory_id=uuid.uuid4(),
    )

    new_memory_id = uuid.uuid4()
    merged_into = await _upsert(
        session_factory,
        subject_id=subject_id,
        text="G. Hopper",
        embedding=_V,
        embedding_model="model-a",
        memory_id=new_memory_id,
    )

    assert merged_into.id == same_model_row.id
    assert merged_into.id != cross_model_row.id

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 2, "must not have inserted a third, fresh row"


async def test_conflict_path_keeps_first_non_null_embedding_model(session_factory):
    """Mirrors the existing embedding-coalesce test: a later writer carrying
    an embedding_model must fill a NULL, and a later NULL must not clobber
    an existing embedding_model, on the ON CONFLICT DO UPDATE branch."""
    import asyncio

    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"

    async def upsert_with(embedding_model, memory_id):
        async with session_factory() as session:
            await repo.upsert_entity_with_link(
                session,
                subject_id=subject_id,
                tenant_id=None,
                entity_text="Berlin",
                entity_normalized="berlin",
                entity_kind="GPE",
                embedding=None,
                embedding_model=embedding_model,
                memory_id=memory_id,
            )
            await session.commit()

    m1, m2 = uuid.uuid4(), uuid.uuid4()
    await asyncio.gather(upsert_with(None, m1), upsert_with("model-a", m2))

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1
    assert rows[0].embedding_model == "model-a"


async def test_provider_prefix_is_not_treated_as_a_different_model(session_factory):
    """`openai/text-embedding-3-small` and `text-embedding-3-small` are the
    same model, so a row tagged with one must still merge with the other.

    The guard normalizes rather than comparing raw strings, but nothing at
    this level pinned that: with a bare `!=` the two ids below look
    different and a real duplicate would be inserted instead of merged.
    """
    subject_id = f"emb-model-{uuid.uuid4().hex[:8]}"
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="Grace Hopper",
        embedding=_V,
        embedding_model="openai/text-embedding-3-small",
        memory_id=uuid.uuid4(),
    )
    await _upsert(
        session_factory,
        subject_id=subject_id,
        text="G. Hopper",
        embedding=_V,
        embedding_model="text-embedding-3-small",
        memory_id=uuid.uuid4(),
    )

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1, "the same model under a provider prefix must still merge"
