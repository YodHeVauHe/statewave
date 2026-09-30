"""Migration safety utilities.

Shared logic for preflight checks, startup guards, and admin endpoints.
Provides schema version introspection without requiring the full app to boot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

# Expected head revision — update this when adding new migrations
EXPECTED_HEAD = "0033_entities_embedding_model"

# Path to alembic.ini relative to the repo root
_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def resolve_database_url(explicit: str | None = None) -> str:
    """Resolve DB URL: explicit arg, env vars, then settings (.env)."""
    import os

    if explicit is not None:
        return explicit
    if os.environ.get("STATEWAVE_DATABASE_URL"):
        return os.environ["STATEWAVE_DATABASE_URL"]
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    from server.core.config import settings

    return settings.database_url


@dataclass
class MigrationStatus:
    """Result of a migration state check."""

    current_revision: str | None = None
    expected_head: str = EXPECTED_HEAD
    pending_count: int = 0
    pending_revisions: list[str] = field(default_factory=list)
    is_compatible: bool = False
    error: str | None = None

    @property
    def needs_migration(self) -> bool:
        return self.pending_count > 0

    @property
    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if self.is_compatible:
            return "Schema is up to date"
        return f"{self.pending_count} pending migration(s): {self.current_revision} → {self.expected_head}"


def get_alembic_config() -> Config:
    """Build Alembic config from alembic.ini."""
    cfg = Config(str(_ALEMBIC_INI))
    url = resolve_database_url()
    if url:
        cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def get_script_directory() -> ScriptDirectory:
    """Get the Alembic script directory for revision introspection."""
    return ScriptDirectory.from_config(get_alembic_config())


def get_all_revisions() -> list[str]:
    """Return ordered list of all revision IDs from base to head."""
    script = get_script_directory()
    revs = []
    for rev in script.walk_revisions():
        revs.append(rev.revision)
    revs.reverse()  # walk_revisions goes head→base
    return revs


async def check_migration_status(database_url: str | None = None) -> MigrationStatus:
    """Check current DB revision against expected head.

    This creates its own connection (no dependency on the app engine)
    so it can be used in preflight scripts and startup guards.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    url = resolve_database_url(database_url)
    if not url:
        return MigrationStatus(
            error="Database URL not set (STATEWAVE_DATABASE_URL or DATABASE_URL)"
        )

    status = MigrationStatus()

    try:
        engine = create_async_engine(url, pool_pre_ping=True)
        async with engine.connect() as conn:
            # Check if alembic_version table exists
            result = await conn.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT FROM information_schema.tables "
                    "  WHERE table_name = 'alembic_version'"
                    ")"
                )
            )
            table_exists = result.scalar()

            if not table_exists:
                status.current_revision = None
                status.pending_count = len(get_all_revisions())
                status.pending_revisions = get_all_revisions()
                return status

            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            row = result.first()
            status.current_revision = row[0] if row else None

        await engine.dispose()
    except Exception as exc:
        status.error = str(exc)[:300]
        return status

    # Calculate pending migrations
    return _resolve_pending(status)


def _resolve_pending(status: MigrationStatus) -> MigrationStatus:
    """Given a status with current_revision set, calculate pending info."""
    all_revs = get_all_revisions()
    if status.current_revision is None:
        status.pending_revisions = all_revs
        status.pending_count = len(all_revs)
    elif status.current_revision == EXPECTED_HEAD:
        status.is_compatible = True
        status.pending_count = 0
    else:
        try:
            idx = all_revs.index(status.current_revision)
            status.pending_revisions = all_revs[idx + 1 :]
            status.pending_count = len(status.pending_revisions)
        except ValueError:
            status.error = (
                f"Current revision '{status.current_revision}' not found in migration chain"
            )

    return status


def configured_embedding_model(bind) -> str | None:
    """Best-effort resolution of the model presently producing embeddings.

    Mirrors the precedence `server.core.dynamic_settings.get_setting()`
    documents (tenant_override -> global_db -> env -> hardcoded default),
    minus the tenant step: `server.services.embeddings.get_provider()`
    builds one process-wide singleton from global config only and never
    consults a tenant override, so a tenant-scoped `litellm_embedding_model`
    row isn't the model actually stamping fresh vectors either; mirroring
    tenant precedence here would claim a precision the running server
    doesn't have.

    Reads `system_settings` (the admin-UI override layer, #26) directly so
    an operator-configured deployment backfills the value actually set
    there, then falls back to `server.core.config.settings`, which loads
    `.env` itself, unlike a raw `os.environ` read, for the env/.env step.
    `alembic/env.py` already imports `server.db.tables` and
    `server.services.migrations`, so importing `server.core.config` here
    is the same thing this chain already does elsewhere.

    A deployment topology that hands this migration's process a narrower
    env than the application itself sees (e.g. a Helm migration Job given
    only `STATEWAVE_DATABASE_URL`) and has never written a `system_settings`
    override either is a chart-wiring gap outside what any in-process read
    can recover. The "none"/unrecognized branch below leaves the column
    NULL rather than guessing in that case.
    """
    from sqlalchemy import select

    from server.core.config import settings as env_settings
    from server.core.dynamic_settings import system_settings

    overrides: dict[str, object] = {}
    try:
        # Selecting through the real `system_settings` Table (rather than a
        # hand-typed `sa.text` SELECT) makes SQLAlchemy apply the JSONB
        # column's own result decoding, so this reads the same Python value
        # `apply_global_override` wrote, independent of what the driver
        # returns for a raw-text query.
        result = bind.execute(
            select(system_settings.c.key, system_settings.c.value).where(
                system_settings.c.key.in_(("embedding_provider", "litellm_embedding_model"))
            )
        )
        overrides = dict(result.fetchall())
    except Exception:
        # `system_settings` is created by migration 0026, strictly earlier
        # in this chain, so this is defensive (e.g. an offline `--sql`
        # render with no live connection) rather than an expected path.
        overrides = {}

    provider = overrides.get("embedding_provider", env_settings.embedding_provider)
    if provider == "litellm":
        return overrides.get("litellm_embedding_model", env_settings.litellm_embedding_model)
    if provider == "stub":
        return "stub"
    return None  # "none" (or unrecognized): no live provider to attribute rows to
