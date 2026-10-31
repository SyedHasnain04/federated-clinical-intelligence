"""Migration runner for the Clinical Intelligence Platform database schema.

Allows running versioned forward and backward migrations directly on a
SQLAlchemy engine or connection without requiring external CLI tools.
"""

import importlib
import logging
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

MIGRATION_SEQUENCE = [
    "database.migrations.versions.001_initial_core_schema",
]


def run_migrations(engine: Engine, direction: str = "upgrade") -> List[str]:
    """Applies database migrations in order.

    Parameters:
    -----------
    engine : Engine
        SQLAlchemy engine.
    direction : str
        'upgrade' or 'downgrade'.

    Returns:
    --------
    List[str]:
        List of applied migration revision IDs.
    """
    applied = []
    sequence = MIGRATION_SEQUENCE if direction == "upgrade" else list(reversed(MIGRATION_SEQUENCE))

    for mod_path in sequence:
        mod = importlib.import_module(mod_path)
        rev = getattr(mod, "revision", mod_path)

        if direction == "upgrade":
            if hasattr(mod, "upgrade"):
                mod.upgrade(engine)
                applied.append(rev)
        elif direction == "downgrade":
            if hasattr(mod, "downgrade"):
                mod.downgrade(engine)
                applied.append(rev)

    return applied
