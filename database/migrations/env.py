"""Migration environment configuration and metadata binding."""

from database.models import Base
from database.connection import get_db_url, get_engine

target_metadata = Base.metadata


def get_metadata():
    """Returns SQLAlchemy metadata for migration auto-generation and validation."""
    return target_metadata
