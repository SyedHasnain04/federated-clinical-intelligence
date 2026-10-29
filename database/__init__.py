"""Database foundation module for the Federated Clinical Intelligence Platform.

Provides core PostgreSQL relational schema DDL, SQLAlchemy 2.0 ORM models,
connection management, migration runners, and synthetic cohort generation.
"""

from database.models import (
    Base,
    Patient,
    Admission,
    DiagnosisICD,
    ICUStay,
    LabEvent,
    ChartEvent,
)
from database.connection import (
    get_db_config,
    get_db_url,
    get_engine,
    get_session_factory,
    check_connection,
)

__all__ = [
    "Base",
    "Patient",
    "Admission",
    "DiagnosisICD",
    "ICUStay",
    "LabEvent",
    "ChartEvent",
    "get_db_config",
    "get_db_url",
    "get_engine",
    "get_session_factory",
    "check_connection",
]
