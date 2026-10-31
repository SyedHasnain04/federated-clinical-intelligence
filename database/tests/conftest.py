"""Pytest fixtures for relational database testing.

Supports dual-mode execution:
1. Live PostgreSQL instance (if DB_HOST/DB_PORT is reachable).
2. Supplemental in-memory relational engine (with PRAGMA foreign_keys=ON) for CI
   and local developer validation when a live PostgreSQL instance is not provisioned.
"""

import os
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from database.models import Base
from database.connection import check_connection, get_engine
from database.synthetic.relational_generator import generate_synthetic_relational_cohort


@event.listens_for(Engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Enforces strict foreign key validation on SQLite in-memory connections."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            pass


@pytest.fixture(scope="session")
def is_postgres_available() -> bool:
    """Checks whether a live PostgreSQL server is accessible."""
    try:
        eng = get_engine()
        is_conn, _ = check_connection(eng)
        return is_conn
    except Exception:
        return False


@pytest.fixture(scope="function")
def db_engine(is_postgres_available: bool):
    """Yields a clean, isolated relational database engine for testing."""
    if is_postgres_available and os.getenv("TEST_WITH_LIVE_POSTGRES"):
        # Live PostgreSQL test database
        engine = get_engine()
        # Drop and recreate tables in test schema
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        yield engine
        Base.metadata.drop_all(engine)
    else:
        # Isolated in-memory engine with strict foreign key enforcement
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        yield engine
        Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(db_engine) -> Session:
    """Yields an active database session for a test function."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def synthetic_cohort():
    """Generates a fixed synthetic relational cohort for tests."""
    return generate_synthetic_relational_cohort(n_patients=25, seed=42)

