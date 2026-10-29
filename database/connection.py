"""Database connection and engine configuration for the Clinical Intelligence Platform.

Manages connection pooling, credentials masking, and environment configuration.
Credentials are read strictly from environment variables and are never logged.
"""

import os
from typing import Dict, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool

# Environment variable keys
ENV_DB_HOST = "DB_HOST"
ENV_DB_PORT = "DB_PORT"
ENV_DB_NAME = "DB_NAME"
ENV_DB_USER = "DB_USER"
ENV_DB_PASSWORD = "DB_PASSWORD"
ENV_DATABASE_URL = "DATABASE_URL"


def get_db_config() -> Dict[str, str]:
    """Retrieves database configuration from environment variables without exposing secrets."""
    return {
        "host": os.getenv(ENV_DB_HOST, "localhost"),
        "port": os.getenv(ENV_DB_PORT, "5432"),
        "database": os.getenv(ENV_DB_NAME, "mimic_iv_dev"),
        "user": os.getenv(ENV_DB_USER, "postgres"),
        # Password is not included in the generic config dump for safety
    }


def get_db_url(hide_password: bool = True) -> str:
    """Builds the PostgreSQL SQLAlchemy URL.

    Parameters:
    -----------
    hide_password : bool
        If True, masks the password (e.g., 'postgresql+psycopg2://user:***@host:5432/db').
    """
    override = os.getenv(ENV_DATABASE_URL)
    if override:
        if hide_password and "@" in override and ":" in override:
            # Mask user:pass in override URL if present
            try:
                u = URL.create(override)
                return u.render_as_string(hide_password=True)
            except Exception:
                return "postgresql://***:***@custom-override-url"
        return override

    password = os.getenv(ENV_DB_PASSWORD, "")
    url = URL.create(
        drivername="postgresql+asyncpg" if os.getenv("USE_ASYNCPG") else "postgresql",
        username=os.getenv(ENV_DB_USER, "postgres"),
        password=password if not hide_password else ("***" if password else None),
        host=os.getenv(ENV_DB_HOST, "localhost"),
        port=int(os.getenv(ENV_DB_PORT, "5432")),
        database=os.getenv(ENV_DB_NAME, "mimic_iv_dev"),
    )
    return url.render_as_string(hide_password=hide_password)


def get_engine(
    db_url: Optional[str] = None,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """Creates a SQLAlchemy engine with connection pooling.

    Parameters:
    -----------
    db_url : str, optional
        Custom connection string (defaults to building from environment variables).
    echo : bool
        Whether to log SQL queries.
    pool_size : int
        Base connection pool size.
    max_overflow : int
        Maximum extra connections allowed under load.
    """
    if db_url is None:
        raw_password = os.getenv(ENV_DB_PASSWORD, "")
        url = URL.create(
            drivername="postgresql",
            username=os.getenv(ENV_DB_USER, "postgres"),
            password=raw_password if raw_password else None,
            host=os.getenv(ENV_DB_HOST, "localhost"),
            port=int(os.getenv(ENV_DB_PORT, "5432")),
            database=os.getenv(ENV_DB_NAME, "mimic_iv_dev"),
        )
        target_url = url
    else:
        target_url = db_url

    # Check if target is SQLite (in-memory or file) for tests
    is_sqlite = str(target_url).startswith("sqlite")

    if is_sqlite:
        engine = create_engine(
            target_url,
            echo=echo,
            poolclass=NullPool,
        )
    else:
        engine = create_engine(
            target_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=30,
            pool_pre_ping=True,  # Disconnect detector
        )

    return engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Creates a sessionmaker bound to the given or default engine."""
    eng = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


def check_connection(engine: Optional[Engine] = None, timeout_seconds: float = 2.0) -> Tuple[bool, str]:
    """Tests if the database connection is live and responsive.

    Returns:
    --------
    Tuple[bool, str]:
        (is_connected, message)
    """
    eng = engine or get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Database connection successful."
    except Exception as e:
        safe_msg = f"Database connection unavailable: {type(e).__name__}."
        return False, safe_msg
