"""SQLite engine and session management.

Each case owns a database file. :class:`Database` wraps engine creation,
schema initialisation and scoped sessions, and enables SQLite pragmas that
matter for a single-writer desktop application (WAL, foreign keys).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

logger = logging.getLogger(__name__)

__all__ = ["Database"]


class Database:
    """Owns one SQLite file and hands out sessions.

    Example:
        >>> db = Database("cases/CASE-0001/case.db")
        >>> db.create_all()
        >>> with db.session() as session:
        ...     session.add(some_row)
    """

    def __init__(self, path: os.PathLike | str, echo: bool = False) -> None:
        """Create (but do not yet connect) a database handle.

        Args:
            path: Path to the SQLite file. Parent directories are created.
            echo: Emit SQL to the logger; useful when debugging.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Engine = create_engine(
            f"sqlite:///{self._path.as_posix()}",
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30.0},
        )
        self._configure_pragmas(self._engine)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False, future=True
        )
        logger.debug("Database handle created for %s", self._path)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _configure_pragmas(engine: Engine) -> None:
        """Apply SQLite pragmas on every new DBAPI connection."""

        @event.listens_for(engine, "connect")
        def _set_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")
            finally:
                cursor.close()

    @property
    def path(self) -> Path:
        """Path of the SQLite file."""
        return self._path

    @property
    def engine(self) -> Engine:
        """The underlying SQLAlchemy engine."""
        return self._engine

    def create_all(self) -> None:
        """Create any missing tables."""
        Base.metadata.create_all(self._engine)
        logger.info("Schema ensured for %s", self._path.name)

    # --------------------------------------------------------------- sessions
    def new_session(self) -> Session:
        """Return a new unmanaged session; the caller must close it."""
        return self._session_factory()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Context manager yielding a session that commits on success.

        Any exception rolls the transaction back and is re-raised.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def read_session(self) -> Iterator[Session]:
        """Context manager for read-only access; never commits."""
        session = self._session_factory()
        try:
            yield session
        finally:
            session.close()

    def dispose(self) -> None:
        """Close all pooled connections."""
        self._engine.dispose()
        logger.debug("Database disposed: %s", self._path)

    def __enter__(self) -> "Database":  # pragma: no cover - convenience
        return self

    def __exit__(self, *_exc) -> None:  # pragma: no cover - convenience
        self.dispose()


def open_database(path: os.PathLike | str, create: bool = True) -> Database:
    """Open (and optionally initialise) a case database.

    Args:
        path: SQLite file path.
        create: Create the schema when missing.
    """
    database = Database(path)
    if create:
        database.create_all()
    return database


_shared: Optional[Database] = None
