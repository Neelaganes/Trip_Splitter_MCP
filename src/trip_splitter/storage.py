import secrets
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from trip_splitter.domain import Category, Conversion, Expense, RateSource, Trip


class NotFoundError(LookupError):
    """No trip or expense with that id (for this caller)."""


class Storage(ABC):
    @abstractmethod
    def create_trip(self, name: str, members: list[str], base_currency: str = "INR") -> Trip:
        """Save a new trip with a server-made id and return it."""

    @abstractmethod
    def get_trip(self, trip_id: str) -> Trip:
        """Return the trip, or raise NotFoundError."""

    @abstractmethod
    def list_trips(self) -> list[Trip]:
        """Every trip, oldest first."""

    @abstractmethod
    def add_member(self, trip_id: str, name: str) -> Trip:
        """Add a member and return the updated trip, or raise NotFoundError."""

    @abstractmethod
    def add_expense(self, trip_id: str, expense: Expense) -> Expense:
        """Save the expense with a server-made id and timestamp; return the saved copy."""

    @abstractmethod
    def list_expenses(self, trip_id: str) -> list[Expense]:
        """The trip's expenses that haven't been deleted, oldest first."""

    @abstractmethod
    def delete_expense(self, trip_id: str, expense_id: str) -> Expense:
        """Mark the expense deleted (it stays in history) and return it, or raise NotFoundError."""

SCHEMA = """
        CREATE TABLE IF NOT EXISTS trips (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_currency TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS members (
            trip_id TEXT NOT NULL REFERENCES trips (id),
            name TEXT NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (trip_id, name)
        );
        CREATE TABLE IF NOT EXISTS expenses (
            id TEXT PRIMARY KEY,
            trip_id TEXT NOT NULL REFERENCES trips (id),
            paid_by TEXT NOT NULL,
            amount_minor INTEGER NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS shares (
            expense_id TEXT NOT NULL REFERENCES expenses (id),
            member TEXT NOT NULL,
            amount_minor INTEGER NOT NULL,
            PRIMARY KEY (expense_id, member)
        );
        """

MIGRATIONS = [
    [
        "ALTER TABLE expenses ADD COLUMN spent_on TEXT",
        "ALTER TABLE expenses ADD COLUMN original_currency TEXT",
        "ALTER TABLE expenses ADD COLUMN original_amount_minor INTEGER",
        "ALTER TABLE expenses ADD COLUMN fx_rate TEXT",
        "ALTER TABLE expenses ADD COLUMN fx_date TEXT",
        "ALTER TABLE expenses ADD COLUMN fx_source TEXT",
    ],
]


def migrate(db: sqlite3.Connection) -> None:
    """Bring an older database up to date, once, even when several servers open it at the same time."""
    if db.execute("PRAGMA user_version").fetchone()[0] >= len(MIGRATIONS):
        return
    db.execute("BEGIN IMMEDIATE")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    for number, statements in enumerate(MIGRATIONS[version:], start=version + 1):
        for statement in statements:
            db.execute(statement)
        db.execute(f"PRAGMA user_version = {number}")
    db.commit()


def conversion_columns(conversion: Conversion | None) -> tuple:
    if conversion is None:
        return (None, None, None, None, None)
    return (conversion.currency, conversion.amount_minor, str(conversion.rate),
            conversion.rate_date.isoformat(), conversion.source.value)


def read_conversion(currency: str | None, amount_minor: int | None, rate: str | None,
                    rate_date: str | None, source: str | None) -> Conversion | None:
    if currency is None:
        return None
    return Conversion(currency, amount_minor, Decimal(rate), date.fromisoformat(rate_date), RateSource(source))


class SQLiteStorage(Storage):
    def __init__(self, path: str | Path) -> None:
        self.path = path
        with self._connect() as db:
            db.executescript(SCHEMA)
            migrate(db)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def create_trip(self, name: str, members: list[str], base_currency: str = "INR") -> Trip:
        trip = Trip(f"trip_{secrets.token_hex(4)}", name, tuple(members), base_currency, datetime.now(UTC))
        with self._connect() as db:
            db.execute(
                "INSERT INTO trips (id, name, base_currency, created_at) VALUES (?, ?, ?, ?)",
                (trip.id, trip.name, trip.base_currency, trip.created_at.isoformat()),
            )
            db.executemany(
                "INSERT INTO members (trip_id, name, position) VALUES (?, ?, ?)",
                [(trip.id, member, position) for position, member in enumerate(trip.members)],
            )
        return trip

    def get_trip(self, trip_id: str) -> Trip:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, name, base_currency, created_at FROM trips WHERE id = ?", (trip_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"No trip with id {trip_id!r}.")
            members = db.execute(
                "SELECT name FROM members WHERE trip_id = ? ORDER BY position", (trip_id,)
            ).fetchall()
        return Trip(row[0], row[1], tuple(name for (name,) in members), row[2], datetime.fromisoformat(row[3]))

    def list_trips(self) -> list[Trip]:
        with self._connect() as db:
            rows = db.execute("SELECT id FROM trips ORDER BY created_at, id").fetchall()
        return [self.get_trip(trip_id) for (trip_id,) in rows]

    def add_member(self, trip_id: str, name: str) -> Trip:
        trip = self.get_trip(trip_id)
        updated = replace(trip, members=(*trip.members, name))
        with self._connect() as db:
            db.execute(
                "INSERT INTO members (trip_id, name, position) VALUES (?, ?, ?)",
                (trip_id, name, len(trip.members)),
            )
        return updated

    def add_expense(self, trip_id: str, expense: Expense) -> Expense:
        self.get_trip(trip_id)
        saved = replace(expense, id=f"exp_{secrets.token_hex(4)}", created_at=datetime.now(UTC))
        with self._connect() as db:
            db.execute(
                "INSERT INTO expenses (id, trip_id, paid_by, amount_minor, description, category, note, created_at,"
                " spent_on, original_currency, original_amount_minor, fx_rate, fx_date, fx_source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (saved.id, trip_id, saved.paid_by, saved.amount_minor, saved.description,
                 saved.category.value, saved.note, saved.created_at.isoformat(),
                 saved.spent_on.isoformat() if saved.spent_on else None, *conversion_columns(saved.conversion)),
            )
            db.executemany(
                "INSERT INTO shares (expense_id, member, amount_minor) VALUES (?, ?, ?)",
                [(saved.id, member, share) for member, share in saved.shares.items()],
            )
        return saved

    def list_expenses(self, trip_id: str) -> list[Expense]:
        self.get_trip(trip_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, paid_by, amount_minor, description, category, note, created_at, spent_on,"
                " original_currency, original_amount_minor, fx_rate, fx_date, fx_source FROM expenses"
                " WHERE trip_id = ? AND deleted_at IS NULL ORDER BY created_at, id",
                (trip_id,),
            ).fetchall()
            share_rows = db.execute(
                "SELECT shares.expense_id, shares.member, shares.amount_minor FROM shares"
                " JOIN expenses ON expenses.id = shares.expense_id WHERE expenses.trip_id = ?",
                (trip_id,),
            ).fetchall()
        shares: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for expense_id, member, share in share_rows:
            shares[expense_id][member] = share
        return [
            Expense(paid_by, amount_minor, shares[expense_id], description, Category(category), note,
                    expense_id, datetime.fromisoformat(created_at),
                    date.fromisoformat(spent_on) if spent_on else None, read_conversion(*fx))
            for expense_id, paid_by, amount_minor, description, category, note, created_at, spent_on, *fx in rows
        ]

    def delete_expense(self, trip_id: str, expense_id: str) -> Expense:
        expense = next((e for e in self.list_expenses(trip_id) if e.id == expense_id), None)
        if expense is None:
            raise NotFoundError(f"No expense with id {expense_id!r} in trip {trip_id!r}.")
        with self._connect() as db:
            db.execute(
                "UPDATE expenses SET deleted_at = ? WHERE id = ? AND trip_id = ? AND deleted_at IS NULL",
                (datetime.now(UTC).isoformat(), expense_id, trip_id),
            )
        return expense


POSTGRES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS trips (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        base_currency TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS members (
        trip_id TEXT NOT NULL REFERENCES trips (id),
        name TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (trip_id, name)
    );
    CREATE TABLE IF NOT EXISTS expenses (
        id TEXT PRIMARY KEY,
        trip_id TEXT NOT NULL REFERENCES trips (id),
        paid_by TEXT NOT NULL,
        amount_minor BIGINT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        note TEXT NOT NULL,
        created_at TEXT NOT NULL,
        deleted_at TEXT,
        spent_on TEXT,
        original_currency TEXT,
        original_amount_minor BIGINT,
        fx_rate TEXT,
        fx_date TEXT,
        fx_source TEXT
    );
    CREATE TABLE IF NOT EXISTS shares (
        expense_id TEXT NOT NULL REFERENCES expenses (id),
        member TEXT NOT NULL,
        amount_minor BIGINT NOT NULL,
        PRIMARY KEY (expense_id, member)
    );
"""


def with_tls(conninfo: str) -> str:
    """The connection string with sslmode=require, unless it already sets an sslmode."""
    if "sslmode" in conninfo_to_dict(conninfo):
        return conninfo
    return make_conninfo(conninfo, sslmode="require")


class PostgresStorage(Storage):
    def __init__(self, conninfo: str) -> None:
        self.conninfo = with_tls(conninfo)
        with self._connect() as db:
            db.execute(POSTGRES_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.conninfo) as connection:
            yield connection

    def create_trip(self, name: str, members: list[str], base_currency: str = "INR") -> Trip:
        trip = Trip(f"trip_{secrets.token_hex(4)}", name, tuple(members), base_currency, datetime.now(UTC))
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO trips (id, name, base_currency, created_at) VALUES (%s, %s, %s, %s)",
                (trip.id, trip.name, trip.base_currency, trip.created_at.isoformat()),
            )
            cursor.executemany(
                "INSERT INTO members (trip_id, name, position) VALUES (%s, %s, %s)",
                [(trip.id, member, position) for position, member in enumerate(trip.members)],
            )
        return trip

    def get_trip(self, trip_id: str) -> Trip:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, name, base_currency, created_at FROM trips WHERE id = %s", (trip_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"No trip with id {trip_id!r}.")
            members = db.execute(
                "SELECT name FROM members WHERE trip_id = %s ORDER BY position", (trip_id,)
            ).fetchall()
        return Trip(row[0], row[1], tuple(name for (name,) in members), row[2], datetime.fromisoformat(row[3]))

    def list_trips(self) -> list[Trip]:
        with self._connect() as db:
            rows = db.execute("SELECT id FROM trips ORDER BY created_at, id").fetchall()
        return [self.get_trip(trip_id) for (trip_id,) in rows]

    def add_member(self, trip_id: str, name: str) -> Trip:
        trip = self.get_trip(trip_id)
        updated = replace(trip, members=(*trip.members, name))
        with self._connect() as db:
            db.execute(
                "INSERT INTO members (trip_id, name, position) VALUES (%s, %s, %s)",
                (trip_id, name, len(trip.members)),
            )
        return updated

    def add_expense(self, trip_id: str, expense: Expense) -> Expense:
        self.get_trip(trip_id)
        saved = replace(expense, id=f"exp_{secrets.token_hex(4)}", created_at=datetime.now(UTC))
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO expenses (id, trip_id, paid_by, amount_minor, description, category, note, created_at,"
                " spent_on, original_currency, original_amount_minor, fx_rate, fx_date, fx_source)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (saved.id, trip_id, saved.paid_by, saved.amount_minor, saved.description,
                 saved.category.value, saved.note, saved.created_at.isoformat(),
                 saved.spent_on.isoformat() if saved.spent_on else None, *conversion_columns(saved.conversion)),
            )
            cursor.executemany(
                "INSERT INTO shares (expense_id, member, amount_minor) VALUES (%s, %s, %s)",
                [(saved.id, member, share) for member, share in saved.shares.items()],
            )
        return saved

    def list_expenses(self, trip_id: str) -> list[Expense]:
        self.get_trip(trip_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, paid_by, amount_minor, description, category, note, created_at, spent_on,"
                " original_currency, original_amount_minor, fx_rate, fx_date, fx_source FROM expenses"
                " WHERE trip_id = %s AND deleted_at IS NULL ORDER BY created_at, id",
                (trip_id,),
            ).fetchall()
            share_rows = db.execute(
                "SELECT shares.expense_id, shares.member, shares.amount_minor FROM shares"
                " JOIN expenses ON expenses.id = shares.expense_id WHERE expenses.trip_id = %s",
                (trip_id,),
            ).fetchall()
        shares: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for expense_id, member, share in share_rows:
            shares[expense_id][member] = share
        return [
            Expense(paid_by, amount_minor, shares[expense_id], description, Category(category), note,
                    expense_id, datetime.fromisoformat(created_at),
                    date.fromisoformat(spent_on) if spent_on else None, read_conversion(*fx))
            for expense_id, paid_by, amount_minor, description, category, note, created_at, spent_on, *fx in rows
        ]

    def delete_expense(self, trip_id: str, expense_id: str) -> Expense:
        expense = next((e for e in self.list_expenses(trip_id) if e.id == expense_id), None)
        if expense is None:
            raise NotFoundError(f"No expense with id {expense_id!r} in trip {trip_id!r}.")
        with self._connect() as db:
            db.execute(
                "UPDATE expenses SET deleted_at = %s WHERE id = %s AND trip_id = %s AND deleted_at IS NULL",
                (datetime.now(UTC).isoformat(), expense_id, trip_id),
            )
        return expense
