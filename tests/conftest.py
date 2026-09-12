from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select

from expense_approval.db import Database
from expense_approval.models import Base, Category, ExpenseClaim, User
from expense_approval.seed import seed_if_empty
from expense_approval.services import ExpenseService


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    database = Database(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    Base.metadata.create_all(database.engine)
    seed_if_empty(database)
    yield database
    database.engine.dispose()


@pytest.fixture
def service(database: Database) -> ExpenseService:
    return ExpenseService(database)


@pytest.fixture
def ids(database: Database) -> dict[str, int]:
    with database.session() as session:
        users = {user.email: user.id for user in session.scalars(select(User))}
        categories = {category.name: category.id for category in session.scalars(select(Category))}
        claims = {
            claim.description: claim.id for claim in session.scalars(select(ExpenseClaim))
        }
    return {**users, **categories, **claims}

