from __future__ import annotations

from datetime import date, timedelta

from argon2 import PasswordHasher
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from expense_approval.db import Database
from expense_approval.models import (
    AIAssessment,
    Category,
    ExpenseClaim,
    ExpenseDocument,
    ExpenseStatus,
    RoleName,
    User,
    UserRole,
    utc_now,
)

DEMO_PASSWORD = "Demo123!"
DEMO_ACCOUNTS = (
    ("employee@expense-demo.local", "Maya Patel", (RoleName.EMPLOYEE,)),
    ("approver@expense-demo.local", "Jordan Lee", (RoleName.APPROVER,)),
    (
        "dual@expense-demo.local",
        "Alex Morgan",
        (RoleName.EMPLOYEE, RoleName.APPROVER),
    ),
    ("second.employee@expense-demo.local", "Sam Rivera", (RoleName.EMPLOYEE,)),
)


def seed_if_empty(database: Database) -> bool:
    with database.session() as session:
        user_count = session.scalar(select(func.count(User.id))) or 0
        if user_count:
            return False
        _seed(session)
        return True


def reset_demo_data(database: Database) -> None:
    with database.session() as session:
        session.execute(delete(AIAssessment))
        session.execute(delete(ExpenseDocument))
        session.execute(delete(ExpenseClaim))
        session.execute(delete(Category))
        session.execute(delete(UserRole))
        session.execute(delete(User))
        _seed(session)


def _seed(session: Session) -> None:
    password_hasher = PasswordHasher()
    password_hash = password_hasher.hash(DEMO_PASSWORD)
    users: dict[str, User] = {}
    for email, full_name, roles in DEMO_ACCOUNTS:
        user = User(
            email=email,
            full_name=full_name,
            password_hash=password_hash,
            is_active=True,
        )
        user.role_links = [UserRole(role=role) for role in roles]
        session.add(user)
        users[email] = user
    session.flush()

    primary_approver = users["approver@expense-demo.local"]
    dual_user = users["dual@expense-demo.local"]
    categories = {
        "Office": Category(name="Office", approver_id=primary_approver.id),
        "Travel": Category(name="Travel", approver_id=primary_approver.id),
        "Software/Subscriptions": Category(
            name="Software/Subscriptions", approver_id=primary_approver.id
        ),
        "Client Entertainment": Category(
            name="Client Entertainment", approver_id=dual_user.id
        ),
        "Other": Category(name="Other", approver_id=dual_user.id),
    }
    session.add_all(categories.values())
    session.flush()

    today = date.today()
    now = utc_now()
    employee = users["employee@expense-demo.local"]
    second_employee = users["second.employee@expense-demo.local"]

    claims = [
        ExpenseClaim(
            employee_id=employee.id,
            category_id=categories["Office"].id,
            amount_cents=14990,
            description="Ergonomic keyboard and mouse for the home office",
            expense_date=today - timedelta(days=4),
            payment_details="Demo reimbursement account ending 8842",
            status=ExpenseStatus.PENDING,
            created_at=now - timedelta(days=3, hours=2),
            updated_at=now - timedelta(days=3, hours=2),
        ),
        ExpenseClaim(
            employee_id=employee.id,
            category_id=categories["Office"].id,
            amount_cents=78000,
            description="Round-trip flight to London for a partner meeting",
            expense_date=today - timedelta(days=3),
            payment_details="Demo reimbursement account ending 8842",
            status=ExpenseStatus.PENDING,
            created_at=now - timedelta(days=2, hours=5),
            updated_at=now - timedelta(days=2, hours=5),
        ),
        ExpenseClaim(
            employee_id=employee.id,
            category_id=categories["Software/Subscriptions"].id,
            amount_cents=2900,
            description="Monthly Figma subscription for the product design team",
            expense_date=today - timedelta(days=16),
            payment_details="Demo reimbursement account ending 8842",
            status=ExpenseStatus.APPROVED,
            decision_comment="Approved for the current billing cycle.",
            decided_by_id=primary_approver.id,
            decided_at=now - timedelta(days=14),
            created_at=now - timedelta(days=15),
            updated_at=now - timedelta(days=14),
        ),
        ExpenseClaim(
            employee_id=employee.id,
            category_id=categories["Travel"].id,
            amount_cents=95000,
            description="Train and hotel for the Berlin customer workshop",
            expense_date=today - timedelta(days=21),
            payment_details="Demo reimbursement account ending 8842",
            status=ExpenseStatus.REJECTED,
            decision_comment="Please attach the workshop agenda and itemized hotel receipt.",
            decided_by_id=primary_approver.id,
            decided_at=now - timedelta(days=18),
            created_at=now - timedelta(days=20),
            updated_at=now - timedelta(days=18),
        ),
        ExpenseClaim(
            employee_id=employee.id,
            category_id=categories["Other"].id,
            amount_cents=4200,
            description="Parking during an onsite client visit",
            expense_date=today - timedelta(days=10),
            payment_details="Demo reimbursement account ending 8842",
            status=ExpenseStatus.WITHDRAWN,
            withdrawn_at=now - timedelta(days=8),
            created_at=now - timedelta(days=9),
            updated_at=now - timedelta(days=8),
        ),
        ExpenseClaim(
            employee_id=second_employee.id,
            category_id=categories["Office"].id,
            amount_cents=22000,
            description="Printer toner for the Warsaw branch office",
            expense_date=today - timedelta(days=2),
            payment_details="Demo reimbursement account ending 1734",
            status=ExpenseStatus.PENDING,
            created_at=now - timedelta(days=1, hours=4),
            updated_at=now - timedelta(days=1, hours=4),
        ),
        ExpenseClaim(
            employee_id=dual_user.id,
            category_id=categories["Client Entertainment"].id,
            amount_cents=31000,
            description="Dinner with the Acme client team after the quarterly review",
            expense_date=today - timedelta(days=1),
            payment_details="Demo reimbursement account ending 4471",
            status=ExpenseStatus.PENDING,
            created_at=now - timedelta(hours=6),
            updated_at=now - timedelta(hours=6),
        ),
    ]
    session.add_all(claims)
