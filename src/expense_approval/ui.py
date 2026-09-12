from __future__ import annotations

from datetime import datetime

import streamlit as st

from expense_approval.models import ExpenseClaim, ExpenseStatus

STATUS_LABELS = {
    ExpenseStatus.PENDING: "Pending",
    ExpenseStatus.APPROVED: "Approved",
    ExpenseStatus.REJECTED: "Rejected",
    ExpenseStatus.WITHDRAWN: "Withdrawn",
}

STATUS_ICONS = {
    ExpenseStatus.PENDING: "◷",
    ExpenseStatus.APPROVED: "✓",
    ExpenseStatus.REJECTED: "×",
    ExpenseStatus.WITHDRAWN: "↩",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7f6; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #153f38;
            border-right: 0;
        }
        [data-testid="stSidebar"] * { color: #f4fbf8; }
        [data-testid="stSidebar"] .stButton button {
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.2);
            color: white;
        }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { letter-spacing: -0.03em; color: #173b34; }
        h1 { font-size: 2.25rem !important; }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e0e8e5;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 28px rgba(24,64,54,.04);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #dde7e3;
            border-radius: 16px;
            background: white;
            box-shadow: 0 10px 32px rgba(24,64,54,.045);
        }
        .eyebrow {
            color: #2f6b5f;
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }
        .muted { color: #64756f; }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            border-radius: 999px;
            padding: .25rem .65rem;
            font-size: .78rem;
            font-weight: 700;
        }
        .status-pending { color: #895b07; background: #fff4d6; }
        .status-approved { color: #1c684f; background: #def5ea; }
        .status-rejected { color: #9a3030; background: #fde7e7; }
        .status-withdrawn { color: #5d6670; background: #edf0f2; }
        .claim-id { color: #70807b; font-size: .78rem; font-weight: 700; }
        .claim-amount {
            color: #173b34;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -.025em;
            text-align: right;
            white-space: nowrap;
        }
        .hero {
            padding: 1.4rem 1.5rem;
            color: white;
            border-radius: 20px;
            background: linear-gradient(135deg, #143f37 0%, #2f6b5f 72%, #4e8b79 100%);
            box-shadow: 0 18px 40px rgba(21,63,56,.16);
            margin-bottom: 1.4rem;
        }
        .hero h2 { color: white !important; margin: 0 0 .35rem; }
        .hero p { color: #d8ebe5; margin: 0; }
        .ai-safe {
            border-left: 4px solid #2f8b67;
            background: #edf9f4;
            padding: .85rem 1rem;
            border-radius: 0 12px 12px 0;
        }
        .ai-flag {
            border-left: 4px solid #d68b1c;
            background: #fff7e7;
            padding: .85rem 1rem;
            border-radius: 0 12px 12px 0;
        }
        .demo-note {
            background: #e9f2ef;
            border: 1px solid #d4e4df;
            padding: .8rem 1rem;
            border-radius: 12px;
            color: #31574e;
            font-size: .88rem;
        }
        .stButton button, .stFormSubmitButton button { border-radius: 10px; font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: ExpenseStatus) -> None:
    label = STATUS_LABELS[status]
    icon = STATUS_ICONS[status]
    st.markdown(
        f'<span class="status-badge status-{status.value}">{icon} {label}</span>',
        unsafe_allow_html=True,
    )


def render_claim_summary(claim: ExpenseClaim, *, show_employee: bool = False) -> None:
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(f'<div class="claim-id">{claim.display_id}</div>', unsafe_allow_html=True)
        st.markdown(f"**{claim.description}**")
    with top_right:
        st.markdown(
            f'<div class="claim-amount">${claim.amount_usd}</div>', unsafe_allow_html=True
        )
    details = f"{claim.category.name} · {claim.expense_date.strftime('%b %d, %Y')}"
    if show_employee:
        details = f"{claim.employee.full_name} · {details}"
    st.caption(details)
    status_badge(claim.status)


def render_claim_details(claim: ExpenseClaim, *, show_employee: bool = False) -> None:
    columns = st.columns(3)
    columns[0].caption("CLAIM")
    columns[0].write(claim.display_id)
    columns[1].caption("AMOUNT")
    columns[1].write(f"${claim.amount_usd} USD")
    columns[2].caption("STATUS")
    with columns[2]:
        status_badge(claim.status)

    st.caption("DESCRIPTION")
    st.write(claim.description)
    left, right = st.columns(2)
    left.caption("CATEGORY")
    left.write(claim.category.name)
    right.caption("EXPENSE DATE")
    right.write(claim.expense_date.strftime("%B %d, %Y"))
    if show_employee:
        st.caption("SUBMITTED BY")
        st.write(f"{claim.employee.full_name} · {claim.employee.email}")

    with st.expander("Payment details", icon="💳"):
        st.warning("Demo data only. Never enter real banking or card information here.")
        st.write(claim.payment_details)

    if claim.decision_comment:
        st.caption("DECISION COMMENT")
        st.info(claim.decision_comment)

    st.caption(f"Last updated {format_datetime(claim.updated_at)}")


def format_datetime(value: datetime) -> str:
    return value.strftime("%b %d, %Y · %H:%M UTC")
