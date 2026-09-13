from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st

from expense_approval.ai import AssessmentResult, ExpenseAnalyzer, build_ai_payload, payload_hash
from expense_approval.config import Settings, load_settings
from expense_approval.db import Database
from expense_approval.documents import (
    DocumentProcessingError,
    ProcessedDocument,
    process_document,
)
from expense_approval.models import AssessmentSource, ExpenseClaim, ExpenseStatus, RoleName
from expense_approval.seed import DEMO_PASSWORD, reset_demo_data, seed_if_empty
from expense_approval.services import ExpenseApprovalError, ExpenseService, StoredAssessment
from expense_approval.ui import (
    inject_styles,
    render_claim_details,
    render_claim_summary,
    status_badge,
)

st.set_page_config(
    page_title="ExpenseFlow · Expense Approval",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class AppResources:
    settings: Settings
    database: Database
    service: ExpenseService
    analyzer: ExpenseAnalyzer
    executor: ThreadPoolExecutor


@st.cache_resource
def get_resources(settings: Settings) -> AppResources:
    database = Database(settings.database_url)
    database.migrate()
    seed_if_empty(database)
    return AppResources(
        settings=settings,
        database=database,
        service=ExpenseService(database),
        analyzer=ExpenseAnalyzer(
            settings.openai_api_key,
            settings.openai_model,
            settings.ai_timeout_seconds,
        ),
        executor=ThreadPoolExecutor(max_workers=4, thread_name_prefix="expense-ai"),
    )


def main() -> None:
    inject_styles()
    settings = load_settings(_streamlit_secrets())
    resources = get_resources(settings)
    _render_flash()

    user_id = st.session_state.get("user_id")
    if not user_id:
        render_login(resources)
        return

    try:
        user = resources.service.get_user(int(user_id))
    except Exception as exc:
        if not _is_user_safe_error(exc):
            raise
        st.session_state.pop("user_id", None)
        st.rerun()
        return

    render_sidebar(resources, user)
    roles = user.roles
    if roles == {RoleName.EMPLOYEE}:
        render_employee_workspace(resources, user.id)
    elif roles == {RoleName.APPROVER}:
        render_approver_workspace(resources, user.id)
    else:
        view = st.session_state.get("role_view", RoleName.EMPLOYEE.value)
        if view == RoleName.APPROVER.value:
            render_approver_workspace(resources, user.id)
        else:
            render_employee_workspace(resources, user.id)


def _streamlit_secrets() -> dict[str, object]:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _process_document_cached(
    original_name: str, mime_type: str | None, content: bytes
) -> ProcessedDocument:
    return process_document(original_name, mime_type, content)


def render_login(resources: AppResources) -> None:
    st.markdown('<div class="eyebrow">EXPENSEFLOW</div>', unsafe_allow_html=True)
    st.title("Expense approval, minus the friction.")
    st.write(
        "Submit expenses, route them to the right approver, and make informed decisions "
        "with an AI consistency check."
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left, st.container(border=True):
        st.subheader("Sign in to the demo")
        st.caption("Choose a shortcut or enter any demo account below.")
        _initialize_login_fields()
        shortcut_columns = st.columns(3)
        shortcuts = [
            ("Employee", "employee@expense-demo.local"),
            ("Approver", "approver@expense-demo.local"),
            ("Dual role", "dual@expense-demo.local"),
        ]
        for column, (label, email) in zip(shortcut_columns, shortcuts, strict=True):
            with column:
                if st.button(label, use_container_width=True, key=f"shortcut_{email}"):
                    st.session_state.login_email = email
                    st.session_state.login_password = DEMO_PASSWORD
                    st.rerun()

        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button(
                "Sign in", type="primary", use_container_width=True
            )
        if submitted:
            user = resources.service.authenticate(email, password)
            if user is None:
                st.error("Email or password is incorrect.")
            else:
                st.session_state.user_id = user.id
                st.session_state.role_view = (
                    RoleName.EMPLOYEE.value
                    if RoleName.EMPLOYEE in user.roles
                    else RoleName.APPROVER.value
                )
                _set_flash("success", f"Welcome back, {user.full_name}.")
                st.rerun()

        st.markdown(
            f'<div class="demo-note">All demo accounts use password '
            f'<strong>{DEMO_PASSWORD}</strong>.</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="hero">
                <div style="opacity:.8;font-size:.8rem;font-weight:700;letter-spacing:.1em;">
                    HOW IT WORKS
                </div>
                <h2>One clear path from spend to decision.</h2>
                <p>Category-based routing · Private queues · Live status · Human-controlled AI</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("**Demo scenarios**")
            st.markdown(
                """
                1. Submit an expense as **Employee**.
                2. Review the routed claim as **Approver**.
                3. See the AI flag the Office / flight mismatch.
                4. Try both workspaces with the **Dual role** account.
                """
            )
            if resources.analyzer.openai_enabled:
                st.success(f"OpenAI analysis enabled · {resources.settings.openai_model}")
            else:
                st.info("OpenAI key not configured · rule-based fallback is active")

        with st.expander("Reset demo data"):
            st.caption("This removes all changes and restores the original fictional dataset.")
            confirmed = st.checkbox("I understand that all current demo changes will be erased.")
            if st.button("Reset now", disabled=not confirmed, use_container_width=True):
                reset_demo_data(resources.database)
                st.session_state.pop("_ai_jobs", None)
                _set_flash("success", "Demo data has been restored.")
                st.rerun()


def _initialize_login_fields() -> None:
    st.session_state.setdefault("login_email", "employee@expense-demo.local")
    st.session_state.setdefault("login_password", DEMO_PASSWORD)


def render_sidebar(resources: AppResources, user: object) -> None:
    with st.sidebar:
        st.markdown("## ◈ ExpenseFlow")
        st.caption("EXPENSE APPROVAL MVP")
        st.divider()
        st.markdown(f"**{user.full_name}**")
        st.caption(user.email)
        role_labels = " · ".join(role.value.title() for role in sorted(user.roles))
        st.caption(role_labels)

        if len(user.roles) > 1:
            st.divider()
            st.radio(
                "Workspace",
                options=[RoleName.EMPLOYEE.value, RoleName.APPROVER.value],
                format_func=lambda value: {
                    RoleName.EMPLOYEE.value: "My expenses",
                    RoleName.APPROVER.value: "Approval queue",
                }[value],
                key="role_view",
            )

        st.divider()
        ai_label = (
            f"AI: {resources.settings.openai_model}"
            if resources.analyzer.openai_enabled
            else "AI: fallback mode"
        )
        st.caption(ai_label)
        if st.button("Sign out", use_container_width=True):
            _sign_out()


def render_employee_workspace(resources: AppResources, user_id: int) -> None:
    st.markdown('<div class="eyebrow">EMPLOYEE WORKSPACE</div>', unsafe_allow_html=True)
    st.title("My expenses")
    st.caption("Submit a reimbursement request and follow every decision in one place.")

    create_tab, claims_tab = st.tabs(["＋ New expense", "My claims"])
    with create_tab:
        render_claim_form(resources, user_id)
    with claims_tab:
        render_employee_claims(resources, user_id)


def render_claim_form(resources: AppResources, user_id: int) -> None:
    categories = resources.service.list_categories()
    category_by_name = {category.name: category for category in categories}
    nonce = st.session_state.setdefault("claim_form_nonce", 0)
    with st.container(border=True):
        st.subheader("New reimbursement request")
        st.caption(
            "Upload an invoice, receipt, or travel document to prefill the claim, "
            "or enter it manually. All submitted amounts remain fixed in USD."
        )
        uploaded_file = st.file_uploader(
            "Supporting document",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            key=f"claim_document_{nonce}",
            help=(
                "PDF or image, up to 8 MB. The file is processed locally and is not "
                "sent to OpenAI."
            ),
            max_upload_size=8,
        )
        document: ProcessedDocument | None = None
        document_error: str | None = None
        if uploaded_file is not None:
            try:
                with st.spinner("Reading the document locally…"):
                    document = _process_document_cached(
                        uploaded_file.name,
                        uploaded_file.type,
                        uploaded_file.getvalue(),
                    )
            except Exception as exc:
                if not _is_user_safe_error(exc):
                    raise
                document_error = str(exc)
                st.error(document_error)
            else:
                _render_document_extraction(document)

        extraction = document.extraction if document else None
        suggested_category = (
            extraction.category_hint
            if extraction and extraction.category_hint in category_by_name
            else next(iter(category_by_name))
        )
        suggested_amount = (
            float(extraction.amount)
            if extraction and extraction.amount and extraction.currency == "USD"
            else 0.0
        )
        suggested_date = (
            extraction.expense_date
            if extraction and extraction.expense_date and extraction.expense_date <= date.today()
            else date.today()
        )
        suggested_description = extraction.description if extraction else ""
        document_token = document.sha256[:10] if document else "manual"

        with st.form(f"claim_form_{nonce}_{document_token}"):
            first, second = st.columns(2)
            amount = first.number_input(
                "Amount (USD) *",
                min_value=0.0,
                max_value=1_000_000.0,
                value=suggested_amount,
                step=0.01,
            )
            category_options = list(category_by_name)
            category_name = second.selectbox(
                "Category *",
                options=category_options,
                index=category_options.index(suggested_category),
            )
            expense_date = first.date_input(
                "Expense date *", value=suggested_date, max_value=date.today()
            )
            second.text_input("Currency", value="USD", disabled=True)
            description = st.text_area(
                "Description *",
                value=suggested_description,
                placeholder="What was purchased and why was it needed?",
                max_chars=500,
            )
            payment_details = st.text_area(
                "Payment details *",
                placeholder="Demo reimbursement reference only — do not enter real bank details",
                max_chars=500,
            )
            st.warning("Demo environment: never enter real banking or card information.")
            extraction_confirmed = True
            if document is not None and document.extraction.is_expense_evidence:
                extraction_confirmed = st.checkbox(
                    "I reviewed the extracted fields and entered the correct USD amount."
                )
            submitted = st.form_submit_button(
                "Submit for approval",
                type="primary",
                use_container_width=True,
                disabled=bool(
                    document_error
                    or (document and not document.extraction.is_expense_evidence)
                ),
            )
        if submitted:
            if not extraction_confirmed:
                st.error("Review and confirm the document extraction before submitting.")
                return
            try:
                claim = resources.service.create_claim(
                    user_id,
                    amount=f"{amount:.2f}",
                    category_id=category_by_name[category_name].id,
                    description=description,
                    expense_date=expense_date,
                    payment_details=payment_details,
                    document=document,
                )
            except Exception as exc:
                if not _is_user_safe_error(exc):
                    raise
                st.error(str(exc))
            else:
                st.session_state.claim_form_nonce = nonce + 1
                _set_flash(
                    "success",
                    f"{claim.display_id} was submitted to {claim.category.approver.full_name}.",
                )
                st.rerun()


def _render_document_extraction(document: ProcessedDocument) -> None:
    extraction = document.extraction
    if extraction.is_expense_evidence:
        st.success(f"Recognized {extraction.kind.value.replace('_', ' ')}")
    else:
        st.error(f"Rejected document type: {extraction.kind.value.replace('_', ' ')}")

    columns = st.columns(4)
    columns[0].caption("VENDOR")
    columns[0].write(extraction.vendor or "Not found")
    columns[1].caption("DOCUMENT")
    columns[1].write(extraction.document_number or "Not found")
    columns[2].caption("ORIGINAL TOTAL")
    amount_label = (
        f"{extraction.amount:,.2f} {extraction.currency or ''}".strip()
        if extraction.amount
        else "Not found"
    )
    columns[2].write(amount_label)
    columns[3].caption("SUGGESTED CATEGORY")
    columns[3].write(extraction.category_hint)
    st.caption(
        f"{extraction.processing_method} · {round(extraction.confidence * 100)}% confidence · "
        "raw document content is never sent to OpenAI"
    )
    for warning in extraction.warnings:
        st.warning(warning)


@st.fragment(run_every="2s")
def render_employee_claims(resources: AppResources, user_id: int) -> None:
    claims = resources.service.list_employee_claims(user_id)
    counts = resources.service._counts(claims)
    metrics = st.columns(4)
    metrics[0].metric("Total", counts.total)
    metrics[1].metric("Pending", counts.pending)
    metrics[2].metric("Approved", counts.approved)
    metrics[3].metric("Needs attention", counts.rejected)

    status_filter = st.selectbox(
        "Filter by status",
        ["All", "Pending", "Approved", "Rejected", "Withdrawn"],
        key="employee_status_filter",
    )
    if status_filter != "All":
        claims = [claim for claim in claims if claim.status.value == status_filter.lower()]

    if not claims:
        st.info("No claims match this view.")
        return

    for claim in claims:
        with st.container(border=True):
            render_claim_summary(claim)
            with st.expander("View details"):
                render_claim_details(claim)
                if claim.status == ExpenseStatus.PENDING:
                    confirmed = st.checkbox(
                        "I want to withdraw this pending claim.", key=f"withdraw_confirm_{claim.id}"
                    )
                    if st.button(
                        "Withdraw claim",
                        key=f"withdraw_{claim.id}",
                        disabled=not confirmed,
                    ):
                        try:
                            resources.service.withdraw_claim(user_id, claim.id)
                        except Exception as exc:
                            if not _is_user_safe_error(exc):
                                raise
                            st.error(str(exc))
                        else:
                            _set_flash("success", f"{claim.display_id} was withdrawn.")
                            st.rerun()


def render_approver_workspace(resources: AppResources, user_id: int) -> None:
    st.markdown('<div class="eyebrow">APPROVER WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Approval queue")
    st.caption("Review only the claims routed to your assigned categories.")
    render_approval_queue(resources, user_id)


@st.fragment(run_every="2s")
def render_approval_queue(resources: AppResources, user_id: int) -> None:
    claims = resources.service.list_approval_queue(user_id)
    history = resources.service.list_approver_history(user_id)

    header_left, header_right = st.columns([3, 1])
    header_left.subheader(f"{len(claims)} waiting for your decision")
    header_right.metric("Processed", len(history))

    if not claims:
        st.success("You're all caught up. No pending claims are assigned to you.")
    else:
        selected_id = st.session_state.get("selected_approval_claim")
        available_ids = {claim.id for claim in claims}
        if selected_id not in available_ids:
            selected_id = claims[0].id
            st.session_state.selected_approval_claim = selected_id

        list_column, detail_column = st.columns([0.9, 1.45], gap="large")
        with list_column:
            category_filter = st.selectbox(
                "Category",
                ["All", *sorted({claim.category.name for claim in claims})],
                key="queue_category_filter",
            )
            visible_claims = claims
            if category_filter != "All":
                visible_claims = [
                    claim for claim in claims if claim.category.name == category_filter
                ]
            for claim in visible_claims:
                with st.container(border=True):
                    render_claim_summary(claim, show_employee=True)
                    if st.button(
                        "Review claim",
                        key=f"review_{claim.id}",
                        type="primary" if claim.id == selected_id else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.selected_approval_claim = claim.id
                        st.rerun(scope="fragment")

        selected_claim = next((claim for claim in claims if claim.id == selected_id), None)
        if selected_claim is not None:
            with detail_column:
                render_approver_detail(resources, user_id, selected_claim)

    if history:
        with st.expander(f"Decision history · {len(history)} claims"):
            for claim in history[:10]:
                left, middle, right = st.columns([4, 2, 1.4])
                left.write(f"**{claim.display_id}** · {claim.description}")
                middle.write(f"${claim.amount_usd} · {claim.category.name}")
                with right:
                    status_badge(claim.status)


def render_approver_detail(
    resources: AppResources, user_id: int, claim: ExpenseClaim
) -> None:
    with st.container(border=True):
        st.subheader("Claim details")
        render_claim_details(claim, show_employee=True)

    with st.container(border=True):
        st.subheader("Your decision")
        st.caption("AI insight is advisory. Your controls remain available at all times.")
        with st.form(f"decision_{claim.id}"):
            comment = st.text_area(
                "Decision comment",
                placeholder="Required for rejection; optional for approval",
                max_chars=500,
            )
            approve_column, reject_column = st.columns(2)
            approved = approve_column.form_submit_button(
                "Approve", type="primary", use_container_width=True
            )
            rejected = reject_column.form_submit_button("Reject", use_container_width=True)
        if approved or rejected:
            decision = ExpenseStatus.APPROVED if approved else ExpenseStatus.REJECTED
            try:
                resources.service.decide_claim(user_id, claim.id, decision, comment)
            except Exception as exc:
                if not _is_user_safe_error(exc):
                    raise
                st.error(str(exc))
            else:
                st.session_state.pop("selected_approval_claim", None)
                _set_flash("success", f"{claim.display_id} was {decision.value}.")
                st.rerun()

    with st.container(border=True):
        st.subheader("AI consistency check")
        render_ai_assessment(resources, claim)


def render_ai_assessment(resources: AppResources, claim: ExpenseClaim) -> None:
    payload = build_ai_payload(claim)
    input_hash = payload_hash(payload)
    cached = resources.service.get_cached_assessment(claim.id, input_hash)
    if cached is not None and (
        cached.source == AssessmentSource.OPENAI or not resources.analyzer.openai_enabled
    ):
        _display_assessment(cached)
        return

    jobs: dict[str, Future[StoredAssessment]] = st.session_state.setdefault("_ai_jobs", {})
    job_key = f"{claim.id}:{input_hash}:{resources.settings.openai_model}"
    job = jobs.get(job_key)
    if job is None:
        job = resources.executor.submit(
            _analyze_and_store,
            resources,
            claim.id,
            payload,
            input_hash,
        )
        jobs[job_key] = job

    if not job.done():
        st.info("Analyzing amount, category, and description… You can decide without waiting.")
        st.caption("Payment details and personal identity are never sent to the AI provider.")
        return

    try:
        assessment = job.result()
    except Exception:
        fallback = resources.analyzer.analyze(payload)
        assessment = resources.service.save_assessment(
            claim_id=claim.id,
            input_hash=input_hash,
            summary=fallback.summary,
            is_inconsistent=fallback.is_inconsistent,
            reasons=fallback.reasons,
            source=AssessmentSource.FALLBACK,
            model="rule-engine-v1",
            latency_ms=fallback.latency_ms,
        )
    _display_assessment(assessment)


def _analyze_and_store(
    resources: AppResources,
    claim_id: int,
    payload: dict[str, str],
    input_hash: str,
) -> StoredAssessment:
    result: AssessmentResult = resources.analyzer.analyze(payload)
    return resources.service.save_assessment(
        claim_id=claim_id,
        input_hash=input_hash,
        summary=result.summary,
        is_inconsistent=result.is_inconsistent,
        reasons=result.reasons,
        source=result.source,
        model=result.model,
        latency_ms=result.latency_ms,
    )


def _display_assessment(assessment: StoredAssessment) -> None:
    if assessment.is_inconsistent:
        st.markdown(
            '<div class="ai-flag"><strong>Potential inconsistency found</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ai-safe"><strong>No obvious inconsistency found</strong></div>',
            unsafe_allow_html=True,
        )
    st.write(assessment.summary)
    for reason in assessment.reasons:
        st.warning(reason)
    source_label = (
        f"OpenAI · {assessment.model} · {assessment.latency_ms} ms"
        if assessment.source == AssessmentSource.OPENAI
        else "Rule-based fallback · AI unavailable or not configured"
    )
    st.caption(f"{source_label} · Advisory only — no decision was made automatically.")


def _sign_out() -> None:
    for key in (
        "user_id",
        "role_view",
        "selected_approval_claim",
        "_ai_jobs",
        "employee_status_filter",
        "queue_category_filter",
    ):
        st.session_state.pop(key, None)
    _set_flash("success", "You have signed out.")
    st.rerun()


def _set_flash(kind: str, message: str) -> None:
    st.session_state["flash_message"] = (kind, message)


def _render_flash() -> None:
    flash = st.session_state.pop("flash_message", None)
    if not flash:
        return
    kind, message = flash
    if kind == "success":
        st.toast(message, icon="✅")
    else:
        st.toast(message, icon="ℹ️")


def _is_user_safe_error(exc: Exception) -> bool:
    """Recognize display-safe domain errors even across Streamlit module reloads."""
    safe_bases = {
        ("expense_approval.documents", DocumentProcessingError.__name__),
        ("expense_approval.services", ExpenseApprovalError.__name__),
    }
    return any(
        (base.__module__, base.__name__) in safe_bases for base in type(exc).__mro__
    )


if __name__ == "__main__":
    main()
