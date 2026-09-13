# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ExpenseFlow is a role-based expense approval MVP: Streamlit UI, SQLAlchemy/SQLite storage,
and an advisory OpenAI consistency check that can never decide a claim.

## Commands

Python 3.12 is required.

```bash
pip install -r requirements-dev.txt
streamlit run app.py           # creates data/expense_approval.sqlite3, migrates, seeds demo data
ruff check .                   # E,F,I,B,UP,SIM · line-length 100
pytest --cov=expense_approval  # same pair of commands CI runs
```

Single test: `pytest tests/test_services.py::test_authentication_and_dual_roles`.
`pythonpath = ["src"]` is set in `[tool.pytest.ini_options]`, so no `pip install -e .` is needed.

New migration: `alembic revision --autogenerate -m "..."` from the repo root
(`alembic.ini` sets `prepend_sys_path = . src`).

## Architecture

- [app.py](app.py) — the entire Streamlit composition in one module. It inserts `src/` into
  `sys.path` before its imports, which is why `app.py` carries the ruff `E402` exemption.
  `Database`, `ExpenseService`, `ExpenseAnalyzer` and a `ThreadPoolExecutor` live in a single
  `AppResources` behind `@st.cache_resource`.
- [src/expense_approval/services.py](src/expense_approval/services.py) — **the authorization
  boundary**. Every public method re-checks the actor via `_require_role` and narrows its query
  by `employee_id` / `Category.approver_id`. UI filtering is not protection: add new data
  operations here, not in `app.py`.
- [src/expense_approval/models.py](src/expense_approval/models.py) — SQLAlchemy 2.0
  (`Mapped`/`mapped_column`). Money exists only as `amount_cents: int`; display goes through
  `ExpenseClaim.amount_usd` and `display_id`.
- [src/expense_approval/db.py](src/expense_approval/db.py) — the `session()` context manager
  (commit/rollback/close), SQLite in WAL mode with `busy_timeout=5000`, and `migrate()`, which
  runs `alembic upgrade head` programmatically at startup.
- [src/expense_approval/ui.py](src/expense_approval/ui.py) — stateless presentation only
  (styles, status badges, claim cards).
- [src/expense_approval/seed.py](src/expense_approval/seed.py) — the fictional dataset. Tests
  address rows by their exact description strings (see the `ids` fixture in
  [tests/conftest.py](tests/conftest.py)), so editing seed text breaks tests.

### Invariants to preserve

- **Status transitions are atomic.** `decide_claim` and `withdraw_claim` issue
  `UPDATE ... WHERE status = 'pending'` and check `rowcount != 1`. That is what stops a second
  concurrent decision from overwriting the first — do not replace it with read-modify-write.
- **The AI privacy boundary.** `build_ai_payload` in
  [src/expense_approval/ai.py](src/expense_approval/ai.py) is the only place data leaves the
  system, and it emits exactly four fields: `amount_usd`, `category`, `description`,
  `expense_date`. Never add `payment_details` or identity —
  `test_ai_payload_excludes_identity_and_payment_details` guards this.
- **Document images are opt-in.** `analyze_document` sends an `input_image` only when
  `send_document_images` is set (`AI_SEND_DOCUMENT_IMAGES`); otherwise images go as
  `sanitize_document_text_for_ai` output, like PDFs. An image is the one payload no sanitizer can
  redact, so the default is off.
- **The model cannot grant expense evidence.** `_merge_document_extraction` computes
  `local.is_expense_evidence and parsed.is_expense_evidence` — a downgrade-only rule. Uploaded
  document text reaches the model as input, so this is what stops a prompt injection inside a
  file from clearing the `_validate_document` gate in `services.py`. Never take
  `parsed.is_expense_evidence` directly.
- **The AI is advisory only.** Neither the OpenAI path nor `rule_based_assessment` mutates claim
  status, and both decision buttons stay enabled. `ExpenseAnalyzer.analyze` never raises: any
  failure becomes a fallback result with `source=AssessmentSource.FALLBACK`.
- **Assessment cache** is keyed by `(claim_id, sha256(payload))` via `payload_hash`, enforced by
  the `uq_assessment_claim_input` constraint. `save_assessment` upgrades an existing fallback row
  to an OpenAI result, but never the reverse.
- **Background jobs.** `render_ai_assessment` stores a `Future` in
  `st.session_state["_ai_jobs"]` under `claim_id:hash:model` so the 2-second fragments do not
  re-submit a request on every rerender. Demo reset and sign-out clear that key.
- **"Live" lists** are `@st.fragment(run_every="2s")`, not database push — everything inside such
  a fragment re-executes every two seconds.

## Configuration

`load_settings()` in [src/expense_approval/config.py](src/expense_approval/config.py) reads
`st.secrets` first, then environment variables: `DATABASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
(default `gpt-5-mini`), `AI_TIMEOUT_SECONDS`, `AI_SEND_DOCUMENT_IMAGES` (default `false`).

Without `OPENAI_API_KEY` the app is fully functional and labels the AI panel as a rule-based
fallback — keep the key optional. `.streamlit/secrets.toml` is gitignored; the template is
`.streamlit/secrets.example.toml`.

## Related docs

Demo accounts, the five-minute review script, and the deliberate MVP boundaries live in
[README.md](README.md) — keep it in sync when behavior changes.
