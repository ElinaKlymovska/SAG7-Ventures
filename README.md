# ExpenseFlow — Expense Approval MVP

ExpenseFlow is an English-language, role-based expense approval demo built with Python and
Streamlit. Employees submit USD reimbursement claims, the system routes each category to its
configured approver, and approvers receive an advisory AI consistency check before making the
final decision. An optional supporting-document intake reads invoices, receipts, and travel
documents locally to prefill the claim.

**Live demo:** [sag7-expense-approval.streamlit.app](https://sag7-expense-approval.streamlit.app/)

The AI never approves, rejects, or blocks a claim. If OpenAI is slow, unavailable, or not
configured, the app immediately remains usable and labels its deterministic fallback clearly.

## What is included

- Employee and approver roles; one user can hold both roles.
- Five routed expense categories.
- `pending`, `approved`, `rejected`, and `withdrawn` workflows.
- Mandatory rejection comments and owner-only withdrawal.
- Object-level access checks in the service layer, not just UI filtering.
- PDF, JPG, PNG, and WebP intake with embedded-text extraction and local OCR.
- Multilingual local OCR plus consent-based AI extraction for unfamiliar document layouts.
- Dynamic vendor, document number, total, currency, date, and business-category suggestions.
- Two-second live refresh for employee status and approver queues.
- OpenAI Responses API integration with structured output.
- Editable AI suggestion for a non-sensitive payment/reimbursement reference.
- Rule-based fallback and cached assessments.
- Fictional seed data, reset control, migrations, tests, and CI.

## Run locally

Requirements: Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

The final command, `streamlit run app.py`, is the single command used to launch the app. It
creates `data/expense_approval.sqlite3`, applies the Alembic migrations, and loads the demo
dataset automatically. No API key is required; without one, the AI panel is explicitly labeled
as a rule-based fallback. Image and scanned-PDF OCR additionally requires Poppler and Tesseract;
Community Cloud installs them from `packages.txt`.

To enable OpenAI analysis:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then add `OPENAI_API_KEY` to `.streamlit/secrets.toml`. The default model is `gpt-5-mini` and can
be changed with `OPENAI_MODEL`. The real secrets file is ignored by Git.

## Demo accounts

All demo accounts use password `Demo123!`.

| Account | Roles | Assigned categories |
|---|---|---|
| `employee@expense-demo.local` | Employee | — |
| `approver@expense-demo.local` | Approver | Office, Travel, Software/Subscriptions |
| `dual@expense-demo.local` | Employee + Approver | Client Entertainment, Other |
| `second.employee@expense-demo.local` | Employee | — |

Passwords are stored only as Argon2 hashes. The login shortcuts intentionally expose the shared
password because every account and claim is fictional and the deployment is a public test demo.

## Five-minute review script

1. Sign in as `employee@expense-demo.local`. Open **My claims** and verify all four statuses.
2. Confirm that the second employee's printer-toner claim is not visible.
3. Open **New expense** and upload an invoice. Verify the suggested vendor, original total,
   category, date, and description. For a non-USD invoice, enter the converted USD amount
   manually; the MVP never invents an exchange rate. For an unfamiliar layout, consent and click
   **Analyze or improve fields with AI**. Then click **Suggest payment details with AI**, review
   the non-sensitive reimbursement reference, and edit it if needed.
4. Submit the expense, then open a second private browser window.
5. Sign in there as `approver@expense-demo.local`; the claim appears within two seconds. Its
   supporting file is available only inside the authorized claim view.
6. Open the seeded `$780.00` Office claim describing a London flight. The AI or labeled fallback
   flags the category mismatch while both decision buttons remain active.
7. Try Reject without a comment, then reject with a reason. The employee view updates within two
   seconds.
8. Sign in as `dual@expense-demo.local` and switch between both workspaces. Self-approval is
   intentionally allowed for this MVP, as specified in the implementation brief.
9. Sign out, expand **Reset demo data**, confirm, and restore the initial dataset.

## Supporting-document flow

The browser-provided filename and MIME type are treated as untrusted. The backend validates the
file signature, applies an 8 MB and 25-page limit, and then extracts embedded PDF text or runs
local Tesseract OCR with Spanish, English, Ukrainian, Polish, and Russian language data. Images
are normalized, enlarged, contrast-adjusted, and sharpened before OCR.

```text
upload → signature/size validation → local text extraction/OCR → evidence classification
       → optional consent-based AI extraction → dynamic fields + configured routing category
       → employee review → submit
```

Invoices and receipts are accepted; boarding passes are accepted with a manual-amount warning.
Promotional graphics, resumes, policies/terms, unreadable files, and unrelated documents are
rejected as evidence. AI can analyze unfamiliar languages and layouts, propose a free-form
business category, and separately map it to one configured routing category. Missing source
fields remain `Not found` rather than being invented. Extraction is advisory: the employee
reviews every field before submission. A detected UAH, PLN, or INR total is displayed in its
original currency, but the employee must enter the correct USD claim amount because currency
conversion is outside the MVP.

The raw attachment is never stored. Once the fields are extracted, the file is discarded and the
claim keeps only the extracted values, the original filename, and a checksum, which the employee
and the assigned approver see on the claim. Local extraction never calls an external service. Optional AI extraction is
explicitly consent-based: images are sent to OpenAI vision, while PDFs send locally extracted
text after common banking identifiers, long IDs, addresses, and emails are removed. Payment
details and user identity are never included. Responses use `store=false`. The supplied example
archive is intentionally not included in this public repository because it contains personal and
payment information; automated tests use synthetic equivalents.

## Architecture

```mermaid
flowchart LR
    D[Document upload] --> X[Local parser / OCR]
    X --> UI[Streamlit UI]
    X -. consent .-> DX[OpenAI document extraction]
    DX --> UI
    UI --> S[ExpenseService]
    S --> DB[(SQLite / SQLAlchemy)]
    UI --> W[Background AI worker]
    W --> OA[OpenAI Responses API]
    W --> FB[Rule fallback]
    W --> DB
```

- `app.py` contains the Streamlit composition, session identity, live fragments, and background
  AI job coordination.
- `src/expense_approval/services.py` is the authorization and workflow boundary. Every read and
  mutation checks the acting user again.
- `src/expense_approval/documents.py` validates uploads, extracts text locally, classifies
  evidence, and produces reviewable field suggestions. It never calls an external service.
- `src/expense_approval/ai.py` has three bounded Structured Output flows. Approver assessment and
  payment-reference suggestions receive only `amount_usd`, `category`, `description`, and
  `expense_date`. Consent-based document extraction receives an image or sanitized OCR text plus
  configured routing category names. It never receives existing payment details or user identity.
- SQLAlchemy stores money as integer cents and Alembic owns the schema.
- SQLite runs in WAL mode with a five-second busy timeout. Status decisions use an atomic
  conditional update, so a stale second decision cannot overwrite the first.

### State transitions

```text
pending ──approve──> approved
pending ──reject───> rejected   (comment required)
pending ──withdraw─> withdrawn  (claim owner only)
```

Resolved claims cannot transition again.

### AI behavior

When an assigned approver opens a claim, a background worker asks the OpenAI Responses API for:

```json
{
  "summary": "One or two short sentences",
  "is_inconsistent": false,
  "reasons": []
}
```

The request has a configurable ten-second timeout, no retries, `store=false`, minimal reasoning,
and a strict Pydantic response schema. The OpenAI client and its connection pool are reused. The
result is cached against a SHA-256 hash of the four allowed input fields. API errors fall back to
transparent keyword and amount checks. Neither path can mutate claim status.

## Test and quality checks

```bash
ruff check .
pytest --cov=expense_approval
```

The suite covers validation, routing, dual roles, access isolation, all transitions, mandatory
rejection comments, concurrent decisions, document classification and extraction, attachment
integrity, the OCR contract with Tesseract and Poppler, AI payload privacy, money parsing across
separator styles, structured response parsing, fallback behavior,
caching, sanitized OCR, dynamic AI document fields and category routing, editable payment-detail
suggestions, and Streamlit workflow smoke tests.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io), create an app from the repository
   and set `app.py` as the entrypoint.
3. Select Python 3.12. Community Cloud installs the OCR system packages declared in the root
   `packages.txt` during deployment.
4. Add the following in the app's **Secrets** settings:

   ```toml
   OPENAI_API_KEY = "your-key"
   OPENAI_MODEL = "gpt-5-mini"
   AI_TIMEOUT_SECONDS = 10.0
   ```

5. Share the generated `streamlit.app` URL together with the repository and demo credentials.

Never commit `.streamlit/secrets.toml` or an API key.

## Deliberate MVP boundaries

- SQLite is appropriate for this reviewable demo, not for a production financial system. The
  local database may reset when a Community Cloud instance is rebuilt.
- "Live" status uses a two-second Streamlit fragment refresh, not database push subscriptions.
- Attachments are stored as SQLite BLOBs and share the demo database's ephemeral lifecycle. A
  production version should use private object storage, malware scanning, retention policies,
  and an audit trail.
- Payment details are synthetic free text and are visible only to the claimant and assigned
  approver. The UI warns users not to enter real financial data.
- Registration, editing submitted claims, notifications, currency conversion, SSO, audit
  exports, and production database operations are out of scope.
