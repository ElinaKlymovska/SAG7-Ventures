from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_login_page_renders(tmp_path: Path, monkeypatch: object) -> None:
    database_url = f"sqlite:///{tmp_path / 'ui.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    assert any("Expense approval" in title.value for title in app.title)
    assert any(button.label == "Sign in" for button in app.button)


def test_employee_page_renders_document_uploader(
    tmp_path: Path, monkeypatch: object
) -> None:
    database_url = f"sqlite:///{tmp_path / 'employee-ui.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    next(button for button in app.button if button.label == "Employee").click().run()
    next(button for button in app.button if button.label == "Sign in").click().run()

    assert not app.exception
    assert any(title.value == "My expenses" for title in app.title)
    uploaders = app.get("file_uploader")
    assert len(uploaders) == 1
    assert uploaders[0].label == "Supporting document"


def test_claim_validation_is_rendered_without_app_error(
    tmp_path: Path, monkeypatch: object
) -> None:
    database_url = f"sqlite:///{tmp_path / 'validation-ui.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    next(button for button in app.button if button.label == "Employee").click().run()
    next(button for button in app.button if button.label == "Sign in").click().run()
    next(field for field in app.number_input if field.label == "Amount (USD) *").set_value(
        25.00
    ).run()
    next(field for field in app.text_area if field.label == "Description *").set_value(
        "Office supplies for the demo"
    ).run()
    next(
        button for button in app.button if button.label == "Submit for approval"
    ).click().run()

    assert not app.exception
    assert any(
        "Payment details must be at least 5 characters" in error.value
        for error in app.error
    )


def test_payment_details_can_be_suggested_and_edited(
    tmp_path: Path, monkeypatch: object
) -> None:
    database_url = f"sqlite:///{tmp_path / 'payment-suggestion-ui.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    next(button for button in app.button if button.label == "Employee").click().run()
    next(button for button in app.button if button.label == "Sign in").click().run()
    next(field for field in app.number_input if field.label == "Amount (USD) *").set_value(
        25.00
    ).run()
    next(field for field in app.text_area if field.label == "Description *").set_value(
        "Office supplies for the demo"
    ).run()
    next(
        button
        for button in app.button
        if button.label == "Suggest payment details with AI"
    ).click().run()

    assert not app.exception
    payment_field = next(
        field for field in app.text_area if field.label == "Payment details *"
    )
    assert payment_field.value.startswith(
        "Reimbursement reference: Client Entertainment expense dated "
    )
    assert payment_field.value.endswith("for $25.00 USD.")
    payment_field.set_value("Edited demo reimbursement reference").run()
    assert next(
        field for field in app.text_area if field.label == "Payment details *"
    ).value == "Edited demo reimbursement reference"


def test_dual_role_user_can_switch_workspaces(
    tmp_path: Path, monkeypatch: object
) -> None:
    database_url = f"sqlite:///{tmp_path / 'dual-ui.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    next(button for button in app.button if button.label == "Dual role").click().run()
    next(button for button in app.button if button.label == "Sign in").click().run()

    assert not app.exception
    workspace = next(radio for radio in app.radio if radio.label == "Workspace")
    assert workspace.value == "employee"
    workspace.set_value("approver").run()
    assert not app.exception
    assert any(title.value == "Approval queue" for title in app.title)
