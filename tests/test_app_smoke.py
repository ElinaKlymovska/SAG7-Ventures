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
