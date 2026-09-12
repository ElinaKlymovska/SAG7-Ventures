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

