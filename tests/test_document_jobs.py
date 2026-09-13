"""Tests for the background document reader behind the claim form.

The OCR work runs in a worker thread, so these check the handover: the form sees a
pending state instead of blocking, a finished job is cached, a display-safe failure is
kept as a message, and the cache does not grow without bound.
"""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

from expense_approval.documents import DocumentProcessingError


def _load_app() -> ModuleType:
    """Import app.py under its own name, without running Streamlit's page setup."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("expense_app_under_test", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app()


@dataclass
class FakeSessionState(dict):
    """The two dict methods _document_job_state relies on."""

    def setdefault(self, key: str, default: object) -> object:
        return super().setdefault(key, default)


@pytest.fixture
def session(monkeypatch: object) -> dict:
    state: dict = FakeSessionState()
    monkeypatch.setattr(app.st, "session_state", state)
    return state


@pytest.fixture
def resources() -> SimpleNamespace:
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        yield SimpleNamespace(document_executor=executor)
    finally:
        executor.shutdown(wait=True)


def _read(resources: object, digest: str = "abc123") -> object:
    return app._document_job_state(resources, digest, "invoice.pdf", None, b"bytes")


def test_slow_document_reports_pending_instead_of_blocking(
    session: dict, resources: SimpleNamespace, monkeypatch: object
) -> None:
    release = Event()
    monkeypatch.setattr(
        app, "process_document", lambda *_args: release.wait(5) or "parsed"
    )

    state = _read(resources)

    assert state.pending is True
    assert state.document is None and state.error is None
    release.set()


def test_finished_document_is_returned_and_cached(
    session: dict, resources: SimpleNamespace, monkeypatch: object
) -> None:
    calls: list[str] = []

    def fake_process(name: str, *_args: object) -> str:
        calls.append(name)
        return "parsed-document"

    monkeypatch.setattr(app, "process_document", fake_process)

    while (state := _read(resources)).pending:
        pass

    assert state.document == "parsed-document"
    assert state.error is None
    # A second read is served from the cache rather than re-running the parser.
    assert _read(resources).document == "parsed-document"
    assert calls == ["invoice.pdf"]
    assert session[app.DOCUMENT_JOBS_KEY] == {}


def test_safe_failure_is_kept_as_a_message(
    session: dict, resources: SimpleNamespace, monkeypatch: object
) -> None:
    def failing(*_args: object) -> None:
        raise DocumentProcessingError("The PDF could not be read safely.")

    monkeypatch.setattr(app, "process_document", failing)

    while (state := _read(resources)).pending:
        pass

    assert state.document is None
    assert state.error == "The PDF could not be read safely."
    assert _read(resources).error == "The PDF could not be read safely."


def test_unexpected_failure_is_raised(
    session: dict, resources: SimpleNamespace, monkeypatch: object
) -> None:
    def exploding(*_args: object) -> None:
        raise RuntimeError("a genuine bug")

    monkeypatch.setattr(app, "process_document", exploding)

    with pytest.raises(RuntimeError, match="a genuine bug"):
        while _read(resources).pending:
            pass


def test_cached_documents_are_capped(
    session: dict, resources: SimpleNamespace, monkeypatch: object
) -> None:
    monkeypatch.setattr(app, "process_document", lambda *_args: "parsed")

    for index in range(app.MAX_CACHED_DOCUMENTS + 3):
        while _read(resources, digest=f"digest-{index}").pending:
            pass

    results = session[app.DOCUMENT_RESULTS_KEY]
    assert len(results) == app.MAX_CACHED_DOCUMENTS
    # The oldest entries are the ones dropped.
    assert f"{app.DOCUMENT_PROCESSOR_VERSION}:digest-0" not in results
    assert f"{app.DOCUMENT_PROCESSOR_VERSION}:digest-10" in results
