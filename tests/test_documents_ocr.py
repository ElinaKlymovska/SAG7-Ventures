"""Tests for the OCR branch of the document pipeline.

These cover the contract with the external binaries — the argv we build, the timeouts
we set, and how every failure turns into a display-safe DocumentProcessingError — plus
the pure Pillow preparation step. Recognition quality itself needs real tesseract data
and stays a manual check from the README demo script, so nothing here shells out: CI
installs neither tesseract nor poppler.
"""

from __future__ import annotations

import io
import subprocess
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from expense_approval.documents import (
    DocumentProcessingError,
    _extract_pdf_text,
    _prepare_image_for_ocr,
    process_document,
)

RECEIPT_TEXT = b"ACME OFFICE SUPPLY\nInvoice No 42\nTotal 25.00 USD\n04.09.2026"


def _png_bytes(width: int = 400, height: int = 120, color: int = 255) -> bytes:
    """Build a real PNG so the genuine Pillow path runs."""
    buffer = io.BytesIO()
    Image.new("L", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_which(monkeypatch: object, missing: str | None = None) -> None:
    monkeypatch.setattr(
        "expense_approval.documents.shutil.which",
        lambda name: None if name == missing else f"/usr/bin/{name}",
    )


def test_image_ocr_extracts_fields(monkeypatch: object) -> None:
    _fake_which(monkeypatch)
    monkeypatch.setattr(
        "expense_approval.documents.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=RECEIPT_TEXT),
    )

    document = process_document("receipt.png", None, _png_bytes())

    assert document.mime_type == "image/png"
    assert document.extraction.processing_method == "local OCR"
    assert document.extraction.amount == Decimal("25.00")
    assert document.extraction.document_number == "42"


def test_image_ocr_receives_prepared_png_and_language_packs(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(stdout=RECEIPT_TEXT)

    _fake_which(monkeypatch)
    monkeypatch.setattr("expense_approval.documents.subprocess.run", fake_run)

    process_document("receipt.png", None, _png_bytes())

    assert captured["argv"][:3] == ["tesseract", "stdin", "stdout"]
    assert "spa+eng+ukr+pol+rus" in captured["argv"]
    assert captured["argv"][-2:] == ["--psm", "11"]
    assert captured["timeout"] == 15
    # The prepared grayscale PNG is sent, never the caller's original bytes.
    assert isinstance(captured["input"], bytes)
    assert captured["input"].startswith(b"\x89PNG\r\n\x1a\n")


def test_missing_tesseract_is_reported_safely(monkeypatch: object) -> None:
    _fake_which(monkeypatch, missing="tesseract")

    with pytest.raises(DocumentProcessingError, match="Local image OCR is unavailable"):
        process_document("receipt.png", None, _png_bytes())


def test_image_ocr_timeout_is_reported_safely(monkeypatch: object) -> None:
    def timing_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["tesseract"], timeout=15)

    _fake_which(monkeypatch)
    monkeypatch.setattr("expense_approval.documents.subprocess.run", timing_out)

    with pytest.raises(DocumentProcessingError, match="took too long"):
        process_document("receipt.png", None, _png_bytes())


def test_image_ocr_process_failure_is_reported_safely(monkeypatch: object) -> None:
    def failing(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "tesseract")

    _fake_which(monkeypatch)
    monkeypatch.setattr("expense_approval.documents.subprocess.run", failing)

    with pytest.raises(DocumentProcessingError, match="could not be read by local OCR"):
        process_document("receipt.png", None, _png_bytes())


def test_blank_ocr_output_is_reported_safely(monkeypatch: object) -> None:
    _fake_which(monkeypatch)
    monkeypatch.setattr(
        "expense_approval.documents.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"   \n  "),
    )

    with pytest.raises(DocumentProcessingError, match="No readable text was found in the image"):
        process_document("receipt.png", None, _png_bytes())


def _pdf_with_pages(page_count: int, *, encrypted: bool = False, text: str = "") -> object:
    class FakePage:
        def extract_text(self) -> str:
            return text

    class FakeReader:
        def __init__(self, *_args: object, **_kwargs: object):
            self.is_encrypted = encrypted
            self.pages = [FakePage() for _ in range(page_count)]

    return FakeReader


def test_encrypted_pdf_is_rejected(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "expense_approval.documents.PdfReader", _pdf_with_pages(1, encrypted=True)
    )

    with pytest.raises(DocumentProcessingError, match="Password-protected PDFs"):
        _extract_pdf_text(b"%PDF-1.4 stub")


def test_pdf_page_limit_is_enforced(monkeypatch: object) -> None:
    monkeypatch.setattr("expense_approval.documents.PdfReader", _pdf_with_pages(26))

    with pytest.raises(DocumentProcessingError, match="limited to 25 pages"):
        _extract_pdf_text(b"%PDF-1.4 stub")


def test_embedded_pdf_text_skips_ocr(monkeypatch: object) -> None:
    """A text PDF must never shell out; which() returning None proves OCR was skipped."""
    long_enough = (
        "ACME OFFICE SUPPLY LIMITED\n"
        "Invoice No 42 issued on 04.09.2026\n"
        "Total 25.00 USD payable within thirty days"
    )
    monkeypatch.setattr(
        "expense_approval.documents.PdfReader", _pdf_with_pages(1, text=long_enough)
    )
    _fake_which(monkeypatch, missing="pdftoppm")

    text, method = _extract_pdf_text(b"%PDF-1.4 stub")

    assert method == "embedded PDF text"
    assert "Invoice No 42" in text


def _rendering_run(page_count: int = 1) -> object:
    """Stand in for pdftoppm by writing the page images it would have produced."""

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "pdftoppm":
            prefix = Path(argv[-1])
            for page in range(1, page_count + 1):
                prefix.with_name(f"{prefix.name}-{page}.png").write_bytes(_png_bytes())
            return SimpleNamespace(stdout=b"")
        return SimpleNamespace(stdout=RECEIPT_TEXT)

    return fake_run


def test_scanned_pdf_falls_back_to_ocr(monkeypatch: object) -> None:
    monkeypatch.setattr("expense_approval.documents.PdfReader", _pdf_with_pages(1, text=""))
    _fake_which(monkeypatch)
    monkeypatch.setattr("expense_approval.documents.subprocess.run", _rendering_run())

    text, method = _extract_pdf_text(b"%PDF-1.4 stub")

    assert method == "local PDF OCR"
    assert "Invoice No 42" in text


def test_pdf_rendering_is_capped_at_three_pages(monkeypatch: object) -> None:
    captured: dict[str, object] = {}
    render = _rendering_run()

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "pdftoppm":
            captured["argv"] = argv
            captured.update(kwargs)
        return render(argv, **kwargs)

    monkeypatch.setattr("expense_approval.documents.PdfReader", _pdf_with_pages(10, text=""))
    _fake_which(monkeypatch)
    monkeypatch.setattr("expense_approval.documents.subprocess.run", fake_run)

    _extract_pdf_text(b"%PDF-1.4 stub")

    argv = captured["argv"]
    assert argv[argv.index("-f") + 1] == "1"
    assert argv[argv.index("-l") + 1] == "3"
    assert argv[argv.index("-scale-to") + 1] == "2200"
    assert captured["timeout"] == 20


def test_missing_pdftoppm_is_reported_safely(monkeypatch: object) -> None:
    monkeypatch.setattr("expense_approval.documents.PdfReader", _pdf_with_pages(1, text=""))
    _fake_which(monkeypatch, missing="pdftoppm")

    with pytest.raises(DocumentProcessingError, match="needs OCR, which is unavailable"):
        _extract_pdf_text(b"%PDF-1.4 stub")


def test_pdf_rendering_failure_is_reported_safely(monkeypatch: object) -> None:
    def failing(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "pdftoppm")

    monkeypatch.setattr("expense_approval.documents.PdfReader", _pdf_with_pages(1, text=""))
    _fake_which(monkeypatch)
    monkeypatch.setattr("expense_approval.documents.subprocess.run", failing)

    with pytest.raises(DocumentProcessingError, match="could not be prepared for OCR"):
        _extract_pdf_text(b"%PDF-1.4 stub")


def test_unreadable_pdf_is_reported_safely() -> None:
    with pytest.raises(DocumentProcessingError, match="could not be read safely"):
        _extract_pdf_text(b"%PDF-1.4 truncated garbage")


def test_oversized_image_is_rejected() -> None:
    with pytest.raises(DocumentProcessingError, match="dimensions are too large"):
        _prepare_image_for_ocr(_png_bytes(6_000, 6_000))


def test_small_image_is_upscaled_for_ocr() -> None:
    prepared = _prepare_image_for_ocr(_png_bytes(200, 100))

    with Image.open(io.BytesIO(prepared)) as image:
        # The 2000px target is capped at a 3x enlargement.
        assert image.size == (600, 300)
        assert image.mode == "L"


def test_large_image_keeps_its_size() -> None:
    prepared = _prepare_image_for_ocr(_png_bytes(2_400, 1_000))

    with Image.open(io.BytesIO(prepared)) as image:
        assert image.size == (2_400, 1_000)


def test_corrupt_image_is_reported_safely() -> None:
    with pytest.raises(DocumentProcessingError, match="could not be read safely"):
        _prepare_image_for_ocr(b"\x89PNG\r\n\x1a\n" + b"garbage" * 20)
