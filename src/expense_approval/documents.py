from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pypdf import PdfReader

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 25
MAX_EXTRACTED_CHARACTERS = 100_000


class DocumentProcessingError(Exception):
    """A document error that is safe to display to the user."""


class DocumentKind(StrEnum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    BOARDING_PASS = "boarding_pass"
    PROMOTIONAL = "promotional"
    POLICY = "policy"
    RESUME = "resume"
    OTHER = "other"


@dataclass(frozen=True)
class DocumentExtraction:
    kind: DocumentKind
    is_expense_evidence: bool
    vendor: str | None
    document_number: str | None
    amount: Decimal | None
    currency: str | None
    expense_date: date | None
    category_hint: str
    description: str
    confidence: float
    processing_method: str
    warnings: tuple[str, ...]
    routing_category: str | None = None


@dataclass(frozen=True)
class ProcessedDocument:
    original_name: str
    mime_type: str
    size_bytes: int
    sha256: str
    content: bytes
    extraction: DocumentExtraction
    extracted_text: str = ""


def process_document(
    original_name: str,
    declared_mime_type: str | None,
    content: bytes,
) -> ProcessedDocument:
    del declared_mime_type  # File extensions and browser-provided MIME types are not trusted.
    clean_name = _clean_filename(original_name)
    if not content:
        raise DocumentProcessingError("The uploaded file is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentProcessingError("The supporting document must be 8 MB or smaller.")

    mime_type = _detect_mime_type(content)
    if mime_type == "application/pdf":
        text, processing_method = _extract_pdf_text(content)
    else:
        text = _ocr_image(content)
        processing_method = "local OCR"

    extraction = analyze_document_text(text, clean_name, processing_method)
    return ProcessedDocument(
        original_name=clean_name,
        mime_type=mime_type,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        extraction=extraction,
        extracted_text=_normalize_text(text)[:MAX_EXTRACTED_CHARACTERS],
    )


def analyze_document_text(
    text: str,
    original_name: str,
    processing_method: str = "embedded PDF text",
) -> DocumentExtraction:
    normalized = _normalize_text(text)[:MAX_EXTRACTED_CHARACTERS]
    searchable = f"{original_name}\n{normalized}".casefold()
    kind, is_evidence = _classify_document(searchable)
    category = _category_hint(searchable) if is_evidence else "Other"
    vendor = _extract_vendor(normalized, searchable)
    document_number = _extract_document_number(normalized)
    amount = _extract_payable_amount(normalized) if is_evidence else None
    currency = _detect_currency(searchable)
    expense_date = _extract_expense_date(normalized) if is_evidence else None
    description = _build_description(
        kind=kind,
        category=category,
        vendor=vendor,
        document_number=document_number,
        searchable=searchable,
    )

    warnings: list[str] = []
    if not is_evidence:
        warnings.append(
            "This file is not an invoice, receipt, or travel document and cannot support a claim."
        )
    else:
        if amount is None:
            warnings.append("No single payable total was found; enter the USD amount manually.")
        if currency and currency != "USD":
            warnings.append(
                f"Detected {currency}. Currency conversion is outside this MVP; "
                "enter the converted USD amount manually."
            )
        if expense_date is None:
            warnings.append("The expense date could not be extracted; verify it manually.")
        if vendor is None:
            warnings.append("The vendor could not be extracted; verify the description manually.")
        if "OCR" in processing_method:
            warnings.append("Image text was extracted locally with OCR; verify every field.")

    completeness = sum(value is not None for value in (vendor, amount, expense_date))
    confidence = 0.15 if not is_evidence else min(0.95, 0.55 + completeness * 0.12)
    return DocumentExtraction(
        kind=kind,
        is_expense_evidence=is_evidence,
        vendor=vendor,
        document_number=document_number,
        amount=amount,
        currency=currency,
        expense_date=expense_date,
        category_hint=category,
        description=description,
        confidence=confidence,
        processing_method=processing_method,
        warnings=tuple(warnings),
        routing_category=category,
    )


def _detect_mime_type(content: bytes) -> str:
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    raise DocumentProcessingError(
        "Only genuine PDF, JPG, JPEG, PNG, and WebP files are supported."
    )


def _extract_pdf_text(content: bytes) -> tuple[str, str]:
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise DocumentProcessingError("Password-protected PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentProcessingError(f"PDFs are limited to {MAX_PDF_PAGES} pages.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError("The PDF could not be read safely.") from exc

    if len(text.strip()) >= 80:
        return text, "embedded PDF text"
    return _ocr_pdf(content), "local PDF OCR"


def _ocr_pdf(content: bytes) -> str:
    if shutil.which("pdftoppm") is None:
        raise DocumentProcessingError("This scanned PDF needs OCR, which is unavailable.")
    with tempfile.TemporaryDirectory(prefix="expense-doc-") as temporary_directory:
        directory = Path(temporary_directory)
        input_path = directory / "document.pdf"
        output_prefix = directory / "page"
        input_path.write_bytes(content)
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    "1",
                    "-l",
                    "3",
                    "-scale-to",
                    "2200",
                    "-png",
                    str(input_path),
                    str(output_prefix),
                ],
                capture_output=True,
                check=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DocumentProcessingError("The scanned PDF could not be prepared for OCR.") from exc
        page_text = [_ocr_image(path.read_bytes()) for path in sorted(directory.glob("page-*.png"))]
    text = "\n".join(page_text).strip()
    if not text:
        raise DocumentProcessingError("No readable text was found in the scanned PDF.")
    return text


def _ocr_image(content: bytes) -> str:
    if shutil.which("tesseract") is None:
        raise DocumentProcessingError("Local image OCR is unavailable on this deployment.")
    prepared_content = _prepare_image_for_ocr(content)
    try:
        result = subprocess.run(
            [
                "tesseract",
                "stdin",
                "stdout",
                "-l",
                "spa+eng+ukr+pol+rus",
                "--psm",
                "11",
            ],
            input=prepared_content,
            capture_output=True,
            check=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise DocumentProcessingError("Image OCR took too long. Try a smaller image.") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise DocumentProcessingError("The image could not be read by local OCR.") from exc
    text = result.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        raise DocumentProcessingError("No readable text was found in the image.")
    return text


def _prepare_image_for_ocr(content: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            if source.width * source.height > 30_000_000:
                raise DocumentProcessingError("The image dimensions are too large for OCR.")
            image = ImageOps.exif_transpose(source).convert("L")
    except DocumentProcessingError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise DocumentProcessingError("The image could not be read safely.") from exc

    longest_edge = max(image.size)
    if longest_edge < 2_000:
        scale = min(3.0, 2_000 / longest_edge)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    image = ImageOps.autocontrast(image).filter(ImageFilter.SHARPEN)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _classify_document(searchable: str) -> tuple[DocumentKind, bool]:
    resume_markers = (
        "curriculum vitae",
        "ai engineer",
        "profile",
        "skills",
        "experience",
        "education",
        "certifications",
    )
    if sum(marker in searchable for marker in resume_markers) >= 3:
        return DocumentKind.RESUME, False
    if "terms and conditions" in searchable or (
        "regulamin" in searchable and any(word in searchable for word in ("terms", "policy"))
    ):
        return DocumentKind.POLICY, False
    if "netflix plans" in searchable or (
        "old price" in searchable and "new price" in searchable
    ):
        return DocumentKind.PROMOTIONAL, False
    if any(marker in searchable for marker in ("boarding pass", "boarding time", "flight no")):
        return DocumentKind.BOARDING_PASS, True
    if any(marker in searchable for marker in ("receipt", "квитанц", "paragon")):
        return DocumentKind.RECEIPT, True
    if any(marker in searchable for marker in ("recibo", "comprobante de pago")):
        return DocumentKind.RECEIPT, True
    if any(
        marker in searchable
        for marker in (
            "рахунок",
            "invoice",
            "faktura",
            "акт-рахунок",
            "factura",
            "importe total",
        )
    ):
        return DocumentKind.INVOICE, True
    return DocumentKind.OTHER, False


def _category_hint(searchable: str) -> str:
    category_keywords: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "Travel",
            ("boarding pass", "boarding time", "flight", "airfare", "hotel", "train ticket"),
        ),
        (
            "Client Entertainment",
            ("restaurant", "dinner", "client entertainment", "catering"),
        ),
        (
            "Software/Subscriptions",
            (
                "internet",
                "інтернет",
                "телеком",
                "lifecell",
                "укртелеком",
                "subscription",
                "software",
                "netflix",
                "мобільн",
                "зв’язку",
                "зв'язку",
            ),
        ),
        (
            "Office",
            (
                "електроенерг",
                "оренд",
                "експлуатаційн",
                "office",
                "printer",
                "marketing fee",
                "маркетинговий збір",
            ),
        ),
    )
    for category, keywords in category_keywords:
        if any(keyword in searchable for keyword in keywords):
            return category
    return "Other"


def _extract_vendor(text: str, searchable: str) -> str | None:
    known_vendors = (
        ("прикарпатенерготрейд", 'ТОВ "Прикарпатенерготрейд"'),
        ("лайфселл", "lifecell"),
        ("lifecell", "lifecell"),
        ("укртелеком", 'АТ "Укртелеком"'),
        ("мульті весте україна 3", 'ПрАТ "Мульті Весте Україна 3"'),
        ("атлантік-пасіфік венчурз", 'ТОВ "Атлантік-Пасіфік Венчурз"'),
        ("неоком-плюс", 'ТОВ "Неоком-Плюс"'),
        ("air company", "Air Company"),
    )
    for marker, display_name in known_vendors:
        if marker in searchable:
            return display_name

    pattern = re.compile(
        r"(?:Постачальник|Одержувач|Raz[oó]n Social|Proveedor|Vendor)"
        r"[ \t]*:?[ \t]*([^\n]{3,160})",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    candidate = re.sub(r"\s+", " ", match.group(1)).strip(" :-")
    return candidate[:120] or None


def _extract_document_number(text: str) -> str | None:
    patterns = (
        r"(?:Рахунок(?:-акт| на оплату| за [^\n№]{0,60})?|Invoice)\s*№\s*([\w./-]+)",
        r"АКТ-РАХУНОК[^\n]{0,100}?№\s*([\w./-]+)",
        r"Comp\.?\s*Nro\.?\s*:?\s*([\w./-]+)",
        r"(?:Factura|Comprobante)\s*(?:Nro\.?|No\.?|#)\s*:?\s*([\w./-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(".,")[:80]
    return None


MONEY_PATTERN = re.compile(
    r"(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d{2,4})|\d{1,9}[.,]\d{2,4})(?!\d)"
)


def _extract_payable_amount(text: str) -> Decimal | None:
    labels = (
        r"усього найменувань[^\n]{0,100}?на суму",
        r"сума до сплати",
        r"всього до оплати",
        r"до сплати з пдв(?:,?\s*грн\.?)?",
        r"всього із пдв",
        r"всього до сплати",
        r"загалом,?\s*враховуючи пдв та пф",
        r"враховуючи пдв та пф",
        r"total(?: amount| due)?",
        r"importe total",
        r"importe neto gravado",
        r"разом",
    )
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    for label in labels:
        values: list[Decimal] = []
        pattern = re.compile(label, re.IGNORECASE)
        for index, line in enumerate(lines):
            match = pattern.search(line)
            if not match:
                continue
            value_text = " ".join((line[match.end() :], *lines[index + 1 : index + 3]))
            for raw_value in MONEY_PATTERN.findall(value_text):
                parsed = _parse_money(raw_value)
                if parsed is not None:
                    values.append(parsed)
        if values:
            return max(values)
    return None


def _parse_money(value: str) -> Decimal | None:
    compact = value.replace(" ", "").replace("\u00a0", "")
    if "," in compact and "." in compact:
        decimal_separator = "," if compact.rfind(",") > compact.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        compact = compact.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif "," in compact:
        compact = compact.replace(",", ".")
    try:
        amount = Decimal(compact).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
    return amount if amount > 0 else None


def _detect_currency(searchable: str) -> str | None:
    if re.search(r"\bUAH\b", searchable, re.IGNORECASE) or any(
        marker in searchable for marker in ("грн", "грив")
    ):
        return "UAH"
    if re.search(r"\bUSD\b", searchable, re.IGNORECASE) or "$" in searchable:
        return "USD"
    if re.search(r"\bPLN\b", searchable, re.IGNORECASE) or "zł" in searchable:
        return "PLN"
    if re.search(r"\bINR\b", searchable, re.IGNORECASE) or "₹" in searchable:
        return "INR"
    return None


UKRAINIAN_MONTHS = {
    "січня": 1,
    "лютого": 2,
    "березня": 3,
    "квітня": 4,
    "травня": 5,
    "червня": 6,
    "липня": 7,
    "серпня": 8,
    "вересня": 9,
    "жовтня": 10,
    "листопада": 11,
    "грудня": 12,
}


def _extract_expense_date(text: str) -> date | None:
    month_names = "|".join(UKRAINIAN_MONTHS)
    written_invoice_date = re.search(
        rf"(?:Рахунок|Invoice)[^\n]{{0,140}}?(\d{{1,2}})\s+({month_names})\s+(\d{{4}})",
        text,
        re.IGNORECASE,
    )
    if written_invoice_date:
        try:
            return date(
                int(written_invoice_date.group(3)),
                UKRAINIAN_MONTHS[written_invoice_date.group(2).casefold()],
                int(written_invoice_date.group(1)),
            )
        except ValueError:
            pass

    numeric_patterns = (
        r"(?:Рахунок|Invoice)[^\n]{0,120}?(?:від|date)?\s*(\d{2}[./-]\d{2}[./-]\d{4})",
        r"Дата(?: формування)?[^:\n]{0,60}:?\s*(\d{2}[./-]\d{2}[./-]\d{4})",
        r"(?:від|dated?)\s*(\d{2}[./-]\d{2}[./-]\d{4})",
        r"Fecha de Emisi[oó]n\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{4})",
    )
    for pattern in numeric_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed:
                return parsed

    match = re.search(rf"(\d{{1,2}})\s+({month_names})\s+(\d{{4}})", text, re.IGNORECASE)
    if match:
        try:
            return date(
                int(match.group(3)),
                UKRAINIAN_MONTHS[match.group(2).casefold()],
                int(match.group(1)),
            )
        except ValueError:
            return None
    generic_numeric = re.search(r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b", text)
    if generic_numeric:
        return _parse_date(generic_numeric.group(1))
    return None


def _parse_date(value: str) -> date | None:
    parts = re.split(r"[./-]", value)
    if len(parts) != 3:
        return None
    try:
        return date(int(parts[2]), int(parts[1]), int(parts[0]))
    except ValueError:
        return None


def _build_description(
    *,
    kind: DocumentKind,
    category: str,
    vendor: str | None,
    document_number: str | None,
    searchable: str,
) -> str:
    if kind == DocumentKind.RESUME:
        return "Resume detected; this is not supporting evidence for an expense claim"
    if kind == DocumentKind.POLICY:
        return "Terms or policy document detected; this is not expense evidence"
    if kind == DocumentKind.PROMOTIONAL:
        return "Promotional price graphic detected; no completed purchase is evidenced"
    if kind == DocumentKind.OTHER:
        return "Unrecognized document; enter the claim manually or upload expense evidence"
    if kind == DocumentKind.BOARDING_PASS:
        return "Air travel boarding pass; verify route, date, and USD expense amount"
    if "електроенерг" in searchable:
        subject = "Electricity services"
    elif any(marker in searchable for marker in ("internet", "телеком", "lifecell")):
        subject = "Internet or telecommunications services"
    elif "оренд" in searchable:
        subject = "Office rent and operating costs"
    elif any(marker in searchable for marker in ("servicios", "comisi", "factura")):
        subject = "Professional services invoice"
    elif kind == DocumentKind.RECEIPT:
        subject = f"{category} purchase receipt"
    else:
        subject = f"{category} expense invoice"
    details = [subject]
    if vendor:
        details.append(f"from {vendor}")
    if document_number:
        details.append(f"document {document_number}")
    return " ".join(details)[:500]


def _normalize_text(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def _clean_filename(filename: str) -> str:
    clean = Path(filename).name.replace("\x00", "").strip()
    if not clean:
        return "supporting-document"
    return clean[:255]


def sanitize_document_text_for_ai(text: str) -> str:
    """Remove common financial and personal identifiers before text-only AI analysis."""
    sensitive_line_markers = (
        "iban",
        "swift",
        "routing number",
        "account number",
        "bank account",
        "card number",
        "номер рахунку",
        "банківські реквізити",
        "domicilio comercial",
        "cuit",
        "tax id",
    )
    safe_lines = [
        line
        for line in _normalize_text(text).splitlines()
        if not any(marker in line.casefold() for marker in sensitive_line_markers)
    ]
    sanitized = "\n".join(safe_lines)
    sanitized = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email redacted]", sanitized)
    sanitized = re.sub(r"(?<!\d)\d{9,}(?!\d)", "[long identifier redacted]", sanitized)
    return sanitized[:20_000]
