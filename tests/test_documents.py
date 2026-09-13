from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal

import pytest

from expense_approval.documents import (
    DocumentExtraction,
    DocumentKind,
    DocumentProcessingError,
    ProcessedDocument,
    analyze_document_text,
    process_document,
    sanitize_document_text_for_ai,
)
from expense_approval.services import AccessDeniedError, ExpenseService, ValidationError

EMPLOYEE = "employee@expense-demo.local"
APPROVER = "approver@expense-demo.local"
DUAL = "dual@expense-demo.local"
SECOND_EMPLOYEE = "second.employee@expense-demo.local"


def test_ukrainian_electricity_invoice_is_extracted() -> None:
    extraction = analyze_document_text(
        """
        ТОВ "Прикарпатенерготрейд"
        Рахунок за електроенергію № 2050003904260819 від 04.09.2026
        Сума до сплати 14 192,03 грн
        """,
        "electricity.pdf",
    )

    assert extraction.kind == DocumentKind.INVOICE
    assert extraction.is_expense_evidence is True
    assert extraction.vendor == 'ТОВ "Прикарпатенерготрейд"'
    assert extraction.amount == Decimal("14192.03")
    assert extraction.currency == "UAH"
    assert extraction.expense_date == date(2026, 9, 4)
    assert extraction.category_hint == "Office"
    assert any("conversion" in warning for warning in extraction.warnings)


def test_written_invoice_date_wins_over_contract_date() -> None:
    extraction = analyze_document_text(
        """
        ПрАТ "Мульті Весте Україна 3"
        Рахунок № Ар.ЛФ.1026 від 10 вересня 2026
        Договір оренди від 15.01.2024
        Всього до оплати 184 211,68 грн
        """,
        "rent.pdf",
    )

    assert extraction.expense_date == date(2026, 9, 10)
    assert extraction.amount == Decimal("184211.68")
    assert extraction.category_hint == "Office"


def test_internet_invoice_routes_to_software_subscriptions() -> None:
    extraction = analyze_document_text(
        """
        ТОВ "Неоком-Плюс"
        Рахунок № 42 від 01.09.2026
        Послуги доступу до мережі Інтернет
        Всього із ПДВ 300,00 грн
        """,
        "internet.pdf",
    )

    assert extraction.kind == DocumentKind.INVOICE
    assert extraction.category_hint == "Software/Subscriptions"
    assert extraction.amount == Decimal("300.00")


def test_unknown_spanish_invoice_fields_are_extracted_without_vendor_allowlist() -> None:
    extraction = analyze_document_text(
        """
        FACTURA
        Punto de Venta: 00001
        Comp. Nro: 00000047
        Razón Social:
        Fecha de Emisión: 01/08/2022
        Servicios según contrato
        Cesión de comisiones
        Moneda: USD - Dólar Estadounidense
        Importe Total: USD
        11243,07
        """,
        "Comprobante-en-USd.jpg",
        "local OCR",
    )

    assert extraction.kind == DocumentKind.INVOICE
    assert extraction.is_expense_evidence is True
    assert extraction.vendor is None
    assert extraction.document_number == "00000047"
    assert extraction.amount == Decimal("11243.07")
    assert extraction.currency == "USD"
    assert extraction.expense_date == date(2022, 8, 1)
    assert extraction.category_hint == "Other"


def test_document_text_is_sanitized_before_text_only_ai_analysis() -> None:
    sanitized = sanitize_document_text_for_ai(
        "Vendor: Example LLC\nIBAN: XX001234567890123456\n"
        "Email: person@example.com\nInvoice 00000047\nTotal USD 25.00"
    )

    assert "Example LLC" in sanitized
    assert "IBAN" not in sanitized
    assert "person@example.com" not in sanitized
    assert "00000047" in sanitized


def test_boarding_pass_is_evidence_but_needs_manual_amount() -> None:
    extraction = analyze_document_text(
        "BOARDING PASS\nAir Company\nFlight No AB 123\nBoarding time 10:30\n07.05.2018",
        "boarding-pass.webp",
        "local OCR",
    )

    assert extraction.kind == DocumentKind.BOARDING_PASS
    assert extraction.is_expense_evidence is True
    assert extraction.category_hint == "Travel"
    assert extraction.amount is None
    assert any("manually" in warning for warning in extraction.warnings)
    assert any("OCR" in warning for warning in extraction.warnings)


@pytest.mark.parametrize(
    ("text", "filename", "kind"),
    [
        (
            "PROFILE SKILLS EXPERIENCE EDUCATION CERTIFICATIONS AI ENGINEER",
            "resume.pdf",
            DocumentKind.RESUME,
        ),
        (
            "Regulamin Vinted Pay Terms and Conditions policy",
            "terms.pdf",
            DocumentKind.POLICY,
        ),
        (
            "NETFLIX PLANS GET PRICE CUT OLD PRICE NEW PRICE",
            "promotion.jpeg",
            DocumentKind.PROMOTIONAL,
        ),
    ],
)
def test_non_expense_documents_are_rejected(
    text: str, filename: str, kind: DocumentKind
) -> None:
    extraction = analyze_document_text(text, filename)

    assert extraction.kind == kind
    assert extraction.is_expense_evidence is False
    assert extraction.amount is None


def test_file_content_is_validated_instead_of_extension() -> None:
    with pytest.raises(DocumentProcessingError, match="genuine PDF"):
        process_document("invoice.pdf", "application/pdf", b"not really a PDF")


def test_service_stores_validated_document(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    content = b"synthetic-invoice"
    extraction = DocumentExtraction(
        kind=DocumentKind.INVOICE,
        is_expense_evidence=True,
        vendor="Example Office Supply",
        document_number="INV-42",
        amount=Decimal("25.50"),
        currency="USD",
        expense_date=date.today(),
        category_hint="Office",
        description="Office expense invoice from Example Office Supply",
        confidence=0.91,
        processing_method="embedded PDF text",
        warnings=(),
    )
    document = ProcessedDocument(
        original_name="invoice.pdf",
        mime_type="application/pdf",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        extraction=extraction,
    )

    claim = service.create_claim(
        ids[EMPLOYEE],
        amount="25.50",
        category_id=ids["Office"],
        description=extraction.description,
        expense_date=date.today(),
        payment_details="Demo payment reference",
        document=document,
    )

    assert claim.document is not None
    assert claim.document.content == content
    assert claim.document.sha256 == document.sha256
    assert claim.document.document_kind == DocumentKind.INVOICE.value
    assert service.get_claim_for_approver(ids[APPROVER], claim.id).document is not None
    with pytest.raises(AccessDeniedError):
        service.get_claim_for_employee(ids[SECOND_EMPLOYEE], claim.id)
    with pytest.raises(AccessDeniedError):
        service.get_claim_for_approver(ids[DUAL], claim.id)


def test_service_rejects_non_expense_document(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    content = b"synthetic-resume"
    document = ProcessedDocument(
        original_name="resume.pdf",
        mime_type="application/pdf",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        extraction=analyze_document_text(
            "PROFILE SKILLS EXPERIENCE EDUCATION CERTIFICATIONS AI ENGINEER",
            "resume.pdf",
        ),
    )

    with pytest.raises(ValidationError, match="not valid expense evidence"):
        service.create_claim(
            ids[EMPLOYEE],
            amount="25.50",
            category_id=ids["Office"],
            description="A legitimate-looking manual description",
            expense_date=date.today(),
            payment_details="Demo payment reference",
            document=document,
        )


def test_service_rejects_document_with_invalid_hash(
    service: ExpenseService, ids: dict[str, int]
) -> None:
    content = b"synthetic-invoice"
    document = ProcessedDocument(
        original_name="invoice.pdf",
        mime_type="application/pdf",
        size_bytes=len(content),
        sha256="0" * 64,
        content=content,
        extraction=analyze_document_text(
            "Invoice No INV-42 Total amount 25.50 USD",
            "invoice.pdf",
        ),
    )

    with pytest.raises(ValidationError, match="integrity check"):
        service.create_claim(
            ids[EMPLOYEE],
            amount="25.50",
            category_id=ids["Office"],
            description="A legitimate-looking manual description",
            expense_date=date.today(),
            payment_details="Demo payment reference",
            document=document,
        )
