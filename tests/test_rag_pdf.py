"""Разбор PDF и chunking: разделы, страницы, длинные разделы с перекрытием, ошибки."""
import pymupdf
import pytest

from datastory.rag.pdf_parser import PdfError, chunk_document, document_id_for, parse_pdf
from scripts import generate_demo_data as gen
from tests.helpers import make_pdf

DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()

EXPECTED_TITLES = [
    "DemoPay KZ — описание бизнес-показателей",
    "1. Описание платёжной системы",
    "2. Описание колонок исходного датасета",
    "3. Определение показателя Success Rate",
    "4. Формула Success Rate",
    "5. Определение Transaction Volume",
    "6. Определение Average Transaction Amount",
    "7. Формула Average Transaction Amount",
    "8. Описание каналов",
    "9. Контекстное событие",
]


def sentences(count: int) -> str:
    return " ".join(f"Предложение номер {i} описывает показатель качества обработки платежей." for i in range(count))


# ------------------------------------------------------------------ демо-документ
def test_demo_pdf_sections_and_pages():
    parsed = parse_pdf(DEMO_PDF, "business_metrics.pdf")
    assert parsed.page_count == 2
    assert parsed.section_titles == EXPECTED_TITLES
    pages = {s.title: s.page_start for s in parsed.sections}
    assert pages["4. Формула Success Rate"] == 1
    assert pages["8. Описание каналов"] == pages["9. Контекстное событие"] == 2


def test_formulas_are_not_mistaken_for_headings():
    titles = parse_pdf(DEMO_PDF, "b.pdf").section_titles
    assert not any("=" in t for t in titles)


def test_demo_chunks_one_per_section_with_metadata():
    parsed = parse_pdf(DEMO_PDF, "business_metrics.pdf")
    chunks = chunk_document(parsed, dataset_id="abc123abc123")
    assert [c.section for c in chunks] == EXPECTED_TITLES
    assert [c.chunk_index for c in chunks] == list(range(10))
    assert len({c.chunk_id for c in chunks}) == 10
    assert all(c.chunk_id == f"{parsed.document_id}-{c.chunk_index:04d}" for c in chunks)
    assert all(c.document_id == parsed.document_id and c.filename == "business_metrics.pdf" for c in chunks)
    assert all(c.dataset_id == "abc123abc123" for c in chunks)
    assert all(c.page in (1, 2) and c.page_end >= c.page for c in chunks)


def test_key_definitions_are_inside_the_right_chunks():
    chunks = {c.section: c.text for c in chunk_document(parse_pdf(DEMO_PDF, "b.pdf"))}
    assert "Success Rate = Successful / Transactions × 100" in chunks["4. Формула Success Rate"]
    assert "Average Transaction Amount = Amount_KZT / Transactions" in chunks["7. Формула Average Transaction Amount"]
    assert "денежный объём платежей" in chunks["5. Определение Transaction Volume"]
    assert "не доказывает причинную связь" in chunks["9. Контекстное событие"]


def test_bullets_stay_on_separate_lines():
    chunks = {c.section: c.text for c in chunk_document(parse_pdf(DEMO_PDF, "b.pdf"))}
    lines = chunks["8. Описание каналов"].split("\n")
    assert [line.split(" — ")[0] for line in lines] == ["• Mobile", "• Web", "• API"]


def test_document_id_is_content_hash():
    assert document_id_for(DEMO_PDF) == document_id_for(bytes(DEMO_PDF))
    assert document_id_for(DEMO_PDF) != document_id_for(DEMO_PDF + b"\n")
    assert parse_pdf(DEMO_PDF, "a.pdf").document_id == parse_pdf(DEMO_PDF, "renamed.pdf").document_id


# ------------------------------------------------------------------ длинные разделы
def test_long_section_is_split_with_overlap():
    parsed = parse_pdf(make_pdf([("h", "1. Длинный раздел"), ("p", sentences(40))]), "long.pdf")
    chunks = chunk_document(parsed, max_chars=400, overlap=100)

    assert len(chunks) > 3
    assert {c.section for c in chunks} == {"1. Длинный раздел"}
    assert all(len(c.text) <= 400 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = previous.text.rsplit(". ", 1)[-1]
        assert last_sentence in current.text, "соседние фрагменты должны перекрываться"


def test_split_does_not_lose_text():
    body = sentences(30)
    chunks = chunk_document(parse_pdf(make_pdf([("h", "1. Раздел"), ("p", body)]), "l.pdf"), max_chars=350, overlap=80)
    combined = " ".join(c.text for c in chunks)
    for i in range(30):
        assert f"Предложение номер {i} " in combined


def test_zero_overlap_gives_disjoint_chunks():
    parsed = parse_pdf(make_pdf([("h", "1. Раздел"), ("p", sentences(30))]), "l.pdf")
    texts = [c.text for c in chunk_document(parsed, max_chars=300, overlap=0)]
    assert len(texts) > 3 and len(set(texts)) == len(texts)
    assert not any(a.rsplit(". ", 1)[-1] in b for a, b in zip(texts, texts[1:]))


def test_sections_are_not_mixed_in_one_chunk():
    pdf = make_pdf([("h", "1. Первый"), ("p", "Текст первого раздела."), ("h", "2. Второй"), ("p", "Текст второго раздела.")])
    chunks = chunk_document(parse_pdf(pdf, "two.pdf"))
    assert [(c.section, c.text) for c in chunks] == [
        ("1. Первый", "Текст первого раздела."),
        ("2. Второй", "Текст второго раздела."),
    ]


def test_section_spanning_pages_records_page_range():
    parsed = parse_pdf(make_pdf([("h", "1. Большой раздел"), ("p", sentences(60))]), "pages.pdf")
    assert parsed.page_count >= 2
    chunks = chunk_document(parsed, max_chars=4000, overlap=100)
    assert chunks[0].page == 1
    assert any(c.page_end > c.page for c in chunks) or chunks[-1].page > 1


def test_page_numbers_in_footer_are_ignored():
    pdf = make_pdf([("h", "1. Раздел"), ("p", "Основной текст."), ("break", ""), ("p", "Продолжение текста.")], page_footer=True)
    text = " ".join(c.text for c in chunk_document(parse_pdf(pdf, "f.pdf")))
    assert "стр." not in text
    assert "Основной текст." in text and "Продолжение текста." in text


def test_text_before_first_heading_is_kept_without_section():
    pdf = make_pdf([("p", "Вступление без заголовка."), ("h", "1. Раздел"), ("p", "Текст.")])
    chunks = chunk_document(parse_pdf(pdf, "x.pdf"))
    assert chunks[0].section is None and chunks[0].text == "Вступление без заголовка."
    assert chunks[1].section == "1. Раздел"


def test_bold_numbered_headings_of_body_size_are_detected():
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for text, font in (("1. Terms", "hebo"), ("Term definition.", "helv"), ("2. Formulas", "hebo"), ("Formula text.", "helv")):
        page.insert_text((72, y), text, fontname=font, fontsize=11)
        y += 30
    parsed = parse_pdf(doc.tobytes(), "same_size.pdf")
    assert parsed.section_titles == ["1. Terms", "2. Formulas"]


def test_invalid_chunk_parameters():
    parsed = parse_pdf(DEMO_PDF, "b.pdf")
    for max_chars, overlap in ((100, 100), (100, 200), (0, 0), (100, -1)):
        with pytest.raises(ValueError):
            chunk_document(parsed, max_chars=max_chars, overlap=overlap)


# ------------------------------------------------------------------ ошибки
@pytest.mark.parametrize("data", [b"", b"this is not a pdf", b"%PDF-1.7\ngarbage"])
def test_unreadable_pdf_raises_plain_error(data):
    with pytest.raises(PdfError) as exc:
        parse_pdf(data, "bad.pdf")
    assert exc.value.user_message


def test_password_protected_pdf():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner")
    with pytest.raises(PdfError, match="паролем"):
        parse_pdf(data, "locked.pdf")


def test_pdf_without_text_layer_is_reported():
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(PdfError, match="текстового слоя"):
        parse_pdf(doc.tobytes(), "scan.pdf")
