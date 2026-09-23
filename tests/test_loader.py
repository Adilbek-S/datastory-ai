"""Загрузка XLSX/CSV: определение формата, листы, кодировки и понятные ошибки."""
import pytest

from datastory.errors import (
    CorruptFileError,
    DataLoadError,
    EmptyFileError,
    NoDataError,
    UnsupportedFileError,
)
from datastory.file_processing.loader import detect_format, list_sheets, read_table
from tests.helpers import make_xlsx

GOOD_XLSX = make_xlsx({"Sales": [["a", "b"], [1, 2], [3, 4]], "Plan": [["x"], ["p"], ["q"], ["r"]]})


# ------------------------------------------------------------------ формат
def test_detect_format():
    assert detect_format("data.csv", b"a,b\n1,2\n") == "csv"
    assert detect_format("DATA.XLSX", GOOD_XLSX) == "xlsx"


@pytest.mark.parametrize("name", ["notes.txt", "data.json", "old.xls", "archive.zip", "noext"])
def test_unsupported_format(name):
    with pytest.raises(UnsupportedFileError) as exc:
        detect_format(name, b"some content")
    assert "XLSX или CSV" in exc.value.user_message


def test_image_is_reported_as_not_supported_yet():
    with pytest.raises(UnsupportedFileError, match="изображениях"):
        read_table(b"\x89PNG", "table.png")


# ------------------------------------------------------------------ пустые и повреждённые файлы
@pytest.mark.parametrize("name", ["a.csv", "a.xlsx"])
@pytest.mark.parametrize("data", [b"", b"   \n\n  "])
def test_empty_file(name, data):
    with pytest.raises(EmptyFileError, match="пустой"):
        read_table(data, name)


def test_csv_with_header_only_has_no_data():
    with pytest.raises(NoDataError, match="нет данных"):
        read_table(b"a,b,c\n", "a.csv")


def test_csv_with_only_empty_rows_has_no_data():
    with pytest.raises(NoDataError):
        read_table(b"a,b\n,\n,\n", "a.csv")


@pytest.mark.parametrize("data", [b"this is not an excel file", b"PK\x03\x04broken-zip-content"])
def test_corrupt_xlsx(data):
    with pytest.raises(CorruptFileError) as exc:
        read_table(data, "broken.xlsx")
    assert "повреждён" in exc.value.user_message


def test_binary_file_named_csv_is_corrupt():
    with pytest.raises(CorruptFileError):
        read_table(b"\x00\x01\x02binary\x00\x00" * 50, "fake.csv")
    with pytest.raises(CorruptFileError):
        read_table(GOOD_XLSX, "excel_renamed.csv")


def test_ragged_csv_reports_parse_error():
    with pytest.raises(CorruptFileError, match="разобрать"):
        read_table(b"a,b\n1,2\n1,2,3,4\n", "ragged.csv")


def test_all_load_errors_share_base_class_with_user_message():
    for exc_type in (UnsupportedFileError, EmptyFileError, CorruptFileError, NoDataError):
        assert issubclass(exc_type, DataLoadError)
        assert exc_type("сообщение").user_message == "сообщение"


# ------------------------------------------------------------------ CSV
@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_csv_delimiters(delimiter):
    text = delimiter.join(["city", "sales"]) + "\n" + delimiter.join(["Almaty", "10"]) + "\n"
    frame = read_table(text.encode("utf-8"), "d.csv")
    assert list(frame.columns) == ["city", "sales"] and len(frame) == 1


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1251", "utf-16"])
def test_csv_encodings(encoding):
    text = "город;выручка\nАлматы;10\nАстана;20\n"
    frame = read_table(text.encode(encoding), "d.csv")
    assert list(frame.columns) == ["город", "выручка"]
    assert frame["город"].tolist() == ["Алматы", "Астана"]


def test_csv_blank_rows_are_dropped():
    frame = read_table(b"a,b\n1,2\n,\n3,4\n\n", "d.csv")
    assert len(frame) == 2


# ------------------------------------------------------------------ XLSX
def test_list_sheets_and_default_sheet():
    assert list_sheets(GOOD_XLSX) == ["Sales", "Plan"]
    frame = read_table(GOOD_XLSX, "book.xlsx")
    assert list(frame.columns) == ["a", "b"]


def test_read_selected_sheet():
    frame = read_table(GOOD_XLSX, "book.xlsx", "Plan")
    assert list(frame.columns) == ["x"] and frame["x"].tolist() == ["p", "q", "r"]


def test_unknown_sheet_is_reported():
    with pytest.raises(NoDataError, match="Лист «Nope» не найден.*Sales, Plan"):
        read_table(GOOD_XLSX, "book.xlsx", "Nope")


def test_empty_sheet_has_no_data():
    book = make_xlsx({"Data": [["a"], [1]], "Empty": []})
    assert list_sheets(book) == ["Data", "Empty"]
    with pytest.raises(NoDataError):
        read_table(book, "b.xlsx", "Empty")


def test_xlsx_trailing_empty_rows_and_columns_are_removed():
    book = make_xlsx({"S": [["a", "b", None], [1, 2, None], [None, None, None], [3, 4, None]]})
    frame = read_table(book, "b.xlsx")
    assert list(frame.columns) == ["a", "b"] and len(frame) == 2


def test_missing_header_gets_placeholder_name():
    book = make_xlsx({"S": [["a", None], [1, 5], [2, 6]]})
    frame = read_table(book, "b.xlsx")
    assert list(frame.columns) == ["a", "Unnamed: 1"]
