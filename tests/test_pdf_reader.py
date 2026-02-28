from pathlib import Path
from types import SimpleNamespace

from pdf_reader import extract_pdf_text


class DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, tuple[object, ...]]] = []

    def warning(self, msg: str, *args: object) -> None:
        self.warnings.append((msg, args))


def test_extract_pdf_text_success(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [
                SimpleNamespace(extract_text=lambda: "Hello"),
                SimpleNamespace(extract_text=lambda: "World"),
            ]

    monkeypatch.setattr("pdf_reader.PdfReader", FakeReader)
    logger = DummyLogger()

    text = extract_pdf_text(Path("file.pdf"), logger)

    assert text == "Hello\nWorld"
    assert logger.warnings == []


def test_extract_pdf_text_empty_returns_none(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [SimpleNamespace(extract_text=lambda: "")]

    monkeypatch.setattr("pdf_reader.PdfReader", FakeReader)
    logger = DummyLogger()

    text = extract_pdf_text(Path("file.pdf"), logger)

    assert text is None


def test_extract_pdf_text_failure_returns_none_and_logs(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, _path: str) -> None:
            raise RuntimeError("bad pdf")

    monkeypatch.setattr("pdf_reader.PdfReader", FakeReader)
    logger = DummyLogger()

    text = extract_pdf_text(Path("broken.pdf"), logger)

    assert text is None
    assert len(logger.warnings) == 1
