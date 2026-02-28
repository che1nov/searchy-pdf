import pickle
from pathlib import Path

from indexer import DocumentEntry, IndexData, Indexer, _compute_idf, _tokenize_to_counts


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def info(self, msg: str, *args: object) -> None:
        self.records.append(("info", msg % args if args else msg))

    def warning(self, msg: str, *args: object) -> None:
        self.records.append(("warning", msg % args if args else msg))

    def debug(self, msg: str, *args: object) -> None:
        self.records.append(("debug", msg % args if args else msg))


def _create_pdf_file(path: Path, content: str = "pdf") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tokenize_to_counts() -> None:
    counts = _tokenize_to_counts("Hello hello, world!")
    assert counts == {"Hello": 1, "hello": 1, "world": 1}


def test_compute_idf_empty_documents() -> None:
    assert _compute_idf([]) == {}


def test_compute_idf_non_empty_documents() -> None:
    docs = [
        DocumentEntry("a.pdf", "/a.pdf", 1.0, 1, {"a": 1, "b": 1}, 2),
        DocumentEntry("b.pdf", "/b.pdf", 1.0, 1, {"b": 1}, 1),
    ]
    idf = _compute_idf(docs)
    assert idf["a"] > idf["b"]


def test_build_index_from_scratch(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "data"
    file_path = docs_dir / "doc1.pdf"
    _create_pdf_file(file_path)

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: "alpha beta alpha")
    logger = DummyLogger()
    indexer = Indexer([docs_dir], tmp_path / "index.pkl", logger)

    result = indexer.build_or_load_index()

    assert str(file_path.resolve()) in result.documents
    assert (tmp_path / "index.pkl").exists()


def test_build_or_load_reuses_unchanged_files(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "data"
    file_path = docs_dir / "doc1.pdf"
    _create_pdf_file(file_path)

    logger = DummyLogger()
    index_file = tmp_path / "index.pkl"

    old_doc = DocumentEntry(
        file=file_path.name,
        path=str(file_path.resolve()),
        mtime=file_path.stat().st_mtime,
        size=file_path.stat().st_size,
        token_counts={"alpha": 1},
        total_terms=1,
    )
    old_index = IndexData(
        documents={old_doc.path: old_doc},
        idf={"alpha": 1.0},
        doc_vectors={old_doc.path: {"alpha": 1.0}},
        doc_norms={old_doc.path: 1.0},
        built_at=1.0,
    )
    with index_file.open("wb") as fh:
        pickle.dump(old_index, fh)

    def should_not_be_called(_path, _logger):
        raise AssertionError("extract should not be called for unchanged file")

    monkeypatch.setattr("indexer.extract_pdf_text", should_not_be_called)

    indexer = Indexer([docs_dir], index_file, logger)
    result = indexer.build_or_load_index()

    assert result == old_index


def test_build_or_load_updates_modified_file(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "data"
    file_path = docs_dir / "doc1.pdf"
    _create_pdf_file(file_path, "old")

    logger = DummyLogger()
    index_file = tmp_path / "index.pkl"

    old_doc = DocumentEntry(
        file=file_path.name,
        path=str(file_path.resolve()),
        mtime=1.0,
        size=1,
        token_counts={"old": 1},
        total_terms=1,
    )
    old_index = IndexData(
        documents={old_doc.path: old_doc},
        idf={"old": 1.0},
        doc_vectors={old_doc.path: {"old": 1.0}},
        doc_norms={old_doc.path: 1.0},
        built_at=1.0,
    )
    with index_file.open("wb") as fh:
        pickle.dump(old_index, fh)

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: "new token")

    indexer = Indexer([docs_dir], index_file, logger)
    result = indexer.build_or_load_index()

    assert result is not old_index
    new_doc = result.documents[str(file_path.resolve())]
    assert "new" in new_doc.token_counts


def test_build_or_load_skips_changed_file_when_extract_returns_none(
    monkeypatch, tmp_path: Path
) -> None:
    docs_dir = tmp_path / "data"
    file_path = docs_dir / "doc1.pdf"
    _create_pdf_file(file_path, "old")

    logger = DummyLogger()
    index_file = tmp_path / "index.pkl"

    old_doc = DocumentEntry(
        file=file_path.name,
        path=str(file_path.resolve()),
        mtime=1.0,
        size=1,
        token_counts={"old": 1},
        total_terms=1,
    )
    old_index = IndexData(
        documents={old_doc.path: old_doc},
        idf={"old": 1.0},
        doc_vectors={old_doc.path: {"old": 1.0}},
        doc_norms={old_doc.path: 1.0},
        built_at=1.0,
    )
    with index_file.open("wb") as fh:
        pickle.dump(old_index, fh)

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: None)

    indexer = Indexer([docs_dir], index_file, logger)
    result = indexer.build_or_load_index()

    assert result.documents == {}


def test_build_or_load_handles_removed_file(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "data"
    docs_dir.mkdir(parents=True, exist_ok=True)
    missing_path = docs_dir / "removed.pdf"

    logger = DummyLogger()
    index_file = tmp_path / "index.pkl"

    old_doc = DocumentEntry(
        file="removed.pdf",
        path=str(missing_path.resolve()),
        mtime=1.0,
        size=1,
        token_counts={"old": 1},
        total_terms=1,
    )
    old_index = IndexData(
        documents={old_doc.path: old_doc},
        idf={"old": 1.0},
        doc_vectors={old_doc.path: {"old": 1.0}},
        doc_norms={old_doc.path: 1.0},
        built_at=1.0,
    )
    with index_file.open("wb") as fh:
        pickle.dump(old_index, fh)

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: "")

    indexer = Indexer([docs_dir], index_file, logger)
    result = indexer.build_or_load_index()

    assert result.documents == {}


def test_extract_document_skips_unreadable_and_empty(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "doc.pdf"
    _create_pdf_file(file_path)
    logger = DummyLogger()
    indexer = Indexer([tmp_path], tmp_path / "index.pkl", logger)

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: None)
    assert indexer._extract_document(file_path, 1.0, 10) is None

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: "")
    assert indexer._extract_document(file_path, 1.0, 10) is None


def test_discover_pdf_files_warns_missing_directory(tmp_path: Path) -> None:
    logger = DummyLogger()
    missing_dir = tmp_path / "missing"
    indexer = Indexer([missing_dir], tmp_path / "index.pkl", logger)

    files = indexer._discover_pdf_files()

    assert files == []
    assert any(level == "warning" for level, _ in logger.records)


def test_load_index_with_corrupted_file_returns_none(tmp_path: Path) -> None:
    logger = DummyLogger()
    index_file = tmp_path / "index.pkl"
    index_file.write_bytes(b"not-a-pickle")

    indexer = Indexer([tmp_path], index_file, logger)

    assert indexer._load_index() is None


def test_load_index_with_wrong_type_returns_none(tmp_path: Path) -> None:
    logger = DummyLogger()
    index_file = tmp_path / "index.pkl"
    with index_file.open("wb") as fh:
        pickle.dump({"bad": "format"}, fh)

    indexer = Indexer([tmp_path], index_file, logger)

    assert indexer._load_index() is None


def test_save_index_writes_atomically(tmp_path: Path) -> None:
    logger = DummyLogger()
    index_file = tmp_path / "cache" / "index.pkl"
    indexer = Indexer([tmp_path], index_file, logger)

    index = IndexData(documents={}, idf={}, doc_vectors={}, doc_norms={}, built_at=1.0)
    indexer._save_index(index)

    assert index_file.exists()
    assert not index_file.with_suffix(index_file.suffix + ".tmp").exists()


def test_build_index_from_scratch_skips_unreadable_file(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "data"
    file_path = docs_dir / "bad.pdf"
    _create_pdf_file(file_path)

    monkeypatch.setattr("indexer.extract_pdf_text", lambda _p, _l: None)
    logger = DummyLogger()
    indexer = Indexer([docs_dir], tmp_path / "index.pkl", logger)

    result = indexer.build_or_load_index()

    assert result.documents == {}


def test_discover_pdf_files_ignores_non_file_entries(tmp_path: Path) -> None:
    logger = DummyLogger()
    docs_dir = tmp_path / "data"
    fake_pdf_dir = docs_dir / "nested.pdf"
    fake_pdf_dir.mkdir(parents=True, exist_ok=True)
    real_pdf = docs_dir / "real.pdf"
    _create_pdf_file(real_pdf)

    indexer = Indexer([docs_dir], tmp_path / "index.pkl", logger)
    files = indexer._discover_pdf_files()

    assert files == [real_pdf.resolve()]


def test_rebuild_model_skips_tokens_absent_in_idf(monkeypatch, tmp_path: Path) -> None:
    logger = DummyLogger()
    indexer = Indexer([tmp_path], tmp_path / "index.pkl", logger)
    doc = DocumentEntry(
        file="doc.pdf",
        path=str((tmp_path / "doc.pdf").resolve()),
        mtime=1.0,
        size=1,
        token_counts={"alpha": 1},
        total_terms=1,
    )
    monkeypatch.setattr("indexer._compute_idf", lambda _documents: {})

    result = indexer._rebuild_model({doc.path: doc})

    assert result.idf == {}
    assert result.doc_vectors == {}
    assert result.documents == {}
