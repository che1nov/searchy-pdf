from indexer import DocumentEntry, IndexData
from search_engine import SearchEngine


def _build_engine() -> SearchEngine:
    doc1 = DocumentEntry(
        file="doc1.pdf",
        path="/tmp/doc1.pdf",
        mtime=1.0,
        size=10,
        token_counts={"alpha": 2, "beta": 1},
        total_terms=3,
    )
    doc2 = DocumentEntry(
        file="doc2.pdf",
        path="/tmp/doc2.pdf",
        mtime=1.0,
        size=20,
        token_counts={"beta": 2},
        total_terms=2,
    )

    index = IndexData(
        documents={doc1.path: doc1, doc2.path: doc2},
        idf={"alpha": 1.5, "beta": 1.2},
        doc_vectors={
            doc1.path: {"alpha": 0.7, "beta": 0.2},
            doc2.path: {"beta": 0.9},
        },
        doc_norms={doc1.path: 0.728011, doc2.path: 0.9},
        built_at=1.0,
    )
    return SearchEngine(index)


def test_search_returns_ranked_results_limited_to_10() -> None:
    engine = _build_engine()

    results = engine.search("alpha beta")

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].file == "doc1.pdf"


def test_search_ignores_empty_query() -> None:
    engine = _build_engine()

    assert engine.search("   ") == []


def test_search_ignores_unknown_terms() -> None:
    engine = _build_engine()

    assert engine.search("gamma delta") == []


def test_search_skips_zero_norm_documents() -> None:
    doc = DocumentEntry(
        file="doc.pdf",
        path="/tmp/doc.pdf",
        mtime=1.0,
        size=10,
        token_counts={"alpha": 1},
        total_terms=1,
    )
    index = IndexData(
        documents={doc.path: doc},
        idf={"alpha": 1.2},
        doc_vectors={doc.path: {"alpha": 0.5}},
        doc_norms={doc.path: 0.0},
        built_at=1.0,
    )
    engine = SearchEngine(index)

    assert engine.search("alpha") == []


def test_search_skips_non_positive_dot_product() -> None:
    doc = DocumentEntry(
        file="doc.pdf",
        path="/tmp/doc.pdf",
        mtime=1.0,
        size=10,
        token_counts={"alpha": 1},
        total_terms=1,
    )
    index = IndexData(
        documents={doc.path: doc},
        idf={"alpha": 1.0},
        doc_vectors={doc.path: {"alpha": -0.5}},
        doc_norms={doc.path: 0.5},
        built_at=1.0,
    )
    engine = SearchEngine(index)

    assert engine.search("alpha") == []
