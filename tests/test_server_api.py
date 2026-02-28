"""Integration tests for the HTTP API contract using requests."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

import pytest

requests = pytest.importorskip("requests")

from indexer import DocumentEntry, IndexData
from search_engine import SearchEngine
from server import SearchRequestHandler


class DummyLogger:
    def info(self, _msg: str, *_args: object) -> None:
        return None

    def exception(self, _msg: str, *_args: object) -> None:
        return None


@pytest.fixture()
def api_base_url() -> str:
    doc = DocumentEntry(
        file="doc1.pdf",
        path="/tmp/doc1.pdf",
        mtime=1.0,
        size=1,
        token_counts={"machine": 1, "learning": 1},
        total_terms=2,
    )
    index_data = IndexData(
        documents={doc.path: doc},
        idf={"machine": 1.0, "learning": 1.0},
        doc_vectors={doc.path: {"machine": 0.5, "learning": 0.5}},
        doc_norms={doc.path: 0.707106},
        built_at=1.0,
    )

    SearchRequestHandler.engine = SearchEngine(index_data)
    SearchRequestHandler.logger = DummyLogger()

    server = ThreadingHTTPServer(("127.0.0.1", 0), SearchRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        yield f"http://{host}:{port}/api/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_endpoint_returns_ok_status(api_base_url: str) -> None:
    """Health endpoint should return service status."""
    response = requests.get(f"{api_base_url}/health", timeout=3)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_endpoint_returns_results_payload(api_base_url: str) -> None:
    """Search endpoint should return ranked document matches."""
    response = requests.get(
        f"{api_base_url}/search",
        params={"q": "machine learning", "limit": 5},
        timeout=3,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "machine learning"
    assert payload["total"] == 1
    assert isinstance(payload["items"], list)
    assert payload["items"][0]["file"] == "doc1.pdf"
