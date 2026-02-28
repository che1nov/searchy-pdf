"""Entry point for the PDF search HTTP service."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from config_loader import load_config
from indexer import Indexer
from search_engine import SearchEngine

LOGGER = logging.getLogger("search_service")


class SearchRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing GET /search endpoint."""

    engine: SearchEngine
    logger: logging.Logger

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path != "/search":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Not found", "message": "Use GET /search?q=<text>"},
            )
            return

        query_params = parse_qs(parsed.query)
        query = (query_params.get("q") or [""])[0].strip()

        if not query:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Missing query parameter 'q'"},
            )
            return

        try:
            results = self.engine.search(query, limit=10)
        except Exception as exc:
            self.logger.exception("Search failed for query: %s", query)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal server error", "details": str(exc)},
            )
            return

        response_payload = {
            "query": query,
            "results": [
                {"file": item.file, "path": item.path, "score": item.score}
                for item in results
            ],
        }
        self._send_json(HTTPStatus.OK, response_payload)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        self.logger.info("%s - %s", self.client_address[0], format % args)


def main() -> None:
    """Load configuration, initialize index, and start HTTP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.yml"
    config = load_config(config_path)

    indexer = Indexer(
        directories=config.directories,
        index_file=config.index_file,
        logger=LOGGER,
    )
    index_data = indexer.build_or_load_index()

    SearchRequestHandler.engine = SearchEngine(index_data)
    SearchRequestHandler.logger = LOGGER

    server_address = (config.host, config.port)
    httpd = ThreadingHTTPServer(server_address, SearchRequestHandler)

    LOGGER.info("Search service started on http://%s:%d", config.host, config.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown signal received")
    finally:
        httpd.server_close()
        LOGGER.info("Server stopped")


if __name__ == "__main__":
    main()
