"""Демонстрация работы поисковой логики без запуска HTTP-сервера."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config_loader import load_config
from indexer import Indexer
from search_engine import SearchEngine

LOGGER = logging.getLogger("search_service.demo")


def run_demo() -> None:
    """Запускает демонстрацию индексации и поиска."""
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir / "config.yml")

    indexer = Indexer(
        directories=config.directories,
        index_file=config.index_file,
        logger=LOGGER,
    )
    index_data = indexer.build_or_load_index()

    engine = SearchEngine(index_data)
    demo_queries = [
        "python",
        "machine learning",
        "distributed systems",
    ]

    for query in demo_queries:
        results = engine.search(query)
        payload = {
            "query": query,
            "results": [
                {"file": item.file, "path": item.path, "score": item.score}
                for item in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    """Точка входа демонстрационного режима."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_demo()


if __name__ == "__main__":
    main()
