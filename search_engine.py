"""Thread-safe TF-IDF search engine."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass

from indexer import IndexData

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
    """Single search result returned to API clients."""

    file: str
    path: str
    score: float


class SearchEngine:
    """Performs cosine-similarity search over prebuilt TF-IDF document vectors."""

    def __init__(self, index_data: IndexData) -> None:
        self._index_data = index_data
        self._lock = threading.RLock()

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search documents by query and return top results with positive relevance."""
        tokens = TOKEN_PATTERN.findall(query.lower())
        if not tokens:
            return []

        with self._lock:
            idf = self._index_data.idf
            doc_vectors = self._index_data.doc_vectors
            doc_norms = self._index_data.doc_norms
            documents = self._index_data.documents

            query_counts: dict[str, int] = {}
            for token in tokens:
                if token in idf:
                    query_counts[token] = query_counts.get(token, 0) + 1

            total_terms = sum(query_counts.values())
            if total_terms == 0:
                return []

            query_vector: dict[str, float] = {}
            for token, count in query_counts.items():
                tf = count / total_terms
                query_vector[token] = tf * idf[token]

            query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
            if query_norm == 0:
                return []

            matches: list[SearchResult] = []
            for path_key, doc_vector in doc_vectors.items():
                dot = 0.0
                for token, query_weight in query_vector.items():
                    doc_weight = doc_vector.get(token)
                    if doc_weight is not None:
                        dot += query_weight * doc_weight

                if dot <= 0:
                    continue

                doc_norm = doc_norms.get(path_key, 0.0)
                if doc_norm == 0:
                    continue

                score = dot / (query_norm * doc_norm)
                if score <= 0:
                    continue

                doc = documents[path_key]
                matches.append(
                    SearchResult(
                        file=doc.file,
                        path=doc.path,
                        score=round(score, 6),
                    )
                )

            matches.sort(key=lambda result: result.score, reverse=True)
            return matches[:limit]
