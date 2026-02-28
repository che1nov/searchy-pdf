"""Indexing logic for PDF documents with incremental updates."""

from __future__ import annotations

import logging
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pdf_reader import extract_pdf_text

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class DocumentEntry:
    """Stored representation of a single indexed PDF document."""

    file: str
    path: str
    mtime: float
    size: int
    token_counts: dict[str, int]
    total_terms: int


@dataclass
class IndexData:
    """Complete in-memory search index."""

    documents: dict[str, DocumentEntry]
    idf: dict[str, float]
    doc_vectors: dict[str, dict[str, float]]
    doc_norms: dict[str, float]
    built_at: float


class Indexer:
    """Builds, loads, and incrementally refreshes TF-IDF index for PDFs."""

    def __init__(self, directories: list[Path], index_file: Path, logger: logging.Logger) -> None:
        self._directories = directories
        self._index_file = index_file
        self._logger = logger

    def build_or_load_index(self) -> IndexData:
        """Load persisted index and update only new/modified files if needed."""
        existing = self._load_index()
        current_files = self._discover_pdf_files()

        if existing is None:
            self._logger.info("No existing index found, building index from scratch")
            return self._build_index_from_files(current_files)

        self._logger.info("Loaded existing index with %d documents", len(existing.documents))

        updated_documents: dict[str, DocumentEntry] = {}
        changed_or_new = 0
        reused = 0

        for file_path in current_files:
            stat = file_path.stat()
            path_key = str(file_path)
            old = existing.documents.get(path_key)

            if old and old.mtime == stat.st_mtime and old.size == stat.st_size:
                updated_documents[path_key] = old
                reused += 1
                continue

            extracted = self._extract_document(file_path, stat.st_mtime, stat.st_size)
            if extracted is not None:
                updated_documents[path_key] = extracted
            changed_or_new += 1

        removed = len(set(existing.documents) - {str(path) for path in current_files})

        if changed_or_new == 0 and removed == 0:
            self._logger.info("Index is up to date. Reused %d documents", reused)
            return existing

        self._logger.info(
            "Refreshing index: updated/new=%d, removed=%d, reused=%d",
            changed_or_new,
            removed,
            reused,
        )

        refreshed = self._rebuild_model(updated_documents)
        self._save_index(refreshed)
        return refreshed

    def _build_index_from_files(self, files: list[Path]) -> IndexData:
        documents: dict[str, DocumentEntry] = {}
        for file_path in files:
            stat = file_path.stat()
            entry = self._extract_document(file_path, stat.st_mtime, stat.st_size)
            if entry is not None:
                documents[entry.path] = entry

        index = self._rebuild_model(documents)
        self._save_index(index)
        return index

    def _extract_document(self, file_path: Path, mtime: float, size: int) -> DocumentEntry | None:
        raw_text = extract_pdf_text(file_path, self._logger)
        if raw_text is None:
            return None

        token_counts = _tokenize_to_counts(raw_text.lower())
        total_terms = sum(token_counts.values())
        if total_terms == 0:
            self._logger.info("Skipping empty PDF: %s", file_path)
            return None

        self._logger.debug("Indexed file %s (%d terms)", file_path, total_terms)
        return DocumentEntry(
            file=file_path.name,
            path=str(file_path),
            mtime=mtime,
            size=size,
            token_counts=token_counts,
            total_terms=total_terms,
        )

    def _discover_pdf_files(self) -> list[Path]:
        files: list[Path] = []
        for directory in self._directories:
            if not directory.exists() or not directory.is_dir():
                self._logger.warning("Directory does not exist or is not accessible: %s", directory)
                continue

            for file_path in directory.rglob("*.pdf"):
                if file_path.is_file():
                    files.append(file_path.resolve())

        files.sort()
        self._logger.info("Discovered %d PDF files", len(files))
        return files

    def _rebuild_model(self, documents: dict[str, DocumentEntry]) -> IndexData:
        idf = _compute_idf(documents.values())
        doc_vectors: dict[str, dict[str, float]] = {}
        doc_norms: dict[str, float] = {}

        for path_key, document in documents.items():
            vector: dict[str, float] = {}
            for token, count in document.token_counts.items():
                token_idf = idf.get(token)
                if token_idf is None:
                    continue
                tf = count / document.total_terms
                vector[token] = tf * token_idf

            norm = math.sqrt(sum(value * value for value in vector.values()))
            if norm > 0:
                doc_vectors[path_key] = vector
                doc_norms[path_key] = norm

        indexed_documents = {
            path_key: doc
            for path_key, doc in documents.items()
            if path_key in doc_vectors
        }

        self._logger.info("Prepared TF-IDF model for %d documents", len(indexed_documents))
        return IndexData(
            documents=indexed_documents,
            idf=idf,
            doc_vectors=doc_vectors,
            doc_norms=doc_norms,
            built_at=time.time(),
        )

    def _load_index(self) -> IndexData | None:
        if not self._index_file.exists():
            return None

        try:
            with self._index_file.open("rb") as file:
                loaded = pickle.load(file)
        except Exception as exc:
            self._logger.warning("Failed to load existing index (%s). Rebuilding...", exc)
            return None

        if not isinstance(loaded, IndexData):
            self._logger.warning("Unsupported index format. Rebuilding...")
            return None

        return loaded

    def _save_index(self, index: IndexData) -> None:
        self._index_file.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._index_file.with_suffix(self._index_file.suffix + ".tmp")
        with temp_path.open("wb") as file:
            pickle.dump(index, file)
        temp_path.replace(self._index_file)
        self._logger.info("Index saved to %s", self._index_file)


def _tokenize_to_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in TOKEN_PATTERN.findall(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _compute_idf(documents: Iterable[DocumentEntry]) -> dict[str, float]:
    documents_list = list(documents)
    total_docs = len(documents_list)
    if total_docs == 0:
        return {}

    doc_frequency: dict[str, int] = {}
    for doc in documents_list:
        for token in doc.token_counts:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    idf: dict[str, float] = {}
    for token, df in doc_frequency.items():
        idf[token] = math.log((1 + total_docs) / (1 + df)) + 1.0
    return idf
