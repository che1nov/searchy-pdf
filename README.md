# PDF Search Service (No Web Frameworks)

A production-style RESTful HTTP service for searching textual PDF documents using TF-IDF cosine similarity.

## Features

- HTTP API built only with Python standard library (`http.server`)
- Endpoint: `GET /search?q=<text>`
- Case-insensitive multi-word search
- Top-10 relevant documents by TF-IDF cosine similarity
- Recursive PDF discovery in configured directories
- Incremental indexing: reprocesses only new/modified files
- Persistent index (`pickle`) for fast restarts
- Graceful handling of unreadable/corrupted PDFs
- Thread-safe search execution

## Installation

1. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3. Place PDFs into configured folders (default):

- `./data`
- `./more_docs`

You can edit `config.yml` to change directories, host, port, or index file path.

## Run Server

From the `search_service` directory:

```bash
python server.py
```

Server starts on `http://127.0.0.1:8000` by default.

## API Usage

### Search request

```bash
curl "http://127.0.0.1:8000/search?q=machine+learning"
```

### Example response

```json
{
  "query": "machine learning",
  "results": [
    {
      "file": "doc1.pdf",
      "path": "/absolute/path/to/data/doc1.pdf",
      "score": 0.823114
    }
  ]
}
```

### Validation errors

Missing query:

```bash
curl "http://127.0.0.1:8000/search"
```

Returns:

```json
{
  "error": "Missing query parameter 'q'"
}
```

## Future REST API Specification (`apispec` branch)

This section describes planned endpoints for the next iteration of the project.

### Base URL

`http://127.0.0.1:8000/api/v1`

### `GET /health`

Checks service availability.

Response `200 OK`:

```json
{
  "status": "ok"
}
```

### `POST /documents/index`

Starts or refreshes PDF indexing.

Request body:

```json
{
  "directories": ["./data", "./more_docs"],
  "force_rebuild": false
}
```

Response `202 Accepted`:

```json
{
  "task_id": "idx_20260301_0001",
  "status": "accepted"
}
```

### `GET /documents`

Returns indexed documents with pagination.

Query params:

- `limit` (optional, default `20`, max `100`)
- `offset` (optional, default `0`)

Response `200 OK`:

```json
{
  "total": 2,
  "items": [
    {
      "id": "doc_1",
      "file": "doc1.pdf",
      "path": "/abs/path/data/doc1.pdf",
      "tokens_count": 1834,
      "updated_at": "2026-03-01T00:00:00Z"
    }
  ]
}
```

### `GET /documents/{document_id}`

Returns metadata for one document.

Response `200 OK`:

```json
{
  "id": "doc_1",
  "file": "doc1.pdf",
  "path": "/abs/path/data/doc1.pdf",
  "tokens_count": 1834,
  "updated_at": "2026-03-01T00:00:00Z"
}
```

Response `404 Not Found`:

```json
{
  "error": "document_not_found"
}
```

### `DELETE /documents/{document_id}`

Removes a document from the index.

Response `204 No Content`.

### `GET /search`

Searches indexed documents by query string.

Query params:

- `q` (required, non-empty string)
- `limit` (optional, default `10`, max `50`)
- `min_score` (optional, float between `0` and `1`)

Response `200 OK`:

```json
{
  "query": "machine learning",
  "total": 1,
  "items": [
    {
      "document_id": "doc_1",
      "file": "doc1.pdf",
      "path": "/abs/path/data/doc1.pdf",
      "score": 0.823114
    }
  ]
}
```

### Common error response

All endpoints return structured errors:

```json
{
  "error": "validation_error",
  "message": "q is required",
  "details": {
    "field": "q"
  }
}
```

Planned status codes: `400`, `404`, `409`, `422`, `500`.

## Testing and Coverage

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run lint:

```bash
ruff check .
```

Run tests with branch coverage (required: 100% for logic modules):

```bash
pytest
```

Coverage target is configured in `pytest.ini` with `--cov-branch --cov-fail-under=100`.

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push and pull requests:

- PEP 8 lint checks (`ruff`)
- Tests (`pytest`)
- Branch coverage enforcement (100% for core logic modules)

## Notes

- Only documents with positive relevance are returned.
- Empty-text PDFs are skipped.
- If index file exists and files are unchanged, service loads index without rebuilding.
- Removed PDFs are excluded on next startup refresh.
