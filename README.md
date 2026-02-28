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
