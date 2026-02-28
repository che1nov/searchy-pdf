"""Integration tests for the future REST API server."""

from __future__ import annotations

import os

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")


def test_health_endpoint_returns_ok_status() -> None:
    """Health endpoint should return service status."""
    response = requests.get(f"{BASE_URL}/health", timeout=3)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_endpoint_returns_results_payload() -> None:
    """Search endpoint should return ranked document matches."""
    response = requests.get(
        f"{BASE_URL}/search",
        params={"q": "machine learning", "limit": 5},
        timeout=3,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "machine learning"
    assert isinstance(payload["items"], list)
