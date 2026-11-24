import pytest
from fastapi.testclient import TestClient


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "service" in data
    assert "version" in data


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "ollama_reachable" in body
    assert "vector_store_ready" in body


def test_chat_validation(client: TestClient):
    # empty message should fail
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_schema(client: TestClient):
    # we don't require Ollama to be up for schema validation,
    # but the call itself may return a graceful error message
    r = client.post(
        "/chat",
        json={
            "message": "What is 2 + 2?",
            "use_tools": False,
            "use_rag": False,
        },
    )
    # either 200 (if model responds) or 500 with detail – both ok for CI without Ollama
    assert r.status_code in (200, 500)
    if r.status_code == 200:
        data = r.json()
        assert "response" in data
        assert "session_id" in data
        assert "latency_ms" in data
