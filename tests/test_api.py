"""Testes da API FastAPI.

Testa os endpoints usando TestClient (sem dependências externas).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Cria TestClient com RAG chain mockado."""
    from src.api.main import app
    from src.api.routes import set_rag_chain

    # Mock do RAG chain
    mock_chain = MagicMock()
    mock_chain.ask.return_value = {
        "answer": "A tela modelo X tem 50% de sombreamento.",
        "sources": [
            {"source_file": "catalogo.pdf", "page": 3, "score": 0.95},
        ],
        "pii_detected": False,
        "processing_time_ms": 150.5,
        "masked_entities": [],
    }
    mock_chain.model_manager.is_loaded = True

    set_rag_chain(mock_chain)

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc, mock_chain


class TestRootEndpoint:
    """Testes do endpoint raiz."""

    def test_root_returns_info(self, client):
        tc, _ = client
        response = tc.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "AI Sales Copilot" in data["message"]
        assert "docs" in data


class TestChatEndpoint:
    """Testes do endpoint de chat."""

    def test_chat_success(self, client):
        tc, mock_chain = client
        response = tc.post("/api/v1/chat", json={
            "message": "Qual o sombreamento da tela X?",
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "A tela modelo X tem 50% de sombreamento."

    def test_chat_with_params(self, client):
        tc, mock_chain = client
        response = tc.post("/api/v1/chat", json={
            "message": "Teste",
            "temperature": 0.3,
            "max_tokens": 256,
            "top_k": 3,
            "namespace": "produtos",
        })
        assert response.status_code == 200
        mock_chain.ask.assert_called_once_with(
            question="Teste",
            namespace="produtos",
            temperature=0.3,
            max_tokens=256,
            top_k=3,
        )

    def test_chat_empty_message(self, client):
        tc, _ = client
        response = tc.post("/api/v1/chat", json={
            "message": "",
        })
        assert response.status_code == 422  # Validation error

    def test_chat_temperature_out_of_range(self, client):
        tc, _ = client
        response = tc.post("/api/v1/chat", json={
            "message": "Teste",
            "temperature": 2.0,
        })
        assert response.status_code == 422


class TestHealthEndpoint:
    """Testes do health check."""

    def test_health_check(self, client):
        tc, _ = client
        response = tc.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestDocumentsEndpoint:
    """Testes dos endpoints de documentos."""

    def test_list_documents(self, client):
        tc, _ = client
        response = tc.get("/api/v1/documents/list")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
