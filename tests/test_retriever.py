"""Testes unitários para o PineconeRetriever.

Usa mocks para evitar chamadas reais ao Pinecone.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.retriever import PineconeRetriever


class MockMatch:
    """Mock de um resultado do Pinecone."""

    def __init__(self, score: float, metadata: dict):
        self.score = score
        self.metadata = metadata


class MockQueryResponse:
    """Mock da resposta do Pinecone.query()."""

    def __init__(self, matches: list):
        self.matches = matches


@pytest.fixture
def mock_retriever():
    """Cria retriever com Pinecone e modelo mockados."""
    with patch("src.rag.retriever.SentenceTransformer") as mock_st, \
         patch("src.rag.retriever.Pinecone") as mock_pc:

        # Mock do modelo de embeddings
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384  # Vetor fake
        mock_st.return_value = mock_model

        # Mock do Pinecone
        mock_index = MagicMock()
        mock_pc_instance = MagicMock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pc.return_value = mock_pc_instance

        retriever = PineconeRetriever(top_k=3)
        retriever.index = mock_index
        retriever.model = mock_model

        yield retriever, mock_index, mock_model


class TestPineconeRetriever:
    """Testes do retriever vetorial."""

    def test_retrieve_returns_results(self, mock_retriever):
        retriever, mock_index, _ = mock_retriever

        # Configura resposta mock do Pinecone
        mock_index.query.return_value = MockQueryResponse([
            MockMatch(0.95, {"text": "Tela X tem 50% de sombreamento", "source_file": "catalogo.pdf", "page": 3}),
            MockMatch(0.82, {"text": "Disponível em 2m, 3m e 4m", "source_file": "catalogo.pdf", "page": 4}),
            MockMatch(0.20, {"text": "Irrelevante", "source_file": "outro.pdf", "page": 1}),
        ])

        results = retriever.retrieve("sombreamento tela X", score_threshold=0.3)

        assert len(results) == 2  # Score 0.20 filtrado
        assert results[0]["score"] == 0.95
        assert "catalogo.pdf" in results[0]["source_file"]

    def test_retrieve_empty_results(self, mock_retriever):
        retriever, mock_index, _ = mock_retriever

        mock_index.query.return_value = MockQueryResponse([])
        results = retriever.retrieve("pergunta sem resultado")

        assert len(results) == 0

    def test_format_context_with_documents(self, mock_retriever):
        retriever, _, _ = mock_retriever

        documents = [
            {"text": "Texto do doc 1", "score": 0.95, "source_file": "a.pdf", "page": 1},
            {"text": "Texto do doc 2", "score": 0.80, "source_file": "b.pdf", "page": 5},
        ]

        context = retriever.format_context(documents)

        assert "Texto do doc 1" in context
        assert "Texto do doc 2" in context
        assert "a.pdf" in context
        assert "Fonte 1" in context
        assert "Fonte 2" in context

    def test_format_context_empty(self, mock_retriever):
        retriever, _, _ = mock_retriever

        context = retriever.format_context([])
        assert "Nenhum documento" in context

    def test_score_threshold_filtering(self, mock_retriever):
        retriever, mock_index, _ = mock_retriever

        mock_index.query.return_value = MockQueryResponse([
            MockMatch(0.25, {"text": "Baixa relevância", "source_file": "x.pdf", "page": 1}),
        ])

        # Com threshold padrão (0.3), deve filtrar
        results = retriever.retrieve("query", score_threshold=0.3)
        assert len(results) == 0

        # Com threshold baixo, deve incluir
        results = retriever.retrieve("query", score_threshold=0.1)
        assert len(results) == 1
