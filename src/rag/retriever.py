"""Retriever vetorial usando Pinecone.

Busca os chunks mais relevantes para uma pergunta
do usuário e retorna documentos com scores.
"""

import logging

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class PineconeRetriever:
    """Busca semântica no Pinecone.

    Converte a query em embedding e busca os top-k chunks
    mais similares no banco vetorial.

    Attributes:
        model: Modelo de embeddings.
        index: Índice Pinecone.
        top_k: Número padrão de resultados.
    """

    def __init__(self, top_k: int = 5) -> None:
        """Inicializa o retriever.

        Args:
            top_k: Número padrão de documentos a retornar.
        """
        settings = get_settings()
        self.top_k = top_k

        logger.info("Inicializando PineconeRetriever...")

        try:
            self.model = SentenceTransformer(settings.model.embedding_model)
            self.pc = Pinecone(api_key=settings.pinecone.api_key)
            self.index = self.pc.Index(settings.pinecone.index_name)
        except Exception as e:
            raise EmbeddingError(f"Erro ao inicializar retriever: {e}") from e

        logger.info("PineconeRetriever pronto (top_k=%d).", self.top_k)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str = "default",
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """Busca chunks relevantes para a query.

        Args:
            query: Pergunta do usuário.
            top_k: Número de resultados (usa padrão se None).
            namespace: Namespace do Pinecone para buscar.
            score_threshold: Score mínimo para incluir resultado.

        Returns:
            Lista de dicts com 'text', 'score', 'source_file', 'page'.
        """
        k = top_k or self.top_k

        logger.info("Buscando top-%d para: '%s' (namespace='%s')", k, query[:80], namespace)

        try:
            # Gera embedding da query
            query_embedding = self.model.encode(
                query,
                normalize_embeddings=True,
            ).tolist()

            # Busca no Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=namespace,
                include_metadata=True,
            )

            # Filtra por score mínimo e formata
            documents = []
            for match in results.matches:
                if match.score >= score_threshold:
                    documents.append({
                        "text": match.metadata.get("text", ""),
                        "score": round(match.score, 4),
                        "source_file": match.metadata.get("source_file", "unknown"),
                        "page": match.metadata.get("page", 0),
                    })

            logger.info("Encontrados %d resultados (score >= %.2f).", len(documents), score_threshold)
            return documents

        except Exception as e:
            raise EmbeddingError(f"Erro na busca vetorial: {e}") from e

    def format_context(self, documents: list[dict]) -> str:
        """Formata os documentos recuperados como contexto para o LLM.

        Args:
            documents: Lista de documentos retornados pelo retrieve().

        Returns:
            String formatada com os chunks e suas fontes.
        """
        if not documents:
            return "Nenhum documento relevante encontrado na base de conhecimento."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc["source_file"]
            page = doc.get("page", "N/A")
            score = doc["score"]
            text = doc["text"]

            context_parts.append(
                f"[Fonte {i}: {source} (p.{page}, relevância: {score})]:\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)
