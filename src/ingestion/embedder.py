"""Geração de embeddings e indexação no Pinecone.

Converte chunks de texto em vetores usando sentence-transformers
e faz upsert no Pinecone com suporte a namespaces.
"""

import hashlib
import logging
from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Gerencia embeddings e indexação no Pinecone.

    Attributes:
        model: Modelo de sentence-transformers para gerar embeddings.
        pc: Cliente Pinecone.
        index: Índice Pinecone ativo.
    """

    def __init__(self) -> None:
        """Inicializa o embedder com modelo e conexão Pinecone."""
        settings = get_settings()

        logger.info("Carregando modelo de embeddings: %s", settings.model.embedding_model)
        try:
            self.model = SentenceTransformer(settings.model.embedding_model)
        except Exception as e:
            raise EmbeddingError(f"Erro ao carregar modelo de embeddings: {e}") from e

        logger.info("Conectando ao Pinecone...")
        try:
            self.pc = Pinecone(api_key=settings.pinecone.api_key)
            self._ensure_index_exists(settings)
            self.index = self.pc.Index(settings.pinecone.index_name)
        except Exception as e:
            raise EmbeddingError(f"Erro ao conectar ao Pinecone: {e}") from e

        logger.info("DocumentEmbedder inicializado com sucesso.")

    def _ensure_index_exists(self, settings: Any) -> None:
        """Cria o índice Pinecone se não existir.

        Args:
            settings: Configurações do projeto.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if settings.pinecone.index_name not in existing_indexes:
            logger.info(
                "Criando índice '%s' (dimensão=%d, métrica=cosine)...",
                settings.pinecone.index_name,
                settings.model.embedding_dimension,
            )
            self.pc.create_index(
                name=settings.pinecone.index_name,
                dimension=settings.model.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone.environment,
                ),
            )
            logger.info("Índice criado com sucesso.")
        else:
            logger.info("Índice '%s' já existe.", settings.pinecone.index_name)

    @staticmethod
    def _generate_id(text: str, metadata: dict) -> str:
        """Gera um ID determinístico para o vetor.

        Args:
            text: Conteúdo do chunk.
            metadata: Metadados do documento.

        Returns:
            Hash MD5 do conteúdo + source.
        """
        source = metadata.get("source_file", "unknown")
        chunk_idx = metadata.get("chunk_index", 0)
        content = f"{source}:{chunk_idx}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate_embeddings(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Gera embeddings para uma lista de textos.

        Args:
            texts: Lista de strings para converter em vetores.
            batch_size: Tamanho do batch para processamento.

        Returns:
            Lista de vetores (cada um é uma lista de floats).

        Raises:
            EmbeddingError: Se ocorrer erro na geração.
        """
        logger.info("Gerando embeddings para %d textos (batch_size=%d)...", len(texts), batch_size)

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()

        except Exception as e:
            raise EmbeddingError(f"Erro ao gerar embeddings: {e}") from e

    def embed_and_store(
        self,
        chunks: list[Document],
        namespace: str = "default",
        batch_size: int = 32,
    ) -> int:
        """Gera embeddings e armazena chunks no Pinecone.

        Args:
            chunks: Lista de Documents chunked.
            namespace: Namespace no Pinecone (ex: 'produtos', 'fiscal').
            batch_size: Tamanho do batch para embedding e upsert.

        Returns:
            Número de vetores inseridos com sucesso.

        Raises:
            EmbeddingError: Se ocorrer erro na geração ou upsert.
        """
        if not chunks:
            logger.warning("Lista de chunks vazia — nada para indexar.")
            return 0

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.generate_embeddings(texts, batch_size)

        logger.info("Fazendo upsert de %d vetores no namespace '%s'...", len(embeddings), namespace)

        try:
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vector_id = self._generate_id(chunk.page_content, chunk.metadata)
                metadata = {
                    "text": chunk.page_content,
                    "source_file": chunk.metadata.get("source_file", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                }
                vectors.append((vector_id, embedding, metadata))

            # Upsert em batches
            total_upserted = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.info("  Upsert batch %d/%d (%d vetores)", i // batch_size + 1,
                            (len(vectors) + batch_size - 1) // batch_size, len(batch))

            logger.info("Upsert concluído: %d vetores no namespace '%s'.", total_upserted, namespace)
            return total_upserted

        except Exception as e:
            raise EmbeddingError(f"Erro no upsert ao Pinecone: {e}") from e

    def query(self, query_text: str, top_k: int = 5, namespace: str = "default") -> list[dict]:
        """Busca os chunks mais relevantes para uma query.

        Args:
            query_text: Texto da pergunta do usuário.
            top_k: Número de resultados a retornar.
            namespace: Namespace no Pinecone para buscar.

        Returns:
            Lista de dicts com 'text', 'score' e 'metadata'.
        """
        query_embedding = self.generate_embeddings([query_text])[0]

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )

        return [
            {
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "source_file": match.metadata.get("source_file", "unknown"),
                "page": match.metadata.get("page", 0),
            }
            for match in results.matches
        ]
