"""Text splitting / chunking de documentos.

Usa RecursiveCharacterTextSplitter do LangChain para dividir
documentos em chunks menores preservando metadados.
"""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exceptions import ChunkingError

logger = logging.getLogger(__name__)

# Separadores ordenados por prioridade (parágrafos > linhas > frases > palavras)
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[Document]:
    """Divide documentos em chunks menores para indexação vetorial.

    Args:
        documents: Lista de Documents do LangChain.
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        chunk_overlap: Sobreposição entre chunks consecutivos.
        separators: Lista de separadores (padrão: parágrafos → palavras).

    Returns:
        Lista de Documents chunked, com metadados preservados e chunk_index adicionado.

    Raises:
        ChunkingError: Se ocorrer erro durante o splitting.
    """
    if not documents:
        raise ChunkingError("Lista de documentos vazia — nada para chunkar.")

    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size deve ser positivo, recebido: {chunk_size}")

    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) deve ser menor que chunk_size ({chunk_size})."
        )

    logger.info(
        "Chunking %d documentos (chunk_size=%d, overlap=%d)...",
        len(documents),
        chunk_size,
        chunk_overlap,
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = splitter.split_documents(documents)

        # Adiciona índice do chunk nos metadados
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(
            "Chunking concluído: %d documentos → %d chunks.",
            len(documents),
            len(chunks),
        )

        return chunks

    except Exception as e:
        raise ChunkingError(f"Erro durante o chunking: {e}") from e


def get_chunk_stats(chunks: list[Document]) -> dict:
    """Retorna estatísticas dos chunks gerados.

    Args:
        chunks: Lista de Documents chunked.

    Returns:
        Dicionário com total, média, min e max de caracteres por chunk.
    """
    if not chunks:
        return {"total": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(chunk.page_content) for chunk in chunks]

    return {
        "total": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
