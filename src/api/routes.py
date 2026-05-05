"""Rotas da API FastAPI.

Define os endpoints do AI Sales Copilot:
- POST /api/v1/chat — Pergunta e resposta
- POST /api/v1/chat/stream — Streaming SSE
- GET /api/v1/health — Health check
- POST /api/v1/documents/ingest — Ingestão de documentos
- GET /api/v1/documents/list — Lista documentos
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src import __version__
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from src.config import get_settings
from src.exceptions import RAGChainError, DocumentLoadError, EmbeddingError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["AI Sales Copilot"])

# Referências globais (inicializadas no lifespan do main.py)
_rag_chain = None


def set_rag_chain(chain) -> None:
    """Define a instância do RAG chain (chamado no startup)."""
    global _rag_chain
    _rag_chain = chain


def get_rag_chain():
    """Retorna a instância do RAG chain."""
    if _rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Aguarde o startup.",
        )
    return _rag_chain


# ============================================================
# Chat
# ============================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Enviar pergunta ao assistente",
    description="Processa uma pergunta usando RAG (retrieval + generation) com PII masking.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Endpoint principal de chat.

    Recebe uma pergunta, busca contexto no Pinecone,
    mascara PII e gera resposta com o LLM.
    """
    chain = get_rag_chain()

    try:
        result = chain.ask(
            question=request.message,
            namespace=request.namespace,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
        )

        return ChatResponse(**result)

    except RAGChainError as e:
        logger.error("Erro no RAG chain: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/stream",
    summary="Chat com streaming (SSE)",
    description="Streaming de resposta via Server-Sent Events.",
)
async def chat_stream(request: ChatRequest):
    """Endpoint de chat com streaming SSE.

    Retorna a resposta token por token usando Server-Sent Events.
    """
    chain = get_rag_chain()

    async def generate():
        try:
            result = chain.ask(
                question=request.message,
                namespace=request.namespace,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_k=request.top_k,
            )

            # Simula streaming (o modelo gera tudo de uma vez)
            answer = result["answer"]
            words = answer.split(" ")

            for i, word in enumerate(words):
                separator = " " if i > 0 else ""
                yield f"data: {separator}{word}\n\n"

            # Envia metadados finais
            import json
            metadata = {
                "sources": result["sources"],
                "pii_detected": result["pii_detected"],
                "processing_time_ms": result["processing_time_ms"],
            }
            yield f"data: [METADATA]{json.dumps(metadata)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================
# Health
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verifica o status do serviço e seus componentes.",
)
async def health_check() -> HealthResponse:
    """Retorna o status de saúde do serviço."""
    settings = get_settings()

    model_loaded = False
    pinecone_connected = False

    try:
        if _rag_chain is not None:
            model_loaded = _rag_chain.model_manager.is_loaded
            pinecone_connected = True  # Se o chain inicializou, Pinecone está ok
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        pinecone_connected=pinecone_connected,
        pii_masking_enabled=settings.security.enable_pii_masking,
        version=__version__,
    )


# ============================================================
# Documents
# ============================================================

@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ingerir documentos",
    description="Carrega, chunka e indexa documentos no Pinecone.",
)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Endpoint para ingestão de novos documentos."""
    settings = get_settings()

    directory = Path(request.directory) if request.directory else settings.raw_data_dir

    try:
        from src.ingestion.loader import load_documents
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.embedder import DocumentEmbedder

        # Carrega
        documents = load_documents(directory)

        # Chunka
        chunks = chunk_documents(
            documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # Indexa
        embedder = DocumentEmbedder()
        vectors_count = embedder.embed_and_store(chunks, namespace=request.namespace)

        return IngestResponse(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            vectors_indexed=vectors_count,
            namespace=request.namespace,
        )

    except (DocumentLoadError, EmbeddingError) as e:
        logger.error("Erro na ingestão: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/documents/list",
    summary="Listar documentos",
    description="Lista os documentos disponíveis no diretório de dados.",
)
async def list_documents():
    """Lista arquivos no diretório de dados brutos."""
    settings = get_settings()

    raw_dir = settings.raw_data_dir

    if not raw_dir.exists():
        return {"documents": [], "total": 0}

    from src.ingestion.loader import LOADER_MAPPING

    supported = set(LOADER_MAPPING.keys())
    files = [
        {
            "filename": f.name,
            "file_type": f.suffix.lower(),
            "size_bytes": f.stat().st_size,
        }
        for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ]

    return {"documents": files, "total": len(files)}
