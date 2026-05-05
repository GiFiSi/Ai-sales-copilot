"""Schemas Pydantic para request/response da API.

Define os modelos de dados para validação e serialização
dos endpoints do AI Sales Copilot.
"""

from pydantic import BaseModel, Field


# ============================================================
# Chat
# ============================================================

class ChatRequest(BaseModel):
    """Request para o endpoint de chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pergunta do usuário em linguagem natural.",
        examples=["Qual o sombreamento da tela modelo X?"],
    )
    session_id: str | None = Field(
        default=None,
        description="ID da sessão para manter contexto (futuro).",
    )
    namespace: str = Field(
        default="default",
        description="Namespace do Pinecone para buscar.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperatura de geração (0.0 = determinístico, 1.0 = criativo).",
    )
    max_tokens: int = Field(
        default=512,
        ge=50,
        le=2048,
        description="Máximo de tokens na resposta.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de chunks a recuperar do Pinecone.",
    )


class Source(BaseModel):
    """Fonte documental usada na resposta."""

    source_file: str = Field(description="Nome do arquivo fonte.")
    page: int = Field(default=0, description="Número da página.")
    score: float = Field(description="Score de relevância (0.0 a 1.0).")


class ChatResponse(BaseModel):
    """Response do endpoint de chat."""

    answer: str = Field(description="Resposta gerada pelo modelo.")
    sources: list[Source] = Field(
        default_factory=list,
        description="Fontes documentais utilizadas.",
    )
    pii_detected: bool = Field(
        default=False,
        description="Se PII foi detectado e mascarado na pergunta.",
    )
    processing_time_ms: float = Field(
        description="Tempo de processamento em milissegundos.",
    )
    masked_entities: list[str] = Field(
        default_factory=list,
        description="Lista de entidades PII mascaradas.",
    )


# ============================================================
# Health
# ============================================================

class HealthResponse(BaseModel):
    """Response do health check."""

    status: str = Field(default="healthy", description="Status do serviço.")
    model_loaded: bool = Field(description="Se o modelo LLM está carregado.")
    pinecone_connected: bool = Field(description="Se o Pinecone está conectado.")
    pii_masking_enabled: bool = Field(description="Se o PII masking está ativo.")
    version: str = Field(description="Versão da aplicação.")


# ============================================================
# Documents
# ============================================================

class DocumentInfo(BaseModel):
    """Informação sobre um documento indexado."""

    filename: str = Field(description="Nome do arquivo.")
    file_type: str = Field(description="Tipo do arquivo (pdf, txt, etc).")
    chunks_count: int = Field(description="Número de chunks gerados.")
    namespace: str = Field(description="Namespace no Pinecone.")


class IngestRequest(BaseModel):
    """Request para ingestão de documentos."""

    directory: str | None = Field(
        default=None,
        description="Diretório com documentos. Padrão: data/raw/",
    )
    namespace: str = Field(
        default="default",
        description="Namespace no Pinecone para indexar.",
    )
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Tamanho do chunk em caracteres.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Sobreposição entre chunks.",
    )


class IngestResponse(BaseModel):
    """Response da ingestão de documentos."""

    documents_loaded: int = Field(description="Documentos carregados.")
    chunks_created: int = Field(description="Chunks gerados.")
    vectors_indexed: int = Field(description="Vetores indexados no Pinecone.")
    namespace: str = Field(description="Namespace utilizado.")


class DocumentListResponse(BaseModel):
    """Response da listagem de documentos."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = Field(description="Total de documentos.")


# ============================================================
# Errors
# ============================================================

class ErrorResponse(BaseModel):
    """Response padrão de erro."""

    detail: str = Field(description="Mensagem de erro.")
    error_type: str = Field(description="Tipo do erro.")
