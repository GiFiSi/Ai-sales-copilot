"""Exceções customizadas do AI Sales Copilot.

Cada módulo lança exceções específicas para facilitar
o tratamento de erros e debugging.
"""


class SalesCopilotError(Exception):
    """Exceção base para todos os erros do projeto."""

    pass


class DocumentLoadError(SalesCopilotError):
    """Erro ao carregar documentos (PDF, TXT, DOCX, CSV)."""

    pass


class ChunkingError(SalesCopilotError):
    """Erro durante o text splitting / chunking."""

    pass


class EmbeddingError(SalesCopilotError):
    """Erro ao gerar embeddings ou fazer upsert no Pinecone."""

    pass


class PIIMaskingError(SalesCopilotError):
    """Erro no mascaramento ou desmascaramento de PII."""

    pass


class ModelLoadError(SalesCopilotError):
    """Erro ao carregar o modelo LLM ou tokenizer."""

    pass


class ModelGenerationError(SalesCopilotError):
    """Erro durante a geração de texto pelo LLM."""

    pass


class RAGChainError(SalesCopilotError):
    """Erro no pipeline RAG (retrieval + generation)."""

    pass


class TrainingError(SalesCopilotError):
    """Erro durante o fine-tuning QLoRA."""

    pass


class ConfigurationError(SalesCopilotError):
    """Erro de configuração (variáveis de ambiente ausentes, etc.)."""

    pass
