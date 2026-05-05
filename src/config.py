"""Configuração centralizada do projeto via variáveis de ambiente.

Usa Pydantic Settings para carregar e validar o `.env` automaticamente.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Raiz do projeto (2 níveis acima deste arquivo)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PineconeSettings(BaseSettings):
    """Configurações do Pinecone Vector Store."""

    model_config = SettingsConfigDict(env_prefix="PINECONE_")

    api_key: str = "your_pinecone_api_key_here"
    index_name: str = "sales-copilot"
    environment: str = "us-east-1"


class ModelSettings(BaseSettings):
    """Configurações do modelo LLM e embeddings."""

    model_config = SettingsConfigDict(env_prefix="")

    hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_token: str = "your_huggingface_token_here"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384


class APISettings(BaseSettings):
    """Configurações do servidor FastAPI."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000


class SecuritySettings(BaseSettings):
    """Configurações de segurança e PII masking."""

    model_config = SettingsConfigDict(env_prefix="")

    enable_pii_masking: bool = True
    spacy_model: str = "pt_core_news_sm"


class Settings(BaseSettings):
    """Configuração principal que agrega todas as sub-configurações."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurações
    pinecone: PineconeSettings = PineconeSettings()
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()

    # Geral
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"

    @property
    def raw_data_dir(self) -> Path:
        """Diretório de documentos originais."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Diretório de chunks processados."""
        return self.data_dir / "processed"

    @property
    def training_data_dir(self) -> Path:
        """Diretório de dados de treinamento."""
        return self.data_dir / "training"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna instância singleton das configurações.

    Returns:
        Settings: Configurações carregadas do .env
    """
    return Settings()
