"""FastAPI application — Entry point do backend.

Configura CORS, lifespan (startup/shutdown) e inclui rotas.
O modelo LLM é carregado no startup para evitar latência na primeira request.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import router, set_rag_chain
from src.config import get_settings

# Configura logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação.

    Startup: Carrega o modelo LLM e inicializa o RAG chain.
    Shutdown: Libera recursos.
    """
    logger.info("=" * 60)
    logger.info("AI Sales Copilot v%s — Iniciando...", __version__)
    logger.info("=" * 60)

    try:
        from src.rag.chain import RAGChain

        chain = RAGChain()
        set_rag_chain(chain)

        logger.info("RAG Chain carregado e pronto para uso!")

    except Exception as e:
        logger.error("Erro ao inicializar RAG Chain: %s", e)
        logger.warning("API rodando em modo degradado (sem modelo).")

    yield  # App rodando

    logger.info("Shutting down...")


# Cria app FastAPI
app = FastAPI(
    title="AI Sales Copilot",
    description=(
        "Assistente RAG corporativo com LLM quantizado, "
        "PII masking e busca semântica em documentos internos."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite acesso do Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclui rotas
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect para documentação."""
    return {
        "message": "AI Sales Copilot API",
        "version": __version__,
        "docs": "/docs",
    }
