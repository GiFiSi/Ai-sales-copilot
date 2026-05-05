"""Script CLI — Ingestão de documentos.

Carrega documentos do diretório data/raw/, faz chunking,
gera embeddings e indexa no Pinecone.

Uso:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --directory ./meus_docs --namespace produtos
"""

import argparse
import logging
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents, get_chunk_stats
from src.ingestion.embedder import DocumentEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Função principal do script de ingestão."""
    parser = argparse.ArgumentParser(
        description="Ingestão de documentos para o AI Sales Copilot",
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=None,
        help="Diretório com documentos (padrão: data/raw/)",
    )
    parser.add_argument(
        "--namespace", "-n",
        type=str,
        default="default",
        help="Namespace no Pinecone (padrão: default)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Tamanho do chunk em caracteres (padrão: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Sobreposição entre chunks (padrão: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size para embeddings (padrão: 32)",
    )

    args = parser.parse_args()
    settings = get_settings()

    # Resolve diretório
    directory = Path(args.directory) if args.directory else settings.raw_data_dir

    logger.info("=" * 60)
    logger.info("AI Sales Copilot — Ingestão de Documentos")
    logger.info("=" * 60)
    logger.info("  Diretório: %s", directory)
    logger.info("  Namespace: %s", args.namespace)
    logger.info("  Chunk size: %d", args.chunk_size)
    logger.info("  Chunk overlap: %d", args.chunk_overlap)
    logger.info("=" * 60)

    try:
        # 1. Carrega documentos
        logger.info("[1/3] Carregando documentos...")
        documents = load_documents(directory)

        # 2. Chunking
        logger.info("[2/3] Criando chunks...")
        chunks = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        stats = get_chunk_stats(chunks)
        logger.info("  Estatísticas: %s", stats)

        # 3. Embedding + Indexação
        logger.info("[3/3] Gerando embeddings e indexando no Pinecone...")
        embedder = DocumentEmbedder()
        vectors_count = embedder.embed_and_store(
            chunks,
            namespace=args.namespace,
            batch_size=args.batch_size,
        )

        logger.info("=" * 60)
        logger.info("INGESTÃO CONCLUÍDA!")
        logger.info("  Documentos: %d", len(documents))
        logger.info("  Chunks: %d", len(chunks))
        logger.info("  Vetores indexados: %d", vectors_count)
        logger.info("  Namespace: %s", args.namespace)
        logger.info("=" * 60)

    except Exception as e:
        logger.error("ERRO na ingestão: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
