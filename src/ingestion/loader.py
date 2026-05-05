"""Carregamento de documentos de múltiplos formatos.

Suporta PDF, TXT, CSV e DOCX usando LangChain document loaders.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.exceptions import DocumentLoadError

logger = logging.getLogger(__name__)

# Mapeamento de extensão → loader
LOADER_MAPPING: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

# Tenta importar DOCX loader (dependência opcional)
try:
    from langchain_community.document_loaders import Docx2txtLoader
    LOADER_MAPPING[".docx"] = Docx2txtLoader
except ImportError:
    logger.warning("python-docx não instalado — arquivos .docx não serão suportados.")


def load_single_document(file_path: Path) -> list[Document]:
    """Carrega um único documento e retorna lista de Documents.

    Args:
        file_path: Caminho absoluto para o arquivo.

    Returns:
        Lista de Documents do LangChain (um por página no caso de PDFs).

    Raises:
        DocumentLoadError: Se o formato não é suportado ou ocorre erro no loading.
    """
    ext = file_path.suffix.lower()

    if ext not in LOADER_MAPPING:
        raise DocumentLoadError(
            f"Formato não suportado: '{ext}'. "
            f"Formatos aceitos: {list(LOADER_MAPPING.keys())}"
        )

    loader_cls = LOADER_MAPPING[ext]
    logger.info("Carregando: %s (loader: %s)", file_path.name, loader_cls.__name__)

    try:
        loader = loader_cls(str(file_path))
        documents = loader.load()

        # Adiciona metadados úteis
        for doc in documents:
            doc.metadata["source_file"] = file_path.name
            doc.metadata["file_type"] = ext

        logger.info("  → %d documento(s) carregado(s) de %s", len(documents), file_path.name)
        return documents

    except Exception as e:
        raise DocumentLoadError(f"Erro ao carregar '{file_path.name}': {e}") from e


def load_documents(directory: Path) -> list[Document]:
    """Carrega todos os documentos de um diretório.

    Percorre recursivamente o diretório e carrega todos os arquivos
    com extensões suportadas.

    Args:
        directory: Caminho para o diretório com documentos.

    Returns:
        Lista combinada de todos os Documents carregados.

    Raises:
        DocumentLoadError: Se o diretório não existe ou está vazio.
    """
    if not directory.exists():
        raise DocumentLoadError(f"Diretório não encontrado: {directory}")

    supported_extensions = set(LOADER_MAPPING.keys())
    files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files:
        raise DocumentLoadError(
            f"Nenhum documento encontrado em '{directory}'. "
            f"Formatos aceitos: {supported_extensions}"
        )

    logger.info("Encontrados %d arquivos em '%s'", len(files), directory)

    all_documents: list[Document] = []
    errors: list[str] = []

    for file_path in sorted(files):
        try:
            docs = load_single_document(file_path)
            all_documents.extend(docs)
        except DocumentLoadError as e:
            errors.append(str(e))
            logger.error("Falha ao carregar %s: %s", file_path.name, e)

    if errors and not all_documents:
        raise DocumentLoadError(
            f"Nenhum documento carregado. Erros: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            "%d arquivo(s) falharam, %d documento(s) carregados com sucesso.",
            len(errors),
            len(all_documents),
        )

    logger.info("Total: %d documentos carregados com sucesso.", len(all_documents))
    return all_documents
