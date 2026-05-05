"""RAG Chain — Pipeline completo de Retrieval-Augmented Generation.

Orquestra: PII Masking → Retrieval → Prompt Formatting → LLM → PII Unmask.
"""

import logging
import time

from src.config import get_settings
from src.exceptions import RAGChainError
from src.model.loader import ModelManager
from src.rag.prompts import SYSTEM_PROMPT, format_chat_messages
from src.rag.retriever import PineconeRetriever
from src.security.pii_masker import PIIMasker
from src.security.pii_unmasker import PIIUnmasker

logger = logging.getLogger(__name__)


class RAGChain:
    """Pipeline RAG completo com PII masking.

    Orquestra todo o fluxo desde a pergunta do usuário
    até a resposta final, passando por mascaramento,
    busca vetorial e geração com LLM.

    Attributes:
        retriever: Busca vetorial no Pinecone.
        model_manager: Gerenciador do modelo LLM.
        pii_masker: Mascaramento de dados sensíveis.
        pii_unmasker: Restauração de dados mascarados.
        pii_enabled: Se o mascaramento está ativo.
    """

    def __init__(
        self,
        retriever: PineconeRetriever | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        """Inicializa o pipeline RAG.

        Args:
            retriever: Instância do retriever (cria novo se None).
            model_manager: Instância do model manager (cria novo se None).
        """
        settings = get_settings()

        logger.info("Inicializando RAG Chain...")

        self.retriever = retriever or PineconeRetriever(top_k=5)
        self.model_manager = model_manager or ModelManager()
        self.pii_enabled = settings.security.enable_pii_masking

        if self.pii_enabled:
            self.pii_masker = PIIMasker()
            self.pii_unmasker = PIIUnmasker()
            logger.info("PII Masking: ATIVADO")
        else:
            self.pii_masker = None
            self.pii_unmasker = None
            logger.info("PII Masking: DESATIVADO")

        logger.info("RAG Chain inicializado com sucesso.")

    def ask(
        self,
        question: str,
        namespace: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_k: int = 5,
    ) -> dict:
        """Processa uma pergunta e retorna resposta completa.

        Fluxo:
        1. Mascara PII na pergunta (se ativado)
        2. Busca contexto relevante no Pinecone
        3. Formata prompt com contexto + pergunta
        4. Gera resposta com o LLM
        5. Desmascara PII na resposta (se necessário)

        Args:
            question: Pergunta do usuário em linguagem natural.
            namespace: Namespace do Pinecone para buscar.
            temperature: Temperatura de geração (0.0 a 1.0).
            max_tokens: Máximo de tokens na resposta.
            top_k: Número de chunks a recuperar.

        Returns:
            Dict com:
                - answer (str): Resposta gerada
                - sources (list): Fontes utilizadas
                - pii_detected (bool): Se PII foi detectado
                - processing_time_ms (float): Tempo de processamento
                - masked_entities (list): Entidades mascaradas (se houver)

        Raises:
            RAGChainError: Se ocorrer erro em qualquer etapa.
        """
        start_time = time.time()
        pii_detected = False
        pii_mapping = {}
        masked_entities = []

        try:
            # --- Etapa 1: PII Masking ---
            processed_question = question
            if self.pii_enabled and self.pii_masker:
                processed_question, pii_mapping = self.pii_masker.mask(question)
                if pii_mapping:
                    pii_detected = True
                    masked_entities = list(pii_mapping.keys())
                    logger.info("PII detectado: %d entidades mascaradas.", len(pii_mapping))

            # --- Etapa 2: Retrieval ---
            documents = self.retriever.retrieve(
                query=processed_question,
                top_k=top_k,
                namespace=namespace,
            )
            context = self.retriever.format_context(documents)

            # --- Etapa 3: Format Prompt ---
            messages = format_chat_messages(
                system_prompt=SYSTEM_PROMPT,
                context=context,
                question=processed_question,
            )

            # --- Etapa 4: LLM Generation ---
            answer = self.model_manager.generate(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            # --- Etapa 5: PII Unmasking ---
            if pii_detected and self.pii_unmasker and pii_mapping:
                answer = self.pii_unmasker.unmask(answer, pii_mapping)

            # --- Formata resultado ---
            processing_time_ms = round((time.time() - start_time) * 1000, 2)

            sources = [
                {
                    "source_file": doc["source_file"],
                    "page": doc.get("page", 0),
                    "score": doc["score"],
                }
                for doc in documents
            ]

            result = {
                "answer": answer,
                "sources": sources,
                "pii_detected": pii_detected,
                "processing_time_ms": processing_time_ms,
                "masked_entities": masked_entities,
            }

            logger.info(
                "Resposta gerada em %.0fms (fontes: %d, pii: %s).",
                processing_time_ms,
                len(sources),
                pii_detected,
            )

            return result

        except Exception as e:
            raise RAGChainError(f"Erro no pipeline RAG: {e}") from e
