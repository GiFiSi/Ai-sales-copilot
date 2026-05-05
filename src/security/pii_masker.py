"""Mascaramento de PII (Informações Pessoais Identificáveis).

Combina o Microsoft Presidio com regex customizado
para detectar e mascarar CPF, CNPJ, e-mail, telefone e nomes.
"""

import logging
import re
from typing import Any

from src.exceptions import PIIMaskingError

logger = logging.getLogger(__name__)

# ============================================================
# Regex patterns para documentos brasileiros
# ============================================================

BRAZILIAN_PII_PATTERNS: dict[str, re.Pattern] = {
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNPJ": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    "TELEFONE": re.compile(r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}"),
    "CEP": re.compile(r"\b\d{5}-?\d{3}\b"),
}

# Mapa de placeholder por entidade
PLACEHOLDER_MAP: dict[str, str] = {
    "CPF": "[CPF_MASCARADO]",
    "CNPJ": "[CNPJ_MASCARADO]",
    "TELEFONE": "[TELEFONE_MASCARADO]",
    "CEP": "[CEP_MASCARADO]",
    "EMAIL": "[EMAIL_MASCARADO]",
    "PERSON": "[NOME_MASCARADO]",
}


class PIIMasker:
    """Detecta e mascara informações pessoais identificáveis.

    Usa uma combinação de regex (para CPF, CNPJ, telefone)
    e Presidio (para e-mails e nomes) para máxima cobertura.

    Attributes:
        use_presidio: Se o Presidio está disponível e ativo.
        analyzer: Presidio AnalyzerEngine (se disponível).
        anonymizer: Presidio AnonymizerEngine (se disponível).
    """

    def __init__(self) -> None:
        """Inicializa o masker, tentando carregar o Presidio."""
        self.use_presidio = False
        self.analyzer: Any = None
        self.anonymizer: Any = None

        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

            # Adiciona recognizers customizados para BR
            cpf_pattern = Pattern(name="cpf_pattern", regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", score=0.9)
            cpf_recognizer = PatternRecognizer(
                supported_entity="BR_CPF",
                patterns=[cpf_pattern],
                supported_language="pt",
            )
            self.analyzer.registry.add_recognizer(cpf_recognizer)

            cnpj_pattern = Pattern(name="cnpj_pattern", regex=r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b", score=0.9)
            cnpj_recognizer = PatternRecognizer(
                supported_entity="BR_CNPJ",
                patterns=[cnpj_pattern],
                supported_language="pt",
            )
            self.analyzer.registry.add_recognizer(cnpj_recognizer)

            self.use_presidio = True
            logger.info("PIIMasker inicializado com Presidio + regex customizado.")

        except ImportError:
            logger.warning(
                "Presidio não instalado. Usando apenas regex para PII masking. "
                "Instale com: pip install presidio-analyzer presidio-anonymizer"
            )

    def _mask_with_regex(self, text: str) -> tuple[str, dict[str, str]]:
        """Mascara PII usando regex patterns.

        Args:
            text: Texto para mascarar.

        Returns:
            Tupla (texto_mascarado, mapeamento de placeholder→valor_original).
        """
        mapping: dict[str, str] = {}
        masked_text = text

        for entity_type, pattern in BRAZILIAN_PII_PATTERNS.items():
            matches = pattern.findall(masked_text)
            placeholder_base = PLACEHOLDER_MAP.get(entity_type, f"[{entity_type}_MASCARADO]")

            for i, match in enumerate(matches):
                if match in mapping.values():
                    continue  # Já mascarado

                if len(matches) > 1:
                    placeholder = f"{placeholder_base[:-1]}_{i + 1}]"
                else:
                    placeholder = placeholder_base

                masked_text = masked_text.replace(match, placeholder, 1)
                mapping[placeholder] = match

        # Email via regex (fallback se Presidio não disponível)
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        emails = email_pattern.findall(masked_text)
        for i, email in enumerate(emails):
            placeholder = "[EMAIL_MASCARADO]" if len(emails) == 1 else f"[EMAIL_MASCARADO_{i + 1}]"
            masked_text = masked_text.replace(email, placeholder, 1)
            mapping[placeholder] = email

        return masked_text, mapping

    def _mask_with_presidio(self, text: str) -> tuple[str, dict[str, str]]:
        """Mascara PII usando Presidio + regex.

        Args:
            text: Texto para mascarar.

        Returns:
            Tupla (texto_mascarado, mapeamento de placeholder→valor_original).
        """
        # Primeiro aplica regex para padrões BR
        masked_text, mapping = self._mask_with_regex(text)

        # Depois aplica Presidio para entidades genéricas (nomes, emails)
        try:
            results = self.analyzer.analyze(
                text=masked_text,
                entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
                language="pt",
                score_threshold=0.7,
            )

            # Ordena do final para o início para não bagunçar os índices
            results = sorted(results, key=lambda r: r.start, reverse=True)

            for result in results:
                original = masked_text[result.start:result.end]

                # Pula se já está mascarado
                if original.startswith("[") and original.endswith("]"):
                    continue

                entity = result.entity_type
                placeholder_base = PLACEHOLDER_MAP.get(entity, f"[{entity}_MASCARADO]")

                # Evita duplicatas
                placeholder = placeholder_base
                counter = 1
                while placeholder in mapping:
                    placeholder = f"{placeholder_base[:-1]}_{counter}]"
                    counter += 1

                masked_text = masked_text[:result.start] + placeholder + masked_text[result.end:]
                mapping[placeholder] = original

        except Exception as e:
            logger.warning("Erro no Presidio (fallback para regex): %s", e)

        return masked_text, mapping

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Mascara todas as PII encontradas no texto.

        Args:
            text: Texto do usuário para mascarar.

        Returns:
            Tupla com:
                - Texto com PII substituída por placeholders
                - Dicionário {placeholder: valor_original}

        Raises:
            PIIMaskingError: Se ocorrer erro no mascaramento.
        """
        if not text or not text.strip():
            return text, {}

        try:
            if self.use_presidio:
                masked_text, mapping = self._mask_with_presidio(text)
            else:
                masked_text, mapping = self._mask_with_regex(text)

            if mapping:
                logger.info("PII mascarado: %d entidades encontradas.", len(mapping))
                for placeholder in mapping:
                    logger.debug("  %s → [REDACTED]", placeholder)

            return masked_text, mapping

        except Exception as e:
            raise PIIMaskingError(f"Erro ao mascarar PII: {e}") from e

    def detect(self, text: str) -> list[dict[str, str]]:
        """Detecta PII sem mascarar (apenas identificação).

        Args:
            text: Texto para analisar.

        Returns:
            Lista de dicts com 'entity_type' e 'value'.
        """
        _, mapping = self.mask(text)

        return [
            {"entity_type": placeholder.strip("[]").replace("_MASCARADO", ""), "value": value}
            for placeholder, value in mapping.items()
        ]
