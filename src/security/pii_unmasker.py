"""Restauração de PII mascarado na resposta do LLM.

Substitui os placeholders de volta pelos valores originais
usando o mapeamento gerado pelo PIIMasker.
"""

import logging

from src.exceptions import PIIMaskingError

logger = logging.getLogger(__name__)


class PIIUnmasker:
    """Restaura dados mascarados na resposta do LLM.

    Usa o mapeamento {placeholder: valor_original} gerado
    pelo PIIMasker para substituir os placeholders de volta.
    """

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restaura PII mascarado na resposta.

        Args:
            text: Texto com placeholders (resposta do LLM).
            mapping: Dicionário {placeholder: valor_original} do PIIMasker.

        Returns:
            Texto com os dados originais restaurados.

        Raises:
            PIIMaskingError: Se ocorrer erro na restauração.
        """
        if not mapping:
            return text

        try:
            unmasked = text
            restored_count = 0

            for placeholder, original_value in mapping.items():
                if placeholder in unmasked:
                    unmasked = unmasked.replace(placeholder, original_value)
                    restored_count += 1

            if restored_count > 0:
                logger.info("PII restaurado: %d entidades desmascaradas.", restored_count)

            # Verifica se sobrou algum placeholder não restaurado
            remaining = [p for p in mapping if p in unmasked]
            if remaining:
                logger.warning(
                    "Placeholders não restaurados: %s (o LLM pode ter alterado o formato).",
                    remaining,
                )

            return unmasked

        except Exception as e:
            raise PIIMaskingError(f"Erro ao restaurar PII: {e}") from e

    def should_unmask(self, text: str, mapping: dict[str, str]) -> bool:
        """Verifica se o texto contém placeholders para restaurar.

        Args:
            text: Texto para verificar.
            mapping: Mapeamento de placeholders.

        Returns:
            True se existem placeholders no texto.
        """
        return any(placeholder in text for placeholder in mapping)
