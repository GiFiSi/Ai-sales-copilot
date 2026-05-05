"""Preparação de datasets para fine-tuning supervisionado (SFT).

Lê arquivos JSONL com pares instrução/resposta e formata
no template de chat do modelo para treinamento com SFTTrainer.
"""

import json
import logging
from pathlib import Path

from datasets import Dataset

from src.exceptions import TrainingError

logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> list[dict]:
    """Carrega um arquivo JSONL.

    Args:
        file_path: Caminho para o arquivo .jsonl

    Returns:
        Lista de dicionários, um por linha.

    Raises:
        TrainingError: Se o arquivo não existe ou está mal formatado.
    """
    if not file_path.exists():
        raise TrainingError(f"Dataset não encontrado: {file_path}")

    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning("Linha %d inválida no JSONL: %s", line_num, e)

    except Exception as e:
        raise TrainingError(f"Erro ao ler dataset: {e}") from e

    logger.info("Carregados %d exemplos de '%s'.", len(data), file_path.name)
    return data


def format_for_chat(
    examples: list[dict],
    system_prompt: str = "Você é um assistente corporativo especializado.",
) -> list[str]:
    """Formata exemplos no formato de chat para SFT.

    Converte cada exemplo {instruction, input, output} para
    o formato de mensagens de chat usado pelo tokenizer.

    Args:
        examples: Lista de dicts com 'instruction' e 'output' (e 'input' opcional).
        system_prompt: Prompt de sistema para o assistente.

    Returns:
        Lista de strings formatadas como conversação.
    """
    formatted = []

    for item in examples:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        # Combina instruction + input
        if input_text:
            user_content = f"{instruction}\n\nContexto: {input_text}"
        else:
            user_content = instruction

        # Formato de chat
        conversation = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )

        formatted.append(conversation)

    return formatted


def prepare_sft_dataset(
    jsonl_path: Path,
    system_prompt: str = "Você é um assistente corporativo especializado.",
    test_size: float = 0.1,
) -> dict[str, Dataset]:
    """Prepara dataset completo para SFT training.

    Args:
        jsonl_path: Caminho para o arquivo JSONL com exemplos.
        system_prompt: Prompt de sistema para incluir no chat.
        test_size: Proporção para split de validação (0.0 a 1.0).

    Returns:
        Dict com 'train' e 'test' datasets do HuggingFace.

    Raises:
        TrainingError: Se o dataset é inválido ou muito pequeno.
    """
    # Carrega e valida
    examples = load_jsonl(jsonl_path)

    if len(examples) < 5:
        raise TrainingError(
            f"Dataset muito pequeno ({len(examples)} exemplos). "
            "Mínimo recomendado: 50 exemplos."
        )

    # Valida campos obrigatórios
    required_fields = {"instruction", "output"}
    for i, item in enumerate(examples):
        missing = required_fields - set(item.keys())
        if missing:
            raise TrainingError(
                f"Exemplo #{i + 1} faltando campos: {missing}. "
                f"Campos necessários: {required_fields}"
            )

    # Formata para chat
    formatted_texts = format_for_chat(examples, system_prompt)

    # Cria dataset HuggingFace
    dataset = Dataset.from_dict({"text": formatted_texts})

    # Split train/test
    if test_size > 0 and len(dataset) >= 10:
        split = dataset.train_test_split(test_size=test_size, seed=42)
        logger.info(
            "Dataset preparado: %d train, %d test.",
            len(split["train"]),
            len(split["test"]),
        )
        return {"train": split["train"], "test": split["test"]}
    else:
        logger.info("Dataset preparado: %d exemplos (sem split).", len(dataset))
        return {"train": dataset, "test": None}
