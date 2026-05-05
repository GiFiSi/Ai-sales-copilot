"""Merge de adapter LoRA com modelo base.

Após o fine-tuning, combina os pesos do adapter com o modelo
base para criar um modelo único otimizado.
"""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import get_settings
from src.exceptions import TrainingError

logger = logging.getLogger(__name__)


def merge_adapter(
    adapter_path: Path | None = None,
    output_path: Path | None = None,
    model_name: str | None = None,
    push_to_hub: bool = False,
    hub_model_name: str | None = None,
) -> Path:
    """Merge do adapter LoRA com o modelo base.

    Carrega o modelo base em fp16 (sem quantização),
    carrega o adapter LoRA por cima e faz o merge dos pesos.
    O resultado é um modelo único que pode ser distribuído.

    Args:
        adapter_path: Caminho para o adapter LoRA. Padrão: models/adapter.
        output_path: Caminho para salvar modelo merged. Padrão: models/merged.
        model_name: Nome do modelo base no HuggingFace.
        push_to_hub: Se True, envia o modelo para o HuggingFace Hub.
        hub_model_name: Nome do modelo no Hub (se push_to_hub=True).

    Returns:
        Path do diretório com o modelo merged.

    Raises:
        TrainingError: Se ocorrer erro no merge.
    """
    settings = get_settings()

    adapter_path = adapter_path or settings.models_dir / "adapter"
    output_path = output_path or settings.models_dir / "merged"
    model_name = model_name or settings.model.hf_model_name

    if not adapter_path.exists():
        raise TrainingError(f"Adapter não encontrado: {adapter_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Merge do Adapter LoRA")
    logger.info("=" * 60)
    logger.info("  Modelo base: %s", model_name)
    logger.info("  Adapter: %s", adapter_path)
    logger.info("  Output: %s", output_path)
    logger.info("=" * 60)

    try:
        # 1. Carrega modelo base em fp16 (sem quantização para merge correto)
        logger.info("[1/4] Carregando modelo base em fp16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=settings.model.hf_token,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=settings.model.hf_token,
            trust_remote_code=True,
        )

        # 2. Carrega adapter
        logger.info("[2/4] Carregando adapter LoRA...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        # 3. Merge
        logger.info("[3/4] Fazendo merge dos pesos...")
        model = model.merge_and_unload()

        # 4. Salva
        logger.info("[4/4] Salvando modelo merged...")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        # Push to Hub (opcional)
        if push_to_hub and hub_model_name:
            logger.info("Enviando para HuggingFace Hub: %s", hub_model_name)
            model.push_to_hub(hub_model_name, token=settings.model.hf_token)
            tokenizer.push_to_hub(hub_model_name, token=settings.model.hf_token)

        logger.info("=" * 60)
        logger.info("Merge concluído! Modelo salvo em: %s", output_path)
        logger.info("=" * 60)

        return output_path

    except Exception as e:
        raise TrainingError(f"Erro no merge do adapter: {e}") from e
