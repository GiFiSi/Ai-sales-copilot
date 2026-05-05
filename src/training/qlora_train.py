"""Fine-tuning QLoRA — Treinamento leve sobre modelo quantizado.

Usa PEFT (LoRA) + bitsandbytes (4-bit) + TRL (SFTTrainer)
para adaptar o modelo ao domínio específico com baixo consumo de VRAM.
"""

import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from src.config import get_settings
from src.exceptions import TrainingError
from src.model.quantize import get_bnb_config
from src.training.prepare_dataset import prepare_sft_dataset

logger = logging.getLogger(__name__)

# Hiperparâmetros padrão do blueprint
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "task_type": TaskType.CAUSAL_LM,
}

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "max_grad_norm": 0.3,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "fp16": True,
    "optim": "paged_adamw_32bit",
    "lr_scheduler_type": "cosine",
    "report_to": "none",
}


def run_qlora_training(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    max_seq_length: int = 512,
    lora_config_overrides: dict | None = None,
    training_args_overrides: dict | None = None,
) -> Path:
    """Executa fine-tuning QLoRA completo.

    Fluxo:
    1. Carrega modelo base em 4-bit
    2. Prepara para k-bit training
    3. Aplica LoRA adapter
    4. Treina com SFTTrainer
    5. Salva adapter weights

    Args:
        dataset_path: Caminho para o dataset JSONL. Padrão: data/training/dataset.jsonl.
        output_dir: Diretório para salvar o adapter. Padrão: models/adapter.
        model_name: Nome do modelo no HuggingFace. Padrão: config.
        max_seq_length: Tamanho máximo de sequência para treinamento.
        lora_config_overrides: Overrides para os hiperparâmetros LoRA.
        training_args_overrides: Overrides para os argumentos de treinamento.

    Returns:
        Path do diretório com o adapter salvo.

    Raises:
        TrainingError: Se ocorrer erro durante o treinamento.
    """
    settings = get_settings()

    # Resolve caminhos
    dataset_path = dataset_path or settings.training_data_dir / "dataset.jsonl"
    output_dir = output_dir or settings.models_dir / "adapter"
    model_name = model_name or settings.model.hf_model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QLoRA Fine-Tuning")
    logger.info("=" * 60)
    logger.info("  Modelo: %s", model_name)
    logger.info("  Dataset: %s", dataset_path)
    logger.info("  Output: %s", output_dir)
    logger.info("  Max seq length: %d", max_seq_length)
    logger.info("=" * 60)

    try:
        # --- 1. Prepara dataset ---
        logger.info("[1/5] Preparando dataset...")
        datasets = prepare_sft_dataset(dataset_path)
        train_dataset = datasets["train"]
        eval_dataset = datasets.get("test")

        # --- 2. Carrega modelo base em 4-bit ---
        logger.info("[2/5] Carregando modelo em 4-bit...")
        bnb_config = get_bnb_config()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=settings.model.hf_token,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=settings.model.hf_token,
            trust_remote_code=True,
        )

        # Log VRAM
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            logger.info("  VRAM após carregar modelo: %.2f GB", vram)

        # --- 3. Prepara modelo para k-bit training ---
        logger.info("[3/5] Preparando para k-bit training + LoRA...")
        model = prepare_model_for_kbit_training(model)

        # Configura LoRA
        lora_params = {**DEFAULT_LORA_CONFIG, **(lora_config_overrides or {})}
        lora_config = LoraConfig(**lora_params)

        model = get_peft_model(model, lora_config)

        # Log parâmetros treináveis
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "  Parâmetros: %s treináveis / %s total (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )

        # --- 4. Configura treinamento ---
        logger.info("[4/5] Configurando treinamento...")
        training_params = {**DEFAULT_TRAINING_ARGS, **(training_args_overrides or {})}
        training_params["output_dir"] = str(output_dir / "checkpoints")

        training_arguments = TrainingArguments(**training_params)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            args=training_arguments,
            max_seq_length=max_seq_length,
        )

        # --- 5. Treina ---
        logger.info("[5/5] Iniciando treinamento...")
        trainer.train()

        # Salva adapter
        logger.info("Salvando adapter em '%s'...", output_dir)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Log VRAM final
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            logger.info("VRAM final: %.2f GB", vram)

        logger.info("=" * 60)
        logger.info("Fine-tuning concluído! Adapter salvo em: %s", output_dir)
        logger.info("=" * 60)

        return output_dir

    except Exception as e:
        raise TrainingError(f"Erro no fine-tuning QLoRA: {e}") from e
