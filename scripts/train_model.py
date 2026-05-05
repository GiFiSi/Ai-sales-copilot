"""Script CLI — Fine-tuning QLoRA.

Executa o treinamento do modelo com QLoRA usando o dataset configurado.

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --dataset ./meu_dataset.jsonl --epochs 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Função principal do script de treinamento."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning QLoRA do AI Sales Copilot",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Caminho para o dataset JSONL (padrão: data/training/dataset.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Diretório de saída do adapter (padrão: models/adapter)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas de treinamento (padrão: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (padrão: 2e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size por device (padrão: 2)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Tamanho máximo de sequência (padrão: 512)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Fazer merge do adapter após treinar",
    )

    args = parser.parse_args()
    settings = get_settings()

    dataset_path = Path(args.dataset) if args.dataset else settings.training_data_dir / "dataset.jsonl"
    output_dir = Path(args.output) if args.output else settings.models_dir / "adapter"

    logger.info("=" * 60)
    logger.info("AI Sales Copilot — Fine-Tuning QLoRA")
    logger.info("=" * 60)
    logger.info("  Dataset: %s", dataset_path)
    logger.info("  Output: %s", output_dir)
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  LR: %s", args.lr)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Max seq length: %d", args.max_seq_length)
    logger.info("=" * 60)

    try:
        from src.training.qlora_train import run_qlora_training

        adapter_path = run_qlora_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            max_seq_length=args.max_seq_length,
            training_args_overrides={
                "num_train_epochs": args.epochs,
                "learning_rate": args.lr,
                "per_device_train_batch_size": args.batch_size,
            },
        )

        logger.info("Adapter salvo em: %s", adapter_path)

        # Merge opcional
        if args.merge:
            logger.info("Fazendo merge do adapter...")
            from src.training.merge_adapter import merge_adapter
            merged_path = merge_adapter(adapter_path=adapter_path)
            logger.info("Modelo merged salvo em: %s", merged_path)

    except Exception as e:
        logger.error("ERRO no treinamento: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
