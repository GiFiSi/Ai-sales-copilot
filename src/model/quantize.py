"""Configuração de quantização 4-bit com BitsAndBytes.

Define a configuração NF4 (Normal Float 4) para compressão
do modelo LLM, permitindo inferência em GPUs consumer (4GB VRAM).
"""

import logging

import torch
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


def get_bnb_config() -> BitsAndBytesConfig:
    """Retorna configuração de quantização 4-bit otimizada.

    Configuração NF4 com double quantization para máxima
    compressão mantendo qualidade aceitável de geração.

    Returns:
        BitsAndBytesConfig configurado para 4-bit.

    Notes:
        - load_in_4bit: Ativa quantização 4-bit
        - bnb_4bit_quant_type='nf4': Normal Float 4, melhor qualidade que int4
        - bnb_4bit_compute_dtype=float16: Precisão de computação
        - bnb_4bit_use_double_quant: Quantiza os parâmetros de quantização
          (reduz ~0.4 bits/param extra, economizando ~0.3GB no modelo de 1.5B)
    """
    compute_dtype = torch.float16

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(
        "BitsAndBytes config: 4-bit NF4, compute_dtype=%s, double_quant=True",
        compute_dtype,
    )

    return config


def estimate_vram_usage(model_params_billions: float) -> dict[str, float]:
    """Estima o uso de VRAM para um modelo em diferentes quantizações.

    Args:
        model_params_billions: Número de parâmetros em bilhões (ex: 1.5).

    Returns:
        Dict com estimativas de VRAM em GB para cada tipo de quantização.
    """
    params = model_params_billions

    return {
        "fp32": round(params * 4, 2),       # 4 bytes/param
        "fp16": round(params * 2, 2),       # 2 bytes/param
        "int8": round(params * 1, 2),       # 1 byte/param
        "int4": round(params * 0.5, 2),     # 0.5 bytes/param
        "int4_double_quant": round(params * 0.5 * 0.9, 2),  # ~10% a menos
    }
