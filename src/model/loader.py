"""Gerenciamento do modelo LLM quantizado.

Carrega o modelo e tokenizer do Hugging Face com quantização 4-bit,
gerencia cache e fornece interface de geração unificada.
"""

import logging
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import get_settings
from src.exceptions import ModelLoadError, ModelGenerationError
from src.model.quantize import get_bnb_config

logger = logging.getLogger(__name__)


class ModelManager:
    """Gerencia o modelo LLM quantizado (singleton thread-safe).

    Carrega o modelo uma única vez e reutiliza para todas as gerações.
    Suporta modelos com chat template (Qwen, Llama) e plain text.

    Attributes:
        model: Modelo HuggingFace carregado em 4-bit.
        tokenizer: Tokenizer correspondente ao modelo.
        model_name: Nome do modelo no HuggingFace Hub.
        device: Dispositivo de computação (cuda/cpu).
    """

    _instance: "ModelManager | None" = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs) -> "ModelManager":
        """Singleton pattern thread-safe."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        """Inicializa o model manager.

        Args:
            model_name: Nome do modelo no HuggingFace Hub.
                        Usa o padrão do .env se None.
        """
        if self._initialized:
            return

        settings = get_settings()
        self.model_name = model_name or settings.model.hf_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        logger.info(
            "ModelManager configurado: modelo=%s, device=%s",
            self.model_name,
            self.device,
        )

        self._initialized = True

    def load(self) -> None:
        """Carrega o modelo e tokenizer com quantização 4-bit.

        Raises:
            ModelLoadError: Se o modelo não pode ser carregado.
        """
        if self.model is not None:
            logger.info("Modelo já carregado, reutilizando.")
            return

        settings = get_settings()

        logger.info("Carregando modelo '%s'...", self.model_name)
        logger.info("  Device: %s", self.device)

        try:
            # Carrega tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=settings.model.hf_token,
                trust_remote_code=True,
            )

            # Configura pad token se necessário
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Carrega modelo com quantização
            if self.device == "cuda":
                bnb_config = get_bnb_config()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=settings.model.hf_token,
                    trust_remote_code=True,
                )
                logger.info("Modelo carregado em 4-bit (GPU).")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    token=settings.model.hf_token,
                    trust_remote_code=True,
                )
                logger.warning("Modelo carregado em CPU (sem quantização). Inferência será lenta.")

            # Log de VRAM
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                logger.info("VRAM: %.2f GB / %.2f GB", vram_used, vram_total)

        except Exception as e:
            raise ModelLoadError(f"Erro ao carregar modelo '{self.model_name}': {e}") from e

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Gera texto a partir de mensagens no formato chat.

        Args:
            messages: Lista de dicts [{"role": "system/user", "content": "..."}].
            max_new_tokens: Máximo de tokens a gerar.
            temperature: Temperatura de amostragem (0.0 = determinístico).
            top_p: Nucleus sampling threshold.
            do_sample: Se True, usa amostragem; se False, greedy.

        Returns:
            Texto gerado pelo modelo (apenas a resposta, sem o prompt).

        Raises:
            ModelGenerationError: Se ocorrer erro na geração.
        """
        # Garante que o modelo está carregado
        if self.model is None or self.tokenizer is None:
            self.load()

        try:
            # Aplica chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokeniza
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            # Gera
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decodifica apenas os tokens novos
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logger.info(
                "Geração: %d tokens input → %d tokens output.",
                input_length,
                len(generated_tokens),
            )

            return response.strip()

        except Exception as e:
            raise ModelGenerationError(f"Erro na geração: {e}") from e

    def load_adapter(self, adapter_path: str | Path) -> None:
        """Carrega um adapter LoRA treinado.

        Args:
            adapter_path: Caminho para o diretório do adapter.

        Raises:
            ModelLoadError: Se o adapter não pode ser carregado.
        """
        if self.model is None:
            self.load()

        try:
            from peft import PeftModel

            adapter_path = Path(adapter_path)
            if not adapter_path.exists():
                raise ModelLoadError(f"Adapter não encontrado: {adapter_path}")

            logger.info("Carregando adapter LoRA de '%s'...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            logger.info("Adapter LoRA carregado com sucesso.")

        except ImportError:
            raise ModelLoadError("Biblioteca 'peft' não instalada para carregar adapter.")
        except Exception as e:
            raise ModelLoadError(f"Erro ao carregar adapter: {e}") from e

    @property
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self.model is not None and self.tokenizer is not None
