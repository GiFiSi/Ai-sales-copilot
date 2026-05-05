# 🤖 AI Sales Copilot — Project Blueprint

> Um assistente RAG corporativo com LLM local quantizado, fine-tuning QLoRA, mascaramento de PII e interface de chat moderna.

---

## 1. Visão Geral

| Item                | Descrição                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------- |
| **Objetivo**        | Agente de IA que responde perguntas sobre produtos/serviços com base em documentos internos |
| **Stack Core**      | Python 3.11+, LangChain, Pinecone, Hugging Face, FastAPI, Streamlit                         |
| **Hardware Mínimo** | GPU com 4GB VRAM (ex: GTX 1650) — modelo roda quantizado em 4-bit                           |
| **Modelo Base**     | Qwen2.5-1.5B ou Llama-3.2-1B (open-source, Hugging Face)                                    |
| **Tipo**            | Template genérico — adaptável a qualquer empresa/domínio                                    |

---

## 2. Arquitetura

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Streamlit   │────▶│   FastAPI     │────▶│  PII Masking  │
│  (Frontend)  │◀────│   (Backend)   │     │  (Presidio)   │
└─────────────┘     └──────┬───────┘     └───────┬───────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌───────────────┐
                    │  LangChain   │────▶│   Pinecone    │
                    │  Orchestrator│     │  (VectorDB)   │
                    └──────┬───────┘     └───────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  LLM Local   │
                    │  4-bit QLoRA │
                    └──────────────┘
```

### Fluxo de Dados

1. Usuário envia pergunta pelo **Streamlit**
2. **FastAPI** recebe e encaminha ao **Presidio** para mascarar PII
3. Pergunta limpa vai ao **LangChain**, que busca contexto no **Pinecone**
4. Contexto + pergunta vão ao **LLM quantizado** (4-bit)
5. Resposta retorna ao usuário pela interface

---

## 3. Estrutura do Repositório

```
project-root/
├── README.md                    # Vitrine do projeto
├── PROJECT_BLUEPRINT.md         # Este documento
├── requirements.txt
├── .env.example
├── docker-compose.yml           # (opcional)
│
├── data/
│   ├── raw/                     # PDFs e documentos originais
│   ├── processed/               # Chunks processados
│   └── training/
│       └── dataset.jsonl        # Dataset de fine-tuning
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Variáveis de ambiente e constantes
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py            # Carrega PDFs/docs (LangChain)
│   │   ├── chunker.py           # Text splitting strategies
│   │   └── embedder.py          # Gera embeddings + upsert Pinecone
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Busca vetorial no Pinecone
│   │   ├── chain.py             # RAG chain (LangChain)
│   │   └── prompts.py           # System/User prompt templates
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── pii_masker.py        # Mascaramento com Presidio
│   │   └── pii_unmasker.py      # Restauração pós-resposta
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── loader.py            # Carrega modelo quantizado
│   │   └── quantize.py          # Config bitsandbytes 4-bit
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── prepare_dataset.py   # Formata dados para SFT
│   │   ├── qlora_train.py       # Script de fine-tuning QLoRA
│   │   └── merge_adapter.py     # Merge LoRA weights
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       ├── routes.py            # Endpoints
│       └── schemas.py           # Pydantic models
│
├── frontend/
│   └── app.py                   # Streamlit chat interface
│
├── scripts/
│   ├── ingest_documents.py      # CLI: processa e indexa docs
│   ├── train_model.py           # CLI: executa fine-tuning
│   └── run_all.py               # CLI: sobe API + frontend
│
├── tests/
│   ├── test_pii_masker.py
│   ├── test_retriever.py
│   └── test_api.py
│
└── docs/
    ├── architecture_diagram.png
    └── setup_guide.md
```

---

## 4. Componentes Detalhados

### 4.1 Ingestão de Documentos (RAG Pipeline)

**Arquivo:** `src/ingestion/loader.py`

```python
# Bibliotecas: langchain_community.document_loaders
# Suporte: PDF, DOCX, TXT, CSV
# Loader recomendado: PyPDFLoader, UnstructuredLoader
```

**Regras de Chunking** (`src/ingestion/chunker.py`):
- Estratégia: `RecursiveCharacterTextSplitter`
- `chunk_size`: 512 tokens
- `chunk_overlap`: 50 tokens
- Metadados preservados: nome do arquivo, página, seção

**Embeddings** (`src/ingestion/embedder.py`):
- Modelo: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- Alternativa PT-BR: `neuralmind/bert-base-portuguese-cased`
- Batch size: 32
- Upsert no Pinecone com namespace por categoria de documento

### 4.2 Pinecone (Vector Store)

| Config     | Valor                                                        |
| ---------- | ------------------------------------------------------------ |
| Plano      | Free Tier (suficiente)                                       |
| Dimensão   | 384 (match com embedding model)                              |
| Métrica    | Cosine                                                       |
| Index Name | `sales-copilot`                                              |
| Namespace  | Por categoria de doc (ex: `produtos`, `fiscal`, `politicas`) |
| Top-K      | 3–5 chunks por query                                         |

### 4.3 PII Masking (Segurança)

**Arquivo:** `src/security/pii_masker.py`

Entidades a detectar e mascarar:

| Entidade     | Regex/Presidio                        | Substituição           |
| ------------ | ------------------------------------- | ---------------------- |
| CPF          | `\d{3}\.?\d{3}\.?\d{3}-?\d{2}`        | `[CPF_MASCARADO]`      |
| CNPJ         | `\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}` | `[CNPJ_MASCARADO]`     |
| Email        | Presidio built-in                     | `[EMAIL_MASCARADO]`    |
| Telefone     | `(\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}`     | `[TELEFONE_MASCARADO]` |
| Nome Próprio | Presidio NER                          | `[NOME_MASCARADO]`     |

**Fluxo:**
1. Input → Presidio `AnalyzerEngine` detecta entidades
2. `AnonymizerEngine` substitui por placeholders
3. Armazena mapa `{placeholder: valor_real}` em memória
4. Após resposta do LLM, `DeanonymizeEngine` restaura (se necessário)

### 4.4 Modelo LLM — Quantização 4-bit

**Arquivo:** `src/model/quantize.py`

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)
```

| Parâmetro                   | Valor   | Justificativa                                  |
| --------------------------- | ------- | ---------------------------------------------- |
| `load_in_4bit`              | True    | Reduz VRAM de ~6GB para ~1.5GB                 |
| `bnb_4bit_quant_type`       | nf4     | Normal Float 4 — melhor qualidade              |
| `bnb_4bit_compute_dtype`    | float16 | Balanço performance/precisão                   |
| `bnb_4bit_use_double_quant` | True    | Quantiza os próprios parâmetros de quantização |

**Modelos recomendados (ordem de preferência):**
1. `Qwen/Qwen2.5-1.5B-Instruct` — melhor custo-benefício
2. `meta-llama/Llama-3.2-1B-Instruct` — mais popular
3. `microsoft/phi-2` — 2.7B mas roda em 4-bit na 1650

### 4.5 Fine-Tuning com QLoRA

**Arquivo:** `src/training/qlora_train.py`

**Dataset** (`data/training/dataset.jsonl`) — mínimo 50 exemplos:

```json
{
  "instruction": "Qual o sombreamento da tela agrícola modelo X?",
  "input": "",
  "output": "A tela agrícola modelo X oferece sombreamento de 50%, ideal para culturas que necessitam de meia-sombra. Disponível nas larguras de 2m, 3m e 4m."
}
```

**Hiperparâmetros QLoRA:**

| Parâmetro                     | Valor                                      |
| ----------------------------- | ------------------------------------------ |
| `r` (rank)                    | 16                                         |
| `lora_alpha`                  | 32                                         |
| `lora_dropout`                | 0.05                                       |
| `target_modules`              | `["q_proj", "v_proj", "k_proj", "o_proj"]` |
| `learning_rate`               | 2e-4                                       |
| `num_train_epochs`            | 3                                          |
| `per_device_train_batch_size` | 2                                          |
| `gradient_accumulation_steps` | 4                                          |
| `max_seq_length`              | 512                                        |
| `warmup_ratio`                | 0.03                                       |

**Bibliotecas:** `peft`, `trl` (SFTTrainer), `bitsandbytes`, `datasets`

### 4.6 FastAPI (Backend)

**Arquivo:** `src/api/main.py`

**Endpoints:**

| Método | Rota                       | Descrição                        |
| ------ | -------------------------- | -------------------------------- |
| POST   | `/api/v1/chat`             | Envia pergunta, retorna resposta |
| POST   | `/api/v1/chat/stream`      | Streaming de resposta (SSE)      |
| GET    | `/api/v1/health`           | Health check                     |
| POST   | `/api/v1/documents/ingest` | Trigger ingestão de novos docs   |
| GET    | `/api/v1/documents/list`   | Lista documentos indexados       |

**Schema de Request/Response:**

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    pii_detected: bool
    processing_time_ms: float
```

### 4.7 Streamlit (Frontend)

**Arquivo:** `frontend/app.py`

**Features obrigatórias:**
- Chat interface com histórico de mensagens
- Indicador de "digitando..." durante inferência
- Sidebar com:
  - Upload de novos documentos
  - Slider de temperatura
  - Toggle de PII masking (on/off)
  - Lista de fontes usadas na resposta
- Badge visual quando PII é detectado
- Tempo de resposta exibido

---

## 5. Variáveis de Ambiente

```env
# .env.example
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=sales-copilot

HF_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
HF_TOKEN=your_token_here

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

API_HOST=0.0.0.0
API_PORT=8000

ENABLE_PII_MASKING=true
LOG_LEVEL=INFO
```

---

## 6. Requirements

```txt
# Core
langchain>=0.3.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
langchain-pinecone>=0.2.0
pinecone-client>=5.0.0

# LLM & Quantization
transformers>=4.45.0
torch>=2.1.0
bitsandbytes>=0.43.0
accelerate>=0.34.0

# Fine-tuning
peft>=0.13.0
trl>=0.12.0
datasets>=3.0.0

# PII Security
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
spacy>=3.7.0

# API & Frontend
fastapi>=0.115.0
uvicorn>=0.32.0
streamlit>=1.39.0

# Utils
python-dotenv>=1.0.0
pypdf>=5.0.0
sentence-transformers>=3.0.0
```

---

## 7. Checklist de Execução

### Fase 1 — Setup (Dia 1)
- [ ] Criar repositório GitHub com `.gitignore` Python
- [ ] Setup do ambiente virtual (`python -m venv .venv`)
- [ ] Instalar dependências
- [ ] Criar conta Pinecone (free tier)
- [ ] Criar conta Hugging Face e gerar token
- [ ] Baixar modelo base quantizado

### Fase 2 — RAG Pipeline (Dia 2–3)
- [ ] Implementar `loader.py` — carregar PDFs
- [ ] Implementar `chunker.py` — text splitting
- [ ] Implementar `embedder.py` — gerar embeddings
- [ ] Testar upsert no Pinecone
- [ ] Implementar `retriever.py` — busca vetorial
- [ ] Implementar `chain.py` — RAG chain completa
- [ ] Testar end-to-end: pergunta → contexto → resposta

### Fase 3 — Segurança PII (Dia 4)
- [ ] Implementar `pii_masker.py` com Presidio
- [ ] Adicionar regras para CPF, CNPJ, telefone, email
- [ ] Implementar `pii_unmasker.py`
- [ ] Escrever testes unitários
- [ ] Integrar no pipeline RAG

### Fase 4 — Quantização + Fine-Tuning (Dia 5–6)
- [ ] Implementar `quantize.py` com BitsAndBytes config
- [ ] Testar carregamento do modelo em 4-bit
- [ ] Monitorar uso de VRAM (deve ser < 3.5GB)
- [ ] Criar `dataset.jsonl` com 50+ exemplos
- [ ] Implementar `qlora_train.py`
- [ ] Executar treinamento (esperar ~1-2h na GTX 1650)
- [ ] Implementar `merge_adapter.py`
- [ ] Comparar respostas antes/depois do fine-tuning

### Fase 5 — API + Frontend (Dia 7–8)
- [ ] Implementar FastAPI com todos os endpoints
- [ ] Testar com Swagger UI (`/docs`)
- [ ] Implementar Streamlit chat interface
- [ ] Conectar frontend → API
- [ ] Testar fluxo completo

### Fase 6 — Polish + Deploy (Dia 9–10)
- [ ] Escrever README.md profissional
- [ ] Criar diagrama de arquitetura
- [ ] Gravar vídeo demonstrativo
- [ ] Escrever testes automatizados
- [ ] Revisar código e documentação
- [ ] Push final para GitHub

---

## 8. README.md — Template

O README deve conter:

1. **Header** com badges (Python, License, Stars)
2. **Descrição** de 2 linhas sobre o projeto
3. **Screenshot/GIF** da interface funcionando
4. **Diagrama de Arquitetura** (imagem)
5. **Features** — lista com emojis
6. **Tech Stack** — tabela
7. **Quick Start** — setup em 5 comandos
8. **Uso da API** — exemplos curl
9. **Fine-Tuning** — como treinar com seus dados
10. **Métricas** — VRAM usage, latência, tokens/s
11. **Roadmap** — próximas features
12. **License** — MIT

---

## 9. Métricas de Sucesso

| Métrica                    | Meta                                 |
| -------------------------- | ------------------------------------ |
| VRAM Usage (inferência)    | < 3.5 GB                             |
| Latência média (resposta)  | < 5 segundos                         |
| PII Detection Rate         | > 95%                                |
| RAG Relevance (manual)     | > 80% respostas com contexto correto |
| Tokens/segundo             | > 10 tok/s                           |
| Tempo total de fine-tuning | < 2 horas                            |

---

## 10. Regras de Qualidade de Código

- **Type hints** em todas as funções
- **Docstrings** em todas as classes e funções públicas
- **Logging** estruturado com `logging` (não `print`)
- **Config centralizada** em `src/config.py` via Pydantic Settings
- **Error handling** com exceções customizadas
- **Testes** com `pytest` (cobertura mínima: 70%)
- **Linting** com `ruff`
- **Formatação** com `black`
- **Commits** seguindo Conventional Commits (`feat:`, `fix:`, `docs:`)

---

> **Este documento é um template genérico.** Para adaptá-lo ao seu domínio, substitua os exemplos de dataset, ajuste os prompts em `src/rag/prompts.py` e insira seus documentos em `data/raw/`.
