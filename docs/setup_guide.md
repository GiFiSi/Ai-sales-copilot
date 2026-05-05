# AI Sales Copilot — Guia de Setup

Guia completo para configurar e executar o projeto do zero.

---

## Pré-requisitos

| Requisito | Versão | Como verificar |
|-----------|--------|----------------|
| Python | 3.11+ | `python --version` |
| pip | 23+ | `pip --version` |
| Git | 2.30+ | `git --version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| CUDA | 11.8+ | `nvcc --version` |

> **Nota:** CUDA e GPU são necessários apenas para inferência com quantização. O projeto funciona em CPU (mais lento).

---

## 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/ai-sales-copilot.git
cd ai-sales-copilot
```

## 2. Criar Ambiente Virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## 3. Instalar Dependências

```bash
# Dependências base
pip install -r requirements.txt

# PyTorch com CUDA (ajuste a versão do CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Modelo spaCy para PII em português
python -m spacy download pt_core_news_sm
```

## 4. Configurar Variáveis de Ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas chaves:

1. **Pinecone**: Crie conta em [pinecone.io](https://www.pinecone.io/) → copie a API key
2. **Hugging Face**: Crie conta em [huggingface.co](https://huggingface.co/) → gere token em Settings → Access Tokens

## 5. Ingerir Documentos

Coloque seus documentos em `data/raw/` (PDFs, TXTs, CSVs):

```bash
python scripts/ingest_documents.py
```

## 6. (Opcional) Fine-Tuning

Edite `data/training/dataset.jsonl` com seus exemplos e execute:

```bash
python scripts/train_model.py
```

## 7. Executar

```bash
# Tudo junto
python scripts/run_all.py

# Ou separadamente
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
streamlit run frontend/app.py
```

- **API**: http://localhost:8000/docs
- **Chat**: http://localhost:8501

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| `CUDA out of memory` | Reduza `max_tokens` ou use modelo menor |
| `Pinecone connection error` | Verifique API key no `.env` |
| `ModuleNotFoundError` | Rode `pip install -r requirements.txt` |
| `spacy model not found` | Rode `python -m spacy download pt_core_news_sm` |
