"""AI Sales Copilot — Interface de Chat (Streamlit).

Frontend interativo com:
- Chat com histórico de mensagens
- Sidebar com configurações e upload
- Indicadores de PII e fontes
- Tempo de resposta
"""

import time

import httpx
import streamlit as st

# ============================================================
# Configuração da página
# ============================================================

st.set_page_config(
    page_title="AI Sales Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS Customizado
# ============================================================

st.markdown("""
<style>
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.9; font-size: 0.95rem; }

    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-msg {
        background: #f0f2f6;
        color: #1a1a2e;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
    }

    /* PII badge */
    .pii-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }

    /* Source card */
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }

    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 1.5rem; color: #667eea; }
    .metric-card p { margin: 0.2rem 0 0; font-size: 0.8rem; color: #666; }

    /* Sidebar */
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Estado da sessão
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "total_pii_detected" not in st.session_state:
    st.session_state.total_pii_detected = 0

# ============================================================
# Configuração da API
# ============================================================

API_BASE_URL = "http://localhost:8000/api/v1"


def send_message(
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_k: int = 5,
    namespace: str = "default",
) -> dict:
    """Envia mensagem para a API e retorna a resposta.

    Args:
        message: Pergunta do usuário.
        temperature: Temperatura de geração.
        max_tokens: Máximo de tokens.
        top_k: Número de chunks a buscar.
        namespace: Namespace do Pinecone.

    Returns:
        Dict com resposta da API.
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "namespace": namespace,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.ConnectError:
        return {
            "answer": "❌ **Erro de conexão:** Não foi possível conectar à API. "
                      "Verifique se o servidor está rodando (`uvicorn src.api.main:app`).",
            "sources": [],
            "pii_detected": False,
            "processing_time_ms": 0,
            "masked_entities": [],
        }
    except Exception as e:
        return {
            "answer": f"❌ **Erro:** {str(e)}",
            "sources": [],
            "pii_detected": False,
            "processing_time_ms": 0,
            "masked_entities": [],
        }


def check_api_health() -> dict:
    """Verifica o status da API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return response.json()
    except Exception:
        return {"status": "offline", "model_loaded": False, "pinecone_connected": False}


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Configurações")

    # Status da API
    st.markdown("### 📡 Status da API")
    health = check_api_health()
    status_color = "🟢" if health.get("status") == "healthy" else "🟡" if health.get("status") == "degraded" else "🔴"
    st.markdown(f"{status_color} **Status:** {health.get('status', 'offline')}")
    st.markdown(f"{'✅' if health.get('model_loaded') else '❌'} Modelo LLM")
    st.markdown(f"{'✅' if health.get('pinecone_connected') else '❌'} Pinecone")

    st.markdown("---")

    # Parâmetros de geração
    st.markdown("### 🎛️ Parâmetros")

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="0.0 = respostas mais determinísticas, 1.0 = mais criativas",
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2048,
        value=512,
        step=50,
        help="Tamanho máximo da resposta em tokens",
    )

    top_k = st.slider(
        "Top-K (chunks)",
        min_value=1,
        max_value=10,
        value=5,
        help="Número de trechos de documentos a recuperar",
    )

    namespace = st.text_input(
        "Namespace",
        value="default",
        help="Namespace do Pinecone (ex: produtos, fiscal)",
    )

    st.markdown("---")

    # PII Toggle
    st.markdown("### 🛡️ Segurança")
    pii_enabled = st.toggle("PII Masking", value=True, help="Mascarar dados sensíveis automaticamente")

    st.markdown("---")

    # Upload de documentos
    st.markdown("### 📄 Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Envie PDFs ou TXTs para indexar",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
    )

    if uploaded_files and st.button("📤 Indexar Documentos", type="primary"):
        with st.spinner("Indexando documentos..."):
            # Salvar arquivos e chamar API de ingestão
            st.info("⚠️ Para indexar, coloque os arquivos em `data/raw/` e use o script `ingest_documents.py`.")

    st.markdown("---")

    # Estatísticas da sessão
    st.markdown("### 📊 Sessão")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Perguntas", st.session_state.total_queries)
    with col2:
        st.metric("PII Detectado", st.session_state.total_pii_detected)

    # Limpar chat
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_pii_detected = 0
        st.rerun()

# ============================================================
# Área principal — Chat
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Sales Copilot</h1>
    <p>Assistente inteligente com RAG, PII Masking e LLM quantizado</p>
</div>
""", unsafe_allow_html=True)

# Histórico de mensagens
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

            # Mostra metadados (se disponíveis)
            metadata = msg.get("metadata", {})

            if metadata.get("pii_detected"):
                entities = metadata.get("masked_entities", [])
                st.markdown(
                    f'<span class="pii-badge">🛡️ PII DETECTADO: {", ".join(entities)}</span>',
                    unsafe_allow_html=True,
                )

            sources = metadata.get("sources", [])
            if sources:
                with st.expander(f"📚 Fontes ({len(sources)})"):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'📄 **{src["source_file"]}** (p.{src.get("page", "N/A")}) '
                            f'— relevância: {src.get("score", "N/A")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            if metadata.get("processing_time_ms"):
                st.caption(f"⏱️ {metadata['processing_time_ms']:.0f}ms")

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta sobre produtos, serviços ou políticas..."):
    # Mostra mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Gera resposta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Buscando informações e gerando resposta..."):
            response = send_message(
                message=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                namespace=namespace,
            )

        # Mostra resposta
        st.markdown(response["answer"])

        # PII badge
        if response.get("pii_detected"):
            entities = response.get("masked_entities", [])
            st.markdown(
                f'<span class="pii-badge">🛡️ PII DETECTADO: {", ".join(entities)}</span>',
                unsafe_allow_html=True,
            )
            st.session_state.total_pii_detected += 1

        # Fontes
        sources = response.get("sources", [])
        if sources:
            with st.expander(f"📚 Fontes ({len(sources)})"):
                for src in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 **{src["source_file"]}** (p.{src.get("page", "N/A")}) '
                        f'— relevância: {src.get("score", "N/A")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Tempo
        if response.get("processing_time_ms"):
            st.caption(f"⏱️ {response['processing_time_ms']:.0f}ms")

    # Salva no histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "metadata": {
            "sources": response.get("sources", []),
            "pii_detected": response.get("pii_detected", False),
            "masked_entities": response.get("masked_entities", []),
            "processing_time_ms": response.get("processing_time_ms", 0),
        },
    })

    st.session_state.total_queries += 1
