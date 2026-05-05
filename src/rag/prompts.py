"""Templates de prompt para o pipeline RAG.

Contém o system prompt e o template de RAG
formatados para modelos de chat (Qwen/Llama).
"""

# ============================================================
# System Prompt — Define o comportamento do assistente
# ============================================================

SYSTEM_PROMPT = """Você é um assistente de IA corporativo especializado em responder perguntas \
sobre produtos, serviços e políticas internas da organização.

## Regras:
1. Responda APENAS com base no contexto fornecido abaixo. Não invente informações.
2. Se o contexto não contiver informação suficiente para responder, diga claramente: \
"Não encontrei essa informação na base de conhecimento."
3. Cite as fontes utilizadas ao final da resposta (nome do documento e página).
4. Responda em português brasileiro, de forma clara e profissional.
5. Se a pergunta envolver cálculos, mostre o passo a passo.
6. Nunca revele dados pessoais, CPFs, CNPJs ou informações sensíveis de clientes."""


# ============================================================
# RAG Prompt Template — Combina contexto + pergunta
# ============================================================

RAG_PROMPT_TEMPLATE = """## Contexto (documentos internos relevantes):

{context}

---

## Pergunta do usuário:

{question}

---

## Instruções:
Com base EXCLUSIVAMENTE no contexto acima, responda à pergunta do usuário. \
Seja preciso, objetivo e cite as fontes."""


# ============================================================
# Chat Format — Para modelos que usam chat template
# ============================================================

def format_chat_messages(
    system_prompt: str,
    context: str,
    question: str,
) -> list[dict[str, str]]:
    """Formata mensagens no formato de chat para o modelo.

    Compatível com Qwen2.5, Llama-3.2 e outros modelos
    que usam o formato de chat do Hugging Face.

    Args:
        system_prompt: Prompt de sistema com regras do assistente.
        context: Contexto recuperado do Pinecone.
        question: Pergunta do usuário.

    Returns:
        Lista de dicts no formato [{"role": "...", "content": "..."}].
    """
    user_message = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def format_simple_prompt(context: str, question: str) -> str:
    """Formata prompt simples (sem chat template).

    Para modelos que não usam formato de chat, gera um
    prompt único concatenando system + contexto + pergunta.

    Args:
        context: Contexto recuperado do Pinecone.
        question: Pergunta do usuário.

    Returns:
        String com o prompt completo.
    """
    user_content = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    return f"{SYSTEM_PROMPT}\n\n{user_content}\n\nResposta:"
