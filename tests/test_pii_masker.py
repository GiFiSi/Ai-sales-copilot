"""Testes unitários para o PII Masker.

Testa detecção e mascaramento de:
- CPF (com e sem pontuação)
- CNPJ (com e sem pontuação)
- E-mail
- Telefone (com e sem DDD)
- CEP
- Múltiplas entidades no mesmo texto
"""

import pytest

from src.security.pii_masker import PIIMasker


@pytest.fixture
def masker():
    """Cria instância do PIIMasker para os testes."""
    return PIIMasker()


# ============================================================
# CPF
# ============================================================

class TestCPFMasking:
    """Testes de mascaramento de CPF."""

    def test_cpf_com_pontuacao(self, masker):
        text = "O CPF do cliente é 123.456.789-00."
        masked, mapping = masker.mask(text)
        assert "123.456.789-00" not in masked
        assert "[CPF_MASCARADO]" in masked
        assert len(mapping) >= 1

    def test_cpf_sem_pontuacao(self, masker):
        text = "CPF: 12345678900"
        masked, mapping = masker.mask(text)
        assert "12345678900" not in masked
        assert len(mapping) >= 1

    def test_multiplos_cpfs(self, masker):
        text = "CPFs: 123.456.789-00 e 987.654.321-00"
        masked, mapping = masker.mask(text)
        assert "123.456.789-00" not in masked
        assert "987.654.321-00" not in masked
        assert len(mapping) >= 2


# ============================================================
# CNPJ
# ============================================================

class TestCNPJMasking:
    """Testes de mascaramento de CNPJ."""

    def test_cnpj_com_pontuacao(self, masker):
        text = "CNPJ da empresa: 12.345.678/0001-99"
        masked, mapping = masker.mask(text)
        assert "12.345.678/0001-99" not in masked
        assert "[CNPJ_MASCARADO]" in masked

    def test_cnpj_sem_pontuacao(self, masker):
        text = "CNPJ: 12345678000199"
        masked, mapping = masker.mask(text)
        assert "12345678000199" not in masked


# ============================================================
# Email
# ============================================================

class TestEmailMasking:
    """Testes de mascaramento de e-mail."""

    def test_email_simples(self, masker):
        text = "Contato: joao@empresa.com.br"
        masked, mapping = masker.mask(text)
        assert "joao@empresa.com.br" not in masked
        assert "[EMAIL_MASCARADO]" in masked

    def test_email_complexo(self, masker):
        text = "Email: jose.silva+vendas@gmail.com"
        masked, mapping = masker.mask(text)
        assert "jose.silva+vendas@gmail.com" not in masked


# ============================================================
# Telefone
# ============================================================

class TestTelefoneMasking:
    """Testes de mascaramento de telefone."""

    def test_telefone_com_ddd(self, masker):
        text = "Ligar para (11) 98765-4321"
        masked, mapping = masker.mask(text)
        assert "98765-4321" not in masked
        assert "[TELEFONE_MASCARADO]" in masked

    def test_telefone_sem_ddd(self, masker):
        text = "Telefone: 98765-4321"
        masked, mapping = masker.mask(text)
        assert "98765-4321" not in masked


# ============================================================
# CEP
# ============================================================

class TestCEPMasking:
    """Testes de mascaramento de CEP."""

    def test_cep_com_hifen(self, masker):
        text = "CEP: 01310-100"
        masked, mapping = masker.mask(text)
        assert "01310-100" not in masked
        assert "[CEP_MASCARADO]" in masked


# ============================================================
# Combinados
# ============================================================

class TestMultipleEntities:
    """Testes com múltiplas entidades PII."""

    def test_texto_com_multiplas_entidades(self, masker):
        text = (
            "Cliente João Silva, CPF 123.456.789-00, "
            "email joao@email.com, telefone (11) 98765-4321."
        )
        masked, mapping = masker.mask(text)

        assert "123.456.789-00" not in masked
        assert "joao@email.com" not in masked
        assert "98765-4321" not in masked
        assert len(mapping) >= 3

    def test_texto_sem_pii(self, masker):
        text = "Qual o preço da tela modelo X na largura de 3 metros?"
        masked, mapping = masker.mask(text)
        assert masked == text
        assert len(mapping) == 0

    def test_texto_vazio(self, masker):
        masked, mapping = masker.mask("")
        assert masked == ""
        assert len(mapping) == 0


# ============================================================
# Unmasking
# ============================================================

class TestUnmasking:
    """Testes do ciclo completo mask → unmask."""

    def test_roundtrip(self, masker):
        from src.security.pii_unmasker import PIIUnmasker

        unmasker = PIIUnmasker()
        original = "Cliente CPF 123.456.789-00 email teste@test.com"

        masked, mapping = masker.mask(original)
        assert "123.456.789-00" not in masked
        assert "teste@test.com" not in masked

        # Simula resposta do LLM contendo placeholders
        llm_response = f"O orçamento para {masked.split('Cliente ')[1].split(' email')[0]} foi gerado."

        # Na prática, se o LLM repete os placeholders, o unmask restaura
        for placeholder, value in mapping.items():
            if placeholder in llm_response:
                restored = unmasker.unmask(llm_response, mapping)
                assert value in restored


# ============================================================
# Detect
# ============================================================

class TestDetect:
    """Testes da função de detecção sem mascaramento."""

    def test_detect_identifica_entidades(self, masker):
        text = "CPF: 123.456.789-00, email: x@y.com"
        entities = masker.detect(text)
        assert len(entities) >= 2
        types = [e["entity_type"] for e in entities]
        assert "CPF" in types
