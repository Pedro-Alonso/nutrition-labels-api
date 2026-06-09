from __future__ import annotations

import logging

from groq import AsyncGroq

from app.analysis.schemas import IngredientAnalysisSchema

logger = logging.getLogger(__name__)

_CLEAN_SYSTEM = """\
Você recebe texto bruto extraído por OCR de um rótulo alimentício.
Sua ÚNICA tarefa: retornar exclusivamente os nomes de ingredientes listados.
REMOVER sem exceção:
- Alegações nutricionais: "zero açúcar", "sem gorduras trans", "fonte de fibras", "light", "diet", "zero", "reduzido em"
- Dados corporativos: CNPJ, endereço, telefone, distribuidor, fabricante, importador
- Instruções de conservação: "conservar em local fresco", "consumir preferencialmente até"
- Certificações: "ANVISA", "USDA Organic", "Kosher", "Halal"
- Advertências alergênicas: "contém glúten", "pode conter traços de"
- Percentuais e valores: "% VD", "porção de X g", "valor energético"
Retorne APENAS os ingredientes separados por vírgula.
Se um texto for genuinamente ambíguo (pode ser ingrediente), mantenha-o.
NUNCA explique, adicione texto ou invente ingredientes. Copie os nomes verbatim."""

_SUMMARY_SYSTEM_BASE = """\
Você é um nutricionista que reporta análise clínica de rótulos a pacientes diabéticos.
REGRAS OBRIGATÓRIAS — sem exceção:
1. Use APENAS as informações presentes no JSON de análise fornecido. Nunca adicione fatos externos.
2. Cite somente ingredientes presentes em "ingredientes_identificados". Não invente nem especule sobre outros.
3. Use os valores de "indice_glicemico" e "risco" exatamente como fornecidos — não arredonde nem estime.
4. Se "risco_global" for "NENHUM" ou "BAIXO", diga isso diretamente. Não invente riscos onde não há.
5. Se "ingredientes_identificados" estiver vazio, informe que nenhum ingrediente de risco foi identificado na análise.
6. Seja direto e objetivo. Proibido usar: "pode", "talvez", "possivelmente", "provavelmente". Use afirmações factuais.
7. Máximo 3 frases. Português do Brasil.
{language_hint}
{diabetes_hint}"""

_LANGUAGE_HINTS: dict[str, str] = {
    "simples": "Use linguagem simples e direta, sem termos técnicos ou siglas médicas.",
    "padrão": "Use linguagem clara com termos nutricionais básicos como IG e carboidratos.",
    "técnico": "Use terminologia técnica clínica e bioquímica.",
}

_DIABETES_HINTS: dict[str, str] = {
    "type1": (
        "O paciente tem Diabetes Tipo 1 (insulino-dependente). "
        "Priorize informações sobre contagem de carboidratos e impacto no bolus de insulina."
    ),
    "type2": (
        "O paciente tem Diabetes Tipo 2. "
        "Priorize índice glicêmico, carga glicêmica e padrão dietético geral."
    ),
    "dmg": (
        "A paciente tem Diabetes Mellitus Gestacional. "
        "Priorize controle glicêmico pós-prandial, limite de carboidratos por refeição "
        "e ingredientes contraindicados na gestação."
    ),
}


async def clean_ingredients_text(raw_text: str, api_key: str) -> str:
    """Remove alegações e ruído OCR do texto de ingredientes. Retorna original em caso de erro."""
    try:
        client = AsyncGroq(api_key=api_key)
        completion = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": _CLEAN_SYSTEM},
                {"role": "user", "content": raw_text},
            ],
        )
        return completion.choices[0].message.content or raw_text
    except Exception:
        logger.exception("Falha na limpeza LLM de ingredientes — usando texto original")
        return raw_text


async def generate_summary(
    analysis: IngredientAnalysisSchema,
    api_key: str,
    language_level: str | None = None,
    diabetes_type: str | None = None,
) -> str | None:
    """Gera resumo em linguagem natural. Retorna None em caso de erro."""
    try:
        system = _SUMMARY_SYSTEM_BASE.format(
            language_hint=_LANGUAGE_HINTS.get(language_level or "", ""),
            diabetes_hint=_DIABETES_HINTS.get(diabetes_type or "", ""),
        )
        client = AsyncGroq(api_key=api_key)
        completion = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": analysis.model_dump_json(
                    exclude={"natural_language_summary"}
                )},
            ],
        )
        return completion.choices[0].message.content
    except Exception:
        logger.exception("Falha ao gerar resumo LLM via Groq")
        return None
