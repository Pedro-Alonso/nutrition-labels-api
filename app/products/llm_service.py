from __future__ import annotations

import json
import logging

from groq import AsyncGroq

from app.analysis.schemas import IngredientAnalysisSchema
from app.products.schemas import NutritionalTableData

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

_TABLE_SYSTEM = """\
Você recebe o texto bruto extraído por OCR de uma tabela nutricional de rótulo \
alimentício brasileiro. Sua ÚNICA tarefa: estruturar essas informações em JSON.

Responda APENAS com um objeto JSON neste formato:
{
  "portion_description": "<descrição da porção, ex.: 'Porção de 30g (2 colheres de sopa)', ou null>",
  "columns": ["<rótulo da 1ª coluna de valores>", "<rótulo da 2ª coluna, se houver>"],
  "rows": [
    {"nutrient": "<nome do nutriente>", "values": ["<valor1>", "<valor2>"]}
  ]
}

REGRAS OBRIGATÓRIAS:
- Cole a unidade (g, mg, kcal) junto ao número, sem espaço (ex.: "15g", "120kcal").
- A coluna de %VD deve conter o número seguido de "%", ou "**" quando o rótulo não
  traz %VD para aquele nutriente.
- Copie os números EXATAMENTE como aparecem no texto (verbatim) — não arredonde,
  não converta unidades, não calcule valores.
- NUNCA invente nutrientes, valores ou colunas que não estejam no texto.
- Se o texto não contiver uma tabela nutricional reconhecível, responda com
  {"portion_description": null, "columns": [], "rows": []}.
- NUNCA explique, adicione markdown ou texto fora do JSON."""

_REFUSAL_MARKERS = (
    "não há",
    "nao ha",
    "nenhum ingrediente",
    "texto fornecido",
)


def _is_refusal(text: str) -> bool:
    """Detecta recusas/explicações da LLM em vez da lista de ingredientes esperada."""
    normalized = text.strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in _REFUSAL_MARKERS):
        return True
    # Resposta longa sem vírgula provavelmente é uma frase explicativa, não uma lista.
    return len(normalized) > 60 and "," not in normalized


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
8. Se o nome do produto for informado, cite-o naturalmente no resumo (ex.: "O produto X contém...").
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
        content = completion.choices[0].message.content or raw_text
        if _is_refusal(content):
            return ""
        return content
    except Exception:
        logger.exception("Falha na limpeza LLM de ingredientes — usando texto original")
        return raw_text


async def clean_nutritional_table(raw_text: str, api_key: str) -> NutritionalTableData | None:
    """Extrai a tabela nutricional estruturada via LLM. Retorna None em caso de erro,
    JSON inválido ou tabela sem linhas (OCR ilegível)."""
    try:
        client = AsyncGroq(api_key=api_key)
        completion = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _TABLE_SYSTEM},
                {"role": "user", "content": raw_text},
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            return None
        data = json.loads(content)
        table = NutritionalTableData(**data)
        if not table.rows:
            return None
        return table
    except Exception:
        logger.exception("Falha na extração LLM da tabela nutricional")
        return None


async def generate_summary(
    analysis: IngredientAnalysisSchema,
    api_key: str,
    language_level: str | None = None,
    diabetes_type: str | None = None,
    name: str | None = None,
    brand: str | None = None,
) -> str | None:
    """Gera resumo em linguagem natural. Retorna None em caso de erro."""
    try:
        system = _SUMMARY_SYSTEM_BASE.format(
            language_hint=_LANGUAGE_HINTS.get(language_level or "", ""),
            diabetes_hint=_DIABETES_HINTS.get(diabetes_type or "", ""),
        )
        client = AsyncGroq(api_key=api_key)
        product_label = " ".join(p for p in (name, brand) if p)
        analysis_json = analysis.model_dump_json(exclude={"natural_language_summary"})
        user_content = (
            f'Produto: "{product_label}"\n{analysis_json}' if product_label else analysis_json
        )
        completion = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return completion.choices[0].message.content
    except Exception:
        logger.exception("Falha ao gerar resumo LLM via Groq")
        return None
