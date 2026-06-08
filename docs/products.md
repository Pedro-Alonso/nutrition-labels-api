# Base Comunitária de Produtos

## Visão Geral

O módulo `products` implementa uma base de dados colaborativa de produtos
alimentícios identificados por código de barras. Usuários autenticados podem
cadastrar e atualizar tabelas nutricionais e listas de ingredientes. O endpoint
de análise clínica DM é público (sem autenticação).

---

## Modelos de Dados

### `Product`

Chave primária natural: `barcode` (código de barras como string).

```
Product
  barcode (PK)       — código de barras
  name               — nome do produto
  brand              — marca
  created_by_user_id — FK → users.id (SET NULL ao deletar usuário)
  created_at
  updated_at
  │
  ├─ nutritional_table (1:1, cascade all/delete-orphan)
  └─ ingredient_list  (1:1, cascade all/delete-orphan)
```

### `NutritionalTable`

Estrutura flexível via JSONB:

```json
{
  "portion_description": "Porção de 200ml (1 copo)",
  "columns": ["Quantidade por porção", "%VD(*)"],
  "rows": [
    { "nutrient": "Valor Energético", "values": ["84 kcal", "4%"] },
    { "nutrient": "Carboidratos",     "values": ["21 g",    "7%"] },
    { "nutrient": "Açúcares Totais",  "values": ["21 g",    "**"] }
  ]
}
```

- `columns`: lista de cabeçalhos (geralmente "Quantidade por porção" e "%VD(*)")
- `rows`: lista de `{ nutrient: string, values: string[] }` — uma linha por nutriente

### `IngredientList`

```json
{
  "items": ["água carbonatada", "açúcar", "extrato de noz de cola", "caramelo"]
}
```

Lista de strings — um item por ingrediente, na ordem em que aparecem no rótulo.

---

## Endpoints

Prefixo: `/api/v1/products`

| Método | Path | Auth | Descrição |
|---|---|---|---|
| GET | `/{barcode}` | Não | Retorna produto completo com análise clínica |
| POST | `/{barcode}` | JWT | Cria produto |
| PUT | `/{barcode}` | JWT | Atualiza produto (patch semântico) |
| POST | `/{barcode}/ocr` | JWT | Preview OCR sem persistir |
| GET | `/{barcode}/analysis` | Opcional (soft auth) | Análise clínica dos ingredientes + resumo LLM |

### Semântica de `PUT` (patch)

Campos ausentes no body mantêm o valor atual do banco:

```json
// Body com apenas brand → apenas brand é atualizado; name permanece
{ "brand": "Nova Marca" }
```

Para apagar `nutritional_table` ou `ingredient_list`, envie o campo com
valor `null` explícito. Campos omitidos não são tocados.

---

## Fluxo de Cadastro via OCR

O fluxo recomendado para o app mobile:

```
1. Ler código de barras da embalagem
2. GET /products/{barcode}
   → 200: produto já existe, mostrar dados
   → 404: cadastrar

3. (Se 404) Fotografar tabela nutricional e/ou ingredientes
4. POST /products/{barcode}/ocr
      image_nutrition + image_ingredients
   → OcrPreviewResponse com dados estruturados

5. Usuário revisa/corrige os dados no app

6. POST /products/{barcode}
   → cria produto com dados revisados

7. (Atualização futura)
   PUT /products/{barcode}
   → campos alterados pelo usuário
```

---

## `AnalysisService.read_outcome()`

O endpoint `/ocr` usa `AnalysisService.read_outcome()` — variante de
`analyze()` que retorna o `ReadOutcome` completo em vez do dict serializado,
permitindo acesso ao `ingredient_report` para extrair `tokens_found`.

```python
outcome = await asyncio.to_thread(
    ocr_service.read_outcome,
    nutrition_bytes,
    "table",   # category_override
)
```

O processamento é executado em thread pool (`asyncio.to_thread`) para não
bloquear o event loop durante o OCR.

---

## Parse de Tabela Nutricional

`product_service.parse_postprocessed_to_nutritional_table(text)` converte o
texto pós-processado do preset vencedor em `NutritionalTableData`:

- Detecta linhas com `\t` (saída de `CellBasedPipeline`) como estrutura tabular
- Extrai cabeçalho (primeira linha) como `columns`
- Cada linha subsequente vira `NutritionalRowData(nutrient, values[])`
- Fallback: texto linear sem `\t` é tratado heuristicamente

---

## Análise Clínica Integrada

`GET /products/{barcode}` e `GET /products/{barcode}/analysis` retornam análise
via `IngredientAnalyzer` (se disponível):

```python
analyzer = reader.ingredient_analyzer  # None se ontologia ausente
report = analyzer.analyze(" ".join(items), image_name=barcode)
```

O `IngredientAnalyzer` é o mesmo usado no fluxo de análise de rótulos — lê
os itens da `IngredientList` como texto OCR e aplica o pipeline
tokenizer → Viterbi → OntologyMatcher.

Se o `IngredientAnalyzer` não estiver disponível (ontologia ausente no
container), `GET /{barcode}/analysis` retorna HTTP 503 e `analysis` em
`GET /{barcode}` retorna `null`.

### Pipeline de análise em `GET /{barcode}/analysis` (com Groq)

Quando `GROQ_API_KEY` está configurado, o endpoint executa três etapas em sequência:

```
ingredient_list.items
    │
    ├─ 1. clean_ingredients_text (llm_service.py)
    │       Groq llama-3.3-70b-versatile, temperature=0
    │       Remove: alegações de marketing, dados corporativos,
    │               instruções de conservação, advertências alergênicas
    │       Fallback silencioso → itens originais em caso de erro
    │
    ├─ 2. _compute_analysis (service.py)
    │       IngredientAnalyzer sobre itens limpos
    │       → IngredientAnalysisSchema
    │
    └─ 3. generate_summary (llm_service.py)
            Groq llama-3.3-70b-versatile, temperature=0
            Personalizado por language_level e diabetes_type do usuário
            Regras: máx 3 frases, apenas dados do JSON, sem especulação
            Fallback silencioso → natural_language_summary = null
```

### Soft auth em `GET /{barcode}/analysis`

O endpoint não exige autenticação, mas **lê o token Bearer se presente**:

```python
# bearer válido → personaliza o resumo LLM com perfil do usuário
# bearer ausente/inválido → resumo usa prompts defaults neutros
```

O token inválido (expirado, assinatura errada) é silenciosamente ignorado —
não retorna 401. Isso garante que clientes não autenticados nunca recebam erro
por enviar um token desatualizado.

---

## Referências de Código

| Arquivo | Conteúdo |
|---|---|
| `app/products/models.py` | `Product`, `NutritionalTable`, `IngredientList` |
| `app/products/schemas.py` | `ProductResponse`, `ProductCreateRequest`, `ProductUpdateRequest`, `OcrPreviewResponse`, `NutritionalTableData`, `IngredientsData` |
| `app/products/router.py` | Rotas GET/POST/PUT + `/ocr` + `/analysis` (com soft auth e Groq) |
| `app/products/service.py` | `get_by_barcode`, `create_product`, `update_product`, `build_product_response`, `parse_postprocessed_to_nutritional_table`, `_compute_analysis` |
| `app/products/llm_service.py` | `clean_ingredients_text`, `generate_summary` — integração Groq |
| `alembic/versions/c3d4e5f6a7b8_add_products.py` | Migration que criou as três tabelas de produtos |
| `alembic/versions/d1e2f3a4b5c6_add_language_level_diabetes_type_to_users.py` | Migration que adicionou `language_level` e `diabetes_type` em `users` |
