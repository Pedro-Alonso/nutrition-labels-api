# Referência da API

Base URL (desenvolvimento): `http://localhost:8000`

Documentação interativa (Swagger UI): `http://localhost:8000/docs`

---

## Índice

- [Health](#health)
- [Auth](#auth)
- [Users](#users)
- [Analysis](#analysis)
- [Products](#products)
- [Códigos de Erro Comuns](#códigos-de-erro-comuns)

---

## Health

### `GET /api/v1/health`

Verifica se o serviço está respondendo. Não requer autenticação.

**Resposta 200**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "dependencies": {
    "database": "not_checked",
    "tesseract": "not_checked",
    "gcv_configured": false
  }
}
```

`gcv_configured` é `true` quando `GOOGLE_APPLICATION_CREDENTIALS` está definido
no ambiente.

---

## Auth

### `POST /api/v1/auth/register`

Cria uma nova conta de usuário.

**Body (JSON)**

```json
{
  "email": "usuario@example.com",
  "password": "minimo8chars",
  "display_name": "Nome Opcional"
}
```

| Campo | Tipo | Obrigatório | Validação |
|---|---|---|---|
| `email` | string | sim | — |
| `password` | string | sim | mínimo 8 caracteres |
| `display_name` | string | não | — |

**Resposta 201**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "usuario@example.com",
  "display_name": "Nome Opcional",
  "created_at": "2026-06-06T12:00:00Z"
}
```

**Erros**

| Código | Motivo |
|---|---|
| 409 | E-mail já cadastrado |
| 422 | Senha com menos de 8 caracteres |

---

### `POST /api/v1/auth/login`

Autentica um usuário e retorna tokens JWT.

**Body (JSON)**

```json
{
  "email": "usuario@example.com",
  "password": "minimo8chars"
}
```

**Resposta 200**

```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer"
}
```

`access_token` expira em 15 minutos. `refresh_token` expira em 30 dias.

**Erros**

| Código | Motivo |
|---|---|
| 401 | E-mail ou senha inválidos |

---

### `POST /api/v1/auth/refresh`

Emite um novo `access_token` a partir de um `refresh_token` válido.

**Body (JSON)**

```json
{
  "refresh_token": "<jwt>"
}
```

**Resposta 200**

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

**Erros**

| Código | Motivo |
|---|---|
| 401 | Refresh token inválido, expirado ou revogado |

---

### `POST /api/v1/auth/logout`

Revoga o `refresh_token` atual, impedindo renovação futura.

**Header:** `Authorization: Bearer <access_token>`

**Body (JSON)**

```json
{
  "refresh_token": "<jwt>"
}
```

**Resposta 204** (sem corpo)

**Erros**

| Código | Motivo |
|---|---|
| 401 | Access token inválido |

---

## Users

Todos os endpoints exigem `Authorization: Bearer <access_token>`.

### `GET /api/v1/users/me`

Retorna o perfil do usuário autenticado.

**Resposta 200**

```json
{
  "id": "550e8400-...",
  "email": "usuario@example.com",
  "display_name": "Nome",
  "language_level": "padrão",
  "diabetes_type": "type2",
  "created_at": "2026-06-06T12:00:00Z"
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `language_level` | `"simples"` \| `"padrão"` \| `"técnico"` \| `null` | Nível de linguagem do resumo clínico gerado por LLM |
| `diabetes_type` | `"type1"` \| `"type2"` \| `"dmg"` \| `null` | Tipo de diabetes — personaliza o foco do resumo clínico |

---

### `PUT /api/v1/users/me`

Atualiza o perfil do usuário autenticado.

**Body (JSON)**

```json
{
  "display_name": "Novo Nome",
  "language_level": "simples",
  "diabetes_type": "type1"
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `display_name` | string \| null | Nome de exibição (opcional) |
| `language_level` | `"simples"` \| `"padrão"` \| `"técnico"` \| null | Nível de linguagem para resumos LLM |
| `diabetes_type` | `"type1"` \| `"type2"` \| `"dmg"` \| null | Tipo de diabetes |

**Resposta 200** — mesmo schema de `GET /me`

---

### `GET /api/v1/users/me/scans`

Retorna o histórico de análises do usuário (paginado).

**Query params**

| Param | Tipo | Default | Descrição |
|---|---|---|---|
| `page` | int | `1` | Número da página |
| `per_page` | int | `20` | Itens por página |

**Resposta 200**

```json
{
  "items": [
    {
      "id": "uuid",
      "created_at": "2026-06-06T12:00:00Z",
      "detected_format": "table",
      "passed": true,
      "winning_preset": "gcv_doc_text",
      "risco_global": null
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 20
}
```

---

## Analysis

### `POST /api/v1/analyze`

**Header:** `Authorization: Bearer <access_token>`

Analisa uma foto de rótulo alimentício via OCR. Recebe `multipart/form-data`.

**Form fields**

| Campo | Tipo | Obrigatório | Default | Descrição |
|---|---|---|---|---|
| `file` | file | sim | — | Imagem do rótulo (JPEG, PNG, WEBP, BMP, TIFF) |
| `category_override` | string | não | `null` | Força categoria: `"table"`, `"text"` ou `"ingredient"` |
| `roi_enabled` | bool | não | `true` | Aplica detecção de ROI antes do OCR |
| `stop_on_first_pass` | bool | não | `true` | Para na primeira cascata aprovada |
| `postprocess` | bool | não | `true` | Aplica pós-processador nutricional ao texto |

**Resposta 200**

```json
{
  "scan_id": "uuid",
  "detected_format": {
    "category": "table",
    "score": 0.032,
    "grid_density": 0.032,
    "reasoning": "densidade de grade acima do limiar"
  },
  "winning_preset": "gcv_doc_text",
  "winning_attempt_index": 1,
  "passed": true,
  "final_ocr_text": "INFORMAÇÃO NUTRICIONAL\n...",
  "final_postprocessed_text": "Valor Energético 75 kcal ...",
  "attempts": [
    {
      "attempt_index": 1,
      "preset": "gcv_doc_text",
      "passed": true,
      "score": 0.87,
      "mean_confidence": 94.2,
      "text_length": 312,
      "keyword_hits": 5
    }
  ],
  "ingredient_analysis": null
}
```

`ingredient_analysis` é preenchido quando `detected_format.category == "ingredient"`:

```json
{
  "risco_global": "ALTO",
  "ingredientes_identificados": [
    {
      "nome_lido": "açúcar",
      "classe": "acucar_simples",
      "risco": "ALTO",
      "alerta": "Eleva glicemia rapidamente. IG ≈ 65.",
      "indice_glicemico": 65,
      "nota_clinica": null
    }
  ],
  "nao_identificados": ["vitamina c"],
  "high_risk_ingredients": ["açúcar"],
  "safe_sweeteners": []
}
```

**Erros**

| Código | Motivo |
|---|---|
| 400 | Imagem inválida/corrompida ou formato não suportado |
| 401 | Token ausente ou inválido |
| 413 | Arquivo maior que `MAX_UPLOAD_SIZE_MB` (default 10 MB) |

---

### `GET /api/v1/presets`

Lista todos os presets OCR disponíveis por categoria. Não requer autenticação.

**Resposta 200**

```json
{
  "table": [
    {
      "name": "gcv_doc_text",
      "description": "Google Cloud Vision DOCUMENT_TEXT_DETECTION.",
      "kind": "cloud_vision",
      "priority": 5
    }
  ],
  "text": [...],
  "ingredients": [...]
}
```

---

## Products

Base comunitária de produtos identificados por código de barras. Permite cadastro
colaborativo de tabelas nutricionais e listas de ingredientes.

### `GET /api/v1/products/{barcode}`

Retorna dados completos de um produto. Não requer autenticação.

**Resposta 200**

```json
{
  "barcode": "7891234567890",
  "name": "Coca-Cola 350ml",
  "brand": "Coca-Cola",
  "nutritional_table": {
    "portion_description": "Porção de 200ml",
    "columns": ["Quantidade por porção", "%VD(*)"],
    "rows": [
      { "nutrient": "Valor Energético", "values": ["84 kcal", "4%"] }
    ]
  },
  "ingredients": {
    "items": ["água carbonatada", "açúcar", "extrato de noz de cola"]
  },
  "analysis": {
    "risco_global": "ALTO",
    "ingredientes_identificados": [...],
    "nao_identificados": [],
    "high_risk_ingredients": ["açúcar"],
    "safe_sweeteners": [],
    "natural_language_summary": null
  },
  "created_at": "2026-06-06T12:00:00Z",
  "updated_at": "2026-06-06T12:00:00Z"
}
```

`analysis` é `null` quando o produto não tem lista de ingredientes ou quando o
`IngredientAnalyzer` não está disponível (ontologia ausente).

**Erros**

| Código | Motivo |
|---|---|
| 404 | Produto não encontrado |

---

### `POST /api/v1/products/{barcode}`

Cria um novo produto. **Requer autenticação.**

**Body (JSON)**

```json
{
  "name": "Coca-Cola 350ml",
  "brand": "Coca-Cola",
  "nutritional_table": {
    "portion_description": "Porção de 200ml",
    "columns": ["Quantidade por porção", "%VD(*)"],
    "rows": [
      { "nutrient": "Valor Energético", "values": ["84 kcal", "4%"] }
    ]
  },
  "ingredients": {
    "items": ["água carbonatada", "açúcar", "extrato de noz de cola"]
  }
}
```

Todos os campos são opcionais. O produto pode ser criado com dados parciais e
completado depois via `PUT`.

**Resposta 201** — mesmo schema de `GET /{barcode}`

**Erros**

| Código | Motivo |
|---|---|
| 409 | Código de barras já cadastrado |

---

### `PUT /api/v1/products/{barcode}`

Atualiza um produto existente (semântica de patch: campos ausentes mantêm o
valor atual). **Requer autenticação.**

**Body (JSON)** — mesmo schema de `POST`, todos os campos opcionais.

**Resposta 200** — mesmo schema de `GET /{barcode}`

**Erros**

| Código | Motivo |
|---|---|
| 404 | Produto não encontrado |

---

### `POST /api/v1/products/{barcode}/ocr`

Processa imagens via OCR e retorna preview estruturado **sem persistir nada**.
Útil para o app mobile montar o formulário de cadastro antes de confirmar.
**Requer autenticação.**

**Form fields** (multipart)

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `image_nutrition` | file | não | Foto da tabela nutricional |
| `image_ingredients` | file | não | Foto da lista de ingredientes |

Pelo menos um dos dois deve ser enviado.

**Resposta 200**

```json
{
  "barcode": "7891234567890",
  "nutritional_table": {
    "portion_description": "Porção de 200ml",
    "columns": ["Quantidade por porção", "%VD(*)"],
    "rows": [...]
  },
  "ingredients": {
    "items": ["água carbonatada", "açúcar"]
  }
}
```

Campos `null` quando a imagem correspondente não foi enviada.

---

### `GET /api/v1/products/{barcode}/analysis`

Retorna análise clínica DM dos ingredientes do produto. Não requer autenticação.

**Autenticação opcional (soft auth):** se um `Bearer` token válido for enviado, o
campo `natural_language_summary` é personalizado com `language_level` e
`diabetes_type` do perfil do usuário. Sem token, o resumo usa os defaults neutros.

**Fluxo interno** (quando `GROQ_API_KEY` está configurado):

1. **Limpeza LLM** — `llama-3.3-70b-versatile` remove alegações de marketing
   ("zero açúcar", "fonte de fibras", CNPJ, etc.) e ruído OCR da lista de
   ingredientes antes da análise ontológica.
2. **Análise ontológica** — `IngredientAnalyzer` roda sobre a lista limpa.
3. **Resumo LLM** — modelo gera até 3 frases em PT-BR, baseadas exclusivamente
   nos dados da análise (sem adicionar fatos externos). Personalizado pelo perfil
   do usuário autenticado.

**Resposta 200**

```json
{
  "risco_global": "ALTO",
  "ingredientes_identificados": [
    {
      "nome_lido": "açúcar",
      "classe": "acucar_simples",
      "risco": "ALTO",
      "alerta": "Eleva glicemia rapidamente. IG ≈ 65.",
      "indice_glicemico": 65,
      "nota_clinica": null
    }
  ],
  "nao_identificados": ["vitamina c"],
  "high_risk_ingredients": ["açúcar"],
  "safe_sweeteners": [],
  "natural_language_summary": "Este produto contém açúcar (IG 65), classificado como ALTO risco para pacientes diabéticos. Recomenda-se evitar o consumo ou verificar a porção com o nutricionista."
}
```

`natural_language_summary` é `null` quando `GROQ_API_KEY` não está configurado
ou quando o Groq retorna erro (o endpoint **não falha** nesses casos).

**Erros**

| Código | Motivo |
|---|---|
| 404 | Produto não encontrado ou sem lista de ingredientes |
| 503 | `IngredientAnalyzer` indisponível (ontologia ausente) |

---

### `GET /api/v1/products/{barcode}/summary`

Retorna resumo personalizado enxuto para um produto. Não requer autenticação.

**Autenticação opcional (soft auth):** se um `Bearer` token válido for enviado, o
resumo é personalizado com `diabetes_type` e `language_level` do perfil do usuário.
Sem token (ou quando a LLM falha para o perfil personalizado), retorna o resumo
genérico cacheado `(barcode, null, null)`.

**Resposta 200**

```json
{
  "summary": "Este produto contém açúcar, classificado como ALTO risco...",
  "diabetes_type": "DM2",
  "language_level": "leigo",
  "risco_global": "ALTO"
}
```

Todos os campos podem ser `null`: `summary` é `null` quando não há ingredientes,
o `IngredientAnalyzer` não está disponível, ou a LLM não está configurada e não
existe cache. `diabetes_type` e `language_level` refletem a personalização
utilizada (ambos `null` para resumo genérico/anônimo).

**Erros**

| Código | Motivo |
|---|---|
| 404 | Produto não encontrado |

---

## Códigos de Erro Comuns

| Código | Significado | Quando ocorre |
|---|---|---|
| 400 | Bad Request | Imagem inválida, formato não suportado |
| 401 | Unauthorized | Token ausente, expirado, inválido ou com `type` errado |
| 404 | Not Found | Recurso não existe |
| 409 | Conflict | E-mail ou código de barras já cadastrado |
| 413 | Payload Too Large | Arquivo acima do limite |
| 422 | Unprocessable Entity | Validação Pydantic falhou (ex.: senha < 8 chars) |
| 429 | Too Many Requests | Rate limit excedido |
| 503 | Service Unavailable | Dependência indisponível (ex.: ontologia ausente) |

**Formato padrão de erro**

```json
{ "detail": "mensagem descritiva" }
```

Para erros 422 (validação Pydantic):

```json
{
  "detail": [
    {
      "loc": ["body", "password"],
      "msg": "A senha deve ter pelo menos 8 caracteres.",
      "type": "value_error"
    }
  ]
}
```

---

## Autenticação via Bearer Token

Endpoints protegidos exigem o header:

```
Authorization: Bearer <access_token>
```

O `access_token` é obtido via `POST /api/v1/auth/login` e expira em 15 minutos.
Para renová-lo sem re-login, use `POST /api/v1/auth/refresh` com o `refresh_token`
(válido por 30 dias).

Detalhes do fluxo JWT em [authentication.md](authentication.md).
