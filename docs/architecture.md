# Arquitetura

## Visão Geral

O `nutrition-labels-api` é um serviço REST em FastAPI que expõe o motor OCR local
(herdado do monolito `teste-pytesseract`) como API HTTP para consumo por um
app mobile voltado a pacientes diabéticos. O sistema recebe fotos de rótulos
alimentícios, executa OCR + análise clínica simbólica e persiste resultados por
usuário.

**Contexto acadêmico:** componente de TCC (UNESP) — Projeto Científico II.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Runtime | Python 3.11 |
| Framework HTTP | FastAPI ≥ 0.110 + Uvicorn ≥ 0.29 |
| Banco de dados | PostgreSQL 16 |
| ORM | SQLAlchemy 2.0 async + asyncpg |
| Migrations | Alembic ≥ 1.13 |
| Auth | python-jose (JWT) + passlib/bcrypt |
| OCR local | Tesseract (pytesseract) + OpenCV |
| OCR nuvem | Google Cloud Vision API |
| Containerização | Docker + Docker Compose |
| Rate limiting | slowapi |

---

## Estrutura de Pacotes

```
nutrition-labels-api/
├── app/                          # Camada HTTP (FastAPI)
│   ├── main.py                   # Entrypoint: lifespan, routers, middlewares
│   ├── core/
│   │   ├── config.py             # Settings (pydantic-settings, lê .env)
│   │   ├── database.py           # Engine async, session factory, Base
│   │   ├── dependencies.py       # get_current_user_id (JWT → user_id)
│   │   ├── security.py           # hash/verify password, create/verify JWT
│   │   ├── limiter.py            # Rate limiter (slowapi)
│   │   └── middleware.py         # LoggingMiddleware (X-Request-ID, latência)
│   ├── auth/                     # /api/v1/auth — register, login, refresh, logout
│   ├── users/                    # /api/v1/users — perfil, histórico de scans
│   ├── analysis/                 # /api/v1/analyze, /api/v1/presets
│   └── products/                 # /api/v1/products — base comunitária de produtos
│
├── ocr_engine/                   # Motor OCR (cópia do monolito, sem alterações)
│   ├── __init__.py               # build_reader() — único ponto de entrada
│   ├── config/                   # app.json, routing.json, presets/, ontologia, wordlist
│   ├── nutrition/                # NutritionReader, FormatDetector, PresetRepository
│   │   └── pipelines/            # LinearPipeline, CellBasedPipeline, CloudVisionPipeline
│   ├── ocr/                      # OcrService, QualityEvaluator, postprocessing
│   │   └── cloud_vision/         # GcvClient, GcvCache, RateLimiter, auth
│   ├── imaging/                  # OPERATION_REGISTRY, morphology, roi, io
│   ├── ingredients/              # tokenizer → segmenter Viterbi → OntologyMatcher
│   └── audit/                    # AuditRecorder (grava PNGs e JSONs de auditoria)
│
├── alembic/                      # Migrations (env.py + versions/)
├── tests/
│   ├── conftest.py               # Fixtures globais: client, db_session, test_user, auth_token
│   ├── api/                      # Testes de integração HTTP
│   ├── unit/                     # Testes unitários de services
│   └── ocr_engine/gcv/           # Testes unitários + property do módulo GCV
│
├── Dockerfile
├── docker-compose.yml            # api + db (dev)
├── docker-compose.test.yml       # api-test + db-test (CI)
├── requirements.txt
└── requirements-dev.txt          # pytest, hypothesis (opt-in)
```

---

## Decisões Arquiteturais

### NutritionReader como singleton

`NutritionReader` é construído **uma vez** no `lifespan` do FastAPI e armazenado
em `app.state.reader`. A inicialização é cara: carrega presets JSON,
`FormatDetector`, `RoiDetector`, `IngredientAnalyzer` (ontologia + wordlist) e
`NutritionTextPostProcessor`.

```python
@asynccontextmanager
async def lifespan(app):
    from ocr_engine import build_reader
    app.state.reader = build_reader()   # única chamada; bloqueia até finalizar
    yield
```

**Consequência:** `uvicorn --workers 1` em produção. Para escalar, use múltiplos
containers (não múltiplos workers no mesmo processo), com PostgreSQL como
estado compartilhado.

### ocr_engine/ e sys.path

`ocr_engine/__init__.py` insere `ocr_engine/` no início de `sys.path` antes de
importar qualquer módulo do motor. Isso permite que imports absolutos do motor
(ex.: `from nutrition.reader import ...`) resolvam corretamente dentro do
diretório copiado, sem alterar nenhum arquivo do monolito.

### AnalysisService usa arquivo temporário

`NutritionReader.read()` espera `pathlib.Path`. O service grava bytes em
`tempfile.NamedTemporaryFile(delete=False)`, processa e remove com
`unlink(missing_ok=True)` no `finally` — garantindo remoção mesmo em exceção.

### Scan persiste result_json JSONB

Toda a resposta do OCR (incluindo `ingredient_analysis`) é armazenada como JSONB
na coluna `result_json` para consultas futuras sem reprocessamento.

### Engine lazy no banco

O engine SQLAlchemy e a session factory são criados na **primeira chamada**, não
na importação do módulo. Isso permite que testes sobrescrevam `DATABASE_URL` via
variável de ambiente antes de qualquer engine ser instanciado.

### NullPool em testes

Testes usam `NullPool` para evitar reúso de conexões com transações pendentes
entre fixtures. Cada fixture `db_session` abre conexão limpa.

### Limpeza de tokens revogados

Uma `asyncio.Task` iniciada no `lifespan` limpa tokens expirados da tabela
`revoked_tokens` a cada hora, evitando crescimento ilimitado.

---

## Fluxo de uma Requisição de Análise

```
POST /api/v1/analyze (multipart)
    │
    ├─ Validação de tamanho e content-type
    ├─ Cálculo SHA-256 dos bytes originais  →  image_hash
    ├─ AnalysisService.analyze()
    │      NamedTemporaryFile (bytes → disco)
    │      NutritionReader.read(tmp_path, options)
    │          FormatDetector → categoria (table|text|ingredient)
    │          RoiDetector → crop
    │          Cascata de presets → melhor OCR
    │          QualityEvaluator → score + passed
    │          NutritionTextPostProcessor → texto normalizado
    │          [IngredientAnalyzer] → análise clínica DM
    │      outcome_to_dict(ReadOutcome) → dict
    │      unlink(tmp_path)              (finally)
    ├─ Persist Scan (JSONB) no PostgreSQL
    └─ AnalyzeResponse (Pydantic v2)
```

---

## Middlewares e Infraestrutura Transversal

| Componente | Função |
|---|---|
| `LoggingMiddleware` | Loga método, path, status, latência e `X-Request-ID` em cada request |
| `CORSMiddleware` | Configura CORS; origens lidas de `ALLOWED_ORIGINS` (.env) |
| `slowapi` rate limiter | Limita requisições por IP; configurado em `app/core/limiter.py` |
| `RateLimitExceeded` handler | Retorna HTTP 429 formatado em JSON |

---

## Relação com o Monolito

O diretório `ocr_engine/` é uma cópia fiel do monolito `teste-pytesseract`.
Nenhum arquivo interno foi modificado. A integração é feita exclusivamente via
`ocr_engine/__init__.py` (manipulação de `sys.path` + fachada `build_reader()`).

Quando o monolito evoluir (novos presets, nova ontologia, modelo SSD re-treinado),
basta substituir os arquivos dentro de `ocr_engine/` — a camada HTTP não muda.

---

## Diagrama de Dependências de Módulos

```
app.main
  ├── app.core.config         (Settings, get_settings)
  ├── app.core.database       (get_engine, get_session_factory, get_db)
  ├── app.core.security       (JWT, bcrypt)
  ├── app.core.dependencies   (get_current_user_id)
  ├── app.core.limiter        (slowapi limiter)
  ├── app.core.middleware     (LoggingMiddleware)
  ├── app.auth.router
  ├── app.users.router
  ├── app.analysis.router
  │     └── app.analysis.service
  │           └── ocr_engine (NutritionReader singleton via app.state.reader)
  └── app.products.router
        └── app.products.service
              └── ocr_engine (IngredientAnalyzer via reader.ingredient_analyzer)
```
