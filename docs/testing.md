# Testes

## Stack

| Biblioteca | Função |
|---|---|
| `pytest >= 7.4` | Runner principal |
| `pytest-asyncio >= 0.23` | Suporte a `async def` em testes |
| `httpx >= 0.27` | Cliente HTTP assíncrono para testes de API |
| `hypothesis >= 6.90` | Property-based testing (módulo GCV) |

Dependências de teste estão em `requirements-dev.txt` (não em `requirements.txt`).

---

## Configuração

`pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
```

`asyncio_mode = auto` trata **todas** as funções `async def` como corrotinas
automaticamente, sem exigir `@pytest.mark.asyncio` em cada uma.

---

## Como rodar

```bash
# Suite completa (requer PostgreSQL ativo + .env com DATABASE_URL)
python -m pytest tests/ -v --tb=short

# Apenas testes de API
python -m pytest tests/api/ -v

# Apenas testes unitários
python -m pytest tests/unit/ -v

# Apenas testes do módulo GCV
python -m pytest tests/ocr_engine/gcv/ -v

# Apenas property tests GCV (sem exemplos concretos)
python -m pytest tests/ocr_engine/gcv/ -v -k "not example"

# Via Docker (CI isolado)
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

---

## Estrutura de Arquivos

```
tests/
├── conftest.py                        # Fixtures globais
├── fixtures/
│   └── images/
│       ├── coca_tabela.jpg            # Tabela nutricional (test_analyze.py)
│       └── coca_ingredientes.jpg      # Lista de ingredientes
├── api/
│   ├── test_auth.py                   # /register, /login, /refresh, /logout
│   ├── test_auth_completeness.py      # Casos de borda de auth
│   ├── test_analyze.py                # /analyze, /presets
│   ├── test_analyze_edge_cases.py     # Casos de borda (arquivo vazio, tipo inválido, etc.)
│   ├── test_products.py               # CRUD + OCR preview + análise DM
│   └── test_scans.py                  # GET /users/me/scans
├── unit/
│   ├── test_analysis_service.py       # AnalysisService (mocks do reader)
│   └── test_users_service.py          # CRUD de usuários
└── ocr_engine/
    └── gcv/
        ├── conftest.py                # tmp_project_root, gcv_app_config_default, fake_sa
        ├── strategies.py             # Geradores Hypothesis
        ├── fixtures/
        │   ├── images/tiny_label.png
        │   ├── responses/             # JSONs sintéticos de resposta GCV
        │   └── service_accounts/fake_sa.json
        └── test_*.py                  # Property tests P1–P21 + exemplos
```

---

## Fixtures Globais (`tests/conftest.py`)

### Cadeia de dependências

```
session_client   ← AsyncClient (app ASGI, NutritionReader singleton)
    │
    └─ db_session    ← AsyncSession com NullPool por teste
           │
           └─ client     ← session_client + override de get_db
                  │
                  └─ test_user    ← POST /register via client
                         │
                         └─ auth_token   ← POST /login via client
```

### `session_client`

```python
@pytest_asyncio.fixture
async def session_client() -> AsyncClient:
    if not hasattr(app.state, "reader") or app.state.reader is None:
        from ocr_engine import build_reader
        app.state.reader = build_reader()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
```

`ASGITransport` não executa o lifespan ASGI. A guarda `hasattr` constrói o
reader **uma vez por sessão** de teste.

### `db_session` (NullPool)

```python
_test_engine = create_async_engine(settings.database_url, poolclass=NullPool)

@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    factory = async_sessionmaker(bind=_test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
```

**Por que NullPool:** evita o erro de transação aninhada do asyncpg. Cada
fixture cria conexão limpa sem pooling.

### `client`

```python
@pytest_asyncio.fixture
async def client(session_client, db_session) -> AsyncClient:
    async def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    yield session_client
    app.dependency_overrides.pop(get_db, None)   # cleanup obrigatório
```

### `clean_db` (autouse)

```python
@pytest_asyncio.fixture(autouse=True)
async def clean_db() -> None:
    async with _test_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE scans, users RESTART IDENTITY CASCADE")
        )
```

Trunca antes de **cada teste** automaticamente. `RESTART IDENTITY CASCADE`
garante isolamento total sem recriar schema.

**Ao adicionar nova tabela:** inclua-a no TRUNCATE.

---

## Como escrever um novo teste de endpoint

```python
# tests/api/test_meu_endpoint.py
from __future__ import annotations
from httpx import AsyncClient

async def test_endpoint_sucesso(client: AsyncClient, auth_token: str) -> None:
    resp = await client.get(
        "/api/v1/meu-path",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    assert "campo_esperado" in resp.json()

async def test_endpoint_sem_auth(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/meu-path")
    assert resp.status_code == 401
```

**Regras:**
- `async def` obrigatório — `asyncio_mode = auto` dispensa `@pytest.mark.asyncio`
- `auth_token` implica `test_user` e `client` — pytest resolve a cadeia
- `db_session` só quando necessário para queries diretas no banco
- `clean_db` é autouse — não declare explicitamente

---

## Testes GCV — Property-Based Testing

Os testes do módulo GCV usam **Hypothesis** para verificar invariantes:

```python
from hypothesis import given, settings
from tests.ocr_engine.gcv.strategies import gcv_response_dict

@given(gcv_response_dict())
@settings(max_examples=100, deadline=None)
def test_mean_confidence_bounded(response: dict) -> None:
    # Feature: gcv-ocr-preset, Property 7
    parsed = parse_response(response, "DOCUMENT_TEXT_DETECTION")
    assert 0.0 <= parsed.mean_confidence <= 100.0
```

### Geradores disponíveis (`strategies.py`)

| Gerador | Produz |
|---|---|
| `gcv_response_dict()` | dict simulando `AnnotateImageResponse` com confs `float ∈ [0,1]` |
| `gcv_response_with_non_numeric_conf()` | idem com confidence não-numérico |
| `bcp47_hints()` | `tuple[str,...]` com 0–4 language hints |
| `image_arrays()` | `np.ndarray` BGR `uint8`, shape `(H,W,3)`, H/W ∈ [16,64] |
| `kind_strings_invalid()` | strings ∉ `ALLOWED_KINDS` |
| `feature_strings_invalid()` | strings ∉ `ALLOWED_FEATURES` |
| `error_class_subsets()` | subconjuntos de `ERROR_PRECEDENCE` |
| `rate_limiter_event_sequences()` | sequências de eventos para rate limiter |
| `cache_states()` | estados do `cache_dir` com entradas válidas e corrompidas |

### Propriedades verificadas (P1–P21)

| # | Componente | Invariante |
|---|---|---|
| P1 | `CloudVisionPipeline.execute` | Sempre retorna `PipelineResult` válido; stages[0]="input", stages[-1]="output" |
| P2 | `CloudVisionPipeline.execute` | `apply_operation` nunca chamado; `ignored_steps_count==len(steps)` |
| P3 | `Preset.category` | GCV preset em `presets/<dir>/` → `category==dir` |
| P4 | `GcvPresetOptions.from_dict` | Defaults corretos para `feature`, `language_hints`, `model` |
| P5 | `GcvCache` | Round-trip com filtro `(feature, hints)` ordem-sensível |
| P6 | `CloudVisionPipeline` (skip) | Falha → `ocr_text=""`, `mean_confidence=0.0`, erro classificado |
| P7 | `parse_response` | `mean_confidence ∈ [0.0, 100.0]` para qualquer resposta |
| P8 | `parse_response` | Campo de texto correto por feature |
| P9 | `CloudVisionPipeline` + `AuditRecorder` | Auditoria idêntica entre cache hit e miss |
| P10 | `RateLimiter` | `≤ N` chamadas em qualquer janela de 60s |
| P11 | `GcvAppConfig.from_dict` | `max_requests_per_minute` inválido → `None` + warning |
| P12 | `GcvClient.fetch` | Falha de credencial aplica `on_failure` corretamente |
| P13 | `PresetRepository._parse` | `ValueError` para `kind` fora de `ALLOWED_KINDS` |
| P14 | `CloudVisionPipeline.execute` | `gcv.feature` inválido → `error=="invalid_feature"` sem chamar API |
| P15 | Todo o fluxo | Conteúdo do Service Account não aparece em nenhum artefato |
| P16 | `GcvCache` (desabilitado) | `cache_enabled=False` → zero I/O no `cache_dir` |
| P17 | `AuditRecorder` | `clean_previous=True` não toca `extractions/.gcv_cache/` |
| P18 | `GcvCache.get` | Corrupção de uma entrada não invalida as demais |
| P19 | `GcvCache` | Entradas não expiram por timestamp |
| P20 | `encode_png` | PNG round-trip preserva pixels |
| P21 | `NutritionReader` | `postprocess=False` → `final_postprocessed.txt` idêntico a `final.txt` |

### Injeção para testes sem rede real

```python
from unittest.mock import MagicMock

mock_client = MagicMock()
mock_client.annotate_image.return_value = {...}  # resposta sintética

gcv_client = GcvClient.build(config, project_root, api_client=mock_client)
```

Nenhum teste faz chamada real à GCV.

---

## CI via Docker

`docker-compose.test.yml` define dois serviços:

- **`db-test`:** PostgreSQL 16 com healthcheck `pg_isready`
- **`api-test`:** stage `test` do Dockerfile; executa
  `alembic upgrade head && python -m pytest tests/ -v --tb=short`

O serviço `api-test` aguarda `db-test` estar saudável antes de iniciar
(`depends_on.condition: service_healthy`). `--abort-on-container-exit` garante
que o exit code do pytest chega ao CI.

```bash
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```
