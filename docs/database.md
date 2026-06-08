# Banco de Dados

## Stack

- **PostgreSQL 16** (produção e testes)
- **SQLAlchemy 2.0** async (`create_async_engine`, `AsyncSession`)
- **asyncpg** como driver
- **Alembic** para migrations

---

## Engine e Session Factory

`app/core/database.py` usa globals com guarda `None` (lazy init):

```python
_engine = None
_session_factory = None

def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return _engine
```

**Por que lazy:** permite que testes sobrescrevam `DATABASE_URL` via env var
antes de qualquer engine ser instanciado, independentemente da ordem de import.

**Parâmetros relevantes**

| Parâmetro | Valor | Motivo |
|---|---|---|
| `pool_pre_ping` | `True` | Detecta conexões PostgreSQL mortas em containers |
| `expire_on_commit` | `False` | Objetos ORM acessíveis após commit sem refresh extra |

---

## Dependency `get_db`

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session
```

O `async with` gerencia commit/rollback/close automaticamente. Não feche
a sessão manualmente dentro do endpoint.

---

## Modelos ORM

### Convenções

- Herdam de `Base` (única `DeclarativeBase` em `app/core/database.py`)
- Chaves primárias: `str` UUID gerado em Python (`default=lambda: str(uuid.uuid4())`)
- Timestamps: `DateTime(timezone=True)` → `TIMESTAMPTZ` no PostgreSQL
- Imports: `from __future__ import annotations` no topo de cada arquivo

### `User` (`app/users/models.py`)

| Coluna | Tipo SQL | Constraints | Descrição |
|---|---|---|---|
| `id` | VARCHAR | PK | UUID v4 string |
| `email` | VARCHAR | UNIQUE, INDEX | Normalizado em lowercase |
| `password_hash` | VARCHAR | NOT NULL | Hash bcrypt |
| `display_name` | VARCHAR | NULL | Nome opcional |
| `language_level` | VARCHAR | NULL | Nível de linguagem LLM: `"simples"` \| `"padrão"` \| `"técnico"` |
| `diabetes_type` | VARCHAR | NULL | Tipo de diabetes: `"type1"` \| `"type2"` |
| `created_at` | TIMESTAMPTZ | NOT NULL | UTC de criação |
| `updated_at` | TIMESTAMPTZ | NOT NULL | Atualizado via `onupdate` |

`language_level` e `diabetes_type` personalizam o campo `natural_language_summary`
gerado pelo Groq em `GET /products/{barcode}/analysis` quando o usuário está
autenticado.

Relacionamentos: `scans` (lazy=`selectin`).

### `Scan` (`app/analysis/models.py`)

| Coluna | Tipo SQL | Constraints | Descrição |
|---|---|---|---|
| `id` | VARCHAR | PK | UUID v4 string |
| `user_id` | VARCHAR | FK → users.id, INDEX | Dono do scan |
| `image_hash` | VARCHAR(64) | NOT NULL | SHA-256 hex dos bytes originais |
| `detected_format` | VARCHAR | NULL | `"table"` / `"text"` / `"ingredient"` |
| `winning_preset` | VARCHAR | NULL | Nome do preset vencedor |
| `passed` | BOOLEAN | NOT NULL, default False | Atingiu limiares de qualidade |
| `risco_global` | VARCHAR | NULL | Pior risco clínico; null se não é ingrediente |
| `result_json` | JSONB | NOT NULL | Payload completo do OCR |
| `created_at` | TIMESTAMPTZ | NOT NULL | Imutável após criação |

`Scan` **não possui `updated_at`** — registros são imutáveis. Re-análise cria
novo `Scan`.

### `RevokedToken` (`app/auth/models.py`)

| Coluna | Tipo SQL | Constraints | Descrição |
|---|---|---|---|
| `id` | VARCHAR | PK | UUID v4 string |
| `token` | TEXT | UNIQUE, INDEX | JWT completo do refresh token |
| `expires_at` | TIMESTAMPTZ | INDEX | Expiração original do token |
| `revoked_at` | TIMESTAMPTZ | NOT NULL | Momento da revogação |

Tokens expirados são limpos automaticamente a cada hora (ver `app/main.py`).

### `Product` (`app/products/models.py`)

| Coluna | Tipo SQL | Constraints | Descrição |
|---|---|---|---|
| `barcode` | VARCHAR | PK | Código de barras (chave natural) |
| `name` | VARCHAR | NULL | Nome do produto |
| `brand` | VARCHAR | NULL | Marca |
| `created_by_user_id` | VARCHAR | FK → users.id (SET NULL), INDEX | Criador |
| `created_at` | TIMESTAMPTZ | NOT NULL | UTC de criação |
| `updated_at` | TIMESTAMPTZ | NOT NULL | Atualizado via `onupdate` |

Relacionamentos: `nutritional_table` (1:1, lazy=`selectin`, cascade=`all, delete-orphan`)
e `ingredient_list` (1:1, mesmo padrão).

### `NutritionalTable` (`app/products/models.py`)

| Coluna | Tipo SQL | Constraints | Descrição |
|---|---|---|---|
| `id` | VARCHAR | PK | UUID v4 string |
| `product_barcode` | VARCHAR | FK → products.barcode (CASCADE), UNIQUE | 1:1 com Product |
| `portion_description` | VARCHAR | NULL | Ex.: "Porção de 200ml" |
| `columns` | JSONB | NOT NULL | Lista de cabeçalhos de coluna |
| `rows` | JSONB | NOT NULL | Lista de linhas `{nutrient, values[]}` |
| `updated_at` | TIMESTAMPTZ | NOT NULL | — |
| `updated_by_user_id` | VARCHAR | FK → users.id (SET NULL), NULL | Último editor |

### `IngredientList` (`app/products/models.py`)

| Coluna | Tipo SQL | Constraints | Descrição |
|---|---|---|---|
| `id` | VARCHAR | PK | UUID v4 string |
| `product_barcode` | VARCHAR | FK → products.barcode (CASCADE), UNIQUE | 1:1 com Product |
| `items` | JSONB | NOT NULL | Lista de strings (ingredientes) |
| `updated_at` | TIMESTAMPTZ | NOT NULL | — |
| `updated_by_user_id` | VARCHAR | FK → users.id (SET NULL), NULL | Último editor |

---

## Diagrama de Relacionamentos

```
users ──────────────────────────────────────────────────────────┐
  │  id (PK)                                                    │
  │                                                             │
  ├─── scans (1:N)                                              │
  │      user_id  FK → users.id                                 │
  │      result_json  JSONB                                     │
  │                                                             │
  ├─── products (1:N, created_by)                               │
  │      barcode (PK)                                           │
  │      created_by_user_id  FK → users.id (SET NULL)           │
  │      │                                                      │
  │      ├─── nutritional_tables (1:1)                          │
  │      │      product_barcode  FK → products.barcode CASCADE  │
  │      │      updated_by_user_id  FK → users.id (SET NULL) ──┘
  │      │                                                      │
  │      └─── ingredient_lists (1:1)                            │
  │             product_barcode  FK → products.barcode CASCADE  │
  │             updated_by_user_id  FK → users.id (SET NULL) ──┘
  │
  └─── revoked_tokens (sem relação direta; user_id não rastreado)
```

---

## Migrations (Alembic)

### Versões existentes

| Revisão | Descrição |
|---|---|
| `f372a4df148e` | Initial schema: tabelas `users` e `scans` |
| `a1b2c3d4e5f6` | Adiciona `revoked_tokens` |
| `c3d4e5f6a7b8` | Adiciona `products`, `nutritional_tables`, `ingredient_lists` |
| `d1e2f3a4b5c6` | Adiciona `language_level` e `diabetes_type` em `users` |

### Comandos essenciais

```bash
# Aplicar todas as migrations pendentes
alembic upgrade head

# Reverter uma migration
alembic downgrade -1

# Reverter tudo
alembic downgrade base

# Ver versão atual no banco
alembic current

# Ver histórico completo
alembic history --verbose

# Criar nova migration após alterar model ORM
alembic revision --autogenerate -m "descricao_da_mudanca"
```

### Regra de ouro: nunca editar migrations já aplicadas

Alembic rastreia a revisão aplicada na tabela `alembic_version`. Editar
`upgrade()` de uma migration já aplicada não tem efeito no banco — Alembic
não re-executa. **Sempre crie uma nova revisão.**

### Como `alembic/env.py` resolve `DATABASE_URL`

```python
DATABASE_URL = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
config.set_main_option("sqlalchemy.url", DATABASE_URL)
```

Variável de ambiente tem prioridade sobre `alembic.ini`.

### Adicionando um novo modelo

1. Criar `app/<dominio>/models.py` herdando de `Base`.
2. Importar em `alembic/env.py` com `# noqa: F401`.
3. `alembic revision --autogenerate -m "add_<nome>"`.
4. **Revisar** o arquivo gerado antes de aplicar.
5. `alembic upgrade head`.
6. Adicionar a nova tabela no `TRUNCATE` de `clean_db` em `tests/conftest.py`.

---

## Por que UUIDs como strings (não auto-increment)

- Chaves sequenciais permitem enumerar registros por força bruta.
- UUIDs v4 são opacos e não enumeráveis.
- Usar `String` (não tipo `UUID` nativo do SQLAlchemy) elimina fricção com
  asyncpg e mantém IDs como `str` em Python — compatível com JWT e schemas
  Pydantic sem conversão.

---

## Por que `DateTime(timezone=True)`

Mapeia para `TIMESTAMPTZ` no PostgreSQL, que armazena o instante em UTC.
Sem `timezone=True`, o banco armazena timestamp naïve — bugs em ambientes
com fusos diferentes.

```python
created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
)
```

**Não use `datetime.utcnow()`** — está deprecated e retorna datetime naïve.

---

## `Scan.result_json` — JSONB

Armazena o payload completo da análise OCR, incluindo `ingredient_analysis`.
Permite consultas futuras sobre o banco sem reprocessar imagens:

```sql
SELECT * FROM scans WHERE result_json->>'winning_preset' = 'gcv_doc_text';
SELECT * FROM scans WHERE result_json->'ingredient_analysis'->>'risco_global' = 'ALTO';
```
