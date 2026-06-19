# nutrition-labels-api

Backend REST para análise de rótulos alimentícios voltado a pacientes diabéticos.
Expõe a API consumida pelo app mobile, integrando OCR (Tesseract / Google Cloud Vision),
processamento de imagem (OpenCV) e análise clínica simbólica via ontologia DM.

Stack: Python 3.11 · FastAPI · PostgreSQL · SQLAlchemy (async) · Docker

---

## Setup Rápido

```bash
# 1. Copiar variáveis de ambiente
cp .env.example .env
# editar .env com seus valores reais

# 2. (Opcional) Adicionar service account GCV
mkdir -p secrets
cp /caminho/para/service_account.json secrets/service_account.json

# 3. Subir containers
docker compose up --build

# 4. Verificar saúde da API
curl http://localhost:8000/api/v1/health
```

Swagger UI (documentação interativa): http://localhost:8000/docs

---

## Documentação Técnica

| Documento | Conteúdo |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Arquitetura, stack, decisões de design, fluxo de dados |
| [docs/api-reference.md](docs/api-reference.md) | Referência completa de todos os endpoints |
| [docs/authentication.md](docs/authentication.md) | JWT, bcrypt, fluxo de tokens, revogação |
| [docs/database.md](docs/database.md) | Modelos ORM, migrations, relacionamentos, PostgreSQL |
| [docs/ocr-engine.md](docs/ocr-engine.md) | Motor OCR, presets, pipelines, GCV, análise clínica DM |
| [docs/products.md](docs/products.md) | Base comunitária de produtos (cadastro colaborativo) |
| [docs/testing.md](docs/testing.md) | Fixtures, como rodar testes, property-based testing GCV |
| [docs/deployment.md](docs/deployment.md) | Docker, variáveis de ambiente, produção, migrations |

---

## Endpoints Resumidos

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/api/v1/health` | Não | Health check |
| GET | `/api/v1/presets` | Não | Lista presets OCR por categoria |
| POST | `/api/v1/analyze` | JWT | Analisa foto de rótulo alimentício |
| POST | `/api/v1/auth/register` | Não | Cria usuário |
| POST | `/api/v1/auth/login` | Não | Login (retorna JWT) |
| POST | `/api/v1/auth/refresh` | Não | Renova access token |
| POST | `/api/v1/auth/logout` | JWT | Revoga refresh token |
| GET | `/api/v1/users/me` | JWT | Perfil do usuário autenticado |
| PUT | `/api/v1/users/me` | JWT | Atualiza perfil |
| GET | `/api/v1/users/me/scans` | JWT | Histórico de análises |
| GET | `/api/v1/products/{barcode}` | Não | Produto com análise clínica |
| POST | `/api/v1/products/{barcode}` | JWT | Cria produto |
| PUT | `/api/v1/products/{barcode}` | JWT | Atualiza produto |
| POST | `/api/v1/products/{barcode}/ocr` | JWT | Preview OCR (sem persistir) |
| GET | `/api/v1/products/{barcode}/analysis` | Não | Análise clínica dos ingredientes |
| GET | `/api/v1/products/{barcode}/summary` | Não | Resumo personalizado enxuto |

Ver [docs/api-reference.md](docs/api-reference.md) para schemas completos.

---

## Comandos Docker

```bash
docker compose up --build                                          # subir tudo
docker compose up db                                               # só o banco
docker compose exec api alembic upgrade head                       # migrations
docker compose exec api alembic revision --autogenerate -m "desc" # nova migration
docker compose exec db psql -U rotulos_user -d rotulos_db          # acesso ao banco
docker compose logs -f api                                         # logs em tempo real
docker compose down                                                # parar (preserva volume)
docker compose down -v                                             # reset completo
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit  # testes CI
```

---

## Convenção de Commits e Branches

### Branch naming

```
(feat|fix|release)/<número-2-dígitos>/<nome-descritivo>

Exemplos:
  feat/00/setup-inicial
  feat/04/tests
  fix/02/null-recorder-interface
```

### Commit message

```
(feat|fix|release|refactor|docs|chore|merge): <mensagem descritiva>

Exemplos:
  feat: add health check endpoint
  fix: correct import path in null_recorder
  docs: add README with commit conventions
```

Formatos inválidos são **rejeitados pelos git hooks** em `.git/hooks/`.

---

## Variáveis de Ambiente Obrigatórias em Produção

| Variável | Descrição |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@host:5432/db` |
| `SECRET_KEY` | Gerar com `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ALLOWED_ORIGINS` | Origins do app mobile (substitua `"*"` em produção) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path do service account GCV (opcional) |

Ver `.env.example` para a lista completa e [docs/deployment.md](docs/deployment.md)
para instruções detalhadas.
