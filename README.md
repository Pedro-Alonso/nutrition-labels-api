# rotulos-backend

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

---

## Endpoints

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/api/v1/health` | Não | Health check |
| GET | `/api/v1/presets` | Não | Lista presets OCR por categoria |
| POST | `/api/v1/analyze` | JWT | Analisa foto de rótulo alimentício |
| POST | `/api/v1/auth/register` | Não | Cria usuário |
| POST | `/api/v1/auth/login` | Não | Login (retorna JWT) |
| POST | `/api/v1/auth/refresh` | Não | Renova access token |
| GET | `/api/v1/users/me` | JWT | Perfil do usuário autenticado |
| PUT | `/api/v1/users/me` | JWT | Atualiza perfil |
| GET | `/api/v1/users/me/scans` | JWT | Histórico de análises |

Documentação interativa: http://localhost:8000/docs (Swagger UI)

---

## Comandos Docker Úteis

```bash
# Subir tudo (com rebuild)
docker compose up --build

# Só o banco (para dev sem Docker da API)
docker compose up db

# Aplicar migrations manualmente
docker compose exec api alembic upgrade head

# Criar nova migration após alterar model ORM
docker compose exec api alembic revision --autogenerate -m "descricao"

# Acessar o banco
docker compose exec db psql -U rotulos_user -d rotulos_db

# Ver logs em tempo real
docker compose logs -f api

# Rebuild sem cache (quando requirements.txt mudar)
docker compose build --no-cache api

# Parar containers (preserva volume do banco)
docker compose down

# Reset completo (apaga banco)
docker compose down -v

# Rodar testes no container
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

---

## Estrutura do Projeto

```
rotulos-backend/
├── app/                    # Aplicação FastAPI
│   ├── main.py             # Ponto de entrada, lifespan, routers, CORS
│   ├── core/               # Infraestrutura horizontal
│   │   ├── config.py       # Pydantic Settings (lê .env)
│   │   ├── database.py     # SQLAlchemy async engine + session
│   │   ├── security.py     # JWT + bcrypt
│   │   └── dependencies.py # get_db(), get_reader(), get_current_user()
│   ├── analysis/           # Feature: análise de rótulos
│   ├── auth/               # Feature: autenticação
│   └── users/              # Feature: perfil e histórico
│
├── ocr_engine/             # Motor OCR (migrado do monolito)
│   ├── __init__.py         # build_reader() — único ponto de entrada
│   ├── nutrition/          # NutritionReader, presets, pipelines
│   ├── ocr/                # OcrService, qualidade, pós-processamento
│   ├── imaging/            # Operações PDI, ROI
│   ├── ingredients/        # Análise clínica DM
│   ├── audit/              # NullAuditRecorder (sem I/O em disco)
│   └── config/             # Presets JSON, ontologia, wordlist
│
├── alembic/                # Migrations de banco de dados
├── tests/                  # Testes automatizados
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Convenção de Commits e Branches

### Branch naming

Todo branch deve seguir o padrão:

```
(feat|fix|release)/<número-2-dígitos>/<nome-descritivo>

Exemplos válidos:
  feat/00/setup-inicial
  feat/01/ocr-engine
  fix/02/null-recorder-interface
  release/10/v1.0.0
```

Branches com nome inválido são **rejeitados no commit e no push** pelos git hooks instalados.

### Commit message

Todo commit deve seguir o padrão:

```
(feat|fix|release|refactor|docs|chore|merge): <mensagem descritiva>

Exemplos válidos:
  feat: add health check endpoint
  chore: initial project structure
  fix: correct import path in null_recorder
  docs: add README with commit conventions
  refactor: extract outcome serialization helper
```

Mensagens com formato inválido são **rejeitadas pelo commit-msg hook**.

### Git hooks instalados

Os hooks estão em `.git/hooks/` e são instalados automaticamente na criação do repositório:

| Hook | Valida |
|---|---|
| `pre-commit` | Nome do branch antes de cada commit |
| `commit-msg` | Nome do branch e formato da mensagem |
| `pre-push` | Nome do branch antes de qualquer push |

---

## Variáveis de Ambiente

Ver `.env.example` para a lista completa. Variáveis obrigatórias em produção:

| Variável | Descrição |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@host:5432/db` |
| `SECRET_KEY` | Chave aleatória 256 bits — gerar com `python -c "import secrets; print(secrets.token_hex(32))"` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path do service account GCV (opcional se não usar GCV) |
