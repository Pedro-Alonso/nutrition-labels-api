# Deploy e Operação

## Setup Rápido (Desenvolvimento)

```bash
# 1. Copiar variáveis de ambiente
cp .env.example .env
# Editar .env com valores reais (SECRET_KEY obrigatório)

# 2. (Opcional) Adicionar service account do Google Cloud Vision
mkdir -p secrets
cp /caminho/para/service_account.json secrets/service_account.json

# 3. Subir containers
docker compose up --build

# 4. Verificar API
curl http://localhost:8000/api/v1/health
```

A API fica disponível em `http://localhost:8000`.
Swagger UI: `http://localhost:8000/docs`.

---

## Variáveis de Ambiente

Definidas em `.env` (não commitado) — ver `.env.example` para a lista completa.

| Variável | Obrigatório | Default | Descrição |
|---|---|---|---|
| `DATABASE_URL` | sim | `postgresql+asyncpg://rotulos_user:rotulos_pass@db:5432/rotulos_db` | Conexão PostgreSQL |
| `SECRET_KEY` | sim (prod) | `"dev-secret-key-..."` | Chave JWT — **gerar aleatoriamente em produção** |
| `ALGORITHM` | não | `"HS256"` | Algoritmo JWT |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | não | `15` | Expiração do access token |
| `REFRESH_TOKEN_EXPIRE_DAYS` | não | `30` | Expiração do refresh token |
| `GOOGLE_APPLICATION_CREDENTIALS` | não | `null` | Path do service account GCV |
| `MAX_UPLOAD_SIZE_MB` | não | `10` | Limite de upload em `/analyze` |
| `TESSDATA_PREFIX` | não | `null` | Override do diretório tessdata |
| `ALLOWED_ORIGINS` | não | `"*"` | CORS origins (separados por vírgula) |

### Gerar `SECRET_KEY` seguro

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Docker Compose (Desenvolvimento)

`docker-compose.yml` sobe dois serviços:

| Serviço | Imagem | Porta | Descrição |
|---|---|---|---|
| `api` | Build local (Dockerfile) | 8000 | FastAPI + Uvicorn + motor OCR |
| `db` | `postgres:16-alpine` | 5432 | PostgreSQL com volume persistente |

O serviço `api` aguarda `db` estar saudável (`pg_isready`) antes de iniciar.

O container `api` executa automaticamente:
```
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Comandos Docker úteis

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
```

---

## Docker Compose (Testes / CI)

`docker-compose.test.yml` sobe dois serviços isolados:

| Serviço | Descrição |
|---|---|
| `db-test` | PostgreSQL 16 com healthcheck `pg_isready` |
| `api-test` | Stage `test` do Dockerfile; executa migrations + pytest |

```bash
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

`--abort-on-container-exit` garante que o exit code do pytest retorna ao CI.

---

## Dockerfile

O Dockerfile usa multi-stage build (implícito via `target` no compose):

**Stage `base`:**
- `python:3.11-slim`
- Instala Tesseract OCR + pacote de idioma `por`
- Instala libgl1 (dependência do OpenCV)
- Instala dependências Python do `requirements.txt`

**Stage `test`:**
- Baseado em `base`
- Instala dependências adicionais de `requirements-dev.txt`
- Copia o código-fonte (incluindo `tests/`)

---

## Escalabilidade

### Por que `--workers 1`

`NutritionReader` é um singleton **em memória de processo**. Com múltiplos
workers no mesmo processo, cada worker manteria sua própria cópia do reader em
memória — multiplicando o consumo de RAM (≈500 MB por instância) e invalidando
o cache GCV em disco por condições de corrida.

**Estratégia correta para escalar:**

```
Load Balancer
    ├─ Container 1 (--workers 1) → PostgreSQL
    ├─ Container 2 (--workers 1) → PostgreSQL
    └─ Container N (--workers 1) → PostgreSQL
```

Cada container tem seu reader isolado; PostgreSQL é o estado compartilhado.

---

## Configuração do Google Cloud Vision

1. Criar um projeto no Google Cloud Console.
2. Habilitar a **Cloud Vision API**.
3. Criar uma Service Account com role **Cloud Vision > Cloud Vision API User**.
4. Baixar a chave JSON da service account.
5. Copiar para `secrets/service_account.json`.
6. Configurar `docker-compose.yml` ou `.env`:

```bash
GOOGLE_APPLICATION_CREDENTIALS="/app/secrets/service_account.json"
```

O `docker-compose.yml` já monta `./secrets/service_account.json:/app/secrets/service_account.json:ro`.

Para desabilitar GCV, não configure a variável. Os presets `cloud_vision` serão
ignorados (carregados, mas sem credencial disponível retornam erro que é tratado
com `on_failure: "skip"`).

---

## Segurança em Produção

| Item | Recomendação |
|---|---|
| `SECRET_KEY` | Gerar com `secrets.token_hex(32)`; nunca usar o default |
| `ALLOWED_ORIGINS` | Restringir para a URL do app mobile em vez de `"*"` |
| `DATABASE_URL` | Usar senha forte; não expor porta 5432 publicamente |
| `secrets/` | Nunca commitar; adicionar ao `.gitignore` |
| HTTPS | Usar reverse proxy (nginx/traefik) com TLS na frente do Uvicorn |
| Rate limiting | `MAX_UPLOAD_SIZE_MB` e slowapi limitam abuso de `/analyze` |

---

## Monitoramento

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

Pode ser usado como probe de liveness em Kubernetes ou Docker Swarm.

### Logs

O `LoggingMiddleware` emite uma linha por request:

```
method=POST path=/api/v1/analyze status=200 duration_ms=3421.5 request_id=abc123
```

O `request_id` é propagado via header `X-Request-ID` (gerado automaticamente
se não enviado pelo cliente).

### Tesseract

Verificar instalação:
```bash
docker compose exec api tesseract --version
docker compose exec api tesseract --list-langs
```

O idioma `por` deve aparecer na lista.

---

## Migrations em Produção

**Nunca rode migrations sem backup do banco.** Procedimento recomendado:

```bash
# 1. Backup
pg_dump rotulos_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Verificar status
docker compose exec api alembic current

# 3. Preview das mudanças (não aplica)
docker compose exec api alembic upgrade head --sql

# 4. Aplicar
docker compose exec api alembic upgrade head
```
