# Autenticação

## Visão Geral

O sistema usa **JWT (JSON Web Tokens)** para autenticação stateless, com dois
tipos de token: `access` (vida curta) e `refresh` (vida longa). Tokens revogados
são rastreados em tabela `revoked_tokens`.

---

## Algoritmo e Chave

| Parâmetro | Valor |
|---|---|
| Algoritmo | `HS256` |
| Chave de assinatura | `SECRET_KEY` (variável de ambiente) |
| Biblioteca | `python-jose[cryptography]` |

O default `"dev-secret-key-change-in-production"` é explicitamente inseguro.
**Troque em produção** via variável de ambiente.

---

## Estrutura do Payload JWT

```json
{
  "sub": "550e8400-e29b-41d4-a716-446655440000",
  "exp": 1717668000,
  "type": "access"
}
```

| Campo | Descrição |
|---|---|
| `sub` | UUID do usuário |
| `exp` | Unix timestamp de expiração |
| `type` | `"access"` ou `"refresh"` |

O campo `type` é validado antes de usar o payload — um `refresh_token` não pode
ser aceito onde apenas `access_token` é válido, e vice-versa.

---

## Expiração dos Tokens

| Token | Campo de config | Default |
|---|---|---|
| Access token | `ACCESS_TOKEN_EXPIRE_MINUTES` | 15 min |
| Refresh token | `REFRESH_TOKEN_EXPIRE_DAYS` | 30 dias |

---

## Fluxo Completo

```
1. POST /auth/register  →  cria User (senha em bcrypt hash)
                           retorna UserResponse

2. POST /auth/login     →  verifica senha (bcrypt.verify)
                           emite access_token + refresh_token
                           retorna TokenResponse

3. GET  /users/me       →  Authorization: Bearer <access_token>
                           get_current_user_id valida JWT
                           retorna UserResponse

4. POST /auth/refresh   →  body: { refresh_token }
                           verifica tipo ("refresh") e expiração
                           verifica se não está em revoked_tokens
                           emite novo access_token
                           retorna AccessTokenResponse

5. POST /auth/logout    →  Authorization: Bearer <access_token>
                           body: { refresh_token }
                           insere refresh_token em revoked_tokens
                           retorna 204
```

---

## Hash de Senha

```python
from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

hash_password(password)       # bcrypt
verify_password(plain, hashed)
```

**Por que `bcrypt<4`:** `passlib 1.7.x` é incompatível com `bcrypt>=4` —
a versão 4 removeu o método `detect_wrap_bug` usado internamente. O
`requirements.txt` pina `bcrypt<4` explicitamente.

---

## Dependency `get_current_user_id`

```python
# app/core/dependencies.py
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(401, "Token inválido ou expirado.",
                            headers={"WWW-Authenticate": "Bearer"})
    return payload["sub"]   # str UUID
```

- Retorna `str` (UUID), não o objeto `User`.
- Consultas adicionais ao banco ficam no service — não na dependency.
- Usado via `user_id: str = Depends(get_current_user_id)` na assinatura do endpoint.

---

## Revogação de Tokens

A tabela `revoked_tokens` armazena JTI (ou o token completo) com `expires_at`.
A cada hora, uma `asyncio.Task` apaga tokens já expirados:

```python
delete(RevokedToken).where(RevokedToken.expires_at < datetime.now(timezone.utc))
```

Endpoints que usam `refresh_token` verificam se ele não está na tabela antes de
emitir novo `access_token`.

---

## Validação de Senha

O `RegisterRequest` valida o tamanho da senha via `@field_validator`:

```python
@field_validator("password")
@classmethod
def password_min_length(cls, v):
    if len(v) < 8:
        raise ValueError("A senha deve ter pelo menos 8 caracteres.")
    return v
```

Falha resulta em HTTP 422.

---

## E-mail

- Armazenado em **lowercase** (`email.lower()` em `create_user`).
- Não validado como `EmailStr` — a unicidade é garantida pela constraint
  `UNIQUE` no banco e verificação explícita no service.
- Mensagem de erro do login é **intencionalmente genérica** ("E-mail ou senha
  inválidos") para não revelar se um e-mail está cadastrado.

---

## Referências de Código

| Arquivo | Conteúdo |
|---|---|
| `app/core/security.py` | `hash_password`, `verify_password`, `create_access_token`, `create_refresh_token`, `verify_access_token`, `verify_refresh_token` |
| `app/core/dependencies.py` | `get_current_user_id` |
| `app/auth/router.py` | Endpoints `/register`, `/login`, `/refresh`, `/logout` |
| `app/auth/service.py` | `get_user_by_email`, `create_user`, `authenticate_user` |
| `app/auth/models.py` | `RevokedToken` ORM model |
| `app/auth/schemas.py` | `RegisterRequest`, `LoginRequest`, `TokenResponse`, etc. |
