from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address


def _key_func(request):
    if os.getenv("TESTING"):
        # Chave única por requisição para que o rate limiter nunca dispare nos testes
        import uuid
        return str(uuid.uuid4())
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)
