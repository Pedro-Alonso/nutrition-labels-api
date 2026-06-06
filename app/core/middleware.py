from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            ms,
            req_id,
        )
        response.headers["X-Request-ID"] = req_id
        return response
