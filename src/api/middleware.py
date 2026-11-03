"""
Authentication-ready and telemetry middleware.

Provides API key and Bearer token verification hooks for mobile (Android) clients,
CORS security, request timing, and correlation IDs.
"""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CorrelationAndTimingMiddleware(BaseHTTPMiddleware):
    """Injects a unique request correlation ID and measures request execution time."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Optional authentication middleware for mobile / API clients.

    When APP_API_KEY_ENABLED is true, validates X-API-Key or Bearer token header.
    Bypasses authentication for health checks, docs, and UI routes.
    """

    EXEMPT_PATHS = {
        "/",
        "/ui",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
        "/api/v1/model-info",
    }

    async def dispatch(self, request: Request, call_next):
        # Allow static files and exempt endpoints freely
        path = request.url.path
        if path in self.EXEMPT_PATHS or path.startswith("/static/"):
            return await call_next(request)

        if settings.api_key_enabled:
            api_key = request.headers.get("X-API-Key")
            auth_header = request.headers.get("Authorization")

            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
                if token == settings.api_key:
                    return await call_next(request)

            if not api_key or (settings.api_key and api_key != settings.api_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "UNAUTHORIZED",
                        "detail": "Invalid or missing API key in X-API-Key or Authorization header.",
                    },
                )

        return await call_next(request)
