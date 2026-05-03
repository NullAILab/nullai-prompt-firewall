"""
fastapi_middleware.py — Drop-in FastAPI middleware for prompt injection detection.

Usage
-----
    from middleware.fastapi_middleware import PromptFirewallMiddleware

    app = FastAPI()
    app.add_middleware(
        PromptFirewallMiddleware,
        prompt_field="prompt",      # JSON body field to inspect
        block_on=["CRITICAL", "HIGH"],  # severities that return HTTP 400
        add_headers=True,           # attach X-Firewall-* headers to every response
    )

The middleware:
  - Reads the JSON request body and extracts the configured field.
  - Runs the prompt injection firewall.
  - If the verdict severity is in *block_on*, returns HTTP 400 with a JSON error.
  - Otherwise, passes the request through and optionally adds verdict headers.

Note: The middleware buffers the request body so that downstream handlers can
still read it after inspection.
"""

from __future__ import annotations

import json
from typing import Sequence

try:
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detector.firewall import check as firewall_check, Verdict


# ─── Middleware ───────────────────────────────────────────────────────────────

if _FASTAPI_AVAILABLE:

    class PromptFirewallMiddleware(BaseHTTPMiddleware):
        """
        Starlette/FastAPI middleware that screens request bodies for prompt
        injection before forwarding to the application.
        """

        def __init__(
            self,
            app: ASGIApp,
            prompt_field: str = "prompt",
            block_on: Sequence[str] = ("CRITICAL", "HIGH"),
            add_headers: bool = True,
            use_classifier: bool = True,
        ) -> None:
            super().__init__(app)
            self.prompt_field = prompt_field
            self.block_on = set(block_on)
            self.add_headers = add_headers
            self.use_classifier = use_classifier

        async def dispatch(self, request: Request, call_next):
            # Only inspect requests with a JSON body
            content_type = request.headers.get("content-type", "")
            if "application/json" not in content_type:
                return await call_next(request)

            try:
                body_bytes = await request.body()
                body = json.loads(body_bytes)
            except (json.JSONDecodeError, Exception):
                return await call_next(request)

            prompt = body.get(self.prompt_field)
            if not isinstance(prompt, str):
                return await call_next(request)

            verdict: Verdict = firewall_check(prompt, use_classifier=self.use_classifier)

            # Block if severity matches the block list
            if verdict.severity in self.block_on:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "prompt_injection_detected",
                        "severity": verdict.severity,
                        "confidence": round(verdict.confidence, 3),
                        "reasons": verdict.reasons,
                    },
                )

            # Pass through — optionally add verdict headers
            response = await call_next(request)
            if self.add_headers:
                response.headers["X-Firewall-Severity"] = verdict.severity
                response.headers["X-Firewall-Confidence"] = f"{verdict.confidence:.3f}"
                response.headers["X-Firewall-Injection"] = str(verdict.is_injection).lower()
            return response

else:
    # Graceful stub when FastAPI is not installed
    class PromptFirewallMiddleware:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FastAPI is not installed. "
                "Run: pip install fastapi uvicorn"
            )


# ─── Standalone FastAPI demo app (run with: uvicorn middleware.fastapi_middleware:demo_app) ──

def _build_demo_app():
    if not _FASTAPI_AVAILABLE:
        return None
    from fastapi import FastAPI

    demo = FastAPI(title="Prompt Injection Firewall Demo")
    demo.add_middleware(
        PromptFirewallMiddleware,
        prompt_field="prompt",
        block_on=["CRITICAL", "HIGH"],
        add_headers=True,
    )

    @demo.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        return {"echo": body.get("prompt", ""), "status": "passed_firewall"}

    @demo.get("/health")
    async def health():
        return {"status": "ok"}

    return demo


demo_app = _build_demo_app()
