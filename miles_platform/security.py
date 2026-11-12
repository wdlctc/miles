"""Authenticate before choosing a workspace, including bearer-key inference routes."""

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from miles_platform.store import digest
from miles_platform.tenancy import CURRENT

PUBLIC = {"/api/login", "/api/invitations/accept", "/api/account/reset", "/api/stripe/webhook"}


class WorkspaceMiddleware:
    def __init__(self, app, registry):
        self.app, self.registry = app, registry

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        path = scope["path"]
        store = self.registry.default
        try:
            if path.startswith("/v1/"):
                authorization = request.headers.get("authorization", "")
                if not authorization.startswith("Bearer "):
                    raise HTTPException(401, "A Bearer API key is required")
                token = authorization[7:]
                parts = token.split(".")
                tenant = parts[1] if len(parts) == 3 and parts[0] == "mp" else "default"
                try:
                    store = self.registry.store(tenant)
                except HTTPException:
                    raise HTTPException(401, "Invalid API key") from None
                key = store.one("SELECT id FROM api_keys WHERE hash=? AND revoked IS NULL", (digest(token),))
                if not key:
                    raise HTTPException(401, "Invalid API key")
                self.registry.active(tenant)
            elif path.startswith("/api/") and path not in PUBLIC:
                principal = self.registry.principal(request.cookies.get("miles_session", ""))
                scope.setdefault("state", {})["principal"] = principal
                store = self.registry.store(principal["tenant"])
            # Enforce the actual streamed body size, not just Content-Length.
            if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}:
                chunks, size = [], 0
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunks.append(message.get("body", b""))
                    size += len(chunks[-1])
                    if size > (262144 if path == "/api/stripe/webhook" else 6_100_000):
                        raise HTTPException(413, "Request too large")
                    if not message.get("more_body"):
                        break
                original, body = receive, b"".join(chunks)
                sent = False

                async def replay():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await original()

                receive = replay
        except HTTPException as exc:
            return await JSONResponse({"detail": exc.detail}, status_code=exc.status_code)(scope, receive, send)
        token = CURRENT.set(store)
        try:
            await self.app(scope, receive, send)
        finally:
            CURRENT.reset(token)
