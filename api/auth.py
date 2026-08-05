"""
Access-token gate for NPSketch.

The app is exposed publicly through the Cloudflare tunnel (https://npsketch.jenai.de),
so every /api/* endpoint and the static UI are protected by a single shared access token.

Flow:
  1. A visitor lands on /gate.html and types the token.
  2. POST /api/auth/login validates it and sets an HMAC-signed, HttpOnly session cookie.
  3. AccessTokenMiddleware rejects any /api/* request that lacks a valid cookie (401).
  4. nginx `auth_request` calls GET /api/auth/check to gate the static pages too.

The token itself is NEVER stored in the cookie. The cookie carries only a signed
"<expiry>.<hmac>" proof, signed with a key derived from the token — so rotating the
token (in .env) instantly invalidates all existing sessions, and there is no second
secret to manage.

Fail-closed: if NPSKETCH_ACCESS_TOKEN is unset/empty the app is fully locked
(login → 503, every cookie invalid) rather than silently open.
"""

import hashlib
import hmac
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# The shared secret the user types on the gate page (from the gitignored root .env).
TOKEN = os.getenv("NPSKETCH_ACCESS_TOKEN", "")

COOKIE_NAME = "npsketch_session"
MAX_AGE = 30 * 24 * 3600  # 30 days
_SIGN_PREFIX = "npsketch-session-v1::"

# Requests under /api/ that must stay reachable without a valid session.
_PUBLIC_PATHS = {"/api/auth/login", "/api/auth/check", "/api/auth/logout"}


def _signing_key() -> bytes:
    """Derive the cookie-signing key from the access token."""
    return hashlib.sha256((_SIGN_PREFIX + TOKEN).encode()).digest()


def issue_cookie_value() -> str:
    """Mint a fresh signed session value: '<expiry_epoch>.<hex_hmac>'."""
    expiry = str(int(time.time()) + MAX_AGE)
    sig = hmac.new(_signing_key(), expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def valid_cookie(value: str) -> bool:
    """True iff `value` is a well-formed, correctly-signed, non-expired session cookie."""
    if not TOKEN or not value:
        return False  # fail-closed when no token is configured
    try:
        expiry_str, sig = value.split(".", 1)
        expected = hmac.new(_signing_key(), expiry_str.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        return int(expiry_str) > int(time.time())
    except (ValueError, AttributeError):
        return False


class AccessTokenMiddleware(BaseHTTPMiddleware):
    """Reject any /api/* request without a valid session cookie (401).

    Non-/api paths, the auth endpoints, and CORS preflight are passed through.
    This is the authoritative guard — it also protects the backend when reached
    directly on :8000, independent of nginx.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            not path.startswith("/api/")
            or request.method == "OPTIONS"
            or path in _PUBLIC_PATHS
        ):
            return await call_next(request)

        if valid_cookie(request.cookies.get(COOKIE_NAME, "")):
            return await call_next(request)

        return JSONResponse({"detail": "Authentication required"}, status_code=401)


router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    token: str


def _set_session_cookie(response: Response) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=issue_cookie_value(),
        max_age=MAX_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )


@router.post("/login")
async def login(payload: LoginRequest):
    """Exchange the shared access token for a signed session cookie."""
    if not TOKEN:
        return JSONResponse(
            {"detail": "Access token not configured on the server"}, status_code=503
        )
    if not hmac.compare_digest(payload.token or "", TOKEN):
        return JSONResponse({"detail": "Invalid token"}, status_code=401)

    response = JSONResponse({"ok": True})
    _set_session_cookie(response)
    return response


@router.get("/check")
async def check(request: Request):
    """200 if the session cookie is valid, else 401 (used by nginx auth_request)."""
    if valid_cookie(request.cookies.get(COOKIE_NAME, "")):
        return Response(status_code=200)
    return Response(status_code=401)


@router.post("/logout")
async def logout():
    """Clear the session cookie."""
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response
