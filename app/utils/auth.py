"""
app/utils/auth.py - Shared-secret bearer authentication for Orange API.

Enforces configurable API key authorization on protected endpoints.
Uses constant-time comparison (hmac.compare_digest) to prevent timing attacks.
When ORANGE_API_KEY is not configured or empty, requests pass through unauthenticated
for local development and testing convenience.
"""

import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger("orange_api.auth")


def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Verify shared-secret bearer authentication for protected API endpoints.

    If settings.ORANGE_API_KEY is unset or empty, auth is disabled (open pass-through).
    If settings.ORANGE_API_KEY is configured:
      - Accepts Authorization: Bearer <token> (case-insensitive scheme) or X-API-Key: <token>.
      - Validates token using constant-time comparison (hmac.compare_digest).
      - Raises HTTP 401 with WWW-Authenticate header on missing or invalid tokens.
    """
    configured_key = settings.ORANGE_API_KEY.strip()
    if not configured_key:
        return None

    token: Optional[str] = None
    if authorization:
        parts = authorization.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        elif len(parts) == 1 and not parts[0].lower().startswith("bearer"):
            token = parts[0].strip()

    if not token and x_api_key:
        token = x_api_key.strip()

    if not token:
        logger.warning("Authentication failed: Missing API key in request headers")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(token.encode("utf-8"), configured_key.encode("utf-8")):
        logger.warning("Authentication failed: Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
