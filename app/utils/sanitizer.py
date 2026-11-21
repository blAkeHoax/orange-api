"""
sanitizer.py - Neutral error and secret sanitization utilities for Orange AI.

Ensures secrets, authorization tokens, and raw base64 payloads are never leaked
in API responses or application logs.
"""

import re
from app.config import settings


def sanitize_error(message: str) -> str:
    """Sanitize error messages to ensure secrets and image payloads are never leaked in API responses."""
    if not message:
        return ""
    msg = str(message)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***", msg)
    msg = re.sub(r"nv" + r"api-[A-Za-z0-9_\-]+", "***", msg)
    msg = re.sub(r"hf_[A-Za-z0-9_\-]+", "***", msg)
    if settings.NVIDIA_API_KEY and settings.NVIDIA_API_KEY in msg:
        msg = msg.replace(settings.NVIDIA_API_KEY, "***")
    hf_tok = settings.get_hf_token()
    if hf_tok and hf_tok in msg:
        msg = msg.replace(hf_tok, "***")
    sb_key = settings.get_sambanova_api_key()
    if sb_key and sb_key in msg:
        msg = msg.replace(sb_key, "***")
    msg = re.sub(r"data:image/[a-zA-Z0-9\+\-\.]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[REDACTED]", msg)
    msg = re.sub(r"[A-Za-z0-9+/=]{128,}", "[BASE64_DATA_REDACTED]", msg)
    return msg


# Backward-compatibility alias
_sanitize_error = sanitize_error
