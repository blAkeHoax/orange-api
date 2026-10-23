"""
latency.py - Diagnostic latency instrumentation for Orange API.

Provides minimal, privacy-safe timing instrumentation across the request lifecycle:
- Request start and completion
- Intelligence routing duration
- NVIDIA provider round-trip duration
- Time-to-first-token (TTFT) availability tracking
- Response serialization and total elapsed time

All stages are correlated via request_id without logging prompts, replies, or credentials.
"""

import time
import uuid
import logging
from contextvars import ContextVar
from typing import Optional, Dict, Any, List

logger = logging.getLogger("orange_api.latency")

# Context-local correlation ID propagated across async and thread boundaries
current_request_id: ContextVar[str] = ContextVar("current_request_id", default="")

# In-memory diagnostic ring buffer for automated test verification (up to 100 entries)
_MAX_TIMING_RECORDS = 100
_timing_records: List[Dict[str, Any]] = []

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "token",
    "secret",
    "password",
    "prompt",
    "message",
    "reply",
    "content",
    "history",
    "profile",
    "system_instruction",
}


def generate_request_id() -> str:
    """Generate a short unique request correlation ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def get_current_request_id() -> str:
    """Get the current request ID from context or return a fallback."""
    req_id = current_request_id.get()
    return req_id if req_id else "no_req_id"


def get_or_create_request_id(explicit_id: Optional[str] = None) -> str:
    """Get existing context request ID or set/create one if none exists."""
    if explicit_id and explicit_id.strip():
        req_id = explicit_id.strip()
        current_request_id.set(req_id)
        return req_id

    existing = current_request_id.get()
    if existing and existing != "no_req_id":
        return existing

    new_id = generate_request_id()
    current_request_id.set(new_id)
    return new_id


def set_current_request_id(req_id: Optional[str] = None) -> str:
    """Explicitly set or generate the current request ID in context."""
    return get_or_create_request_id(req_id)


def log_latency(
    stage: str,
    duration_ms: Optional[float] = None,
    request_id: Optional[str] = None,
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Log a privacy-safe timing measurement for a specific request stage.
    Never logs prompts, replies, or sensitive credentials.
    """
    req_id = request_id or get_current_request_id()

    record: Dict[str, Any] = {
        "request_id": req_id,
        "stage": stage,
    }

    if duration_ms is not None:
        record["duration_ms"] = round(float(duration_ms), 2)

    # Sanitize metadata: only allow non-sensitive scalar fields
    safe_metadata: Dict[str, Any] = {}
    for k, v in metadata.items():
        if k.lower() in _SENSITIVE_KEYS:
            continue
        if isinstance(v, (int, float, bool)):
            safe_metadata[k] = v
        elif isinstance(v, str):
            clean_val = v[:100]
            if "Bearer" in clean_val or "nvapi-" in clean_val:
                continue
            safe_metadata[k] = clean_val

    record.update(safe_metadata)

    # Format human-readable log line
    parts = [f"request_id={req_id}", f"stage={stage}"]
    if duration_ms is not None:
        parts.append(f"duration_ms={record['duration_ms']:.2f}ms")
    for k, v in safe_metadata.items():
        parts.append(f"{k}={v}")

    log_line = "[LATENCY] " + " ".join(parts)
    logger.info(log_line)

    # Retain in diagnostic memory for local test assertions
    _timing_records.append(record)
    if len(_timing_records) > _MAX_TIMING_RECORDS:
        _timing_records.pop(0)

    return record


def get_recorded_timings(
    request_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recorded timing events, optionally filtered by request_id or stage."""
    records = list(_timing_records)
    if request_id:
        records = [r for r in records if r.get("request_id") == request_id]
    if stage:
        records = [r for r in records if r.get("stage") == stage]
    return records


def clear_recorded_timings() -> None:
    """Clear recorded timings buffer (used between test runs)."""
    _timing_records.clear()
