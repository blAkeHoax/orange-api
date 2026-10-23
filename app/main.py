import logging
from typing import Any, Tuple

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.assist import router as assist_router
from app.utils.sanitizer import _sanitize_error
from app.utils.latency import current_request_id, set_current_request_id

logger = logging.getLogger("orange_api.main")

_OMIT_LOC_NAMES = {
    "b64_json",
    "authorization",
    "api_key",
    "token",
    "secret",
    "password",
}


def _sanitize_validation_input(val: Any, loc: Tuple[Any, ...]) -> Any:
    """Sanitize and bound input values echoed in validation errors."""
    if any(isinstance(part, str) and part.lower() in _OMIT_LOC_NAMES for part in loc):
        return "[omitted]"

    if loc and str(loc[-1]).lower() == "image":
        return "[omitted]"

    if isinstance(val, dict):
        has_sensitive_key = any(
            isinstance(k, str) and k.lower() in (_OMIT_LOC_NAMES | {"image"})
            for k in val.keys()
        )
        if has_sensitive_key or len(str(val)) > 256:
            return "[omitted]"
        return {
            k: _sanitize_validation_input(v, (*loc, k))
            for k, v in val.items()
        }

    if isinstance(val, list):
        if len(str(val)) > 256:
            return "[omitted]"
        return [_sanitize_validation_input(item, loc) for item in val]

    val_str = str(val)
    if "data:image/" in val_str or ";base64," in val_str:
        return "[omitted]"

    clean_str = _sanitize_error(val_str)
    if len(clean_str) > 256:
        return clean_str[:256] + "... [truncated]"

    if isinstance(val, (int, float, bool, type(None))):
        return val

    return clean_str


app = FastAPI(
    title="Orange API",
    version="1.0.0",
    description="Backend API for Orange AI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Ensure every HTTP request has a correlation ID attached to headers."""
    incoming_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    req_id = set_current_request_id(incoming_id)
    token = current_request_id.set(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        current_request_id.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle FastAPI RequestValidationError securely:
    - Never echo sensitive fields, base64 data, or credentials.
    - Omit or truncate inputs exceeding 256 characters.
    - Redact sensitive credentials/tokens.
    - Avoid logging oversized input bodies.
    - Retain HTTP 422 status and standard {"detail": [...]} envelope.
    """
    cleaned_errors = []
    for err in exc.errors():
        clean_err = dict(err)
        loc = tuple(clean_err.get("loc", ()))

        if "input" in clean_err:
            clean_err["input"] = _sanitize_validation_input(clean_err["input"], loc)

        if "ctx" in clean_err and isinstance(clean_err["ctx"], dict):
            clean_ctx = {}
            for ctx_k, ctx_v in clean_err["ctx"].items():
                ctx_str = _sanitize_error(str(ctx_v))
                if len(ctx_str) > 256:
                    ctx_str = ctx_str[:256] + "... [truncated]"
                clean_ctx[ctx_k] = ctx_str
            clean_err["ctx"] = clean_ctx

        if "msg" in clean_err:
            clean_err["msg"] = _sanitize_error(str(clean_err["msg"]))

        cleaned_errors.append(clean_err)

    error_summary = [
        {"loc": e.get("loc"), "type": e.get("type"), "msg": e.get("msg")}
        for e in cleaned_errors
    ]
    logger.warning("Request validation failed for %s: %s", request.url.path, error_summary)

    response = JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(cleaned_errors)},
    )
    req_id = current_request_id.get()
    if req_id:
        response.headers["X-Request-ID"] = req_id
    return response


app.include_router(assist_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to Orange API"
    }

@app.get("/health")
def health():
    return {
        "status": "online"
    }
