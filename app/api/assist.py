import asyncio
import functools
import logging
import anyio
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import (
    AssistRequest,
    AssistResponse,
)
from app.services.assist_service import AssistService
from app.services.intelligence_router import (
    RoutingError,
    NoEligibleModelError,
    IneligibleOverrideError,
)
from app.services.model_registry import UnknownModelError
from app.providers import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
)
from app.utils.sanitizer import _sanitize_error

logger = logging.getLogger("orange_api.assist")
router = APIRouter()
assistant = AssistService()


@router.post(
    "/assist",
    response_model=AssistResponse,
)
async def assist(request: AssistRequest):
    """
    Handle assistance requests via the configured AIProvider.
    Enforces a genuine 90.0-second route deadline for overall request completion.
    """
    route_deadline = settings.ROUTE_TIMEOUT
    try:
        worker_call = functools.partial(
            assistant.assist,
            message=request.message,
            history=request.history,
            profile=request.profile,
            system_instruction=request.system_instruction,
            preferred_provider=request.provider,
            model=request.model,
            task_type=request.task_type,
            modalities=request.modalities,
        )
        result = await asyncio.wait_for(
            anyio.to_thread.run_sync(worker_call),
            timeout=route_deadline,
        )
        return AssistResponse(**result)

    except asyncio.TimeoutError:
        logger.warning("Assist route deadline exceeded (%.1fs)", route_deadline)
        raise HTTPException(
            status_code=504,
            detail=f"AI Provider timeout: Request deadline exceeded ({route_deadline:.1f}s)",
        )

    except (RoutingError, NoEligibleModelError, IneligibleOverrideError, UnknownModelError, ValueError) as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Routing / Bad request error: %s", clean_e)
        raise HTTPException(status_code=400, detail=clean_e)

    except ProviderRateLimitError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Provider rate limit error: %s", clean_e)
        raise HTTPException(status_code=429, detail=f"AI Provider rate limit: {clean_e}")

    except ProviderUnavailableError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Provider unavailable error: %s", clean_e)
        raise HTTPException(status_code=503, detail=f"AI Provider unavailable: {clean_e}")

    except ProviderTimeoutError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Provider timeout error: %s", clean_e)
        raise HTTPException(status_code=504, detail=f"AI Provider timeout: {clean_e}")

    except ProviderError as e:
        clean_e = _sanitize_error(str(e))
        logger.error("Provider error: %s", clean_e)
        raise HTTPException(status_code=502, detail=f"AI Provider error: {clean_e}")

    except HTTPException:
        raise

    except Exception as e:
        clean_e = _sanitize_error(str(e))
        logger.error("Assist request unexpected failure: %s", clean_e)
        raise HTTPException(status_code=500, detail=f"Internal assist service error: {clean_e}")