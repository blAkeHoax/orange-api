from fastapi import APIRouter, HTTPException
import logging

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

logger = logging.getLogger("orange_api.assist")
router = APIRouter()
assistant = AssistService()


@router.post(
    "/assist",
    response_model=AssistResponse,
)
def assist(request: AssistRequest):
    """
    Handle assistance requests via the configured AIProvider.
    """
    try:
        result = assistant.assist(
            message=request.message,
            history=request.history,
            profile=request.profile,
            system_instruction=request.system_instruction,
            preferred_provider=request.provider,
            model=request.model,
            task_type=request.task_type,
            modalities=request.modalities,
        )
        return AssistResponse(**result)

    except (RoutingError, NoEligibleModelError, IneligibleOverrideError, UnknownModelError, ValueError) as e:
        logger.warning("Routing / Bad request error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except ProviderRateLimitError as e:
        logger.warning("Provider rate limit error: %s", e)
        raise HTTPException(status_code=429, detail=f"AI Provider rate limit: {e}")

    except ProviderUnavailableError as e:
        logger.warning("Provider unavailable error: %s", e)
        raise HTTPException(status_code=503, detail=f"AI Provider unavailable: {e}")

    except ProviderTimeoutError as e:
        logger.warning("Provider timeout error: %s", e)
        raise HTTPException(status_code=504, detail=f"AI Provider timeout: {e}")

    except ProviderError as e:
        logger.error("Provider error: %s", e)
        raise HTTPException(status_code=502, detail=f"AI Provider error: {e}")

    except Exception as e:
        logger.error("Assist request unexpected failure: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal assist service error: {e}")