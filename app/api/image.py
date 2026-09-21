"""
image.py - FastAPI router for Image Generation in Orange API (Phase 7B).

Exposes POST /generate/image.
Delegates strictly to ImageService and maps provider exceptions to HTTP status codes.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import (
    ImageGenerateRequest,
    ImageGenerateResponse,
)
from app.services.image_service import ImageService
from app.services.image_model_registry import UnknownImageModelError
from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
)

import re
from app.config import settings

logger = logging.getLogger("orange_api.image")
router = APIRouter()
image_service = ImageService()


def _sanitize_error(message: str) -> str:
    """Sanitize error messages to ensure secrets are never leaked in API responses."""
    msg = str(message)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***", msg)
    msg = re.sub(r"nv" + r"api-[A-Za-z0-9_\-]+", "***", msg)
    if settings.NVIDIA_API_KEY and settings.NVIDIA_API_KEY in msg:
        msg = msg.replace(settings.NVIDIA_API_KEY, "***")
    return msg


@router.post(
    "/generate/image",
    response_model=ImageGenerateResponse,
)
def generate_image(request: ImageGenerateRequest):
    """
    Generate one or more images from a text prompt via the configured image provider.
    """
    try:
        return image_service.generate_image(request)

    except UnknownImageModelError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Unknown image model requested: %s", clean_e)
        raise HTTPException(status_code=400, detail=clean_e)

    except ProviderRateLimitError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Image provider rate limit error: %s", clean_e)
        raise HTTPException(status_code=429, detail=f"AI Provider rate limit: {clean_e}")

    except ProviderUnavailableError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Image provider unavailable error: %s", clean_e)
        raise HTTPException(status_code=503, detail=f"AI Provider unavailable: {clean_e}")

    except ProviderTimeoutError as e:
        clean_e = _sanitize_error(str(e))
        logger.warning("Image provider timeout error: %s", clean_e)
        raise HTTPException(status_code=504, detail=f"AI Provider timeout: {clean_e}")

    except ProviderError as e:
        clean_e = _sanitize_error(str(e))
        logger.error("Image provider error: %s", clean_e)
        raise HTTPException(status_code=502, detail=f"AI Provider error: {clean_e}")

    except Exception as e:
        clean_e = _sanitize_error(str(e))
        logger.error("Image generation unexpected failure: %s", clean_e)
        raise HTTPException(status_code=500, detail=f"Internal image service error: {clean_e}")
