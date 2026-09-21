"""
image_service.py - Business logic service for Image Generation in Orange API (Phase 7B).

Coordinates image model resolution from ImageModelRegistry and delegates
to the appropriate BaseImageProvider implementation.
Completely isolated from conversational routing (IntelligenceRouter) and text ModelRegistry.
"""

import time
import logging
from typing import Optional

from app.models.schemas import (
    ImageGenerateRequest,
    ImageGenerateResponse,
    GeneratedImage,
)
from app.services.image_model_registry import (
    get_image_model,
    ImageModelInfo,
)
from app.providers.image_base import BaseImageProvider
from app.providers.nvidia_image_provider import NVIDIAImageProvider

logger = logging.getLogger("orange_api.image_service")


class ImageService:
    """
    Service layer for image generation requests.
    Validates requests, resolves models from ImageModelRegistry,
    and delegates to BaseImageProvider without fallback or conversational dependencies.
    """

    def __init__(self, provider: Optional[BaseImageProvider] = None):
        self._provider = provider

    def _resolve_provider(self, model_info: ImageModelInfo) -> BaseImageProvider:
        """Resolve the active image provider for the resolved image model."""
        if self._provider is not None:
            return self._provider

        if model_info.provider == "nvidia":
            return NVIDIAImageProvider(model=model_info.provider_model_id)

        raise ValueError(f"Unsupported image provider: '{model_info.provider}'")

    def generate_image(self, request: ImageGenerateRequest) -> ImageGenerateResponse:
        """
        Process an image generation request and return structured response data.
        Measures execution duration with a monotonic clock.
        """
        # 1. Resolve requested image model through ImageModelRegistry
        model_key = request.model or "flux_2_klein_4b"
        model_info = get_image_model(model_key)

        # 2. Resolve image provider
        provider = self._resolve_provider(model_info)

        # 3. Dispatch to image provider with monotonic duration measurement
        start_time = time.monotonic()
        result = provider.generate_image(
            prompt=request.prompt,
            width=request.width or 1024,
            height=request.height or 1024,
            steps=request.steps or 4,
            seed=request.seed,
            samples=request.samples or 1,
            model=model_info.provider_model_id,
        )
        duration_ms = max(0, int((time.monotonic() - start_time) * 1000))

        # 4. Map provider ImageGenerationResult to ImageGenerateResponse
        images = [
            GeneratedImage(
                b64_json=artifact.base64,
                mime_type=artifact.mime_type,
                width=artifact.width,
                height=artifact.height,
            )
            for artifact in result.artifacts
        ]

        primary_seed = result.artifacts[0].seed if result.artifacts else None

        logger.info(
            "[IMAGE SERVICE] Generated %d image(s) via model '%s' in %dms",
            len(images),
            model_info.key,
            duration_ms,
        )

        return ImageGenerateResponse(
            success=True,
            model=model_info.key,
            images=images,
            seed=primary_seed,
            duration_ms=duration_ms,
            error=None,
        )
