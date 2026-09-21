"""
image_model_registry.py - Dedicated Image Generation Model Registry for Orange AI.

Registers supported image synthesis models independently from the four-model
language intelligence foundation. Automatic routing, text classification, and
conversational pipelines do not interact with this registry.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


class UnknownImageModelError(ValueError):
    """Raised when a requested image model key or ID is not found in the registry."""
    pass


@dataclass(frozen=True)
class ImageModelInfo:
    """
    Metadata representation of an Image Generation model in the registry.
    """
    key: str
    provider_model_id: str
    provider: str
    display_name: str
    capabilities: List[str]
    supported_resolutions: List[Tuple[int, int]]
    default_steps: int
    max_steps: int
    default_samples: int
    description: Optional[str] = None


# Authoritative Image Model Registry (FLUX.2-klein-4B initially)
_IMAGE_MODELS: Dict[str, ImageModelInfo] = {
    "flux_2_klein_4b": ImageModelInfo(
        key="flux_2_klein_4b",
        provider_model_id="black-forest-labs/flux.2-klein-4b",
        provider="nvidia",
        display_name="FLUX.2 Klein 4B",
        capabilities=["text-to-image"],
        supported_resolutions=[(1024, 1024)],
        default_steps=4,
        max_steps=4,
        default_samples=1,
        description="High-speed rectified flow distilled transformer for text-to-image synthesis.",
    ),
}

# Reverse lookup mapping from provider_model_id to ImageModelInfo
_PROVIDER_ID_MAP: Dict[str, ImageModelInfo] = {
    model.provider_model_id: model for model in _IMAGE_MODELS.values()
}


def get_image_model(model_key_or_id: str) -> ImageModelInfo:
    """
    Look up an image model by internal key (e.g. 'flux_2_klein_4b') or
    by provider model ID (e.g. 'black-forest-labs/flux.2-klein-4b').
    Raises UnknownImageModelError if unrecognized.
    """
    cleaned = (model_key_or_id or "").strip()
    if cleaned in _IMAGE_MODELS:
        return _IMAGE_MODELS[cleaned]
    if cleaned in _PROVIDER_ID_MAP:
        return _PROVIDER_ID_MAP[cleaned]

    known_keys = list(_IMAGE_MODELS.keys())
    known_ids = list(_PROVIDER_ID_MAP.keys())
    raise UnknownImageModelError(
        f"Unknown image model identifier '{model_key_or_id}'. "
        f"Registered model keys: {known_keys}, provider IDs: {known_ids}"
    )


def list_image_models() -> List[ImageModelInfo]:
    """Return all registered image generation models."""
    return list(_IMAGE_MODELS.values())


def image_model_exists(model_key_or_id: str) -> bool:
    """Check if an image model exists by internal key or provider model ID."""
    cleaned = (model_key_or_id or "").strip()
    return cleaned in _IMAGE_MODELS or cleaned in _PROVIDER_ID_MAP
