from typing import Optional, Dict, Type
import logging

from app.config import settings
from app.providers.base import BaseProvider
from app.providers.nvidia_provider import NVIDIAProvider

logger = logging.getLogger("orange_api.providers")

_PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "nvidia": NVIDIAProvider,
}


def get_provider(provider_name: Optional[str] = None) -> BaseProvider:
    """
    Resolve and instantiate the single active AI provider.
    Uses explicitly supplied provider if provided, otherwise uses settings.AI_PROVIDER.
    Never automatically selects another provider upon failure.
    """
    target = (provider_name or settings.AI_PROVIDER).strip().lower()
    provider_cls = _PROVIDER_REGISTRY.get(target)
    if not provider_cls:
        available = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported AI provider: '{target}'. Configured available providers: {available}"
        )

    return provider_cls()
