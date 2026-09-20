from typing import List, Dict, Any, Optional, Tuple
import logging

from app.providers import get_provider, BaseProvider
from app.services.model_registry import resolve_provider_model_id

logger = logging.getLogger("orange_api.router")


class ModelRouter:
    """
    Backwards-compatibility resolver for legacy model routing imports.
    Directly delegates to the configured active AIProvider without multi-provider
    routing or automatic fallback.
    """

    def __init__(self, provider: Optional[BaseProvider] = None):
        self.provider = provider or get_provider()

    def route_and_generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        preferred_provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """
        Directly delegates generation to the active configured provider.
        Does NOT perform automatic fallback between providers.
        """
        active_provider = get_provider(preferred_provider) if preferred_provider else self.provider
        resolved_model_id = resolve_provider_model_id(model, default_id=active_provider.model)

        reply = active_provider.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history,
            model=resolved_model_id,
            **kwargs,
        )
        return reply, f"{active_provider.name}:{resolved_model_id}"

    def generate(self, message: str, model: Optional[str] = None) -> str:
        """Helper for single message generation."""
        reply, _ = self.route_and_generate(message, model=model)
        return reply