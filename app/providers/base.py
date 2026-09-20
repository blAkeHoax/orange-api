from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ProviderError(Exception):
    """Base exception for all AI provider errors."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when the provider is unconfigured, unreachable, or missing credentials."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limits, quotas, or concurrency limits are exceeded."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when inference with the provider times out."""
    pass


class BaseProvider(ABC):
    """
    Abstract base interface for AI model providers.
    Providers focus strictly on model inference communication and do not
    contain memory, planning, routing, or tool execution logic.
    """

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response for the given prompt and optional context."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether this provider is configured and fundamentally usable
        (e.g., required credentials and settings exist).
        Must NOT be used to check runtime rate limits or cooldowns.
        """
        pass