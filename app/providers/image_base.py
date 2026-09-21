"""
image_base.py - Abstract base interface for Image Generation Providers in Orange AI.

Defines the contract and typed result representations for image generation.
Completely decoupled from conversational/text BaseProvider semantics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class ImageArtifact:
    """
    Represents a single generated image artifact from an image provider.
    """
    base64: str
    seed: Optional[int] = None
    mime_type: str = "image/png"
    width: int = 1024
    height: int = 1024


@dataclass(frozen=True)
class ImageGenerationResult:
    """
    Represents the complete result of an image generation request.
    """
    artifacts: List[ImageArtifact]
    model: str
    provider: str = "nvidia"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseImageProvider(ABC):
    """
    Abstract base interface for AI image generation providers.
    Image providers focus strictly on visual synthesis communication and do not
    participate in conversational memory, LLM planning, or text routing.
    """

    name: str = "base_image"
    model: str = "unknown"

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: Optional[int] = None,
        samples: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        Generate one or more image artifacts matching the prompt and parameters.
        Raises typed ProviderError subclasses on failures.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether this image provider is configured and fundamentally usable
        (e.g., required credentials and settings exist).
        Must NOT be used to check runtime rate limits or cooldowns.
        """
        pass
