"""
nvidia_image_provider.py - NVIDIA GenAI Image Generation Provider for Orange AI.

Communicates with NVIDIA's hosted Generative AI API:
https://ai.api.nvidia.com/v1/genai

Specifically targets Black Forest Labs FLUX.2-klein-4B:
https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b

Adheres to:
- Independent provider boundary (BaseImageProvider).
- Strict parameter validation before making HTTP requests.
- Maps native NVIDIA artifacts[].base64 responses into typed ImageGenerationResult.
- Never exposes API keys, authorization headers, or raw secrets in logs or errors.
"""

from typing import List, Dict, Any, Optional
import logging
import httpx

from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
)
from app.providers.image_base import (
    BaseImageProvider,
    ImageArtifact,
    ImageGenerationResult,
)
from app.config import settings

logger = logging.getLogger("orange_api.nvidia_image")


class NVIDIAImageProvider(BaseImageProvider):
    """
    NVIDIA GenAI image provider implementation for FLUX.2-klein-4B.
    """

    name: str = "nvidia"
    default_model: str = "black-forest-labs/flux.2-klein-4b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key or settings.NVIDIA_API_KEY
        self.base_url = (base_url or settings.NVIDIA_GENAI_BASE_URL).rstrip("/")
        self.model = model or self.default_model
        self.timeout = timeout if timeout is not None else settings.IMAGE_GENERATION_TIMEOUT

    def is_available(self) -> bool:
        """
        Returns True if the provider is configured with credentials.
        Does NOT check runtime rate limits or network connectivity.
        """
        return bool(self.api_key and self.api_key.strip())

    def _sanitize_message(self, message: str) -> str:
        """Strip raw API key if inadvertently embedded in upstream error strings."""
        if self.api_key and self.api_key in message:
            return message.replace(self.api_key, "***")
        return message

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
        Send image generation request to NVIDIA GenAI API and return ImageGenerationResult.
        Raises typed ProviderError subclasses on failures.
        Never exposes the API key in logs or error messages.
        """
        if not self.is_available():
            raise ProviderUnavailableError(
                "NVIDIA image provider is not configured (missing NVIDIA_API_KEY)."
            )

        # 1. Pre-request parameter validation
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            raise ProviderError("Image generation prompt cannot be empty.")

        if width != 1024 or height != 1024:
            raise ProviderError(
                f"Unsupported resolution ({width}x{height}). "
                "FLUX.2-klein-4B on NVIDIA hosted NIM currently supports only 1024x1024."
            )

        if not (1 <= steps <= 4):
            raise ProviderError(
                f"Invalid steps ({steps}). FLUX.2-klein-4B supports steps between 1 and 4."
            )

        if seed is not None and seed < 0:
            raise ProviderError(
                f"Invalid seed ({seed}). Seed must be a non-negative integer (>= 0)."
            )

        if samples < 1:
            raise ProviderError(
                f"Invalid samples count ({samples}). Samples must be at least 1."
            )

        target_model = kwargs.get("model") or self.model
        endpoint = f"{self.base_url}/{target_model.lstrip('/')}"

        payload: Dict[str, Any] = {
            "prompt": clean_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "samples": samples,
        }
        if seed is not None:
            payload["seed"] = seed

        if "mode" in kwargs and kwargs["mode"]:
            payload["mode"] = kwargs["mode"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "OrangeAI-NVIDIAImageProvider/1.0",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload, headers=headers)

            status_code = response.status_code

            if status_code == 200:
                data = response.json()
                artifacts = data.get("artifacts")
                if not artifacts or not isinstance(artifacts, list):
                    raise ProviderError(
                        "NVIDIA GenAI API returned empty or missing artifacts in response."
                    )

                parsed_artifacts: List[ImageArtifact] = []
                for idx, art in enumerate(artifacts):
                    if not isinstance(art, dict) or not art.get("base64"):
                        raise ProviderError(
                            f"NVIDIA GenAI API response artifact at index {idx} is missing base64 image data."
                        )
                    parsed_artifacts.append(
                        ImageArtifact(
                            base64=art["base64"],
                            seed=art.get("seed"),
                            mime_type="image/png",
                            width=width,
                            height=height,
                        )
                    )

                return ImageGenerationResult(
                    artifacts=parsed_artifacts,
                    model=target_model,
                    provider=self.name,
                    metadata={
                        "prompt": clean_prompt,
                        "steps": steps,
                        "samples": samples,
                    },
                )

            # Parse error details safely
            err_msg = ""
            try:
                err_data = response.json()
                if isinstance(err_data, dict):
                    err_msg = err_data.get("error", {}).get("message") or str(err_data.get("error", err_data))
                else:
                    err_msg = str(err_data)
            except Exception:
                err_msg = response.text[:200]

            err_msg = self._sanitize_message(err_msg)
            logger.warning("NVIDIA GenAI API returned HTTP %s: %s", status_code, err_msg)

            if status_code == 400:
                raise ProviderError(f"NVIDIA GenAI API invalid request (HTTP 400): {err_msg}")
            if status_code in (401, 403):
                raise ProviderUnavailableError(
                    f"NVIDIA authentication failed (HTTP {status_code}): {err_msg}"
                )
            if status_code == 422:
                raise ProviderError(f"NVIDIA GenAI API unprocessable entity (HTTP 422): {err_msg}")
            if status_code == 429:
                raise ProviderRateLimitError(f"NVIDIA rate limit or quota exceeded: {err_msg}")
            if status_code in (500, 502, 503):
                raise ProviderUnavailableError(
                    f"NVIDIA image service temporarily unavailable (HTTP {status_code}): {err_msg}"
                )
            if status_code == 504:
                raise ProviderTimeoutError(f"NVIDIA image generation gateway timeout (HTTP 504): {err_msg}")

            raise ProviderError(f"NVIDIA GenAI API error (HTTP {status_code}): {err_msg}")

        except (httpx.ConnectError, httpx.NetworkError) as e:
            sanitized_e = self._sanitize_message(str(e))
            logger.warning("NVIDIA GenAI connection error: %s", sanitized_e)
            raise ProviderUnavailableError(f"Failed to connect to NVIDIA GenAI API: {sanitized_e}") from e

        except httpx.TimeoutException as e:
            sanitized_e = self._sanitize_message(str(e))
            logger.warning("NVIDIA GenAI request timed out: %s", sanitized_e)
            raise ProviderTimeoutError(f"NVIDIA image generation request timed out: {sanitized_e}") from e

        except (ProviderError, ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError):
            raise

        except Exception as e:
            sanitized_e = self._sanitize_message(str(e))
            logger.error("Unexpected error during NVIDIA image generation: %s", sanitized_e)
            raise ProviderError(f"NVIDIA image provider error: {sanitized_e}") from e
