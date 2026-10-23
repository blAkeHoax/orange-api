"""
nvidia_provider.py - NVIDIA OpenAI-compatible cloud provider for Orange AI.

Communicates with NVIDIA's OpenAI-compatible API endpoint:
https://integrate.api.nvidia.com/v1

Supports the four-model foundation:
- openai/gpt-oss-20b
- nvidia/nemotron-3.5-lightning-30b-a3b
- z-ai/glm-5.3-flash
- google/gemma-4-31b-it

Implements the BaseProvider contract and maps errors to typed exceptions:
- ProviderRateLimitError (HTTP 429)
- ProviderUnavailableError (HTTP 502/503, connection failures, auth errors)
- ProviderTimeoutError (HTTP 504, request timeouts)
- ProviderError (general API errors)
"""

from typing import List, Dict, Any, Optional
import logging
import re
import time
import httpx

from app.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
)
from app.config import settings
from app.services.model_registry import model_exists, get_model
from app.utils.latency import log_latency

logger = logging.getLogger("orange_api.nvidia")


class NVIDIAProvider(BaseProvider):
    """
    NVIDIA model provider implementation using NVIDIA's OpenAI-compatible API.
    Handles communication with NVIDIA-hosted models.
    """

    name: str = "nvidia"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.NVIDIA_API_KEY
        self.base_url = (base_url or settings.NVIDIA_BASE_URL).rstrip("/")
        self.model = model or settings.DEFAULT_NVIDIA_MODEL
        self.timeout = timeout if timeout is not None else settings.NVIDIA_INFERENCE_TIMEOUT

    def is_available(self) -> bool:
        """
        Returns True if the provider is configured with credentials.
        Does NOT check runtime rate limits or network connectivity.
        """
        return bool(self.api_key and self.api_key.strip())

    def _sanitize_message(self, message: str) -> str:
        """Strip raw API key, bearer tokens, or sensitive patterns from error strings."""
        if not message:
            return ""
        msg = str(message)
        if self.api_key and self.api_key in msg:
            msg = msg.replace(self.api_key, "***")
        msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***", msg)
        msg = re.sub(r"nv" + r"api-[A-Za-z0-9_\-]+", "***", msg)
        msg = re.sub(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[REDACTED]", msg)
        return msg

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send request to NVIDIA OpenAI-compatible API and return response text.
        Supports text-only conversational assistance.
        Raises typed ProviderError subclasses on failures.
        Never exposes the API key in logs or error messages.
        """
        if not self.is_available():
            raise ProviderUnavailableError("NVIDIA provider is not configured (missing NVIDIA_API_KEY).")

        # Resolve model: kwargs override -> instance model -> default model
        target_model = kwargs.get("model") or self.model

        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if history:
            for turn in history:
                role = turn.get("role", "user")
                if role in ("assistant", "model"):
                    role = "assistant"
                elif role != "system":
                    role = "user"
                messages.append({
                    "role": role,
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OrangeAI-NVIDIAProvider/1.0",
        }

        endpoint = f"{self.base_url}/chat/completions"
        request_timeout = kwargs.get("timeout", self.timeout)

        http_start = time.perf_counter()
        try:
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(endpoint, json=payload, headers=headers)

            http_duration_ms = (time.perf_counter() - http_start) * 1000.0
            status_code = response.status_code

            log_latency(
                stage="nvidia_provider",
                duration_ms=http_duration_ms,
                model=target_model,
                http_status=status_code,
                prompt_len=len(prompt),
                history_turns=len(history or []),
            )
            # Track TTFT diagnostic limitation: blocking synchronous HTTP client
            log_latency(
                stage="ttft",
                status="unavailable",
                reason="non_streaming_provider",
                note="NVIDIAProvider uses blocking chat/completions without streaming chunks",
            )

            if status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ProviderError("NVIDIA API returned empty choices in response.")
                first_choice = choices[0]
                message_content = first_choice.get("message", {}).get("content")
                if message_content is None:
                    finish_reason = first_choice.get("finish_reason")
                    if finish_reason == "length":
                        raise ProviderError(
                            "NVIDIA model exceeded the output token budget before producing response content."
                        )
                    raise ProviderError("NVIDIA API response did not contain assistant message content.")
                return message_content

            # Parse error details safely without leaking credentials or raw payload
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
            logger.warning("NVIDIA API returned HTTP %s: %s", status_code, err_msg)

            if status_code == 429:
                raise ProviderRateLimitError(f"NVIDIA rate limit or quota exceeded: {err_msg}")
            if status_code in (401, 403):
                raise ProviderUnavailableError(f"NVIDIA authentication failed (HTTP {status_code}): {err_msg}")
            if status_code in (502, 503):
                raise ProviderUnavailableError(f"NVIDIA service temporarily unavailable (HTTP {status_code}): {err_msg}")
            if status_code == 504:
                raise ProviderTimeoutError(f"NVIDIA gateway timeout (HTTP 504): {err_msg}")

            raise ProviderError(f"NVIDIA API error (HTTP {status_code}): {err_msg}")

        except (httpx.ConnectError, httpx.NetworkError) as e:
            clean_e = self._sanitize_message(str(e))
            logger.warning("NVIDIA connection error: %s", clean_e)
            raise ProviderUnavailableError(f"Failed to connect to NVIDIA API: {clean_e}") from e

        except httpx.TimeoutException as e:
            http_duration_ms = (time.perf_counter() - http_start) * 1000.0
            log_latency(
                stage="nvidia_provider_timeout",
                duration_ms=http_duration_ms,
                model=target_model,
            )
            clean_e = self._sanitize_message(str(e))
            logger.warning("NVIDIA request timed out: %s", clean_e)
            raise ProviderTimeoutError(f"NVIDIA request timed out: {clean_e}") from e

        except (ProviderError, ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError):
            raise

        except Exception as e:
            clean_e = self._sanitize_message(str(e))
            logger.error("Unexpected error during NVIDIA invocation: %s", clean_e)
            raise ProviderError(f"NVIDIA provider error: {clean_e}") from e
