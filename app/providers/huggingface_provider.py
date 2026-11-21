"""
huggingface_provider.py - Hugging Face Inference Providers adapter for Orange AI.

Communicates with Hugging Face Inference Providers via their OpenAI-compatible endpoint:
https://router.huggingface.co/v1

Supports high-performance open weights models such as:
- Qwen/Qwen3.8-27B:fastest
- openai/gpt-oss-120b:fastest

Implements the BaseProvider contract and maps errors to typed exceptions:
- ProviderRateLimitError (HTTP 429)
- ProviderUnavailableError (HTTP 401/403/502/503, connection failures, auth errors)
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
from app.utils.latency import log_latency
from app.utils.sanitizer import _sanitize_error

logger = logging.getLogger("orange_api.huggingface")


class HuggingFaceProvider(BaseProvider):
    """
    Hugging Face Inference Providers model provider implementation using
    Hugging Face's OpenAI-compatible router API (https://router.huggingface.co/v1).
    """

    name: str = "huggingface"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.get_hf_token()
        self.base_url = (base_url or settings.HUGGINGFACE_BASE_URL).rstrip("/")
        self.model = model or settings.DEFAULT_HUGGINGFACE_MODEL
        self.timeout = timeout if timeout is not None else settings.HUGGINGFACE_INFERENCE_TIMEOUT

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
        return _sanitize_error(msg)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send request to Hugging Face OpenAI-compatible API and return response text.
        Supports text-only conversational, coding, reasoning, and agentic assistance.
        Raises typed ProviderError subclasses on failures.
        Never exposes the API key in logs or error messages.
        """
        if not self.is_available():
            raise ProviderUnavailableError("Hugging Face provider is not configured (missing HF_TOKEN).")

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
            "User-Agent": "OrangeAI-HuggingFaceProvider/1.0",
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
                stage="huggingface_provider",
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
                note="HuggingFaceProvider uses blocking chat/completions without streaming chunks",
            )

            if status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ProviderError("Hugging Face API returned empty choices in response.")
                first_choice = choices[0]
                message_content = first_choice.get("message", {}).get("content")
                if message_content is None:
                    finish_reason = first_choice.get("finish_reason")
                    if finish_reason == "length":
                        raise ProviderError(
                            "Hugging Face model exceeded the output token budget before producing response content."
                        )
                    raise ProviderError("Hugging Face API response did not contain assistant message content.")
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
            logger.warning("Hugging Face API returned HTTP %s: %s", status_code, err_msg)

            if status_code == 429:
                raise ProviderRateLimitError(f"Hugging Face rate limit or quota exceeded: {err_msg}")
            if status_code in (401, 403):
                raise ProviderUnavailableError(f"Hugging Face authentication failed (HTTP {status_code}): {err_msg}")
            if status_code in (502, 503):
                raise ProviderUnavailableError(f"Hugging Face service temporarily unavailable (HTTP {status_code}): {err_msg}")
            if status_code == 504:
                raise ProviderTimeoutError(f"Hugging Face gateway timeout (HTTP 504): {err_msg}")

            raise ProviderError(f"Hugging Face API error (HTTP {status_code}): {err_msg}")

        except (httpx.ConnectError, httpx.NetworkError) as e:
            clean_e = self._sanitize_message(str(e))
            logger.warning("Hugging Face connection error: %s", clean_e)
            raise ProviderUnavailableError(f"Failed to connect to Hugging Face API: {clean_e}") from e

        except httpx.TimeoutException as e:
            http_duration_ms = (time.perf_counter() - http_start) * 1000.0
            log_latency(
                stage="huggingface_provider_timeout",
                duration_ms=http_duration_ms,
                model=target_model,
            )
            clean_e = self._sanitize_message(str(e))
            logger.warning("Hugging Face request timed out: %s", clean_e)
            raise ProviderTimeoutError(f"Hugging Face request timed out: {clean_e}") from e

        except (ProviderError, ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError):
            raise

        except Exception as e:
            clean_e = self._sanitize_message(str(e))
            logger.error("Unexpected error during Hugging Face invocation: %s", clean_e)
            raise ProviderError(f"Hugging Face provider error: {clean_e}") from e
