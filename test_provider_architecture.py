"""
test_provider_architecture.py - Verification suite for Orange API Provider Architecture (Phase 6A).

Tests:
1. NVIDIA provider imports correctly.
2. NVIDIA provider satisfies BaseProvider.
3. NVIDIA base URL is configured correctly.
4. NVIDIA model IDs are correct.
5. All four models exist in the registry.
6. Registry lookup by internal ID works.
7. Registry lookup by provider model ID works.
8. Unknown model produces a clean error.
9. Provider factory returns NVIDIAProvider.
10. No Gemini provider remains active.
11. No Ollama provider remains active.
12. No fallback provider is called.
13. Provider typed errors still behave correctly.
14. LLMService delegates correctly to NVIDIAProvider.
15. NVIDIA API key is never exposed through Desktop configuration.
16. AgentEngine has no direct NVIDIA dependency.
17. JobRuntime has no direct NVIDIA dependency.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure Orange API root is on sys.path
api_root = os.path.dirname(os.path.abspath(__file__))
if api_root not in sys.path:
    sys.path.insert(0, api_root)

# Desktop root for dependency isolation tests
desktop_root = os.path.join(os.path.dirname(api_root), "Orange AI Desktop")

from app.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
)
from app.providers.nvidia_provider import NVIDIAProvider
from app.providers.factory import get_provider, _PROVIDER_REGISTRY
from app.services.model_registry import (
    get_model,
    get_model_by_provider_id,
    list_models,
    list_available_models,
    model_exists,
    resolve_provider_model_id,
    UnknownModelError,
    ModelInfo,
)
from app.services.llm_service import LLMService
from app.services.model_router import ModelRouter
from app.config import settings


class MockFailingProvider(BaseProvider):
    name = "nvidia"
    model = "nvidia/nemotron-3.5-lightning-30b-a3b"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, **kwargs):
        raise ProviderRateLimitError("Simulated 429 quota exhaustion")


class MockSuccessProvider(BaseProvider):
    name = "nvidia"
    model = "nvidia/nemotron-3.5-lightning-30b-a3b"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, **kwargs):
        model = kwargs.get("model", self.model)
        return f"Mock reply to: '{prompt}' via {model}"


def test_1_nvidia_provider_imports():
    print("1. Testing NVIDIA provider imports...")
    from app.providers.nvidia_provider import NVIDIAProvider as P
    assert P is not None
    print("  NVIDIA provider imports passed [OK]")


def test_2_nvidia_provider_satisfies_base():
    print("2. Testing NVIDIA provider satisfies BaseProvider...")
    p = NVIDIAProvider(api_key="test-key")
    assert isinstance(p, BaseProvider)
    assert p.name == "nvidia"
    assert hasattr(p, "generate")
    assert callable(p.generate)
    assert hasattr(p, "is_available")
    assert callable(p.is_available)
    assert p.is_available() is True

    # Unconfigured provider reports is_available False
    p_empty = NVIDIAProvider(api_key="")
    assert p_empty.is_available() is False
    print("  NVIDIA provider satisfies BaseProvider [OK]")


def test_3_nvidia_base_url_configured():
    print("3. Testing NVIDIA base URL configuration...")
    p = NVIDIAProvider(api_key="test-key")
    assert p.base_url == "https://integrate.api.nvidia.com/v1"
    print("  NVIDIA base URL configuration passed [OK]")


def test_4_nvidia_model_ids_correct():
    print("4. Testing foundation model IDs...")
    expected_ids = {
        "gpt_oss_20b": "openai/gpt-oss-20b",
        "nemotron_3_5_lightning_30b_a3b": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia_coder": "minimaxai/minimax-m3.1-flash",
        "nvidia_reasoning": "moonshotai/kimi-k3.1-pro",
        "glm_5_3_flash": "z-ai/glm-5.3-flash",
        "gemma_4_31b_it": "google/gemma-4-31b-it",
        "qwen_3_8_27b": "Qwen/Qwen3.8-27B:fastest",
        "gpt_oss_120b": "openai/gpt-oss-120b:fastest",
    }
    for key, expected_id in expected_ids.items():
        model_info = get_model(key)
        assert model_info.provider_model_id == expected_id, (
            f"Expected {expected_id} for {key}, got {model_info.provider_model_id}"
        )
    print("  Foundation model IDs verified [OK]")


def test_5_all_four_models_in_registry():
    print("5. Testing models in registry and GLM retirement...")
    models = list_models()
    keys = {m.key for m in models}
    assert "gpt_oss_20b" in keys
    assert "nemotron_3_5_lightning_30b_a3b" in keys
    assert "nvidia_coder" in keys
    assert "nvidia_reasoning" in keys
    assert "gemma_4_31b_it" in keys
    assert "glm_5_3_flash" in keys
    assert "qwen_3_8_27b" in keys
    assert "gpt_oss_120b" in keys
    assert "sambanova_coder" in keys
    assert "sambanova_reasoning" in keys
    assert "sambanova_gemma_4_31b" in keys
    assert "sambanova_minimax_m3_1_flash" in keys
    assert "sambanova_glm_5_3" in keys
    assert "muse_glimmer_30b" not in keys

    # Check GLM is retired
    glm = get_model("glm_5_3_flash")
    assert glm.is_available is False, "Expected glm_5_3_flash to be retired (is_available=False)"

    # Check NVIDIA models
    assert settings.NVIDIA_CODER_MODEL == "minimaxai/minimax-m3.1-flash"
    assert settings.NVIDIA_REASONING_MODEL == "moonshotai/kimi-k3.1-pro"
    nv_coder = get_model("nvidia_coder")
    assert nv_coder.provider == "nvidia"
    assert nv_coder.provider_model_id == "minimaxai/minimax-m3.1-flash"
    assert nv_coder.is_available is True
    assert set(nv_coder.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    nv_reason = get_model("nvidia_reasoning")
    assert nv_reason.provider == "nvidia"
    assert nv_reason.provider_model_id == "moonshotai/kimi-k3.1-pro"
    assert nv_reason.is_available is True
    assert set(nv_reason.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    # Check Hugging Face models
    qwen = get_model("qwen_3_8_27b")
    assert qwen.provider == "huggingface"
    assert qwen.is_available is True
    assert set(qwen.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    gpt120 = get_model("gpt_oss_120b")
    assert gpt120.provider == "huggingface"
    assert gpt120.is_available is True
    assert set(gpt120.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    # Check SambaNova models
    assert settings.DEFAULT_SAMBANOVA_MODEL == "gpt-oss-120b"
    sb_coder = get_model("sambanova_coder")
    assert sb_coder.provider == "sambanova"
    assert sb_coder.provider_model_id == "gpt-oss-120b"
    assert sb_coder.is_available is True
    assert set(sb_coder.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    sb_reason = get_model("sambanova_reasoning")
    assert sb_reason.provider == "sambanova"
    assert sb_reason.provider_model_id == "Meta-Llama-3.3-70B-Instruct"
    assert sb_reason.is_available is True
    assert set(sb_reason.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    sb_gemma = get_model("sambanova_gemma_4_31b")
    assert sb_gemma.provider == "sambanova"
    assert sb_gemma.provider_model_id == "gemma-4-31B-it"
    assert sb_gemma.is_available is True
    assert set(sb_gemma.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    sb_minimax = get_model("sambanova_minimax_m3_1_flash")
    assert sb_minimax.provider == "sambanova"
    assert sb_minimax.provider_model_id == "MiniMax-M3.1-Flash"
    assert sb_minimax.is_available is True
    assert set(sb_minimax.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}

    sb_glm = get_model("sambanova_glm_5_3")
    assert sb_glm.provider == "sambanova"
    assert sb_glm.provider_model_id == "GLM-5.3"
    assert sb_glm.is_available is True
    assert set(sb_glm.capabilities) == {"text", "reasoning", "tool_use", "agentic"}

    # Check capabilities match text-only specifications (no vision)
    nemotron = get_model("nemotron_3_5_lightning_30b_a3b")
    assert set(nemotron.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}
    gemma = get_model("gemma_4_31b_it")
    assert set(gemma.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic", "video"}
    gpt = get_model("gpt_oss_20b")
    assert set(gpt.capabilities) == {"text", "reasoning", "coding"}
    print("  Registry model metadata verified [OK]")


def test_6_registry_lookup_by_internal_id():
    print("6. Testing registry lookup by internal key...")
    m = get_model("gemma_4_31b_it")
    assert m.display_name == "Gemma 4 31B IT"
    assert m.provider == "nvidia"
    assert m.provider_model_id == "google/gemma-4-31b-it"
    assert model_exists("gemma_4_31b_it") is True
    print("  Internal key lookup passed [OK]")


def test_7_registry_lookup_by_provider_id():
    print("7. Testing registry lookup by provider model ID...")
    m = get_model("google/gemma-4-31b-it")
    assert m.key == "gemma_4_31b_it"
    assert m.display_name == "Gemma 4 31B IT"
    m_direct = get_model_by_provider_id("google/gemma-4-31b-it")
    assert m_direct.key == "gemma_4_31b_it"
    assert model_exists("google/gemma-4-31b-it") is True
    print("  Provider model ID lookup passed [OK]")


def test_8_unknown_model_error():
    print("8. Testing unknown model rejection...")
    try:
        get_model("unknown_nonexistent_model")
        assert False, "Expected UnknownModelError"
    except UnknownModelError as e:
        assert "Unknown model identifier" in str(e)

    assert model_exists("unknown_nonexistent_model") is False
    print("  Unknown model rejection passed [OK]")


def test_9_provider_factory_returns_nvidia():
    print("9. Testing provider factory returns NVIDIAProvider...")
    p = get_provider("nvidia")
    assert isinstance(p, NVIDIAProvider)
    assert p.name == "nvidia"

    # Default provider should resolve to nvidia
    p_default = get_provider()
    assert isinstance(p_default, NVIDIAProvider)
    print("  Provider factory returns NVIDIAProvider [OK]")


def test_10_no_gemini_active():
    print("10. Testing no Gemini provider remains active...")
    assert "gemini" not in _PROVIDER_REGISTRY
    try:
        get_provider("gemini")
        assert False, "Expected ValueError when requesting gemini"
    except ValueError as e:
        assert "Unsupported AI provider: 'gemini'" in str(e)
    print("  No Gemini provider active verified [OK]")


def test_11_no_ollama_active():
    print("11. Testing no Ollama provider remains active...")
    assert "ollama" not in _PROVIDER_REGISTRY
    try:
        get_provider("ollama")
        assert False, "Expected ValueError when requesting ollama"
    except ValueError as e:
        assert "Unsupported AI provider: 'ollama'" in str(e)
    print("  No Ollama provider active verified [OK]")


def test_12_no_fallback_on_failure():
    print("12. Testing strict NO FALLBACK on provider failure...")
    failing = MockFailingProvider()
    llm = LLMService(provider=failing)
    try:
        llm.generate_response(message="Test query")
        assert False, "Expected ProviderRateLimitError"
    except ProviderRateLimitError as e:
        assert "Simulated 429 quota exhaustion" in str(e)

    router = ModelRouter(provider=failing)
    try:
        router.route_and_generate(prompt="Test query")
        assert False, "Expected ProviderRateLimitError"
    except ProviderRateLimitError as e:
        assert "Simulated 429 quota exhaustion" in str(e)
    print("  Strict NO FALLBACK passed [OK]")


def test_13_provider_typed_errors():
    print("13. Testing provider typed error hierarchy...")
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(ProviderUnavailableError, ProviderError)
    assert issubclass(ProviderTimeoutError, ProviderError)
    assert issubclass(ProviderError, Exception)
    print("  Typed error hierarchy passed [OK]")


def test_14_llm_service_delegation():
    print("14. Testing LLMService delegation to NVIDIAProvider...")
    mock_prov = MockSuccessProvider()
    llm = LLMService(provider=mock_prov)

    # 1. Default conversational routing (Phase 6B Intelligence Router selects nemotron_3_5_lightning_30b_a3b)
    reply, model_name = llm.generate_response(message="Hello")
    assert "Mock reply" in reply
    assert model_name == "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b"

    # 2. Explicit model key resolution via ModelRegistry
    reply2, model_name2 = llm.generate_response(message="Code something", model="nemotron_3_5_lightning_30b_a3b")
    assert "nvidia/nemotron-3.5-lightning-30b-a3b" in reply2
    assert model_name2 == "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b"

    # 3. ModelRouter compatibility
    router = ModelRouter(provider=mock_prov)
    r_reply, r_model = router.route_and_generate(prompt="Inspect", model="glm_5_3_flash")
    assert "z-ai/glm-5.3-flash" in r_reply
    assert r_model == "nvidia:z-ai/glm-5.3-flash"
    print("  LLMService delegation passed [OK]")


def test_15_desktop_no_nvidia_key():
    print("15. Testing Desktop configuration has zero provider keys...")
    desktop_env = os.path.join(desktop_root, ".env")
    if os.path.exists(desktop_env):
        with open(desktop_env, "r", encoding="utf-8") as f:
            content = f.read()
            assert "NVIDIA_API_KEY" not in content
            assert "GEMINI_API_KEY" not in content
    print("  Desktop zero provider keys verified [OK]")


def test_16_agent_engine_no_nvidia_dependency():
    print("16. Testing AgentEngine has zero direct NVIDIA dependencies...")
    agent_file = os.path.join(desktop_root, "brain", "agent.py")
    with open(agent_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "NVIDIAProvider" not in content
        assert "nvidia_provider" not in content
        assert "integrate.api.nvidia.com" not in content
    print("  AgentEngine provider independence verified [OK]")


def test_17_job_runtime_no_nvidia_dependency():
    print("17. Testing JobRuntime has zero direct NVIDIA dependencies...")
    runtime_file = os.path.join(desktop_root, "brain", "jobs", "runtime.py")
    with open(runtime_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "NVIDIAProvider" not in content
        assert "nvidia_provider" not in content
        assert "integrate.api.nvidia.com" not in content
    print("  JobRuntime provider independence verified [OK]")


def test_nvidia_provider_mocked_generate():
    print("Testing NVIDIAProvider mocked HTTP generation & error mapping...")
    prov = NVIDIAProvider(api_key="test-api-key")

    # Mock 200 OK response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Generated reply from NVIDIA"}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = prov.generate("Hello NVIDIA")
        assert res == "Generated reply from NVIDIA"

    # Mock 429 Rate limit response
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.json.return_value = {"error": {"message": "Rate limit exceeded (429)"}}
    with patch("httpx.Client.post", return_value=mock_429):
        try:
            prov.generate("Rate limit test")
            assert False, "Expected ProviderRateLimitError"
        except ProviderRateLimitError as e:
            assert "Rate limit exceeded" in str(e)

    # Mock 503 Unavailable response
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.json.return_value = {"error": {"message": "Server temporarily overloaded"}}
    with patch("httpx.Client.post", return_value=mock_503):
        try:
            prov.generate("Overload test")
            assert False, "Expected ProviderUnavailableError"
        except ProviderUnavailableError as e:
            assert "temporarily unavailable" in str(e)

    # Mock 504 Timeout response
    mock_504 = MagicMock()
    mock_504.status_code = 504
    mock_504.json.return_value = {"error": {"message": "Gateway Timeout"}}
    with patch("httpx.Client.post", return_value=mock_504):
        try:
            prov.generate("Timeout test")
            assert False, "Expected ProviderTimeoutError"
        except ProviderTimeoutError as e:
            assert "gateway timeout" in str(e)

    print("  NVIDIAProvider mocked HTTP generation passed [OK]")


def test_18_nvidia_provider_response_hardening_and_timeout():
    print("18. Testing NVIDIAProvider response hardening & configurable timeout...")
    prov = NVIDIAProvider(api_key="test-api-key")

    # 5. Timeout configuration: uses settings.NVIDIA_INFERENCE_TIMEOUT (default 85.0) when no explicit timeout supplied
    assert prov.timeout == settings.NVIDIA_INFERENCE_TIMEOUT
    assert prov.timeout == 85.0

    # Explicit timeout override wins if provided
    prov_custom = NVIDIAProvider(api_key="test-api-key", timeout=45.0)
    assert prov_custom.timeout == 45.0

    # 1. Normal content response
    mock_normal = MagicMock()
    mock_normal.status_code = 200
    mock_normal.json.return_value = {
        "choices": [{"message": {"content": "Generated reply"}}]
    }
    with patch("httpx.Client.post", return_value=mock_normal):
        res = prov.generate("Hello normal")
        assert res == "Generated reply"

    # 2. content=None with finish_reason="length"
    mock_length = MagicMock()
    mock_length.status_code = 200
    mock_length.json.return_value = {
        "choices": [
            {
                "message": {"content": None},
                "finish_reason": "length",
            }
        ]
    }
    with patch("httpx.Client.post", return_value=mock_length):
        try:
            prov.generate("Length test")
            assert False, "Expected ProviderError for content=None with finish_reason='length'"
        except ProviderError as e:
            assert "exceeded the output token budget" in str(e)

    # 3. content=None without finish_reason="length"
    mock_none = MagicMock()
    mock_none.status_code = 200
    mock_none.json.return_value = {
        "choices": [
            {
                "message": {"content": None},
                "finish_reason": "stop",
            }
        ]
    }
    with patch("httpx.Client.post", return_value=mock_none):
        try:
            prov.generate("None test")
            assert False, "Expected ProviderError for content=None"
        except ProviderError as e:
            assert "did not contain assistant message content" in str(e)

    # 4. reasoning_content is present but content is None
    secret_reasoning = "private reasoning text"
    mock_reasoning = MagicMock()
    mock_reasoning.status_code = 200
    mock_reasoning.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": secret_reasoning,
                },
                "finish_reason": "length",
            }
        ]
    }
    with patch("httpx.Client.post", return_value=mock_reasoning):
        try:
            prov.generate("Reasoning leakage test")
            assert False, "Expected ProviderError when reasoning_content is present but content is None"
        except ProviderError as e:
            # Verify reasoning_content is NOT returned and NOT exposed in error message
            assert secret_reasoning not in str(e)

    print("  NVIDIAProvider response hardening & configurable timeout passed [OK]")


def test_19_huggingface_provider_imports():
    print("19. Testing Hugging Face provider imports...")
    from app.providers.huggingface_provider import HuggingFaceProvider as HFP
    assert HFP is not None
    print("  Hugging Face provider imports passed [OK]")


def test_20_huggingface_provider_satisfies_base():
    print("20. Testing Hugging Face provider satisfies BaseProvider...")
    from app.providers.huggingface_provider import HuggingFaceProvider
    p = HuggingFaceProvider(api_key="test-hf-key")
    assert isinstance(p, BaseProvider)
    assert p.name == "huggingface"
    assert hasattr(p, "generate") and callable(p.generate)
    assert hasattr(p, "is_available") and callable(p.is_available)
    assert p.is_available() is True

    p_empty = HuggingFaceProvider(api_key="")
    assert p_empty.is_available() is False
    print("  Hugging Face provider satisfies BaseProvider [OK]")


def test_21_huggingface_base_url_and_timeout():
    print("21. Testing Hugging Face base URL and timeout defaults...")
    from app.providers.huggingface_provider import HuggingFaceProvider
    p = HuggingFaceProvider(api_key="test-key")
    assert p.base_url == "https://router.huggingface.co/v1"
    assert p.timeout == settings.HUGGINGFACE_INFERENCE_TIMEOUT
    assert p.timeout == 85.0
    print("  Hugging Face base URL and timeout verified [OK]")


def test_22_provider_factory_returns_huggingface():
    print("22. Testing provider factory returns HuggingFaceProvider...")
    from app.providers.huggingface_provider import HuggingFaceProvider
    p = get_provider("huggingface")
    assert isinstance(p, HuggingFaceProvider)
    assert p.name == "huggingface"
    assert "huggingface" in _PROVIDER_REGISTRY
    assert "nvidia" in _PROVIDER_REGISTRY
    print("  Provider factory returns HuggingFaceProvider [OK]")


def test_23_huggingface_mocked_generate_and_errors():
    print("23. Testing HuggingFaceProvider mocked generation and error mapping...")
    from app.providers.huggingface_provider import HuggingFaceProvider
    prov = HuggingFaceProvider(api_key="test-hf-token")

    # Mock 200 Success
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Generated reply from Hugging Face"}}]
    }
    with patch("httpx.Client.post", return_value=mock_resp):
        res = prov.generate("Hello HF")
        assert res == "Generated reply from Hugging Face"

    # Mock 429 Rate Limit
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.json.return_value = {"error": {"message": "Rate limit exceeded (429)"}}
    with patch("httpx.Client.post", return_value=mock_429):
        try:
            prov.generate("Rate limit HF")
            assert False, "Expected ProviderRateLimitError"
        except ProviderRateLimitError as e:
            assert "rate limit or quota exceeded" in str(e).lower()

    # Mock 503 Service Unavailable
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.json.return_value = {"error": {"message": "HF engine overloaded"}}
    with patch("httpx.Client.post", return_value=mock_503):
        try:
            prov.generate("Overload HF")
            assert False, "Expected ProviderUnavailableError"
        except ProviderUnavailableError as e:
            assert "temporarily unavailable" in str(e)

    # Mock 504 Gateway Timeout
    mock_504 = MagicMock()
    mock_504.status_code = 504
    mock_504.json.return_value = {"error": {"message": "Gateway Timeout"}}
    with patch("httpx.Client.post", return_value=mock_504):
        try:
            prov.generate("Timeout HF")
            assert False, "Expected ProviderTimeoutError"
        except ProviderTimeoutError as e:
            assert "gateway timeout" in str(e)

    # Mock TimeoutException
    import httpx
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
        try:
            prov.generate("Timeout exception probe")
            assert False, "Expected ProviderTimeoutError"
        except ProviderTimeoutError as e:
            assert "request timed out" in str(e)

    print("  HuggingFaceProvider mocked generation and error mapping passed [OK]")


def test_24_sambanova_provider_imports():
    print("24. Testing SambaNova provider imports...")
    from app.providers.sambanova_provider import SambaNovaProvider as SBP
    assert SBP is not None
    print("  SambaNova provider imports passed [OK]")


def test_25_sambanova_provider_satisfies_base():
    print("25. Testing SambaNova provider satisfies BaseProvider...")
    from app.providers.sambanova_provider import SambaNovaProvider
    p = SambaNovaProvider(api_key="test-sb-key")
    assert isinstance(p, BaseProvider)
    assert p.name == "sambanova"
    assert hasattr(p, "generate") and callable(p.generate)
    assert hasattr(p, "is_available") and callable(p.is_available)
    assert p.is_available() is True

    p_empty = SambaNovaProvider(api_key="")
    assert p_empty.is_available() is False
    print("  SambaNova provider satisfies BaseProvider [OK]")


def test_26_sambanova_base_url_and_timeout():
    print("26. Testing SambaNova base URL and timeout defaults...")
    from app.providers.sambanova_provider import SambaNovaProvider
    p = SambaNovaProvider(api_key="test-key")
    assert p.base_url == "https://api.sambanova.ai/v1"
    assert p.timeout == settings.SAMBANOVA_INFERENCE_TIMEOUT
    assert p.timeout == 85.0
    print("  SambaNova base URL and timeout verified [OK]")


def test_27_provider_factory_returns_sambanova():
    print("27. Testing provider factory returns SambaNovaProvider...")
    from app.providers.sambanova_provider import SambaNovaProvider
    p = get_provider("sambanova")
    assert isinstance(p, SambaNovaProvider)
    assert p.name == "sambanova"
    assert "sambanova" in _PROVIDER_REGISTRY
    assert "huggingface" in _PROVIDER_REGISTRY
    assert "nvidia" in _PROVIDER_REGISTRY
    print("  Provider factory returns SambaNovaProvider [OK]")


def test_28_sambanova_mocked_generate_and_errors():
    print("28. Testing SambaNovaProvider mocked generation and error mapping...")
    from app.providers.sambanova_provider import SambaNovaProvider
    prov = SambaNovaProvider(api_key="test-sb-token")

    # Mock 200 Success
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Generated reply from SambaNova"}}]
    }
    with patch("httpx.Client.post", return_value=mock_resp):
        res = prov.generate("Hello SambaNova")
        assert res == "Generated reply from SambaNova"

    # Mock 429 Rate Limit
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.json.return_value = {"error": {"message": "Rate limit reached"}}
    with patch("httpx.Client.post", return_value=mock_429):
        try:
            prov.generate("Rate limit SB")
            assert False, "Expected ProviderRateLimitError"
        except ProviderRateLimitError as e:
            assert "rate limit or quota exceeded" in str(e).lower()

    # Mock 401 Auth Error
    mock_401 = MagicMock()
    mock_401.status_code = 401
    mock_401.json.return_value = {"error": {"message": "Invalid API key"}}
    with patch("httpx.Client.post", return_value=mock_401):
        try:
            prov.generate("Auth fail SB")
            assert False, "Expected ProviderUnavailableError"
        except ProviderUnavailableError as e:
            assert "authentication failed" in str(e)

    # Mock 503 Service Unavailable
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.json.return_value = {"error": {"message": "SambaNova engine overloaded"}}
    with patch("httpx.Client.post", return_value=mock_503):
        try:
            prov.generate("Overload SB")
            assert False, "Expected ProviderUnavailableError"
        except ProviderUnavailableError as e:
            assert "temporarily unavailable" in str(e)

    # Mock 504 Gateway Timeout
    mock_504 = MagicMock()
    mock_504.status_code = 504
    mock_504.json.return_value = {"error": {"message": "Gateway Timeout"}}
    with patch("httpx.Client.post", return_value=mock_504):
        try:
            prov.generate("Timeout SB")
            assert False, "Expected ProviderTimeoutError"
        except ProviderTimeoutError as e:
            assert "gateway timeout" in str(e)

    # Mock TimeoutException
    import httpx
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
        try:
            prov.generate("Timeout exception probe")
            assert False, "Expected ProviderTimeoutError"
        except ProviderTimeoutError as e:
            assert "request timed out" in str(e)

    print("  SambaNovaProvider mocked generation and error mapping passed [OK]")


class ProviderArchitectureTestCase(unittest.TestCase):
    def test_all_provider_architecture(self):
        test_1_nvidia_provider_imports()
        test_2_nvidia_provider_satisfies_base()
        test_3_nvidia_base_url_configured()
        test_4_nvidia_model_ids_correct()
        test_5_all_four_models_in_registry()
        test_6_registry_lookup_by_internal_id()
        test_7_registry_lookup_by_provider_id()
        test_8_unknown_model_error()
        test_9_provider_factory_returns_nvidia()
        test_10_no_gemini_active()
        test_11_no_ollama_active()
        test_12_no_fallback_on_failure()
        test_13_provider_typed_errors()
        test_14_llm_service_delegation()
        test_15_desktop_no_nvidia_key()
        test_16_agent_engine_no_nvidia_dependency()
        test_17_job_runtime_no_nvidia_dependency()
        test_18_nvidia_provider_response_hardening_and_timeout()
        test_nvidia_provider_mocked_generate()
        test_19_huggingface_provider_imports()
        test_20_huggingface_provider_satisfies_base()
        test_21_huggingface_base_url_and_timeout()
        test_22_provider_factory_returns_huggingface()
        test_23_huggingface_mocked_generate_and_errors()
        test_24_sambanova_provider_imports()
        test_25_sambanova_provider_satisfies_base()
        test_26_sambanova_base_url_and_timeout()
        test_27_provider_factory_returns_sambanova()
        test_28_sambanova_mocked_generate_and_errors()


if __name__ == "__main__":
    print("=" * 60)
    print("ORANGE API: PROVIDER ARCHITECTURE VERIFICATION TESTS (PHASE 6A)")
    print("=" * 60)
    test_1_nvidia_provider_imports()
    test_2_nvidia_provider_satisfies_base()
    test_3_nvidia_base_url_configured()
    test_4_nvidia_model_ids_correct()
    test_5_all_four_models_in_registry()
    test_6_registry_lookup_by_internal_id()
    test_7_registry_lookup_by_provider_id()
    test_8_unknown_model_error()
    test_9_provider_factory_returns_nvidia()
    test_10_no_gemini_active()
    test_11_no_ollama_active()
    test_12_no_fallback_on_failure()
    test_13_provider_typed_errors()
    test_14_llm_service_delegation()
    test_15_desktop_no_nvidia_key()
    test_16_agent_engine_no_nvidia_dependency()
    test_17_job_runtime_no_nvidia_dependency()
    test_18_nvidia_provider_response_hardening_and_timeout()
    test_nvidia_provider_mocked_generate()
    print("=" * 60)
    print("ALL 18+ PROVIDER ARCHITECTURE TESTS PASSED [OK]")
    print("=" * 60)
