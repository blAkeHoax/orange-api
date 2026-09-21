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
    print("4. Testing four-model foundation IDs...")
    expected_ids = {
        "gpt_oss_20b": "openai/gpt-oss-20b",
        "nemotron_3_5_lightning_30b_a3b": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "glm_5_3_flash": "z-ai/glm-5.3-flash",
        "gemma_4_31b_it": "google/gemma-4-31b-it",
    }
    for key, expected_id in expected_ids.items():
        model_info = get_model(key)
        assert model_info.provider_model_id == expected_id, (
            f"Expected {expected_id} for {key}, got {model_info.provider_model_id}"
        )
    print("  Four-model foundation IDs verified [OK]")


def test_5_all_four_models_in_registry():
    print("5. Testing all four models exist in registry...")
    models = list_models()
    assert len(models) == 4
    keys = {m.key for m in models}
    assert keys == {
        "gpt_oss_20b",
        "nemotron_3_5_lightning_30b_a3b",
        "glm_5_3_flash",
        "gemma_4_31b_it",
    }
    # Check capabilities match Phase 6A specifications
    nemotron = get_model("nemotron_3_5_lightning_30b_a3b")
    assert set(nemotron.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic"}
    gemma = get_model("gemma_4_31b_it")
    assert set(gemma.capabilities) == {"text", "reasoning", "coding", "tool_use", "agentic", "vision", "video"}
    glm = get_model("glm_5_3_flash")
    assert set(glm.capabilities) == {"text", "reasoning", "tool_use", "agentic", "vision"}
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

    # 1. Default conversational routing (Phase 6B Intelligence Router selects gpt_oss_20b)
    reply, model_name = llm.generate_response(message="Hello")
    assert "Mock reply" in reply
    assert model_name == "nvidia:openai/gpt-oss-20b"

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

    # 5. Timeout configuration: uses settings.NVIDIA_INFERENCE_TIMEOUT (default 120.0) when no explicit timeout supplied
    assert prov.timeout == settings.NVIDIA_INFERENCE_TIMEOUT
    assert prov.timeout == 120.0

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
