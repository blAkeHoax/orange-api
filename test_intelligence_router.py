"""
test_intelligence_router.py - Unit & Integration Test Suite for Phase 6B Intelligence Router.

Verifies:
1. Deterministic task classification via TaskProfileBuilder without LLM calls.
2. Modality normalization and strict validation.
3. Explicit task_type validation.
4. Capability and modality matching.
5. 7-step hierarchy routing decisions for all 6 task archetypes.
6. Explicit model overrides (valid, ineligible, unknown).
7. Error cases (NoEligibleModelError, IneligibleOverrideError, UnknownModelError, ValueError).
8. Deterministic tie-breaking across 1,000 iterations.
9. FastAPI POST /assist endpoint integration.
10. Desktop client decoupling and contract compliance.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add Orange API to sys.path
API_DIR = os.path.abspath(os.path.dirname(__file__))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from app.models.schemas import AssistRequest, AssistResponse
from app.models.task_profile import TaskType, TaskProfile, LatencyPreference
from app.services.task_profile_builder import (
    build_task_profile,
    normalize_modalities,
    _SUPPORTED_MODALITIES,
)
from app.services.intelligence_router import (
    IntelligenceRouter,
    RouteDecision,
    RoutingError,
    NoEligibleModelError,
    IneligibleOverrideError,
)
from app.services.model_registry import (
    get_model,
    list_available_models,
    model_exists,
    UnknownModelError,
)
from app.services.llm_service import LLMService


class TestTaskProfileBuilder(unittest.TestCase):
    """Test deterministic TaskProfile generation and validation."""

    def test_modality_normalization_default(self):
        self.assertEqual(normalize_modalities(None), frozenset({"text"}))
        self.assertEqual(normalize_modalities([]), frozenset({"text"}))

    def test_modality_normalization_valid(self):
        mods = normalize_modalities(["text", "vision"])
        self.assertEqual(mods, frozenset({"text", "vision"}))

        mods_video = normalize_modalities(["video"])
        self.assertEqual(mods_video, frozenset({"text", "video"}))

    def test_modality_normalization_invalid(self):
        with self.assertRaises(ValueError):
            normalize_modalities(["audio"])

        with self.assertRaises(ValueError):
            normalize_modalities(["text", "smell"])

    def test_task_type_explicit_valid(self):
        req = AssistRequest(message="hello", task_type="coding")
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.CODING)

        req_agent = AssistRequest(message="do task", task_type="agent_decision")
        profile_agent = build_task_profile(req_agent)
        self.assertEqual(profile_agent.task_type, TaskType.AGENT_DECISION)
        self.assertTrue(profile_agent.requires_structured_output)
        self.assertTrue({"tool_use", "agentic"}.issubset(profile_agent.required_capabilities))

    def test_task_type_explicit_invalid(self):
        req = AssistRequest(message="hello", task_type="quantum_teleportation")
        with self.assertRaises(ValueError):
            build_task_profile(req)

    def test_task_type_inferred_coding(self):
        req = AssistRequest(message="debug this Python code:\ndef add(a, b):\n    return a - b")
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.CODING)
        self.assertIn("coding", profile.required_capabilities)

    def test_task_type_inferred_coding_code_fence(self):
        req = AssistRequest(message="Check this:\n```\nprint('hello')\n```")
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.CODING)

    def test_task_type_inferred_vision(self):
        req = AssistRequest(message="Describe this picture", modalities=["vision"])
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.VISION_ANALYSIS)
        self.assertIn("vision", profile.required_capabilities)

    def test_task_type_inferred_video(self):
        req = AssistRequest(message="What happens in this clip?", modalities=["video"])
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.VIDEO_ANALYSIS)
        self.assertIn("video", profile.required_capabilities)

    def test_task_type_inferred_conversational(self):
        req = AssistRequest(message="Tell me a joke about bananas.")
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.CONVERSATIONAL)


class TestIntelligenceRouter(unittest.TestCase):
    """Test 7-step hierarchy routing and error handling."""

    def setUp(self):
        self.router = IntelligenceRouter()

    def test_conversational_routing_gpt_oss(self):
        req = AssistRequest(message="What is the capital of France?")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "gpt_oss_20b")
        self.assertEqual(decision.provider_model_id, "openai/gpt-oss-20b")
        self.assertIn("conversational", decision.reason.lower())

    def test_coding_routing_gemma(self):
        req = AssistRequest(message="debug this Python function:\ndef test(): pass")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "gemma_4_31b_it")
        self.assertEqual(decision.provider_model_id, "google/gemma-4-31b-it")

    def test_agent_decision_routing_nemotron(self):
        req = AssistRequest(message="Analyze state and decide tool", task_type="agent_decision")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "nemotron_3_5_lightning_30b_a3b")
        self.assertEqual(decision.provider_model_id, "nvidia/nemotron-3.5-lightning-30b-a3b")

    def test_job_step_routing_nemotron(self):
        req = AssistRequest(message="Execute step 3", task_type="job_step")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "nemotron_3_5_lightning_30b_a3b")
        self.assertEqual(decision.provider_model_id, "nvidia/nemotron-3.5-lightning-30b-a3b")

    def test_vision_routing_glm(self):
        req = AssistRequest(message="Inspect image", modalities=["vision"])
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "glm_5_3_flash")
        self.assertEqual(decision.provider_model_id, "z-ai/glm-5.3-flash")

    def test_video_routing_gemma(self):
        req = AssistRequest(message="Inspect clip", modalities=["video"])
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "gemma_4_31b_it")
        self.assertEqual(decision.provider_model_id, "google/gemma-4-31b-it")

    def test_valid_override_honored(self):
        # Conversational request overridden to Gemma
        req = AssistRequest(message="Hello there", model="gemma_4_31b_it")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "gemma_4_31b_it")
        self.assertEqual(decision.selection_step, "EXPLICIT_OVERRIDE")

    def test_ineligible_override_lacks_capability(self):
        # GPT-OSS lacks 'tool_use' and 'agentic' required for agent_decision
        req = AssistRequest(
            message="Plan next action",
            task_type="agent_decision",
            model="gpt_oss_20b",
        )
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError):
            self.router.route(profile)

    def test_ineligible_override_lacks_modality(self):
        # GPT-OSS lacks 'vision'
        req = AssistRequest(
            message="Describe photo",
            modalities=["vision"],
            model="gpt_oss_20b",
        )
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError):
            self.router.route(profile)

    def test_unknown_override_model(self):
        req = AssistRequest(
            message="Hello",
            model="non_existent_model_xyz",
        )
        profile = build_task_profile(req)
        with self.assertRaises(UnknownModelError):
            self.router.route(profile)

    def test_no_eligible_model_for_impossible_capabilities(self):
        impossible_profile = TaskProfile(
            task_type=TaskType.CONVERSATIONAL,
            required_capabilities=frozenset({"telepathy", "quantum_compute"}),
            modalities=frozenset({"text"}),
        )
        with self.assertRaises(NoEligibleModelError):
            self.router.route(impossible_profile)

    def test_deterministic_consistency_1000_runs(self):
        req = AssistRequest(message="Explain the theory of relativity in simple terms.")
        profile = build_task_profile(req)
        first_decision = self.router.route(profile)

        for _ in range(1000):
            dec = self.router.route(profile)
            self.assertEqual(dec.selected_model_key, first_decision.selected_model_key)
            self.assertEqual(dec.provider_model_id, first_decision.provider_model_id)
            self.assertEqual(dec.selection_step, first_decision.selection_step)
            self.assertEqual(dec.fallback_candidates, first_decision.fallback_candidates)

    def test_fallback_candidates_excludes_selected(self):
        req = AssistRequest(message="Debug code", task_type="coding")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertNotIn(decision.selected_model_key, decision.fallback_candidates)
        self.assertTrue(len(decision.fallback_candidates) > 0)


class TestAPIEndpointRouting(unittest.TestCase):
    """Test FastAPI /assist endpoint with task_type, modalities, and error mapping."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_conversational_defaults_to_gpt_oss(self, mock_generate):
        mock_generate.return_value = "Hello! I am ready to assist."

        resp = self.client.post("/assist", json={"message": "Good morning"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("openai/gpt-oss-20b", data["model"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_coding_routes_to_gemma(self, mock_generate):
        mock_generate.return_value = "Fixed the syntax error."

        resp = self.client.post("/assist", json={"message": "debug this python code: def foo(): return 1"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("google/gemma-4-31b-it", data["model"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_agent_decision_routes_to_nemotron(self, mock_generate):
        mock_generate.return_value = '{"action": {"type": "FINISH", "response": "Done"}}'

        resp = self.client.post(
            "/assist",
            json={"message": "State evaluation", "task_type": "agent_decision"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("nvidia/nemotron-3.5-lightning-30b-a3b", data["model"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_vision_routes_to_glm(self, mock_generate):
        mock_generate.return_value = "The image depicts a sunset over mountains."

        resp = self.client.post(
            "/assist",
            json={"message": "Analyze image", "modalities": ["vision"]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("z-ai/glm-5.3-flash", data["model"])

    def test_api_invalid_modality_returns_400(self):
        resp = self.client.post(
            "/assist",
            json={"message": "Analyze sound", "modalities": ["audio"]},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported modality", resp.json()["detail"])

    def test_api_invalid_task_type_returns_400(self):
        resp = self.client.post(
            "/assist",
            json={"message": "Run task", "task_type": "unknown_type"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported task_type", resp.json()["detail"])

    def test_api_ineligible_override_returns_400(self):
        resp = self.client.post(
            "/assist",
            json={
                "message": "Run agent step",
                "task_type": "agent_decision",
                "model": "gpt_oss_20b",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("lacks required capabilities", resp.json()["detail"])

    def test_api_unknown_model_returns_400(self):
        resp = self.client.post(
            "/assist",
            json={
                "message": "Say hello",
                "model": "fictional_model_999",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown model identifier", resp.json()["detail"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_default_system_instruction_when_omitted(self, mock_generate):
        mock_generate.return_value = "Hello! I am Orange AI."

        resp = self.client.post("/assist", json={"message": "Introduce yourself"})
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        self.assertEqual(
            kwargs.get("system_instruction"),
            "You are Orange AI, an intelligent, helpful, and concise personal AI assistant.",
        )

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_explicit_system_instruction_preserved(self, mock_generate):
        mock_generate.return_value = "Custom instruction output"
        custom_inst = "Custom instruction: output valid JSON only."

        resp = self.client.post(
            "/assist",
            json={
                "message": "Do task",
                "system_instruction": custom_inst,
            },
        )
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), custom_inst)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_empty_system_instruction_falls_back_to_default(self, mock_generate):
        mock_generate.return_value = "Fallback reply"

        resp = self.client.post(
            "/assist",
            json={
                "message": "Hi",
                "system_instruction": "   ",
            },
        )
        self.assertEqual(resp.status_code, 200)
        _, kwargs = mock_generate.call_args
        self.assertEqual(
            kwargs.get("system_instruction"),
            "You are Orange AI, an intelligent, helpful, and concise personal AI assistant.",
        )


class TestCORSConfiguration(unittest.TestCase):
    """Test CORS headers, preflight requests, and origin allowlist."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    def test_cors_production_frontend_preflight_allowed(self):
        resp = self.client.options(
            "/assist",
            headers={
                "Origin": "https://orange-ai-steel.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers.get("access-control-allow-origin"),
            "https://orange-ai-steel.vercel.app",
        )
        self.assertIn("POST", resp.headers.get("access-control-allow-methods", ""))

    def test_cors_local_vite_allowed(self):
        resp = self.client.options(
            "/assist",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_cors_local_127_vite_allowed(self):
        resp = self.client.options(
            "/assist",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers.get("access-control-allow-origin"),
            "http://127.0.0.1:5173",
        )

    def test_cors_unrelated_origin_not_allowed(self):
        resp = self.client.options(
            "/assist",
            headers={
                "Origin": "https://unauthorized-domain.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertNotEqual(
            resp.headers.get("access-control-allow-origin"),
            "https://unauthorized-domain.com",
        )

    def test_cors_health_endpoint_functional_with_and_without_cors(self):
        # 1. Without Origin header (e.g. desktop client)
        resp_no_origin = self.client.get("/health")
        self.assertEqual(resp_no_origin.status_code, 200)
        self.assertEqual(resp_no_origin.json(), {"status": "online"})
        self.assertNotIn("access-control-allow-origin", resp_no_origin.headers)

        # 2. With allowed Origin header (e.g. web frontend)
        resp_with_origin = self.client.get(
            "/health",
            headers={"Origin": "https://orange-ai-steel.vercel.app"},
        )
        self.assertEqual(resp_with_origin.status_code, 200)
        self.assertEqual(resp_with_origin.json(), {"status": "online"})
        self.assertEqual(
            resp_with_origin.headers.get("access-control-allow-origin"),
            "https://orange-ai-steel.vercel.app",
        )

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_cors_post_assist_functional(self, mock_generate):
        mock_generate.return_value = "Hello from Orange AI!"

        resp = self.client.post(
            "/assist",
            json={"message": "hey"},
            headers={"Origin": "https://orange-ai-steel.vercel.app"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["reply"], "Hello from Orange AI!")
        self.assertEqual(
            resp.headers.get("access-control-allow-origin"),
            "https://orange-ai-steel.vercel.app",
        )


class TestDesktopDecoupling(unittest.TestCase):
    """Ensure Desktop client is decoupled from router and provider implementations."""

    def test_desktop_zero_provider_and_router_imports(self):
        desktop_dir = os.path.abspath(os.path.join(API_DIR, "..", "Orange AI Desktop"))
        forbidden_tokens = [
            "IntelligenceRouter",
            "NVIDIAProvider",
            "from app.services.intelligence_router",
            "from app.providers",
        ]

        for root, dirs, files in os.walk(desktop_dir):
            if "venv" in root or "__pycache__" in root or ".git" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        for token in forbidden_tokens:
                            self.assertNotIn(
                                token,
                                content,
                                f"Desktop file {file} illegally imports/references '{token}'",
                            )

    def test_desktop_api_client_payload_includes_task_type_and_modalities(self):
        desktop_dir = os.path.abspath(os.path.join(API_DIR, "..", "Orange AI Desktop"))
        if desktop_dir not in sys.path:
            sys.path.insert(0, desktop_dir)
        from brain.api_client import OrangeAPIClient
        client = OrangeAPIClient(api_url="http://127.0.0.1:8000")

        # Mock urllib.request.urlopen to inspect sent payload
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b'{"reply": "ok", "model": "test", "success": true}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            client.assist(
                message="hello",
                task_type="agent_decision",
                modalities=["text", "vision"],
                model="glm_5_3_flash",
            )

            req = mock_urlopen.call_args[0][0]
            import json
            payload = json.loads(req.data.decode("utf-8"))
            self.assertEqual(payload.get("task_type"), "agent_decision")
            self.assertEqual(payload.get("modalities"), ["text", "vision"])
            self.assertEqual(payload.get("model"), "glm_5_3_flash")


if __name__ == "__main__":
    unittest.main()
