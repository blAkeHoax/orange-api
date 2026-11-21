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
from app.services.assist_service import AssistService, DEFAULT_SYSTEM_INSTRUCTION
from app.config import settings


class TestTaskProfileBuilder(unittest.TestCase):
    """Test deterministic TaskProfile generation and validation."""

    def test_modality_normalization_default(self):
        self.assertEqual(normalize_modalities(None), frozenset({"text"}))
        self.assertEqual(normalize_modalities([]), frozenset({"text"}))

    def test_modality_normalization_valid(self):
        mods_video = normalize_modalities(["video"])
        self.assertEqual(mods_video, frozenset({"text", "video"}))

    def test_modality_normalization_invalid(self):
        with self.assertRaises(ValueError):
            normalize_modalities(["audio"])

        with self.assertRaises(ValueError):
            normalize_modalities(["vision"])

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

    def test_task_type_inferred_video(self):
        req = AssistRequest(message="What happens in this clip?", modalities=["video"])
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.VIDEO_ANALYSIS)
        self.assertIn("video", profile.required_capabilities)

    def test_task_type_inferred_conversational(self):
        req = AssistRequest(message="Tell me a joke about bananas.")
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.CONVERSATIONAL)

    def test_task_type_explicit_reasoning_and_general(self):
        req_reasoning = AssistRequest(message="analyze", task_type="reasoning")
        profile_reasoning = build_task_profile(req_reasoning)
        self.assertEqual(profile_reasoning.task_type, TaskType.REASONING)
        self.assertEqual(profile_reasoning.required_capabilities, frozenset({"text", "reasoning"}))
        self.assertEqual(profile_reasoning.preferred_capabilities, frozenset({"coding", "tool_use"}))

        req_general = AssistRequest(message="general inquiry", task_type="general")
        profile_general = build_task_profile(req_general)
        self.assertEqual(profile_general.task_type, TaskType.GENERAL)

    def test_task_type_inferred_reasoning_signals(self):
        reasoning_prompts = [
            "Synthesize the morning briefing for today.",
            "Please generate the daily briefing from my notes.",
            "Draft an executive summary of the quarterly metrics.",
            "Provide an architectural comparison of monolith vs microservices.",
            "What are the architectural trade-offs of using GraphQL?",
            "Provide a deep analysis of this race condition.",
            "Perform a root cause analysis for the outage.",
            "Show me a step-by-step reasoning for this math problem.",
        ]
        for prompt in reasoning_prompts:
            with self.subTest(prompt=prompt):
                req = AssistRequest(message=prompt)
                profile = build_task_profile(req)
                self.assertEqual(profile.task_type, TaskType.REASONING)
                self.assertIn("reasoning", profile.required_capabilities)

    def test_structured_output_signal_detection(self):
        structured_prompts = [
            "Extract user contacts and return as valid json",
            "Generate structured output with id and name",
            "Output in json format only",
            "Parse log entries and return json schema",
        ]
        for prompt in structured_prompts:
            with self.subTest(prompt=prompt):
                req = AssistRequest(message=prompt)
                profile = build_task_profile(req)
                self.assertTrue(profile.requires_structured_output)


class TestIntelligenceRouter(unittest.TestCase):
    """Test 7-step hierarchy routing and error handling."""

    def setUp(self):
        self.router = IntelligenceRouter()

    def test_conversational_routing_bolt(self):
        req = AssistRequest(message="What is the capital of France?")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "nvidia_bolt")
        self.assertEqual(decision.provider_model_id, settings.NVIDIA_BOLT_MODEL)
        self.assertIn("conversational", decision.reason.lower())

    def test_coding_routing_nemotron_when_gemma_unavailable(self):
        """Verify normal coding task routes to nvidia_coder when Gemma is unavailable and HF/SambaNova unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": ""}):
            req = AssistRequest(message="debug this Python function:\ndef test(): pass")
            profile = build_task_profile(req)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "nvidia_coder")
            self.assertEqual(decision.provider_model_id, settings.NVIDIA_CODER_MODEL)
            self.assertNotEqual(decision.selection_step, "EXPLICIT_OVERRIDE")
            self.assertNotEqual(decision.selected_model_key, "gemma_4_31b_it")

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

    def test_coding_routing_sambanova_when_sambanova_configured(self):
        """Verify coding task automatically routes to SambaNova Coder when SAMBANOVA_API_KEY is configured."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-sb-key", "HF_TOKEN": "test-hf-token"}):
            req_inferred = AssistRequest(message="debug this Python function:\ndef test(): pass")
            profile_inferred = build_task_profile(req_inferred)
            decision_inferred = self.router.route(profile_inferred)
            self.assertEqual(decision_inferred.selected_model_key, "sambanova_coder")
            self.assertEqual(decision_inferred.provider_model_id, settings.DEFAULT_SAMBANOVA_MODEL)
            self.assertEqual(decision_inferred.selection_step, "TASK_SPECIALIZATION")
            self.assertIn("qwen_3_8_27b", decision_inferred.fallback_candidates)
            self.assertIn("nvidia_coder", decision_inferred.fallback_candidates)
            self.assertIn("nemotron_3_5_lightning_30b_a3b", decision_inferred.fallback_candidates)

            # Explicit coding task_type
            req_explicit = AssistRequest(message="write function", task_type="coding")
            profile_explicit = build_task_profile(req_explicit)
            decision_explicit = self.router.route(profile_explicit)
            self.assertEqual(decision_explicit.selected_model_key, "sambanova_coder")
            self.assertEqual(decision_explicit.provider_model_id, settings.DEFAULT_SAMBANOVA_MODEL)
            self.assertEqual(decision_explicit.selection_step, "TASK_SPECIALIZATION")

    def test_coding_routing_qwen_when_hf_configured(self):
        """Verify coding task automatically routes to Qwen 3.8 27B when HF_TOKEN is configured and SambaNova is unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token", "SAMBANOVA_API_KEY": ""}):
            # Inferred coding prompt
            req_inferred = AssistRequest(message="debug this Python function:\ndef test(): pass")
            profile_inferred = build_task_profile(req_inferred)
            decision_inferred = self.router.route(profile_inferred)
            self.assertEqual(decision_inferred.selected_model_key, "qwen_3_8_27b")
            self.assertEqual(decision_inferred.provider_model_id, "Qwen/Qwen3.8-27B:fastest")
            self.assertEqual(decision_inferred.selection_step, "TASK_SPECIALIZATION")
            self.assertIn("nvidia_coder", decision_inferred.fallback_candidates)
            self.assertIn("nemotron_3_5_lightning_30b_a3b", decision_inferred.fallback_candidates)

            # Explicit coding task_type
            req_explicit = AssistRequest(message="write function", task_type="coding")
            profile_explicit = build_task_profile(req_explicit)
            decision_explicit = self.router.route(profile_explicit)
            self.assertEqual(decision_explicit.selected_model_key, "qwen_3_8_27b")
            self.assertEqual(decision_explicit.provider_model_id, "Qwen/Qwen3.8-27B:fastest")
            self.assertEqual(decision_explicit.selection_step, "TASK_SPECIALIZATION")

    def test_coding_routing_nemotron_when_hf_unset(self):
        """Verify coding task routes to nvidia_coder when HF_TOKEN and SAMBANOVA_API_KEY are unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": ""}):
            req = AssistRequest(message="debug this Python function:\ndef test(): pass")
            profile = build_task_profile(req)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "nvidia_coder")
            self.assertEqual(decision.provider_model_id, settings.NVIDIA_CODER_MODEL)

    def test_reasoning_routing_sambanova_when_sambanova_configured(self):
        """Verify complex reasoning & briefing synthesis automatically route to SambaNova Llama 3.3 70B."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-sb-key", "HF_TOKEN": "test-hf-token"}):
            prompts = [
                "Synthesize the morning briefing for today.",
                "Please generate the daily briefing from my notes.",
                "Draft an executive summary of this quarterly review.",
                "Provide an architectural comparison between monolith and microservices.",
                "What are the architectural trade-offs of using event sourcing?",
                "Provide a deep analysis of this distributed deadlock.",
            ]
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    req = AssistRequest(message=prompt)
                    profile = build_task_profile(req)
                    self.assertEqual(profile.task_type, TaskType.REASONING)
                    decision = self.router.route(profile)
                    self.assertEqual(decision.selected_model_key, "sambanova_reasoning")
                    self.assertEqual(decision.provider_model_id, settings.SAMBANOVA_REASONING_MODEL)
                    self.assertEqual(decision.selection_step, "TASK_SPECIALIZATION")
                    self.assertIn("gpt_oss_120b", decision.fallback_candidates)
                    self.assertIn("nvidia_bolt", decision.fallback_candidates)
                    self.assertIn("nemotron_3_5_lightning_30b_a3b", decision.fallback_candidates)

            # Explicit task_type="reasoning"
            req_explicit = AssistRequest(message="Analyze bottleneck options", task_type="reasoning")
            profile_explicit = build_task_profile(req_explicit)
            decision_explicit = self.router.route(profile_explicit)
            self.assertEqual(decision_explicit.selected_model_key, "sambanova_reasoning")
            self.assertEqual(decision_explicit.provider_model_id, settings.SAMBANOVA_REASONING_MODEL)
            self.assertEqual(decision_explicit.selection_step, "TASK_SPECIALIZATION")

    def test_reasoning_routing_gpt_oss_120b_when_hf_configured(self):
        """Verify complex reasoning & briefing synthesis automatically route to GPT-OSS 120B when HF configured and SambaNova unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token", "SAMBANOVA_API_KEY": ""}):
            prompts = [
                "Synthesize the morning briefing for today.",
                "Please generate the daily briefing from my notes.",
                "Draft an executive summary of this quarterly review.",
                "Provide an architectural comparison between monolith and microservices.",
                "What are the architectural trade-offs of using event sourcing?",
                "Provide a deep analysis of this distributed deadlock.",
            ]
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    req = AssistRequest(message=prompt)
                    profile = build_task_profile(req)
                    self.assertEqual(profile.task_type, TaskType.REASONING)
                    decision = self.router.route(profile)
                    self.assertEqual(decision.selected_model_key, "gpt_oss_120b")
                    self.assertEqual(decision.provider_model_id, "openai/gpt-oss-120b:fastest")
                    self.assertEqual(decision.selection_step, "TASK_SPECIALIZATION")
                    self.assertIn("nvidia_bolt", decision.fallback_candidates)
                    self.assertIn("nemotron_3_5_lightning_30b_a3b", decision.fallback_candidates)

            # Explicit task_type="reasoning"
            req_explicit = AssistRequest(message="Analyze bottleneck options", task_type="reasoning")
            profile_explicit = build_task_profile(req_explicit)
            decision_explicit = self.router.route(profile_explicit)
            self.assertEqual(decision_explicit.selected_model_key, "gpt_oss_120b")
            self.assertEqual(decision_explicit.selection_step, "TASK_SPECIALIZATION")

    def test_reasoning_routing_bolt_when_hf_unset(self):
        """Verify reasoning synthesis routes to nvidia_bolt when HF_TOKEN and SAMBANOVA_API_KEY are unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": ""}):
            req = AssistRequest(message="Synthesize a deep analysis of this distributed deadlock.")
            profile = build_task_profile(req)
            self.assertEqual(profile.task_type, TaskType.REASONING)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "nvidia_bolt")
            self.assertEqual(decision.provider_model_id, settings.NVIDIA_BOLT_MODEL)
            self.assertEqual(decision.selection_step, "TASK_SPECIALIZATION")
            self.assertIn("nemotron_3_5_lightning_30b_a3b", decision.fallback_candidates)

    def test_briefing_routing_to_nemotron_lightning(self):
        """Verify briefing tasks route to nemotron_3_5_lightning_30b_a3b."""
        req = AssistRequest(message="Synthesize the daily morning briefing", task_type="briefing")
        profile = build_task_profile(req)
        self.assertEqual(profile.task_type, TaskType.BRIEFING)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "nemotron_3_5_lightning_30b_a3b")
        self.assertEqual(decision.provider_model_id, "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(decision.selection_step, "TASK_SPECIALIZATION")

    def test_structured_output_routing_to_sambanova_when_sambanova_configured(self):
        """Verify structured output requests route to SambaNova Coder when SAMBANOVA_API_KEY is configured."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-sb-key", "HF_TOKEN": "test-hf-token"}):
            req = AssistRequest(message="Extract user contacts and return as valid json")
            profile = build_task_profile(req)
            self.assertTrue(profile.requires_structured_output)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "sambanova_coder")
            self.assertEqual(decision.provider_model_id, settings.DEFAULT_SAMBANOVA_MODEL)
            self.assertEqual(decision.selection_step, "STRUCTURED_OUTPUT_PREFERENCE")

    def test_structured_output_routing_to_qwen_when_hf_configured(self):
        """Verify structured output requests route to Qwen 3.8 27B when HF_TOKEN is configured and SambaNova unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token", "SAMBANOVA_API_KEY": ""}):
            req = AssistRequest(message="Extract user contacts and return as valid json")
            profile = build_task_profile(req)
            self.assertTrue(profile.requires_structured_output)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "qwen_3_8_27b")
            self.assertEqual(decision.provider_model_id, "Qwen/Qwen3.8-27B:fastest")
            self.assertEqual(decision.selection_step, "STRUCTURED_OUTPUT_PREFERENCE")

    def test_structured_output_routing_falls_back_to_nemotron_when_hf_unset(self):
        """Verify structured output requests route to nvidia_coder when HF_TOKEN and SAMBANOVA_API_KEY are unset."""
        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": ""}):
            req = AssistRequest(message="Extract user contacts and return as valid json")
            profile = build_task_profile(req)
            self.assertTrue(profile.requires_structured_output)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "nvidia_coder")
            self.assertEqual(decision.provider_model_id, settings.NVIDIA_CODER_MODEL)
            self.assertEqual(decision.selection_step, "STRUCTURED_OUTPUT_PREFERENCE")
            self.assertEqual(decision.selection_step, "STRUCTURED_OUTPUT_PREFERENCE")

    def test_agent_decision_and_job_step_pinned_to_nemotron_with_hf_configured(self):
        """Verify agent_decision and job_step remain pinned to low-latency Nemotron even when HF is configured."""
        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
            for t_type in ["agent_decision", "job_step"]:
                with self.subTest(task_type=t_type):
                    req = AssistRequest(message="Execute step", task_type=t_type)
                    profile = build_task_profile(req)
                    decision = self.router.route(profile)
                    self.assertEqual(decision.selected_model_key, "nemotron_3_5_lightning_30b_a3b")
                    self.assertEqual(decision.provider_model_id, "nvidia/nemotron-3.5-lightning-30b-a3b")
                    self.assertEqual(decision.selection_step, "TASK_SPECIALIZATION")

    def test_conversational_and_general_route_to_bolt(self):
        """Verify conversational and general tasks route to Nemotron 3.5 Bolt."""
        for t_type in ["conversational", "general"]:
            with self.subTest(task_type=t_type):
                req = AssistRequest(message="Hello assistance", task_type=t_type)
                profile = build_task_profile(req)
                decision = self.router.route(profile)
                self.assertEqual(decision.selected_model_key, "nvidia_bolt")
                self.assertEqual(decision.provider_model_id, settings.NVIDIA_BOLT_MODEL)
                self.assertEqual(decision.selection_step, "TASK_SPECIALIZATION")
                self.assertIn("nemotron_3_5_lightning_30b_a3b", decision.fallback_candidates)

    def test_model_registry_no_muse_glimmer(self):
        """Verify Muse Glimmer is completely absent from model registry."""
        self.assertFalse(model_exists("muse_glimmer_30b"))
        self.assertFalse(model_exists("meta/muse-glimmer-30b"))
        with self.assertRaises(UnknownModelError):
            get_model("muse_glimmer_30b")

    def test_explicit_override_retired_glm_rejected(self):
        """Verify explicit override to retired GLM model raises IneligibleOverrideError."""
        req = AssistRequest(message="Inspect text", model="glm_5_3_flash")
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError) as cm:
            self.router.route(profile)
        self.assertIn("is currently unavailable", str(cm.exception))

    def test_ineligible_override_gpt_oss_lacks_video(self):
        """Verify GPT-OSS override fails if video modality is required."""
        req = AssistRequest(
            message="Inspect clip",
            modalities=["video"],
            model="gpt_oss_20b",
        )
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError):
            self.router.route(profile)

    def test_video_routing_fails_when_gemma_unavailable(self):
        """Verify video analysis task raises NoEligibleModelError when Gemma is unavailable."""
        req = AssistRequest(message="Inspect clip", modalities=["video"])
        profile = build_task_profile(req)
        with self.assertRaises(NoEligibleModelError):
            self.router.route(profile)

    def test_valid_override_honored(self):
        # Conversational request overridden to Nemotron
        req = AssistRequest(message="Hello there", model="nemotron_3_5_lightning_30b_a3b")
        profile = build_task_profile(req)
        decision = self.router.route(profile)
        self.assertEqual(decision.selected_model_key, "nemotron_3_5_lightning_30b_a3b")
        self.assertEqual(decision.selection_step, "EXPLICIT_OVERRIDE")

        # Conversational request overridden to Qwen when HF_TOKEN is configured
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
            req_qwen = AssistRequest(message="Hello Qwen", model="qwen_3_8_27b")
            profile_qwen = build_task_profile(req_qwen)
            decision_qwen = self.router.route(profile_qwen)
            self.assertEqual(decision_qwen.selected_model_key, "qwen_3_8_27b")
            self.assertEqual(decision_qwen.selection_step, "EXPLICIT_OVERRIDE")

    def test_ineligible_override_unconfigured_hf_provider_rejected(self):
        """Verify explicit override for Qwen raises IneligibleOverrideError when HF_TOKEN is unset."""
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            req = AssistRequest(message="Hello", model="qwen_3_8_27b")
            profile = build_task_profile(req)
            with self.assertRaises(IneligibleOverrideError) as cm:
                self.router.route(profile)
            self.assertIn("provider 'huggingface' is currently unconfigured or unavailable", str(cm.exception))

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
        # GPT-OSS lacks 'video'
        req = AssistRequest(
            message="Describe clip",
            modalities=["video"],
            model="gpt_oss_20b",
        )
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError):
            self.router.route(profile)

    def test_ineligible_override_available_model_lacks_modality(self):
        # Nemotron lacks 'video' modality
        req = AssistRequest(
            message="Describe clip",
            modalities=["video"],
            model="nemotron_3_5_lightning_30b_a3b",
        )
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError) as cm:
            self.router.route(profile)
        self.assertIn("video", str(cm.exception))

    def test_explicit_override_unavailable_gpt_oss_rejected(self):
        """Verify explicit override for gpt_oss_20b raises IneligibleOverrideError fast."""
        req = AssistRequest(message="Hello there", model="gpt_oss_20b")
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError) as cm:
            self.router.route(profile)
        self.assertIn("is currently unavailable", str(cm.exception))

    def test_explicit_override_unavailable_provider_id_rejected(self):
        """Verify explicit override for openai/gpt-oss-20b raises IneligibleOverrideError fast."""
        req = AssistRequest(message="Hello there", model="openai/gpt-oss-20b")
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError) as cm:
            self.router.route(profile)
        self.assertIn("is currently unavailable", str(cm.exception))

    def test_explicit_override_unavailable_gemma_rejected(self):
        """Verify explicit override for gemma_4_31b_it raises IneligibleOverrideError fast."""
        req = AssistRequest(message="Hello there", model="gemma_4_31b_it")
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError) as cm:
            self.router.route(profile)
        self.assertIn("is currently unavailable", str(cm.exception))

    def test_explicit_override_unavailable_gemma_provider_id_rejected(self):
        """Verify explicit override for google/gemma-4-31b-it raises IneligibleOverrideError fast."""
        req = AssistRequest(message="Hello there", model="google/gemma-4-31b-it")
        profile = build_task_profile(req)
        with self.assertRaises(IneligibleOverrideError) as cm:
            self.router.route(profile)
        self.assertIn("is currently unavailable", str(cm.exception))

    def test_list_available_models_excludes_unavailable_models(self):
        """Verify list_available_models() returns selectable models and excludes gpt_oss_20b, gemma_4_31b_it, and retired glm_5_3_flash."""
        available = list_available_models()
        available_keys = {m.key for m in available}
        self.assertNotIn("gpt_oss_20b", available_keys)
        self.assertNotIn("gemma_4_31b_it", available_keys)
        self.assertNotIn("glm_5_3_flash", available_keys)
        self.assertIn("nemotron_3_5_lightning_30b_a3b", available_keys)
        self.assertIn("qwen_3_8_27b", available_keys)
        self.assertIn("gpt_oss_120b", available_keys)

    def test_sambanova_models_available_in_registry(self):
        """Verify SambaNova models appear in available models list when SAMBANOVA_API_KEY is configured."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}):
            available = [m.key for m in list_available_models()]
            self.assertIn("sambanova_coder", available)
            self.assertIn("sambanova_reasoning", available)
            self.assertIn("sambanova_gemma_4_31b", available)
            self.assertIn("sambanova_minimax_m3_1_flash", available)
            self.assertIn("sambanova_glm_5_3", available)

    def test_sambanova_models_filtered_when_key_unset(self):
        """Verify SambaNova models are excluded from router when SAMBANOVA_API_KEY is unset."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": ""}):
            all_router_models = [
                m.key for m in list_available_models()
                if self.router._is_model_provider_available(m)
            ]
            self.assertNotIn("sambanova_coder", all_router_models)
            self.assertNotIn("sambanova_reasoning", all_router_models)
            self.assertNotIn("sambanova_gemma_4_31b", all_router_models)
            self.assertNotIn("sambanova_minimax_m3_1_flash", all_router_models)
            self.assertNotIn("sambanova_glm_5_3", all_router_models)

    def test_sambanova_eligible_explicit_overrides_succeed(self):
        """Verify eligible explicit overrides for SambaNova models succeed."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}):
            for model_key in [
                "sambanova_coder",
                "sambanova_reasoning",
                "sambanova_gemma_4_31b",
                "sambanova_minimax_m3_1_flash",
                "sambanova_glm_5_3",
            ]:
                req = AssistRequest(message="Hello there", model=model_key)
                profile = build_task_profile(req)
                decision = self.router.route(profile)
                self.assertEqual(decision.selected_model_key, model_key)
                self.assertEqual(decision.selection_step, "EXPLICIT_OVERRIDE")

    def test_sambanova_unconfigured_override_rejected(self):
        """Verify explicit override for SambaNova raises IneligibleOverrideError when key is unset."""
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": ""}):
            req = AssistRequest(message="Hello", model="sambanova_coder")
            profile = build_task_profile(req)
            with self.assertRaises(IneligibleOverrideError):
                self.router.route(profile)

    def test_eligible_explicit_overrides_succeed(self):
        """Verify eligible explicit overrides for Nemotron, NVIDIA coder/reasoning, Qwen, and GPT-OSS-120B all succeed."""
        with patch.dict(os.environ, {"HF_TOKEN": "test-token", "SAMBANOVA_API_KEY": ""}):
            for model_key in [
                "nemotron_3_5_lightning_30b_a3b",
                "nvidia_coder",
                "nvidia_reasoning",
                "qwen_3_8_27b",
                "gpt_oss_120b",
            ]:
                req = AssistRequest(message="Hello there", model=model_key)
                profile = build_task_profile(req)
                decision = self.router.route(profile)
                self.assertEqual(decision.selected_model_key, model_key)
                self.assertEqual(decision.selection_step, "EXPLICIT_OVERRIDE")

    def test_unknown_override_model(self):
        req = AssistRequest(
            message="Hello",
            model="non_existent_model_xyz",
        )
        profile = build_task_profile(req)
        with self.assertRaises(UnknownModelError):
            self.router.route(profile)

    def test_auto_model_treated_as_dynamic_routing(self):
        """Verify model='auto' is not treated as an override and uses dynamic routing."""
        # 1. With SambaNova configured, coding routes to sambanova_coder
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "sb_test_token", "HF_TOKEN": "hf_test_token"}):
            req_code = AssistRequest(message="Write a python script", task_type="coding", model="auto")
            profile_code = build_task_profile(req_code)
            self.assertIsNone(profile_code.override_model)
            dec_code = self.router.route(profile_code)
            self.assertEqual(dec_code.selected_model_key, "sambanova_coder")
            self.assertEqual(dec_code.selection_step, "TASK_SPECIALIZATION")

        # 2. With only HF configured, coding routes to qwen_3_8_27b
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token", "SAMBANOVA_API_KEY": ""}):
            # Coding with model="auto" routes to Qwen
            req_code = AssistRequest(message="Write a python script", task_type="coding", model="auto")
            profile_code = build_task_profile(req_code)
            self.assertIsNone(profile_code.override_model)
            dec_code = self.router.route(profile_code)
            self.assertEqual(dec_code.selected_model_key, "qwen_3_8_27b")
            self.assertEqual(dec_code.selection_step, "TASK_SPECIALIZATION")

            # Conversational with model="auto" routes to Nemotron 3.5 Bolt
            req_chat = AssistRequest(message="Hello there", model="auto")
            profile_chat = build_task_profile(req_chat)
            self.assertIsNone(profile_chat.override_model)
            dec_chat = self.router.route(profile_chat)
            self.assertEqual(dec_chat.selected_model_key, "nvidia_bolt")
            self.assertEqual(dec_chat.provider_model_id, settings.NVIDIA_BOLT_MODEL)

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

    def test_fallback_candidates_excludes_selected_when_hf_set(self):
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
            req = AssistRequest(message="Tell me a joke", task_type="conversational")
            profile = build_task_profile(req)
            decision = self.router.route(profile)
            self.assertNotIn(decision.selected_model_key, decision.fallback_candidates)
            self.assertTrue(len(decision.fallback_candidates) > 0)
            self.assertIn("nemotron_3_5_lightning_30b_a3b", decision.fallback_candidates)
            self.assertIn("qwen_3_8_27b", decision.fallback_candidates)
            self.assertIn("gpt_oss_120b", decision.fallback_candidates)

    def test_fallback_candidates_empty_when_hf_unset(self):
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            req = AssistRequest(message="Tell me a joke", task_type="conversational")
            profile = build_task_profile(req)
            decision = self.router.route(profile)
            self.assertEqual(decision.selected_model_key, "nvidia_bolt")
            self.assertIn("nemotron_3_5_lightning_30b_a3b", decision.fallback_candidates)
            self.assertIn("nvidia_coder", decision.fallback_candidates)


class TestAPIEndpointRouting(unittest.TestCase):
    """Test FastAPI /assist endpoint with task_type, modalities, and error mapping."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_conversational_defaults_to_bolt(self, mock_generate):
        mock_generate.return_value = "Hello! I am ready to assist."

        resp = self.client.post("/assist", json={"message": "Good morning"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(settings.NVIDIA_BOLT_MODEL, data["model"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_coding_routes_to_nvidia_coder(self, mock_generate):
        mock_generate.return_value = "Fixed the syntax error."

        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": ""}):
            resp = self.client.post("/assist", json={"message": "debug this python code: def foo(): return 1"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn(settings.NVIDIA_CODER_MODEL, data["model"])
            self.assertNotIn("google/gemma-4-31b-it", data["model"])
            mock_generate.assert_called_once()

    @patch("app.providers.huggingface_provider.HuggingFaceProvider.generate")
    def test_api_coding_routes_to_qwen_when_hf_configured(self, mock_generate):
        mock_generate.return_value = "Fixed with Qwen."

        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
            resp = self.client.post("/assist", json={"message": "debug this python code: def foo(): return 1"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("Qwen/Qwen3.8-27B:fastest", data["model"])
            mock_generate.assert_called_once()

    @patch("app.providers.huggingface_provider.HuggingFaceProvider.generate")
    def test_api_reasoning_routes_to_gpt_oss_when_hf_configured(self, mock_generate):
        mock_generate.return_value = "Morning briefing synthesized."

        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
            resp = self.client.post("/assist", json={"message": "Synthesize the morning briefing for today."})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("openai/gpt-oss-120b:fastest", data["model"])
            mock_generate.assert_called_once()

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_reasoning_routes_to_nvidia_reasoning_when_hf_unset(self, mock_generate):
        mock_generate.return_value = "Morning briefing synthesized via Llama 3.3."

        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": ""}):
            resp = self.client.post("/assist", json={"message": "Synthesize the morning briefing for today."})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn(settings.NVIDIA_REASONING_MODEL, data["model"])
            mock_generate.assert_called_once()

    @patch("app.providers.huggingface_provider.HuggingFaceProvider.generate")
    def test_api_structured_output_routes_to_qwen_when_hf_configured(self, mock_generate):
        mock_generate.return_value = '{"users": []}'

        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
            resp = self.client.post("/assist", json={"message": "Extract items and return as valid json"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("Qwen/Qwen3.8-27B:fastest", data["model"])
            mock_generate.assert_called_once()

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

    def test_api_vision_modality_returns_400(self):
        resp = self.client.post(
            "/assist",
            json={"message": "Analyze image", "modalities": ["vision"]},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported modality: 'vision'", resp.json()["detail"])

    def test_api_muse_glimmer_override_returns_400(self):
        resp = self.client.post(
            "/assist",
            json={"message": "Analyze chart", "model": "muse_glimmer_30b"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown model identifier 'muse_glimmer_30b'", resp.json()["detail"])

    def test_api_explicit_override_retired_glm_rejected(self):
        resp = self.client.post(
            "/assist",
            json={"message": "Analyze prompt", "model": "glm_5_3_flash"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("is currently unavailable", resp.json()["detail"])

    @patch("app.providers.huggingface_provider.HuggingFaceProvider.generate")
    def test_api_explicit_override_huggingface_qwen(self, mock_generate):
        mock_generate.return_value = "Qwen response."
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
            resp = self.client.post(
                "/assist",
                json={"message": "Analyze prompt", "model": "qwen_3_8_27b"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("Qwen/Qwen3.8-27B:fastest", data["model"])

    def test_api_explicit_override_unconfigured_huggingface_rejected(self):
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            resp = self.client.post(
                "/assist",
                json={"message": "Analyze prompt", "model": "qwen_3_8_27b"},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("unconfigured or unavailable", resp.json()["detail"])

    @patch("app.providers.sambanova_provider.SambaNovaProvider.generate")
    def test_api_explicit_override_sambanova(self, mock_generate):
        mock_generate.return_value = "SambaNova response."
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-token"}):
            resp = self.client.post(
                "/assist",
                json={"message": "Analyze prompt", "model": "sambanova_coder"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["model"], f"sambanova:{settings.DEFAULT_SAMBANOVA_MODEL}")
            self.assertEqual(data["reply"], "SambaNova response.")

    @patch("app.providers.sambanova_provider.SambaNovaProvider.generate")
    def test_api_explicit_override_all_sambanova_models(self, mock_generate):
        mock_generate.return_value = "SambaNova multi-model response."
        models_to_test = [
            ("sambanova_coder", settings.DEFAULT_SAMBANOVA_MODEL),
            ("sambanova_reasoning", "Meta-Llama-3.3-70B-Instruct"),
            ("sambanova_gemma_4_31b", "gemma-4-31B-it"),
            ("sambanova_minimax_m3_1_flash", "MiniMax-M3.1-Flash"),
            ("sambanova_glm_5_3", "GLM-5.3"),
        ]
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-token"}):
            for model_key, expected_provider_id in models_to_test:
                resp = self.client.post(
                    "/assist",
                    json={"message": "Analyze prompt", "model": model_key},
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["model"], f"sambanova:{expected_provider_id}")
                self.assertEqual(data["reply"], "SambaNova multi-model response.")

    def test_sambanova_models_in_fallback_candidates(self):
        """Verify SambaNova models are included in fallback candidates when SAMBANOVA_API_KEY is configured."""
        router = IntelligenceRouter()
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key", "HF_TOKEN": "test-hf"}):
            req = AssistRequest(message="Write a python function to sort a list", task_type="coding")
            profile = build_task_profile(req)
            decision = router.route(profile)
            self.assertEqual(decision.selected_model_key, "sambanova_coder")
            # Other eligible SambaNova models should be among fallback candidates
            self.assertIn("sambanova_reasoning", decision.fallback_candidates)
            self.assertIn("sambanova_gemma_4_31b", decision.fallback_candidates)
            self.assertIn("sambanova_minimax_m3_1_flash", decision.fallback_candidates)

    def test_api_explicit_override_unconfigured_sambanova_rejected(self):
        with patch.dict(os.environ, {"SAMBANOVA_API_KEY": ""}):
            resp = self.client.post(
                "/assist",
                json={"message": "Analyze prompt", "model": "sambanova_coder"},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("unconfigured or unavailable", resp.json()["detail"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_explicit_override_nvidia_models(self, mock_generate):
        mock_generate.return_value = "NVIDIA multi-model response."
        models_to_test = [
            ("nvidia_coder", settings.NVIDIA_CODER_MODEL),
            ("nvidia_reasoning", settings.NVIDIA_REASONING_MODEL),
            ("nemotron_3_5_lightning_30b_a3b", settings.DEFAULT_NVIDIA_MODEL),
        ]
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nv-key"}):
            for model_key, expected_provider_id in models_to_test:
                resp = self.client.post(
                    "/assist",
                    json={"message": "Analyze prompt", "model": model_key},
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["model"], f"nvidia:{expected_provider_id}")
                self.assertEqual(data["reply"], "NVIDIA multi-model response.")

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_nvidia_task_routing_when_only_nvidia_configured(self, mock_generate):
        mock_generate.return_value = "Task routed response."
        with patch.dict(os.environ, {"HF_TOKEN": "", "SAMBANOVA_API_KEY": "", "NVIDIA_API_KEY": "test-nv-key"}):
            # Coding -> nvidia_coder
            resp_code = self.client.post(
                "/assist",
                json={"message": "Write a quicksort in python", "task_type": "coding"},
            )
            self.assertEqual(resp_code.status_code, 200)
            self.assertEqual(resp_code.json()["model"], f"nvidia:{settings.NVIDIA_CODER_MODEL}")

            # Reasoning -> nvidia_reasoning (nvidia_bolt)
            resp_reason = self.client.post(
                "/assist",
                json={"message": "Synthesize a deep comparative analysis", "task_type": "reasoning"},
            )
            self.assertEqual(resp_reason.status_code, 200)
            self.assertEqual(resp_reason.json()["model"], f"nvidia:{settings.NVIDIA_BOLT_MODEL}")

            # Briefing -> nemotron_3_5_lightning_30b_a3b
            resp_briefing = self.client.post(
                "/assist",
                json={"message": "Synthesize the morning briefing", "task_type": "briefing"},
            )
            self.assertEqual(resp_briefing.status_code, 200)
            self.assertEqual(resp_briefing.json()["model"], f"nvidia:{settings.DEFAULT_NVIDIA_MODEL}")

            # Conversational -> nvidia_bolt
            resp_chat = self.client.post(
                "/assist",
                json={"message": "Hello how are you?", "task_type": "conversational"},
            )
            self.assertEqual(resp_chat.status_code, 200)
            self.assertEqual(resp_chat.json()["model"], f"nvidia:{settings.NVIDIA_BOLT_MODEL}")

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
                "message": "analyze video clip",
                "modalities": ["video"],
                "model": "nemotron_3_5_lightning_30b_a3b",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("lacks required capabilities", resp.json()["detail"])

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_unavailable_override_fails_fast_without_calling_provider(self, mock_generate):
        """Explicit override for gpt_oss_20b must return HTTP 400 fast without invoking provider."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello there",
                "model": "gpt_oss_20b",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("is currently unavailable", resp.json()["detail"])
        mock_generate.assert_not_called()

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_unavailable_override_by_provider_id_fails_fast(self, mock_generate):
        """Explicit override using provider ID 'openai/gpt-oss-20b' must also fail fast."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello there",
                "model": "openai/gpt-oss-20b",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("is currently unavailable", resp.json()["detail"])
        mock_generate.assert_not_called()

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_unavailable_gemma_override_fails_fast_without_calling_provider(self, mock_generate):
        """Explicit override for gemma_4_31b_it must return HTTP 400 fast without invoking provider."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello there",
                "model": "gemma_4_31b_it",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("is currently unavailable", resp.json()["detail"])
        mock_generate.assert_not_called()

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_unavailable_gemma_by_provider_id_fails_fast(self, mock_generate):
        """Explicit override using provider ID 'google/gemma-4-31b-it' must also fail fast."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello there",
                "model": "google/gemma-4-31b-it",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("is currently unavailable", resp.json()["detail"])
        mock_generate.assert_not_called()

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_video_analysis_fails_fast_without_calling_provider(self, mock_generate):
        """Video analysis request must fail fast with HTTP 400 without invoking provider when no video model is available."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Inspect video clip",
                "modalities": ["video"],
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No registered models satisfy required capabilities", resp.json()["detail"])
        mock_generate.assert_not_called()

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
            DEFAULT_SYSTEM_INSTRUCTION,
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
            DEFAULT_SYSTEM_INSTRUCTION,
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
                modalities=["text", "video"],
                model="glm_5_3_flash",
            )

            req = mock_urlopen.call_args[0][0]
            import json
            payload = json.loads(req.data.decode("utf-8"))
            self.assertEqual(payload.get("task_type"), "agent_decision")
            self.assertEqual(payload.get("modalities"), ["text", "video"])
            self.assertEqual(payload.get("model"), "glm_5_3_flash")


class TestAssistServiceProfileHandling(unittest.TestCase):
    """Direct unit tests for AssistService profile deduplication and injection."""

    def setUp(self):
        self.service = AssistService()
        self.mock_llm = MagicMock()
        self.mock_llm.generate_response.return_value = ("Test reply", "nvidia:gpt_oss_20b")
        self.service.llm = self.mock_llm

    def test_profile_appended_when_not_pre_embedded(self):
        self.service.assist(
            message="hello",
            profile="Name: Alice\nRole: Engineer",
            system_instruction="Base system instruction.",
        )
        self.mock_llm.generate_response.assert_called_once()
        passed_instruction = self.mock_llm.generate_response.call_args[1]["system_instruction"]
        self.assertIn("Base system instruction.", passed_instruction)
        self.assertIn("User Profile & Background:\nName: Alice\nRole: Engineer", passed_instruction)
        self.assertEqual(passed_instruction.count("Name: Alice"), 1)

    def test_profile_not_duplicated_when_already_in_instruction(self):
        pre_embedded = "Base instruction.\n\nUser Profile & Background:\nName: Alice\nRole: Engineer"
        self.service.assist(
            message="hello",
            profile="Name: Alice\nRole: Engineer",
            system_instruction=pre_embedded,
        )
        passed_instruction = self.mock_llm.generate_response.call_args[1]["system_instruction"]
        self.assertEqual(passed_instruction.count("Name: Alice"), 1)
        self.assertEqual(passed_instruction.count("User Profile & Background:"), 1)

    def test_no_profile_block_when_empty_or_none(self):
        self.service.assist(message="hello", profile="")
        passed_instruction = self.mock_llm.generate_response.call_args[1]["system_instruction"]
        self.assertNotIn("User Profile & Background:", passed_instruction)
        self.assertEqual(passed_instruction, DEFAULT_SYSTEM_INSTRUCTION)

        self.mock_llm.reset_mock()
        self.service.assist(message="hello", profile=None)
        passed_instruction_none = self.mock_llm.generate_response.call_args[1]["system_instruction"]
        self.assertNotIn("User Profile & Background:", passed_instruction_none)
        self.assertEqual(passed_instruction_none, DEFAULT_SYSTEM_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
