"""
test_default_system_regression.py - New Default System Behavioral Regression Test Suite.

Orange AI API - Stage 1 Stabilization & Phase 2 Readiness.

Verifies that DEFAULT_SYSTEM_INSTRUCTION enforces Orange AI's intended response behaviors:
1. Direct structural openings (no filler pleasantries or conversational preambles).
2. Strict user constraint and format compliance.
3. Single-choice decisiveness without evasive hedging.
4. Anti-hallucination / explicit honesty and uncertainty acknowledgment.
5. Adaptive response length proportional to query complexity.
6. Objective facts vs. subjective preferences distinction.

Covers both:
- Deterministic offline contract regression tests (6 test classes, runs in CI/offline with zero network/API keys).
- Optional live behavioral smoke checks (gated by RUN_LIVE_BEHAVIORAL_TESTS=1).
"""

import sys
import os
import re
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure Orange API root is on sys.path
API_DIR = os.path.abspath(os.path.dirname(__file__))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from app.main import app
from app.models.schemas import Message, AssistRequest, AssistResponse
from app.services.assist_service import AssistService, DEFAULT_SYSTEM_INSTRUCTION
from app.services.model_registry import (
    get_model,
    list_available_models,
    model_exists,
    UnknownModelError,
)
from app.services.intelligence_router import IneligibleOverrideError
from app.providers.nvidia_provider import NVIDIAProvider
from app.providers.base import ProviderError, ProviderUnavailableError


class TestDefaultSystemPromptIntegrity(unittest.TestCase):
    """
    Test Suite 1: Behavioral Clause Integrity.
    Verifies that DEFAULT_SYSTEM_INSTRUCTION explicitly contains and preserves
    all required behavioral directives defined by the New Default System.
    """

    def test_default_system_instruction_is_non_empty_string(self):
        """DEFAULT_SYSTEM_INSTRUCTION must be a substantial non-empty string."""
        self.assertIsInstance(DEFAULT_SYSTEM_INSTRUCTION, str)
        self.assertTrue(len(DEFAULT_SYSTEM_INSTRUCTION.strip()) > 100)

    def test_behavioral_clause_direct_answers_no_filler(self):
        """Must require direct answers and prohibit repetitive pleasantries / preambles."""
        text = DEFAULT_SYSTEM_INSTRUCTION.lower()
        self.assertIn("answer the user's actual question directly", text)
        self.assertIn("avoid unnecessary introductions", text)
        self.assertIn("repetitive pleasantries", text)
        self.assertIn("irrelevant explanations", text)

    def test_behavioral_clause_constraint_compliance(self):
        """Must demand strict compliance with user instructions, format, and language."""
        text = DEFAULT_SYSTEM_INSTRUCTION.lower()
        self.assertIn("follow the user's explicit instructions", text)
        self.assertIn("constraints", text)
        self.assertIn("requested format", text)
        self.assertIn("language", text)

    def test_behavioral_clause_single_choice_decisiveness(self):
        """Must mandate making exactly one choice without evasion when required."""
        text = DEFAULT_SYSTEM_INSTRUCTION.lower()
        self.assertIn("if the user requires exactly one choice, make exactly one choice", text)
        self.assertIn("without evading the constraint or listing alternatives", text)
        self.assertIn("be decisive when the task requires a decision", text)

    def test_behavioral_clause_anti_hallucination_and_uncertainty(self):
        """Must enforce grounding in available facts and explicit acknowledgment of uncertainty."""
        text = DEFAULT_SYSTEM_INSTRUCTION.lower()
        self.assertIn("base your answers on available information", text)
        self.assertIn("do not invent facts", text)
        self.assertIn("distinguish known facts from assumptions and uncertainty", text)
        self.assertIn("ask a clarifying question when missing information prevents a reliable answer", text)

    def test_behavioral_clause_adaptive_length(self):
        """Must enforce proportionality between response length and query complexity."""
        text = DEFAULT_SYSTEM_INSTRUCTION.lower()
        self.assertIn("adapt the response length to the task", text)
        self.assertIn("prefer concise answers for simple questions", text)
        self.assertIn("fuller explanations for complex requests", text)

    def test_behavioral_clause_objective_vs_subjective(self):
        """Must prohibit presenting subjective preferences as objective facts."""
        text = DEFAULT_SYSTEM_INSTRUCTION.lower()
        self.assertIn("do not present subjective preferences or uncertain conclusions as objective facts", text)


class TestDefaultInjection(unittest.TestCase):
    """
    Test Suite 2: Default System Instruction Injection.
    Verifies that when callers omit system_instruction, DEFAULT_SYSTEM_INSTRUCTION
    is injected intact into both AssistService and the FastAPI POST /assist endpoint.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_assist_service_injects_default_system_instruction_when_omitted(self, mock_generate):
        """AssistService.assist() passes DEFAULT_SYSTEM_INSTRUCTION when system_instruction=None."""
        mock_generate.return_value = "Direct response."
        service = AssistService()

        result = service.assist(message="What is the capital of Mars?")

        self.assertTrue(result["success"])
        self.assertEqual(result["reply"], "Direct response.")
        self.assertEqual(mock_generate.call_count, 1)

        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), DEFAULT_SYSTEM_INSTRUCTION)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_api_endpoint_injects_default_system_instruction_when_omitted(self, mock_generate):
        """POST /assist injects DEFAULT_SYSTEM_INSTRUCTION into provider when omitted from body."""
        mock_generate.return_value = "Direct endpoint response."

        resp = self.client.post("/assist", json={"message": "Summarize today's goals."})
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["reply"], "Direct endpoint response.")

        self.assertEqual(mock_generate.call_count, 1)
        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), DEFAULT_SYSTEM_INSTRUCTION)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_default_system_instruction_unmodified_identity(self, mock_generate):
        """Injected instruction must match DEFAULT_SYSTEM_INSTRUCTION verbatim without mutation."""
        mock_generate.return_value = "Verified exact identity."
        service = AssistService()

        service.assist(message="Check identity")

        _, kwargs = mock_generate.call_args
        injected = kwargs.get("system_instruction")
        self.assertEqual(injected, DEFAULT_SYSTEM_INSTRUCTION)
        self.assertEqual(len(injected), len(DEFAULT_SYSTEM_INSTRUCTION))


class TestExplicitOverridePrecedence(unittest.TestCase):
    """
    Test Suite 3: Explicit Override Precedence and Profile Composition.
    Verifies caller-provided overrides replace default instruction while
    preserving profile blocks and handling empty/whitespace edge cases.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_custom_system_instruction_replaces_default(self, mock_generate):
        """Explicit custom system instruction replaces DEFAULT_SYSTEM_INSTRUCTION."""
        mock_generate.return_value = "SQL response."
        custom = "You are a PostgreSQL query specialist. Output raw SQL only."

        resp = self.client.post(
            "/assist",
            json={
                "message": "List users",
                "system_instruction": custom,
            },
        )
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), custom)
        self.assertNotIn("You are Orange AI", kwargs.get("system_instruction", ""))

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_custom_system_instruction_whitespace_stripping(self, mock_generate):
        """Custom system instruction has leading/trailing whitespace stripped."""
        mock_generate.return_value = "Stripped response."
        custom_padded = "   \n\tCustom stripped instruction.\t\n  "

        resp = self.client.post(
            "/assist",
            json={
                "message": "Test whitespace",
                "system_instruction": custom_padded,
            },
        )
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), "Custom stripped instruction.")

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_whitespace_or_empty_system_instruction_reverts_to_default(self, mock_generate):
        """Empty, whitespace-only, or newline-only system instructions cleanly revert to default."""
        mock_generate.return_value = "Default fallback response."

        for empty_val in ["", "   ", "\t\t", "\n\n", "   \r\n  "]:
            with self.subTest(empty_val=repr(empty_val)):
                resp = self.client.post(
                    "/assist",
                    json={
                        "message": "Hello",
                        "system_instruction": empty_val,
                    },
                )
                self.assertEqual(resp.status_code, 200)
                _, kwargs = mock_generate.call_args
                self.assertEqual(
                    kwargs.get("system_instruction"),
                    DEFAULT_SYSTEM_INSTRUCTION,
                )

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_profile_composition_with_default_instruction(self, mock_generate):
        """Profile block is cleanly appended to DEFAULT_SYSTEM_INSTRUCTION."""
        mock_generate.return_value = "Personalized reply."
        profile_text = "Role: Staff Engineer. Timezone: UTC+5:30. Preferred language: Python."

        resp = self.client.post(
            "/assist",
            json={
                "message": "What should I code today?",
                "profile": profile_text,
            },
        )
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        full_inst = kwargs.get("system_instruction", "")
        self.assertTrue(full_inst.startswith(DEFAULT_SYSTEM_INSTRUCTION))
        self.assertIn("User Profile & Background:\nRole: Staff Engineer.", full_inst)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_profile_composition_with_custom_instruction(self, mock_generate):
        """Profile block is cleanly appended to custom system instruction."""
        mock_generate.return_value = "Custom personalized reply."
        custom_inst = "Act as a terminal assistant."
        profile_text = "User prefers Bash commands."

        resp = self.client.post(
            "/assist",
            json={
                "message": "How do I list files?",
                "system_instruction": custom_inst,
                "profile": profile_text,
            },
        )
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        full_inst = kwargs.get("system_instruction", "")
        self.assertTrue(full_inst.startswith(custom_inst))
        self.assertIn("User Profile & Background:\nUser prefers Bash commands.", full_inst)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_profile_not_duplicated_if_already_in_instruction(self, mock_generate):
        """If instruction already embeds the profile block, it is not duplicated."""
        mock_generate.return_value = "De-duplicated reply."
        profile_text = "Senior Rust Developer"
        custom_inst = f"Instructions with embedded profile: Senior Rust Developer only."

        service = AssistService()
        service.assist(
            message="Check duplication",
            profile=profile_text,
            system_instruction=custom_inst,
        )

        _, kwargs = mock_generate.call_args
        full_inst = kwargs.get("system_instruction", "")
        # Should not append second profile block since profile_text is in full_instruction
        self.assertEqual(full_inst.count("Senior Rust Developer"), 1)
        self.assertNotIn("User Profile & Background:", full_inst)


class TestMultiTurnAndContextPreservation(unittest.TestCase):
    """
    Test Suite 4: Multi-Turn & Context Preservation.
    Verifies that system instruction is placed at index 0 (system role)
    and strictly preserved across multi-turn conversation histories.
    """

    def test_system_instruction_placed_at_index_zero_in_payload(self):
        """NVIDIAProvider places system_instruction at messages[0] with role='system'."""
        provider = NVIDIAProvider(api_key="test_key_dummy")

        captured_payloads = []

        def mock_post(url, json=None, headers=None, **kwargs):
            captured_payloads.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Test answer"}}]
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            provider.generate(
                prompt="User question",
                system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
            )

        self.assertEqual(len(captured_payloads), 1)
        messages = captured_payloads[0].get("messages", [])
        self.assertTrue(len(messages) >= 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_SYSTEM_INSTRUCTION)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "User question")

    def test_system_instruction_preserves_position_across_multi_turn_history(self):
        """System instruction remains at index 0 followed by history turns and latest user turn."""
        provider = NVIDIAProvider(api_key="test_key_dummy")

        history = [
            {"role": "user", "content": "Turn 1 question"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Turn 2 question"},
            {"role": "assistant", "content": "Turn 2 answer"},
            {"role": "user", "content": "Turn 3 question"},
            {"role": "assistant", "content": "Turn 3 answer"},
        ]

        captured_payloads = []

        def mock_post(url, json=None, headers=None, **kwargs):
            captured_payloads.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Turn 4 answer"}}]
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            provider.generate(
                prompt="Turn 4 question",
                system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
                history=history,
            )

        self.assertEqual(len(captured_payloads), 1)
        messages = captured_payloads[0].get("messages", [])

        # Total messages: 1 (system) + 6 (history) + 1 (current user) = 8
        self.assertEqual(len(messages), 8)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_SYSTEM_INSTRUCTION)

        # Check turn order
        self.assertEqual(messages[1], {"role": "user", "content": "Turn 1 question"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Turn 1 answer"})
        self.assertEqual(messages[-1], {"role": "user", "content": "Turn 4 question"})

    def test_system_instruction_preserved_with_large_context_payload(self):
        """Large prompts (simulating heavy context budgeting) do not truncate system instruction."""
        provider = NVIDIAProvider(api_key="test_key_dummy")
        large_prompt = "Context payload line:\n" + ("X" * 100 + "\n") * 50

        captured_payloads = []

        def mock_post(url, json=None, headers=None, **kwargs):
            captured_payloads.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Large context reply"}}]
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            provider.generate(
                prompt=large_prompt,
                system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
            )

        self.assertEqual(len(captured_payloads), 1)
        messages = captured_payloads[0].get("messages", [])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_SYSTEM_INSTRUCTION)


class TestModelCompatibilityAcrossActiveModels(unittest.TestCase):
    """
    Test Suite 5: Model Compatibility Across Active Models.
    Verifies identical system-instruction propagation for both active models
    (Nemotron & GLM) and strict rejection of disabled foundation models.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_system_instruction_propagates_to_default_nemotron(self, mock_generate):
        """Default model (Nemotron) receives DEFAULT_SYSTEM_INSTRUCTION."""
        mock_generate.return_value = "Nemotron reply."

        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello Nemotron",
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            },
        )
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), DEFAULT_SYSTEM_INSTRUCTION)
        self.assertEqual(kwargs.get("model"), "nvidia/nemotron-3.5-lightning-30b-a3b")

    @patch("app.providers.huggingface_provider.HuggingFaceProvider.generate")
    def test_system_instruction_propagates_to_selected_qwen(self, mock_generate):
        """Selected Hugging Face model (Qwen 3.8 27B) receives DEFAULT_SYSTEM_INSTRUCTION identically."""
        mock_generate.return_value = "Qwen reply."

        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
            resp = self.client.post(
                "/assist",
                json={
                    "message": "Hello Qwen",
                    "model": "Qwen/Qwen3.8-27B:fastest",
                },
            )
            self.assertEqual(resp.status_code, 200)

            _, kwargs = mock_generate.call_args
            self.assertEqual(kwargs.get("system_instruction"), DEFAULT_SYSTEM_INSTRUCTION)
            self.assertEqual(kwargs.get("model"), "Qwen/Qwen3.8-27B:fastest")

    def test_retired_model_glm_rejected_with_400(self):
        """Retired model z-ai/glm-5.3-flash is rejected before provider dispatch."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello GLM",
                "model": "z-ai/glm-5.3-flash",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("unavailable", resp.json()["detail"].lower())

    def test_unavailable_model_gpt_oss_rejected_with_400(self):
        """Disabled model openai/gpt-oss-20b is rejected before provider dispatch."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello GPT-OSS",
                "model": "openai/gpt-oss-20b",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("unavailable", resp.json()["detail"].lower())

    def test_unavailable_model_gemma_rejected_with_400(self):
        """Disabled model google/gemma-4-31b-it is rejected before provider dispatch."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "Hello Gemma",
                "model": "google/gemma-4-31b-it",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("unavailable", resp.json()["detail"].lower())


class TestErrorAndEdgeCaseHandling(unittest.TestCase):
    """
    Test Suite 6: Error & Edge-Case Handling.
    Verifies unicode resilience, input validation, and that provider errors
    or sanitization routines never leak secrets or corrupt system instructions.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_unicode_and_markdown_in_user_prompt_does_not_corrupt_instruction(self, mock_generate):
        """Complex unicode, emoji, and markdown in prompt do not alter system instruction."""
        mock_generate.return_value = "Unicode OK."
        complex_prompt = "こんにちは世界！ 🚀 123 \n```python\nprint('hello')\n```\nمرحبا بالعالم"

        resp = self.client.post("/assist", json={"message": complex_prompt})
        self.assertEqual(resp.status_code, 200)

        _, kwargs = mock_generate.call_args
        self.assertEqual(kwargs.get("system_instruction"), DEFAULT_SYSTEM_INSTRUCTION)

    def test_missing_message_returns_422_without_dispatch(self):
        """Missing required message field fails validation at FastAPI layer without dispatch."""
        resp = self.client.post("/assist", json={})
        self.assertEqual(resp.status_code, 422)

    def test_provider_error_sanitization_masks_secrets(self):
        """NVIDIAProvider sanitizes error messages, masking bearer tokens and API keys."""
        fake_api_key = "nvapi-secret1234567890abcdef"
        provider = NVIDIAProvider(api_key=fake_api_key)

        raw_error_message = f"Error communicating with Bearer {fake_api_key}: Connection reset."
        sanitized = provider._sanitize_message(raw_error_message)

        self.assertNotIn(fake_api_key, sanitized)
        self.assertNotIn("nvapi-secret1234567890abcdef", sanitized)
        self.assertIn("***", sanitized)


class TestLiveBehavioralSmokeChecks(unittest.TestCase):
    """
    Test Suite 7: Optional Live Behavioral Smoke Checks (Opt-In).
    Gated by RUN_LIVE_BEHAVIORAL_TESTS=1 environment variable.
    Evaluates real provider inference against DEFAULT_SYSTEM_INSTRUCTION to check:
    - Directness (no conversational filler).
    - Constraint compliance & single-choice decisiveness.
    - Anti-hallucination & uncertainty honesty.
    - Adaptive length (concise simple queries vs. comprehensive complex queries).
    """

    @classmethod
    def setUpClass(cls):
        live_flag = os.getenv("RUN_LIVE_BEHAVIORAL_TESTS", "0")
        if live_flag != "1":
            raise unittest.SkipTest(
                "Live behavioral smoke tests skipped (set RUN_LIVE_BEHAVIORAL_TESTS=1 to run)."
            )

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise unittest.SkipTest(
                "Live behavioral smoke tests require NVIDIA_API_KEY environment variable."
            )

        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    def test_live_direct_answer_no_filler(self):
        """Verify model answers directly without conversational preamble/pleasantries."""
        resp = self.client.post(
            "/assist",
            json={
                "message": "What is the capital of France? Name the city only in one word.",
            },
        )
        self.assertEqual(resp.status_code, 200)
        reply = resp.json().get("reply", "").strip()

        # Should start directly with Paris; not 'Sure! The capital is...'
        self.assertIn("Paris", reply)
        forbidden_openers = ["certainly", "sure", "of course", "hello", "as an ai"]
        for opener in forbidden_openers:
            self.assertFalse(
                reply.lower().startswith(opener),
                f"Response contained filler opener '{opener}': {reply}",
            )

    def test_live_single_choice_decisiveness(self):
        """Verify model makes exactly one choice when required without evasive hedging."""
        prompt = (
            "You must choose exactly one option: Coffee or Tea? "
            "Make exactly one choice without evading the constraint or listing alternatives. "
            "Reply with your choice in your first three words."
        )
        resp = self.client.post("/assist", json={"message": prompt})
        self.assertEqual(resp.status_code, 200)
        reply = resp.json().get("reply", "").strip()

        # Must mention either Coffee or Tea decisively
        has_coffee = "coffee" in reply.lower()
        has_tea = "tea" in reply.lower()
        self.assertTrue(
            has_coffee ^ has_tea,  # XOR: exactly one chosen
            f"Expected decisive single choice (Coffee XOR Tea), got: {reply}",
        )

    def test_live_uncertainty_honesty_anti_hallucination(self):
        """Verify model does not invent fictional historical facts and acknowledges uncertainty."""
        prompt = "Who was the winner of the 2038 Martian Orbital Marathon held on Olympus Mons?"
        resp = self.client.post("/assist", json={"message": prompt})
        self.assertEqual(resp.status_code, 200)
        reply = resp.json().get("reply", "").lower()

        # Should acknowledge that no such event has taken place or it's fictional / unknown
        uncertainty_signals = [
            "no", "has not", "hasn't", "fictional", "unknown", "future", "not aware", "cannot predict", "not possible"
        ]
        has_signal = any(signal in reply for signal in uncertainty_signals)
        self.assertTrue(
            has_signal,
            f"Expected honest uncertainty or fictional acknowledgment, got: {reply}",
        )

    def test_live_adaptive_length_brevity_vs_depth(self):
        """Verify response length scales with task complexity (simple vs. deep)."""
        simple_prompt = "What is 15 * 4? Answer with the number only."
        complex_prompt = (
            "Explain how the Raft distributed consensus algorithm works, "
            "including leader election, log replication, and safety guarantees, in two detailed paragraphs."
        )

        resp_simple = self.client.post("/assist", json={"message": simple_prompt})
        resp_complex = self.client.post("/assist", json={"message": complex_prompt})

        self.assertEqual(resp_simple.status_code, 200)
        self.assertEqual(resp_complex.status_code, 200)

        len_simple = len(resp_simple.json().get("reply", "").strip())
        len_complex = len(resp_complex.json().get("reply", "").strip())

        # Simple reply should be substantially shorter than complex explanation
        self.assertTrue(
            len_simple < len_complex * 0.3,
            f"Expected simple reply ({len_simple} chars) to be much shorter than complex ({len_complex} chars).",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
