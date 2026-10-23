"""
test_latency_instrumentation.py - Test Suite for Orange API Latency & Timing Instrumentation.

Verifies:
1. Correlation request ID generation and ContextVar propagation across async and sync calls.
2. Privacy-safe latency logging (never logs prompts, replies, or API keys).
3. Full request lifecycle timing stages recorded with matching correlation ID:
   - request_start
   - routing
   - nvidia_provider
   - ttft (diagnostic tracking of non-streaming limitation)
   - response_serialization
   - request_complete
4. Custom X-Request-ID header propagation from client to all stages and response headers.
5. Error and timeout latency capture (request_failed).
"""

import io
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.providers.nvidia_provider import NVIDIAProvider
from app.providers.base import ProviderTimeoutError
from app.utils.latency import (
    generate_request_id,
    get_current_request_id,
    set_current_request_id,
    get_or_create_request_id,
    setup_latency_logger,
    log_latency,
    get_recorded_timings,
    clear_recorded_timings,
)


class TestLatencyInstrumentation(unittest.TestCase):
    """Test suite verifying diagnostic latency and correlation ID instrumentation."""

    def setUp(self):
        self.client = TestClient(app)
        clear_recorded_timings()

    def tearDown(self):
        clear_recorded_timings()
        setup_latency_logger(stream=sys.stdout)

    def test_request_id_generator_and_context(self):
        """Test correlation ID format and ContextVar setter/getter."""
        req_id = generate_request_id()
        self.assertTrue(req_id.startswith("req_"))
        self.assertGreaterEqual(len(req_id), 16)

        set_current_request_id(req_id)
        self.assertEqual(get_current_request_id(), req_id)

        # Test get_or_create_request_id preserves active ID
        same_id = get_or_create_request_id()
        self.assertEqual(same_id, req_id)

    def test_log_latency_privacy_safeguards(self):
        """Test log_latency suppresses sensitive fields and formats valid records."""
        rec = log_latency(
            stage="test_stage",
            duration_ms=12.345,
            request_id="req_test_safe",
            authorization="Bearer secret_token_12345",
            api_key="nvapi-testsecretkey",
            password="supersecretpassword",
            prompt="User private prompt that should be omitted",
            reply="Assistant private response",
            safe_metric=42,
            model_name="openai/gpt-oss-20b",
        )

        self.assertEqual(rec["request_id"], "req_test_safe")
        self.assertEqual(rec["stage"], "test_stage")
        self.assertEqual(rec["duration_ms"], 12.35)
        self.assertEqual(rec["safe_metric"], 42)
        self.assertEqual(rec["model_name"], "openai/gpt-oss-20b")

        # Confirm all sensitive keys were strictly suppressed
        self.assertNotIn("authorization", rec)
        self.assertNotIn("api_key", rec)
        self.assertNotIn("password", rec)
        self.assertNotIn("prompt", rec)
        self.assertNotIn("reply", rec)

    def test_assist_request_records_lifecycle_timing_stages(self):
        """Verify standard /assist request logs request_start, routing, serialization, and request_complete."""
        with patch.object(NVIDIAProvider, "generate", return_value="The speed of light is 299,792,458 m/s."):
            resp = self.client.post("/assist", json={"message": "What is the speed of light?"})

        self.assertEqual(resp.status_code, 200)
        req_id = resp.headers.get("X-Request-ID")
        self.assertIsNotNone(req_id)
        self.assertTrue(req_id.startswith("req_"))

        # Retrieve recorded stages for this request
        records = get_recorded_timings(request_id=req_id)
        stages = [r["stage"] for r in records]

        # 1. request_start
        self.assertIn("request_start", stages)
        start_rec = next(r for r in records if r["stage"] == "request_start")
        self.assertEqual(start_rec["endpoint"], "/assist")
        self.assertIn("message_len", start_rec)
        self.assertIn("history_turns", start_rec)

        # 2. routing
        self.assertIn("routing", stages)
        routing_rec = next(r for r in records if r["stage"] == "routing")
        self.assertIn("duration_ms", routing_rec)
        self.assertEqual(routing_rec["selected_model"], "nvidia/nemotron-3.5-lightning-30b-a3b")

        # 3. response_serialization
        self.assertIn("response_serialization", stages)
        serial_rec = next(r for r in records if r["stage"] == "response_serialization")
        self.assertIn("duration_ms", serial_rec)
        self.assertIn("reply_len", serial_rec)

        # 4. request_complete
        self.assertIn("request_complete", stages)
        complete_rec = next(r for r in records if r["stage"] == "request_complete")
        self.assertIn("total_elapsed_ms", complete_rec)
        self.assertEqual(complete_rec["status_code"], 200)

    def test_nvidia_provider_timing_and_ttft(self):
        """Verify NVIDIAProvider.generate records provider duration and TTFT non-streaming diagnostic."""
        prov = NVIDIAProvider(api_key="test-api-key")
        test_req_id = "req_provider_test_123"
        set_current_request_id(test_req_id)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Generated reply from NVIDIA"}}]
        }

        with patch("app.providers.nvidia_provider.httpx.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_instance.post.return_value = mock_resp
            mock_client_instance.__enter__.return_value = mock_client_instance
            mock_client_instance.__exit__.return_value = False
            mock_client_cls.return_value = mock_client_instance

            reply = prov.generate("Hello NVIDIA")
            self.assertEqual(reply, "Generated reply from NVIDIA")

        records = get_recorded_timings(request_id=test_req_id)
        stages = [r["stage"] for r in records]

        # Verify nvidia_provider stage
        self.assertIn("nvidia_provider", stages)
        prov_rec = next(r for r in records if r["stage"] == "nvidia_provider")
        self.assertIn("duration_ms", prov_rec)
        self.assertEqual(prov_rec["http_status"], 200)
        self.assertIn("prompt_len", prov_rec)

        # Verify ttft stage
        self.assertIn("ttft", stages)
        ttft_rec = next(r for r in records if r["stage"] == "ttft")
        self.assertEqual(ttft_rec["status"], "unavailable")
        self.assertEqual(ttft_rec["reason"], "non_streaming_provider")

    def test_custom_correlation_id_propagates(self):
        """Verify client-supplied X-Request-ID is preserved across all logged stages and response headers."""
        custom_id = "client-trace-id-998877"
        with patch.object(NVIDIAProvider, "generate", return_value="def fib(n): return n"):
            resp = self.client.post(
                "/assist",
                json={"message": "Write a python function", "task_type": "coding"},
                headers={"X-Request-ID": custom_id},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("X-Request-ID"), custom_id)

        records = get_recorded_timings(request_id=custom_id)
        self.assertGreaterEqual(len(records), 4)
        for r in records:
            self.assertEqual(r["request_id"], custom_id)

    def test_timeout_failure_records_request_failed(self):
        """Verify request timeout captures failed stage timing with correlation ID."""
        with patch.object(NVIDIAProvider, "generate", side_effect=ProviderTimeoutError("Read timed out")):
            resp = self.client.post("/assist", json={"message": "Simulate slow upstream"})

        self.assertEqual(resp.status_code, 504)
        req_id = resp.headers.get("X-Request-ID")
        self.assertIsNotNone(req_id)

        records = get_recorded_timings(request_id=req_id)
        stages = [r["stage"] for r in records]

        self.assertIn("request_failed", stages)
        failed_rec = next(r for r in records if r["stage"] == "request_failed")
        self.assertEqual(failed_rec["status_code"], 504)
        self.assertEqual(failed_rec["error_type"], "ProviderTimeoutError")
        self.assertIn("duration_ms", failed_rec)

    def test_latency_logger_emits_info_to_captured_stream(self):
        """Verify INFO-level [LATENCY] records are emitted to the configured stream."""
        capture_stream = io.StringIO()
        setup_latency_logger(stream=capture_stream)

        rec = log_latency(
            stage="test_stream_emission",
            duration_ms=45.67,
            request_id="req_capture_stream_01",
            task_type="conversational",
        )

        output = capture_stream.getvalue()
        self.assertIn("[LATENCY]", output)
        self.assertIn("request_id=req_capture_stream_01", output)
        self.assertIn("stage=test_stream_emission", output)
        self.assertIn("duration_ms=45.67ms", output)
        self.assertIn("task_type=conversational", output)
        self.assertTrue(output.endswith("\n"))

    def test_repeated_logger_setup_prevents_duplicate_handlers_and_duplicate_entries(self):
        """Verify repeated setup_latency_logger calls do not duplicate handlers or log lines."""
        capture_stream = io.StringIO()

        # Call setup multiple times
        for _ in range(5):
            lat_logger = setup_latency_logger(stream=capture_stream)

        stream_handlers = [
            h for h in lat_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(stream_handlers), 1)

        # Log a record and verify it is emitted exactly once
        log_latency(
            stage="test_single_entry",
            request_id="req_single_instance_check",
            duration_ms=1.23,
        )

        lines = [line for line in capture_stream.getvalue().splitlines() if "req_single_instance_check" in line]
        self.assertEqual(len(lines), 1)

    def test_root_and_unrelated_loggers_unaffected(self):
        """Verify orange_api.latency is isolated: propagate is False and other loggers are untouched."""
        lat_logger = setup_latency_logger()
        self.assertEqual(lat_logger.level, logging.INFO)
        self.assertFalse(lat_logger.propagate)

        # Verify unrelated loggers did not have handlers or propagation altered
        unrelated = logging.getLogger("orange_api.assist")
        self.assertFalse(any(h in unrelated.handlers for h in lat_logger.handlers))
        self.assertTrue(unrelated.propagate)


if __name__ == "__main__":
    unittest.main()
