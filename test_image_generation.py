"""
test_image_generation.py - Verification suite for Phase 7A Image Generation Provider.

Tests:
1. Registry contains FLUX.2-klein-4B.
2. Registry rejects unknown model.
3. Provider sends correct endpoint URL.
4. Provider sends correct Authorization header.
5. Provider sends correct Content-Type & Accept headers.
6. Provider sends correct default payload (1024x1024, steps=4, samples=1).
7. Provider accepts valid seed.
8. Provider rejects invalid resolution without calling HTTP.
9. Provider rejects invalid steps without calling HTTP.
10. Provider rejects negative seed without calling HTTP.
11. Successful response parsing (artifacts[0].base64 & seed).
12. Empty artifacts list rejection.
13. Missing base64 in artifact rejection.
14. HTTP 400 maps to ProviderError.
15. HTTP 401 maps to ProviderUnavailableError.
16. HTTP 422 maps to ProviderError.
17. HTTP 429 maps to ProviderRateLimitError.
18. HTTP 503 maps to ProviderUnavailableError.
19. HTTP 504 maps to ProviderTimeoutError.
20. Connection timeout maps to ProviderTimeoutError.
21. Connection error maps to ProviderUnavailableError.
22. Secret safety (API key never exposed in errors or logs).
23. Existing intelligence provider isolation.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import httpx

# Ensure Orange API root is on sys.path
api_root = os.path.dirname(os.path.abspath(__file__))
if api_root not in sys.path:
    sys.path.insert(0, api_root)

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
from app.providers.nvidia_image_provider import NVIDIAImageProvider
from app.services.image_model_registry import (
    get_image_model,
    list_image_models,
    image_model_exists,
    UnknownImageModelError,
    ImageModelInfo,
)


class TestImageGenerationProvider(unittest.TestCase):
    """Test suite for Phase 7A Image Generation Provider and Registry."""

    def test_1_registry_contains_flux(self):
        """TEST 1: Registry contains FLUX.2-klein-4B with exact verified metadata."""
        self.assertTrue(image_model_exists("flux_2_klein_4b"))
        self.assertTrue(image_model_exists("black-forest-labs/flux.2-klein-4b"))

        model = get_image_model("flux_2_klein_4b")
        self.assertEqual(model.key, "flux_2_klein_4b")
        self.assertEqual(model.provider_model_id, "black-forest-labs/flux.2-klein-4b")
        self.assertEqual(model.provider, "nvidia")
        self.assertEqual(model.display_name, "FLUX.2 Klein 4B")
        self.assertEqual(model.supported_resolutions, [(1024, 1024)])
        self.assertEqual(model.default_steps, 4)
        self.assertEqual(model.max_steps, 4)
        self.assertEqual(model.default_samples, 1)
        self.assertIn("text-to-image", model.capabilities)

    def test_2_registry_rejects_unknown_model(self):
        """TEST 2: Registry rejects unknown model keys with UnknownImageModelError."""
        self.assertFalse(image_model_exists("nonexistent_model_xyz"))
        with self.assertRaises(UnknownImageModelError):
            get_image_model("nonexistent_model_xyz")

    def test_3_provider_sends_correct_endpoint(self):
        """TEST 3: Provider sends request to exact FLUX endpoint."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"base64": "MOCK_BASE64", "seed": 42}]
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            provider.generate_image(prompt="A test prompt")
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            self.assertEqual(
                call_url,
                "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b",
            )

    def test_4_provider_sends_correct_authentication(self):
        """TEST 4: Provider sends Authorization: Bearer <key> header."""
        test_key = "nvapi-test-secret-key-12345"
        provider = NVIDIAImageProvider(api_key=test_key)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"base64": "MOCK_BASE64", "seed": 42}]
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            provider.generate_image(prompt="Auth test")
            headers = mock_post.call_args[1]["headers"]
            self.assertEqual(headers.get("Authorization"), f"Bearer {test_key}")

    def test_5_provider_sends_correct_headers(self):
        """TEST 5: Provider sends Accept and Content-Type application/json headers."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"base64": "MOCK_BASE64", "seed": 42}]
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            provider.generate_image(prompt="Header test")
            headers = mock_post.call_args[1]["headers"]
            self.assertEqual(headers.get("Accept"), "application/json")
            self.assertEqual(headers.get("Content-Type"), "application/json")

    def test_6_provider_sends_correct_default_payload(self):
        """TEST 6: Provider sends correct default payload (1024x1024, steps=4, samples=1)."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"base64": "MOCK_BASE64", "seed": 42}]
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            provider.generate_image(prompt="Default payload test")
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["prompt"], "Default payload test")
            self.assertEqual(payload["width"], 1024)
            self.assertEqual(payload["height"], 1024)
            self.assertEqual(payload["steps"], 4)
            self.assertEqual(payload["samples"], 1)
            self.assertNotIn("seed", payload)

    def test_7_provider_accepts_valid_seed(self):
        """TEST 7: Provider forwards non-negative seed in payload."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"base64": "MOCK_BASE64", "seed": 777}]
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            provider.generate_image(prompt="Seed test", seed=777)
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload.get("seed"), 777)

    def test_8_provider_rejects_invalid_resolution(self):
        """TEST 8: Provider rejects width != 1024 or height != 1024 without calling HTTP."""
        provider = NVIDIAImageProvider(api_key="test-key")

        with patch("httpx.Client.post") as mock_post:
            with self.assertRaises(ProviderError) as ctx1:
                provider.generate_image(prompt="Res test", width=512, height=1024)
            self.assertIn("Unsupported resolution", str(ctx1.exception))
            mock_post.assert_not_called()

            with self.assertRaises(ProviderError) as ctx2:
                provider.generate_image(prompt="Res test", width=1024, height=768)
            self.assertIn("Unsupported resolution", str(ctx2.exception))
            mock_post.assert_not_called()

    def test_9_provider_rejects_invalid_steps(self):
        """TEST 9: Provider rejects steps=0 or steps=5 without calling HTTP."""
        provider = NVIDIAImageProvider(api_key="test-key")

        with patch("httpx.Client.post") as mock_post:
            with self.assertRaises(ProviderError) as ctx1:
                provider.generate_image(prompt="Steps test", steps=0)
            self.assertIn("Invalid steps", str(ctx1.exception))
            mock_post.assert_not_called()

            with self.assertRaises(ProviderError) as ctx2:
                provider.generate_image(prompt="Steps test", steps=5)
            self.assertIn("Invalid steps", str(ctx2.exception))
            mock_post.assert_not_called()

    def test_10_provider_rejects_invalid_seed(self):
        """TEST 10: Provider rejects negative seed without calling HTTP."""
        provider = NVIDIAImageProvider(api_key="test-key")

        with patch("httpx.Client.post") as mock_post:
            with self.assertRaises(ProviderError) as ctx:
                provider.generate_image(prompt="Negative seed test", seed=-1)
            self.assertIn("Invalid seed", str(ctx.exception))
            mock_post.assert_not_called()

    def test_11_successful_response_parsing(self):
        """TEST 11: Correctly parses artifacts[0].base64 and artifacts[0].seed."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [
                {
                    "base64": "TEST_BASE64_IMAGE_DATA_XYZ",
                    "seed": 12345
                }
            ]
        }

        with patch("httpx.Client.post", return_value=mock_resp):
            result = provider.generate_image(prompt="Parsing test")
            self.assertIsInstance(result, ImageGenerationResult)
            self.assertEqual(len(result.artifacts), 1)
            artifact = result.artifacts[0]
            self.assertEqual(artifact.base64, "TEST_BASE64_IMAGE_DATA_XYZ")
            self.assertEqual(artifact.seed, 12345)
            self.assertEqual(artifact.width, 1024)
            self.assertEqual(artifact.height, 1024)
            self.assertEqual(artifact.mime_type, "image/png")
            self.assertEqual(result.model, "black-forest-labs/flux.2-klein-4b")

    def test_12_empty_artifacts(self):
        """TEST 12: Rejects empty artifacts list as ProviderError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"artifacts": []}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderError) as ctx:
                provider.generate_image(prompt="Empty artifacts")
            self.assertIn("empty or missing artifacts", str(ctx.exception))

    def test_13_missing_base64(self):
        """TEST 13: Rejects artifact missing base64 field as ProviderError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"seed": 12345}]  # Missing base64
        }

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderError) as ctx:
                provider.generate_image(prompt="Missing base64")
            self.assertIn("missing base64 image data", str(ctx.exception))

    def test_14_http_400_mapping(self):
        """TEST 14: HTTP 400 maps to ProviderError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": {"message": "Bad request syntax"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderError) as ctx:
                provider.generate_image(prompt="Error 400 test")
            self.assertIn("HTTP 400", str(ctx.exception))
            self.assertFalse(isinstance(ctx.exception, (ProviderUnavailableError, ProviderRateLimitError, ProviderTimeoutError)))

    def test_15_http_401_mapping(self):
        """TEST 15: HTTP 401 maps to ProviderUnavailableError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "Invalid API key"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderUnavailableError) as ctx:
                provider.generate_image(prompt="Auth failure test")
            self.assertIn("NVIDIA authentication failed", str(ctx.exception))

    def test_16_http_422_mapping(self):
        """TEST 16: HTTP 422 maps to ProviderError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"error": {"message": "Unprocessable entity parameters"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderError) as ctx:
                provider.generate_image(prompt="Error 422 test")
            self.assertIn("HTTP 422", str(ctx.exception))

    def test_17_http_429_mapping(self):
        """TEST 17: HTTP 429 maps to ProviderRateLimitError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderRateLimitError) as ctx:
                provider.generate_image(prompt="Rate limit test")
            self.assertIn("rate limit or quota exceeded", str(ctx.exception))

    def test_18_http_503_mapping(self):
        """TEST 18: HTTP 503 maps to ProviderUnavailableError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {"error": {"message": "Service unavailable"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderUnavailableError) as ctx:
                provider.generate_image(prompt="Service unavailable test")
            self.assertIn("temporarily unavailable", str(ctx.exception))

    def test_19_http_504_mapping(self):
        """TEST 19: HTTP 504 maps to ProviderTimeoutError."""
        provider = NVIDIAImageProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 504
        mock_resp.json.return_value = {"error": {"message": "Gateway timeout"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderTimeoutError) as ctx:
                provider.generate_image(prompt="Gateway timeout test")
            self.assertIn("gateway timeout", str(ctx.exception))

    def test_20_connection_timeout(self):
        """TEST 20: Connection timeout maps to ProviderTimeoutError."""
        provider = NVIDIAImageProvider(api_key="test-key")

        with patch("httpx.Client.post", side_effect=httpx.ReadTimeout("The read operation timed out")):
            with self.assertRaises(ProviderTimeoutError) as ctx:
                provider.generate_image(prompt="Read timeout test")
            self.assertIn("request timed out", str(ctx.exception))

    def test_21_connection_error(self):
        """TEST 21: Connection error maps to ProviderUnavailableError."""
        provider = NVIDIAImageProvider(api_key="test-key")

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
            with self.assertRaises(ProviderUnavailableError) as ctx:
                provider.generate_image(prompt="Connect error test")
            self.assertIn("Failed to connect", str(ctx.exception))

    def test_22_secret_safety(self):
        """TEST 22: Secret key is never exposed in exceptions or logs."""
        super_secret_key = "nvapi-SUPER-CONFIDENTIAL-KEY-99999"
        provider = NVIDIAImageProvider(api_key=super_secret_key)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        # Simulate upstream echoing the Authorization header or secret in its error
        mock_resp.json.return_value = {
            "error": {"message": f"Failed with header Bearer {super_secret_key}"}
        }

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(ProviderError) as ctx:
                provider.generate_image(prompt="Secret leak test")
            error_message = str(ctx.exception)
            self.assertNotIn(super_secret_key, error_message)
            self.assertIn("***", error_message)

    def test_23_existing_intelligence_provider_isolation(self):
        """TEST 23: Phase 7A does not import or alter existing intelligence modules."""
        # 1. Inspect app/services/model_registry.py
        import app.services.model_registry as text_registry
        self.assertFalse(hasattr(text_registry, "flux_2_klein_4b"))
        self.assertFalse(text_registry.model_exists("flux_2_klein_4b"))
        models = text_registry.list_models()
        self.assertEqual(len(models), 4)

        # 2. Inspect app/services/intelligence_router.py
        with open(os.path.join(api_root, "app", "services", "intelligence_router.py"), "r", encoding="utf-8") as f:
            router_code = f.read()
            self.assertNotIn("flux_2_klein_4b", router_code)
            self.assertNotIn("image_base", router_code)
            self.assertNotIn("nvidia_image_provider", router_code)

        # 3. Inspect app/providers/nvidia_provider.py
        with open(os.path.join(api_root, "app", "providers", "nvidia_provider.py"), "r", encoding="utf-8") as f:
            nvidia_code = f.read()
            self.assertNotIn("flux_2_klein_4b", nvidia_code)
            self.assertNotIn("image_base", nvidia_code)


if __name__ == "__main__":
    unittest.main()
