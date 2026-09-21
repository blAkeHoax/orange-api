"""
test_image_api.py - Verification suite for Phase 7B Image Generation API & Service.

Tests:
1. POST /generate/image route registration.
2. Valid request succeeds with expected structure.
3. Default dimensions (1024x1024) forwarded to provider.
4. Invalid width rejected with HTTP 422 without calling provider.
5. Invalid height rejected with HTTP 422 without calling provider.
6. Invalid steps (0, 5) rejected with HTTP 422.
7. Negative seed rejected with HTTP 422.
8. Prompt > 2000 chars rejected with HTTP 422.
9. Empty prompt rejected with HTTP 422.
10. Unknown image model returns HTTP 400 without calling provider.
11. Provider result correctly translated to images[].b64_json and seed.
12. ProviderTimeoutError mapped to HTTP 504.
13. ProviderRateLimitError mapped to HTTP 429.
14. ProviderUnavailableError mapped to HTTP 503.
15. ProviderError mapped to HTTP 502.
16. duration_ms measured as non-negative integer.
17. API secret safety (no leaked API keys or Authorization tokens).
18. Assist regression (POST /assist unaffected).
19. Health regression (GET /health unaffected).
20. OpenAPI docs registration (POST /generate/image in /openapi.json).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure Orange API root is on sys.path
api_root = os.path.dirname(os.path.abspath(__file__))
if api_root not in sys.path:
    sys.path.insert(0, api_root)

from app.main import app
from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
)
from app.providers.image_base import (
    ImageArtifact,
    ImageGenerationResult,
)


class TestImageAPI(unittest.TestCase):
    """Test suite for Phase 7B Image Generation API and Service."""

    def setUp(self):
        self.client = TestClient(app)

    def test_1_route_exists(self):
        """TEST 1: Verify POST /generate/image route is registered in FastAPI app."""
        route_paths = [getattr(r, "path", None) for r in app.routes if getattr(r, "path", None)]
        for r in app.routes:
            if hasattr(r, "original_router"):
                route_paths.extend([getattr(x, "path", None) for x in r.original_router.routes])
        self.assertIn("/generate/image", route_paths)

    def test_2_valid_request_succeeds(self):
        """TEST 2: Valid request succeeds with 200 OK and expected structure."""
        mock_result = ImageGenerationResult(
            artifacts=[
                ImageArtifact(
                    base64="VALID_BASE64_IMAGE",
                    seed=42,
                    mime_type="image/png",
                    width=1024,
                    height=1024,
                )
            ],
            model="black-forest-labs/flux.2-klein-4b",
            provider="nvidia",
        )

        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image", return_value=mock_result):
            resp = self.client.post(
                "/generate/image",
                json={
                    "prompt": "a futuristic orange robot",
                    "model": "flux_2_klein_4b",
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["model"], "flux_2_klein_4b")
            self.assertIsInstance(data["images"], list)
            self.assertEqual(len(data["images"]), 1)
            self.assertEqual(data["images"][0]["b64_json"], "VALID_BASE64_IMAGE")
            self.assertEqual(data["images"][0]["mime_type"], "image/png")
            self.assertEqual(data["images"][0]["width"], 1024)
            self.assertEqual(data["images"][0]["height"], 1024)
            self.assertEqual(data["seed"], 42)
            self.assertIsInstance(data["duration_ms"], int)
            self.assertGreaterEqual(data["duration_ms"], 0)
            self.assertIsNone(data["error"])

    def test_3_default_dimensions(self):
        """TEST 3: When width/height are omitted, provider receives 1024, 1024."""
        mock_result = ImageGenerationResult(
            artifacts=[ImageArtifact(base64="MOCK", seed=1)],
            model="black-forest-labs/flux.2-klein-4b",
        )

        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image", return_value=mock_result) as mock_gen:
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Omit dimensions test"},
            )
            self.assertEqual(resp.status_code, 200)
            mock_gen.assert_called_once()
            _, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get("width"), 1024)
            self.assertEqual(kwargs.get("height"), 1024)

    def test_4_invalid_width(self):
        """TEST 4: Send width != 1024 -> HTTP 422, provider not called."""
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Invalid width", "width": 512, "height": 1024},
            )
            self.assertEqual(resp.status_code, 422)
            mock_gen.assert_not_called()

    def test_5_invalid_height(self):
        """TEST 5: Send height != 1024 -> HTTP 422, provider not called."""
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Invalid height", "width": 1024, "height": 768},
            )
            self.assertEqual(resp.status_code, 422)
            mock_gen.assert_not_called()

    def test_6_invalid_steps(self):
        """TEST 6: steps=0 and steps=5 rejected with HTTP 422."""
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp1 = self.client.post(
                "/generate/image",
                json={"prompt": "Steps 0", "steps": 0},
            )
            self.assertEqual(resp1.status_code, 422)

            resp2 = self.client.post(
                "/generate/image",
                json={"prompt": "Steps 5", "steps": 5},
            )
            self.assertEqual(resp2.status_code, 422)
            mock_gen.assert_not_called()

    def test_7_negative_seed(self):
        """TEST 7: seed < 0 rejected with HTTP 422."""
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Negative seed", "seed": -1},
            )
            self.assertEqual(resp.status_code, 422)
            mock_gen.assert_not_called()

    def test_8_prompt_too_long(self):
        """TEST 8: prompt > 2000 chars rejected with HTTP 422."""
        long_prompt = "a" * 2001
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp = self.client.post(
                "/generate/image",
                json={"prompt": long_prompt},
            )
            self.assertEqual(resp.status_code, 422)
            mock_gen.assert_not_called()

    def test_9_empty_prompt(self):
        """TEST 9: empty and whitespace-only prompts rejected with HTTP 422."""
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp1 = self.client.post(
                "/generate/image",
                json={"prompt": ""},
            )
            self.assertEqual(resp1.status_code, 422)

            resp2 = self.client.post(
                "/generate/image",
                json={"prompt": "   "},
            )
            self.assertEqual(resp2.status_code, 422)
            mock_gen.assert_not_called()

    def test_10_unknown_model(self):
        """TEST 10: Unknown model key returns deterministic HTTP 400, provider not called."""
        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image") as mock_gen:
            resp = self.client.post(
                "/generate/image",
                json={
                    "prompt": "Unknown model test",
                    "model": "fictional_unregistered_model",
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Unknown image model identifier", resp.json()["detail"])
            mock_gen.assert_not_called()

    def test_11_provider_result_translation(self):
        """TEST 11: Provider result translated to images[].b64_json; artifacts not exposed."""
        mock_result = ImageGenerationResult(
            artifacts=[ImageArtifact(base64="RAW_IMAGE_DATA_123", seed=9876)],
            model="black-forest-labs/flux.2-klein-4b",
        )

        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image", return_value=mock_result):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Translation test"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["images"][0]["b64_json"], "RAW_IMAGE_DATA_123")
            self.assertEqual(data["seed"], 9876)
            self.assertNotIn("artifacts", data)

    def test_12_provider_timeout(self):
        """TEST 12: ProviderTimeoutError maps to HTTP 504."""
        with patch(
            "app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image",
            side_effect=ProviderTimeoutError("NVIDIA request timed out"),
        ):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Timeout test"},
            )
            self.assertEqual(resp.status_code, 504)
            self.assertIn("AI Provider timeout", resp.json()["detail"])

    def test_13_provider_rate_limit(self):
        """TEST 13: ProviderRateLimitError maps to HTTP 429."""
        with patch(
            "app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image",
            side_effect=ProviderRateLimitError("Rate limit exceeded"),
        ):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Rate limit test"},
            )
            self.assertEqual(resp.status_code, 429)
            self.assertIn("AI Provider rate limit", resp.json()["detail"])

    def test_14_provider_unavailable(self):
        """TEST 14: ProviderUnavailableError maps to HTTP 503."""
        with patch(
            "app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image",
            side_effect=ProviderUnavailableError("NVIDIA service down"),
        ):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Unavailable test"},
            )
            self.assertEqual(resp.status_code, 503)
            self.assertIn("AI Provider unavailable", resp.json()["detail"])

    def test_15_provider_error(self):
        """TEST 15: ProviderError maps to HTTP 502."""
        with patch(
            "app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image",
            side_effect=ProviderError("General NVIDIA failure"),
        ):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Provider error test"},
            )
            self.assertEqual(resp.status_code, 502)
            self.assertIn("AI Provider error", resp.json()["detail"])

    def test_16_duration_measurement(self):
        """TEST 16: duration_ms exists and is a non-negative integer."""
        mock_result = ImageGenerationResult(
            artifacts=[ImageArtifact(base64="MOCK", seed=1)],
            model="black-forest-labs/flux.2-klein-4b",
        )

        with patch("app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image", return_value=mock_result):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Duration test"},
            )
            self.assertEqual(resp.status_code, 200)
            duration = resp.json()["duration_ms"]
            self.assertIsInstance(duration, int)
            self.assertGreaterEqual(duration, 0)

    def test_17_api_secret_safety(self):
        """TEST 17: Error details never expose API key or Authorization headers."""
        secret_key = "nvapi-SECRET-LEAK-PREVENTION-TEST"
        with patch(
            "app.providers.nvidia_image_provider.NVIDIAImageProvider.generate_image",
            side_effect=ProviderError(f"Upstream error with Bearer {secret_key}"),
        ):
            resp = self.client.post(
                "/generate/image",
                json={"prompt": "Secret safety test"},
            )
            self.assertEqual(resp.status_code, 502)
            detail = resp.json()["detail"]
            self.assertNotIn(secret_key, detail)

    def test_18_assist_regression(self):
        """TEST 18: POST /assist continues to work with mocked intelligence provider."""
        with patch("app.providers.nvidia_provider.NVIDIAProvider.generate", return_value="Mocked intelligence reply"):
            resp = self.client.post(
                "/assist",
                json={"message": "Hello assist"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["reply"], "Mocked intelligence reply")
            self.assertEqual(data["model"], "nvidia:openai/gpt-oss-20b")

    def test_19_health_regression(self):
        """TEST 19: GET /health continues to return online status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "online"})

    def test_20_docs_registration(self):
        """TEST 20: FastAPI exposes POST /generate/image in OpenAPI schema."""
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()
        paths = schema.get("paths", {})
        self.assertIn("/generate/image", paths)
        self.assertIn("post", paths["generate/image"] if "generate/image" in paths else paths["/generate/image"])


if __name__ == "__main__":
    unittest.main()
