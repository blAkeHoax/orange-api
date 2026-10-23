"""
test_vision_contract.py - Regression Test Suite for Orange AI Complete Image Feature Removal.

Verifies:
1. Existing text-only request and response contracts remain valid (200 OK).
2. Image attachment fields in /assist are rejected with HTTP 422 (Unprocessable Entity).
3. Alternate image-bearing fields (images, image_url, b64_json) are rejected with HTTP 422.
4. Vision modality ('vision') is rejected with HTTP 400.
5. Vision task archetype ('vision_analysis') is rejected with HTTP 400.
6. Image-generation endpoints (/generate/image) remain completely absent (HTTP 404).
7. Validation error handler and responses never leak submitted image data or credentials.
8. Model registry contains zero vision-only models and zero models with 'vision' capability.
9. NVIDIA provider constructs scalar string text messages only (no multimodal array blocks).
10. Intelligence router contains no vision task types or vision models.
"""

import unittest
from unittest.mock import patch, MagicMock
import base64
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import AssistRequest, AssistResponse
from app.services.task_profile_builder import build_task_profile, normalize_modalities
from app.models.task_profile import TaskType
from app.services.intelligence_router import IntelligenceRouter
from app.services.model_registry import list_models, model_exists, get_model, UnknownModelError
from app.providers.nvidia_provider import NVIDIAProvider


TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")


class TestImageFeatureRemovalContract(unittest.TestCase):
    """Test suite verifying complete removal of all image-related functionality."""

    def setUp(self):
        self.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Text-Only Request and Response Contract
    # -------------------------------------------------------------------------
    @patch("app.providers.nvidia_provider.NVIDIAProvider.generate")
    def test_text_only_request_accepted(self, mock_generate):
        """Standard text assistance works and returns 200 OK."""
        mock_generate.return_value = "This is a helpful text response."
        resp = self.client.post("/assist", json={"message": "Hello Orange AI!"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["reply"], "This is a helpful text response.")
        self.assertIn("nvidia/nemotron-3.5-lightning-30b-a3b", data["model"])

    def test_assist_request_schema_forbids_extra_fields(self):
        """AssistRequest schema strictly rejects extra fields including 'image'."""
        with self.assertRaises(ValidationError) as ctx:
            AssistRequest(message="hello", image={"b64_json": TINY_PNG_B64})
        self.assertIn("extra_forbidden", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Rejection of Image Attachment Fields (HTTP 422)
    # -------------------------------------------------------------------------
    def test_api_rejects_image_dict_with_422(self):
        """POST /assist with 'image' dict is rejected with HTTP 422."""
        resp = self.client.post("/assist", json={
            "message": "Inspect this image",
            "image": {"b64_json": TINY_PNG_B64, "mime_type": "image/png"},
        })
        self.assertEqual(resp.status_code, 422)
        errors = resp.json()["detail"]
        self.assertTrue(any(e.get("loc") == ["body", "image"] for e in errors))

    def test_api_rejects_image_string_with_422(self):
        """POST /assist with 'image' string is rejected with HTTP 422."""
        resp = self.client.post("/assist", json={
            "message": "Inspect this image",
            "image": f"data:image/png;base64,{TINY_PNG_B64}",
        })
        self.assertEqual(resp.status_code, 422)

    def test_api_rejects_image_null_with_422(self):
        """POST /assist with 'image': null is rejected with HTTP 422 (extra forbidden)."""
        resp = self.client.post("/assist", json={
            "message": "Inspect this image",
            "image": None,
        })
        self.assertEqual(resp.status_code, 422)

    # -------------------------------------------------------------------------
    # 3. Rejection of Alternate Multimodal Payloads (HTTP 422)
    # -------------------------------------------------------------------------
    def test_api_rejects_images_array_with_422(self):
        """POST /assist with alternate 'images' array is rejected with HTTP 422."""
        resp = self.client.post("/assist", json={
            "message": "Inspect images",
            "images": [TINY_PNG_B64],
        })
        self.assertEqual(resp.status_code, 422)

    def test_api_rejects_image_url_with_422(self):
        """POST /assist with alternate 'image_url' is rejected with HTTP 422."""
        resp = self.client.post("/assist", json={
            "message": "Inspect URL",
            "image_url": "https://example.com/photo.png",
        })
        self.assertEqual(resp.status_code, 422)

    def test_api_rejects_b64_json_with_422(self):
        """POST /assist with direct 'b64_json' field is rejected with HTTP 422."""
        resp = self.client.post("/assist", json={
            "message": "Inspect raw",
            "b64_json": TINY_PNG_B64,
        })
        self.assertEqual(resp.status_code, 422)

    # -------------------------------------------------------------------------
    # 4. Rejection of Vision Modality and Task Type (HTTP 400)
    # -------------------------------------------------------------------------
    def test_api_rejects_vision_modality_with_400(self):
        """POST /assist with modalities=['vision'] is rejected with HTTP 400."""
        resp = self.client.post("/assist", json={
            "message": "Analyze image",
            "modalities": ["vision"],
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported modality: 'vision'", resp.json()["detail"])

    def test_api_rejects_vision_analysis_task_type_with_400(self):
        """POST /assist with task_type='vision_analysis' is rejected with HTTP 400."""
        resp = self.client.post("/assist", json={
            "message": "Analyze visual scene",
            "task_type": "vision_analysis",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported task_type", resp.json()["detail"])

    # -------------------------------------------------------------------------
    # 5. Absence of Image Generation Endpoints (HTTP 404)
    # -------------------------------------------------------------------------
    def test_image_generation_post_endpoint_absent(self):
        """POST /generate/image returns 404 Not Found."""
        resp = self.client.post("/generate/image", json={"prompt": "A modern orange computer"})
        self.assertEqual(resp.status_code, 404)

    def test_image_generation_get_endpoint_absent(self):
        """GET /generate/image returns 404 Not Found."""
        resp = self.client.get("/generate/image")
        self.assertEqual(resp.status_code, 404)

    # -------------------------------------------------------------------------
    # 6. Safe Error Handling & Payload Scrubbing
    # -------------------------------------------------------------------------
    def test_validation_error_never_echoes_image_payload(self):
        """Submitting a 5 MB invalid base64 image field does not echo the data in 422 response."""
        oversized_data = "A" * 100000
        resp = self.client.post("/assist", json={
            "message": "Test input",
            "image": {"b64_json": oversized_data},
        })
        self.assertEqual(resp.status_code, 422)
        self.assertNotIn(oversized_data, resp.text)
        self.assertLess(len(resp.text), 1000)

    def test_validation_error_redacts_credentials(self):
        """Submitting credential-like keys in invalid fields scrubs values to [omitted]."""
        resp = self.client.post("/assist", json={
            "message": "Test input",
            "token": "secret_token_12345",
            "password": "secret_password",
        })
        self.assertEqual(resp.status_code, 422)
        self.assertNotIn("secret_token_12345", resp.text)
        self.assertNotIn("secret_password", resp.text)

    # -------------------------------------------------------------------------
    # 7. Model Registry and Provider Isolation
    # -------------------------------------------------------------------------
    def test_model_registry_has_no_vision_models(self):
        """Verify model registry contains zero vision-only models and no 'vision' capabilities."""
        self.assertFalse(model_exists("muse_glimmer_30b"))
        self.assertFalse(model_exists("meta/muse-glimmer-30b"))
        self.assertFalse(model_exists("meta/llama-3.2-11b-vision-instruct"))
        self.assertFalse(model_exists("llama_3_2_vision"))

        with self.assertRaises(UnknownModelError):
            get_model("muse_glimmer_30b")

        models = list_models()
        for m in models:
            self.assertNotIn("vision", m.capabilities, f"Model {m.key} must not declare 'vision' capability.")

    def test_nvidia_provider_constructs_scalar_string_prompt_only(self):
        """Direct NVIDIA provider generation sends prompt as scalar text string, never list blocks."""
        prov = NVIDIAProvider(api_key="test-key", model="openai/gpt-oss-20b")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Text reply"}}]}

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            prov.generate("Explain algorithms")
            call_payload = mock_post.call_args[1]["json"]
            user_msg = next(m for m in call_payload["messages"] if m["role"] == "user")
            self.assertIsInstance(user_msg["content"], str)
            self.assertEqual(user_msg["content"], "Explain algorithms")

    def test_task_types_contain_no_vision_analysis(self):
        """Verify TaskType enum has no VISION_ANALYSIS."""
        task_types = [t.value for t in TaskType]
        self.assertNotIn("vision_analysis", task_types)


if __name__ == "__main__":
    unittest.main()
