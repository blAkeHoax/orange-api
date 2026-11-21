"""
test_auth.py - Comprehensive test suite for Orange API shared-secret authentication.

Verifies:
1. Open pass-through mode when ORANGE_API_KEY is empty/unset (local dev/test compatibility).
2. Strict enforcement when ORANGE_API_KEY is configured:
   - 401 Unauthorized on missing credentials
   - 401 Unauthorized on incorrect credentials
   - 401 includes WWW-Authenticate: Bearer header
   - 200 OK on valid Authorization: Bearer <token>
   - 200 OK on case-insensitive bearer prefix (e.g., "bearer")
   - 200 OK on valid X-API-Key header
3. GET / and GET /health remain unauthenticated in all configurations.
4. Constant-time comparison (hmac.compare_digest) is enforced.
5. Direct verify_api_key dependency behavior and whitespace handling.
"""

import hmac
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

# Ensure Orange API root is on sys.path
api_root = os.path.dirname(os.path.abspath(__file__))
if api_root not in sys.path:
    sys.path.insert(0, api_root)

from app.main import app, startup_auth_check
from app.config import settings
from app.utils.auth import verify_api_key


class TestAPIAuthentication(unittest.TestCase):
    """Test suite for shared-secret bearer authentication on POST /assist."""

    def setUp(self):
        self.client = TestClient(app)
        self.mock_assist_response = {
            "reply": "Test assistance response",
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "success": True,
        }

    # =========================================================================
    # Group 1: Open pass-through mode (ORANGE_API_KEY unset or empty)
    # =========================================================================

    def test_01_open_passthrough_when_key_unset(self):
        """When ORANGE_API_KEY is empty, POST /assist succeeds without auth header."""
        with patch.object(settings, "ORANGE_API_KEY", ""):
            with patch("app.api.assist.assistant.assist", return_value=self.mock_assist_response):
                resp = self.client.post("/assist", json={"message": "Hello world"})
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["reply"], "Test assistance response")

    def test_02_open_passthrough_whitespace_only_treated_as_empty(self):
        """When ORANGE_API_KEY contains only whitespace, auth is disabled."""
        with patch.object(settings, "ORANGE_API_KEY", "   \t  "):
            with patch("app.api.assist.assistant.assist", return_value=self.mock_assist_response):
                resp = self.client.post("/assist", json={"message": "Hello world"})
                self.assertEqual(resp.status_code, 200)

    # =========================================================================
    # Group 2: Public routes remain unauthenticated
    # =========================================================================

    def test_03_root_endpoint_always_unauthenticated(self):
        """GET / returns 200 OK without auth headers even when ORANGE_API_KEY is set."""
        with patch.object(settings, "ORANGE_API_KEY", "prod-secret-key-456"):
            resp = self.client.get("/")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["message"], "Welcome to Orange API")

    def test_04_health_endpoint_always_unauthenticated(self):
        """GET /health returns 200 OK without auth headers even when ORANGE_API_KEY is set."""
        with patch.object(settings, "ORANGE_API_KEY", "prod-secret-key-456"):
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "online")

    # =========================================================================
    # Group 3: Strict enforcement when ORANGE_API_KEY is set
    # =========================================================================

    def test_05_missing_auth_header_returns_401(self):
        """When key is configured, omitting auth headers returns 401 with WWW-Authenticate."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            resp = self.client.post("/assist", json={"message": "Hello"})
            self.assertEqual(resp.status_code, 401)
            self.assertIn("WWW-Authenticate", resp.headers)
            self.assertEqual(resp.headers["WWW-Authenticate"], "Bearer")
            self.assertIn("Invalid or missing API key", resp.json()["detail"])

    def test_06_invalid_bearer_token_returns_401(self):
        """When key is configured, supplying an incorrect token returns 401."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            resp = self.client.post(
                "/assist",
                json={"message": "Hello"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.headers.get("WWW-Authenticate"), "Bearer")
            self.assertIn("Invalid or missing API key", resp.json()["detail"])

    def test_07_valid_bearer_token_returns_200(self):
        """When key is configured, valid Authorization: Bearer <token> allows request."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            with patch("app.api.assist.assistant.assist", return_value=self.mock_assist_response):
                resp = self.client.post(
                    "/assist",
                    json={"message": "Hello"},
                    headers={"Authorization": "Bearer secret-token-xyz"},
                )
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["reply"], "Test assistance response")

    def test_08_case_insensitive_bearer_prefix(self):
        """Authorization header scheme prefix is case-insensitive (bearer vs Bearer)."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            with patch("app.api.assist.assistant.assist", return_value=self.mock_assist_response):
                resp = self.client.post(
                    "/assist",
                    json={"message": "Hello"},
                    headers={"Authorization": "bearer secret-token-xyz"},
                )
                self.assertEqual(resp.status_code, 200)

    def test_09_valid_x_api_key_header_returns_200(self):
        """Alternative X-API-Key header is accepted and returns 200."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            with patch("app.api.assist.assistant.assist", return_value=self.mock_assist_response):
                resp = self.client.post(
                    "/assist",
                    json={"message": "Hello"},
                    headers={"X-API-Key": "secret-token-xyz"},
                )
                self.assertEqual(resp.status_code, 200)

    def test_10_invalid_x_api_key_header_returns_401(self):
        """Invalid X-API-Key header returns 401."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            resp = self.client.post(
                "/assist",
                json={"message": "Hello"},
                headers={"X-API-Key": "bad-key"},
            )
            self.assertEqual(resp.status_code, 401)

    def test_11_empty_bearer_token_returns_401(self):
        """Authorization header with only 'Bearer' (empty token) returns 401."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            resp = self.client.post(
                "/assist",
                json={"message": "Hello"},
                headers={"Authorization": "Bearer "},
            )
            self.assertEqual(resp.status_code, 401)

    # =========================================================================
    # Group 4: Constant-time comparison & Direct dependency unit tests
    # =========================================================================

    def test_12_constant_time_comparison_used(self):
        """verify_api_key calls hmac.compare_digest for constant-time comparison."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            with patch("app.utils.auth.hmac.compare_digest", wraps=hmac.compare_digest) as mock_compare:
                token = verify_api_key(authorization="Bearer secret-token-xyz")
                self.assertEqual(token, "secret-token-xyz")
                mock_compare.assert_called_once()
                args = mock_compare.call_args[0]
                self.assertEqual(args[0], b"secret-token-xyz")
                self.assertEqual(args[1], b"secret-token-xyz")

    def test_13_direct_verify_api_key_whitespace_token_stripped(self):
        """verify_api_key strips surrounding whitespace from supplied tokens."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            token = verify_api_key(authorization="Bearer   secret-token-xyz   ")
            self.assertEqual(token, "secret-token-xyz")

    def test_14_direct_verify_api_key_raises_http_401_exception(self):
        """verify_api_key directly raises HTTPException with status 401."""
        with patch.object(settings, "ORANGE_API_KEY", "secret-token-xyz"):
            with self.assertRaises(HTTPException) as ctx:
                verify_api_key(authorization="Bearer wrong")
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertEqual(ctx.exception.headers.get("WWW-Authenticate"), "Bearer")

    def test_15_startup_warning_when_key_unset(self):
        """Startup event logs a security warning when ORANGE_API_KEY is not configured."""
        with patch.object(settings, "ORANGE_API_KEY", ""):
            with patch("app.main.logger.warning") as mock_warn:
                startup_auth_check()
                mock_warn.assert_called_once()
                self.assertIn("SECURITY WARNING: ORANGE_API_KEY is not set", mock_warn.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
