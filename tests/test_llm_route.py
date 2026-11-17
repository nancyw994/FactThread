"""Free model requests must not fall through to paid models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import llm_client  # noqa: E402


class RouteModelTests(unittest.TestCase):
    def test_default_model_uses_groq_then_openrouter(self) -> None:
        self.assertEqual(llm_client.DEFAULT_MODEL, "openai/gpt-oss-120b")
        chain = llm_client.provider_chain(llm_client.DEFAULT_MODEL)
        self.assertEqual([item["provider"] for item in chain], ["groq", "openrouter"])
        self.assertEqual(chain[0]["model"], "openai/gpt-oss-120b")
        self.assertEqual(chain[1]["model"], "openai/gpt-oss-120b")
        self.assertEqual(chain[1]["models"], ["openai/gpt-oss-120b"])
        self.assertNotIn("openai/gpt-5.6-terra", chain[1]["models"])

    def test_free_router_stays_free(self) -> None:
        self.assertEqual(llm_client.route_models("openrouter/free"), ["openrouter/free"])

    def test_free_variant_stays_free(self) -> None:
        self.assertEqual(llm_client.route_models("example/model:free"), ["example/model:free"])

    def test_nemotron_free_stays_free(self) -> None:
        model = "nvidia/nemotron-3.5-super-vl:free"
        self.assertEqual(llm_client.route_models(model), [model])

    def test_paid_model_does_not_add_fallbacks_unless_authorized(self) -> None:
        model = "anthropic/claude-sonnet-5"
        self.assertEqual(llm_client.route_models(model), [model])
        original = llm_client.ALLOW_PAID
        llm_client.ALLOW_PAID = True
        try:
            routed = llm_client.route_models(model)
        finally:
            llm_client.ALLOW_PAID = original
        self.assertEqual(routed[0], model)
        self.assertEqual(routed[1:], list(llm_client.FALLBACK_MODELS))

    def test_disabled_openrouter_fallback_does_not_call_openrouter(self) -> None:
        calls: list[str] = []
        original_lookup, original_client = self._patch_keys()
        original = llm_client.ALLOW_OPENROUTER_FALLBACK
        llm_client.ALLOW_OPENROUTER_FALLBACK = False

        def send(client, target):
            calls.append(target["provider"])
            error = RuntimeError("rate limit")
            error.status_code = 429
            raise error

        try:
            with self.assertRaises(RuntimeError) as caught:
                llm_client.call_with_fallback(llm_client.DEFAULT_MODEL, send)
        finally:
            llm_client.ALLOW_OPENROUTER_FALLBACK = original
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(calls, ["groq"])
        self.assertIsInstance(caught.exception, llm_client.RateLimited)
        self.assertIn("groq", str(caught.exception))
        self.assertNotIn("openrouter", calls)

    def test_retry_after_is_kept_on_the_rate_limit(self) -> None:
        original_lookup, original_client = self._patch_keys()
        original = llm_client.ALLOW_OPENROUTER_FALLBACK
        llm_client.ALLOW_OPENROUTER_FALLBACK = False

        class Response:
            headers = {"retry-after": "15"}

        def send(client, target):
            error = RuntimeError("429")
            error.status_code = 429
            error.response = Response()
            raise error

        try:
            with self.assertRaises(llm_client.RateLimited) as caught:
                llm_client.call_with_fallback(llm_client.DEFAULT_MODEL, send)
        finally:
            llm_client.ALLOW_OPENROUTER_FALLBACK = original
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(caught.exception.retry_after, 15)

    def _patch_keys(self):
        original_lookup = llm_client.lookup_key
        original_client = llm_client._client
        llm_client.lookup_key = lambda name: ("test-key", "ok")
        llm_client._client = lambda target: object()
        return original_lookup, original_client

    def test_groq_failure_uses_openrouter_once(self) -> None:
        calls: list[str] = []
        original_lookup, original_client = self._patch_keys()

        def send(client, target):
            calls.append(target["provider"])
            if target["provider"] == "groq":
                error = RuntimeError("rate limit")
                error.status_code = 429
                raise error
            return "openrouter-response"

        try:
            result, target = llm_client.call_with_fallback("openai/gpt-oss-120b", send)
        finally:
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(calls, ["groq", "openrouter"])
        self.assertEqual(target["provider"], "openrouter")
        self.assertEqual(result, "openrouter-response")

    def test_groq_success_does_not_call_openrouter(self) -> None:
        calls: list[str] = []
        original_lookup, original_client = self._patch_keys()

        def send(client, target):
            calls.append(target["provider"])
            return "from groq"

        try:
            result, target = llm_client.call_with_fallback(llm_client.DEFAULT_MODEL, send)
        finally:
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(calls, ["groq"])
        self.assertEqual(target["provider"], "groq")
        self.assertEqual(result, "from groq")

    def test_missing_both_keys_names_each_provider(self) -> None:
        original = llm_client.lookup_key
        llm_client.lookup_key = lambda name: ("", "missing")
        try:
            with self.assertRaises(RuntimeError) as caught:
                llm_client.require_provider_key()
        finally:
            llm_client.lookup_key = original
        message = str(caught.exception)
        self.assertIn("GROQ_API_KEY", message)
        self.assertIn("OPENROUTER_API_KEY", message)


if __name__ == "__main__":
    unittest.main()
