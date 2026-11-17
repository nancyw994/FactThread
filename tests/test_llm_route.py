"""Free model requests must not fall through to paid models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import llm_client  # noqa: E402


class RouteModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._openrouter = llm_client.ALLOW_OPENROUTER_FALLBACK
        self._paid = llm_client.ALLOW_PAID

    def tearDown(self) -> None:
        llm_client.ALLOW_OPENROUTER_FALLBACK = self._openrouter
        llm_client.ALLOW_PAID = self._paid

    def _flags(self, *, openrouter: bool, paid: bool) -> None:
        llm_client.ALLOW_OPENROUTER_FALLBACK = openrouter
        llm_client.ALLOW_PAID = paid

    def test_free_mode_keeps_groq_and_skips_the_billed_openrouter_id(self) -> None:
        self._flags(openrouter=True, paid=False)
        self.assertEqual(llm_client.DEFAULT_MODEL, "openai/gpt-oss-120b")
        chain = llm_client.provider_chain(llm_client.DEFAULT_MODEL)
        self.assertEqual([item["provider"] for item in chain], ["groq"])
        self.assertEqual(chain[0]["model"], "openai/gpt-oss-120b")
        self.assertEqual(chain[0]["route_reason"], "groq_primary")
        self.assertTrue(llm_client.confirmed_free("groq", "openai/gpt-oss-120b"))
        self.assertFalse(llm_client.confirmed_free("openrouter", "openai/gpt-oss-120b"))

    def test_paid_mode_can_send_the_same_id_to_openrouter(self) -> None:
        self._flags(openrouter=True, paid=True)
        chain = llm_client.provider_chain(llm_client.DEFAULT_MODEL)
        self.assertEqual([item["provider"] for item in chain], ["groq", "openrouter"])
        self.assertEqual(chain[1]["model"], "openai/gpt-oss-120b")
        self.assertEqual(chain[1]["route_reason"], "openrouter_same_model")
        self.assertEqual(chain[1]["models"], ["openai/gpt-oss-120b"])
        self.assertNotIn("openai/gpt-5.6-terra", chain[1]["models"])

    def test_free_router_stays_free(self) -> None:
        self.assertEqual(llm_client.route_models("openrouter/free"), ["openrouter/free"])

    def test_free_variant_stays_free(self) -> None:
        self.assertEqual(llm_client.route_models("example/model:free"), ["example/model:free"])

    def test_nemotron_free_stays_free(self) -> None:
        model = "nvidia/nemotron-3.5-super-vl:free"
        self.assertEqual(llm_client.route_models(model), [model])

    def test_an_unconfirmed_model_is_not_routed_when_paid_is_off(self) -> None:
        self._flags(openrouter=True, paid=False)
        model = "anthropic/claude-sonnet-5"
        self.assertEqual(llm_client.route_models(model), [])
        self.assertEqual(llm_client.provider_chain(model), [])
        with self.assertRaises(RuntimeError) as caught:
            llm_client.available_targets(model)
        self.assertIn("not a confirmed free model", str(caught.exception))
        self.assertIn("Paid models are not allowed", str(caught.exception))

    def test_paid_model_adds_fallbacks_only_when_paid_is_allowed(self) -> None:
        self._flags(openrouter=True, paid=True)
        model = "anthropic/claude-sonnet-5"
        routed = llm_client.route_models(model)
        self.assertEqual(routed[0], model)
        self.assertEqual(routed[1:], list(llm_client.FALLBACK_MODELS))

    def test_disabled_openrouter_fallback_does_not_call_openrouter(self) -> None:
        calls: list[str] = []
        original_lookup, original_client = self._patch_keys()
        self._flags(openrouter=False, paid=True)

        def send(client, target):
            calls.append(target["provider"])
            error = RuntimeError("rate limit")
            error.status_code = 429
            raise error

        try:
            with self.assertRaises(RuntimeError) as caught:
                llm_client.call_with_fallback(llm_client.DEFAULT_MODEL, send)
        finally:
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(calls, ["groq"])
        self.assertIsInstance(caught.exception, llm_client.RateLimited)
        self.assertIn("groq", str(caught.exception))
        self.assertNotIn("openrouter", calls)

    def test_retry_after_is_kept_on_the_rate_limit(self) -> None:
        original_lookup, original_client = self._patch_keys()
        self._flags(openrouter=False, paid=False)

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
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(caught.exception.retry_after, 15)

    def _patch_keys(self):
        original_lookup = llm_client.lookup_key
        original_client = llm_client._client
        llm_client.lookup_key = lambda name: ("test-key", "ok")
        llm_client._client = lambda target: object()
        return original_lookup, original_client

    def test_free_mode_does_not_fall_through_to_openrouter(self) -> None:
        calls: list[str] = []
        original_lookup, original_client = self._patch_keys()
        self._flags(openrouter=True, paid=False)

        def send(client, target):
            calls.append(target["provider"])
            error = RuntimeError("rate limit")
            error.status_code = 429
            raise error

        try:
            with self.assertRaises(llm_client.RateLimited) as caught:
                llm_client.call_with_fallback(llm_client.DEFAULT_MODEL, send)
        finally:
            llm_client.lookup_key = original_lookup
            llm_client._client = original_client
        self.assertEqual(calls, ["groq"])
        self.assertIn("not a confirmed free model", str(caught.exception))
        attempt = caught.exception.attempts[0]
        self.assertEqual(attempt["provider"], "groq")
        self.assertEqual(attempt["model"], "openai/gpt-oss-120b")
        self.assertEqual(attempt["route_reason"], "groq_primary")
        self.assertEqual(attempt["cost_usd"], "unknown")
        self.assertFalse(attempt["cache_hit"])
        self.assertFalse(attempt["ok"])

    def test_groq_failure_uses_openrouter_once_when_paid_is_allowed(self) -> None:
        calls: list[str] = []
        original_lookup, original_client = self._patch_keys()
        self._flags(openrouter=True, paid=True)

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

    def test_a_free_openrouter_model_does_not_need_paid_mode(self) -> None:
        self._flags(openrouter=True, paid=False)
        model = "nvidia/nemotron-3.5-super-vl:free"
        chain = llm_client.provider_chain(model)
        self.assertEqual(chain[0]["provider"], "openrouter")
        self.assertEqual(chain[0]["models"], [model])
        self._flags(openrouter=False, paid=False)
        self.assertEqual(llm_client.provider_chain(model), [])

    def test_missing_both_keys_names_each_provider(self) -> None:
        self._flags(openrouter=True, paid=True)
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
