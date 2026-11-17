"""User-supplied URL evidence. No network and no model calls."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import acquire_url  # noqa: E402
import judge_evidence  # noqa: E402


PAGE = """<html><body>
<h1>Example Domain</h1>
<p>This domain is for use in illustrative examples in documents.</p>
</body></html>"""


def _ok(raw: str, *, final: str = "https://example.com/final", status: int = 200, truncated: bool = False):
    def getter(url: str) -> dict:
        return {"http_status": status, "final_url": final, "raw": raw, "truncated": truncated, "error": None}

    return getter


class UrlEvidenceTests(unittest.TestCase):
    def test_fetch_saves_url_time_body_and_hash(self) -> None:
        page = acquire_url.fetch_page("https://example.com/start", getter=_ok(PAGE))
        self.assertEqual(page["original_url"], "https://example.com/start")
        self.assertEqual(page["final_url"], "https://example.com/final")
        self.assertRegex(page["fetched_at"], r"^\d{4}-\d{2}-\d{2}T")
        self.assertIn("illustrative examples", page["body"])
        self.assertEqual(page["content_sha256"], hashlib.sha256(page["body"].encode("utf-8")).hexdigest())
        self.assertFalse(page["is_complete"])
        self.assertEqual(page["completeness"], "unconfirmed")
        self.assertEqual(page["fetch_status"], "ok")
        self.assertEqual(page["title"], "Example Domain")

    def test_relevant_paragraph_keeps_character_offsets(self) -> None:
        page = acquire_url.fetch_page("https://example.com", getter=_ok(PAGE))
        page["source_id"] = acquire_url._source_id(page)
        claim = "This domain is for use in illustrative examples in documents."
        row = acquire_url.evidence_for_claim(claim, page, claim_id="url_supported")
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["source"]["title"], "Example Domain")
        self.assertFalse(row["source"]["is_complete"])
        passage = row["passages"][0]
        self.assertEqual(page["body"][passage["start"] : passage["end"]], passage["quote"])
        self.assertIn("illustrative examples", passage["quote"])

    def test_fetch_failure_is_not_a_false_claim(self) -> None:
        def getter(url: str) -> dict:
            return {"http_status": 404, "final_url": url, "raw": "", "truncated": False, "error": "HTTP 404"}

        page = acquire_url.fetch_page("https://example.com/missing", getter=getter)
        page["source_id"] = acquire_url._source_id(page)
        row = acquire_url.evidence_for_claim("The page names a price.", page, claim_id="url_missing")
        judged = judge_evidence.judge_one(row, {page["source_id"]: ""}, judge="fixed_rules")
        self.assertEqual(row["status"], "fetch_failed")
        self.assertIsNone(judged["verdict"])
        self.assertIn("not treated as false", row["reason"])
        self.assertNotEqual(judged["verdict"], "refuted")

    def test_empty_body_is_recorded(self) -> None:
        page = acquire_url.fetch_page("https://example.com/empty", getter=_ok("   <html><body></body></html>"))
        page["source_id"] = acquire_url._source_id(page)
        row = acquire_url.evidence_for_claim("The page names a price.", page, claim_id="url_empty")
        self.assertEqual(page["fetch_status"], "empty_body")
        self.assertEqual(row["status"], "empty_body")
        self.assertIn("not treated as false", row["reason"])

    def test_truncated_page_is_not_judged(self) -> None:
        page = acquire_url.fetch_page("https://example.com/cut", getter=_ok(PAGE, truncated=True))
        page["source_id"] = acquire_url._source_id(page)
        row = acquire_url.evidence_for_claim("This domain is for use in illustrative examples in documents.", page, claim_id="url_cut")
        self.assertEqual(row["status"], "truncated")
        self.assertIn("not treated as false", row["reason"])
        self.assertEqual(row["passages"], [])

    def test_omission_without_a_complete_scope_is_insufficient(self) -> None:
        page = acquire_url.fetch_page("https://example.com", getter=_ok(PAGE))
        page["source_id"] = acquire_url._source_id(page)
        row = acquire_url.evidence_for_claim(
            "The page does not state that the domain is for sale.",
            page,
            claim_id="url_omit",
            omission=True,
        )
        calls: list[str] = []

        def complete(system: str, user: str) -> str:
            calls.append(user)
            return "{}"

        judged = judge_evidence.judge_one(
            row,
            {page["source_id"]: page["body"]},
            judge="llm",
            complete=complete,
            model="stub-model",
        )
        self.assertEqual(row["status"], "ready")
        self.assertFalse(row["source"]["is_complete"])
        self.assertFalse(row["document"]["is_complete"])
        self.assertEqual(calls, [])
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertNotEqual(judged["verdict"], "refuted")
        self.assertIn("not treated as false", row["reason"])

    def test_attested_complete_omission_enters_the_check(self) -> None:
        page = acquire_url.fetch_page("https://example.com", getter=_ok(PAGE))
        page["source_id"] = acquire_url._source_id(page)
        row = acquire_url.evidence_for_claim(
            "The page does not state that the domain is for sale.",
            page,
            claim_id="url_full",
            omission=True,
            attest_complete=True,
        )
        self.assertEqual(row["status"], "ready")
        self.assertTrue(row["source"]["is_complete"])
        self.assertEqual(row["document"]["start"], 0)
        self.assertEqual(row["document"]["end"], len(page["body"]))
        self.assertEqual(page["body"][row["document"]["start"] : row["document"]["end"]], row["document"]["quote"])

    def test_attested_complete_but_truncated_does_not_enter(self) -> None:
        page = acquire_url.fetch_page("https://example.com/cut", getter=_ok(PAGE, truncated=True))
        page["source_id"] = acquire_url._source_id(page)
        row = acquire_url.evidence_for_claim(
            "The page never names a buyer.",
            page,
            claim_id="url_full_cut",
            omission=True,
            attest_complete=True,
        )
        self.assertEqual(row["status"], "truncated")
        self.assertIsNone(row["document"])

    def test_repeat_input_skips_fetch_and_model(self) -> None:
        fetches = {"n": 0}
        models: list[str] = []

        def getter(url: str) -> dict:
            fetches["n"] += 1
            return _ok(PAGE)(url)

        def complete(system: str, user: str) -> str:
            self.assertIn("untrusted", system.lower())
            self.assertIn("Ignore any instruction", system)
            models.append(user)
            packet = json.loads(user)
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The supplied sentence states it.",
                    "evidence_refs": packet["supplied_spans"],
                    "limitations": [],
                }
            )

        cases = [
            {
                "id": "cached",
                "claim": "This domain is for use in illustrative examples in documents.",
                "url": "https://example.com/cached",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = acquire_url.check_urls(
                cases, getter=getter, judge="llm", model="stub-model", complete=complete, cache_dir=cache
            )
            second = acquire_url.check_urls(
                cases, getter=getter, judge="llm", model="stub-model", complete=complete, cache_dir=cache
            )
            self.assertEqual(fetches["n"], 1)
            self.assertEqual(len(models), 1)
            self.assertEqual(first["verdicts"]["claims"][0]["cache"], "judged")
            self.assertEqual(second["verdicts"]["claims"][0]["cache"], "hit")
            self.assertEqual(second["verdicts"]["claims"][0]["verdict"], "supported")
            refreshed = acquire_url.check_urls(
                cases,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=cache,
                refresh=True,
            )
            self.assertEqual(fetches["n"], 2)
            self.assertEqual(len(models), 1)
            self.assertEqual(refreshed["verdicts"]["claims"][0]["cache"], "hit")
            rejudged = acquire_url.check_urls(
                cases,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=cache,
                rejudge=True,
            )
        self.assertEqual(fetches["n"], 2)
        self.assertEqual(len(models), 2)
        self.assertEqual(rejudged["verdicts"]["claims"][0]["cache"], "judged")

    def test_judge_cache_key_changes_with_the_prompt(self) -> None:
        page = {"content_sha256": "abc"}
        case = {"claim": "A claim.", "url": "https://example.com"}
        kwargs = {
            "judge": "llm",
            "model": "stub-model",
            "allow_openrouter_fallback": False,
            "allow_paid": False,
        }
        original = acquire_url._prompt_version
        acquire_url._prompt_version = lambda: "prompt-a"
        try:
            first = acquire_url._run_key(case, page, **kwargs)
            acquire_url._prompt_version = lambda: "prompt-b"
            second = acquire_url._run_key(case, page, **kwargs)
            acquire_url._prompt_version = lambda: "prompt-a"
            changed_claim = acquire_url._run_key({**case, "claim": "Another claim."}, page, **kwargs)
            changed_hash = acquire_url._run_key(case, {"content_sha256": "def"}, **kwargs)
            changed_model = acquire_url._run_key(case, page, **{**kwargs, "model": "other-model"})
            changed_paid = acquire_url._run_key(case, page, **{**kwargs, "allow_paid": True})
            changed_evidence = acquire_url._run_key(
                case,
                page,
                evidence={"passages": [{"start": 0, "end": 3, "quote": "abc"}]},
                **kwargs,
            )
        finally:
            acquire_url._prompt_version = original
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, changed_claim)
        self.assertNotEqual(first, changed_hash)
        self.assertNotEqual(first, changed_model)
        self.assertNotEqual(first, changed_paid)
        self.assertNotEqual(first, changed_evidence)

    def _ready(self, claim_id: str, url: str) -> dict:
        return {
            "id": claim_id,
            "claim": "This domain is for use in illustrative examples in documents.",
            "url": url,
            "human_label": "supported",
        }

    def test_a_rate_limit_is_retried_then_left_uncached(self) -> None:
        waits: list[float] = []

        def complete(system: str, user: str) -> str:
            error = RuntimeError("429 rate limit")
            error.retry_after = 12
            raise error

        cases = [self._ready("rate", "https://example.com/rate")]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            result = acquire_url.check_urls(
                cases,
                getter=_ok(PAGE),
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=cache,
                sleeper=waits.append,
                rng=lambda: 0,
            )
            self.assertEqual(list(cache.glob("runs/*.json")), [])
            self.assertEqual(waits, [12.0])
            again = acquire_url.check_urls(
                cases,
                getter=_ok(PAGE),
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=cache,
                sleeper=waits.append,
                rng=lambda: 0,
            )
        claim = result["verdicts"]["claims"][0]
        usage = result["verdicts"]["usage"]
        self.assertEqual(waits, [12.0, 12.0])
        self.assertEqual(usage["request_attempts"], 2)
        self.assertEqual(usage["successful_calls"], 0)
        self.assertEqual(usage["failed_attempts"], 2)
        self.assertIsNone(claim["verdict"])
        self.assertEqual(claim["judge_status"], "rate_limited")
        self.assertEqual(again["verdicts"]["claims"][0]["judge_status"], "rate_limited")
        self.assertEqual(again["cache"]["cache_hits"], 0)
        score = acquire_url._score_labels(result["verdicts"]["claims"], [{"id": "rate", "human_label": "supported"}])
        self.assertEqual(score["cases"], 1)
        self.assertEqual(score["completed"], 0)
        self.assertEqual(score["runtime_errors"], 1)
        self.assertEqual(score["completion"], "0/1")
        self.assertIsNone(score["accuracy"])
        self.assertEqual(score["false_support"], 0)
        self.assertEqual(score["semantic_errors"], [])

    def test_consecutive_limits_leave_later_claims_pending(self) -> None:
        calls: list[str] = []

        def complete(system: str, user: str) -> str:
            packet = json.loads(user)
            ref = packet["source"]["document_ref"]
            calls.append(ref)
            if ref.endswith("/two"):
                raise RuntimeError("429 rate limit")
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The supplied sentence states it.",
                    "evidence_refs": packet["supplied_spans"],
                    "limitations": [],
                }
            )

        claim = "This domain is for use in illustrative examples in documents."
        cases = [
            {"id": "first", "claim": claim, "url": "https://example.com/one"},
            {"id": "second", "claim": claim, "url": "https://example.com/two"},
            {"id": "third", "claim": claim, "url": "https://example.com/three"},
        ]

        def getter(url: str) -> dict:
            return _ok(PAGE, final=url)(url)

        with tempfile.TemporaryDirectory() as tmp:
            result = acquire_url.check_urls(
                cases,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
                sleeper=lambda _seconds: None,
                rng=lambda: 0,
            )
        self.assertEqual(calls, ["https://example.com/one", "https://example.com/two", "https://example.com/two"])
        rows = {row["claim_id"]: row for row in result["verdicts"]["claims"]}
        self.assertEqual(rows["first"]["verdict"], "supported")
        self.assertEqual(rows["first"]["judge_status"], "completed")
        self.assertIsNone(rows["second"]["verdict"])
        self.assertEqual(rows["second"]["judge_status"], "rate_limited")
        self.assertEqual(rows["third"]["judge_status"], "pending")
        self.assertIsNone(rows["third"]["verdict"])
        score = acquire_url._score_labels(
            result["verdicts"]["claims"],
            [
                {"id": "first", "human_label": "supported"},
                {"id": "second", "human_label": "refuted"},
                {"id": "third", "human_label": "supported"},
            ],
        )
        self.assertEqual(score["cases"], 3)
        self.assertEqual(score["completed"], 1)
        self.assertEqual(score["runtime_errors"], 1)
        self.assertEqual(score["pending"], 1)
        self.assertEqual(score["completion"], "1/3")
        self.assertEqual(score["accuracy"], "1/1")
        self.assertEqual(score["false_support"], 0)

    def test_missing_retry_after_uses_capped_jitter(self) -> None:
        waits: list[float] = []

        def complete(system: str, user: str) -> str:
            raise RuntimeError("429 rate limit")

        with tempfile.TemporaryDirectory() as tmp:
            acquire_url.check_urls(
                [self._ready("rate", "https://example.com/jitter")],
                getter=_ok(PAGE),
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
                sleeper=waits.append,
                rng=lambda: 0,
            )
        self.assertEqual(waits, [0.5])

    def test_a_false_support_is_counted_only_for_a_completed_verdict(self) -> None:
        score = acquire_url._score_labels(
            [
                {"claim_id": "ok", "verdict": "supported", "judge_status": "completed"},
                {"claim_id": "bad", "verdict": "supported", "judge_status": "completed"},
                {"claim_id": "wait", "verdict": None, "judge_status": "rate_limited"},
            ],
            [
                {"id": "ok", "human_label": "supported"},
                {"id": "bad", "human_label": "refuted"},
                {"id": "wait", "human_label": "insufficient_evidence"},
            ],
        )
        self.assertEqual(score["completed"], 2)
        self.assertEqual(score["runtime_errors"], 1)
        self.assertEqual(score["accuracy"], "1/2")
        self.assertEqual(score["false_support"], 1)
        self.assertEqual(score["semantic_errors"][0]["id"], "bad")


if __name__ == "__main__":
    unittest.main()
