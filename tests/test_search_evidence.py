"""Search discovers URLs. The judge sees fetched text only."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import acquire_url
import search_evidence  # noqa: E402


CLAIM = "RFC 2606 reserves example.com for documentation."
PLAIN = "The recorded count was 12 lanes in 1998."
BODY = f"<html><title>Example</title><body><p>{CLAIM}</p></body></html>"


def _page(url: str) -> dict:
    return {"http_status": 200, "final_url": url, "raw": BODY, "truncated": False, "error": None}


class SearchEvidenceTests(unittest.TestCase):
    def test_queries_candidates_and_pages_stay_inside_the_budget(self) -> None:
        calls: list[str] = []

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            calls.append(query)
            self.assertLessEqual(limit, 5)
            if len(calls) == 1:
                return [
                    {"url": "https://www.example.com/page-0/", "title": "duplicate", "quote": "neutral snippet"},
                    {"url": "https://example.com/page-0", "title": "Page 0", "quote": "neutral snippet"},
                    {"url": "https://example.com/page-1", "title": "Page 1", "quote": "neutral snippet"},
                ], None
            return [
                {"url": "https://example.com/page-2", "title": "Page 2", "quote": "neutral snippet"},
                {"url": "https://example.com/page-3", "title": "Page 3", "quote": "neutral snippet"},
                {
                    "url": "https://example.com/oppose",
                    "title": "Opposing page",
                    "quote": "example.com is not reserved for documentation",
                },
                {"url": "https://example.com/extra", "title": "Extra", "quote": "neutral snippet"},
            ], None

        fetched: list[str] = []

        def getter(url: str) -> dict:
            fetched.append(url)
            return _page(url)

        def complete(system: str, user: str) -> str:
            self.assertNotIn("neutral snippet", user)
            self.assertNotIn("is not reserved", user)
            packet = json.loads(user)
            spans = packet["supplied_spans"]
            self.assertLessEqual(len(spans), 3)
            self.assertGreaterEqual(len(spans), 2)
            self.assertEqual(len({item["source_id"] for item in spans}), len(spans))
            self.assertEqual(len({item["url"] for item in spans}), len(spans))
            self.assertEqual(
                {item["source_id"] for item in packet["sources"]},
                {item["source_id"] for item in spans},
            )
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The fetched sentence states it.",
                    "evidence_refs": spans,
                    "limitations": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "budget", "claim": CLAIM, "human_label": "supported"}],
                search=search,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
                allow_openrouter_fallback=False,
                allow_paid=False,
            )
        self.assertEqual(search_evidence.build_queries(CLAIM), ["RFC 2606 reserves example.com for documentation", "RFC 2606 example.com"])
        self.assertEqual(len(calls), 2)
        claim = result["claims"][0]
        self.assertEqual(len(claim["candidates"]), 5)
        self.assertEqual(len(fetched), 3)
        self.assertIn("https://example.com/oppose", fetched)
        self.assertEqual(claim["search_status"], "ok")
        self.assertEqual(claim["fetch_status"], "ok")
        self.assertEqual(claim["judge_status"], "completed")
        self.assertTrue(claim["evidence_found"])
        self.assertEqual(claim["verdict"], "supported")
        self.assertTrue(claim["citation_passed"])
        for ref in claim["evidence_refs"]:
            passage = next(item for item in claim["passages"] if item["source_id"] == ref["source_id"])
            self.assertEqual(ref["url"], passage["url"])
            self.assertEqual((ref["start"], ref["end"]), (passage["start"], passage["end"]))
        self.assertEqual(result["evaluation"]["claims"][0]["judge_correct"], True)
        self.assertFalse(result["paid_fallback"])

    def test_a_missing_search_service_is_not_a_refutation(self) -> None:
        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [], "No search service is available. A paid search service was not enabled."

        result = search_evidence.investigate(
            [{"id": "none", "claim": PLAIN, "human_label": "supported"}],
            search=search,
            judge="none",
        )
        claim = result["claims"][0]
        self.assertEqual(claim["search_status"], "search_unavailable")
        self.assertEqual(claim["fetch_status"], "not_run")
        self.assertEqual(claim["judge_status"], "not_run")
        self.assertIsNone(claim["verdict"])
        self.assertIn("paid search service was not enabled", claim["limitations"][0])
        self.assertIn("not treated as false", " ".join(claim["limitations"]))
        self.assertFalse(claim["evidence_found"])
        self.assertIsNone(result["evaluation"]["claims"][0]["judge_correct"])

    def test_no_hits_are_source_not_found(self) -> None:
        result = search_evidence.investigate(
            [{"id": "empty", "claim": PLAIN}],
            search=lambda query, limit: ([], None),
            judge="none",
        )
        claim = result["claims"][0]
        self.assertEqual(claim["search_status"], "source_not_found")
        self.assertEqual(claim["fetch_status"], "not_run")
        self.assertEqual(claim["judge_status"], "not_run")
        self.assertFalse(claim["evidence_found"])
        self.assertIsNone(claim["verdict"])
        self.assertNotEqual(claim["verdict"], "refuted")

    def test_a_failed_fetch_is_not_a_refutation(self) -> None:
        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [{"url": "https://example.com/missing", "title": "Missing", "quote": "snippet"}], None

        def getter(url: str) -> dict:
            return {"http_status": 404, "final_url": url, "raw": "", "truncated": False, "error": "HTTP 404"}

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "miss", "claim": CLAIM, "human_label": "supported"}],
                search=search,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=lambda system, user: (_ for _ in ()).throw(AssertionError("the judge was called")),
                cache_dir=Path(tmp),
            )
        claim = result["claims"][0]
        self.assertEqual(claim["fetches"][0]["fetch_status"], "fetch_failed")
        self.assertEqual(claim["search_status"], "ok")
        self.assertEqual(claim["fetch_status"], "fetch_failed")
        self.assertEqual(claim["judge_status"], "not_run")
        self.assertFalse(claim["evidence_found"])
        self.assertIsNone(claim["verdict"])
        self.assertIn("not treated as false", " ".join(claim["limitations"]))

    def test_a_mismatched_quote_fails_validation_without_a_verdict(self) -> None:
        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            span = dict(packet["supplied_spans"][0])
            span["quote"] = "this quote is not on the page"
            return json.dumps(
                {
                    "verdict": "refuted",
                    "explanation": "The page says the opposite.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "bad-cite", "claim": CLAIM, "human_label": "supported"}],
                search=lambda query, limit: ([{"url": "https://example.com/rfc", "title": "RFC", "quote": "snippet"}], None),
                getter=lambda url: _page(url),
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
            )
        claim = result["claims"][0]
        self.assertIsNone(claim["verdict"])
        self.assertIsNone(claim["explanation"])
        self.assertNotEqual(claim["verdict"], "refuted")
        self.assertFalse(claim["citation_passed"])
        self.assertEqual(claim["validation_status"], "validation_failed")
        self.assertEqual(claim["judge_status"], "validation_failed")
        self.assertIn("this quote is not on the page", claim["judge_requests"][0]["raw_reply"])

    def test_a_rate_limit_stops_later_claims(self) -> None:
        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [{"url": "https://example.com/rfc", "title": "RFC", "quote": "snippet"}], None

        def complete(system: str, user: str) -> str:
            raise RuntimeError("429 rate limit")

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [
                    {"id": "first", "claim": CLAIM, "human_label": "supported"},
                    {"id": "second", "claim": CLAIM, "human_label": "refuted"},
                ],
                search=search,
                getter=lambda url: _page(url),
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
                sleeper=lambda _seconds: None,
                rng=lambda: 0,
            )
        self.assertEqual(result["claims"][0]["judge_status"], "rate_limited")
        self.assertIsNone(result["claims"][0]["verdict"])
        self.assertIsNone(result["claims"][0]["citation_passed"])
        self.assertTrue(result["claims"][0]["evidence_found"])
        second = result["claims"][1]
        self.assertEqual(second["search_status"], "ok")
        self.assertEqual(second["fetch_status"], "ok")
        self.assertEqual(second["judge_status"], "not_run")
        self.assertTrue(second["evidence_found"])
        self.assertTrue(second["queries"])
        self.assertTrue(second["passages"])
        self.assertIsNone(second["verdict"])
        self.assertIsNone(result["evaluation"]["claims"][0]["judge_correct"])
        self.assertEqual(result["evaluation"]["search_completed"], 2)
        self.assertEqual(result["evaluation"]["evidence_acquired"], 2)
        self.assertEqual(result["evaluation"]["judge_completed"], 0)

    def test_a_completed_claim_is_not_searched_again(self) -> None:
        saved = {
            "id": "done",
            "claim": PLAIN,
            "human_label": "supported",
            "queries": ["RFC 2606 reserves example.com for documentation", "RFC 2606 example.com"],
            "candidates": [
                {"query": "RFC 2606 reserves example.com for documentation", "url": f"https://example.com/{i}", "title": str(i), "snippet": "s"}
                for i in range(5)
            ],
            "selected": [{"url": "https://example.com/0", "title": "0", "query": "q"}],
            "fetches": [
                {
                    "url": "https://example.com/0",
                    "final_url": "https://example.com/0",
                    "fetch_status": "ok",
                    "passage": {"source_id": "src_x", "url": "https://example.com/0", "start": 0, "end": 3, "quote": "RFC"},
                }
            ],
            "passages": [{"source_id": "src_x", "url": "https://example.com/0", "start": 0, "end": 3, "quote": "RFC"}],
            "evidence_found": True,
            "verdict": "supported",
            "judge_status": "completed",
            "citation_passed": True,
            "explanation": "kept",
            "pipeline_version": search_evidence.PIPELINE_VERSION,
        }
        result = search_evidence.investigate(
            [{"id": "done", "claim": PLAIN, "human_label": "supported"}],
            search=lambda query, limit: (_ for _ in ()).throw(AssertionError("search repeated")),
            getter=lambda url: (_ for _ in ()).throw(AssertionError("page refetched")),
            judge="llm",
            complete=lambda system, user: (_ for _ in ()).throw(AssertionError("judge repeated")),
            previous={"claims": [saved]},
        )
        claim = result["claims"][0]
        self.assertEqual(claim["verdict"], "supported")
        self.assertEqual(claim["explanation"], "kept")
        self.assertEqual(len(claim["candidates"]), 5)
        self.assertEqual(result["evaluation"]["judge_completed"], 1)
        self.assertEqual(result["evaluation"]["judge_accuracy"], "1/1")

    def test_a_search_rate_limit_does_not_start_the_next_search(self) -> None:
        calls: list[str] = []

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            calls.append(query)
            return [], "429 rate limit. Retry after 900s. A paid search service was not enabled."

        result = search_evidence.investigate(
            [
                {"id": "first", "claim": PLAIN},
                {"id": "second", "claim": PLAIN},
            ],
            search=search,
            judge="none",
            sleeper=lambda _seconds: (_ for _ in ()).throw(AssertionError("waited past the cap")),
            rng=lambda: 0,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["claims"][0]["search_status"], "search_rate_limited")
        self.assertEqual(result["claims"][0]["fetch_status"], "not_run")
        self.assertIsNone(result["claims"][0]["evidence_found"])
        self.assertIsNone(result["claims"][0]["verdict"])
        self.assertEqual(result["claims"][1]["search_status"], "not_run")
        self.assertIsNone(result["claims"][1]["evidence_found"])
        self.assertFalse(result["paid_fallback"])

    def test_search_honors_retry_after_then_continues(self) -> None:
        waits: list[float] = []
        calls = {"n": 0}

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            calls["n"] += 1
            if calls["n"] == 1:
                return [], "429 rate limit. Retry after 12s."
            return [{"url": "https://example.com/rfc", "title": "RFC", "quote": "snippet"}], None

        result = search_evidence.investigate(
            [{"id": "retry", "claim": CLAIM}],
            search=search,
            judge="none",
            sleeper=waits.append,
            rng=lambda: 0,
        )
        self.assertEqual(waits, [12.0])
        self.assertEqual(result["claims"][0]["search_status"], "ok")
        self.assertGreaterEqual(len(result["claims"][0]["candidates"]), 1)

    def test_resume_reuses_saved_search_and_pages(self) -> None:
        searches: list[str] = []
        fetches: list[str] = []

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            searches.append(query)
            return [{"url": "https://example.com/rfc", "title": "RFC", "quote": "snippet"}], None

        def getter(url: str) -> dict:
            fetches.append(url)
            return _page(url)

        def blocked(system: str, user: str) -> str:
            raise RuntimeError("429 rate limit")

        cases = [
            {"id": "first", "claim": CLAIM, "human_label": "supported"},
            {"id": "second", "claim": CLAIM, "human_label": "supported"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = search_evidence.investigate(
                cases,
                search=search,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=blocked,
                cache_dir=cache,
                sleeper=lambda _seconds: None,
                rng=lambda: 0,
            )
            searches.clear()
            fetches.clear()

            def finish(system: str, user: str) -> str:
                packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
                return json.dumps(
                    {
                        "verdict": "supported",
                        "explanation": "The fetched sentence states it.",
                        "evidence_refs": packet["supplied_spans"],
                        "limitations": [],
                    }
                )

            second = search_evidence.investigate(
                cases,
                search=lambda query, limit: (_ for _ in ()).throw(AssertionError("search repeated")),
                getter=lambda url: (_ for _ in ()).throw(AssertionError("page refetched")),
                judge="llm",
                model="stub-model",
                complete=finish,
                cache_dir=cache,
                previous=first,
                allow_openrouter_fallback=False,
                allow_paid=False,
            )
        self.assertEqual(searches, [])
        self.assertEqual(fetches, [])
        self.assertEqual(second["claims"][0]["verdict"], "supported")
        self.assertEqual(second["claims"][1]["verdict"], "supported")
        self.assertEqual(second["evaluation"]["judge_completed"], 2)
        self.assertEqual(second["evaluation"]["judge_accuracy"], "2/2")
        self.assertTrue(second["claims"][0]["citation_passed"])
        self.assertFalse(second["openrouter_fallback"])

    def test_a_swapped_source_is_not_a_valid_citation(self) -> None:
        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [
                {"url": "https://example.com/a", "title": "A", "quote": "snippet"},
                {"url": "https://example.com/b", "title": "B", "quote": "snippet"},
            ], None

        def getter(url: str) -> dict:
            text = PLAIN if url.endswith("/a") else f"Note. {PLAIN}"
            return {
                "http_status": 200,
                "final_url": url,
                "raw": f"<html><title>T</title><body><p>{text}</p></body></html>",
                "truncated": False,
                "error": None,
            }

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            spans = packet["supplied_spans"]
            swapped = []
            for span, other in zip(spans, reversed(spans)):
                bad = dict(span)
                bad["source_id"] = other["source_id"]
                bad["url"] = other["url"]
                swapped.append(bad)
            return json.dumps(
                {
                    "verdict": "refuted",
                    "explanation": "The other page says so.",
                    "evidence_refs": swapped,
                    "limitations": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "swap", "claim": PLAIN, "human_label": "supported"}],
                search=search,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
            )
        claim = result["claims"][0]
        self.assertEqual(len(claim["passages"]), 2)
        self.assertIsNone(claim["verdict"])
        self.assertIsNone(claim["explanation"])
        self.assertNotEqual(claim["verdict"], "refuted")
        self.assertFalse(claim["citation_passed"])
        self.assertEqual(claim["validation_status"], "validation_failed")
        self.assertEqual([item["purpose"] for item in claim["judge_requests"]], ["initial", "repair"])

    def test_a_named_site_is_fetched_before_search_results(self) -> None:
        claim_text = "example.com states that this domain is for use in documentation examples without needing permission."
        sentence = "This domain is for use in documentation examples without needing permission."
        fetched: list[str] = []

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "quote": sentence}], None

        def getter(url: str) -> dict:
            fetched.append(url)
            return {
                "http_status": 200,
                "final_url": url if url.startswith("http") else f"https://{url}",
                "raw": f"<html><title>T</title><body><p>{sentence}</p></body></html>",
                "truncated": False,
                "error": None,
            }

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            site = next(span for span in packet["supplied_spans"] if span["url"].startswith("https://example.com"))
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The named site states it.",
                    "evidence_refs": [site],
                    "limitations": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "named", "claim": claim_text, "human_label": "supported", "human_basis": "The example.com page states that sentence."}],
                search=search,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
            )
        claim = result["claims"][0]
        self.assertEqual(search_evidence._host(claim["selected"][0]["url"]), "example.com")
        self.assertEqual(search_evidence._host(fetched[0]), "example.com")
        self.assertTrue(claim["key_evidence_obtained"])
        self.assertEqual(claim["verdict"], "supported")
        self.assertEqual(claim["attribution_check"], "passed")
        self.assertTrue(result["evaluation"]["claims"][0]["end_to_end_match"])

    def test_a_failed_named_source_is_not_replaced_by_a_third_party(self) -> None:
        claim_text = "example.com states that this domain is for use in documentation examples without needing permission."
        sentence = "This domain is for use in documentation examples without needing permission."

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "quote": sentence}], None

        def getter(url: str) -> dict:
            if search_evidence._host(url) == "example.com":
                return {"http_status": 404, "final_url": url, "raw": "", "truncated": False, "error": "HTTP 404"}
            return {
                "http_status": 200,
                "final_url": url,
                "raw": f"<html><title>Wiki</title><body><p>{sentence}</p></body></html>",
                "truncated": False,
                "error": None,
            }

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            wiki = next(span for span in packet["supplied_spans"] if "wikipedia" in span["url"])
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "A wiki page says the same thing.",
                    "evidence_refs": [wiki],
                    "limitations": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "third", "claim": claim_text, "human_label": "supported"}],
                search=search,
                getter=getter,
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
            )
        claim = result["claims"][0]
        direct = next(item for item in claim["fetches"] if search_evidence._host(item["url"]) == "example.com")
        self.assertEqual(direct["fetch_status"], "fetch_failed")
        self.assertEqual(direct["error"], "HTTP 404")
        self.assertIn("could not be read", " ".join(claim["limitations"]))
        self.assertEqual(claim["verdict"], "insufficient_evidence")
        self.assertNotEqual(claim["verdict"], "refuted")
        self.assertEqual(claim["attribution_check"], "failed")
        self.assertEqual(claim["content_check"], "supported")
        self.assertFalse(claim["key_evidence_obtained"])
        self.assertFalse(result["evaluation"]["claims"][0]["end_to_end_match"])

    def test_an_rfc_statement_reads_the_editor_before_a_summary(self) -> None:
        claim_text = "RFC 2606 states that top level domain names are not reserved for use in documentation."
        snippet = "Platform summary: the RFC reserves documentation names."
        official = "a few top level domain names are reserved for use in documentation"

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            return [{"url": "https://grokipedia.com/page/RFC_2606", "title": "Overview", "quote": snippet}], None

        def getter(url: str) -> dict:
            body = official if "rfc-editor.org" in url or "ietf.org" in url else "A third-party overview of the RFC."
            return {
                "http_status": 200,
                "final_url": url,
                "raw": f"<html><title>RFC 2606</title><body><p>{body}</p></body></html>",
                "truncated": False,
                "error": None,
            }

        result = search_evidence.investigate(
            [{"id": "rfc", "claim": claim_text}],
            search=search,
            getter=getter,
            judge="none",
        )
        claim = result["claims"][0]
        selected = [item["url"] for item in claim["selected"]]
        self.assertTrue(selected[0].startswith("https://www.rfc-editor.org/rfc/rfc2606"))
        self.assertIn("datatracker.ietf.org", selected[1])
        quotes = " ".join(str((item.get("passage") or {}).get("quote") or "") for item in claim["fetches"])
        self.assertIn(official, quotes)
        self.assertNotIn(snippet, quotes)
        self.assertNotIn("Platform summary", quotes)
        missed = search_evidence.investigate(
            [{"id": "miss", "claim": claim_text}],
            search=lambda query, limit: ([], None),
            getter=getter,
            judge="none",
        )
        self.assertTrue(missed["claims"][0]["selected"][0]["url"].startswith("https://www.rfc-editor.org/rfc/rfc2606"))

    def test_an_explicit_url_is_fetched_and_a_bare_title_is_not_invented(self) -> None:
        self.assertEqual(search_evidence.named_sources("RFC 2606 is widely cited."), [])
        rfc = search_evidence.named_sources("RFC 2606 states that names are reserved.")
        self.assertEqual([item["host"] for item in rfc], ["rfc-editor.org", "datatracker.ietf.org"])
        self.assertTrue(all(item["role"] == "original" for item in rfc))
        cern = search_evidence.named_sources("http://info.cern.ch is the home of the first website.")
        self.assertEqual(cern[0]["host"], "info.cern.ch")
        self.assertFalse(cern[0]["attributed"])
        result = search_evidence.investigate(
            [{"id": "url", "claim": "http://info.cern.ch is the home of the first website."}],
            search=lambda query, limit: ([{"url": "https://home.cern/", "title": "CERN", "quote": "snippet"}], None),
            getter=lambda url: {
                "http_status": 200,
                "final_url": url,
                "raw": "<html><title>T</title><body><p>http://info.cern.ch is the home of the first website.</p></body></html>",
                "truncated": False,
                "error": None,
            },
            judge="none",
        )
        self.assertEqual(result["claims"][0]["selected"][0]["url"], "http://info.cern.ch")

    def test_only_updates_the_requested_claim(self) -> None:
        saved = {
            "id": "keep",
            "claim": CLAIM,
            "queries": ["RFC 2606 reserves example.com for documentation"],
            "candidates": [{"url": "https://example.com/rfc", "title": "RFC", "query": "q"}],
            "selected": [{"url": "https://example.com/rfc", "title": "RFC", "query": "q"}],
            "fetches": [],
            "passages": [],
            "evidence_found": False,
            "verdict": None,
            "search_status": "ok",
            "fetch_status": "not_run",
            "judge_status": "rate_limited",
            "pipeline_version": search_evidence.PIPELINE_VERSION,
        }
        calls: list[str] = []

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
            calls.append(query)
            return [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "quote": "snippet"}], None

        result = search_evidence.investigate(
            [
                {"id": "keep", "claim": CLAIM},
                {"id": "fill", "claim": "example.com states that this domain is for use in documentation examples without needing permission."},
            ],
            search=search,
            getter=lambda url: {
                "http_status": 200,
                "final_url": url,
                "raw": "<html><title>T</title><body><p>example.com states that this domain is for use in documentation examples without needing permission.</p></body></html>",
                "truncated": False,
                "error": None,
            },
            judge="none",
            previous={"claims": [saved]},
            only={"fill"},
        )
        self.assertEqual(result["claims"][0]["judge_status"], "rate_limited")
        self.assertEqual(result["claims"][0]["candidates"][0]["url"], "https://example.com/rfc")
        self.assertEqual(search_evidence._host(result["claims"][1]["selected"][0]["url"]), "example.com")
        self.assertTrue(calls)

    def test_a_verdict_without_the_named_site_is_judged_again(self) -> None:
        claim_text = "example.com states that this domain is for use in documentation examples without needing permission."
        sentence = "This domain is for use in documentation examples without needing permission."
        saved = {
            "id": "again",
            "claim": claim_text,
            "human_label": "supported",
            "queries": [{"query": claim_text, "status": "ok", "error": None}],
            "candidates": [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "query": "q"}],
            "selected": [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "query": "q"}],
            "fetches": [
                {
                    "url": "https://en.wikipedia.org/wiki/Example.com",
                    "final_url": "https://en.wikipedia.org/wiki/Example.com",
                    "fetch_status": "ok",
                    "passage": {"source_id": "src_wiki", "url": "https://en.wikipedia.org/wiki/Example.com", "start": 0, "end": 10, "quote": "wiki text"},
                }
            ],
            "passages": [{"source_id": "src_wiki", "url": "https://en.wikipedia.org/wiki/Example.com", "start": 0, "end": 10, "quote": "wiki text"}],
            "evidence_found": True,
            "verdict": "insufficient_evidence",
            "judge_status": "completed",
            "search_status": "ok",
            "fetch_status": "ok",
            "explanation": "old third-party judgment",
            "pipeline_version": search_evidence.PIPELINE_VERSION,
        }

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            site = next(span for span in packet["supplied_spans"] if span["url"].startswith("https://example.com"))
            return json.dumps(
                {"verdict": "supported", "explanation": "The site states it.", "evidence_refs": [site], "limitations": []}
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = search_evidence.investigate(
                [{"id": "again", "claim": claim_text, "human_label": "supported"}],
                search=lambda query, limit: (_ for _ in ()).throw(AssertionError("search repeated")),
                getter=lambda url: {
                    "http_status": 200,
                    "final_url": url,
                    "raw": f"<html><title>Example Domain</title><body><p>{sentence}</p></body></html>",
                    "truncated": False,
                    "error": None,
                },
                judge="llm",
                model="stub-model",
                complete=complete,
                cache_dir=Path(tmp),
                previous={"claims": [saved]},
            )
        claim = result["claims"][0]
        self.assertEqual(search_evidence._host(claim["fetches"][0]["final_url"]), "example.com")
        self.assertEqual(claim["verdict"], "supported")
        self.assertTrue(claim["key_evidence_obtained"])

    def test_an_old_source_list_is_selected_again_without_clearing_page_cache(self) -> None:
        claim_text = "example.com states that this domain is for use in documentation examples without needing permission."
        sentence = "This domain is for use in documentation examples without needing permission."
        saved = {
            "id": "old",
            "claim": claim_text,
            "pipeline_version": "source-selection-1",
            "created_at": "2026-11-16T00:00:00+00:00",
            "queries": [{"query": "old query", "status": "ok", "error": None}],
            "candidates": [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "query": "old query"}],
            "selected": [{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "query": "old query"}],
            "fetches": [
                {
                    "url": "https://en.wikipedia.org/wiki/Example.com",
                    "final_url": "https://en.wikipedia.org/wiki/Example.com",
                    "fetch_status": "ok",
                    "passage": {"source_id": "src_wiki", "url": "https://en.wikipedia.org/wiki/Example.com", "start": 0, "end": 4, "quote": "wiki"},
                }
            ],
            "search_status": "ok",
            "fetch_status": "ok",
            "judge_status": "completed",
            "verdict": "supported",
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            page_path = acquire_url._page_cache_path(cache, "https://example.com")
            page_path.parent.mkdir(parents=True)
            page_path.write_text(
                json.dumps(
                    {
                        "original_url": "https://example.com",
                        "final_url": "https://example.com/",
                        "title": "Example Domain",
                        "fetched_at": "2026-11-16T00:00:00Z",
                        "http_status": 200,
                        "body": sentence,
                        "content_sha256": "cached",
                        "truncated": False,
                        "fetch_status": "ok",
                        "error": None,
                        "source_id": "src_example",
                    }
                ),
                encoding="utf-8",
            )
            cached_bytes = page_path.read_bytes()

            def getter(url: str) -> dict:
                if url == "https://example.com":
                    raise AssertionError("cached page was fetched again")
                return {
                    "http_status": 200,
                    "final_url": url,
                    "raw": f"<html><title>Wiki</title><body><p>{sentence}</p></body></html>",
                    "truncated": False,
                    "error": None,
                }

            result = search_evidence.investigate(
                [{"id": "old", "claim": claim_text}],
                search=lambda query, limit: ([{"url": "https://en.wikipedia.org/wiki/Example.com", "title": "Wiki", "quote": "overview"}], None),
                getter=getter,
                judge="none",
                cache_dir=cache,
                previous={"claims": [saved]},
            )
            claim = result["claims"][0]
            self.assertEqual(page_path.read_bytes(), cached_bytes)
            self.assertEqual(claim["created_at"], "2026-11-16T00:00:00+00:00")
            self.assertEqual(claim["pipeline_version"], search_evidence.PIPELINE_VERSION)
            self.assertEqual(claim["cache_version"], search_evidence.CACHE_VERSION)
            self.assertEqual(claim["selected"][0]["url"], "https://example.com")
            named = next(item for item in claim["fetches"] if item["url"] == "https://example.com")
            self.assertEqual(named["page_cache"], "hit")
            self.assertEqual(named["fetch_status"], "ok")
            self.assertNotEqual(result["paid_fallback"], True)

    def test_a_provider_cooldown_blocks_the_next_model_call(self) -> None:
        calls = {"n": 0}
        searches: list[str] = []
        first = "The recorded count was 12 lanes in 1998."
        second = "The recorded span was 14 lanes in 1999."
        body = f"{first} {second}"

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searches.append(query)
            return [{"url": "https://example.com/lanes", "title": "Lanes", "quote": "snippet"}], None

        def complete(system: str, user: str) -> str:
            calls["n"] += 1
            error = RuntimeError("429 rate limit")
            error.retry_after = 120
            error.status_code = 429
            raise error

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            kwargs = {
                "search": search,
                "getter": lambda url: {
                    "http_status": 200,
                    "final_url": url,
                    "raw": f"<html><title>Lanes</title><body><p>{body}</p></body></html>",
                    "truncated": False,
                    "error": None,
                },
                "judge": "llm",
                "model": "stub-model",
                "complete": complete,
                "cache_dir": cache,
                "sleeper": lambda seconds: None,
                "rng": lambda: 0,
            }
            earlier = search_evidence.investigate([{"id": "one", "claim": first}], **kwargs)
            after_first = len(searches)
            later = search_evidence.investigate([{"id": "two", "claim": second}], **kwargs)
        self.assertEqual(earlier["claims"][0]["judge_status"], "rate_limited")
        self.assertEqual(calls["n"], 2)
        self.assertGreater(len(searches), after_first)
        self.assertEqual(later["claims"][0]["judge_status"], "rate_limited")
        self.assertEqual(later["claims"][0]["provider"], "groq")
        self.assertEqual(later["claims"][0]["model"], "stub-model")
        self.assertEqual(later["claims"][0]["fetch_status"], "ok")
        self.assertTrue(later["claims"][0]["retry_at"])
        self.assertFalse(later["paid_fallback"])
        self.assertFalse(later["openrouter_fallback"])

    def test_a_named_page_is_kept_when_the_claim_words_are_absent(self) -> None:
        claim_text = "example.com states that commercial advertising is permitted without permission."
        sentence = "This domain is for use in documentation examples without needing permission."
        result = search_evidence.investigate(
            [{"id": "ad", "claim": claim_text}],
            search=lambda query, limit: ([], None),
            getter=lambda url: {
                "http_status": 200,
                "final_url": url,
                "raw": f"<html><title>Example Domain</title><body><p>{sentence}</p></body></html>",
                "truncated": False,
                "error": None,
            },
            judge="none",
        )
        claim = result["claims"][0]
        self.assertEqual(claim["selected"][0]["url"], "https://example.com")
        passage = claim["fetches"][0]["passage"]
        self.assertIn(sentence, passage["quote"])
        self.assertTrue(claim["evidence_found"])


if __name__ == "__main__":
    unittest.main()
