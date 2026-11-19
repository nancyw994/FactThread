"""Article mode extracts claims, then checks at most three without rerunning finished ones."""

from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "web"))

import check_server  # noqa: E402

ARTICLE = (
    "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. "
    "The World Wide Web was invented in 2005."
)
OTHER = "The recorded count was 12 lanes in 1998."
SHORT_ARTICLE = (
    "Apollo 11 landed the first humans on the Moon on July 20, 1969. "
    "The first Moon landing happened in 1972. "
    "Apollo 11 was humanity's greatest achievement. "
    "NASA will establish a permanent lunar base by 2030."
)


def _extract(text: str, repair: list[str] | None) -> dict:
    claims = []
    for sentence in [part.strip() for part in text.split(".") if part.strip()]:
        quote = sentence + "."
        claims.append(
            {
                "source_quote": quote,
                "normalized_claim": quote,
                "category": "checkable_fact",
                "missing_context": ["which product the price applies to"] if "2005" in quote else [],
            }
        )
    return {"claims": claims}


def _page(url: str) -> dict:
    body = ARTICLE + " Tim Berners-Lee invented the World Wide Web in 1991. " + OTHER
    return {
        "http_status": 200,
        "final_url": url,
        "raw": f"<html><title>Source</title><body><p>{body}</p></body></html>",
        "truncated": False,
        "error": None,
    }


def _hits(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
    return [{"url": "https://example.com/source", "title": "Source", "quote": "snippet"}], None


class ArticleModeTests(unittest.TestCase):
    def test_article_extraction_requests_json_object_without_fallbacks(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        observed = {}

        def configure_providers(**options):
            observed["provider_options"] = options

        def ask_llm_usage(spec, user, **options):
            observed["response_format"] = options.get("response_format")
            observed["max_completion_tokens"] = options.get("max_completion_tokens")
            return {"content": '{"claims": []}', "finish_reason": "stop", "usage": {}}

        fake_client = SimpleNamespace(configure_providers=configure_providers, ask_llm_usage=ask_llm_usage)
        with patch.dict(sys.modules, {"llm_client": fake_client}):
            result = check_server.article_mode.groq_extract(ARTICLE, None)
        self.assertEqual(result, {"claims": []})
        self.assertEqual(observed["provider_options"], {"allow_openrouter_fallback": False, "allow_paid": False})
        self.assertEqual(observed["response_format"], {"type": "json_object"})
        self.assertEqual(observed["max_completion_tokens"], 4096)

    def test_unparseable_model_text_is_a_failed_extraction(self) -> None:
        with self.assertRaises(check_server.article_mode.extract_claims.ExtractionParseError):
            check_server.article_mode._parse_extract("model commentary with no JSON object")

        def parse_failure(text, repair):
            raise check_server.article_mode.extract_claims.ExtractionParseError("invalid JSON")

        book = check_server.article_mode.ArticleBook(view=check_server.public_view, extract=parse_failure)
        snap = book.start_extract(ARTICLE)
        snap = self._wait_snapshot(book, snap["id"])
        self.assertEqual(snap["status"], "error")
        self.assertEqual(snap["extraction_status"], "parse_failed")

    @staticmethod
    def _wait_snapshot(book, article_id):
        deadline = time.time() + 3
        while time.time() < deadline:
            snap = book.snapshot(article_id)
            if snap["status"] != "running":
                return snap
            time.sleep(0.01)
        raise AssertionError("article extraction did not finish")

    def test_extract_shows_quote_claim_type_and_missing_context(self) -> None:
        with self._server() as base:
            job = self._post(base, "/api/articles", {"text": ARTICLE, "allow_paid": True})
            done = self._wait(base, job["id"])
        self.assertEqual(done["status"], "extracted")
        self.assertFalse(done["paid_fallback"])
        self.assertFalse(done["openrouter_fallback"])
        self.assertEqual(done["paid_label"], "本次不允许付费模型。")
        self.assertEqual(len(done["candidates"]), 2)
        first, second = done["candidates"]
        self.assertEqual(first["source_quote"], "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN.")
        first_range = first["source_ranges"][0]
        self.assertEqual(ARTICLE[first_range[0]:first_range[1]], first["source_quote"])
        self.assertIn("1989", first["normalized_claim"])
        self.assertEqual(first["fact_type"], "checkable_fact")
        self.assertEqual(first["fact_type_label"], "可核查事实")
        self.assertEqual(first["missing_context"], [])
        self.assertEqual(second["missing_context"], ["which product the price applies to"])
        self.assertNotIn("human_label", json.dumps(done))

    def test_short_article_defaults_opinion_and_prediction_off_and_checks_only_selected(self) -> None:
        quotes = [part.strip().rstrip(".") + "." for part in SHORT_ARTICLE.split(". ")]

        def extract(text: str, repair: list[str] | None) -> dict:
            categories = ["checkable_fact", "checkable_fact", "opinion", "prediction"]
            return {
                "claims": [
                    {
                        "source_quote": quote,
                        "normalized_claim": quote,
                        "category": category,
                        "missing_context": [],
                    }
                    for quote, category in zip(quotes, categories)
                ]
            }

        searched: list[str] = []
        judged: list[str] = []

        def get_page(url: str) -> dict:
            body = (
                "Apollo 11 landed the first humans on the Moon on July 20, 1969. "
                "The first Moon landing was in 1969, not 1972."
            )
            return {
                "http_status": 200,
                "final_url": url,
                "raw": f"<html><title>NASA</title><body><p>{body}</p></body></html>",
                "truncated": False,
                "error": None,
            }

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searched.append(query)
            return _hits(query, limit)

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            judged.append(packet["claim"])
            span = packet["supplied_spans"][0]
            verdict = "refuted" if "1972" in packet["claim"] else "supported"
            return json.dumps(
                {
                    "verdict": verdict,
                    "explanation": "Compared with the fetched source.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        with self._server(search, complete, extract=extract, getter=get_page) as base:
            job = self._wait(base, self._post(base, "/api/articles", {"text": SHORT_ARTICLE})["id"])
            self.assertEqual([row["source_quote"] for row in job["candidates"]], quotes)
            self.assertEqual([row["normalized_claim"] for row in job["candidates"]], quotes)
            self.assertEqual([row["fact_type"] for row in job["candidates"]], [
                "checkable_fact", "checkable_fact", "opinion", "prediction"
            ])
            self.assertEqual([row["default_selected"] for row in job["candidates"]], [True, True, False, False])
            selected = [row["id"] for row in job["candidates"][:2]]
            checked = self._wait(
                base,
                self._post(base, f"/api/articles/{job['id']}/check", {"ids": selected})["id"],
            )

        self.assertEqual(judged, [quotes[0], quotes[1]], checked)
        self.assertTrue(searched)
        self.assertFalse(any("achievement" in query.lower() or "2030" in query for query in searched))
        self.assertEqual([item["verdict"] for item in checked["claims"]], ["supported", "refuted"])
        self.assertTrue(all(item["citations"] for item in checked["claims"]))
        self.assertFalse(checked["paid_fallback"])
        self.assertFalse(checked["openrouter_fallback"])

    def test_a_changed_article_drops_the_old_extraction(self) -> None:
        with self._server() as base:
            first = self._wait(base, self._post(base, "/api/articles", {"text": ARTICLE})["id"])
            second = self._wait(base, self._post(base, "/api/articles", {"text": OTHER})["id"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(second["candidates"]), 1)
        self.assertNotIn("CERN", json.dumps(second["candidates"]))
        self.assertIn("12 lanes", second["candidates"][0]["source_quote"])

    def test_more_than_three_claims_are_rejected(self) -> None:
        text = "One is 1. Two is 2. Three is 3. Four is 4."
        with self._server() as base:
            job = self._wait(base, self._post(base, "/api/articles", {"text": text})["id"])
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self._post(base, f"/api/articles/{job['id']}/check", {"ids": [item["id"] for item in job["candidates"]]})
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("3", caught.exception.read().decode("utf-8"))

    def test_selected_claims_are_checked_in_order_and_finished_ones_stay(self) -> None:
        searches: list[str] = []
        judged: list[str] = []
        blocked = {"value": True}

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searches.append(query)
            return _hits(query, limit)

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            judged.append(packet["claim"])
            if blocked["value"] and "2005" in packet["claim"]:
                error = RuntimeError("429 rate limit")
                error.status_code = 429
                raise error
            span = packet["supplied_spans"][0]
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The fetched page states it.",
                    "evidence_refs": [span],
                    "limitations": [],
                    "debug": "SECRET-RAW-REPLY",
                }
            )

        with self._server(search, complete) as base:
            job = self._wait(base, self._post(base, "/api/articles", {"text": ARTICLE})["id"])
            ids = [item["id"] for item in job["candidates"]]
            started = self._post(base, f"/api/articles/{job['id']}/check", {"ids": ids, "allow_paid": True})
            held = self._wait(base, started["id"])
            self.assertTrue(held["can_continue"])
            self.assertEqual(held["claims"][0]["verdict"], "supported")
            self.assertIsNone(held["claims"][1]["verdict"])
            self.assertNotIn("SECRET-RAW-REPLY", json.dumps(held))
            searched = len(searches)
            judged_before = list(judged)
            blocked["value"] = False
            continued = self._post(base, f"/api/articles/{job['id']}/continue", {})
            done = self._wait(base, continued["id"])
        first_at = next(index for index, query in enumerate(searches) if query.startswith("Tim Berners-Lee"))
        later_fact_at = next(index for index, query in enumerate(searches) if "invented" in query and "2005" in query)
        self.assertLess(first_at, later_fact_at)
        self.assertEqual(len(searches), searched)
        self.assertEqual(done["claims"][0]["verdict"], "supported")
        self.assertEqual(done["claims"][1]["verdict"], "supported")
        self.assertIn("https://example.com/source", json.dumps(done["claims"][1]["citations"]))
        self.assertTrue(all("2005" in item for item in judged[len(judged_before) :]))
        self.assertFalse(done["paid_fallback"])
        self.assertFalse(done["can_continue"])

    def test_an_edited_claim_is_not_given_the_old_verdict(self) -> None:
        seen: list[str] = []

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            seen.append(packet["claim"])
            span = packet["supplied_spans"][0]
            return json.dumps(
                {
                    "verdict": "supported" if "1991" not in packet["claim"] else "refuted",
                    "explanation": packet["claim"],
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        with self._server(complete=complete) as base:
            job = self._wait(base, self._post(base, "/api/articles", {"text": ARTICLE})["id"])
            first_id = job["candidates"][0]["id"]
            self._wait(base, self._post(base, f"/api/articles/{job['id']}/check", {"ids": [first_id]})["id"])
            edited = "Tim Berners-Lee invented the World Wide Web in 1991."
            again = self._wait(
                base,
                self._post(
                    base,
                    f"/api/articles/{job['id']}/check",
                    {"ids": [first_id], "claims": {first_id: edited}},
                )["id"],
            )
        self.assertEqual(again["claims"][0]["claim"], edited)
        self.assertEqual(again["claims"][0]["verdict"], "refuted")
        self.assertIn(edited, seen)

    def test_saved_article_and_completed_claim_survive_server_restart(self) -> None:
        import tempfile

        searches: list[str] = []
        blocked = {"value": True}

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searches.append(query)
            return _hits(query, limit)

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            if blocked["value"] and "2005" in packet["claim"]:
                error = RuntimeError("429 rate limit")
                error.status_code = 429
                raise error
            span = packet["supplied_spans"][0]
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The page states it.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = check_server.App(
                search=search, getter=_page, complete=complete, extract=_extract,
                jobs_dir=root / "jobs", cache_dir=root / "pages",
                sleeper=lambda seconds: None, rng=lambda: 0,
            )
            job = first.articles.start_extract(ARTICLE)
            job = _wait_for(first, job["id"])
            ids = [item["id"] for item in job["candidates"]]
            first.articles.start_check(job["id"], ids, {})
            held = _wait_for(first, job["id"])
            self.assertEqual(held["claims"][0]["verdict"], "supported")
            self.assertTrue(held["can_continue"])
            searched = len(searches)

            blocked["value"] = False
            restored = check_server.App(
                search=search, getter=_page, complete=complete,
                jobs_dir=root / "jobs", cache_dir=root / "pages",
                sleeper=lambda seconds: None, rng=lambda: 0,
            )
            loaded = restored.articles.snapshot(job["id"])
            self.assertEqual(loaded["claims"][0]["verdict"], "supported")
            restored.articles.continue_article(job["id"])
            done = _wait_for(restored, job["id"])

        self.assertEqual(len(searches), searched)
        self.assertEqual([item["verdict"] for item in done["claims"]], ["supported", "supported"])

    def test_legacy_saved_article_backfills_duplicate_offsets_and_causal_gaps(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = "These measures promoted development. More context. These measures promoted development."
            quote = "These measures promoted development."
            article_id = "saved"
            article_file = root / "articles" / f"{article_id}.json"
            article_file.parent.mkdir(parents=True)
            article_file.write_text(json.dumps({
                "id": article_id,
                "text": text,
                "text_sha": check_server.article_mode._sha(text),
                "candidates": [{
                    "id": "c1",
                    "source_quote": quote,
                    "normalized_claim": quote,
                    "fact_type": "checkable_fact",
                    "missing_context": [],
                }],
                "selection": [],
                "records": {},
                "status": "extracted",
            }), encoding="utf-8")
            app = check_server.App(jobs_dir=root / "jobs", cache_dir=root / "cache")
            snapshot = app.articles.snapshot(article_id)
            candidate = snapshot["candidates"][0]
            self.assertEqual(len(candidate["source_ranges"]), 2)
            self.assertFalse(candidate["default_selected"])
            self.assertIn("需要明确因果关系所指的行动或变化", candidate["missing_context"])
            persisted = json.loads(article_file.read_text(encoding="utf-8"))
            self.assertEqual(persisted["candidates"][0]["source_ranges"], candidate["source_ranges"])

    def _server(self, search=None, complete=None, extract=None, getter=None):
        return _Running(search or _hits, complete or _supported, extract=extract, getter=getter)

    def _post(self, base: str, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 202)
            return json.load(response)

    def _wait(self, base: str, article_id: str) -> dict:
        deadline = time.time() + 5
        last = {}
        while time.time() < deadline:
            with urllib.request.urlopen(base + f"/api/articles/{article_id}") as response:
                last = json.load(response)
            if last["status"] != "running":
                self.assertIn(last["status"], {"extracted", "done"}, last.get("error"))
                return last
            time.sleep(0.02)
        self.fail(f"article stayed running: {last}")


def _supported(system: str, user: str) -> str:
    packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
    span = packet["supplied_spans"][0]
    return json.dumps(
        {
            "verdict": "supported",
            "explanation": "The fetched page states it.",
            "evidence_refs": [span],
            "limitations": [],
        }
    )


def _wait_for(app: check_server.App, article_id: str) -> dict:
    deadline = time.time() + 5
    last = {}
    while time.time() < deadline:
        last = app.articles.snapshot(article_id)
        if last["status"] != "running":
            if last["status"] == "error":
                raise AssertionError(last.get("error"))
            return last
        time.sleep(0.02)
    raise AssertionError(f"article stayed running: {last}")


class _Running:
    def __init__(self, search, complete, extract=None, getter=None) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.app = check_server.App(
            search=search,
            getter=getter or _page,
            complete=complete,
            extract=extract or _extract,
            jobs_dir=root / "jobs",
            cache_dir=root / "pages",
            sleeper=lambda seconds: None,
            rng=lambda: 0,
        )
        self.server = check_server.serve(self.app, 0)
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def __enter__(self) -> str:
        return self.base

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()
