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
        self.assertIn("1989", first["normalized_claim"])
        self.assertEqual(first["fact_type"], "checkable_fact")
        self.assertEqual(first["fact_type_label"], "可核查事实")
        self.assertEqual(first["missing_context"], [])
        self.assertEqual(second["missing_context"], ["which product the price applies to"])
        self.assertNotIn("human_label", json.dumps(done))

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
        self.assertLess(first_at, searches.index("The World Wide Web was invented in 2005"))
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

    def _server(self, search=None, complete=None):
        return _Running(search or _hits, complete or _supported)

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


class _Running:
    def __init__(self, search, complete) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.app = check_server.App(
            search=search,
            getter=_page,
            complete=complete,
            extract=_extract,
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
