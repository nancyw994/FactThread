"""The local page calls the existing search pipeline and hides human labels."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "web"))

import check_server  # noqa: E402

SENTENCE = "RFC 2606 reserves example.com for documentation."
PAGE = f"<html><title>RFC 2606</title><body><p>{SENTENCE}</p></body></html>"
URL = "https://example.com/rfc2606"


def _hits() -> tuple[list[dict[str, str]], None]:
    return [{"url": URL, "title": "RFC 2606", "quote": "snippet that the judge must not see"}], None


class CheckPageTests(unittest.TestCase):
    def test_a_supported_result_shows_the_quote_while_search_is_running(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            started.set()
            self.assertTrue(release.wait(2))
            return _hits()

        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            span = packet["supplied_spans"][0]
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The fetched sentence states the reservation.",
                    "evidence_refs": [span],
                    "limitations": ["A third-party summary was not used."],
                }
            )

        with self._server(search, complete) as base:
            job = self._post(base, "/api/jobs", {"claim": SENTENCE, "allow_paid": True, "human_label": "refuted"})
            self.assertTrue(started.wait(2))
            running = self._get(base, f"/api/jobs/{job['id']}")
            self.assertEqual(running["stages"]["search"]["label"], "进行中")
            self.assertEqual(running["stages"]["fetch"]["label"], "未开始")
            self.assertEqual(running["stages"]["judge"]["label"], "未开始")
            self.assertNotIn("human_label", json.dumps(running))
            release.set()
            done = self._wait(base, job["id"])
        self.assertEqual(done["verdict"], "supported")
        self.assertEqual(done["verdict_label"], "支持")
        self.assertEqual(done["explanation"], "The fetched sentence states the reservation.")
        self.assertEqual(done["citations"][0]["quote"], SENTENCE)
        self.assertEqual(done["citations"][0]["title"], "RFC 2606")
        self.assertIn("rfc-editor.org/rfc/rfc2606", done["citations"][0]["url"])
        self.assertEqual(done["citations"][0]["role_label"], "原始文档")
        self.assertTrue(done["citations"][0]["cited"])
        self.assertNotIn("snippet that the judge must not see", json.dumps(done))
        self.assertTrue(any("不是完整文档" in line for line in done["scope"]))
        self.assertIn("A third-party summary was not used.", done["limitations"])
        self.assertFalse(done["openrouter_fallback"])
        self.assertFalse(done["paid_fallback"])
        self.assertEqual(done["paid_label"], "本次不允许付费模型。")
        self.assertEqual(done["openrouter_label"], "本次不允许调用 OpenRouter。")
        self.assertNotIn("human_label", json.dumps(done))
        self.assertNotIn("confidence", json.dumps(done))

    def test_insufficient_evidence_is_a_result_not_a_score(self) -> None:
        def complete(system: str, user: str) -> str:
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            return json.dumps(
                {
                    "verdict": "insufficient_evidence",
                    "explanation": "The page reserves the name and does not state a launch year.",
                    "evidence_refs": [packet["supplied_spans"][0]],
                    "limitations": [],
                }
            )

        with self._server(lambda query, limit: _hits(), complete) as base:
            job = self._post(base, "/api/jobs", {"claim": SENTENCE})
            done = self._wait(base, job["id"])
        self.assertEqual(done["verdict"], "insufficient_evidence")
        self.assertEqual(done["verdict_label"], "证据不足")
        self.assertFalse(done["can_continue"])
        self.assertNotIn("%", done["verdict_label"])
        self.assertNotIn("confidence", json.dumps(done))

    def test_a_rate_limit_can_continue_without_searching_again(self) -> None:
        searches: list[str] = []
        fetches: list[str] = []
        calls = {"n": 0}

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searches.append(query)
            return _hits()

        def getter(url: str) -> dict:
            fetches.append(url)
            return {"http_status": 200, "final_url": url, "raw": PAGE, "truncated": False, "error": None}

        def complete(system: str, user: str) -> str:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("429 rate limit")
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The saved page states the reservation.",
                    "evidence_refs": [packet["supplied_spans"][0]],
                    "limitations": [],
                }
            )

        with self._server(search, complete, getter=getter) as base:
            job = self._post(base, "/api/jobs", {"claim": SENTENCE})
            waiting = self._wait(base, job["id"])
            self.assertEqual(waiting["verdict"], None)
            self.assertEqual(waiting["verdict_label"], "判断：等待重试")
            self.assertEqual(waiting["status_note"], "模型服务暂时限流，证据已保存")
            self.assertEqual(waiting["stages"]["judge"]["label"], "等待重试")
            self.assertEqual(waiting["stages"]["search"]["label"], "完成")
            self.assertTrue(waiting["can_continue"])
            self.assertFalse(waiting["retry_known"])
            self.assertIsNone(waiting["retry_at"])
            self.assertEqual(waiting["evidence_heading"], "已获取的证据")
            self.assertIsNone(waiting["explanation"])
            self.assertNotIn("not treated as false", " ".join(waiting["limitations"]))
            self.assertTrue(waiting["citations"])
            self.assertFalse(waiting["citations"][0]["cited"])
            self.assertTrue(any("不会重新搜索" in line for line in waiting["scope"]))
            search_count = len(searches)
            fetch_count = len(fetches)
            again = self._post(base, f"/api/jobs/{job['id']}/continue", {})
            done = self._wait(base, again["id"])
        self.assertEqual(len(searches), search_count)
        self.assertEqual(len(fetches), fetch_count)
        self.assertEqual(done["verdict"], "supported")
        self.assertEqual(done["verdict_label"], "支持")
        self.assertEqual(done["evidence_heading"], "引用")
        self.assertIn("rfc2606", done["citations"][0]["url"])
        self.assertFalse(done["can_continue"])

    def test_a_known_retry_time_disables_continue_until_it_passes(self) -> None:
        searches: list[str] = []

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searches.append(query)
            return _hits()

        def complete(system: str, user: str) -> str:
            error = RuntimeError("429 rate limit")
            error.retry_after = 120
            error.status_code = 429
            raise error

        with self._server(search, complete) as base:
            job = self._post(base, "/api/jobs", {"claim": SENTENCE})
            waiting = self._wait(base, job["id"])
            self.assertEqual(waiting["verdict_label"], "判断：等待重试")
            self.assertEqual(waiting["status_note"], "模型服务暂时限流，证据已保存")
            self.assertFalse(waiting["can_continue"])
            self.assertTrue(waiting["retry_known"])
            self.assertIsNotNone(waiting["retry_at"])
            self.assertNotIn("retry_after_s", waiting)
            counted = len(searches)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self._post(base, f"/api/jobs/{job['id']}/continue", {})
        self.assertEqual(caught.exception.code, 409)
        self.assertEqual(len(searches), counted)

    def test_a_new_claim_can_fetch_during_a_shared_cooldown(self) -> None:
        calls = {"n": 0}
        searches: list[str] = []
        fetches: list[str] = []
        first = "The handbook states that the pier opened in 2001."
        second = "The ledger states that the pier closed in 2002."

        def search(query: str, limit: int) -> tuple[list[dict[str, str]], None]:
            searches.append(query)
            return [{"url": "https://example.com/pier", "title": "Pier", "quote": "snippet"}], None

        def getter(url: str) -> dict:
            fetches.append(url)
            return {
                "http_status": 200,
                "final_url": url,
                "raw": f"<html><title>Pier</title><body><p>{first} {second}</p></body></html>",
                "truncated": False,
                "error": None,
            }

        def complete(system: str, user: str) -> str:
            calls["n"] += 1
            error = RuntimeError("429 rate limit")
            error.retry_after = 120
            error.status_code = 429
            raise error

        with self._server(search, complete, getter=getter) as base:
            job = self._post(base, "/api/jobs", {"claim": first})
            waiting = self._wait(base, job["id"])
            used = calls["n"]
            searched = len(searches)
            other = self._post(base, "/api/jobs", {"claim": second})
            held = self._wait(base, other["id"])
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self._post(base, f"/api/jobs/{other['id']}/continue", {})
        self.assertEqual(waiting["verdict_label"], "判断：等待重试")
        self.assertGreater(calls["n"], 0)
        self.assertEqual(calls["n"], used)
        self.assertGreater(len(searches), searched)
        self.assertEqual(held["reads"][0]["page_cache"], "hit")
        self.assertEqual(held["reads"][0]["detail"], "已读取")
        self.assertEqual(held["verdict_label"], "判断：等待重试")
        self.assertEqual(held["status_note"], "模型服务暂时限流，证据已保存")
        self.assertFalse(held["can_continue"])
        self.assertEqual(held["stages"]["search"]["label"], "完成")
        self.assertEqual(held["stages"]["fetch"]["label"], "完成")
        self.assertTrue(held["reads"])
        self.assertFalse(held["paid_fallback"])
        self.assertEqual(caught.exception.code, 409)

    def test_a_long_passage_stays_short_until_it_is_expanded(self) -> None:
        quote = "reserved for documentation " * 20
        view = check_server.public_view(
            {
                "judge_status": "rate_limited",
                "verdict": None,
                "passages": [
                    {"source_id": "src_rfc", "url": "https://www.rfc-editor.org/rfc/rfc2606.html", "quote": quote},
                    {"source_id": "src_wiki", "url": "https://en.wikipedia.org/wiki/RFC_2606", "quote": "A short overview."},
                ],
                "sources": [
                    {"source_id": "src_rfc", "title": "RFC 2606", "url": "https://www.rfc-editor.org/rfc/rfc2606.html"},
                    {"source_id": "src_wiki", "title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/RFC_2606"},
                ],
                "named_sources": [{"host": "rfc-editor.org", "role": "original"}],
                "limitations": ["The claim is not treated as false.", "The provider rate limit continued."],
            }
        )
        self.assertEqual(view["evidence_heading"], "已获取的证据")
        hidden = check_server.public_view(
            {
                "judge_status": "validation_failed",
                "verdict": None,
                "explanation": "This failed conclusion must stay off the page.",
                "judge_requests": [{"purpose": "initial", "raw_reply": "SECRET-RAW-REPLY", "validation_errors": ["quote does not match"]}],
            }
        )
        self.assertEqual(hidden["stages"]["judge"]["label"], "引用未通过")
        self.assertIsNone(hidden["verdict"])
        self.assertNotIn("SECRET-RAW-REPLY", json.dumps(hidden))
        self.assertNotIn("judge_requests", json.dumps(hidden))
        self.assertIsNone(view["explanation"])
        self.assertLess(len(view["citations"][0]["excerpt"]), len(quote.strip()))
        self.assertEqual(view["citations"][0]["role_label"], "原始文档")
        self.assertEqual(view["citations"][1]["role_label"], "辅助材料")
        self.assertEqual(view["limitations"], [])
        self.assertFalse(view["paid_fallback"])
        failed = check_server.public_view(
            {
                "judge_status": "rate_limited",
                "claim": "example.com states that this domain is for use in documentation examples without needing permission.",
                "named_sources": [
                    {"url": "https://example.com", "host": "example.com", "role": "original", "fetch_status": "fetch_failed", "error": "HTTP 404"}
                ],
                "fetches": [
                    {"url": "https://example.com", "final_url": "https://example.com/", "fetch_status": "fetch_failed", "error": "HTTP 404"},
                    {
                        "url": "https://en.wikipedia.org/wiki/Example.com",
                        "final_url": "https://en.wikipedia.org/wiki/Example.com",
                        "fetch_status": "ok",
                        "passage": {"source_id": "src_wiki", "url": "https://en.wikipedia.org/wiki/Example.com", "quote": "A short overview."},
                    },
                ],
                "passages": [{"source_id": "src_wiki", "url": "https://en.wikipedia.org/wiki/Example.com", "quote": "A short overview."}],
            }
        )
        self.assertEqual(failed["reads"][0]["role_label"], "原始文档")
        self.assertIn("HTTP 404", failed["reads"][0]["detail"])
        self.assertEqual(failed["reads"][1]["role_label"], "辅助材料")
        self.assertEqual(failed["citations"][0]["role_label"], "辅助材料")

    def test_the_page_has_no_label_or_confidence_field(self) -> None:
        page = (ROOT / "web" / "check.html").read_text(encoding="utf-8")
        self.assertIn("核查", page)
        self.assertIn("继续任务", page)
        self.assertIn("已获取的证据", page)
        self.assertIn("展开完整证据", page)
        self.assertIn("重试时间未知", page)
        self.assertIn("判断：等待重试", page)
        self.assertIn("模型服务暂时限流，证据已保存", page)
        self.assertIn("剩余", page)
        self.assertIn("本次不允许付费模型。", page)
        self.assertIn("本次不允许调用 OpenRouter。", page)
        self.assertNotIn("human_label", page)
        self.assertNotIn("confidence", page)
        self.assertNotIn("api_key", page)

    def _server(self, search, complete, getter=None):
        return _Running(search, complete, getter)

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

    def _get(self, base: str, path: str) -> dict:
        with urllib.request.urlopen(base + path) as response:
            return json.load(response)

    def _wait(self, base: str, job_id: str) -> dict:
        deadline = time.time() + 5
        last = {}
        while time.time() < deadline:
            last = self._get(base, f"/api/jobs/{job_id}")
            if last["status"] != "running":
                self.assertEqual(last["status"], "done", last.get("error"))
                return last
            time.sleep(0.02)
        self.fail(f"job stayed running: {last}")


class _Limited(RuntimeError):
    def __init__(self) -> None:
        super().__init__("429 rate limit")
        self.retry_after = 900
        self.status_code = 429


class _Running:
    def __init__(self, search, complete, getter) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.app = check_server.App(
            search=search,
            getter=getter or _page,
            complete=complete,
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


def _page(url: str) -> dict:
    return {"http_status": 200, "final_url": url, "raw": PAGE, "truncated": False, "error": None}


if __name__ == "__main__":
    unittest.main()
