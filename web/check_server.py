#!/usr/bin/env python3
"""Local page for one claim. The check itself is pipeline/search_evidence.py.

    .venv/bin/python web/check_server.py

OpenRouter and paid model fallback stay off. The API key is read only by the
pipeline on this machine. This page does not sign anyone in and does not check
a whole article.

The article checker remains ``python web/server.py``.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import acquire_url  # noqa: E402
import articles as article_mode  # noqa: E402
import search_evidence  # noqa: E402

PAGE = WEB / "check.html"
ARTICLE_PAGE = WEB / "article.html"
JOBS = ROOT / "data" / "web" / "jobs"
VERDICTS = acquire_url.VERDICTS
VERDICT_LABELS = {
    "supported": "支持",
    "refuted": "反驳",
    "insufficient_evidence": "证据不足",
}
SEARCH_LABELS = {
    "not_run": "未开始",
    "ok": "完成",
    "source_not_found": "未找到来源",
    "search_unavailable": "搜索不可用",
    "search_rate_limited": "搜索限流",
}
FETCH_LABELS = {
    "not_run": "未开始",
    "ok": "完成",
    "fetch_failed": "读取失败",
    "passage_not_found": "未选到段落",
    "truncated": "完成",
    "empty_body": "页面为空",
}
JUDGE_LABELS = {
    "not_run": "未开始",
    "completed": "完成",
    "rate_limited": "等待重试",
    "failed": "引用未通过",
    "validation_failed": "引用未通过",
    "model_error": "判断失败",
    "pending": "等待重试",
}
READ_LABELS = {
    "ok": "已读取",
    "truncated": "已读取",
    "fetch_failed": "读取失败",
    "empty_body": "页面为空",
    "not_run": "未读取",
}
SECRET = re.compile(r"(api[_-]?key|secret|token|sk-|gsk_|bearer\s)", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"^(the claim is not treated as false\.?|the provider rate limit|not requested because the provider rate limit|the model was not called because the provider rate limit|saved evidence is kept|the named source )",
    re.IGNORECASE,
)
MAX_CLAIM = 4_000
MAX_BODY = 100_000
EXCERPT = 180


def public_view(
    record: dict[str, Any],
    active: str | None = None,
    *,
    allow_paid: bool = False,
    allow_openrouter: bool = False,
) -> dict[str, Any]:
    """Fields the page may show. Human labels and provider secrets stay out."""
    judge_status = record.get("judge_status") or "not_run"
    verdict = record.get("verdict") if record.get("verdict") in VERDICTS and judge_status == "completed" else None
    pending = judge_status == "rate_limited"
    retry_at = _retry_at(record) if pending else None
    cooling = bool(retry_at and retry_at > datetime.now(timezone.utc))
    return {
        "claim": record.get("claim") or "",
        "stages": {
            "search": _stage(record.get("search_status"), SEARCH_LABELS, active == "search"),
            "fetch": _stage(record.get("fetch_status"), FETCH_LABELS, active == "fetch"),
            "judge": _stage(judge_status, JUDGE_LABELS, active == "judge"),
        },
        "verdict": verdict,
        "verdict_label": "判断：等待重试" if pending else VERDICT_LABELS.get(verdict or ""),
        "status_note": "模型服务暂时限流，证据已保存" if pending else None,
        "explanation": record.get("explanation") if verdict else None,
        "evidence_heading": "引用" if verdict else "已获取的证据",
        "reads": _reads(record),
        "citations": _citations(record, published=bool(verdict)),
        "scope": _scope(record, pending),
        "limitations": _limits(record),
        "can_continue": pending and not cooling,
        "retry_at": retry_at.isoformat() if cooling else None,
        "retry_known": retry_at is not None,
        "openrouter_fallback": allow_openrouter,
        "paid_fallback": allow_paid,
        "paid_label": "本次允许付费模型。" if allow_paid else "本次不允许付费模型。",
        "openrouter_label": "本次允许调用 OpenRouter。" if allow_openrouter else "本次不允许调用 OpenRouter。",
    }


def _retry_at(record: dict[str, Any]) -> datetime | None:
    """The stored server deadline. A missing deadline is not turned into a guess."""
    raw = record.get("retry_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _stage(status: Any, labels: dict[str, str], active: bool) -> dict[str, str]:
    current = str(status or "not_run")
    if active and current == "not_run":
        return {"status": "running", "label": "进行中"}
    return {"status": current, "label": labels.get(current, current)}


def _scope(record: dict[str, Any], pending: bool) -> list[str]:
    lines = ["只依据已读取的页面段落，不是完整文档。"]
    if pending:
        lines.append("已保存的证据会在继续时复用，不会重新搜索。")
    return lines


def _limits(record: dict[str, Any]) -> list[str]:
    kept: list[str] = []
    for item in record.get("limitations") or []:
        text = str(item).strip()
        if not text or _BOILERPLATE.match(text) or text in kept:
            continue
        if "not a complete document" in text.lower() or text == record.get("document_scope"):
            continue
        kept.append(text)
    return kept


def _citations(record: dict[str, Any], *, published: bool) -> list[dict[str, Any]]:
    titles: dict[str, str] = {}
    for source in record.get("sources") or []:
        titles[str(source.get("source_id"))] = str(source.get("title") or "")
    for fetched in record.get("fetches") or []:
        passage = fetched.get("passage") or {}
        if passage.get("source_id"):
            titles.setdefault(str(passage["source_id"]), str(fetched.get("title") or ""))
    rows = record.get("evidence_refs") if published else record.get("passages")
    citations = []
    for item in rows or []:
        url = str(item.get("url") or "")
        if urlparse(url).scheme not in {"http", "https"}:
            url = ""
        quote = str(item.get("quote") or "")
        role = _evidence_role(url, record)
        citations.append(
            {
                "quote": quote,
                "excerpt": _excerpt(quote),
                "title": titles.get(str(item.get("source_id"))) or "",
                "url": url,
                "cited": published,
                "role": role,
                "role_label": "原始文档" if role == "original" else "辅助材料",
            }
        )
    return citations


def _excerpt(quote: str) -> str:
    text = re.sub(r"\s+", " ", quote).strip()
    if len(text) <= EXCERPT:
        return text
    cut = text[:EXCERPT].rsplit(" ", 1)[0]
    return f"{cut}…"


def _reads(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Named originals stay visible even when a third-party passage is the only quote."""
    fetches = list(record.get("fetches") or [])
    rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    for source in record.get("named_sources") or []:
        host = str(source.get("host") or "")
        match = next((item for item in fetches if search_evidence._host(item.get("final_url") or item.get("url") or "") == host), None)
        rows.append(_read_row(str(source.get("url") or ""), match, "original", source.get("fetch_status"), source.get("error")))
        if host:
            covered.add(host)
    for fetched in fetches:
        host = search_evidence._host(fetched.get("final_url") or fetched.get("url") or "")
        if host in covered:
            continue
        rows.append(_read_row(str(fetched.get("url") or ""), fetched, _evidence_role(str(fetched.get("url") or ""), record), fetched.get("fetch_status"), fetched.get("error")))
    return rows


def _read_row(url: str, fetched: dict[str, Any] | None, role: str, status: Any, error: Any) -> dict[str, Any]:
    current = str((fetched or {}).get("fetch_status") or status or "not_run")
    reason = (fetched or {}).get("error") or error
    label = READ_LABELS.get(current, current)
    detail = f"{label}：{reason}" if reason and current not in {"ok", "truncated"} else label
    return {
        "url": str((fetched or {}).get("final_url") or url),
        "fetch_status": current,
        "detail": detail,
        "page_cache": (fetched or {}).get("page_cache"),
        "role": role,
        "role_label": "原始文档" if role == "original" else "辅助材料",
    }


def _evidence_role(url: str, record: dict[str, Any]) -> str:
    host = search_evidence._host(url)
    for source in record.get("named_sources") or []:
        if source.get("host") == host and source.get("role", "original") == "original":
            return "original"
    if host == "rfc-editor.org" or host.endswith(".rfc-editor.org") or host == "ietf.org" or host.endswith(".ietf.org"):
        return "original"
    return "auxiliary"


def _safe_error(exc: BaseException) -> str:
    text = str(exc)
    if SECRET.search(text):
        text = ""
    detail = f"{text[:300]}" if text else ""
    return f"核查没有完成（{type(exc).__name__}）。{detail}".strip()


class App:
    """One in-memory claim at a time. Hooks exist so tests can drive the same page."""

    def __init__(
        self,
        *,
        search: Callable[..., Any] | None = None,
        getter: Callable[[str], dict[str, Any]] | None = None,
        complete: Callable[[str, str], str] | None = None,
        extract: Callable[..., Any] | None = None,
        jobs_dir: Path | None = None,
        cache_dir: Path | None = None,
        sleeper: Callable[[float], None] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self.search = search
        self.getter = getter
        self.complete = complete
        self.jobs_dir = jobs_dir or JOBS
        self.cache_dir = cache_dir
        self.sleeper = sleeper
        self.rng = rng
        self.allow_paid = False
        self.allow_openrouter = False
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.articles = article_mode.ArticleBook(
            view=public_view,
            extract=extract,
            search=search,
            getter=getter,
            complete=complete,
            cache_dir=cache_dir,
            articles_dir=(jobs_dir.parent / "articles") if jobs_dir is not None else (JOBS.parent / "articles"),
            sleeper=sleeper,
            rng=rng,
        )

    def start(self, claim: str) -> dict[str, Any]:
        text = _claim(claim)
        with self.lock:
            existing = next((job for job in self.jobs.values() if job["claim"] == text and job.get("pipeline")), None)
        if existing and existing["status"] != "running":
            record = existing["pipeline"]["claims"][0]
            self._apply_shared_cooldown(record)
            stale = record.get("pipeline_version") != search_evidence.PIPELINE_VERSION
            if stale:
                with self.lock:
                    existing["status"] = "running"
                    existing["error"] = None
                self._spawn(existing, previous=existing.get("pipeline"))
                return self.snapshot(existing["id"])
            if record.get("search_status") in search_evidence.SEARCH_DONE or record.get("fetches"):
                if _cooling(record):
                    return self.snapshot(existing["id"])
                if self._view(record, existing.get("pipeline")).get("can_continue"):
                    return self.continue_job(existing["id"])
                return self.snapshot(existing["id"])
        job_id = _job_id()
        job = {
            "id": job_id,
            "claim": text,
            "status": "running",
            "error": None,
            "pipeline": None,
            "view": self._view({"claim": text, "search_status": "not_run", "fetch_status": "not_run", "judge_status": "not_run"}),
        }
        with self.lock:
            self.jobs[job_id] = job
        self._spawn(job, previous=None)
        return self.snapshot(job_id)

    def continue_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "running":
                raise RuntimeError("running")
            record = ((job.get("pipeline") or {}).get("claims") or [{}])[0]
            self._apply_shared_cooldown(record)
            if _cooling(record):
                raise RuntimeError("cooldown")
            fresh = self._view(record, job.get("pipeline"))
            if not fresh.get("can_continue"):
                raise RuntimeError("finished")
            job["status"] = "running"
            job["error"] = None
            previous = job.get("pipeline")
        self._spawn(job, previous=previous)
        return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            record = ((job.get("pipeline") or {}).get("claims") or [None])[0]
            if record and job["status"] != "running":
                self._apply_shared_cooldown(record)
                job["view"] = self._view(record, job.get("pipeline"))
            view = dict(job["view"])
            view["id"] = job["id"]
            view["status"] = job["status"]
            view["error"] = job["error"]
            return view

    def _spawn(self, job: dict[str, Any], previous: dict[str, Any] | None) -> None:
        thread = threading.Thread(target=self._run, args=(job, previous), daemon=True)
        thread.start()

    def _run(self, job: dict[str, Any], previous: dict[str, Any] | None) -> None:
        def progress(stage: str, record: dict[str, Any]) -> None:
            with self.lock:
                if job["status"] == "running":
                    job["view"] = self._view(record, active=stage)

        try:
            result = search_evidence.investigate(
                [{"id": job["id"], "claim": job["claim"]}],
                search=self.search,
                getter=self.getter,
                judge="llm",
                allow_openrouter_fallback=self.allow_openrouter,
                allow_paid=self.allow_paid,
                cache_dir=self.cache_dir,
                complete=self.complete,
                sleeper=self.sleeper or acquire_url.time.sleep,
                rng=self.rng or acquire_url.random.random,
                previous=previous,
                checkpoint=self.jobs_dir / f"{job['id']}.json",
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001 — the page shows a short failure, not a secret
            traceback.print_exc()
            with self.lock:
                job["status"] = "error"
                job["error"] = _safe_error(exc)
            return
        with self.lock:
            job["pipeline"] = result
            job["view"] = self._view(result["claims"][0], result)
            job["status"] = "done"

    def _view(self, record: dict[str, Any], pipeline: dict[str, Any] | None = None, active: str | None = None) -> dict[str, Any]:
        """Show the switches used for this run. A stored verdict is not rewritten."""
        if pipeline and "paid_fallback" in pipeline:
            allow_paid = bool(pipeline.get("paid_fallback"))
            allow_openrouter = bool(pipeline.get("openrouter_fallback"))
        else:
            allow_paid = self.allow_paid
            allow_openrouter = self.allow_openrouter
        return public_view(record, active, allow_paid=allow_paid, allow_openrouter=allow_openrouter)

    def _apply_shared_cooldown(self, record: dict[str, Any]) -> None:
        """A later deadline for this provider and model replaces a shorter one on the task."""
        if record.get("judge_status") != "rate_limited":
            return
        provider = str(record.get("provider") or search_evidence.DEFAULT_PROVIDER)
        model = str(record.get("model") or search_evidence.DEFAULT_MODEL)
        shared = search_evidence.shared_retry_at(self.cache_dir or acquire_url.CACHE, provider, model)
        if shared is None or shared <= datetime.now(timezone.utc):
            return
        current = _retry_at(record)
        if current is None or shared > current:
            record["retry_at"] = shared.isoformat()


def _cooling(record: dict[str, Any]) -> bool:
    moment = _retry_at(record)
    return bool(moment and moment > datetime.now(timezone.utc))


def _claim(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("输入一条 claim。")
    if len(text) > MAX_CLAIM:
        raise ValueError("这条 claim 太长。")
    return text


def _job_id() -> str:
    import secrets

    return secrets.token_hex(8)


def handler_for(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/check"}:
                body = PAGE.read_bytes()
                return self._send(200, body, "text/html; charset=utf-8")
            if path == "/article":
                body = ARTICLE_PAGE.read_bytes()
                return self._send(200, body, "text/html; charset=utf-8")
            if path.startswith("/api/articles/"):
                article_id = path.removeprefix("/api/articles/").strip("/")
                if not article_id or "/" in article_id:
                    return self._send_json(404, {"error": "没有这篇文章。"})
                try:
                    return self._send_json(200, app.articles.snapshot(article_id))
                except KeyError:
                    return self._send_json(404, {"error": "没有这篇文章。"})
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/").strip("/")
                try:
                    return self._send_json(200, app.snapshot(job_id))
                except KeyError:
                    return self._send_json(404, {"error": "没有这项核查。"})
            return self._send_json(404, {"error": "没有这个页面。"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
            try:
                if path == "/api/jobs":
                    return self._send_json(202, app.start(payload.get("claim")))
                if path == "/api/articles":
                    return self._send_json(202, app.articles.start_extract(payload.get("text")))
                if path.startswith("/api/articles/") and path.endswith("/check"):
                    article_id = path.removeprefix("/api/articles/").removesuffix("/check").strip("/")
                    return self._send_json(202, app.articles.start_check(article_id, payload.get("ids"), payload.get("claims")))
                if path.startswith("/api/articles/") and path.endswith("/continue"):
                    article_id = path.removeprefix("/api/articles/").removesuffix("/continue").strip("/")
                    return self._send_json(202, app.articles.continue_article(article_id))
                if path.startswith("/api/jobs/") and path.endswith("/continue"):
                    job_id = path.removeprefix("/api/jobs/").removesuffix("/continue").strip("/")
                    return self._send_json(202, app.continue_job(job_id))
            except KeyError:
                return self._send_json(404, {"error": "没有这项核查。"})
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
            except RuntimeError as exc:
                if str(exc) == "running":
                    return self._send_json(409, {"error": "这项核查还在进行。"})
                if str(exc) == "cooldown":
                    return self._send_json(409, {"error": "冷却尚未结束。"})
                return self._send_json(409, {"error": "这项核查没有待完成的判断。"})
            return self._send_json(404, {"error": "没有这个页面。"})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length < 0 or length > MAX_BODY:
                raise ValueError("请求太大。")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("请求不是 JSON。") from exc
            if not isinstance(data, dict):
                raise ValueError("请求不是 JSON。")
            return data

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(app: App, port: int = 8000) -> ThreadingHTTPServer:
    app.jobs_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = App()
    server = serve(app, args.port)
    print(f"FactThread check: http://127.0.0.1:{server.server_address[1]}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
