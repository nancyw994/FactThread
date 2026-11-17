"""Paste one article, extract candidate claims, then check up to three of them.

The single-claim page is unchanged. OpenRouter and paid models stay off.
A changed article drops the previous extraction. A changed claim does not
reuse the previous judgment. Completed claims are not run again.
"""

from __future__ import annotations

import hashlib
import json
import threading
import traceback
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import extract_claims
import search_evidence

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "config" / "prompts" / "extract_claims.md"
MAX_SELECT = 3
MAX_ARTICLE = 12_000
VERDICTS = search_evidence.VERDICTS
FINISHED = frozenset({"completed", "failed", "validation_failed"})
TYPE_LABELS = {
    "checkable_fact": "可核查事实",
    "opinion": "观点",
    "prediction": "预测",
    "intention": "意图",
    "needs_context": "缺少上下文",
}


def groq_extract(text: str, repair_errors: list[str] | None) -> dict[str, Any]:
    """Extract with the free Groq model. OpenRouter and paid models stay off."""
    import sys

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import llm_client

    llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
    user = json.dumps({"article": text, "repair_errors": repair_errors or []}, ensure_ascii=False)
    raw = llm_client.ask_llm({"instructions": PROMPT.read_text(encoding="utf-8")}, user)
    return _parse_extract(raw)


def _parse_extract(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {"claims": []}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"claims": []}
    return parsed if isinstance(parsed, dict) else {"claims": []}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ArticleBook:
    """In-memory articles. Tests pass extract, search, and judge hooks."""

    def __init__(
        self,
        *,
        view: Callable[..., dict[str, Any]],
        extract: Callable[[str, list[str] | None], Any] | None = None,
        search: Callable[..., Any] | None = None,
        getter: Callable[[str], dict[str, Any]] | None = None,
        complete: Callable[[str, str], str] | None = None,
        cache_dir: Path | None = None,
        articles_dir: Path | None = None,
        sleeper: Callable[[float], None] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self.view = view
        self.extract_complete = extract or groq_extract
        self.search = search
        self.getter = getter
        self.complete = complete
        self.cache_dir = cache_dir
        self.articles_dir = articles_dir
        self.sleeper = sleeper
        self.rng = rng
        self.allow_paid = False
        self.allow_openrouter = False
        self.articles: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._restore()

    def start_extract(self, text: str, article_source_url: Any = None) -> dict[str, Any]:
        article_text = _article(text)
        source_url = _source_url(article_source_url)
        digest = _sha(article_text)
        with self.lock:
            existing = next(
                (
                    item
                    for item in self.articles.values()
                    if item["text_sha"] == digest
                    and _canonical_url(item.get("article_source_url")) == _canonical_url(source_url)
                    and item["status"] in {"extracted", "done"}
                ),
                None,
            )
        if existing:
            return self.snapshot(existing["id"])
        article_id = _id()
        article = {
            "id": article_id,
            "text": article_text,
            "text_sha": digest,
            "article_source_url": source_url,
            "status": "running",
            "phase": "extract",
            "error": None,
            "candidates": [],
            "selection": [],
            "records": {},
            "attempt_history": {},
        }
        with self.lock:
            self.articles[article_id] = article
        threading.Thread(target=self._extract, args=(article,), daemon=True).start()
        return self.snapshot(article_id)

    def start_check(
        self,
        article_id: str,
        ids: Any,
        edits: Any,
        *,
        retry: bool = False,
        article_source_url: Any = None,
    ) -> dict[str, Any]:
        article = self._get(article_id)
        if article["status"] == "running":
            raise RuntimeError("running")
        selection = self._selection(article, ids, edits)
        with self.lock:
            if article_source_url is not None:
                article["article_source_url"] = _source_url(article_source_url)
            current = {item["candidate_id"]: item for item in article.get("selection") or []}
            for item in selection:
                previous = article["records"].get(item["pipeline_id"])
                if retry and previous is not None:
                    history = article.setdefault("attempt_history", {}).setdefault(item["pipeline_id"], [])
                    history.append(previous)
                    article["records"].pop(item["pipeline_id"], None)
                current[item["candidate_id"]] = item
            article["selection"] = list(current.values())
            article["status"] = "running"
            article["phase"] = "check"
            article["error"] = None
            self._save(article)
        threading.Thread(target=self._check, args=(article,), daemon=True).start()
        return self.snapshot(article_id)

    def continue_article(self, article_id: str) -> dict[str, Any]:
        article = self._get(article_id)
        if article["status"] == "running":
            raise RuntimeError("running")
        view = self.snapshot(article_id)
        if not view["can_continue"]:
            if view.get("cooling"):
                raise RuntimeError("cooldown")
            raise RuntimeError("finished")
        with self.lock:
            article["status"] = "running"
            article["phase"] = "check"
            article["error"] = None
            self._save(article)
        threading.Thread(target=self._check, args=(article,), daemon=True).start()
        return self.snapshot(article_id)

    def snapshot(self, article_id: str) -> dict[str, Any]:
        with self.lock:
            article = self.articles.get(article_id)
            if article is None:
                raise KeyError(article_id)
            selection = list(article.get("selection") or [])
            records = dict(article.get("records") or {})
            status = article["status"]
            shown_article = {
                "id": article["id"],
                "status": status,
                "phase": article.get("phase"),
                "error": article.get("error"),
                "text_sha": article["text_sha"],
                "article_source_url": article.get("article_source_url"),
                "candidates": list(article.get("candidates") or []),
            }
        claims = []
        cooling = False
        retry_at = None
        unfinished = False
        for item in selection:
            record = records.get(item["pipeline_id"])
            if record is None:
                shown = self.view(
                    {"claim": item["claim"], "search_status": "not_run", "fetch_status": "not_run", "judge_status": "not_run"},
                    allow_paid=False,
                    allow_openrouter=False,
                )
            else:
                active = record.get("active_stage") if status == "running" else None
                view_record = dict(record)
                view_record["article_source_url"] = shown_article.get("article_source_url")
                shown = self.view(view_record, active=active, allow_paid=False, allow_openrouter=False)
                if shown.get("retry_at") and (retry_at is None or shown["retry_at"] < retry_at):
                    retry_at = shown["retry_at"]
                if _cooling(record):
                    cooling = True
            shown["candidate_id"] = item["candidate_id"]
            claims.append(shown)
            if _unfinished(record):
                unfinished = True
        return {
            "id": shown_article["id"],
            "status": shown_article["status"],
            "phase": shown_article.get("phase"),
            "error": shown_article.get("error"),
            "text_sha": shown_article["text_sha"],
            "article_source_url": shown_article.get("article_source_url"),
            "candidates": [_candidate_view(item) for item in shown_article["candidates"]],
            "selected": [item["candidate_id"] for item in selection],
            "claims": claims,
            "can_continue": unfinished and not cooling and status != "running",
            "cooling": cooling,
            "retry_at": retry_at,
            "max_select": MAX_SELECT,
            "paid_fallback": False,
            "openrouter_fallback": False,
            "paid_label": "本次不允许付费模型。",
            "openrouter_label": "本次不允许调用 OpenRouter。",
        }

    def _extract(self, article: dict[str, Any]) -> None:
        try:
            extracted = extract_claims.extract_claims(article["text"], self.extract_complete)
            candidates = [_stored_candidate(item) for item in extracted.get("claims") or []]
        except Exception as exc:  # noqa: BLE001 — the page gets a short failure, not a key
            traceback.print_exc()
            with self.lock:
                article["status"] = "error"
                article["error"] = _safe_error(exc)
            return
        with self.lock:
            article["candidates"] = candidates
            article["selection"] = []
            article["records"] = {}
            article["attempt_history"] = {}
            article["status"] = "extracted"
            article["phase"] = None
        self._save(article)

    def _check(self, article: dict[str, Any]) -> None:
        selection = list(article.get("selection") or [])
        pending = [item for item in selection if _unfinished(article["records"].get(item["pipeline_id"]))]
        if not pending:
            with self.lock:
                article["status"] = "done"
                article["phase"] = None
            self._save(article)
            return
        previous = {"claims": [article["records"][item["pipeline_id"]] for item in pending if item["pipeline_id"] in article["records"]]}

        def progress(stage: str, record: dict[str, Any]) -> None:
            with self.lock:
                if article["status"] == "running":
                    current = dict(record)
                    current["active_stage"] = stage
                    article["records"][str(record.get("id"))] = current
                    self._save(article)

        options: dict[str, Any] = {}
        if self.sleeper is not None:
            options["sleeper"] = self.sleeper
        if self.rng is not None:
            options["rng"] = self.rng
        try:
            result = search_evidence.investigate(
                [{"id": item["pipeline_id"], "claim": item["claim"]} for item in pending],
                search=self.search,
                getter=self.getter,
                judge="llm",
                allow_openrouter_fallback=False,
                allow_paid=False,
                cache_dir=self.cache_dir,
                complete=self.complete,
                input_source_url=article.get("article_source_url"),
                previous=previous if previous["claims"] else None,
                progress=progress,
                **options,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            with self.lock:
                article["status"] = "error"
                article["error"] = _safe_error(exc)
            self._save(article)
            return
        with self.lock:
            for record in result["claims"]:
                article["records"][str(record.get("id"))] = record
            article["status"] = "done"
            article["phase"] = None
        self._save(article)

    def _selection(self, article: dict[str, Any], ids: Any, edits: Any) -> list[dict[str, str]]:
        if article["status"] == "running":
            raise RuntimeError("running")
        if not article.get("candidates"):
            raise ValueError("先提取文章中的条目。")
        wanted = ids if isinstance(ids, list) else []
        changes = edits if isinstance(edits, dict) else {}
        by_id = {item["id"]: item for item in article["candidates"]}
        chosen: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw_id in wanted:
            candidate_id = str(raw_id)
            if candidate_id in seen or candidate_id not in by_id:
                continue
            seen.add(candidate_id)
            edited = changes.get(candidate_id, by_id[candidate_id]["normalized_claim"])
            text = str(edited or "").strip()
            if not text:
                raise ValueError("要核查的条目不能是空的。")
            legacy_id = _sha(text)
            scoped_id = _sha(f"{article['text_sha']}\0{text}")
            pipeline_id = legacy_id if legacy_id in article.get("records", {}) else scoped_id
            chosen.append({"candidate_id": candidate_id, "claim": text, "pipeline_id": pipeline_id})
        if not chosen:
            raise ValueError("先选择要核查的条目。")
        if len(chosen) > MAX_SELECT:
            raise ValueError("每批最多选择 3 条。")
        return chosen

    def _get(self, article_id: str) -> dict[str, Any]:
        article = self.articles.get(article_id)
        if article is None:
            raise KeyError(article_id)
        return article

    def _save(self, article: dict[str, Any]) -> None:
        if self.articles_dir is None:
            return
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        path = self.articles_dir / f"{article['id']}.json"
        stored = {
            "id": article["id"],
            "text": article["text"],
            "text_sha": article["text_sha"],
            "article_source_url": article.get("article_source_url"),
            "candidates": article.get("candidates") or [],
            "selection": article.get("selection") or [],
            "records": article.get("records") or {},
            "attempt_history": article.get("attempt_history") or {},
            "status": article.get("status"),
            "phase": article.get("phase"),
            "error": article.get("error"),
        }
        path.write_text(json.dumps(stored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _restore(self) -> None:
        """Reload locally saved articles so a paused batch can continue after restart."""
        if self.articles_dir is None or not self.articles_dir.is_dir():
            return
        for path in self.articles_dir.glob("*.json"):
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                text = _article(saved.get("text"))
                digest = _sha(text)
                if digest != saved.get("text_sha") or path.stem != saved.get("id"):
                    continue
                selection = saved.get("selection") if isinstance(saved.get("selection"), list) else []
                records = saved.get("records") if isinstance(saved.get("records"), dict) else {}
                status = saved.get("status") or ("done" if selection else "extracted")
                # A process that stopped mid-request leaves its last stage checkpoint
                # available; mark it idle so the user can continue that item.
                if status == "running":
                    status = "extracted" if saved.get("phase") == "extract" else "done"
                self.articles[path.stem] = {
                    "id": path.stem,
                    "text": text,
                    "text_sha": digest,
                    "article_source_url": _source_url(saved.get("article_source_url")),
                    "status": status,
                    "phase": None,
                    "error": saved.get("error"),
                    "candidates": saved.get("candidates") if isinstance(saved.get("candidates"), list) else [],
                    "selection": selection,
                    "records": records,
                    "attempt_history": saved.get("attempt_history") if isinstance(saved.get("attempt_history"), dict) else {},
                }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue


def _stored_candidate(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": claim.get("id"),
        "source_quote": claim.get("source_quote") or "",
        "normalized_claim": claim.get("normalized_claim") or "",
        "fact_type": claim.get("fact_type") or claim.get("category") or "",
        "missing_context": list(claim.get("missing_context") or []),
        "status_reason": claim.get("status_reason"),
    }


def _candidate_view(claim: dict[str, Any]) -> dict[str, Any]:
    fact_type = str(claim.get("fact_type") or "")
    missing_context = list(claim.get("missing_context") or [])
    return {
        "id": claim.get("id"),
        "source_quote": claim.get("source_quote") or "",
        "normalized_claim": claim.get("normalized_claim") or "",
        "fact_type": fact_type,
        "fact_type_label": TYPE_LABELS.get(fact_type, fact_type),
        "missing_context": missing_context,
        "context_note": (
            "尚需补充声明中的主体、日期或比较基准。"
            if fact_type == "needs_context" and not missing_context
            else None
        ),
        "default_selected": (
            fact_type not in {"opinion", "prediction", "intention", "needs_context"}
            and not missing_context
        ),
    }


def _unfinished(record: dict[str, Any] | None) -> bool:
    if record is None:
        return True
    status = record.get("judge_status") or "not_run"
    if status in FINISHED:
        return False
    if status == "completed" and record.get("verdict") in VERDICTS:
        return False
    return True


def _cooling(record: dict[str, Any]) -> bool:
    from datetime import datetime, timezone

    raw = record.get("retry_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment > datetime.now(timezone.utc)


def _article(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("先粘贴一段文章。")
    if len(text) > MAX_ARTICLE:
        raise ValueError("这篇文章太长。")
    return text


def _source_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) > 2_000:
        raise ValueError("文章来源 URL 太长。")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("文章来源必须是有效的 http 或 https URL。")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, ""))


def _canonical_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        return None
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), f"{host}{port}", path, "", ""))


def _id() -> str:
    import secrets

    return secrets.token_hex(8)


def _safe_error(exc: BaseException) -> str:
    text = str(exc)
    if "api_key" in text.lower() or "sk-" in text or "gsk_" in text:
        text = ""
    detail = text[:300] if text else ""
    return f"核查没有完成（{type(exc).__name__}）。{detail}".strip()
