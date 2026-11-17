#!/usr/bin/env python3
"""Fetch a user-supplied URL and attach passages for the existing judge.

The caller provides the claim and the URL. This does not search. A successful
fetch is stored with its original URL, final URL, fetch time, body, and content
hash, and that success does not mark the page as a complete document.

A read failure, an empty body, or a truncated body is recorded. Those statuses
are not findings that the claim is false. A fetched page is not a complete
document. An omission whose scope is not confirmed is insufficient evidence.

A repeated claim and URL reuses the saved page and any completed verdict.
--refresh refetches pages and keeps those verdicts. --rejudge discards verdicts
and keeps the saved pages. A rate limit is not a verdict and is not cached.
Paid model fallback stays off unless --allow-paid is set.

    .venv/bin/python pipeline/acquire_url.py --cases data/dev/url/cases.json --no-openrouter-fallback
    .venv/bin/python pipeline/acquire_url.py --cases data/dev/url/cases_pages.json --no-openrouter-fallback --out data/dev/url/last_url_pages.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import acquire_evidence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dev" / "url" / "last_url.json"
CACHE = ROOT / "data" / "dev" / "url" / "cache"
CONTEXT_LIMIT = 12000
TIMEOUT = 30
VERDICTS = frozenset({"supported", "refuted", "insufficient_evidence"})
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
RETRY_AFTER_CAP = 600.0
MAX_CONSECUTIVE_RATE_LIMITS = 2
USER_AGENT = "FactThread/0.1 (user-supplied URL; evidence fetch)"
NOT_FALSE = "The claim is not judged and is not treated as false."


def fetch_page(url: str, *, getter: Callable[[str], dict[str, Any]] | None = None) -> dict[str, Any]:
    """GET one URL. The returned body is the text that later quotes must match."""
    original = (url or "").strip()
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not original:
        return _page(original, original, fetched_at, None, "", True, "fetch_failed", "The URL is empty.")
    try:
        response = (getter or _http_get)(original)
    except Exception as exc:  # noqa: BLE001 — a failed read is a recorded status, not a verdict
        return _page(original, original, fetched_at, None, "", False, "fetch_failed", f"{type(exc).__name__}: {exc}")
    status = response.get("http_status")
    final_url = str(response.get("final_url") or original)
    error = response.get("error")
    if error or (isinstance(status, int) and status >= 400):
        return _page(original, final_url, fetched_at, status, "", False, "fetch_failed", str(error or f"HTTP {status}"))
    raw = str(response.get("raw") or "")
    title = page_title(raw)
    body = html_to_text(raw)
    truncated = bool(response.get("truncated"))
    if not body.strip():
        return _page(original, final_url, fetched_at, status, "", truncated, "empty_body", "The response had no text.", title=title)
    fetch_status = "truncated" if truncated else "ok"
    return _page(original, final_url, fetched_at, status, body, truncated, fetch_status, None, title=title)


def evidence_for_claim(
    claim: str,
    page: dict[str, Any],
    *,
    claim_id: str,
    omission: bool = False,
    attest_complete: bool = False,
    context_limit: int = CONTEXT_LIMIT,
) -> dict[str, Any]:
    """Turn one fetched page into an evidence row. Does not assign a verdict."""
    source = _source_view(page, complete=False)
    row: dict[str, Any] = {
        "claim_id": claim_id,
        "normalized_claim": claim,
        "fact_type": "checkable_fact",
        "verification_scope": "document_content" if omission else "external_fact",
        "document_ref": page.get("final_url"),
        "status": None,
        "reason": None,
        "source": source,
        "passages": [],
        "document": None,
        "proves_absence": False if omission else None,
    }
    status = page.get("fetch_status")
    if status == "fetch_failed":
        return _undecided(row, "fetch_failed", f"The page could not be read ({page.get('error') or 'request failed'}).")
    if status == "empty_body" or not str(page.get("body") or "").strip():
        return _undecided(row, "empty_body", "The page body is empty.")
    body = str(page["body"])
    if omission:
        return _omission_row(row, page, body, attest_complete=attest_complete, context_limit=context_limit)
    if page.get("truncated"):
        return _undecided(row, "truncated", "The page content was truncated before a passage could be checked.")
    passage = _best_paragraph(claim, page["source_id"], body)
    if passage is None:
        return _undecided(row, "passage_not_found", "The page was read, but no paragraph contained this claim's anchors.")
    acquire_evidence._require_match(body, passage)
    row["status"] = "ready"
    row["passages"] = [passage]
    row["reason"] = "The cited paragraph is an exact slice of the fetched text. The page is not marked as a complete document."
    return row


def acquire_cases(
    cases: list[dict[str, Any]],
    *,
    getter: Callable[[str], dict[str, Any]] | None = None,
    context_limit: int = CONTEXT_LIMIT,
) -> dict[str, Any]:
    """Fetch each distinct URL once, then build one evidence row per claim."""
    cache: dict[str, dict[str, Any]] = {}
    pages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        url = str(case.get("url") or "")
        if url not in cache:
            page = fetch_page(url, getter=getter)
            page["source_id"] = _source_id(page)
            cache[url] = page
            pages.append(page)
        page = cache[url]
        rows.append(
            evidence_for_claim(
                str(case.get("claim") or ""),
                page,
                claim_id=str(case.get("id") or f"url{index}"),
                omission=bool(case.get("omission")),
                attest_complete=bool(case.get("attest_complete")),
                context_limit=context_limit,
            )
        )
    return {"pages": pages, "claims": rows}


def _rate_limit_wait(exc: BaseException) -> tuple[bool, float | None]:
    """Whether this is a rate limit, and Retry-After seconds when the server sent one."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    limited = (
        status == 429
        or type(exc).__name__ in {"RateLimited", "RateLimitError"}
        or "429" in text
        or "rate limit" in text
    )
    if not limited:
        return False, None
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, int | float):
        return True, float(direct)
    minutes = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", str(exc))
    if minutes:
        return True, int(minutes.group(1)) * 60 + float(minutes.group(2))
    return True, None


def _backoff_seconds(step: int, rng: Callable[[], float]) -> float:
    """Bounded exponential delay with jitter. step starts at 0."""
    raw = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** step))
    return raw * (0.5 + 0.5 * rng())


def _wait_for_limit(retry_after: float | None, step: int, rng: Callable[[], float]) -> float | None:
    """Honor Retry-After. Without it, use capped jittered backoff. None means do not wait."""
    if retry_after is not None:
        if retry_after > RETRY_AFTER_CAP:
            return None
        return max(0.0, float(retry_after))
    return _backoff_seconds(step, rng)


def _unfinished(row: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "claim_id": row.get("claim_id"),
        "verdict": None,
        "judge_status": status,
        "explanation": None,
        "evidence_refs": [],
        "limitations": [reason],
        "evidence_status": status,
        "validation_status": status,
        "publish": False,
        "cache": status,
    }


def judge_acquired(
    acquired: dict[str, Any],
    *,
    judge: str,
    model: str | None = None,
    allow_openrouter_fallback: bool = True,
    allow_paid: bool = False,
    complete: Callable[[str, str], str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Judge ready URL rows with the existing judge. Human labels are not sent.

    A rate limit is retried once after Retry-After or bounded backoff. A second
    consecutive limit stops the batch and leaves the later claims pending.
    """
    import judge_evidence

    if judge == "llm":
        client = judge_evidence._llm_client()
        client.configure_providers(allow_openrouter_fallback=allow_openrouter_fallback, allow_paid=allow_paid)
    bodies = {page["source_id"]: page.get("body") or "" for page in acquired["pages"]}
    meter = judge_evidence.CallMeter() if judge == "llm" else None
    bound, chosen, meter = judge_evidence._bind_judge(judge, complete, model, meter)
    attempts = {"attempts": 0, "failures": 0, "successes": 0}

    def counted(system: str, user: str) -> str:
        attempts["attempts"] += 1
        try:
            if bound is None:
                raise RuntimeError("The LLM judge was selected, but no model call was configured.")
            text = bound(system, user)
        except Exception:
            attempts["failures"] += 1
            raise
        attempts["successes"] += 1
        return text

    judged: list[dict[str, Any]] = []
    consecutive = 0
    backoff_step = 0
    stopped = False
    index = 0
    claims = acquired["claims"]
    while index < len(claims):
        row = claims[index]
        if stopped:
            judged.append(
                _unfinished(
                    row,
                    "pending",
                    "Not requested because the provider rate limit continued. The claim stays pending and is not treated as false.",
                )
            )
            index += 1
            continue
        before = len(meter.calls) if meter is not None else 0
        try:
            result = judge_evidence.judge_one(
                row,
                bodies,
                judge=judge,
                complete=counted if bound is not None else None,
                model=chosen,
                meter=meter,
            )
        except RuntimeError as exc:
            limited, wait = _rate_limit_wait(exc)
            if not limited:
                result = _unfinished(
                    row,
                    "model_error",
                    f"The model request failed ({type(exc).__name__}). No provider fallback was used. The claim is not treated as false.",
                )
            else:
                consecutive += 1
                delay = _wait_for_limit(wait, backoff_step, rng)
                if consecutive >= MAX_CONSECUTIVE_RATE_LIMITS or delay is None:
                    stopped = True
                    result = _unfinished(
                        row,
                        "rate_limited",
                        "The provider rate limit continued. This request was not cached as a verdict, and the remaining claims stay pending. The claim is not treated as false.",
                    )
                    if isinstance(wait, int | float):
                        result["retry_after_s"] = wait
                else:
                    sleeper(delay)
                    backoff_step += 1
                    continue
        else:
            consecutive = 0
            backoff_step = 0
            if result.get("verdict") in VERDICTS:
                result["judge_status"] = "completed"
                result["cache"] = "judged"
            elif result.get("judge_status") is None:
                result["judge_status"] = result.get("validation_status") or "model_error"
                result["cache"] = result["judge_status"]
        if meter is not None:
            result["call_usage"] = _usage_of(meter.calls[before:])
        judged.append(result)
        index += 1
    result = {"method": judge_evidence._method(judge, chosen), "claims": judged}
    usage = meter.summary() if meter is not None else {"calls": 0, "elapsed_s": 0, "cost_usd": "unknown", "served_models": []}
    if not usage.get("requests"):
        usage["request_attempts"] = attempts["attempts"]
        usage["successful_calls"] = attempts["successes"]
        usage["failed_attempts"] = attempts["failures"]
        usage["cache_hits"] = 0
    else:
        usage["failed_attempts"] = sum(1 for item in usage["requests"] if not item.get("ok"))
    result["usage"] = usage
    return result


def _usage_of(calls: list[dict[str, Any]]) -> dict[str, Any]:
    import judge_evidence

    meter = judge_evidence.CallMeter()
    for call in calls:
        meter.add(call)
    return meter.summary()


def _prompt_version() -> str:
    """Hash the judge prompt and checklist so either edit misses the old verdict."""
    import judge_evidence

    text = judge_evidence.JUDGE_PROMPT.read_text(encoding="utf-8") + "\n" + judge_evidence._CHECKLIST
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_sha256(row: dict[str, Any] | None) -> str:
    """Hash the spans that will be judged, not only the stored page."""
    parts: list[str] = []
    for passage in (row or {}).get("passages") or []:
        parts.append(f"{passage.get('start')}:{passage.get('end')}:{passage.get('quote')}")
    document = (row or {}).get("document") or {}
    if document:
        parts.append(f"doc:{document.get('start')}:{document.get('end')}:{document.get('quote')}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _run_key(
    case: dict[str, Any],
    page: dict[str, Any],
    *,
    judge: str,
    model: str,
    allow_openrouter_fallback: bool,
    allow_paid: bool,
    context_limit: int = CONTEXT_LIMIT,
    evidence: dict[str, Any] | None = None,
) -> str:
    payload = {
        "allow_openrouter_fallback": allow_openrouter_fallback,
        "allow_paid": allow_paid,
        "attest_complete": bool(case.get("attest_complete")),
        "claim": case.get("claim") or "",
        "content_sha256": page.get("content_sha256"),
        "context_limit": context_limit,
        "evidence_sha256": _evidence_sha256(evidence),
        "judge": judge,
        "model": model,
        "omission": bool(case.get("omission")),
        "prompt_sha256": _prompt_version(),
        "url": case.get("url") or "",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _page_cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / "pages" / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"


def _cached_page(
    url: str,
    *,
    getter: Callable[[str], dict[str, Any]] | None,
    refresh: bool,
    cache_dir: Path,
) -> tuple[dict[str, Any], str]:
    path = _page_cache_path(cache_dir, url)
    if path.is_file() and not refresh:
        page = json.loads(path.read_text(encoding="utf-8"))
        return page, "hit"
    page = fetch_page(url, getter=getter)
    page["source_id"] = _source_id(page)
    if page.get("fetch_status") in {"ok", "truncated"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return page, "fetched"


def _usable_verdict(path: Path) -> dict[str, Any] | None:
    """A cached verdict must be one of the three labels. A rate limit is not reusable."""
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("judge_status") in {"rate_limited", "pending", "model_error"} or saved.get("verdict") not in VERDICTS:
        path.unlink()
        return None
    saved["cache"] = "hit"
    saved["judge_status"] = saved.get("judge_status") or "completed"
    return saved


def check_urls(
    cases: list[dict[str, Any]],
    *,
    getter: Callable[[str], dict[str, Any]] | None = None,
    judge: str = "llm",
    model: str | None = None,
    allow_openrouter_fallback: bool = True,
    allow_paid: bool = False,
    refresh: bool = False,
    rejudge: bool = False,
    cache_dir: Path | None = None,
    context_limit: int = CONTEXT_LIMIT,
    complete: Callable[[str, str], str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Fetch, select evidence, and judge. Pages and verdicts are cached separately."""
    cache_dir = cache_dir or CACHE
    resolved = model
    if judge == "llm" and not resolved:
        import judge_evidence

        resolved = judge_evidence._llm_client().DEFAULT_MODEL
    resolved = resolved or judge
    pages: dict[str, dict[str, Any]] = {}
    page_state: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    keys: list[str] = []
    for index, case in enumerate(cases, start=1):
        url = str(case.get("url") or "").strip()
        if url not in pages:
            page, state = _cached_page(url, getter=getter, refresh=refresh, cache_dir=cache_dir)
            pages[url] = page
            page_state[url] = state
            records.append(page)
        page = pages[url]
        rows.append(
            evidence_for_claim(
                str(case.get("claim") or ""),
                page,
                claim_id=str(case.get("id") or f"url{index}"),
                omission=bool(case.get("omission")),
                attest_complete=bool(case.get("attest_complete")),
                context_limit=context_limit,
            )
        )
        keys.append(
            _run_key(
                case,
                page,
                judge=judge,
                model=resolved,
                allow_openrouter_fallback=allow_openrouter_fallback,
                allow_paid=allow_paid,
                context_limit=context_limit,
                evidence=rows[-1],
            )
        )
    acquired = {"pages": records, "claims": rows}
    judged_claims: list[dict[str, Any] | None] = [None] * len(rows)
    pending: list[int] = []
    for index, key in enumerate(keys):
        path = cache_dir / "runs" / f"{key}.json"
        if not rejudge:
            saved = _usable_verdict(path)
            if saved is not None:
                judged_claims[index] = saved
                continue
        pending.append(index)
    fresh: dict[str, Any] | None = None
    if pending:
        subset = {"pages": records, "claims": [rows[index] for index in pending]}
        fresh = judge_acquired(
            subset,
            judge=judge,
            model=model,
            allow_openrouter_fallback=allow_openrouter_fallback,
            allow_paid=allow_paid,
            complete=complete,
            sleeper=sleeper,
            rng=rng,
        )
        for index, result in zip(pending, fresh["claims"]):
            judged_claims[index] = result
            if result.get("verdict") in VERDICTS and result.get("judge_status") == "completed":
                path = cache_dir / "runs" / f"{keys[index]}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                stored = dict(result)
                stored["cache"] = "judged"
                path.write_text(json.dumps(stored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    usage = (fresh or {}).get("usage") or {
        "calls": 0,
        "elapsed_s": 0,
        "cost_usd": "unknown",
        "served_models": [],
        "request_attempts": 0,
        "successful_calls": 0,
        "failed_attempts": 0,
    }
    claim_cache = [item.get("cache") if item else None for item in judged_claims]
    usage["cases"] = len(rows)
    usage["cache_hits"] = sum(1 for item in claim_cache if item == "hit")
    usage.setdefault("request_attempts", 0)
    usage.setdefault("successful_calls", usage.get("calls", 0))
    usage.setdefault("failed_attempts", 0)
    verdicts = {
        "method": (fresh or {}).get("method"),
        "claims": judged_claims,
        "usage": usage,
    }
    return {
        "acquired": acquired,
        "verdicts": verdicts,
        "cache": {
            "refresh": refresh,
            "rejudge": rejudge,
            "pages": page_state,
            "claims": claim_cache,
            "cases": usage["cases"],
            "cache_hits": usage["cache_hits"],
            "request_attempts": usage["request_attempts"],
            "successful_calls": usage["successful_calls"],
            "failed_attempts": usage["failed_attempts"],
        },
    }


def html_to_text(raw: str) -> str:
    """Drop tags. Offsets later refer to this text, not to the original HTML."""
    if "<" not in raw:
        return raw.strip()
    cleaned = re.sub(r"(?is)<(script|style|noscript|title)[^>]*>.*?</\1>", " ", raw)
    cleaned = re.sub(r"(?is)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?is)</p>", "\n\n", cleaned)
    cleaned = re.sub(r"(?is)</h[1-6]>", "\n\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _http_get(url: str) -> dict[str, Any]:
    """GET through curl so TLS uses the system trust store. A failed read is returned, not raised."""
    with tempfile.TemporaryDirectory() as tmp:
        body_path = Path(tmp) / "body"
        header_path = Path(tmp) / "headers"
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "-A",
                USER_AGENT,
                "--max-time",
                str(TIMEOUT),
                "-D",
                str(header_path),
                "-o",
                str(body_path),
                "-w",
                "%{http_code}\t%{url_effective}",
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or "curl failed").strip()
            return {"http_status": None, "final_url": url, "raw": "", "truncated": False, "error": message}
        code_text, _, final = result.stdout.strip().partition("\t")
        try:
            status = int(code_text or "0")
        except ValueError:
            status = 0
        raw_bytes = body_path.read_bytes() if body_path.is_file() else b""
        header_text = header_path.read_text(encoding="utf-8", errors="replace") if header_path.is_file() else ""
        charset, declared = _final_headers(header_text)
        truncated = declared is not None and declared > len(raw_bytes)
        if status >= 400:
            return {"http_status": status, "final_url": final or url, "raw": "", "truncated": truncated, "error": f"HTTP {status}"}
        return {
            "http_status": status,
            "final_url": final or url,
            "raw": raw_bytes.decode(charset, errors="replace"),
            "truncated": truncated,
            "error": None,
        }


def _final_headers(text: str) -> tuple[str, int | None]:
    blocks = [block for block in re.split(r"\r?\n\r?\n", text) if block.strip()]
    charset = "utf-8"
    declared = None
    for line in (blocks[-1] if blocks else "").splitlines():
        name, _, value = line.partition(":")
        if name.lower() == "content-type":
            match = re.search(r"charset=([^\s;]+)", value, re.IGNORECASE)
            if match:
                charset = match.group(1).strip("\"'")
        if name.lower() == "content-length" and value.strip().isdigit():
            declared = int(value.strip())
    return charset, declared


def page_title(raw: str) -> str:
    """Read the page title as text. It is stored metadata, not an instruction."""
    if "<" not in raw:
        return next((line.strip() for line in raw.splitlines() if line.strip()), "")[:180]
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw) or re.search(r"(?is)<h1[^>]*>(.*?)</h1>", raw)
    if not match:
        return ""
    text = re.sub(r"(?is)<[^>]+>", " ", match.group(1))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _page(
    original: str,
    final_url: str,
    fetched_at: str,
    http_status: int | None,
    body: str,
    truncated: bool,
    fetch_status: str,
    error: str | None,
    title: str = "",
) -> dict[str, Any]:
    return {
        "original_url": original,
        "final_url": final_url,
        "title": title,
        "fetched_at": fetched_at,
        "http_status": http_status,
        "body": body,
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "truncated": truncated,
        "fetch_status": fetch_status,
        "error": error,
        "is_complete": False,
        "completeness": "unconfirmed",
    }


def _source_id(page: dict[str, Any]) -> str:
    digest = hashlib.sha256(f"{page.get('final_url')}\n{page.get('body') or ''}".encode("utf-8")).hexdigest()
    return "src_" + digest[:12]


def _source_view(page: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    return {
        "source_id": page.get("source_id"),
        "title": page.get("title") or "",
        "document_type": "webpage",
        "document_ref": page.get("final_url"),
        "is_complete": complete,
        "fictional": False,
        "original_url": page.get("original_url"),
        "final_url": page.get("final_url"),
        "fetched_at": page.get("fetched_at"),
        "content_sha256": page.get("content_sha256"),
        "http_status": page.get("http_status"),
        "truncated": bool(page.get("truncated")),
        "completeness": "confirmed" if complete else "unconfirmed",
    }


def _undecided(row: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    row["status"] = status
    row["reason"] = f"{reason} {NOT_FALSE}"
    return row


def _omission_row(
    row: dict[str, Any],
    page: dict[str, Any],
    body: str,
    *,
    attest_complete: bool,
    context_limit: int,
) -> dict[str, Any]:
    if not attest_complete:
        span = {"source_id": page["source_id"], "quote": body, "start": 0, "end": len(body), "is_complete": False}
        acquire_evidence._require_match(body, span)
        row["status"] = "ready"
        row["omission"] = True
        row["source"] = _source_view(page, complete=False)
        row["document"] = span
        row["reason"] = (
            "The page was fetched, but a web fetch is not a complete document. "
            "The omission is insufficient evidence until that scope is confirmed. "
            + NOT_FALSE
        )
        return row
    if page.get("truncated") or len(body) > context_limit:
        return _undecided(
            row,
            "truncated",
            "The caller marked the page complete, but the full text did not fit in the check. The omission is not checked.",
        )
    span = {"source_id": page["source_id"], "quote": body, "start": 0, "end": len(body), "is_complete": True}
    acquire_evidence._require_match(body, span)
    row["status"] = "ready"
    row["omission"] = True
    row["source"] = _source_view(page, complete=True)
    row["source"]["completeness_basis"] = "The caller attested that this fetch is the complete document, and the full text entered the check."
    row["document"] = span
    row["reason"] = "The caller confirmed this fetch is complete and the full text is in range. Silence in that text is still not a verdict by itself."
    return row


def _best_paragraph(claim: str, source_id: str, body: str) -> dict[str, Any] | None:
    claim_row = {"source_quote": claim, "normalized_claim": claim}
    best: dict[str, Any] | None = None
    best_score = -1
    for start, end, paragraph in _paragraphs(body):
        candidate, score = _scored_span(claim_row, source_id, body, start, end, paragraph)
        if candidate is None or score <= best_score:
            continue
        best = candidate
        best_score = score
    return best


def _scored_span(
    claim_row: dict[str, Any],
    source_id: str,
    body: str,
    start: int,
    end: int,
    text: str,
) -> tuple[dict[str, Any] | None, int]:
    if len(text) > 800:
        best = None
        best_score = -1
        for sent_start, sent_end, sentence in acquire_evidence._sentences(text):
            score = acquire_evidence._passage_score(claim_row, sentence)
            if score is None or score <= best_score:
                continue
            absolute_start = start + sent_start
            passage = {"source_id": source_id, "quote": sentence, "start": absolute_start, "end": absolute_start + len(sentence)}
            if acquire_evidence.quote_matches(body, sentence, passage["start"], passage["end"]):
                best = passage
                best_score = score
        return best, best_score
    score = acquire_evidence._passage_score(claim_row, text)
    if score is None:
        return None, -1
    passage = {"source_id": source_id, "quote": text, "start": start, "end": end}
    if not acquire_evidence.quote_matches(body, text, start, end):
        return None, -1
    return passage, score


def _paragraphs(body: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"(?:[^\n]|\n(?!\n))+", body):
        raw = match.group(0)
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        start = match.start() + lead
        end = match.end() - trail
        if end <= start:
            continue
        spans.append((start, end, body[start:end]))
    return spans or [(0, len(body), body)]


def _print_acquired(acquired: dict[str, Any], judged: dict[str, Any] | None, reviews: list[dict[str, Any]]) -> None:
    by_id = {page["source_id"]: page for page in acquired["pages"]}
    verdicts = {row.get("claim_id"): row for row in (judged or {}).get("claims") or []}
    human = {item.get("id"): item for item in reviews}
    for row in acquired["claims"]:
        page = by_id.get((row.get("source") or {}).get("source_id")) or {}
        print(f"\n{row['claim_id']}  status={row['status']}")
        print(f"  claim: {row.get('normalized_claim')}")
        print(f"  original_url: {page.get('original_url')}")
        print(f"  final_url: {page.get('final_url')}")
        print(f"  title: {page.get('title')}")
        print(f"  fetched_at: {page.get('fetched_at')}")
        print(f"  http_status: {page.get('http_status')}  fetch_status: {page.get('fetch_status')}  truncated: {page.get('truncated')}")
        print(f"  content_sha256: {page.get('content_sha256')}")
        print(f"  completeness: {(row.get('source') or {}).get('completeness')}  is_complete: {(row.get('source') or {}).get('is_complete')}")
        body = page.get("body") or ""
        print(f"  body_chars: {len(body)}")
        if body and len(body) <= 2000:
            print("  body:")
            print(body)
        elif body:
            print("  body_excerpt:")
            print(body[:800])
        if row.get("reason"):
            print(f"  acquire: {row['reason']}")
        for passage in row.get("passages") or []:
            print(f"  evidence [{passage['start']}:{passage['end']}]: {passage['quote']}")
        document = row.get("document")
        if document:
            print(f"  document [{document['start']}:{document['end']}] chars={len(document['quote'])}")
        verdict = verdicts.get(row["claim_id"])
        if verdict:
            label = verdict.get("verdict") if verdict.get("verdict") in VERDICTS else None
            status = verdict.get("judge_status") or ("completed" if label else "runtime_error")
            print(f"  verdict: {label if label else 'null'}  judge_status={status}  cache={verdict.get('cache')}")
            if verdict.get("explanation"):
                print(f"  explanation: {verdict['explanation']}")
            usage = verdict.get("call_usage") or {}
            if usage:
                print(
                    f"  provider={', '.join(usage.get('served_models') or []) or '-'}  "
                    f"calls={usage.get('calls')}  elapsed_s={usage.get('elapsed_s')}  cost_usd={usage.get('cost_usd')}"
                )
            for ref in verdict.get("evidence_refs") or []:
                print(f"  ref {ref.get('source_id')} [{ref.get('start')}:{ref.get('end')}]")
            for limit in verdict.get("limitations") or []:
                print(f"  limit: {limit}")
        review = human.get(row["claim_id"])
        if review and review.get("human_label"):
            actual = (verdict or {}).get("verdict")
            status = (verdict or {}).get("judge_status")
            if actual in VERDICTS:
                mark = "match" if actual == review["human_label"] else "differ"
                print(f"  expected {review['human_label']}  actual {actual}  {mark}")
            else:
                print(f"  expected {review['human_label']}  judge_status={status or 'runtime_error'}  not a semantic error")
            if review.get("human_basis"):
                print(f"  human_basis: {review['human_basis']}")
    if judged and judged.get("usage"):
        usage = judged["usage"]
        print(
            f"\ncases {usage.get('cases', len(acquired['claims']))}  "
            f"cache_hits {usage.get('cache_hits', 0)}  "
            f"request_attempts {usage.get('request_attempts', usage.get('calls', 0))}  "
            f"successful_calls {usage.get('successful_calls', usage.get('calls', 0))}  "
            f"failed_attempts {usage.get('failed_attempts', 0)}"
        )
        print(
            f"llm_calls {usage['calls']}  elapsed_s {usage['elapsed_s']}  "
            f"cost_usd {usage['cost_usd']}  served {', '.join(usage.get('served_models') or []) or '-'}"
        )
        print("llm_calls counts successful responses only. It does not count cache hits or failed attempts.")
    evaluation = _score_labels((judged or {}).get("claims") or [], reviews)
    print(
        f"evaluation cases {evaluation['cases']}  completed {evaluation['completed']}  "
        f"runtime_errors {evaluation['runtime_errors']}  pending {evaluation['pending']}  "
        f"completion {evaluation['completion']}  accuracy {evaluation['accuracy']}  "
        f"false_support {evaluation['false_support']}"
    )
    for item in evaluation["semantic_errors"]:
        print(f"semantic {item['id']}  expected {item['expected']}  actual {item['actual']}")
    for item in evaluation["runtime_error_cases"]:
        print(f"runtime {item['id']}  judge_status={item['judge_status']}")
    if judged and judged.get("method"):
        print(judged["method"]["description"])
    return evaluation


def _score_labels(claims: list[dict[str, Any] | None], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy uses completed verdicts only. Runtime failures stay in the completion denominator."""
    human = {item.get("id"): item.get("human_label") for item in reviews}
    completed = 0
    runtime_errors = 0
    pending = 0
    correct = 0
    labeled = 0
    false_support = 0
    semantic: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    for row in claims:
        if not row or row.get("judge_status") == "pending":
            pending += 1
            continue
        verdict = row.get("verdict")
        if verdict in VERDICTS:
            completed += 1
            expected = human.get(row.get("claim_id"))
            if expected:
                labeled += 1
                if verdict == expected:
                    correct += 1
                else:
                    semantic.append({"id": row.get("claim_id"), "expected": expected, "actual": verdict})
                if verdict == "supported" and expected != "supported":
                    false_support += 1
            continue
        runtime_errors += 1
        runtime.append({"id": row.get("claim_id"), "judge_status": row.get("judge_status") or "runtime_error"})
    total = len(claims)
    return {
        "cases": total,
        "completed": completed,
        "runtime_errors": runtime_errors,
        "pending": pending,
        "completion": f"{completed}/{total}" if total else "0/0",
        "accuracy": f"{correct}/{labeled}" if labeled else None,
        "false_support": false_support,
        "semantic_errors": semantic,
        "runtime_error_cases": runtime,
    }


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    reviews = []
    for case in cases:
        reviews.append({"id": case.get("id"), "human_label": case.get("human_label"), "human_basis": case.get("human_basis")})
    return cases, reviews


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch a user-supplied URL and optionally judge the saved text.")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--claim")
    parser.add_argument("--url")
    parser.add_argument("--omission", action="store_true")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--judge", choices=("none", "fixed_rules", "llm"), default="llm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-openrouter-fallback", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Refetch pages. Cached verdicts are kept.")
    parser.add_argument("--rejudge", action="store_true", help="Ignore cached verdicts. Cached pages are kept.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.cases:
        cases, reviews = _load_cases(args.cases)
    elif args.claim and args.url:
        cases = [{"id": "url1", "claim": args.claim, "url": args.url, "omission": args.omission, "attest_complete": args.complete}]
        reviews = []
    else:
        parser.error("Provide --cases or both --claim and --url.")
        return
    if args.judge == "none":
        acquired = acquire_cases(cases)
        judged = None
        payload = {"acquired": acquired, "verdicts": None, "human_labels": reviews, "cache": {"refresh": args.refresh, "rejudge": args.rejudge}}
    else:
        checked = check_urls(
            cases,
            judge=args.judge,
            model=args.model,
            allow_openrouter_fallback=not args.no_openrouter_fallback,
            allow_paid=args.allow_paid,
            refresh=args.refresh,
            rejudge=args.rejudge,
        )
        acquired = checked["acquired"]
        judged = checked["verdicts"]
        payload = {**checked, "human_labels": reviews, "input": {"judge": args.judge, "model": args.model, "allow_paid": args.allow_paid, "openrouter_fallback": not args.no_openrouter_fallback, "refresh": args.refresh, "rejudge": args.rejudge}}
    payload["evaluation"] = _print_acquired(acquired, judged, reviews)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
