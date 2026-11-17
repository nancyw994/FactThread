#!/usr/bin/env python3
"""Find candidate pages for a claim, then judge the fetched text.

Search snippets are used only to discover URLs. The judge sees passages cut
from the fetched page body. A missing source, a failed fetch, and a model
rate limit stay separate from a refuted verdict. Paid search and paid model
fallback are not turned on.

A later run reuses finished queries, candidate URLs, and fetched passages.
After a model rate limit, search and fetch still continue for the other claims.
Search retries honor Retry-After and stop at the same cap as the judge.
OpenRouter and paid model fallback stay off unless the command asks for them.

    .venv/bin/python pipeline/search_evidence.py --cases data/dev/search/cases.json --no-openrouter-fallback
    .venv/bin/python pipeline/search_evidence.py --cases data/dev/search/cases.json --no-openrouter-fallback --resume data/dev/search/last_search.json
    .venv/bin/python pipeline/search_evidence.py --cases data/dev/search/cases.json --no-openrouter-fallback --resume data/dev/search/last_search.json --only search_example
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, urlsplit

import acquire_url

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "dev" / "search" / "cases.json"
OUT = ROOT / "data" / "dev" / "search" / "last_search.json"
MAX_QUERIES = 2
MAX_CANDIDATES = 5
MAX_PAGES = 3
PIPELINE_VERSION = "source-selection-2"
CACHE_VERSION = "page-body-v1"
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"
_COOLDOWN_LOCK = threading.Lock()
VERDICTS = acquire_url.VERDICTS
SEARCH_DONE = frozenset({"ok", "source_not_found", "search_unavailable"})
JUDGE_DONE = frozenset({"completed", "failed"})
_OPPOSE = re.compile(r"\b(?:not|no|never|false|incorrect|isn't|is not|aren't|are not)\b", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_URL_IN_CLAIM = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_SPEECH_SOURCE = re.compile(
    r"\b(?P<speaker>https?://[^\s)>\]]+|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})"
    r"\s+(?:states|says|said|claims|writes|wrote|adds|confirms)\b",
    re.IGNORECASE,
)
_RFC_STATEMENT = re.compile(
    r"\bRFC\s+(?P<num>\d{3,5})\s+(?:states|says|said|claims|writes|wrote|adds|confirms|reserves|defines|specifies)\b",
    re.IGNORECASE,
)
SearchFn = Callable[[str, int], tuple[list[dict[str, str]], str | None]]


def build_queries(claim: str, *, limit: int = MAX_QUERIES) -> list[str]:
    """Build compact searches from the claim's entities, relation, and object."""
    text = _SPACE.sub(" ", claim).strip().rstrip(".")
    words = text.split()
    if not words:
        return []
    stop = {"the", "a", "an", "at", "in", "on", "of", "for", "to", "and", "or", "by", "with", "from", "was", "were", "is", "are", "be", "been", "being", "it", "its", "this", "that", "these", "those", "as", "which", "who", "what", "when", "where", "how"}
    content = [word for word in words if word.strip(".,;:()[]\"'").lower() not in stop]
    first = " ".join(words[:12])
    queries = [first]
    relation = re.compile(r"\b(?:dedicat\w*|host\w*|invent\w*|creat\w*|caus\w*|allow\w*|enabl\w*|result\w*|attribut\w*|deriv\w*|own\w*|locat\w*|serv\w*|us\w*|produc\w*|issu\w*|releas\w*|publish\w*|reserv\w*|led|lead\w*|develop\w*|mov\w*|launch\w*|build\w*|establish\w*|represent\w*)\b", re.I)
    match = relation.search(text)
    if match:
        tokens = re.findall(r"[\w.-]+(?:['’][\w-]+)?", text, flags=re.UNICODE)
        second = " ".join(word for word in tokens if word.lower() not in stop)
        if match.group(0).lower() not in second.lower():
            second = f"{second} {match.group(0)}"
    else:
        second = " ".join(word for word in words if _distinctive(word))
        if not second or second == first:
            second = " ".join(content[-8:])
    if second and second != first:
        queries = [second, first]
    return queries[:limit]


def _distinctive(word: str) -> bool:
    bare = word.strip(".,;:()[]\"'")
    if not bare:
        return False
    if any(character.isdigit() for character in bare) or "." in bare or ":" in bare:
        return True
    return len(bare) > 1 and bare.isupper()


def configured_search(query: str, limit: int) -> tuple[list[dict[str, str]], str | None]:
    """Use the DuckDuckGo client already configured in pipeline/lib.py."""
    try:
        import lib
    except Exception as exc:  # noqa: BLE001 — report the missing client instead of switching providers
        return [], (
            f"The configured search client could not be loaded ({type(exc).__name__}: {exc}). "
            "A paid search service was not enabled."
        )
    if getattr(lib, "DDGS", None) is None:
        return [], (
            "No search service is available. FactThread is configured to use DuckDuckGo through ddgs, "
            "and that client is not installed. A paid search service was not enabled."
        )
    hits = lib.search_web(query, limit=limit)
    if len(hits) == 1 and hits[0].get("title") == "search-error":
        return [], (
            f"The configured DuckDuckGo search failed ({hits[0].get('quote') or 'unknown error'}). "
            "A paid search service was not enabled."
        )
    return hits, None


def _rate_limited_text(text: str) -> tuple[bool, float | None]:
    """Reuse the judge's Retry-After parser for a search error string."""
    limited, wait = acquire_url._rate_limit_wait(RuntimeError(text))
    if not limited:
        return False, None
    if wait is not None:
        return True, wait
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*s", text)
    if seconds and ("retry" in text.lower() or "try again" in text.lower()):
        return True, float(seconds.group(1))
    return True, None


def collect_candidates(
    claim: str,
    search: SearchFn,
    *,
    query_limit: int = MAX_QUERIES,
    candidate_limit: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    """Search at most two queries and keep at most five unique public URLs."""
    queries = build_queries(claim, limit=query_limit)
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    errors: list[str] = []
    for query in queries:
        hits, error = search(query, candidate_limit)
        if error:
            errors.append(error)
            continue
        for hit in hits:
            url = str(hit.get("url") or "")
            if not _public_http(url):
                continue
            key = _url_key(url)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "query": query,
                    "url": url,
                    "title": str(hit.get("title") or ""),
                    "snippet": str(hit.get("quote") or hit.get("snippet") or ""),
                }
            )
            if len(candidates) >= candidate_limit:
                break
        if len(candidates) >= candidate_limit:
            break
    status = "ok"
    message = ""
    if not candidates and errors:
        status = "search_unavailable"
        message = errors[0]
    elif not candidates:
        status = "source_not_found"
        message = "The configured search returned no public URLs. The claim is not treated as false."
    return {"queries": queries, "candidates": candidates, "status": status, "message": message}


def named_sources(claim: str) -> list[dict[str, Any]]:
    """URLs written in the claim, and hosts the claim says the statement comes from."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(url: str, *, attributed: bool, explicit_url: bool, role: str = "original") -> None:
        host = _host(url)
        if not host or host in seen:
            if host in seen and attributed:
                for item in found:
                    if item["host"] == host:
                        item["attributed"] = True
                        item["role"] = "original"
            return
        seen.add(host)
        found.append(
            {"url": url, "host": host, "attributed": attributed, "explicit_url": explicit_url, "role": role}
        )

    for match in _SPEECH_SOURCE.finditer(claim):
        speaker = match.group("speaker").rstrip(".,;:")
        if speaker.lower().startswith("http"):
            add(speaker, attributed=True, explicit_url=True)
        else:
            host = speaker.lower().removeprefix("www.")
            add(f"https://{host}", attributed=True, explicit_url=False)
    for match in _RFC_STATEMENT.finditer(claim):
        for url in _rfc_originals(match.group("num")):
            add(url, attributed=True, explicit_url=False)
    for match in _URL_IN_CLAIM.finditer(claim):
        add(match.group(0).rstrip(".,;:"), attributed=False, explicit_url=True)
    return found


def _rfc_originals(number: str) -> list[str]:
    """The RFC Editor and IETF copies of a numbered RFC. Other sites are not this document."""
    num = str(int(number))
    return [
        f"https://www.rfc-editor.org/rfc/rfc{num}.html",
        f"https://datatracker.ietf.org/doc/html/rfc{num}",
    ]


def select_pages(candidates: list[dict[str, str]], *, limit: int = MAX_PAGES) -> list[dict[str, str]]:
    """Read at most three pages. A source named by the claim comes first."""
    named = [item for item in candidates if item.get("named")]
    others = [item for item in candidates if not item.get("named")]
    chosen: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in named + others:
        key = _url_key(item["url"])
        if key in seen:
            continue
        chosen.append(item)
        seen.add(key)
        if len(chosen) >= limit:
            break
    if len(named) + len(others) > limit:
        chosen_keys = set(seen)
        for extra in others:
            if _url_key(extra["url"]) in chosen_keys:
                continue
            if not _OPPOSE.search(extra.get("snippet") or ""):
                continue
            for index in range(len(chosen) - 1, -1, -1):
                if not chosen[index].get("named"):
                    chosen[index] = extra
                    break
            break
    return chosen


def investigate(
    cases: list[dict[str, Any]],
    *,
    search: SearchFn | None = None,
    getter: Callable[[str], dict[str, Any]] | None = None,
    judge: str = "llm",
    model: str | None = None,
    allow_openrouter_fallback: bool = False,
    allow_paid: bool = False,
    cache_dir: Path | None = None,
    complete: Callable[[str, str], str] | None = None,
    sleeper: Callable[[float], None] = acquire_url.time.sleep,
    rng: Callable[[], float] = acquire_url.random.random,
    previous: dict[str, Any] | None = None,
    checkpoint: Path | None = None,
    only: set[str] | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
    input_source_url: str | None = None,
) -> dict[str, Any]:
    """Search, fetch, and judge. A saved run is reused and only unfinished stages run again."""
    search = search or configured_search
    saved = {str(item.get("id")): item for item in (previous or {}).get("claims") or []}
    claims: list[dict[str, Any]] = []
    search_paused = False
    search_tries = 0
    store = cache_dir or acquire_url.CACHE
    for index, case in enumerate(cases, start=1):
        record = _adopt(case, index, saved.get(str(case.get("id") or f"search{index}")))
        record["article_source_url"] = input_source_url or record.get("article_source_url")
        if only and record["id"] not in only:
            claims.append(record)
            continue
        finished = record["judge_status"] == "completed" and record["verdict"] in VERDICTS
        if finished and _named_source_was_read(record):
            claims.append(record)
            continue
        if finished:
            record["judge_status"] = "not_run"
            record["verdict"] = None
            record["citation_passed"] = None
            record["explanation"] = None
        selecting = record["search_status"] not in SEARCH_DONE
        if selecting:
            if record["judge_status"] != "completed":
                record["limitations"] = []
            _report(progress, "search", record)
            if search_paused:
                _note(record, "Search was not run because the search service rate limit continued. A paid search service was not enabled. The claim is not treated as false.")
            else:
                stopped = _continue_search(record, search, sleeper=sleeper, rng=rng, tries=search_tries)
                search_tries = stopped[1]
                if stopped[0]:
                    search_paused = True
            _report(progress, "search", record)
        _ensure_named_sources(record)
        if record["candidates"] and not _fetch_complete(record):
            selecting = True
            _report(progress, "fetch", record)
            _continue_fetch(record, getter=getter, cache_dir=store)
        if selecting:
            record["sources_selected_at"] = datetime.now(timezone.utc).isoformat()
        _record_direct_fetch(record)
        _mark_evidence(record)
        record["key_evidence_obtained"] = _key_evidence_obtained(record)
        _report(progress, "fetch", record)
        claims.append(record)
    _checkpoint(checkpoint, claims, allow_paid=allow_paid, allow_openrouter_fallback=allow_openrouter_fallback)
    judge_paused = False
    for record in claims:
        if only and record["id"] not in only:
            continue
        if judge_paused:
            if record["judge_status"] not in JUDGE_DONE | {"rate_limited"}:
                record["judge_status"] = "not_run"
                _note(record, "The model was not called because the provider rate limit continued. Saved evidence is kept. The claim is not treated as false.")
            continue
        if not _needs_judge(record, judge):
            continue
        provider, model_name = _model_identity(model)
        record["provider"] = provider
        record["model"] = model_name
        deadline = shared_retry_at(store, provider, model_name)
        if deadline is not None and deadline > datetime.now(timezone.utc):
            record["judge_status"] = "rate_limited"
            record["retry_at"] = deadline.isoformat()
            record["retry_after_s"] = (deadline - datetime.now(timezone.utc)).total_seconds()
            _note(record, "The model was not called because the provider rate limit continued. Saved evidence is kept. The claim is not treated as false.")
            judge_paused = True
            _report(progress, "judge", record)
            _checkpoint(checkpoint, claims, allow_paid=allow_paid, allow_openrouter_fallback=allow_openrouter_fallback)
            continue
        _report(progress, "judge", record)
        _judge_record(
            record,
            getter=getter,
            judge=judge,
            model=model,
            allow_openrouter_fallback=allow_openrouter_fallback,
            allow_paid=allow_paid,
            cache_dir=store,
            complete=complete,
            sleeper=sleeper,
            rng=rng,
            input_source_url=record.get("article_source_url"),
        )
        _mark_evidence(record)
        if record.get("judge_status") == "rate_limited" and record.get("retry_at"):
            kept = remember_shared_retry(store, provider, model_name, str(record["retry_at"]))
            if kept is not None:
                record["retry_at"] = kept.isoformat()
                record["retry_after_s"] = (kept - datetime.now(timezone.utc)).total_seconds()
        _report(progress, "judge", record)
        _checkpoint(checkpoint, claims, allow_paid=allow_paid, allow_openrouter_fallback=allow_openrouter_fallback)
        if record["judge_status"] == "rate_limited":
            judge_paused = True
    return _payload(claims, allow_paid=allow_paid, allow_openrouter_fallback=allow_openrouter_fallback)


def _blank(case: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": str(case.get("id") or f"search{index}"),
        "claim": str(case.get("claim") or ""),
        "human_label": case.get("human_label"),
        "human_basis": case.get("human_basis"),
        "queries": [],
        "candidates": [],
        "selected": [],
        "fetches": [],
        "passages": [],
        "sources": [],
        "evidence_found": None,
        "page_fetch_succeeded": None,
        "relevant_passage_found": None,
        "sufficient_evidence_found": None,
        "verdict": None,
        "search_status": "not_run",
        "fetch_status": "not_run",
        "judge_status": "not_run",
        "citation_passed": None,
        "limitations": [],
        "pipeline_version": PIPELINE_VERSION,
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources_selected_at": None,
        "provider": None,
        "model": None,
    }


def _adopt(case: dict[str, Any], index: int, old: dict[str, Any] | None) -> dict[str, Any]:
    """Keep finished stages from a previous run, including a run saved before these statuses existed."""
    record = _blank(case, index)
    if not old:
        return record
    saved = copy.deepcopy(old)
    created = saved.get("created_at") or record["created_at"]
    if saved.get("pipeline_version") != PIPELINE_VERSION:
        record["created_at"] = created
        return record
    record["queries"] = _query_records(saved.get("queries"))
    record["candidates"] = list(saved.get("candidates") or [])
    record["selected"] = list(saved.get("selected") or [])
    record["fetches"] = list(saved.get("fetches") or [])
    record["passages"] = list(saved.get("passages") or [])
    record["sources"] = list(saved.get("sources") or [])
    for field in ("verdict", "explanation", "evidence_refs", "validation_status", "citation_passed", "call_usage", "judge_requests", "retry_after_s", "retry_at", "limitations", "article_source_url", "page_fetch_succeeded", "relevant_passage_found", "sufficient_evidence_found"):
        if field in saved:
            record[field] = saved[field]
    record["human_label"] = case.get("human_label", saved.get("human_label"))
    record["human_basis"] = case.get("human_basis", saved.get("human_basis"))
    record["created_at"] = created
    record["sources_selected_at"] = saved.get("sources_selected_at")
    record["provider"] = saved.get("provider")
    record["model"] = saved.get("model")
    record["search_status"] = saved.get("search_status") or _legacy_search_status(saved)
    record["fetch_status"] = saved.get("fetch_status") or _legacy_fetch_status(saved)
    record["judge_status"] = _legacy_judge_status(saved) if "search_status" not in saved else saved.get("judge_status") or "not_run"
    if record["judge_status"] == "pending":
        record["judge_status"] = "not_run"
    _attach_urls(record)
    _mark_evidence(record)
    if record["verdict"] not in VERDICTS and record["judge_status"] == "completed":
        record["judge_status"] = "not_run"
        record["verdict"] = None
    return record


def _legacy_search_status(saved: dict[str, Any]) -> str:
    old = saved.get("judge_status")
    if old == "search_unavailable":
        return "search_unavailable"
    if old == "source_not_found":
        return "source_not_found"
    if saved.get("candidates") or saved.get("fetches"):
        return "ok"
    if old == "pending" or not saved.get("queries"):
        return "not_run"
    return "ok"


def _legacy_fetch_status(saved: dict[str, Any]) -> str:
    old = saved.get("judge_status")
    if old in {"search_unavailable", "source_not_found", "pending"} and not saved.get("fetches"):
        return "not_run"
    if not saved.get("fetches"):
        return "not_run"
    statuses = [item.get("fetch_status") for item in saved.get("fetches") or []]
    if statuses and all(item == "fetch_failed" for item in statuses):
        return "fetch_failed"
    if saved.get("passages") or saved.get("evidence_found") is True:
        return "ok"
    if any(item == "ok" for item in statuses):
        return "passage_not_found"
    return "fetch_failed"


def _legacy_judge_status(saved: dict[str, Any]) -> str:
    old = saved.get("judge_status")
    if old in {"search_unavailable", "source_not_found", "fetch_failed", "passage_not_found", "pending", "not_judged"}:
        return "not_run"
    if saved.get("verdict") in VERDICTS:
        return "completed"
    return old or "not_run"


def _query_records(raw: Any) -> list[dict[str, Any]]:
    records = []
    for item in raw or []:
        if isinstance(item, str):
            records.append({"query": item, "status": "ok", "error": None})
        elif isinstance(item, dict) and item.get("query"):
            records.append(
                {"query": str(item["query"]), "status": item.get("status") or "ok", "error": item.get("error")}
            )
    return records


def _attach_urls(record: dict[str, Any]) -> None:
    by_id = {}
    for fetched in record.get("fetches") or []:
        passage = fetched.get("passage") or {}
        url = fetched.get("final_url") or fetched.get("url")
        if passage and not passage.get("url"):
            passage["url"] = url
        if passage.get("source_id"):
            by_id[passage["source_id"]] = url
    for ref in record.get("evidence_refs") or []:
        if isinstance(ref, dict) and not ref.get("url") and ref.get("source_id") in by_id:
            ref["url"] = by_id[ref["source_id"]]


def _continue_search(
    record: dict[str, Any],
    search: SearchFn,
    *,
    sleeper: Callable[[float], None],
    rng: Callable[[], float],
    tries: int,
) -> tuple[bool, int]:
    """Run queries that are not already saved. Returns whether search must pause, and the try count."""
    done = {item["query"] for item in record["queries"] if item.get("status") == "ok"}
    seen = {_url_key(item["url"]) for item in record["candidates"]}
    paused = False
    errors: list[str] = []
    record["queries"] = [item for item in record["queries"] if item.get("status") == "ok"]
    for query in build_queries(record["claim"]):
        if len(record["candidates"]) >= MAX_CANDIDATES or query in done:
            continue
        hits, error, paused, tries = _search_once(search, query, sleeper=sleeper, rng=rng, tries=tries)
        if paused:
            record["queries"].append({"query": query, "status": "search_rate_limited", "error": error})
            record["search_status"] = "search_rate_limited"
            if error:
                _note(record, error)
            _note(record, "The claim is not treated as false.")
            break
        if error:
            record["queries"].append({"query": query, "status": "search_unavailable", "error": error})
            errors.append(error)
            continue
        record["queries"].append({"query": query, "status": "ok", "error": None})
        _add_hits(record, query, hits, seen)
        if len(record["candidates"]) >= MAX_CANDIDATES:
            break
    if not paused:
        if record["candidates"]:
            record["search_status"] = "ok"
        elif errors:
            record["search_status"] = "search_unavailable"
            _note(record, errors[0])
            _note(record, "The claim is not treated as false.")
        elif record["queries"]:
            record["search_status"] = "source_not_found"
            _note(record, "The configured search returned no public URLs. The claim is not treated as false.")
        else:
            record["search_status"] = "source_not_found"
            _note(record, "The configured search returned no public URLs. The claim is not treated as false.")
    record["selected"] = [
        {"url": item["url"], "title": item["title"], "query": item["query"]}
        for item in select_pages(record["candidates"])
    ]
    return paused, tries


def _search_once(
    search: SearchFn,
    query: str,
    *,
    sleeper: Callable[[float], None],
    rng: Callable[[], float],
    tries: int,
) -> tuple[list[dict[str, str]], str | None, bool, int]:
    hits, error = search(query, MAX_CANDIDATES)
    limited, wait = _rate_limited_text(error or "")
    if not error or not limited:
        return hits, error, False, 0
    tries += 1
    delay = acquire_url._wait_for_limit(wait, tries - 1, rng)
    if tries >= acquire_url.MAX_CONSECUTIVE_RATE_LIMITS or delay is None:
        return [], error, True, tries
    sleeper(delay)
    hits, error = search(query, MAX_CANDIDATES)
    limited, _wait = _rate_limited_text(error or "")
    if error and limited:
        return [], error, True, tries + 1
    return hits, error, False, 0


def _ensure_named_sources(record: dict[str, Any]) -> None:
    """Put a source the claim itself names at the front, even when search never returned it."""
    named = named_sources(record["claim"])
    record["named_sources"] = named
    if not named:
        record["selected"] = [
            {"url": item["url"], "title": item["title"], "query": item["query"]}
            for item in select_pages(record["candidates"])
        ]
        return
    by_host = {_host(item.get("url") or ""): item for item in record["candidates"]}
    for source in reversed(named):
        current = by_host.get(source["host"])
        if current is not None:
            current["named"] = True
            current["attributed"] = bool(source["attributed"])
            current["role"] = source.get("role") or "original"
            record["candidates"].remove(current)
            record["candidates"].insert(0, current)
            continue
        added = {
            "query": "named source",
            "url": source["url"],
            "title": source.get("document") or source["host"],
            "snippet": "",
            "named": True,
            "attributed": bool(source["attributed"]),
            "role": source.get("role") or "original",
        }
        record["candidates"].insert(0, added)
        by_host[source["host"]] = added
    kept_named = [item for item in record["candidates"] if item.get("named")]
    others = [item for item in record["candidates"] if not item.get("named")]
    room = max(MAX_CANDIDATES, len(kept_named)) - len(kept_named)
    kept_others = others[:room]
    opposing = next((item for item in others if item not in kept_others and _OPPOSE.search(item.get("snippet") or "")), None)
    if opposing is not None and kept_others:
        kept_others[-1] = opposing
    record["candidates"] = kept_named + kept_others
    record["selected"] = [
        {"url": item["url"], "title": item["title"], "query": item["query"]}
        for item in select_pages(record["candidates"])
    ]
    if record["search_status"] == "source_not_found" and record["candidates"]:
        record["search_status"] = "ok"


def _record_direct_fetch(record: dict[str, Any]) -> None:
    """Keep a failed read of a named source. Another site does not replace that failure."""
    fetches = record.get("fetches") or []
    for source in record.get("named_sources") or []:
        match = next((item for item in fetches if _host(item.get("final_url") or item.get("url") or "") == source["host"]), None)
        if match is None:
            source["fetch_status"] = "not_run"
            continue
        source["fetch_status"] = match.get("fetch_status")
        source["error"] = match.get("error")
        source["final_url"] = match.get("final_url")
        if match.get("fetch_status") not in {"ok", "truncated"}:
            reason = match.get("error") or match.get("fetch_status") or "request failed"
            _note(
                record,
                f"The named source {source['url']} could not be read ({reason}). "
                "A third-party page can support the content, but it does not establish that this site said it. "
                "The claim is not treated as false.",
            )


def _cited_hosts(record: dict[str, Any]) -> set[str]:
    hosts = set()
    for ref in record.get("evidence_refs") or []:
        host = _host(str(ref.get("url") or ""))
        if host:
            hosts.add(host)
    return hosts


def _apply_named_attribution(record: dict[str, Any]) -> None:
    """A page about a site does not prove that the site itself made the statement."""
    attributed = [item["host"] for item in record.get("named_sources") or [] if item.get("attributed")]
    if not attributed or record.get("verdict") != "supported":
        return
    if any(host in _cited_hosts(record) for host in attributed):
        record["attribution_check"] = "passed"
        return
    host = attributed[0]
    record["content_check"] = "supported"
    record["attribution_check"] = "failed"
    record["verdict"] = "insufficient_evidence"
    record["judge_status"] = "completed"
    record["explanation"] = (
        f"The supplied pages may support the content, but none is {host}. "
        "A third-party page does not establish that the named site said it, so the whole claim is not supported."
    )
    _note(record, record["explanation"])


def _named_source_was_read(record: dict[str, Any]) -> bool:
    """A saved verdict is reusable only after each source named by the claim has been read or has a recorded failure."""
    required = named_sources(str(record.get("claim") or ""))
    if not required:
        return True
    seen = {_host(str(item.get("final_url") or item.get("url") or "")) for item in record.get("fetches") or []}
    return all(item["host"] in seen for item in required)


def _key_evidence_obtained(record: dict[str, Any]) -> bool | None:
    named = named_sources(str(record.get("claim") or ""))
    passages = record.get("passages") or []
    if named:
        hosts = {item["host"] for item in named}
        return any(_host(str(item.get("url") or "")) in hosts for item in passages)
    basis = _SPACE.sub(" ", str(record.get("human_basis") or "")).lower()
    basis_words = {word for word in re.findall(r"[a-z0-9.]{4,}", basis)}
    if len(basis_words) < 4:
        return None
    for passage in passages:
        quote_words = {word for word in re.findall(r"[a-z0-9.]{4,}", str(passage.get("quote") or "").lower())}
        if len(basis_words & quote_words) >= 4:
            return True
    if record.get("fetch_status") in {None, "not_run"} and not passages:
        return None
    return False


def _add_hits(record: dict[str, Any], query: str, hits: list[dict[str, str]], seen: set[str]) -> None:
    for hit in hits:
        url = str(hit.get("url") or "")
        if not _public_http(url):
            continue
        key = _url_key(url)
        if key in seen:
            continue
        seen.add(key)
        record["candidates"].append(
            {
                "query": query,
                "url": url,
                "title": str(hit.get("title") or ""),
                "snippet": str(hit.get("quote") or hit.get("snippet") or ""),
            }
        )
        if len(record["candidates"]) >= MAX_CANDIDATES:
            return


def _fetch_complete(record: dict[str, Any]) -> bool:
    if record["search_status"] in {"source_not_found", "search_unavailable"} and not record["candidates"]:
        return True
    selected = {item["url"] for item in record.get("selected") or []}
    if not selected:
        return record["search_status"] in SEARCH_DONE
    saved = {item.get("url"): item for item in record.get("fetches") or []}
    for url in selected:
        fetched = saved.get(url) or {}
        if fetched.get("fetch_status") not in {"ok", "truncated", "empty_body"}:
            return False
        if _is_named_page(record, url) and not fetched.get("passage"):
            return False
    return True


def _continue_fetch(record: dict[str, Any], *, getter: Callable[[str], dict[str, Any]] | None, cache_dir: Path) -> None:
    saved = {item.get("url"): item for item in record.get("fetches") or []}
    fetches: list[dict[str, Any]] = []
    for item in record.get("selected") or []:
        previous = saved.get(item["url"])
        if previous and previous.get("fetch_status") in {"ok", "truncated", "empty_body"} and (
            previous.get("passage") or not _is_named_page(record, item["url"])
        ):
            fetches.append(previous)
            continue
        page, state = acquire_url._cached_page(item["url"], getter=getter, refresh=False, cache_dir=cache_dir)
        fetched = _fetch_view(record, item, page, state)
        fetches.append(fetched)
    record["fetches"] = fetches
    record["passages"] = [item["passage"] for item in fetches if item.get("passage")]
    record["sources"] = _sources_from(fetches)
    statuses = [item.get("fetch_status") for item in fetches]
    if not fetches:
        record["fetch_status"] = "not_run"
    elif all(item == "fetch_failed" for item in statuses):
        record["fetch_status"] = "fetch_failed"
        _note(record, "No fetched passage was available to judge. The claim is not treated as false.")
    elif record["passages"]:
        record["fetch_status"] = "ok"
    elif any(item == "ok" for item in statuses):
        record["fetch_status"] = "passage_not_found"
        _note(record, "No fetched passage was available to judge. The claim is not treated as false.")
    else:
        record["fetch_status"] = "fetch_failed"
        _note(record, "No fetched passage was available to judge. The claim is not treated as false.")


def _fetch_view(record: dict[str, Any], item: dict[str, str], page: dict[str, Any], state: str) -> dict[str, Any]:
    fetched: dict[str, Any] = {
        "url": item["url"],
        "final_url": page.get("final_url"),
        "title": page.get("title") or item["title"],
        "fetch_status": page.get("fetch_status"),
        "http_status": page.get("http_status"),
        "content_sha256": page.get("content_sha256"),
        "page_cache": state,
        "error": page.get("error"),
        "passage": None,
    }
    row = acquire_url.evidence_for_claim(record["claim"], page, claim_id=record["id"])
    if row.get("status") == "ready" and row.get("passages"):
        passage = dict(row["passages"][0])
    elif _is_named_page(record, item["url"]):
        passage = _named_body_passage(record["claim"], page, host=_host(item["url"]))
    else:
        passage = None
    if passage:
        fetched["passage"] = {
            "source_id": passage.get("source_id"),
            "url": page.get("final_url") or item["url"],
            "start": passage.get("start"),
            "end": passage.get("end"),
            "quote": passage.get("quote"),
        }
    return fetched


def _kept_named_passage(page: dict[str, Any], saved: dict[str, Any] | None) -> dict[str, Any] | None:
    import acquire_evidence

    if not saved or not saved.get("quote"):
        return None
    body = str(page.get("body") or "")
    start, end = saved.get("start"), saved.get("end")
    if not acquire_evidence.quote_matches(body, str(saved["quote"]), start, end):
        return None
    passage = dict(saved)
    passage["source_id"] = page.get("source_id") or saved.get("source_id")
    return passage


def _is_named_page(record: dict[str, Any], url: str) -> bool:
    host = _host(url)
    return any(item.get("host") == host for item in record.get("named_sources") or [])


def _named_body_passage(claim: str, page: dict[str, Any], host: str | None = None) -> dict[str, Any] | None:
    """Keep an exact slice of a named page when no sentence cleared the scorer."""
    import acquire_evidence

    body = str(page.get("body") or "")
    source_id = page.get("source_id")
    if not source_id or not body.strip():
        return None
    if len(body) <= 1000:
        start, end = 0, len(body)
    else:
        unique = list(dict.fromkeys(acquire_evidence._content_tokens(claim)))
        paragraphs = acquire_url._paragraphs(body)
        scores = []
        for index, (start, end, paragraph) in enumerate(paragraphs):
            tokens = set(acquire_evidence._content_tokens(paragraph))
            overlap = sum(1 for token in unique if token in tokens)
            scores.append(overlap)
        needles = [item for item in re.findall(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", claim.lower()) if item != host]
        named = [
            index
            for index, (start, end, paragraph) in enumerate(paragraphs)
            if any(needle in paragraph.lower() for needle in needles)
        ]
        if named:
            index = max(named, key=lambda item: scores[item])
        elif scores and max(scores) >= 2:
            index = scores.index(max(scores))
        else:
            return None
        low = high = index
        while low > 0 and paragraphs[high][1] - paragraphs[low - 1][0] <= 900:
            low -= 1
        while high + 1 < len(paragraphs) and paragraphs[high + 1][1] - paragraphs[low][0] <= 900:
            high += 1
        start, end = paragraphs[low][0], paragraphs[high][1]
    quote = body[start:end]
    if not acquire_evidence.quote_matches(body, quote, start, end):
        return None
    return {"source_id": source_id, "quote": quote, "start": start, "end": end}


def _sources_from(fetches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for fetched in fetches:
        passage = fetched.get("passage") or {}
        if not passage:
            continue
        sources.append(
            {
                "source_id": passage.get("source_id"),
                "url": passage.get("url") or fetched.get("final_url") or fetched.get("url"),
                "title": fetched.get("title") or "",
            }
        )
    return sources


def _mark_evidence(record: dict[str, Any]) -> None:
    record["page_fetch_succeeded"] = any(item.get("fetch_status") in {"ok", "empty_body", "truncated"} for item in record.get("fetches") or [])
    record["relevant_passage_found"] = bool(record.get("passages"))
    record["sufficient_evidence_found"] = (
        record.get("verdict") in {"supported", "refuted"}
        if record.get("judge_status") == "completed"
        else None
    )
    if record["fetch_status"] == "not_run":
        record["evidence_found"] = False if record["search_status"] in {"source_not_found", "search_unavailable"} else None
        return
    record["evidence_found"] = bool(record.get("passages"))


def _needs_judge(record: dict[str, Any], judge: str) -> bool:
    if judge == "none" or not record.get("evidence_found"):
        return False
    if record.get("judge_status") == "completed" and record.get("verdict") in VERDICTS:
        return False
    if record.get("judge_status") == "failed":
        return False
    return True


def _judge_record(
    record: dict[str, Any],
    *,
    getter: Callable[[str], dict[str, Any]] | None,
    judge: str,
    model: str | None,
    allow_openrouter_fallback: bool,
    allow_paid: bool,
    cache_dir: Path,
    complete: Callable[[str, str], str] | None,
    sleeper: Callable[[float], None],
    rng: Callable[[], float],
    input_source_url: str | None = None,
) -> None:
    pages: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    by_url = {item["url"]: item for item in record.get("selected") or []}
    for fetched in record.get("fetches") or []:
        if not fetched.get("passage"):
            continue
        page, state = acquire_url._cached_page(fetched["url"], getter=getter, refresh=False, cache_dir=cache_dir)
        fetched["page_cache"] = state
        row = acquire_url.evidence_for_claim(record["claim"], page, claim_id=record["id"])
        if row.get("status") == "ready" and row.get("passages"):
            passage = dict(row["passages"][0])
        elif _is_named_page(record, fetched["url"]):
            passage = _kept_named_passage(page, fetched.get("passage"))
            if passage is None:
                continue
        else:
            continue
        passage["url"] = page.get("final_url") or fetched.get("url")
        is_input = bool(input_source_url and _url_key(passage["url"]) == _url_key(input_source_url))
        if is_input:
            passage["source_role"] = "input_article"
            passage["independent"] = False
        passages.append(passage)
        pages.append(page)
        sources.append(
            {
                "source_id": passage.get("source_id"),
                "url": passage.get("url"),
                "title": page.get("title") or (by_url.get(fetched["url"]) or {}).get("title") or "",
                **({"source_role": "input_article", "independent": False} if is_input else {"independent": True}),
            }
        )
        fetched["passage"] = {
            "source_id": passage.get("source_id"),
            "url": passage.get("url"),
            "start": passage.get("start"),
            "end": passage.get("end"),
            "quote": passage.get("quote"),
        }
    record["passages"] = [
        {"source_id": item.get("source_id"), "url": item.get("url"), "start": item.get("start"), "end": item.get("end"), "quote": item.get("quote")}
        for item in passages
    ]
    record["sources"] = sources
    if not passages:
        record["evidence_found"] = False
        record["judge_status"] = "not_run"
        return
    combined = {
        "claim_id": record["id"],
        "normalized_claim": record["claim"],
        "fact_type": "checkable_fact",
        "verification_scope": "external_fact",
        "status": "ready",
        "source": {
            "source_id": sources[0]["source_id"],
            "document_type": "webpage",
            "document_ref": sources[0]["url"],
            "is_complete": False,
        },
        "sources": sources,
        "passages": passages,
        "document": None,
    }
    record["verification_scope"] = combined["verification_scope"]
    record["document_complete"] = False
    judged = acquire_url.judge_acquired(
        {"pages": pages, "claims": [combined]},
        judge=judge,
        model=model,
        allow_openrouter_fallback=allow_openrouter_fallback,
        allow_paid=allow_paid,
        complete=complete,
        sleeper=sleeper,
        rng=rng,
    )
    verdict = judged["claims"][0]
    refs, mismatches = _bind_citations(verdict.get("evidence_refs") or [], pages)
    source_flags = {item.get("source_id"): item for item in sources}
    for ref in refs:
        flag = source_flags.get(ref.get("source_id"), {})
        if flag.get("source_role"):
            ref["source_role"] = flag["source_role"]
        if "independent" in flag:
            ref["independent"] = flag["independent"]
    record["evidence_refs"] = refs
    record["explanation"] = verdict.get("explanation")
    record["document_scope"] = verdict.get("document_scope")
    record["limitations"] = list(verdict.get("limitations") or [])
    record["validation_status"] = verdict.get("validation_status")
    record["call_usage"] = judged.get("usage")
    record["judge_requests"] = list(verdict.get("judge_requests") or [])
    record["verdict"] = verdict.get("verdict")
    record["judge_status"] = "not_run" if verdict.get("judge_status") == "pending" else verdict.get("judge_status")
    _remember_retry(record, verdict)
    _apply_named_attribution(record)
    if record.get("judge_status") != "rate_limited":
        record["retry_at"] = None
    if mismatches:
        record["verdict"] = None
        record["judge_status"] = "failed"
        record["validation_status"] = "failed"
        record["citation_passed"] = False
        _note(record, "A citation did not match its original source. No verdict is published. The claim is not treated as false.")
    elif record["validation_status"] in {"failed", "validation_failed"}:
        record["citation_passed"] = False
    elif record["validation_status"] == "passed" and record["verdict"] in VERDICTS:
        record["citation_passed"] = True
    else:
        record["citation_passed"] = None
    if record["verdict"] not in VERDICTS:
        _note(record, "The claim is not treated as false.")
    independent_refs = [ref for ref in refs if ref.get("independent") is not False and ref.get("source_role") != "input_article"]
    if record["verdict"] in {"supported", "refuted"} and not independent_refs:
        record["verdict"] = "insufficient_evidence"
        note = "The cited evidence comes only from the input article, so it is not independent corroboration."
        record["explanation"] = f"{record.get('explanation') or ''} {note}".strip()
        record.setdefault("limitations", []).append(note)
    _record_direct_fetch(record)


def _bind_citations(refs: list[dict[str, Any]], pages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Each citation must name the source whose stored text contains that exact span."""
    import acquire_evidence

    by_id = {page.get("source_id"): page for page in pages}
    bound: list[dict[str, Any]] = []
    failures: list[str] = []
    for ref in refs:
        page = by_id.get(ref.get("source_id"))
        item = dict(ref)
        if page is None:
            failures.append(f"source {ref.get('source_id')!r} is not one of the fetched pages")
            bound.append(item)
            continue
        url = page.get("final_url") or page.get("original_url")
        cited = item.get("url")
        if cited and _url_key(str(cited)) != _url_key(str(url)):
            failures.append(f"source {item.get('source_id')} was cited with {cited}, not {url}")
        item["url"] = url
        if not acquire_evidence.quote_matches(page.get("body") or "", item.get("quote"), item.get("start"), item.get("end")):
            failures.append(f"source {item.get('source_id')} quote does not match [{item.get('start')}:{item.get('end')}]")
        bound.append(item)
    return bound, failures


def _model_identity(model: str | None) -> tuple[str, str]:
    """The judge uses Groq unless a caller names another model. OpenRouter stays off."""
    return DEFAULT_PROVIDER, model or DEFAULT_MODEL


def _cooldown_path(cache_dir: Path) -> Path:
    return cache_dir / "model_cooldown.json"


def _parse_moment(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def shared_retry_at(cache_dir: Path, provider: str, model: str) -> datetime | None:
    """The provider's saved deadline. A missing deadline is not a guessed wait."""
    path = _cooldown_path(cache_dir)
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    item = saved.get(f"{provider}/{model}") if isinstance(saved, dict) else None
    if not isinstance(item, dict):
        return None
    return _parse_moment(item.get("retry_at"))


def remember_shared_retry(cache_dir: Path, provider: str, model: str, retry_at: str | None) -> datetime | None:
    """Keep the later deadline for this provider and model. Page bodies are not touched."""
    candidate = _parse_moment(retry_at)
    with _COOLDOWN_LOCK:
        current = shared_retry_at(cache_dir, provider, model)
        chosen = candidate
        if current is not None and (chosen is None or current > chosen):
            chosen = current
        if chosen is None:
            return None
        if current is None or chosen > current:
            path = _cooldown_path(cache_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            saved: dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        saved = loaded
                except json.JSONDecodeError:
                    saved = {}
            saved[f"{provider}/{model}"] = {"provider": provider, "model": model, "retry_at": chosen.isoformat()}
            path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if chosen <= datetime.now(timezone.utc):
            return None
        return chosen


def _remember_retry(record: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Keep the server's Retry-After as an absolute time. A missing time stays missing."""
    if record.get("judge_status") != "rate_limited":
        return
    seconds = verdict.get("retry_after_s")
    if isinstance(seconds, int | float):
        record["retry_after_s"] = float(seconds)
        record["retry_at"] = (datetime.now(timezone.utc) + timedelta(seconds=float(seconds))).isoformat()
        return
    record["retry_after_s"] = None
    record["retry_at"] = None


def _report(progress: Callable[[str, dict[str, Any]], None] | None, stage: str, record: dict[str, Any]) -> None:
    """Tell a caller which stage is current. The record is the pipeline's own object."""
    if progress is not None:
        progress(stage, record)


def _note(record: dict[str, Any], text: str) -> None:
    if text and text not in record["limitations"]:
        record["limitations"].append(text)


def _payload(claims: list[dict[str, Any]], *, allow_paid: bool, allow_openrouter_fallback: bool) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "cache_version": CACHE_VERSION,
        "budgets": {"max_queries": MAX_QUERIES, "max_candidates": MAX_CANDIDATES, "max_pages": MAX_PAGES},
        "paid_fallback": allow_paid,
        "openrouter_fallback": allow_openrouter_fallback,
        "claims": claims,
        "evaluation": _evaluate(claims),
    }


def _checkpoint(path: Path | None, claims: list[dict[str, Any]], *, allow_paid: bool, allow_openrouter_fallback: bool) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_payload(claims, allow_paid=allow_paid, allow_openrouter_fallback=allow_openrouter_fallback), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )



def _evaluate(claims: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for claim in claims:
        verdict = claim.get("verdict")
        expected = claim.get("human_label")
        completed = claim.get("judge_status") == "completed" and verdict in VERDICTS
        judge_correct = verdict == expected if completed and expected else None
        key_evidence = _key_evidence_obtained(claim)
        claim["key_evidence_obtained"] = key_evidence
        rows.append(
            {
                "id": claim.get("id"),
                "search_status": claim.get("search_status"),
                "fetch_status": claim.get("fetch_status"),
                "evidence_found": claim.get("evidence_found"),
                "judge_status": claim.get("judge_status"),
                "verdict": verdict,
                "human_label": expected,
                "judge_correct": judge_correct,
                "citation_passed": claim.get("citation_passed"),
                "key_evidence_obtained": claim.get("key_evidence_obtained"),
                "end_to_end_match": verdict == expected if expected and verdict in VERDICTS else False if expected else None,
            }
        )
    labeled = [row for row in rows if row["judge_correct"] is not None]
    correct = sum(1 for row in labeled if row["judge_correct"])
    return {
        "search_completed": sum(1 for claim in claims if claim.get("search_status") in SEARCH_DONE),
        "evidence_acquired": sum(1 for claim in claims if claim.get("evidence_found") is True),
        "judge_completed": sum(1 for row in rows if row["judge_status"] == "completed" and row["verdict"] in VERDICTS),
        "judge_accuracy": f"{correct}/{len(labeled)}" if labeled else None,
        "end_to_end_matches": sum(1 for row in rows if row.get("end_to_end_match") is True),
        "end_to_end_rate": f"{sum(1 for row in rows if row.get('end_to_end_match') is True)}/{len(rows)}" if rows else None,
        "key_evidence_obtained": sum(1 for row in rows if row.get("key_evidence_obtained") is True),
        "claims": rows,
    }


def _public_http(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    return host not in {"localhost", "127.0.0.1", "::1"}


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _url_key(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return f"{host}{path}"


def _print_run(result: dict[str, Any]) -> None:
    for claim in result["claims"]:
        print(
            f"\n{claim['id']}  search={claim.get('search_status')}  fetch={claim.get('fetch_status')}  "
            f"judge={claim.get('judge_status')}  verdict={claim.get('verdict')}"
        )
        print(f"  claim: {claim.get('claim')}")
        print(f"  queries: {[item.get('query') if isinstance(item, dict) else item for item in claim.get('queries') or []]}")
        print(f"  candidates: {len(claim.get('candidates') or [])}")
        for item in claim.get("candidates") or []:
            print(f"    {item.get('url')}  {item.get('title')}")
        for fetched in claim.get("fetches") or []:
            passage = fetched.get("passage") or {}
            span = (
                f"{passage.get('source_id')} [{passage.get('start')}:{passage.get('end')}] {passage.get('url')}"
                if passage
                else "none"
            )
            print(
                f"  fetch {fetched.get('fetch_status')}  cache={fetched.get('page_cache')}  "
                f"passage={span}  {fetched.get('final_url') or fetched.get('url')}"
            )
            if passage.get("quote"):
                print(f"    {passage['quote'][:240]}")
        print(
            f"  evidence_found={claim.get('evidence_found')}  "
            f"citation_passed={claim.get('citation_passed')}"
        )
        if claim.get("explanation"):
            print(f"  explanation: {claim['explanation']}")
        for limit in claim.get("limitations") or []:
            print(f"  limit: {limit}")
        if claim.get("human_label"):
            print(f"  human_label: {claim['human_label']}")
    evaluation = result["evaluation"]
    print(
        f"\nsearch_completed={evaluation['search_completed']}  "
        f"evidence_acquired={evaluation['evidence_acquired']}  "
        f"judge_completed={evaluation['judge_completed']}  "
        f"judge_accuracy={evaluation['judge_accuracy']}  "
        f"end_to_end={evaluation['end_to_end_rate']}  "
        f"key_evidence_obtained={evaluation['key_evidence_obtained']}"
    )
    for row in evaluation["claims"]:
        print(
            f"{row['id']}  evidence_found={row['evidence_found']}  "
            f"judge_correct={row['judge_correct']}  citation_passed={row['citation_passed']}  "
            f"judge_status={row['judge_status']}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search for pages and judge the fetched text.")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--judge", choices=("none", "fixed_rules", "llm"), default="llm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-openrouter-fallback", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--only", action="append", default=None, help="Resume only these claim ids.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    data = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    previous = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
    result = investigate(
        cases,
        judge=args.judge,
        model=args.model,
        allow_openrouter_fallback=not args.no_openrouter_fallback,
        allow_paid=args.allow_paid,
        previous=previous,
        checkpoint=args.out,
        only=set(args.only) if args.only else None,
    )
    _print_run(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
