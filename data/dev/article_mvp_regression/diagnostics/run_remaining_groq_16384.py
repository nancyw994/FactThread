"""Sequential extraction-only run for heatwave, Lucy, and dengue at frozen config."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import certifi
import httpcore2
import ssl
import yaml

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]

import articles
import extract_claims
import llm_client

ORDER = ["heatwave", "lucy", "dengue"]
MODEL = "openai/gpt-oss-120b"
CAP = 16384
PROMPT_SHA = "e52cb13057888cc38bba9f73e2e6ac7fd229593d848a6b16829203269ade1f77"
OUTPUT_ROOT = BASE / "after" / "remaining-16384"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_frozen_config() -> tuple[dict, str, str]:
    if articles.ARTICLE_MODEL != MODEL or articles.EXTRACTION_MAX_COMPLETION_TOKENS != CAP:
        raise RuntimeError("Refusing run: model or output cap differs from frozen config")
    prompt_bytes = articles.PROMPT.read_bytes()
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    if prompt_sha != PROMPT_SHA:
        raise RuntimeError("Refusing run: revised prompt hash changed")
    if not os.environ.get("SSL_CERT_FILE") or Path(os.environ["SSL_CERT_FILE"]).resolve() != Path(certifi.where()).resolve():
        raise RuntimeError("Refusing run: activated venv is not using the repaired certifi CA bundle")
    context = httpcore2.default_ssl_context(_trust_env=True)
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError("Refusing run: TLS verification is not enabled")

    llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
    chain = llm_client.provider_chain(MODEL)
    if len(chain) != 1 or chain[0].get("provider") != "groq" or chain[0].get("model") != MODEL:
        raise RuntimeError(f"Refusing unexpected provider chain: {chain}")
    if llm_client.ALLOW_OPENROUTER_FALLBACK or llm_client.ALLOW_PAID:
        raise RuntimeError("Refusing run: OpenRouter or paid access is enabled")
    _, key_status = llm_client.lookup_key("GROQ_API_KEY")
    if key_status != "ok":
        raise RuntimeError(f"GROQ_API_KEY is {key_status}; no model request was made")
    route = {key: chain[0][key] for key in ("provider", "model", "base_url", "route_reason")}
    return route, prompt_bytes.decode("utf-8"), "sha256:" + prompt_sha


def load_articles() -> dict:
    values = json.loads((BASE / "articles.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in values}


def article_identity(source: dict, prompt_version: str) -> dict:
    text = source["text"]
    return {
        "provider": "groq",
        "model": MODEL,
        "prompt_version": prompt_version,
        "output_cap": CAP,
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "input_chars": len(text),
    }


def perform_one(article_id: str, source: dict, run_dir: Path, route: dict,
                prompt_text: str, prompt_version: str, record: dict | None = None) -> dict:
    text = source["text"]
    identity = article_identity(source, prompt_version)
    if identity["input_chars"] > articles.MAX_ARTICLE:
        raise RuntimeError(f"{article_id} exceeds the article input limit")
    user_message = json.dumps({"article": text, "repair_errors": []}, ensure_ascii=False)
    yaml_spec = yaml.safe_dump({"instructions": prompt_text}, sort_keys=False, allow_unicode=True)
    messages = llm_client._messages(yaml_spec, user_message)
    if record is None:
        record = {
            "article_id": article_id,
            "source_url": source["url"],
            **identity,
            "truncated_input": False,
            "submitted_article_chars": len(text),
            "provider_chain": [route],
            "allow_paid": False,
            "allow_openrouter_fallback": False,
            "response_format": articles.EXTRACTION_RESPONSE_FORMAT,
            "temperature": 0.0,
            "output_parameter": "max_completion_tokens",
            "request_count": 0,
            "request_attempts": [],
            "accepted_claims": [],
            "rejected_claims": [],
            "cache_reused": False,
        }
    record["status"] = "request_started"
    record["request_count"] += 1
    attempt = {
        "sequence": record["request_count"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "request_parameters": {
            "model": MODEL,
            "temperature": 0.0,
            "messages": messages,
            "response_format": articles.EXTRACTION_RESPONSE_FORMAT,
            "max_completion_tokens": CAP,
        },
    }
    record["request_attempts"].append(attempt)
    result_path = run_dir / f"{article_id}.json"
    write_json(result_path, record)

    try:
        response = llm_client.ask_llm_usage(
            {"instructions": prompt_text}, user_message,
            model=MODEL,
            temperature=0.0,
            response_format=articles.EXTRACTION_RESPONSE_FORMAT,
            max_completion_tokens=CAP,
        )
    except Exception as exc:
        provider_attempts = getattr(exc, "attempts", []) or []
        last = provider_attempts[-1] if provider_attempts else {}
        status = last.get("http_status")
        headers = last.get("rate_limit_headers") or {}
        attempt.update({
            "request_status": "rate_limited" if status == 429 or llm_client.is_rate_limit(exc) else "request_failed",
            "http_status": status,
            "error_type": last.get("exception_type") or type(exc).__name__,
            "error_code": last.get("error_code"),
            "error_body": last.get("error_body"),
            "rate_limit_headers": headers,
            "provider_attempts": provider_attempts,
            "error": str(llm_client._redact_provider_value(str(exc)))[:1800],
        })
        record.update({
            "status": "paused_rate_limited" if status == 429 or llm_client.is_rate_limit(exc) else "request_failed",
            "http_status": status,
            "rate_limit_headers": headers,
            "error_type": attempt["error_type"],
            "error_code": attempt["error_code"],
            "error_body": attempt["error_body"],
            "error": attempt["error"],
        })
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
                record["retry_after_seconds"] = seconds
                record["resume_not_before_utc"] = (
                    datetime.now(timezone.utc) + timedelta(seconds=seconds)
                ).isoformat()
            except (TypeError, ValueError):
                record["retry_after_seconds"] = retry_after
        write_json(result_path, record)
        return record

    raw = response.get("content") or ""
    raw_dir = run_dir / "raw"
    raw_path = raw_dir / f"{article_id}-attempt-{record['request_count']}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw, encoding="utf-8")
    attempt.update({
        "request_status": "completed",
        "http_status": response.get("http_status"),
        "served_model": response.get("model"),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "rate_limit_headers": response.get("rate_limit_headers") or {},
        "cost_usd": response.get("cost_usd", "unknown"),
        "elapsed_seconds": response.get("elapsed_s"),
        "raw_reply_chars": len(raw),
        "raw_reply_file": str(raw_path.relative_to(BASE)),
    })
    record.update({
        "http_status": response.get("http_status"),
        "served_model": response.get("model"),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "rate_limit_headers": response.get("rate_limit_headers") or {},
        "cost_usd": response.get("cost_usd", "unknown"),
        "elapsed_seconds": response.get("elapsed_s"),
        "raw_reply_chars": len(raw),
        "raw_reply_file": str(raw_path.relative_to(BASE)),
    })
    try:
        parsed = articles._parse_extract(raw)
    except Exception as exc:
        attempt.update({"parse_status": "parse_failed", "parse_error": str(exc)})
        record.update({"status": "truncated" if response.get("finish_reason") in {"length", "max_tokens"} else "parse_failed",
                       "parsed_items": None, "accepted_items": 0, "rejected_items": 0,
                       "accepted_claims": [], "rejected_claims": [], "parse_error": str(exc)})
        write_json(result_path, record)
        return record

    accepted, rejected = extract_claims._split(text, parsed)
    truncated = response.get("finish_reason") in {"length", "max_tokens"}
    record.update({
        "status": "truncated" if truncated else "completed",
        "parsed_items": len(parsed.get("claims") or []),
        "accepted_items": len(accepted),
        "rejected_items": len(rejected),
        "accepted_claims": accepted,
        "rejected_claims": rejected,
        "parse_method": "articles._parse_extract; extract_claims._split exact-source-quote validation",
        "rejection_reasons": [error for item in rejected for error in item.get("errors", [])],
    })
    attempt.update({
        "parse_status": "parsed",
        "parsed_items": record["parsed_items"],
        "accepted_items": record["accepted_items"],
        "rejected_items": record["rejected_items"],
        "rejection_reasons": record["rejection_reasons"],
    })
    write_json(result_path, record)
    return record


def aggregate(manifest: dict) -> None:
    all_results = manifest["results"]
    completed = [x for x in all_results if x.get("status") == "completed"]
    manifest["completed_articles"] = len(completed)
    manifest["completion_rate"] = {"completed": len(completed), "total": len(ORDER)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(x.get("usage") or {}).get(key) for x in all_results]
        usable = [v for v in values if isinstance(v, (int, float))]
        manifest.setdefault("aggregate_usage", {})[key] = sum(usable) if usable else None
    manifest["aggregate_elapsed_seconds"] = sum(x.get("elapsed_seconds", 0) or 0 for x in all_results)


def main() -> int:
    args = sys.argv[1:]
    route, prompt_text, prompt_version = validate_frozen_config()
    sources = load_articles()
    if args:
        if len(args) != 2 or args[0] != "--resume":
            raise SystemExit("usage: run_remaining_groq_16384.py [--resume RUN_ID]")
        run_id = args[1]
        run_dir = OUTPUT_ROOT / run_id
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"paused_rate_limited", "partial", "stopped_request_failed"}:
            raise RuntimeError(f"Run is not resumable: {manifest.get('status')}")
        if manifest.get("model") != MODEL or manifest.get("prompt_version") != prompt_version or manifest.get("output_cap") != CAP:
            raise RuntimeError("Frozen configuration changed; refusing resume")
        for record in manifest["results"]:
            current = article_identity(sources[record["article_id"]], prompt_version)
            if record.get("input_sha256") != current["input_sha256"]:
                raise RuntimeError(f"Source input changed for {record['article_id']}; refusing resume")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "manifest.json"
        manifest = {
            "run_id": run_id,
            "order": ORDER,
            "provider": route,
            "model": MODEL,
            "prompt_version": prompt_version,
            "output_cap": CAP,
            "allow_paid": False,
            "allow_openrouter_fallback": False,
            "requests_per_article_limit": 1,
            "no_search_or_judge": True,
            "status": "running",
            "results": [],
            "pending": list(ORDER),
        }
        write_json(manifest_path, manifest)

    for article_id in ORDER:
        existing = next((r for r in manifest["results"] if r.get("article_id") == article_id), None)
        if existing and existing.get("status") == "completed":
            continue
        if existing and existing.get("request_count", 0) >= 1 and existing.get("status") not in {"paused_rate_limited", "request_failed", "parse_failed", "truncated"}:
            continue
        record = perform_one(article_id, sources[article_id], run_dir, route, prompt_text, prompt_version, existing)
        if existing is None:
            manifest["results"].append(record)
        done_index = ORDER.index(article_id)
        manifest["pending"] = [item for item in ORDER if not any(
            r.get("article_id") == item and r.get("status") == "completed" for r in manifest["results"]
        )]
        if record.get("status") == "paused_rate_limited":
            manifest["status"] = "paused_rate_limited"
            manifest["paused_article"] = article_id
            manifest["pending"] = ORDER[done_index:]
            aggregate(manifest)
            write_json(manifest_path, manifest)
            break
        aggregate(manifest)
        write_json(manifest_path, manifest)
    else:
        manifest["status"] = "completed" if len([r for r in manifest["results"] if r.get("status") == "completed"]) == len(ORDER) else "partial"
        manifest["pending"] = [item for item in ORDER if not any(
            r.get("article_id") == item and r.get("status") == "completed" for r in manifest["results"]
        )]
        aggregate(manifest)
        write_json(manifest_path, manifest)

    print(json.dumps({
        "manifest_file": str(manifest_path.relative_to(BASE)),
        "status": manifest["status"],
        "completed_articles": manifest.get("completed_articles"),
        "pending": manifest.get("pending"),
        "aggregate_usage": manifest.get("aggregate_usage"),
        "aggregate_elapsed_seconds": manifest.get("aggregate_elapsed_seconds"),
        "results": [{"article_id": r.get("article_id"), "status": r.get("status"),
                     "finish_reason": r.get("finish_reason"), "accepted_items": r.get("accepted_items"),
                     "rejected_items": r.get("rejected_items"), "record_file": f"{run_dir.name}/{r.get('article_id')}.json"}
                    for r in manifest["results"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
