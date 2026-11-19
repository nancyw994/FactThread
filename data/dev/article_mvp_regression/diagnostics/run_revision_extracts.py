"""Sequential extraction-only runner with cache identity and pause-on-rate-limit."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]

import articles
import extract_claims
import llm_client


ORDER = ["chan_chan", "heatwave", "lucy", "dengue"]
MAX_503_RETRIES = 2
GEMINI_MODEL = llm_client.GEMINI_FREE_MODEL
GROQ_MODEL = llm_client.GROQ_MODEL

llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
model = articles.ARTICLE_MODEL
provider_chain = [
    {key: item[key] for key in ("provider", "model", "base_url", "key_name", "route_reason")}
    for item in llm_client.provider_chain(model)
]
expected_routes = {
    GROQ_MODEL: [{"provider": "groq", "model": GROQ_MODEL, "base_url": llm_client.GROQ_BASE_URL,
                  "key_name": "GROQ_API_KEY", "route_reason": "groq_primary"}],
    GEMINI_MODEL: [{"provider": "gemini", "model": GEMINI_MODEL, "base_url": llm_client.GEMINI_BASE_URL,
                    "key_name": "GEMINI_API_KEY", "route_reason": "gemini_requested"}],
}
if model not in expected_routes or provider_chain != expected_routes[model]:
    raise RuntimeError(f"Refusing unexpected provider/model route: {provider_chain}")
if llm_client.ALLOW_PAID or llm_client.ALLOW_OPENROUTER_FALLBACK:
    raise RuntimeError("Refusing to run with paid access or provider fallback enabled")
provider = provider_chain[0]["provider"]
key_name = provider_chain[0]["key_name"]
_, key_status = llm_client.lookup_key(key_name)
if key_status != "ok":
    raise RuntimeError(f"{key_name} is {key_status}; no model request was made")

cap = articles.EXTRACTION_MAX_COMPLETION_TOKENS
if cap != 8192:
    raise RuntimeError(f"Expected output cap 8192, found {cap}; no model request was made")
prompt_bytes = articles.PROMPT.read_bytes()
prompt_text = prompt_bytes.decode("utf-8")
prompt_version = "sha256:" + hashlib.sha256(prompt_bytes).hexdigest()
response_format = articles.EXTRACTION_RESPONSE_FORMAT
temperature = 0.0
sources = json.loads((BASE / "articles.json").read_text(encoding="utf-8"))
by_id = {item["id"]: item for item in sources}
cache_dir = BASE / "cache" / "extractions"
cache_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity_for(article_id: str) -> tuple[dict, str]:
    article = by_id[article_id]["text"]
    identity = {
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "output_cap": cap,
        "input_sha256": hashlib.sha256(article.encode("utf-8")).hexdigest(),
    }
    token = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return identity, hashlib.sha256(token).hexdigest()


def cache_hit(article_id: str, identity: dict, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("cache_identity") != identity or cached.get("cache_key") != key:
        raise RuntimeError(f"Cache identity mismatch for {article_id}; refusing reuse")
    if cached.get("status") != "completed" or cached.get("finish_reason") in {"length", "max_tokens"}:
        return None
    cached["cache_reused"] = True
    cached["cache_file"] = str(path.relative_to(BASE))
    cached["record_file"] = str(path.relative_to(BASE))
    return cached


def run_article(article_id: str, run_dir: Path) -> dict:
    source = by_id[article_id]
    article = source["text"]
    if len(article) > articles.MAX_ARTICLE:
        raise RuntimeError(f"{article_id} exceeds the normal article input limit")
    identity, key = identity_for(article_id)
    cached = cache_hit(article_id, identity, key)
    if cached is not None:
        return cached

    user_message = json.dumps({"article": article, "repair_errors": []}, ensure_ascii=False)
    yaml_spec = yaml.safe_dump({"instructions": prompt_text}, sort_keys=False, allow_unicode=True)
    messages = llm_client._messages(yaml_spec, user_message)
    record = {
        "article_id": article_id,
        "source_url": source["url"],
        "input_chars": len(article),
        "input_bytes": len(article.encode("utf-8")),
        "input_sha256": identity["input_sha256"],
        "truncated": False,
        "submitted_article_chars": len(article),
        "provider_chain": provider_chain,
        "allow_paid": llm_client.ALLOW_PAID,
        "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
        "model": model,
        "prompt_version": prompt_version,
        "response_format": response_format,
        "temperature": temperature,
        "output_parameter": "max_completion_tokens",
        "output_cap": cap,
        "cache_identity": identity,
        "cache_key": key,
        "cache_reused": False,
        "request_attempts": [],
        "accepted_claims": [],
        "rejected_claims": [],
    }
    record_path = run_dir / f"{article_id}.json"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_parameters = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": response_format,
        "max_completion_tokens": cap,
    }

    for retry_503 in range(MAX_503_RETRIES + 1):
        attempt = {
            "sequence": retry_503 + 1,
            "request_status": "started",
            "http_status": None,
            "error_type": None,
            "error_code": None,
            "error_body": None,
            "rate_limit_headers": {},
            "request_parameters": request_parameters,
        }
        record["request_attempts"].append(attempt)
        write_json(record_path, record)
        started = time.perf_counter()
        try:
            response = llm_client.ask_llm_usage(
                {"instructions": prompt_text},
                user_message,
                model=model,
                temperature=temperature,
                response_format=response_format,
                max_completion_tokens=cap,
            )
        except Exception as exc:
            provider_attempts = getattr(exc, "attempts", []) or []
            provider_attempt = provider_attempts[-1] if provider_attempts else {}
            status = provider_attempt.get("http_status")
            attempt.update({
                "request_status": "rate_limited" if status == 429 or llm_client.is_rate_limit(exc) else "request_failed",
                "http_status": status,
                "error_type": provider_attempt.get("exception_type") or type(exc.__cause__ or exc).__name__,
                "error_code": provider_attempt.get("error_code"),
                "error_body": provider_attempt.get("error_body"),
                "rate_limit_headers": provider_attempt.get("rate_limit_headers") or {},
                "provider_attempts": provider_attempts,
                "error": str(llm_client._redact_provider_value(str(exc)))[:1200],
                "elapsed_seconds": time.perf_counter() - started,
            })
            if status == 503 and retry_503 < MAX_503_RETRIES:
                headers = attempt["rate_limit_headers"]
                try:
                    wait_seconds = float(headers.get("retry-after", 2 ** (retry_503 + 1)))
                except (TypeError, ValueError):
                    wait_seconds = float(2 ** (retry_503 + 1))
                wait_seconds = min(max(wait_seconds, 1.0), 15.0)
                attempt["status"] = "temporary_503_waiting"
                attempt["retry_after_seconds"] = wait_seconds
                write_json(record_path, record)
                time.sleep(wait_seconds)
                continue
            record["status"] = "paused_rate_limited" if status == 429 else "request_failed"
            write_json(record_path, record)
            return record

        raw = response.get("content") or ""
        raw_path = raw_dir / f"{article_id}-attempt-{retry_503 + 1}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        attempt.update({
            "request_status": "completed",
            "http_status": response.get("http_status"),
            "served_model": response.get("model"),
            "finish_reason": response.get("finish_reason"),
            "usage": response.get("usage"),
            "rate_limit_headers": response.get("rate_limit_headers") or {},
            "cost_usd": response.get("cost_usd", "unknown"),
            "raw_reply_chars": len(raw),
            "raw_reply_file": str(raw_path.relative_to(BASE)),
            "elapsed_seconds": response.get("elapsed_s"),
        })
        record["http_status"] = response.get("http_status")
        record["finish_reason"] = response.get("finish_reason")
        record["usage"] = response.get("usage")
        record["rate_limit_headers"] = response.get("rate_limit_headers") or {}
        record["raw_reply_file"] = str(raw_path.relative_to(BASE))

        try:
            parsed = articles._parse_extract(raw)
        except Exception as exc:
            attempt["parse_status"] = "parse_failed"
            attempt["parse_error"] = str(exc)
            record["status"] = "truncated" if response.get("finish_reason") in {"length", "max_tokens"} else "parse_failed"
            write_json(record_path, record)
            return record

        accepted, rejected = extract_claims._split(article, parsed)
        record["parsed_items"] = len(parsed.get("claims") or [])
        record["accepted_claims"] = accepted
        record["rejected_claims"] = rejected
        attempt.update({
            "parse_status": "parsed",
            "parsed_items": record["parsed_items"],
            "accepted_items": len(accepted),
            "rejected_items": len(rejected),
            "rejection_reasons": [error for item in rejected for error in item["errors"]],
        })
        truncated = response.get("finish_reason") in {"length", "max_tokens"}
        record["truncated"] = truncated
        record["status"] = "truncated" if truncated else "completed"
        record["accepted_items"] = len(accepted)
        record["rejected_items"] = len(rejected)
        write_json(record_path, record)

        if not truncated:
            cache_file = cache_dir / f"{key}.json"
            write_json(cache_file, record)
            record["cache_file"] = str(cache_file.relative_to(BASE))
            write_json(record_path, record)
        return record

    record["status"] = "request_failed"
    write_json(record_path, record)
    return record


def run_format_probe(run_dir: Path) -> dict:
    """One short extraction using the same prompt, model, format, and output cap."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_articles import SHORT_ARTICLE

    user_message = json.dumps({"article": SHORT_ARTICLE, "repair_errors": []}, ensure_ascii=False)
    yaml_spec = yaml.safe_dump({"instructions": prompt_text}, sort_keys=False, allow_unicode=True)
    messages = llm_client._messages(yaml_spec, user_message)
    record = {
        "kind": "short_format_probe", "provider_chain": provider_chain, "provider": provider,
        "model": model, "prompt_version": prompt_version, "response_format": response_format,
        "temperature": temperature, "output_parameter": "max_completion_tokens", "output_cap": cap,
        "input_chars": len(SHORT_ARTICLE),
        "input_sha256": hashlib.sha256(SHORT_ARTICLE.encode("utf-8")).hexdigest(),
        "truncated": False, "allow_paid": False, "allow_openrouter_fallback": False,
        "request_attempts": [],
    }
    path = run_dir / "format_probe.json"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_parameters = {"model": model, "temperature": temperature, "messages": messages,
                          "response_format": response_format, "max_completion_tokens": cap}
    for retry_503 in range(MAX_503_RETRIES + 1):
        attempt = {"sequence": retry_503 + 1, "request_status": "started",
                   "request_parameters": request_parameters}
        record["request_attempts"].append(attempt)
        write_json(path, record)
        started = time.perf_counter()
        try:
            response = llm_client.ask_llm_usage(
                {"instructions": prompt_text}, user_message, model=model, temperature=temperature,
                response_format=response_format, max_completion_tokens=cap,
            )
        except Exception as exc:
            attempts = getattr(exc, "attempts", []) or []
            provider_attempt = attempts[-1] if attempts else {}
            status = provider_attempt.get("http_status")
            attempt.update({
                "request_status": "rate_limited" if status == 429 or llm_client.is_rate_limit(exc) else "request_failed",
                "http_status": status,
                "error_type": provider_attempt.get("exception_type") or type(exc).__name__,
                "error_code": provider_attempt.get("error_code"),
                "error_body": provider_attempt.get("error_body"),
                "rate_limit_headers": provider_attempt.get("rate_limit_headers") or {},
                "provider_attempts": attempts,
                "error": str(llm_client._redact_provider_value(str(exc)))[:1200],
                "elapsed_seconds": time.perf_counter() - started,
            })
            if status == 503 and retry_503 < MAX_503_RETRIES:
                headers = attempt["rate_limit_headers"]
                try:
                    wait_seconds = float(headers.get("retry-after", 2 ** (retry_503 + 1)))
                except (TypeError, ValueError):
                    wait_seconds = float(2 ** (retry_503 + 1))
                wait_seconds = min(max(wait_seconds, 1.0), 15.0)
                attempt["retry_after_seconds"] = wait_seconds
                write_json(path, record)
                time.sleep(wait_seconds)
                continue
            record["status"] = "paused_rate_limited" if status == 429 else "request_failed"
            write_json(path, record)
            return record

        raw = response.get("content") or ""
        raw_path = raw_dir / "format_probe.txt"
        raw_path.write_text(raw, encoding="utf-8")
        attempt.update({"request_status": "completed", "http_status": response.get("http_status"),
                        "served_model": response.get("model"), "finish_reason": response.get("finish_reason"),
                        "usage": response.get("usage"),
                        "rate_limit_headers": response.get("rate_limit_headers") or {},
                        "raw_reply_chars": len(raw), "raw_reply_file": str(raw_path.relative_to(BASE)),
                        "elapsed_seconds": response.get("elapsed_s")})
        record.update({"http_status": response.get("http_status"),
                       "finish_reason": response.get("finish_reason"), "usage": response.get("usage"),
                       "raw_reply_file": str(raw_path.relative_to(BASE))})
        try:
            parsed = articles._parse_extract(raw)
        except Exception as exc:
            attempt.update({"parse_status": "parse_failed", "parse_error": str(exc)})
            record["status"] = "truncated" if response.get("finish_reason") in {"length", "max_tokens"} else "parse_failed"
            write_json(path, record)
            return record
        accepted, rejected = extract_claims._split(SHORT_ARTICLE, parsed)
        record.update({"parsed_items": len(parsed.get("claims") or []), "accepted_claims": accepted,
                       "rejected_claims": rejected, "status": "completed"})
        attempt.update({"parse_status": "parsed", "parsed_items": record["parsed_items"],
                        "accepted_items": len(accepted), "rejected_items": len(rejected),
                        "rejection_reasons": [error for item in rejected for error in item["errors"]]})
        if response.get("finish_reason") in {"length", "max_tokens"}:
            record["status"] = "truncated"
        elif not accepted:
            record["status"] = "validation_failed"
        write_json(path, record)
        return record
    return record


def main() -> int:
    args = sys.argv[1:]
    probe_first = "--probe" in args
    requested = [arg for arg in args if arg != "--probe"] or ORDER
    if not requested or any(article_id not in ORDER for article_id in requested):
        raise SystemExit("usage: run_revision_extracts.py [--probe] [chan_chan] [heatwave] [lucy] [dengue]")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = BASE / "after" / "prompt-revision" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "order": requested,
        "provider_chain": provider_chain,
        "provider": provider,
        "model": model,
        "allow_paid": llm_client.ALLOW_PAID,
        "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
        "prompt_version": prompt_version,
        "output_cap": cap,
        "results": [],
        "status": "running",
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    if probe_first:
        probe = run_format_probe(run_dir)
        manifest["format_probe"] = str((run_dir / "format_probe.json").relative_to(BASE))
        manifest["format_probe_status"] = probe.get("status")
        if probe.get("status") != "completed":
            manifest["status"] = "paused_rate_limited" if probe.get("status") == "paused_rate_limited" else "stopped_format_probe_failed"
            manifest["pending"] = requested
            write_json(manifest_path, manifest)
            print(json.dumps({"manifest_file": str(manifest_path.relative_to(BASE)), "status": manifest["status"],
                              "pending": manifest["pending"], "format_probe": probe}, ensure_ascii=False, indent=2))
            return 0
        write_json(manifest_path, manifest)
    for article_id in requested:
        result = run_article(article_id, run_dir)
        manifest["results"].append({
            "article_id": article_id,
            "status": result.get("status"),
            "cache_reused": result.get("cache_reused", False),
            "record_file": result.get("record_file") or str((run_dir / f"{article_id}.json").relative_to(BASE)),
            "input_chars": result.get("input_chars"),
            "input_sha256": result.get("input_sha256"),
            "finish_reason": result.get("finish_reason"),
            "parsed_items": result.get("parsed_items"),
            "accepted_items": result.get("accepted_items"),
            "rejected_items": result.get("rejected_items"),
        })
        write_json(manifest_path, manifest)
        if result.get("status") == "paused_rate_limited":
            manifest["status"] = "paused_rate_limited"
            manifest["pending"] = requested[len(manifest["results"]):]
            write_json(manifest_path, manifest)
            break
        if result.get("status") == "request_failed":
            manifest["status"] = "stopped_request_failed"
            manifest["pending"] = requested[len(manifest["results"]):]
            write_json(manifest_path, manifest)
            break
    else:
        manifest["status"] = "completed"
        manifest["pending"] = []
        write_json(manifest_path, manifest)

    print(json.dumps({
        "manifest_file": str(manifest_path.relative_to(BASE)),
        "status": manifest["status"],
        "pending": manifest.get("pending"),
        "results": manifest["results"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
