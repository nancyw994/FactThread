"""Single-provider, extraction-only Chan Chan run with a content-addressed record."""

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


llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
model = articles.ARTICLE_MODEL
provider_chain = [
    {key: item[key] for key in ("provider", "model", "base_url", "key_name", "route_reason")}
    for item in llm_client.provider_chain(model)
]
expected_chain = [{
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "base_url": llm_client.GROQ_BASE_URL,
    "key_name": "GROQ_API_KEY",
    "route_reason": "groq_primary",
}]
if provider_chain != expected_chain:
    raise RuntimeError(f"Refusing unexpected provider/model route: {provider_chain}")
if llm_client.ALLOW_PAID or llm_client.ALLOW_OPENROUTER_FALLBACK:
    raise RuntimeError("Refusing to run with paid access or OpenRouter enabled")
key, key_status = llm_client.lookup_key("GROQ_API_KEY")
if key_status != "ok":
    raise RuntimeError(f"GROQ_API_KEY is {key_status}; no model request was made")
del key

source = next(item for item in json.loads((BASE / "articles.json").read_text(encoding="utf-8"))
              if item["id"] == "chan_chan")
article = source["text"]
article_hash = hashlib.sha256(article.encode("utf-8")).hexdigest()
prompt_bytes = articles.PROMPT.read_bytes()
prompt_text = prompt_bytes.decode("utf-8")
prompt_version = "sha256:" + hashlib.sha256(prompt_bytes).hexdigest()
initial_cap = articles.EXTRACTION_MAX_COMPLETION_TOKENS
if initial_cap != 8192:
    raise RuntimeError(f"Expected configured output cap 8192, found {initial_cap}; no model request was made")
if len(article) > articles.MAX_ARTICLE:
    raise RuntimeError("Chan Chan source exceeds the article-mode input limit; no model request was made")

user_message = json.dumps({"article": article, "repair_errors": []}, ensure_ascii=False)
yaml_spec = yaml.safe_dump({"instructions": prompt_text}, sort_keys=False, allow_unicode=True)
messages = llm_client._messages(yaml_spec, user_message)
response_format = articles.EXTRACTION_RESPONSE_FORMAT
temperature = 0.0
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
output_dir = BASE / "after" / "chan-full-extract" / run_id
raw_dir = output_dir / "raw"
raw_dir.mkdir(parents=True, exist_ok=False)
output_file = output_dir / "result.json"
record = {
    "run_id": run_id,
    "run_type": "single_article_extraction_only",
    "article_id": source["id"],
    "source_url": source["url"],
    "input_chars": len(article),
    "input_bytes": len(article.encode("utf-8")),
    "input_sha256": article_hash,
    "truncated": False,
    "submitted_article_chars": len(article),
    "prompt_version": prompt_version,
    "prompt_path": str(articles.PROMPT.relative_to(ROOT)),
    "provider_chain": provider_chain,
    "allow_paid": llm_client.ALLOW_PAID,
    "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
    "response_format": response_format,
    "temperature": temperature,
    "output_parameter": "max_completion_tokens",
    "cache_reused": False,
    "attempts": [],
    "accepted_claims": [],
    "rejected_claims": [],
    "output_file": str(output_file.relative_to(BASE)),
}


def cache_key(output_cap: int) -> str:
    identity = {
        "provider": "groq",
        "model": model,
        "prompt_version": prompt_version,
        "output_cap": output_cap,
        "input_sha256": article_hash,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save() -> None:
    output_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitized_error(exc: BaseException) -> str:
    return str(llm_client._redact_provider_value(str(exc)))[:1200]


def run_cap(output_cap: int) -> tuple[bool, dict | None]:
    key_for_cap = cache_key(output_cap)
    record.setdefault("cache_keys", []).append({"output_cap": output_cap, "key": key_for_cap})
    record["active_cache_key"] = key_for_cap
    record["active_output_cap"] = output_cap
    retry_503 = 0
    while True:
        attempt_number = len(record["attempts"]) + 1
        request_parameters = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "response_format": response_format,
            "max_completion_tokens": output_cap,
        }
        attempt = {
            "sequence": attempt_number,
            "provider": "groq",
            "model": model,
            "output_cap": output_cap,
            "cache_key": key_for_cap,
            "request_status": "started",
            "http_status": None,
            "error_type": None,
            "error_code": None,
            "error_body": None,
            "rate_limit_headers": {},
            "finish_reason": None,
            "usage": None,
            "request_parameters": request_parameters,
            "parse_status": "not_run",
            "parsed_items": None,
            "accepted_items": None,
            "rejected_items": None,
            "rejection_reasons": [],
        }
        record["attempts"].append(attempt)
        save()
        started = time.perf_counter()
        try:
            result = llm_client.ask_llm_usage(
                {"instructions": prompt_text},
                user_message,
                model=model,
                temperature=temperature,
                response_format=response_format,
                max_completion_tokens=output_cap,
            )
        except Exception as exc:
            provider_attempts = getattr(exc, "attempts", []) or []
            provider_attempt = provider_attempts[-1] if provider_attempts else {}
            attempt.update({
                "request_status": "rate_limited" if llm_client.is_rate_limit(exc) else "request_failed",
                "error": sanitized_error(exc),
                "error_type": provider_attempt.get("exception_type") or type(exc.__cause__ or exc).__name__,
                "error_code": provider_attempt.get("error_code"),
                "http_status": provider_attempt.get("http_status"),
                "error_body": provider_attempt.get("error_body"),
                "rate_limit_headers": provider_attempt.get("rate_limit_headers") or {},
                "provider_attempts": provider_attempts,
                "elapsed_seconds": time.perf_counter() - started,
            })
            status = attempt["http_status"]
            if status == 429:
                attempt["request_status"] = "rate_limited"
                record["stopped_reason"] = "http_429_no_retry"
                save()
                return False, None
            if status == 400:
                record["stopped_reason"] = "http_400_no_budget_retry"
                save()
                return False, None
            if status == 503 and retry_503 < 2:
                headers = attempt["rate_limit_headers"]
                delay = min(max(float(headers.get("retry-after", 2 ** (retry_503 + 1))), 1.0), 15.0)
                retry_503 += 1
                attempt["retry_503_after_seconds"] = delay
                attempt["request_status"] = "temporary_503_waiting"
                save()
                time.sleep(delay)
                continue
            record["stopped_reason"] = "http_503_retries_exhausted" if status == 503 else "request_failed"
            save()
            return False, None

        raw = result.get("content") or ""
        raw_path = raw_dir / f"reply-{attempt_number:02d}-cap-{output_cap}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        attempt.update({
            "request_status": "completed",
            "http_status": result.get("http_status"),
            "served_model": result.get("model"),
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage"),
            "rate_limit_headers": result.get("rate_limit_headers") or {},
            "cost_usd": result.get("cost_usd", "unknown"),
            "elapsed_seconds": result.get("elapsed_s"),
            "raw_reply_chars": len(raw),
            "raw_reply_file": str(raw_path.relative_to(BASE)),
        })
        try:
            parsed = articles._parse_extract(raw)
        except Exception as exc:
            attempt["parse_status"] = "parse_failed"
            attempt["parse_error"] = sanitized_error(exc)
            attempt["parsed_items"] = None
            if attempt.get("finish_reason") not in {"length", "max_tokens"}:
                record["stopped_reason"] = "parse_failed"
                save()
                return False, None
            record["stopped_reason"] = "truncated_before_parse"
            save()
            return True, {"raw": raw, "parsed": None, "accepted": [], "rejected": [], "attempt": attempt}

        claims = parsed.get("claims") or []
        accepted, rejected = extract_claims._split(article, parsed)
        attempt.update({
            "parse_status": "parsed",
            "parsed_items": len(claims),
            "accepted_items": len(accepted),
            "rejected_items": len(rejected),
            "rejection_reasons": [error for item in rejected for error in item["errors"]],
        })
        record["accepted_claims"] = accepted
        record["rejected_claims"] = rejected
        record["stopped_reason"] = None
        save()
        return True, {"raw": raw, "parsed": parsed, "accepted": accepted, "rejected": rejected, "attempt": attempt}


for cap in (initial_cap,):
    completed, outcome = run_cap(cap)
    if not completed or outcome is None:
        break
    if outcome["attempt"].get("finish_reason") not in {"length", "max_tokens"}:
        record["final_status"] = "completed"
        break

    headers = outcome["attempt"].get("rate_limit_headers") or {}
    remaining_raw = headers.get("x-ratelimit-remaining-tokens")
    prompt_tokens = (outcome["attempt"].get("usage") or {}).get("prompt_tokens")
    try:
        remaining = int(remaining_raw)
        prompt_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        record["length_retry_decision"] = {
            "eligible": False,
            "reason": "provider did not return enough remaining-token quota data to confirm a 16384 retry",
            "remaining_tokens": remaining_raw,
        }
        record["final_status"] = "truncated_quota_unknown"
        break
    required = 16384 + prompt_tokens
    if remaining < required:
        record["length_retry_decision"] = {
            "eligible": False,
            "reason": "remaining token quota is below the next request cap plus prompt tokens",
            "remaining_tokens": remaining,
            "required_minimum": required,
        }
        record["final_status"] = "truncated_quota_insufficient"
        break
    record["length_retry_decision"] = {
        "eligible": True,
        "remaining_tokens": remaining,
        "required_minimum": required,
    }
    save()
    completed, outcome = run_cap(16384)
    record["final_status"] = "completed" if completed else "retry_failed"
    break

if "final_status" not in record:
    last = record["attempts"][-1] if record["attempts"] else {}
    record["final_status"] = last.get("request_status", "not_run")
save()
print(json.dumps({
    "record_file": str(output_file.relative_to(BASE)),
    "input_chars": record["input_chars"],
    "input_sha256": article_hash,
    "provider_chain": provider_chain,
    "allow_paid": llm_client.ALLOW_PAID,
    "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
    "prompt_version": prompt_version,
    "caps_attempted": [item["output_cap"] for item in record["attempts"]],
    "final_status": record["final_status"],
    "attempts": [{key: item.get(key) for key in ("sequence", "request_status", "http_status", "error_type", "error_code", "finish_reason", "usage", "parse_status", "parsed_items", "accepted_items", "rejected_items", "retry_503_after_seconds", "raw_reply_file")} for item in record["attempts"]],
    "accepted_count": len(record["accepted_claims"]),
    "rejected_count": len(record["rejected_claims"]),
    "rejection_reasons": [error for item in record["rejected_claims"] for error in item.get("errors", [])],
    "stopped_reason": record.get("stopped_reason"),
}, ensure_ascii=False, indent=2))
