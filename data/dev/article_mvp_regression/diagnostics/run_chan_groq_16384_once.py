"""Single-request Chan Chan full-text extraction at 16,384 completion tokens."""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path

import certifi
import httpcore2
import yaml

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]

import articles
import extract_claims
import llm_client

MODEL = "openai/gpt-oss-120b"
CAP = 16384
EXPECTED_PROMPT_SHA = "e52cb13057888cc38bba9f73e2e6ac7fd229593d848a6b16829203269ade1f77"
EXPECTED_INPUT_SHA = "785df4792b277555c254ff0a8f56f67db6b88c5d9b85f0274aaa65ea4d81034c"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if articles.ARTICLE_MODEL != MODEL or articles.EXTRACTION_MAX_COMPLETION_TOKENS != CAP:
        raise RuntimeError("Refusing unexpected model or completion-token cap")
    if not os.environ.get("SSL_CERT_FILE") or Path(os.environ["SSL_CERT_FILE"]).resolve() != Path(certifi.where()).resolve():
        raise RuntimeError("Refusing request: activated venv is not using its certifi CA bundle")
    ssl_context = httpcore2.default_ssl_context(_trust_env=True)
    if ssl_context.verify_mode != ssl.CERT_REQUIRED or not ssl_context.check_hostname:
        raise RuntimeError("Refusing request: TLS verification is not fully enabled")

    llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
    chain = llm_client.provider_chain(MODEL)
    if len(chain) != 1 or chain[0].get("provider") != "groq" or chain[0].get("model") != MODEL:
        raise RuntimeError(f"Refusing unexpected provider route: {chain}")
    if llm_client.ALLOW_OPENROUTER_FALLBACK or llm_client.ALLOW_PAID:
        raise RuntimeError("Refusing request with provider fallback or paid routes enabled")
    _, key_status = llm_client.lookup_key("GROQ_API_KEY")
    if key_status != "ok":
        raise RuntimeError(f"GROQ_API_KEY is {key_status}; no extraction request was made")

    source = next(item for item in json.loads((BASE / "articles.json").read_text(encoding="utf-8"))
                  if item["id"] == "chan_chan")
    article = source["text"]
    article_sha = hashlib.sha256(article.encode("utf-8")).hexdigest()
    prompt_bytes = articles.PROMPT.read_bytes()
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    if article_sha != EXPECTED_INPUT_SHA or prompt_sha != EXPECTED_PROMPT_SHA:
        raise RuntimeError("Refusing request: article text or revised prompt hash changed")
    if len(article) > articles.MAX_ARTICLE:
        raise RuntimeError("Refusing request: source exceeds the configured full-article input limit")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = BASE / "after" / "chan-16384-once" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    yaml_spec = yaml.safe_dump({"instructions": prompt_bytes.decode("utf-8")}, sort_keys=False, allow_unicode=True)
    user_message = json.dumps({"article": article, "repair_errors": []}, ensure_ascii=False)
    messages = llm_client._messages(yaml_spec, user_message)
    record = {
        "run_id": run_id,
        "article_id": "chan_chan",
        "source_url": source["url"],
        "input_chars": len(article),
        "input_bytes": len(article.encode("utf-8")),
        "input_sha256": article_sha,
        "truncated": False,
        "submitted_article_chars": len(article),
        "provider": "groq",
        "provider_chain": [{k: chain[0][k] for k in ("provider", "model", "base_url", "route_reason")}],
        "model": MODEL,
        "prompt_version": "sha256:" + prompt_sha,
        "response_format": articles.EXTRACTION_RESPONSE_FORMAT,
        "temperature": 0.0,
        "output_parameter": "max_completion_tokens",
        "output_cap": CAP,
        "account_billing_status": "not independently verified in Groq console",
        "allow_paid": False,
        "allow_openrouter_fallback": False,
        "request_count_limit": 1,
        "request_count": 1,
        "request_parameters": {
            "model": MODEL,
            "temperature": 0.0,
            "messages": messages,
            "response_format": articles.EXTRACTION_RESPONSE_FORMAT,
            "max_completion_tokens": CAP,
        },
        "status": "request_started",
    }
    result_path = run_dir / "result.json"
    write_json(result_path, record)

    try:
        response = llm_client.ask_llm_usage(
            {"instructions": prompt_bytes.decode("utf-8")}, user_message,
            model=MODEL,
            temperature=0.0,
            response_format=articles.EXTRACTION_RESPONSE_FORMAT,
            max_completion_tokens=CAP,
        )
    except Exception as exc:
        provider_attempts = getattr(exc, "attempts", []) or []
        last = provider_attempts[-1] if provider_attempts else {}
        record.update({
            "status": "rate_limited" if last.get("http_status") == 429 or llm_client.is_rate_limit(exc) else "request_failed",
            "http_status": last.get("http_status"),
            "request_attempts": provider_attempts,
            "error_type": last.get("exception_type") or type(exc).__name__,
            "error_code": last.get("error_code"),
            "error_body": last.get("error_body"),
            "rate_limit_headers": last.get("rate_limit_headers") or {},
            "error": str(llm_client._redact_provider_value(str(exc)))[:1600],
            "retry_policy": "no retry; one request maximum",
        })
        write_json(result_path, record)
        print(json.dumps({"result_file": str(result_path.relative_to(BASE)), "status": record["status"],
                          "http_status": record.get("http_status"), "rate_limit_headers": record.get("rate_limit_headers")},
                         ensure_ascii=False, indent=2))
        return 0

    raw = response.get("content") or ""
    raw_path = run_dir / "raw_reply.txt"
    raw_path.write_text(raw, encoding="utf-8")
    record.update({
        "http_status": response.get("http_status"),
        "served_model": response.get("model"),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "rate_limit_headers": response.get("rate_limit_headers") or {},
        "cost_usd": response.get("cost_usd", "unknown"),
        "elapsed_seconds": response.get("elapsed_s"),
        "raw_reply_file": str(raw_path.relative_to(BASE)),
        "raw_reply_chars": len(raw),
    })
    try:
        parsed = articles._parse_extract(raw)
    except Exception as exc:
        record.update({"status": "parse_failed", "parse_error": str(exc), "parsed_items": None,
                       "accepted_claims": [], "rejected_claims": []})
        write_json(result_path, record)
        print(json.dumps({"result_file": str(result_path.relative_to(BASE)), "status": record["status"],
                          "finish_reason": record.get("finish_reason"), "usage": record.get("usage")},
                         ensure_ascii=False, indent=2))
        return 0

    accepted, rejected = extract_claims._split(article, parsed)
    truncated = response.get("finish_reason") in {"length", "max_tokens"}
    record.update({
        "status": "truncated" if truncated else "completed",
        "parsed_items": len(parsed.get("claims") or []),
        "accepted_items": len(accepted),
        "rejected_items": len(rejected),
        "accepted_claims": accepted,
        "rejected_claims": rejected,
        "parse_method": "articles._parse_extract JSON decoder; extract_claims._split exact-quote/local validation",
        "rejection_reasons": [error for item in rejected for error in item.get("errors", [])],
        "retry_policy": "no retry; one request maximum",
    })
    write_json(result_path, record)
    print(json.dumps({"result_file": str(result_path.relative_to(BASE)), "status": record["status"],
                      "http_status": record.get("http_status"), "finish_reason": record.get("finish_reason"),
                      "usage": record.get("usage"), "parsed_items": record.get("parsed_items"),
                      "accepted_items": record.get("accepted_items"), "rejected_items": record.get("rejected_items"),
                      "raw_reply_file": record.get("raw_reply_file")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
