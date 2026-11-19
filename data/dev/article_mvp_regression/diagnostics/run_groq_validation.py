"""Run extraction only, Chan Chan first, and stop on the first request failure."""

import ast
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]

import llm_client
import articles
import check_server
import yaml


BASE = ROOT / "data/dev/article_mvp_regression"
OUTPUT = BASE / "after/groq_validation_sequenced_extractions.json"
RAW = BASE / "after/groq-validation/raw"
RAW.mkdir(parents=True, exist_ok=True)
INPUTS = json.loads((BASE / "articles.json").read_text(encoding="utf-8"))
llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
provider_chain = [{"provider": item["provider"], "model": item["model"]}
                  for item in llm_client.provider_chain(llm_client.DEFAULT_MODEL)]
if provider_chain != [{"provider": "groq", "model": llm_client.DEFAULT_MODEL}]:
    raise RuntimeError(f"Unexpected provider route: {provider_chain}")

events = []
runs = []
lock = threading.Lock()
current_article = None
current_event = None
format_probe = None
orig_usage = llm_client.ask_llm_usage
orig_parse = articles._parse_extract


def save():
    result = {
        "provider_chain": provider_chain,
        "allow_paid": llm_client.ALLOW_PAID,
        "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
        "response_format": articles.EXTRACTION_RESPONSE_FORMAT,
        "max_completion_tokens": articles.EXTRACTION_MAX_COMPLETION_TOKENS,
        "format_probe": format_probe,
        "attempts": events,
        "articles": runs,
        "stopped_on": (
            next((item["id"] for item in reversed(runs) if item.get("status") == "error"), None)
            or ("format_probe_apollo" if format_probe is not None and format_probe.get("status") != "passed" else None)
        ),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def usage_capture(spec, user, *args, **kwargs):
    global current_event
    try:
        request = json.loads(user)
    except (TypeError, json.JSONDecodeError):
        request = {}
    model = kwargs.get("model", args[0] if len(args) > 0 else llm_client.DEFAULT_MODEL)
    temperature = kwargs.get("temperature", args[1] if len(args) > 1 else 0.0)
    response_format = kwargs.get("response_format", args[2] if len(args) > 2 else None)
    output_limit = kwargs.get("max_completion_tokens", args[3] if len(args) > 3 else None)
    yaml_spec = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True) if isinstance(spec, dict) else spec
    request_parameters = {
        "model": model,
        "temperature": temperature,
        "messages": llm_client._messages(yaml_spec, user),
    }
    if response_format is not None:
        request_parameters["response_format"] = response_format
    if output_limit is not None:
        request_parameters["max_completion_tokens"] = output_limit
    with lock:
        sequence = len(events) + 1
        event = {
            "sequence": sequence,
            "article_id": current_article,
            "request_status": "started",
            "provider": "groq",
            "requested_model": model,
            "article_chars": len(request.get("article") or ""),
            "request_chars": len(user) if isinstance(user, str) else None,
            "repair_errors": request.get("repair_errors") or [],
            "response_format": response_format,
            "max_completion_tokens": output_limit,
            "request_parameters": request_parameters,
            "finish_reason": None,
            "usage": None,
            "raw_reply_file": None,
            "parse_status": "not_run",
            "parsed_items": None,
            "accepted_items": None,
            "rejected_items": None,
            "rejection_reasons": [],
        }
        events.append(event)
        current_event = event
        save()
    try:
        result = orig_usage(spec, user, *args, **kwargs)
    except Exception as exc:
        cause = exc.__cause__
        cause_message = str(cause or "")
        cause_message = re.sub(r"(?:gsk_|sk-)[A-Za-z0-9_-]+", "[REDACTED_KEY]", cause_message)
        cause_message = re.sub(r"org_[A-Za-z0-9]+", "[REDACTED_ORG]", cause_message)
        event["request_status"] = "rate_limited" if llm_client.is_rate_limit(exc) else "request_failed"
        event["error_type"] = type(exc).__name__
        event["error"] = str(exc)[:600]
        event["provider_error_type"] = type(cause).__name__ if cause else None
        event["provider_status_code"] = getattr(cause, "status_code", None) if cause else None
        event["provider_error_detail"] = cause_message[:1000] or None
        event["retry_after_seconds"] = getattr(exc, "retry_after", None)
        event["provider_attempts"] = getattr(exc, "attempts", [])
        save()
        raise
    raw = result.get("content") or ""
    raw_file = RAW / f"reply-{sequence:02d}.txt"
    raw_file.write_text(raw, encoding="utf-8")
    event.update({
        "request_status": "completed",
        "provider": result.get("provider"),
        "served_model": result.get("model"),
        "finish_reason": result.get("finish_reason"),
        "usage": result.get("usage"),
        "raw_reply_chars": len(raw),
        "raw_reply_file": str(raw_file.relative_to(BASE)),
        "elapsed_seconds": result.get("elapsed_s"),
    })
    save()
    return result


def parse_capture(raw):
    try:
        parsed = orig_parse(raw)
    except Exception as exc:
        if current_event is not None:
            current_event["parse_status"] = "parse_failed"
            current_event["parse_error"] = f"{type(exc).__name__}: {exc}"
            save()
        raise
    if current_event is not None:
        current_event["parse_status"] = "parsed"
        current_event["parsed_items"] = len(parsed.get("claims") or [])
        save()
    return parsed


llm_client.ask_llm_usage = usage_capture
articles._parse_extract = parse_capture
book = articles.ArticleBook(view=check_server.public_view)

# A short same-configuration format probe gates all article requests.
test_tree = ast.parse((ROOT / "tests/test_articles.py").read_text(encoding="utf-8"))
apollo_node = next(
    node.value for node in test_tree.body
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SHORT_ARTICLE" for target in node.targets)
)
probe_text = ast.literal_eval(apollo_node)
current_article = "format_probe_apollo"
try:
    probe_output = articles.groq_extract(probe_text, None)
    probe_claims, probe_rejected = articles.extract_claims._split(probe_text, probe_output)
    if current_event is not None:
        current_event.update({
            "accepted_items": len(probe_claims),
            "rejected_items": len(probe_rejected),
            "rejection_reasons": [error for item in probe_rejected for error in item["errors"]],
        })
    format_probe = {
        "input_chars": len(probe_text),
        "status": "passed" if current_event and current_event.get("parse_status") == "parsed" else "failed",
        "parsed_items": len(probe_output.get("claims") or []),
        "accepted_items": len(probe_claims),
        "rejected_items": len(probe_rejected),
        "rejection_reasons": [error for item in probe_rejected for error in item["errors"]],
        "request_sequence": current_event.get("sequence") if current_event else None,
    }
except Exception as exc:
    format_probe = {
        "input_chars": len(probe_text),
        "status": "rate_limited" if llm_client.is_rate_limit(exc) else "failed",
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
        "request_sequence": current_event.get("sequence") if current_event else None,
    }
    save()

if format_probe.get("status") != "passed":
    save()
    print(json.dumps({
        "format_probe": format_probe,
        "attempts": [{key: item.get(key) for key in ["sequence", "article_id", "request_status", "finish_reason", "usage", "max_completion_tokens", "parse_status"]} for item in events],
        "articles": [],
        "stopped_on": "format_probe_apollo",
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0)
save()

for source in INPUTS:
    current_article = source["id"]
    text = source["text"].strip()
    if len(text) > articles.MAX_ARTICLE:
        raise ValueError(f"{source['id']} exceeds the article-mode limit")
    started = book.start_extract(text, source["url"])
    article_id = started["id"]
    while True:
        snapshot = book.snapshot(article_id)
        if snapshot["status"] != "running":
            break
        time.sleep(0.1)
    row = {
        "id": source["id"],
        "source_url": source["url"],
        "input_chars": len(text),
        "truncated": False,
        "status": snapshot["status"],
        "extraction_status": snapshot.get("extraction_status"),
        "error": snapshot.get("error"),
        "extraction_diagnostics": snapshot.get("extraction_diagnostics") or [],
        "candidates": snapshot.get("candidates") or [],
    }
    article_events = [item for item in events if item.get("article_id") == source["id"]]
    for event, diagnostic in zip(article_events, row["extraction_diagnostics"]):
        if diagnostic.get("stage") == "filter":
            event.update({
                "parsed_items": diagnostic.get("parsed_count"),
                "accepted_items": diagnostic.get("accepted_count"),
                "rejected_items": diagnostic.get("rejected_count"),
                "rejection_reasons": diagnostic.get("rejection_reasons") or [],
            })
        elif diagnostic.get("stage") == "parse":
            event["parse_status"] = "parse_failed"
            event["parse_error"] = diagnostic.get("error")
    runs.append(row)
    save()
    if snapshot["status"] == "error" or snapshot.get("extraction_status") in {"request_failed", "parse_failed", "invalid_input"}:
        break

save()
print(json.dumps({
    "provider_chain": provider_chain,
    "allow_paid": llm_client.ALLOW_PAID,
    "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
    "format_probe": format_probe,
    "attempts": [{key: item.get(key) for key in ["sequence", "article_id", "request_status", "finish_reason", "usage", "max_completion_tokens", "parse_status", "parsed_items", "accepted_items", "rejected_items"]} for item in events],
    "articles": [{"id": item["id"], "status": item["status"], "extraction_status": item["extraction_status"], "candidate_count": len(item["candidates"]), "input_chars": item["input_chars"]} for item in runs],
    "stopped_on": next((item["id"] for item in reversed(runs) if item.get("status") == "error"), None),
}, ensure_ascii=False, indent=2))
