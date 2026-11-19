import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"
inputs = json.loads((BASE / "articles.json").read_text())
post = json.loads((BASE / "diagnostics/replay_entry.json").read_text())
sequential_path = BASE / "after/sequenced_extractions.json"
sequential = json.loads(sequential_path.read_text()) if sequential_path.exists() else None
post_by_id = {item["id"]: item for item in post["runs"]}
rows = []
for item in inputs:
    current = post_by_id[item["id"]]
    rows.append({
        "id": item["id"],
        "source_url": item["url"],
        "input_chars": len(item["text"]),
        "truncated": False,
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "paid_allowed": False,
        "openrouter_allowed": False,
        "before": {
            "reported_status": "extracted_empty",
            "parsed_items": 0,
            "accepted_items": 0,
            "rejected_items": 0,
            "note": "Prior batch extraction helper returned empty claims and rejected lists; no candidate output was retained.",
        },
        "after": {
            "article_mode_status": current["status"],
            "extraction_status": current.get("extraction_status"),
            "request_attempts": len(current.get("extraction_diagnostics") or []),
            "parsed_items": None,
            "accepted_items": None,
            "rejected_items": None,
            "runtime_error": current.get("error"),
            "filter_diagnostics": current.get("extraction_diagnostics") or [],
        },
    })
data = {
    "provider_chain": post["provider_chain"],
    "allow_paid": post["allow_paid"],
    "allow_openrouter_fallback": post["allow_openrouter_fallback"],
    "extraction_response_format": "Groq JSON Object Mode requested; no response was received in the post-fix run because the free Groq endpoint rate-limited requests.",
    "format_trials": [
        {"mode": "strict json_schema", "result": "BadRequest", "model_reply_saved": False},
        {"mode": "json_object", "result": "rate limited on all four articles", "model_reply_saved": False},
    ],
    "independent_apollo_control": {
        "input_chars": 203,
        "truncated": False,
        "before_repair": {"raw_chars": 1193, "parsed_items": 4, "accepted_items": 4, "rejected_items": 0, "status": "extracted"},
        "raw_file": "diagnostics/raw/before/apollo-control-valid.json",
    },
    "failed_article_raw_reply": {
        "id": "chan_chan",
        "raw_chars": 14711,
        "has_json_object": False,
        "has_claims_array": False,
        "content": "Explanatory numbered claim list, copied article text, and quoted items; the response stops before a JSON object.",
        "raw_file": "diagnostics/raw/before/chan_chan-unstructured-response.txt",
        "previous_parser_behavior": "returned claims=[]; ArticleBook retried once and then reported status=extracted",
        "filter_counts": {"parsed": 0, "accepted": 0, "rejected": 0},
    },
    "articles": rows,
    "batch_metrics": {
        "before_extraction_recall_on_selected_gold": "0/12",
        "before_precision": "not measurable: zero output claims",
        "before_normalization_fidelity": "not measurable: zero output claims",
        "after_extraction_recall": "not measurable: 4/4 runtime request failures",
        "after_precision": "not measurable: no output claims",
        "after_normalization_fidelity": "not measurable: no output claims",
        "after_runtime_errors": "4/4 article requests rate-limited",
    },
}
if sequential is not None:
    attempted = {item["id"] for item in sequential.get("articles") or []}
    completed = [item for item in sequential.get("articles") or [] if item.get("status") == "extracted"]
    data["latest_sequential_validation"] = {
        "article_order": [item["id"] for item in inputs],
        "format_probe": sequential.get("format_probe"),
        "article_completion": f"{len(completed)}/{len(inputs)}",
        "articles_completed": [item["id"] for item in completed],
        "articles_not_attempted_after_stop": [item["id"] for item in inputs if item["id"] not in attempted],
        "request_attempts": len(sequential.get("attempts") or []),
        "request_failures": sum(item.get("request_status") in {"request_failed", "rate_limited"} for item in sequential.get("attempts") or []),
        "rate_limited_requests": sum(item.get("request_status") == "rate_limited" for item in sequential.get("attempts") or []),
        "stopped_on": sequential.get("stopped_on"),
        "quality_for_completed_articles": {
            "completed_articles": len(completed),
            "reference_claim_recall": None,
            "extra_item_precision": None,
            "normalization_fidelity": None,
            "reason": "No article completed; there are no successful outputs to manually score.",
        },
        "request_events_file": "after/sequenced_extractions.json",
    }
(BASE / "extraction_comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
