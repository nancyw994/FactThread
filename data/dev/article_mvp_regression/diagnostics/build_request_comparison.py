import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"
run = json.loads((BASE / "after/sequenced_extractions.json").read_text(encoding="utf-8"))
probe = next(item for item in run["attempts"] if item.get("article_id") == "format_probe_apollo")
failed = next(item for item in run["attempts"] if item.get("article_id") == "chan_chan")
provider_error = (failed.get("provider_attempts") or [{}])[0]
probe_filters = run.get("format_probe") or {}

success = {
    "request_status": probe.get("request_status"),
    "provider": probe.get("provider"),
    "served_model": probe.get("served_model"),
    "input_chars": probe.get("article_chars"),
    "request_parameters": probe.get("request_parameters"),
    "finish_reason": probe.get("finish_reason"),
    "usage": probe.get("usage"),
    "output_limit": probe.get("max_completion_tokens"),
    "raw_reply_file": probe.get("raw_reply_file"),
    "raw_reply_chars": probe.get("raw_reply_chars"),
    "parse_status": probe.get("parse_status"),
    "parsed_items": probe.get("parsed_items"),
    "accepted_items": probe.get("accepted_items") if probe.get("accepted_items") is not None else probe_filters.get("accepted_items"),
    "rejected_items": probe.get("rejected_items") if probe.get("rejected_items") is not None else probe_filters.get("rejected_items"),
    "rejection_reasons": probe.get("rejection_reasons") or probe_filters.get("rejection_reasons"),
}
failure = {
    "request_status": failed.get("request_status"),
    "provider": failed.get("provider"),
    "requested_model": failed.get("requested_model"),
    "input_chars": failed.get("article_chars"),
    "request_parameters": failed.get("request_parameters"),
    "http_status": provider_error.get("http_status"),
    "provider_exception_class": provider_error.get("exception_type"),
    "error_type": provider_error.get("error_type"),
    "error_code": provider_error.get("error_code"),
    "error_body": provider_error.get("error_body"),
    "finish_reason": failed.get("finish_reason"),
    "usage": failed.get("usage"),
    "output_limit": failed.get("max_completion_tokens"),
    "raw_reply_file": failed.get("raw_reply_file"),
    "parse_status": failed.get("parse_status"),
    "parsed_items": failed.get("parsed_items"),
    "accepted_items": failed.get("accepted_items"),
    "rejected_items": failed.get("rejected_items"),
    "retry_after_seconds": provider_error.get("retry_after_seconds"),
}
same_config = all(
    success["request_parameters"].get(key) == failure["request_parameters"].get(key)
    for key in ("model", "temperature", "response_format", "max_completion_tokens")
)
result = {
    "credentials": "No API key or authorization header is included.",
    "same_request_configuration": same_config,
    "success": success,
    "failure": failure,
    "attempt1_badrequest": json.loads((BASE / "diagnostics/attempt1_request_comparison.json").read_text(encoding="utf-8"))["failed_attempt1"],
    "interpretation": "The same current request configuration completed the short format probe, while the Chan request failed with an explicit 429 quota body. This explains the latest Chan failure only; attempt1's earlier BadRequest still has no saved provider body and remains unclassified.",
}
out = BASE / "diagnostics/request_success_failure_comparison.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "same_request_configuration": same_config,
    "probe_finish_reason": success["finish_reason"],
    "probe_usage": success["usage"],
    "probe_parse_filter": [success["parsed_items"], success["accepted_items"], success["rejected_items"]],
    "failed_status": failure["http_status"],
    "failed_error_type": failure["error_type"],
    "failed_error_code": failure["error_code"],
    "failed_body_saved": failure["error_body"] is not None,
    "output": str(out.relative_to(ROOT)),
}, ensure_ascii=False, indent=2))
