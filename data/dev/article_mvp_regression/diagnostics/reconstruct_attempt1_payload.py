"""Reconstruct saved SDK request arguments without contacting a provider."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "scripts")]

import llm_client
import yaml

BASE = ROOT / "data/dev/article_mvp_regression"
article = json.loads((BASE / "articles.json").read_text(encoding="utf-8"))[0]
article_text = article["text"].strip()
user_message = json.dumps({"article": article_text, "repair_errors": []}, ensure_ascii=False)
prompt = (ROOT / "config/prompts/extract_claims.md").read_text(encoding="utf-8")
yaml_spec = yaml.safe_dump({"instructions": prompt}, sort_keys=False, allow_unicode=True)
messages = llm_client._messages(yaml_spec, user_message)

test_tree = ast.parse((ROOT / "tests/test_articles.py").read_text(encoding="utf-8"))
apollo_node = next(
    node.value for node in test_tree.body
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SHORT_ARTICLE" for target in node.targets)
)
apollo_text = ast.literal_eval(apollo_node)
apollo_user = json.dumps({"article": apollo_text, "repair_errors": []}, ensure_ascii=False)
apollo_messages = llm_client._messages(yaml_spec, apollo_user)

failed = {
    "source": "Reconstructed from saved attempt1 fields and the exact local request-building code; no API key or auth headers included.",
    "provider": "groq",
    "endpoint": llm_client.GROQ_BASE_URL + "/chat/completions",
    "parameters": {
        "model": llm_client.DEFAULT_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4096,
        "messages": messages,
    },
    "message_chars": {"system": len(messages[0]["content"]), "user": len(messages[1]["content"]), "user_article": len(article_text)},
    "attempt1_saved_error": {
        "message": "Model request failed. groq openai/gpt-oss-120b: BadRequestError The fixed_rules baseline was not substituted.",
        "provider_exception_class": "BadRequestError",
        "http_status": None,
        "body": None,
        "body_saved": False,
        "error_type": None,
        "error_code": None,
        "note": "Attempt1 did not persist the provider HTTP response or exception body. Status/type/code cannot be recovered from the saved record.",
    },
}
successful = {
    "source": "Reconstructed from the saved Apollo raw response and the then-current groq_extract/ask_llm code path; not a server-side request log.",
    "provider": "groq",
    "endpoint": llm_client.GROQ_BASE_URL + "/chat/completions",
    "parameters": {
        "model": llm_client.DEFAULT_MODEL,
        "temperature": 0.0,
        "response_format": None,
        "max_completion_tokens": None,
        "messages": apollo_messages,
    },
    "message_chars": {"system": len(apollo_messages[0]["content"]), "user": len(apollo_messages[1]["content"]), "user_article": len(apollo_text)},
    "saved_response": {
        "raw_file": "diagnostics/raw/before/apollo-control-valid.json",
        "raw_chars": 1193,
        "finish_reason": None,
        "usage": None,
        "parsed_items": 4,
        "accepted_items": 4,
        "rejected_items": 0,
    },
}
result = {
    "credentials": "No API key, bearer header, or credential-bearing client config is stored.",
    "failed_attempt1": failed,
    "successful_apollo_control": successful,
    "cause_classification": {
        "parameter_unsupported": "not established; the same model/response_format/output cap later succeeded on the short format probe, but attempt1's own provider body is missing",
        "structured_output_configuration": "not established; JSON Object Mode succeeded later on the short probe, but the missing attempt1 body prevents direct confirmation",
        "input_limit": "not established; 5,234 article characters is below the app limit, but provider tokenization and the provider error body were not saved",
        "server_generation_or_validation": "not established; no provider body or numeric status was saved",
        "conclusion": "unclassified; do not infer the cause from BadRequestError alone and do not change request parameters based on attempt1",
    },
    "comparison_limit": "The requests used different response-format/output-cap settings. This contrast does not identify the BadRequest cause.",
}
out = BASE / "diagnostics/attempt1_request_comparison.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "failed_model": failed["parameters"]["model"],
    "failed_format": failed["parameters"]["response_format"],
    "failed_output_limit": failed["parameters"]["max_completion_tokens"],
    "failed_message_chars": failed["message_chars"],
    "failed_http_status": None,
    "failed_body_saved": False,
    "successful_apollo_raw_chars": successful["saved_response"]["raw_chars"],
    "comparison_saved": str(out.relative_to(ROOT)),
}, ensure_ascii=False, indent=2))
