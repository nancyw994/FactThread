import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]

import llm_client
import articles
import check_server

llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
raw_dir = ROOT / "data/dev/article_mvp_regression/diagnostics/raw"
raw_dir.mkdir(parents=True, exist_ok=True)
orig = llm_client.ask_llm
calls = []

def capture(spec, user, *args, **kwargs):
    raw = orig(spec, user, *args, **kwargs)
    name = f"reply-{len(calls)+1:02d}.txt"
    (raw_dir / name).write_text(raw, encoding="utf-8")
    try:
        request = json.loads(user)
    except (TypeError, json.JSONDecodeError):
        request = {}
    calls.append({"file": name, "chars": len(raw), "request_chars": len(user),
                  "article_chars": len(request.get("article") or ""),
                  "repair_errors": request.get("repair_errors") or [],
                  "response_format": kwargs.get("response_format", {}).get("type")})
    return raw

llm_client.ask_llm = capture
inputs = json.loads(Path("/private/tmp/factthread_pilot_inputs.json").read_text())
apollo = (
    "Apollo 11 landed the first humans on the Moon on July 20, 1969. "
    "The first Moon landing happened in 1972. "
    "Apollo 11 was humanity's greatest achievement. "
    "NASA will establish a permanent lunar base by 2030."
)
targets = [(item["id"], item["text"], item["url"]) for item in inputs]
targets.append(("apollo_control", apollo, None))
book = articles.ArticleBook(view=check_server.public_view)
out = []
for key, text, url in targets:
    before = len(calls)
    started = book.start_extract(text, url)
    article_id = started["id"]
    while True:
        snap = book.snapshot(article_id)
        if snap["status"] != "running":
            break
        time.sleep(0.25)
    out.append({
        "id": key, "input_chars": len(text), "truncated": False,
        "status": snap["status"], "error": snap.get("error"),
        "extraction_status": snap.get("extraction_status"),
        "extraction_diagnostics": snap.get("extraction_diagnostics") or [],
        "candidate_count": len(snap.get("candidates") or []),
        "candidates": snap.get("candidates") or [], "call_count": len(calls)-before,
        "calls": calls[before:],
    })
result = {
    "provider_chain": [{"provider": x["provider"], "model": x["model"]} for x in llm_client.provider_chain(llm_client.DEFAULT_MODEL)],
    "allow_paid": llm_client.ALLOW_PAID,
    "allow_openrouter_fallback": llm_client.ALLOW_OPENROUTER_FALLBACK,
    "runs": out,
}
(ROOT / "data/dev/article_mvp_regression/diagnostics/replay_entry.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps([{k:v for k,v in x.items() if k != "candidates"} | {"candidate_count":x["candidate_count"], "candidate_quotes":[c.get("source_quote") for c in x["candidates"]]} for x in out], ensure_ascii=False, indent=2))
