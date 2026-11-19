"""Offline replay: re-run current extract_claims._split on saved raw replies (no API calls).

Compares current output with the claims saved at run time and writes a diff.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]
import articles, extract_claims  # noqa: E402

RUNS = [
    ("data/dev/article_mvp_regression", "after/prompt-v2-16384/20261119T084848.429760Z"),
    ("data/dev/article_generalization", "runs/prompt-v2-16384/20261119T093811.469020Z"),
]
KEYS = ["normalized_claim", "status", "missing_context", "normalization_notes"]

def main() -> int:
    report = {"code_version": articles.EXTRACTION_CODE_VERSION, "articles": {}}
    for base_rel, run_rel in RUNS:
        base = ROOT / base_rel
        sources = {a["id"]: a for a in json.loads((base / "articles.json").read_text(encoding="utf-8"))}
        run = base / run_rel
        for record_path in sorted(run.glob("*.json")):
            if record_path.name in {"manifest.json", "checkpoint_scores.json"}:
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            text = sources[record["article_id"]]["text"]
            raw = (base / record["raw_reply_file"]).read_text(encoding="utf-8")
            accepted, rejected = extract_claims._split(text, articles._parse_extract(raw))
            old = {c["source_quote"]: c for c in record["accepted_claims"]}
            changes = []
            for c in accepted:
                prev = old.get(c["source_quote"])
                if prev is None:
                    changes.append({"source_quote": c["source_quote"], "change": "newly_accepted",
                                    "now": {k: c.get(k) for k in KEYS}})
                    continue
                diff = {k: {"before": prev.get(k), "after": c.get(k)} for k in KEYS if prev.get(k) != c.get(k)}
                if diff:
                    changes.append({"source_quote": c["source_quote"], "change": diff})
            report["articles"][record["article_id"]] = {
                "accepted_before": len(record["accepted_claims"]), "accepted_after": len(accepted),
                "rejected_after": len(rejected), "changed": changes,
            }
    out = ROOT / "data/dev/article_generalization/diagnostics/replay_normalization_fidelity.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
