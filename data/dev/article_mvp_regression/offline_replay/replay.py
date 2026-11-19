#!/usr/bin/env python3
"""Replay saved article replies through local extraction code; never contacts a provider."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def run(code_root: Path, data_root: Path) -> dict:
    sys.path.insert(0, str(code_root / "pipeline"))
    sys.path.insert(0, str(code_root / "web"))
    import extract_claims
    import articles as article_mode

    regression = data_root / "data/dev/article_mvp_regression"
    articles = {item["id"]: item["text"] for item in json.loads((regression / "articles.json").read_text(encoding="utf-8"))}
    records = {
        "chan_chan": regression / "after/chan-16384-once/20261119T061032.258640Z/result.json",
        "heatwave": regression / "after/remaining-16384/20261119T065418.025619Z/heatwave.json",
    }
    report = {}
    for article_id, result_path in records.items():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        raw_path = regression / record["raw_reply_file"]
        raw = raw_path.read_text(encoding="utf-8")
        parsed = article_mode._parse_extract(raw)
        finish_reason = record.get("finish_reason")

        def saved_reply(_text, _repair):
            return {**parsed, "finish_reason": finish_reason}

        result = extract_claims.extract_claims(articles[article_id], saved_reply)
        claims = result.get("claims") or []
        rejected = result.get("rejected") or []
        status_counts = dict(Counter([result.get("status", "unknown")]))
        claim_status_counts = dict(Counter(claim.get("status", "unknown") for claim in claims))
        normalization_notes = [note for claim in claims for note in (claim.get("normalization_notes") or [])]
        quote_notes = [note for claim in claims for note in (claim.get("extraction_notes") or [])]
        speaker_gap = "直接引语未在规范化声明中标注说话人"
        flags = {
            "mid_sentence_quote": sum(any("句中开始" in note for note in claim.get("extraction_notes", [])) for claim in claims),
            "unattributed_quotation": sum(speaker_gap in (claim.get("missing_context") or []) for claim in claims),
            "normalization_fallback": sum(bool(claim.get("normalization_notes")) for claim in claims),
        }
        uncovered = extract_claims.uncovered_sentences(articles[article_id], claims, rejected) if hasattr(extract_claims, "uncovered_sentences") else []
        report[article_id] = {
            "raw_reply": str(raw_path.relative_to(data_root)),
            "finish_reason": finish_reason,
            "status_counts": status_counts,
            "claim_status_counts": claim_status_counts,
            "needs_context_count": sum(claim.get("status") == "needs_context" for claim in claims),
            "accepted": len(claims),
            "rejected": len(rejected),
            "uncovered_sentence_count": len(uncovered),
            "flags": flags,
            "normalization_fallback_notes": normalization_notes,
            "quote_position_notes": quote_notes,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--baseline-code-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    code_root = args.code_root.resolve()
    report = run(code_root, data_root)
    if args.baseline_code_root:
        with tempfile.TemporaryDirectory() as folder:
            before_path = Path(folder) / "before.json"
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()),
                "--code-root", str(args.baseline_code_root.resolve()),
                "--data-root", str(data_root), "--output", str(before_path),
            ], check=True, stdout=subprocess.DEVNULL)
            report = {
                "baseline_commit": "7eedc74",
                "before": json.loads(before_path.read_text(encoding="utf-8")),
                "after": report,
                "scope": "saved raw replies through local parser/filter/finalize code; prompt effect not measured",
            }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
