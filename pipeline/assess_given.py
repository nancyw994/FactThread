#!/usr/bin/env python3
"""Evidence-given assessment with strict evidence IDs.

The model receives a claim and passages that were already retrieved. It does not
search, and it does not supply source URLs. Code checks that every cited ID
belongs to this run, that each quote matches the stored passage, and that every
evidence item resolves to a source in the packet. One repair is attempted.
A failed check becomes an incomplete result: invalid citations are not dropped
so that a strong verdict can still be published.

This is the design's evidence-given core. It does not replace pipeline/04_compare.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "assess_given.v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

VERDICTS = frozenset({"supported", "refuted", "insufficient_evidence", "conflicting_evidence"})
RELATIONS = frozenset({"entailment", "contradiction", "neutral"})
CATEGORIES = frozenset({"factual", "opinion", "prediction", "needs_context"})
NON_FACTUAL = frozenset({"opinion", "prediction"})

Complete = Callable[[dict[str, Any], list[str] | None], Any]


def validate_packet(packet: dict[str, Any]) -> list[str]:
    """Reject a packet whose passages do not resolve to its own source text."""
    errors: list[str] = []
    if not isinstance(packet, dict):
        return ["packet is not an object"]
    claim = packet.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        errors.append("claim must be a non-empty string")
    category = packet.get("category")
    if category not in CATEGORIES:
        errors.append("category must be factual, opinion, prediction, or needs_context")
    sources = packet.get("sources")
    passages = packet.get("passages")
    if not isinstance(sources, list):
        errors.append("sources must be a list")
        sources = []
    if not isinstance(passages, list):
        errors.append("passages must be a list")
        passages = []

    by_source: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"sources[{index}] is not an object")
            continue
        sid = source.get("id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"sources[{index}] needs an id")
            continue
        if sid in by_source:
            errors.append(f"duplicate source id {sid!r}")
        text = source.get("text")
        if not isinstance(text, str) or not text:
            errors.append(f"source {sid!r} has no extracted text")
        status = source.get("extraction_status", "ok")
        if status != "ok":
            errors.append(f"source {sid!r} extraction_status is {status!r}, not ok")
        url = source.get("canonical_url")
        parsed = urlparse(url if isinstance(url, str) else "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append(f"source {sid!r} canonical_url must be http or https")
        by_source[sid] = source

    seen_passages: set[str] = set()
    for index, passage in enumerate(passages):
        if not isinstance(passage, dict):
            errors.append(f"passages[{index}] is not an object")
            continue
        pid = passage.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"passages[{index}] needs an id")
            continue
        if pid in seen_passages:
            errors.append(f"duplicate passage id {pid!r}")
        seen_passages.add(pid)
        sid = passage.get("source_id")
        source = by_source.get(sid) if isinstance(sid, str) else None
        if source is None:
            errors.append(f"passage {pid!r} does not resolve to a source")
            continue
        quote = passage.get("quote")
        if not isinstance(quote, str) or not quote:
            errors.append(f"passage {pid!r} needs a quote")
            continue
        text = source.get("text") if isinstance(source.get("text"), str) else ""
        located = _locate(text, quote, passage.get("start_offset"), passage.get("end_offset"))
        if isinstance(located, str):
            errors.append(f"passage {pid!r} {located}")
    return errors


def validate_assessment(packet: dict[str, Any], output: Any) -> list[str]:
    """Check a model object against the packet. Do not rewrite it."""
    if not isinstance(output, dict):
        return ["response is not an object"]
    errors: list[str] = []
    passages = _index(packet.get("passages"))
    verdict = output.get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"verdict {verdict!r} is not a product label")
    explanation = output.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        errors.append("explanation is empty")
    questions = output.get("unresolved_questions")
    if not isinstance(questions, list) or not all(isinstance(item, str) for item in questions):
        errors.append("unresolved_questions must be a list of strings")
    if output.get("needs_review") is not None and not isinstance(output.get("needs_review"), bool):
        errors.append("needs_review must be a boolean")

    citations = output.get("explanation_citations")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        errors.append("explanation_citations must be a list of strings")
        citations = []
    evidence = output.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
        evidence = []

    relations: dict[str, str] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            errors.append(f"evidence[{index}] is not an object")
            continue
        eid = item.get("id")
        if not isinstance(eid, str) or not eid:
            errors.append(f"evidence[{index}] needs an id")
            continue
        if eid in relations:
            errors.append(f"duplicate evidence id {eid!r}")
            continue
        passage = passages.get(eid)
        if passage is None:
            errors.append(f"evidence {eid!r} is not in this run")
            continue
        if item.get("source_id") != passage.get("source_id"):
            errors.append(f"evidence {eid!r} source_id does not match the stored passage")
        if item.get("quote") != passage.get("quote"):
            errors.append(f"evidence {eid!r} quote does not match the stored passage")
        relation = item.get("relation")
        if relation not in RELATIONS:
            errors.append(f"evidence {eid!r} relation {relation!r} is not entailment, contradiction, or neutral")
            continue
        relations[eid] = relation

    for cid in citations:
        if cid not in passages:
            errors.append(f"citation {cid!r} is not in this run")
        elif cid not in relations:
            errors.append(f"citation {cid!r} is not listed in evidence")

    cited = [cid for cid in citations if cid in relations]
    if verdict == "supported" and not any(relations[cid] == "entailment" for cid in cited):
        errors.append("supported verdict has no cited entailment passage")
    if verdict == "refuted" and not any(relations[cid] == "contradiction" for cid in cited):
        errors.append("refuted verdict has no cited contradiction passage")
    if verdict == "conflicting_evidence":
        kinds = {relations[cid] for cid in cited}
        if "entailment" not in kinds or "contradiction" not in kinds:
            errors.append("conflicting_evidence verdict needs cited entailment and contradiction")
    if verdict == "insufficient_evidence" and passages and not cited:
        errors.append("insufficient_evidence with supplied passages must cite the passages it used")
    return errors


def assess(packet: dict[str, Any], complete: Complete, *, run_id: str | None = None) -> dict[str, Any]:
    """Assess one packet. `complete` returns the model object and may be called twice."""
    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    claim = packet.get("claim") if isinstance(packet, dict) else None
    category = packet.get("category") if isinstance(packet, dict) else None
    base = _envelope(run_id, claim if isinstance(claim, str) else "", category if isinstance(category, str) else None, packet)

    packet_errors = validate_packet(packet) if isinstance(packet, dict) else ["packet is not an object"]
    if packet_errors:
        base.update(
            status="failed",
            verdict=None,
            explanation="The evidence packet is invalid, so no factual verdict was assigned.",
            needs_review=True,
            validation_status="failed",
            validation_failures=packet_errors,
            error_code="invalid_input",
        )
        return base

    if category in NON_FACTUAL:
        base.update(
            status="completed",
            verdict=None,
            explanation=f"This statement is categorized as {category}, so FactThread does not assign a factual verdict.",
            needs_review=False,
            validation_status="passed",
        )
        return base

    rejected: list[dict[str, Any]] = []
    output: Any = None
    errors: list[str] = []
    for _ in range(2):
        try:
            output = complete(packet, errors or None)
        except Exception as exc:  # operational failure, not a refutation
            base.update(
                status="failed",
                verdict=None,
                explanation="Assessment did not finish. This is not a factual refutation.",
                needs_review=True,
                validation_status="failed",
                error_code="model_error",
            )
            base["debug"]["error"] = type(exc).__name__
            base["debug"]["rejected"] = rejected
            return base
        errors = validate_assessment(packet, output)
        if not errors:
            break
        rejected.append({"response": copy.deepcopy(output), "validation_failures": errors})
        output = None

    base["debug"]["rejected"] = rejected
    if errors:
        base.update(
            status="incomplete",
            verdict=None,
            explanation="The assessment cited evidence that does not match this run, so no verdict was published.",
            needs_review=True,
            validation_status="failed",
            validation_failures=errors,
            error_code="validation_failed",
        )
        return base

    published = _publish(packet, output)
    review = bool(output.get("needs_review")) or published["verdict"] in {
        "insufficient_evidence",
        "conflicting_evidence",
    }
    base.update(
        status="completed",
        verdict=published["verdict"],
        explanation=published["explanation"],
        explanation_citations=published["explanation_citations"],
        evidence=published["evidence"],
        sources=published["sources"],
        unresolved_questions=published["unresolved_questions"],
        needs_review=review,
        validation_status="passed",
    )
    return base


def load_sample(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f"{path} does not contain a cases list")
    return cases


def packet_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Model input only. Gold labels stay in the sample file."""
    packet = case.get("input")
    if not isinstance(packet, dict):
        raise ValueError(f"case {case.get('id')!r} has no input object")
    return copy.deepcopy(packet)


def default_complete(model: str, base_url: str) -> Complete:
    def complete(packet: dict[str, Any], repair_errors: list[str] | None) -> dict[str, Any]:
        from lib import chat, extract_json_object, load_prompt, make_client

        user = json.dumps(
            {"packet": _model_view(packet), "repair_errors": repair_errors or []},
            ensure_ascii=False,
        )
        text = chat(make_client(base_url), model, load_prompt("assess_given.md"), user, max_tokens=1200)
        return extract_json_object(text)

    return complete


def _envelope(run_id: str, claim: str, category: str | None, packet: dict[str, Any] | None) -> dict[str, Any]:
    as_of = packet.get("evidence_as_of") if isinstance(packet, dict) else None
    return {
        "run_id": run_id,
        "status": "failed",
        "claim": claim,
        "category": category,
        "evidence_as_of": as_of,
        "verdict": None,
        "explanation": "",
        "explanation_citations": [],
        "evidence": [],
        "sources": [],
        "unresolved_questions": [],
        "needs_review": True,
        "validation_status": "failed",
        "validation_failures": [],
        "error_code": None,
        "debug": {"prompt_version": PROMPT_VERSION, "rejected": []},
    }


def _publish(packet: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    passages = _index(packet.get("passages"))
    sources = _index(packet.get("sources"))
    evidence: list[dict[str, Any]] = []
    used_sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in output["evidence"]:
        passage = passages[item["id"]]
        sid = passage["source_id"]
        evidence.append(
            {
                "id": item["id"],
                "source_id": sid,
                "quote": passage["quote"],
                "relation": item["relation"],
            }
        )
        if sid not in seen_sources:
            seen_sources.add(sid)
            source = sources[sid]
            used_sources.append(
                {
                    "id": sid,
                    "canonical_url": source.get("canonical_url"),
                    "title": source.get("title"),
                    "published_at": source.get("published_at"),
                }
            )
    return {
        "verdict": output["verdict"],
        "explanation": output["explanation"].strip(),
        "explanation_citations": list(output["explanation_citations"]),
        "evidence": evidence,
        "sources": used_sources,
        "unresolved_questions": list(output["unresolved_questions"]),
    }


def _model_view(packet: dict[str, Any]) -> dict[str, Any]:
    """Drop stored page text. The model sees passage quotes and source dates, not URLs to invent."""
    sources = [
        {
            "id": source.get("id"),
            "title": source.get("title"),
            "published_at": source.get("published_at"),
        }
        for source in packet.get("sources") or []
        if isinstance(source, dict)
    ]
    passages = [
        {
            "id": passage.get("id"),
            "source_id": passage.get("source_id"),
            "quote": passage.get("quote"),
        }
        for passage in packet.get("passages") or []
        if isinstance(passage, dict)
    ]
    return {
        "claim": packet.get("claim"),
        "category": packet.get("category"),
        "claim_date": packet.get("claim_date"),
        "evidence_as_of": packet.get("evidence_as_of"),
        "sources": sources,
        "passages": passages,
    }


def _index(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    found: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            found[item["id"]] = item
    return found


def _locate(text: str, quote: str, start: Any, end: Any) -> str | None:
    """Return an error string, or None when the quote matches one stored span."""
    if start is None and end is None:
        first = text.find(quote)
        if first < 0:
            return "quote is not in the source text"
        if text.find(quote, first + 1) >= 0:
            return "quote is ambiguous in the source text"
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return "offsets must be integers"
    if start < 0 or end > len(text) or start >= end:
        return "offsets are outside the source text"
    if text[start:end] != quote:
        return "offsets do not match the quote"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess a development sample with supplied evidence.")
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "dev" / "evidence_given.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "dev" / "last_assess.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="openrouter")
    args = parser.parse_args()
    complete = default_complete(args.model, args.base_url)
    results = []
    for case in load_sample(args.sample):
        result = assess(packet_from_case(case), complete, run_id=f"run_{case.get('id')}")
        results.append({"case_id": case.get("id"), "result": result})
        print(f"{case.get('id')}: {result['status']} {result['verdict']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
