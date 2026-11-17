#!/usr/bin/env python3
"""Attach local fixture evidence to extracted claims. No search API and no verdict.

Opinion and prediction claims are skipped. A claim that is missing context stays
needs_context. An absence claim proceeds only when a complete document is in the
fixture set; an empty passage list does not show that the document omits anything.
A claim with no matching fixture is source_not_found, which is not a refutation.

    python3 pipeline/acquire_evidence.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import extract_claims

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "dev" / "extract"
FIXTURES = ROOT / "data" / "dev" / "fixtures" / "harbor" / "sources.json"
DOCUMENT_TYPES = frozenset({"report", "announcement", "headline"})
_PERCENT = re.compile(r"\d+(?:\.\d+)?%")
_MONEY = re.compile(r"\$\s?\d+(?:\.\d+)?")
_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
_STOP = frozenset(
    """
    a an the of on for that this and or to in is at by its it are was were be
    report says said lists paper authors document note same does not say never
    names add adds with from into than then
    """.split()
)


def quote_matches(body: str, quote: str, start: int, end: int) -> bool:
    """True only when body[start:end] is exactly quote."""
    if not isinstance(body, str) or not isinstance(quote, str):
        return False
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < start or end > len(body):
        return False
    return body[start:end] == quote


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data["sources"] if isinstance(data, dict) else data
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{path} has no sources")
    loaded: list[dict[str, Any]] = []
    for document in documents:
        loaded.append(_check_document(document))
    return loaded


def acquire(claims: list[dict[str, Any]], documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one evidence row per claim. Rows have a status and no verdict."""
    return {"claims": [acquire_one(claim, documents) for claim in claims]}


def acquire_one(claim: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    row = _base_row(claim)
    held = _held_status(claim)
    if held:
        row["status"] = held
        row["reason"] = _held_reason(claim, held)
        return row

    chosen = select_documents(claim, documents)
    if not chosen:
        row["status"] = "source_not_found"
        ref = claim.get("document_ref")
        target = f" for {ref}" if ref else ""
        row["reason"] = f"No fixture source matched this claim{target}. The claim is left undecided."
        return row

    if extract_claims.needs_full_document(claim):
        return _absence_row(row, chosen)

    passage, source = _best_supported_passage(claim, chosen)
    if passage is None or source is None:
        row["status"] = "passage_not_found"
        row["source"] = _source_view(chosen[0])
        row["reason"] = "A fixture source matched, but no sentence in it contains this claim's anchors."
        return row
    _require_match(source["body"], passage)
    row["status"] = "ready"
    row["source"] = _source_view(source)
    row["passages"] = [passage]
    return row


def select_documents(claim: dict[str, Any], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick fixture documents by document_ref or source type. This is not a web search."""
    scope = claim.get("verification_scope")
    quote = str(claim.get("source_quote") or "")
    normalized = str(claim.get("normalized_claim") or "")
    if scope == "document_content":
        ref = claim.get("document_ref")
        if not ref:
            return []
        return [document for document in documents if _same_ref(document.get("document_ref"), ref)]
    if extract_claims.is_headline_wording(quote, normalized):
        return [document for document in documents if document["document_type"] == "headline"]
    if re.search(r"\bposted\b", f"{quote} {normalized}", re.IGNORECASE):
        return [document for document in documents if document["document_type"] == "announcement"]
    return []


def _absence_row(row: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Hand over a complete document. Missing sentences do not prove an omission."""
    complete = [document for document in documents if document["is_complete"]]
    row["proves_absence"] = False
    row["passages"] = []
    if not complete:
        row["status"] = "pending_document"
        row["source"] = _source_view(documents[0])
        row["reason"] = (
            "This claim needs the complete document. The fixture on hand is not complete, "
            "so the missing passage is not evidence."
        )
        return row
    document = complete[0]
    span = {
        "source_id": document["source_id"],
        "quote": document["body"],
        "start": 0,
        "end": len(document["body"]),
        "is_complete": True,
    }
    _require_match(document["body"], span)
    row["status"] = "ready"
    row["source"] = _source_view(document)
    row["document"] = span
    row["reason"] = (
        "The complete document is attached for a later check. "
        "An empty passage list does not establish that the document omits this point."
    )
    return row


def _best_supported_passage(
    claim: dict[str, Any], documents: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    best_passage = None
    best_source = None
    best_score = -1
    for document in documents:
        for start, end, sentence in _sentences(document["body"]):
            score = _passage_score(claim, sentence)
            if score is None or score <= best_score:
                continue
            passage = {
                "source_id": document["source_id"],
                "quote": sentence,
                "start": start,
                "end": end,
            }
            if not quote_matches(document["body"], sentence, start, end):
                continue
            best_passage = passage
            best_source = document
            best_score = score
    return best_passage, best_source


def _passage_score(claim: dict[str, Any], sentence: str) -> int | None:
    needed = _anchors(claim)
    if needed and not all(_contains(sentence, anchor) for anchor in needed):
        return None
    claim_tokens = _content_tokens(f"{claim.get('source_quote') or ''} {claim.get('normalized_claim') or ''}")
    sentence_tokens = set(_content_tokens(sentence))
    overlap = sum(1 for token in claim_tokens if token in sentence_tokens)
    if not needed and (overlap < 3 or overlap * 2 < len(claim_tokens)):
        return None
    return overlap + (10 * len(needed))


def _anchors(claim: dict[str, Any]) -> list[str]:
    text = f"{claim.get('source_quote') or ''} {claim.get('normalized_claim') or ''}"
    found = [*_PERCENT.findall(text), *_MONEY.findall(text), *_DATE.findall(text)]
    for match in _QUOTED.finditer(text):
        phrase = next((group for group in match.groups() if group), "")
        if phrase.strip():
            found.append(phrase)
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def _contains(sentence: str, anchor: str) -> bool:
    folded = sentence.lower()
    target = anchor.lower().strip()
    if target in folded:
        return True
    return _words(target) in _words(sentence) if _words(target) else False


def _sentences(body: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?\n]+[.!?]?", body):
        raw = match.group(0)
        sentence = raw.strip()
        if not sentence:
            continue
        start = match.start() + raw.find(sentence)
        spans.append((start, start + len(sentence), sentence))
    return spans


def _content_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9%]+", text.lower()) if token not in _STOP]


def _words(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def _held_status(claim: dict[str, Any]) -> str | None:
    if claim.get("verification_scope") == "text_analysis":
        return "skipped"
    status = claim.get("status")
    if status in {"skipped", "needs_context"}:
        return status
    fact_type = claim.get("fact_type") or claim.get("category")
    if fact_type in {"opinion", "prediction"}:
        return "skipped"
    if claim.get("missing_context") and status != "pending_document":
        return "needs_context"
    return None


def _held_reason(claim: dict[str, Any], status: str) -> str:
    if claim.get("status_reason"):
        return str(claim["status_reason"])
    if status == "skipped":
        fact_type = claim.get("fact_type") or claim.get("category") or "skipped"
        return f"{fact_type} is not sent to retrieval"
    missing = claim.get("missing_context") or []
    if missing:
        return "Missing context: " + "; ".join(missing)
    return "Missing context."


def _base_row(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": claim.get("id"),
        "normalized_claim": claim.get("normalized_claim"),
        "fact_type": claim.get("fact_type"),
        "verification_scope": claim.get("verification_scope"),
        "document_ref": claim.get("document_ref"),
        "status": None,
        "source": None,
        "passages": [],
        "document": None,
    }


def _source_view(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": document["source_id"],
        "title": document["title"],
        "document_type": document["document_type"],
        "document_ref": document.get("document_ref"),
        "is_complete": document["is_complete"],
        "fictional": True,
    }


def _check_document(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("fixture source is not an object")
    if document.get("fictional") is not True:
        raise ValueError(f"{document.get('source_id')} is not marked fictional")
    for field in ("source_id", "title", "body", "document_type", "is_complete"):
        if field not in document:
            raise ValueError(f"{document.get('source_id')} is missing {field}")
    if document["document_type"] not in DOCUMENT_TYPES:
        raise ValueError(f"{document['source_id']} has document_type {document['document_type']!r}")
    if not isinstance(document["body"], str) or not document["body"].strip():
        raise ValueError(f"{document['source_id']} has an empty body")
    if not isinstance(document["is_complete"], bool):
        raise ValueError(f"{document['source_id']} is_complete must be true or false")
    ref = document.get("document_ref")
    if ref is not None and not isinstance(ref, str):
        raise ValueError(f"{document['source_id']} document_ref must be a string")
    return {
        "source_id": document["source_id"],
        "title": document["title"],
        "body": document["body"],
        "document_type": document["document_type"],
        "document_ref": ref,
        "is_complete": document["is_complete"],
        "fictional": True,
    }


def _same_ref(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return _words(left) == _words(right)


def _require_match(body: str, span: dict[str, Any]) -> None:
    if not quote_matches(body, span["quote"], span["start"], span["end"]):
        raise ValueError(f"{span.get('source_id')} quote does not match the stored body")


def _print_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        source = row.get("source") or {}
        source_id = source.get("source_id") or "-"
        print(f"{row['claim_id']}  {row['status']}  {row.get('verification_scope') or '-'}  {source_id}")
        print(f"  claim: {row.get('normalized_claim')}")
        if row.get("reason"):
            print(f"  {row['reason']}")
        for passage in row["passages"]:
            print(f"  quote [{passage['start']}:{passage['end']}]: {passage['quote']}")
        document = row.get("document")
        if document:
            print(
                f"  full document [{document['start']}:{document['end']}] "
                f"complete={document['is_complete']} chars={len(document['quote'])}"
            )


def _load_claims(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("claims"), list):
        return data["claims"]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path} has no claims")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach local fixture evidence. Does not search or verdict.")
    parser.add_argument("--claims", type=Path, default=SAMPLE / "last_extract.json")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--out", type=Path, default=SAMPLE / "last_evidence.json")
    args = parser.parse_args()
    result = acquire(_load_claims(args.claims), load_fixtures(args.fixtures))
    _print_rows(result["claims"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
