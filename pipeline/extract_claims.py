#!/usr/bin/env python3
"""Extract atomic claims from one article and check that each quote is real.

extract_claims(text) asks a model for claims. Code then drops nothing silently:
a quote that is missing, ambiguous, or the wrong shape is returned under
"rejected" with the reason. Valid claims keep the caller's review fields only.

This does not search and does not assign a verdict.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "dev" / "extract"
CATEGORIES = frozenset({"checkable_fact", "opinion", "prediction", "needs_context", "intention"})
FACT_TYPES = frozenset({"checkable_fact", "opinion", "prediction", "intention"})
SCOPES = frozenset({"external_fact", "document_content", "text_analysis"})
STATUSES = frozenset({"ready", "needs_context", "pending_document", "skipped"})
_AND_ATTRIBUTION = re.compile(
    r"^(?P<attr>.+?\b(?:says|said|adds|added|writes|claims|reports)\b)\s+(?P<a>.+?),\s+and\s+(?P<b>.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_ADDED_ATTRIBUTION = re.compile(
    r"^(?:the report says|the authors add that|according to the report,?)\s+",
    re.IGNORECASE,
)
_PRICE = re.compile(r"\$\s?\d|\bper million\b", re.IGNORECASE)
_BILLED_THING = re.compile(r"\b(api|service|product|subscription|model\s+[A-Za-z0-9]+)\b", re.IGNORECASE)
_ABSENCE = re.compile(
    r"\b(does not report|does not say|doesn't say|never names|does not claim|does not state|not reported|no named|not a result)\b",
    re.IGNORECASE,
)
_DOC_NOUN = re.compile(r"\b(report|paper|document|authors|note)\b", re.IGNORECASE)
_DOCUMENT = re.compile(
    r"\b(report|paper|authors|document)\b.{0,48}\b(says|said|lists|add|adds|states|does not|doesn't|never)\b"
    r"|\b(the report says|the authors add|does not report|does not say|never names)\b",
    re.IGNORECASE,
)
_PRICE_GAP = "which product or service the price applies to"
_DOC_GAP = "which document is referred to"
_HEADLINE_GAP = "需要明确标题所指的具体结果或衡量指标"
_NAMED_REPORT = re.compile(
    r"\b(?:its|the|a|an)\s+(?P<name>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+report\b"
)
_NAMED_NOTE = re.compile(r"\b(?P<name>the (?:[\w']+\s+){0,2}note)\b", re.IGNORECASE)
_REPORT_REF = re.compile(r"\b(the same report|this report|that report|the report|the paper)\b", re.IGNORECASE)
_NOTE_REF = re.compile(r"\bthe (?:[\w']+\s+){0,2}note\b", re.IGNORECASE)
_DOC_REF = re.compile(r"\bthe document\b", re.IGNORECASE)
_PRONOUN = re.compile(r"^(?:It|it)\b")
_EXAM_NAME = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:-[A-Z][A-Za-z0-9]+)+)\b")
_INTENT = re.compile(r"\b(on purpose|deliberately|intentionally|intended to|meant to)\b", re.IGNORECASE)
_EVALUATION = re.compile(r"\b(I think|I believe|I find|in my view|the best|the worst)\b", re.IGNORECASE)
_FUTURE = re.compile(r"\bby 20\d\d\b.+\bwill\b", re.IGNORECASE)
_META = re.compile(
    r"\b(the line|the sentence|the prediction|this line|that line)\b.{0,120}\b(is|are)\b.{0,80}\b"
    r"(a preference|an opinion|a forecast|the writer's forecast)\b",
    re.IGNORECASE | re.DOTALL,
)
_COMPOUND = re.compile(
    r"^(?P<left>.+?\b(?:is|are)\b\s+(?:the writer's forecast|a forecast|a preference|an opinion))"
    r",\s+not\s+(?P<right>.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_HEADLINE_ATTR = re.compile(r"\bheadlines?\b.{0,40}\b(called|said|wrote|described)\b", re.IGNORECASE)
_WORDING_GAP = re.compile(r"baseline|compar|system that|which system", re.IGNORECASE)
_HEADLINE_RESULT = re.compile(r"\b(?:gain|increase|improvement|rise|drop|decline|result|score|performance|growth)\b", re.IGNORECASE)
_MODALITY = re.compile(r"\b(?:will|would|may|might|could|can|should|must|cannot|can't|did not|does not|never|not)\b", re.IGNORECASE)
_TEXT_WORD = re.compile(r"[\w]+(?:[-’'][\w]+)*", re.UNICODE)
_EN_STOP = frozenset(
    "a an and are as at be been being by for from had has have he her hers him his i in into is it its me my of on or our ours she that the their theirs them they this those these to was were what when where which while who why will with we you your".split()
)
_NUMBER = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
_ORDINAL = re.compile(r"\b(?:first(?:[- ]ever)?|second|third|fourth|fifth|earliest|initially|initial)\b|首次|最早|第一|第二|第三", re.IGNORECASE)
_EXAM_REF = re.compile(r"\bthat exam\b", re.IGNORECASE)
_VAGUE_ACTION_REFERENCE = re.compile(r"\b(?:these|those|such)\s+(?:actions?|steps?|measures?|changes?|decisions?|efforts?)\b", re.IGNORECASE)
_CAUSAL = re.compile(
    r"\b(?:cause[sd]?|allow(?:s|ed)?|enable[sd]?|lead(?:s)? to|led to|result(?:s|ed)? in|contribut(?:e|es|ed) to|drive[sn]?|drove|promot(?:e|es|ed|ing)|fuel(?:s|ed|ing)|accelerat(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_VAGUE_OUTCOME = re.compile(
    r"\b(?:flourish(?:ed|es|ing)?|thrive[sd]?|prosper(?:ed|s|ing)?|improv(?:e|es|ed|ing)|"
    r"succeed(?:ed|s|ing)?|successful(?:ly)?|grow(?:s|th|ing)?|development|progress|better|stronger)\b",
    re.IGNORECASE,
)
_MEASURE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s?%\b|\b(?:by|from|to)\s+\d|\b(?:rate|count|number|share|percentage|score|measure|metric|index|volume)\b)",
    re.IGNORECASE,
)
_ATTRIBUTION = re.compile(
    r"\b(?:say|says|said|tells|told|telling|write|writes|wrote|written|"
    r"explain|explains|explained|add|adds|added|note|notes|noted|"
    r"report|reports|reported|claim|claims|claimed|according to)\b",
    re.I,
)
_NAMED_ATTRIBUTION = re.compile(
    r"\b(?P<speaker>(?:[A-Z][A-Za-z’'-]+)(?:\s+[A-Z][A-Za-z’'-]+){0,3})\s+"
    r"(?:says?|said|tells|told|telling|writes?|wrote|explains?|explained|adds?|added|notes?|noted|"
    r"reports?|reported|claims?|claimed)\b"
    r"|\baccording to\s+(?P<according_speaker>(?:[A-Z][A-Za-z’'-]+)(?:\s+[A-Z][A-Za-z’'-]+){0,3})",
    re.I,
)
_SPEAKER_GAP = "直接引语未在规范化声明中标注说话人"
_NORMALIZATION_FALLBACK = "模型规范化改变了时态/确定程度，已回退为原文措辞"
_QUOTE_TYPOGRAPHY_NOTE = "引文与原文仅大小写/标点不同，已替换为原文精确片段"
_QUOTE_EQUIVALENTS = {
    '"': '[\"“”]', "“": '[\"“”]', "”": '[\"“”]',
    "'": "['‘’]", "‘": "['‘’]", "’": "['‘’]",
    "-": "[-–—]", "–": "[-–—]", "—": "[-–—]",
}
_QUOTE_TRAILING = frozenset(".,;:\"“”'‘’")
_QUOTE_PUNCTUATION = frozenset(".,;:")

Complete = Callable[[str, list[str] | None], Any]


class ExtractionParseError(ValueError):
    """The model replied, but its response was not a usable extraction object."""


def locate_quote(text: str, quote: str) -> str | tuple[int, int]:
    """Return an error, or the unique span where quote sits in text."""
    if not isinstance(quote, str) or not quote.strip():
        return "source_quote is empty"
    ranges = find_quote_ranges(text, quote)
    if not ranges:
        return "source_quote is not in the article"
    if len(ranges) > 1:
        return "source_quote is ambiguous in the article"
    return tuple(ranges[0])


def find_quote_ranges(text: str, quote: str) -> list[list[int]]:
    """Return exact occurrences first, then narrowly typographic fallback positions."""
    if not isinstance(quote, str) or not quote:
        return []
    ranges = _exact_quote_ranges(text, quote)
    return ranges if ranges else _typographic_quote_ranges(text, quote)


def _exact_quote_ranges(text: str, quote: str) -> list[list[int]]:
    ranges: list[list[int]] = []
    start = 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            break
        ranges.append([index, index + len(quote)])
        start = index + 1
    return ranges


def _typographic_quote_ranges(text: str, quote: str) -> list[list[int]]:
    """Match only first-letter case, quote/dash glyphs, and quote-final punctuation."""
    end = len(quote)
    while end and quote[end - 1] in _QUOTE_TRAILING:
        end -= 1
    core = quote[:end]
    if not core:
        return []
    has_trailing_punctuation = any(char in _QUOTE_PUNCTUATION for char in quote[end:])

    pattern_parts: list[str] = []
    first_alpha_seen = False
    for char in core:
        if char.isalpha() and not first_alpha_seen:
            pattern_parts.append(f"[{re.escape(char.lower())}{re.escape(char.upper())}]")
            first_alpha_seen = True
        elif char in _QUOTE_EQUIVALENTS:
            pattern_parts.append(_QUOTE_EQUIVALENTS[char])
        else:
            pattern_parts.append(re.escape(char))
    if not first_alpha_seen:
        return []
    suffix = r"[.,;:]?[\"“”'‘’]*" if has_trailing_punctuation else r"[\"“”'‘’]*"
    # Word boundaries: a case-folded first letter or a dropped trailing mark
    # must not let the quote match inside a longer word ("no." vs "nothing").
    prefix = r"(?<![^\W_])" if core[0].isalnum() else ""
    boundary = r"(?![^\W_])" if core[-1].isalnum() else ""
    body = prefix + "".join(pattern_parts) + boundary + suffix
    matcher = re.compile("(?=" + body + ")")
    ranges: list[list[int]] = []
    for match in matcher.finditer(text):
        # The lookahead keeps repeated/overlapping fallback matches visible.
        found = re.compile(body).match(text, match.start())
        if found:
            ranges.append([match.start(), found.end()])
    return ranges


def check_claim(text: str, claim: Any) -> list[str]:
    if not isinstance(claim, dict):
        return ["claim is not an object"]
    errors: list[str] = []
    quote = claim.get("source_quote") or ""
    if not _exact_quote_ranges(text, quote):
        fallback_ranges = _typographic_quote_ranges(text, quote)
        if not fallback_ranges:
            errors.append("source_quote is not in the article")
        elif len(fallback_ranges) > 1:
            errors.append("source_quote is ambiguous in the article")
    normalized = claim.get("normalized_claim")
    if not isinstance(normalized, str) or not normalized.strip():
        errors.append("normalized_claim is empty")
    if claim.get("category") not in CATEGORIES:
        errors.append(f"category {claim.get('category')!r} is not one of the four labels")
    missing = claim.get("missing_context")
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        errors.append("missing_context must be a list of strings")
    return errors


def extract_claims(text: str, complete: Complete) -> dict[str, Any]:
    """Return claims plus explicit extraction status and per-attempt filter counts."""
    if not isinstance(text, str) or not text.strip():
        return {"claims": [], "rejected": [{"claim": None, "errors": ["article is empty"]}], "status": "invalid_input", "attempts": []}

    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, 3):
        try:
            output = complete(text, errors or None)
        except ExtractionParseError as exc:
            attempts.append({"attempt": attempt_number, "stage": "parse", "error": str(exc)})
            return {"claims": [], "rejected": rejected, "status": "parse_failed", "error": str(exc), "attempts": attempts}
        except Exception as exc:
            return {
                "claims": [],
                "rejected": [{"claim": None, "errors": [f"model_error:{type(exc).__name__}"]}],
                "status": "request_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempts + [{"attempt": attempt_number, "stage": "request", "error": type(exc).__name__}],
            }
        if isinstance(output, list):
            output = {"claims": output}
        finish_reason = output.get("finish_reason") if isinstance(output, dict) else None
        if isinstance(output, dict) and output.get("provider_parse_error"):
            message = str(output.get("error") or "model response contains no complete JSON object with a claims list")
            return {"claims": [], "rejected": [], "status": "parse_failed", "error": message,
                    "attempts": attempts + [{"attempt": attempt_number, "stage": "parse", "error": message,
                                             "finish_reason": finish_reason}]}
        if isinstance(output, dict) and output.get("status") in {"request_failed", "parse_failed"}:
            status = output["status"]
            message = str(output.get("error") or status)
            return {"claims": [], "rejected": output.get("rejected") or [], "status": status, "error": message,
                    "attempts": attempts + [{"attempt": attempt_number, "stage": status.removesuffix("_failed"), "error": message}]}
        raw_claims = output.get("claims") if isinstance(output, dict) else None
        parsed_count = len(raw_claims) if isinstance(raw_claims, list) else 0
        accepted, rejected = _split(text, output)
        attempts.append({
            "attempt": attempt_number,
            "stage": "filter",
            "parsed_count": parsed_count,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejection_reasons": [error for item in rejected for error in item["errors"]],
            **({"finish_reason": finish_reason} if finish_reason is not None else {}),
        })
        if accepted and not rejected:
            break
        if parsed_count == 0 and not rejected:
            return {"claims": [], "rejected": [],
                    "status": "incomplete" if finish_reason not in (None, "stop") else "empty",
                    "attempts": attempts}
        errors = [error for item in rejected for error in item["errors"]]
        if not errors:
            errors = ["no claims were returned"]
    numbered = _number(accepted)
    # The exposed claims are from the last filter attempt. Earlier truncated
    # replies are useful diagnostics, but must not make a successful repair
    # look partial.
    final_finish_reason = attempts[-1].get("finish_reason") if attempts else None
    status = "incomplete" if final_finish_reason not in (None, "stop") else (
        "filtered_empty" if not numbered and rejected else "completed_with_rejections" if rejected else "completed"
    )
    return {"claims": numbered, "rejected": rejected, "status": status, "attempts": attempts}


def align_to_gold(gold: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    """Suggest misses, extras, and meaning changes. A person still checks each row."""
    used: set[int] = set()
    rows: list[dict[str, Any]] = []
    for item in gold:
        match_at = _best_match(item, predicted, used)
        if match_at is None:
            rows.append({"gold_id": item["id"], "kind": "missed", "predicted_id": None})
            continue
        used.add(match_at)
        pred = predicted[match_at]
        problems = _meaning_problems(item, pred)
        rows.append(
            {
                "gold_id": item["id"],
                "kind": "meaning_changed" if problems else "matched",
                "predicted_id": pred.get("id"),
                "problems": problems,
            }
        )
    extras = [
        {"predicted_id": predicted[index].get("id"), "kind": "extra"}
        for index in range(len(predicted))
        if index not in used
    ]
    return {"rows": rows, "extras": extras}


def verify_claim(claim: dict[str, Any], evidence: dict[str, Any], complete: Complete) -> dict[str, Any]:
    """Judge one extracted claim against evidence a person already supplied."""
    from assess_given import assess

    if _needs_full_document(claim) and not evidence.get("full_document"):
        return {
            "status": "completed",
            "claim": claim.get("normalized_claim") or "",
            "verdict": "insufficient_evidence",
            "explanation": (
                "This claim is about what the document does not say or does not report. "
                "The full document was not available, so the supplied passages are not enough."
            ),
            "explanation_citations": [],
            "evidence": [],
            "unresolved_questions": ["What does the rest of the document say?"],
            "needs_review": True,
            "validation_status": "passed",
            "validation_failures": [],
            "error_code": None,
        }

    packet = {
        "claim": claim.get("normalized_claim") or "",
        "category": _assess_category(claim.get("category")),
        "claim_date": evidence.get("claim_date"),
        "evidence_as_of": evidence.get("evidence_as_of"),
        "sources": evidence.get("sources") or [],
        "passages": evidence.get("passages") or [],
    }
    return assess(packet, complete)


def load_gold(path: Path = SAMPLE / "gold.json") -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    articles = []
    for article in data["articles"]:
        text = (path.parent / article["path"]).read_text(encoding="utf-8")
        articles.append({**article, "text": text})
    return articles


def default_complete(model: str, base_url: str) -> Complete:
    def complete(text: str, repair_errors: list[str] | None) -> dict[str, Any]:
        from lib import chat, extract_json_array, extract_json_object, load_prompt, make_client

        user = json.dumps({"article": text, "repair_errors": repair_errors or []}, ensure_ascii=False)
        raw = chat(make_client(base_url), model, load_prompt("extract_claims.md"), user, max_tokens=2200)
        parsed = extract_json_object(raw)
        if parsed.get("claims"):
            return parsed
        items = extract_json_array(raw)
        return {"claims": items} if items else parsed

    return complete


def _split(text: str, output: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = output.get("claims") if isinstance(output, dict) else None
    if not isinstance(claims, list):
        return [], [{"claim": output, "errors": ["response has no claims list"]}]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for claim in claims:
        errors = check_claim(text, claim)
        if errors:
            rejected.append({"claim": claim, "errors": errors})
        else:
            for part in expand_compound(claim):
                accepted.append(_finalize(text, part))
    return accepted, rejected


def prepare_claims(text: str, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply attribution, reference, and status rules without calling a model."""
    prepared: list[dict[str, Any]] = []
    for claim in claims:
        for part in expand_compound(claim):
            prepared.append(_finalize(text, part))
    return _number(prepared)


def correct_attribution(quote: str, normalized: str) -> str:
    """Do not turn 'The report says A, and B' into 'The report says B'."""
    text = normalized.strip()
    match = _AND_ATTRIBUTION.match(quote.strip())
    if match:
        attr = match.group("attr")
        first = _words(match.group("a"))
        second = _words(match.group("b"))
        claim = _words(text)
        about_second = second[:48] in claim
        about_first = first[:48] in claim
        if about_second and not about_first and text.lower().startswith(attr.lower()):
            clause = match.group("b").strip().rstrip(".")
            return clause[:1].upper() + clause[1:] + "."
    added = _ADDED_ATTRIBUTION.match(text)
    if added and added.group(0).strip().lower().rstrip(",") not in quote.lower():
        rest = text[added.end() :].strip()
        if rest:
            return rest[:1].upper() + rest[1:]
    return text


def price_gap(quote: str, normalized: str) -> str | None:
    """A price with no named product or service in its own clause cannot be retrieved."""
    clause = _price_clause(quote, normalized)
    if _ABSENCE.search(clause) or _ABSENCE.search(normalized):
        return None
    if not _PRICE.search(clause) and not _PRICE.search(normalized):
        return None
    if _BILLED_THING.search(clause):
        return None
    return _PRICE_GAP


def verification_scope(quote: str, normalized: str) -> str:
    blob = _scope_text(quote, normalized)
    if _DOCUMENT.search(blob) or (_ABSENCE.search(blob) and _DOC_NOUN.search(blob)):
        return "document_content"
    return "external_fact"


def _scope_text(quote: str, normalized: str) -> str:
    """Use only the conjunct this claim is about, so A's attribution does not cover B."""
    match = _AND_ATTRIBUTION.match(quote.strip())
    if match:
        first = _words(match.group("a"))[:48]
        second = _words(match.group("b"))[:48]
        claim = _words(normalized)
        if second in claim and first not in claim:
            return f"{match.group('b')} {normalized}"
        if first in claim and second not in claim:
            return f"{match.group('attr')} {match.group('a')} {normalized}"
    return f"{quote} {normalized}"


def expand_compound(claim: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a classification joined by ', not' from a separate absence assertion."""
    quote = str(claim.get("source_quote") or "").strip()
    match = _COMPOUND.match(quote)
    if not match:
        return [claim]
    left = match.group("left").strip()
    right_body = match.group("right").strip().rstrip(".")
    right = f"not {right_body}"
    if left not in quote or right not in quote:
        return [claim]
    subject = left.split(" is ")[0].strip()
    shared = {key: value for key, value in claim.items() if key not in {"source_quote", "normalized_claim", "missing_context"}}
    return [
        {**shared, "source_quote": left, "normalized_claim": _sentence(left), "missing_context": []},
        {
            **shared,
            "source_quote": right,
            "normalized_claim": f"{subject} is not {right_body}.",
            "missing_context": [],
        },
    ]


def _finalize(article: str, claim: dict[str, Any]) -> dict[str, Any]:
    quote = claim["source_quote"]
    source_ranges = _exact_quote_ranges(article, quote)
    if not source_ranges:
        source_ranges = _typographic_quote_ranges(article, quote)
        if len(source_ranges) > 1:
            raise ValueError("source_quote is ambiguous in the article")
    quote_was_adjusted = bool(source_ranges) and article[slice(*source_ranges[0])] != quote
    if quote_was_adjusted:
        quote = article[slice(*source_ranges[0])]
    normalized = correct_attribution(quote, str(claim.get("normalized_claim") or quote).strip())
    normalization_gaps: list[str] = []
    normalization_notes: list[str] = []
    source_numbers = _NUMBER.findall(quote)
    claim_numbers = _NUMBER.findall(normalized)
    if any(number not in claim_numbers for number in source_numbers):
        normalization_gaps.append("规范化声明未保留原文中的数字或年份")
    source_ordinals = {match.group(0).lower() for match in _ORDINAL.finditer(quote)}
    claim_ordinals = {match.group(0).lower() for match in _ORDINAL.finditer(normalized)}
    if claim_ordinals - source_ordinals:
        normalized = quote
        normalization_notes.append("模型添加了原文没有的顺序限定，已回退为原文措辞")
    source_modality = {match.group(0).lower() for match in _MODALITY.finditer(quote)}
    normalized_modality = {match.group(0).lower() for match in _MODALITY.finditer(normalized)}
    if source_modality - normalized_modality:
        normalized = quote
        normalization_notes.append(_NORMALIZATION_FALLBACK)
    # The quote above is now the exact article substring, so preserve the
    # resolver's ranges (including duplicate exact occurrences).
    offset = source_ranges[0][0] if source_ranges else 0
    resolved_claims = [resolve_exam(article, span[0], normalized) for span in source_ranges] or [normalized]
    if all(value == resolved_claims[0] for value in resolved_claims):
        normalized = resolved_claims[0]
    elif _EXAM_REF.search(normalized):
        normalization_gaps.append("原文引用在多处出现，指代对象无法唯一确定")
    focused = _scope_text(quote, normalized)
    references = [resolve_document_ref(article, span[0], quote, focused) for span in source_ranges] or [resolve_document_ref(article, offset, quote, focused)]
    if all(reference == references[0] for reference in references):
        document_ref, ref_state = references[0]
    else:
        document_ref, ref_state = None, "ambiguous"
    fact_type = infer_fact_type(claim, focused, "")
    scope = infer_scope(focused, "", document_ref, ref_state, fact_type)
    missing = relevant_gaps(focused, normalized, list(claim.get("missing_context") or []))
    for gap in normalization_gaps:
        if gap not in missing:
            missing.append(gap)
    gap = price_gap(quote, normalized)
    if (
        gap
        and scope != "text_analysis"
        and fact_type not in {"opinion", "prediction"}
        and gap not in missing
    ):
        missing.append(gap)
    if ref_state == "ambiguous" and _DOC_GAP not in missing:
        missing.append(_DOC_GAP)
    for gap in claim_context_gaps(quote, normalized):
        if gap not in missing:
            missing.append(gap)
    if _quote_has_unattributed_speaker(article, source_ranges, quote, normalized):
        if _SPEAKER_GAP not in missing:
            missing.append(_SPEAKER_GAP)
    absence = is_absence(focused, "")
    status, status_reason = assign_status(fact_type, scope, missing, absence, ref_state)
    return {
        "source_quote": quote,
        "source_ranges": source_ranges,
        "normalized_claim": normalized,
        "fact_type": fact_type,
        "category": fact_type,
        "verification_scope": scope,
        "status": status,
        "status_reason": status_reason,
        "document_ref": document_ref,
        "missing_context": missing,
        "normalization_notes": normalization_notes,
        "extraction_notes": ([_QUOTE_TYPOGRAPHY_NOTE] if quote_was_adjusted else []) + _quote_position_notes(article, source_ranges),
    }


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Simple deterministic sentence spans used only for extraction review diagnostics."""
    spans = []
    start = 0
    for match in re.finditer(r"[.!?](?:[\"'”’)]*)\s+|\n+", text):
        end = match.end()
        left, right = start, end
        while left < right and text[left].isspace(): left += 1
        while right > left and text[right - 1].isspace(): right -= 1
        if left < right: spans.append((left, right))
        start = end
    left, right = start, len(text)
    while left < right and text[left].isspace(): left += 1
    while right > left and text[right - 1].isspace(): right -= 1
    if left < right: spans.append((left, right))
    return spans


def uncovered_sentences(article: str, claims: list[dict[str, Any]], rejected: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    covered = []
    for claim in claims:
        covered.extend(claim.get("source_ranges") or find_quote_ranges(article, str(claim.get("source_quote") or "")))
    for item in rejected or []:
        claim = item.get("claim")
        if isinstance(claim, dict):
            covered.extend(find_quote_ranges(article, str(claim.get("source_quote") or "")))
    result = []
    for start, end in sentence_spans(article):
        if not any(a < end and start < b for a, b in covered):
            result.append({"start": start, "end": end, "text": article[start:end]})
    return result


def _quote_position_notes(article: str, ranges: list[list[int]]) -> list[str]:
    for start, _ in ranges:
        sentence_start = max((m.end() for m in re.finditer(r"[.!?](?:[\"'”’)]*)\s+|\n+", article[:start])), default=0)
        prefix = article[sentence_start:start].strip()
        # A quote or bracket immediately before the quoted words is a sentence
        # boundary marker, not omitted source text. Newlines are boundaries in
        # sentence_spans above; flattened headings remain intentionally opaque.
        if prefix.strip("\"'“”‘’「」『』([{<（【《〈"):
            return ["引文从句中开始，可能省略了该句前面的限定或分句"]
    return []


def _quote_has_unattributed_speaker(article: str, ranges: list[list[int]], quote: str, normalized: str) -> bool:
    if not ranges:
        return False
    for start, end in ranges:
        quote_start, quote_end = start, end
        # The model may quote the delimiters as part of source_quote. Normalize
        # only the bounds used for detection; preserve the stored source quote.
        opening = {'"', "'", "“", "‘", "「", "『", "(", "[", "{"}
        closing = {'"', "'", "”", "’", "」", "』", ")", "]", "}"}
        while quote_start < quote_end and article[quote_start].isspace():
            quote_start += 1
        while quote_end > quote_start and article[quote_end - 1].isspace():
            quote_end -= 1
        includes_open = quote_start < quote_end and article[quote_start] in opening
        includes_close = quote_start < quote_end and article[quote_end - 1] in closing
        left = quote_start
        right = quote_end
        while left > 0 and article[left - 1].isspace(): left -= 1
        while right < len(article) and article[right].isspace(): right += 1
        is_quoted = (includes_open and includes_close) or (
            left > 0 and right < len(article)
            and article[left - 1] in opening and article[right] in closing
        )
        if not is_quoted:
            continue
        # Look at the quote's own sentence and its immediate neighbors. This
        # catches continued direct quotes attributed in the same paragraph
        # without borrowing an unrelated speaker from elsewhere in the article.
        spans = sentence_spans(article)
        sentence_index = next((i for i, (a, b) in enumerate(spans) if a < quote_end and quote_start < b), None)
        if sentence_index is None:
            continue
        paragraph_break = article.rfind("\n\n", 0, quote_start)
        paragraph_start = paragraph_break + 2 if paragraph_break >= 0 else 0
        paragraph_end = article.find("\n\n", quote_end)
        if paragraph_end < 0:
            paragraph_end = len(article)
        context_spans = spans[max(0, sentence_index - 1):sentence_index + 2]
        context = " ".join(article[max(a, paragraph_start):min(b, paragraph_end)] for a, b in context_spans)
        if not _ATTRIBUTION.search(context):
            continue
        if _ATTRIBUTION.search(normalized):
            continue
        named_attributions = _NAMED_ATTRIBUTION.finditer(context)
        speakers = [match.group("speaker") or match.group("according_speaker") for match in named_attributions]
        if not any(speaker and speaker in normalized for speaker in speakers):
            return True
    return False


def causal_context_gaps(source_quote: str, normalized_claim: str) -> list[str]:
    """Flag causal claims whose antecedent or outcome is too vague to check."""
    text = f"{source_quote} {normalized_claim}"
    if not _CAUSAL.search(text):
        return []
    gaps = []
    if _VAGUE_ACTION_REFERENCE.search(normalized_claim):
        gaps.append("需要明确因果关系所指的行动或变化")
    if _VAGUE_OUTCOME.search(text) and not _MEASURE.search(text):
        gaps.append("需要说明因果结果的具体对象或衡量指标")
        gaps.append("需要提供所述行动导致该结果的因果依据")
    return gaps


def headline_context_gaps(source_quote: str, normalized_claim: str) -> list[str]:
    """Ask for the referent behind vague result words in attributed headline claims."""
    source_results = {match.group(0).lower() for match in _HEADLINE_RESULT.finditer(source_quote)}
    normalized = normalized_claim.lower()
    if _HEADLINE_ATTR.search(source_quote) and source_results and not any(term in normalized for term in source_results):
        return [_HEADLINE_GAP]
    return []


def claim_context_gaps(source_quote: str, normalized_claim: str) -> list[str]:
    """Context checks shared by fresh extraction and restored saved claims."""
    gaps = causal_context_gaps(source_quote, normalized_claim)
    for gap in headline_context_gaps(source_quote, normalized_claim):
        if gap not in gaps:
            gaps.append(gap)
    return gaps


def infer_fact_type(claim: dict[str, Any], quote: str, normalized: str) -> str:
    blob = f"{quote} {normalized}"
    if _META.search(blob):
        return "checkable_fact"
    if _INTENT.search(blob):
        return "intention"
    if _EVALUATION.search(blob):
        return "opinion"
    if _FUTURE.search(blob) or claim.get("category") == "prediction" or claim.get("fact_type") == "prediction":
        return "prediction"
    if claim.get("category") == "opinion" or claim.get("fact_type") == "opinion":
        return "opinion"
    return "checkable_fact"


def infer_scope(quote: str, normalized: str, document_ref: str | None, ref_state: str, fact_type: str) -> str:
    blob = f"{quote} {normalized}"
    if _META.search(blob):
        return "text_analysis"
    if fact_type in {"intention", "opinion", "prediction"}:
        return "external_fact"
    if is_headline_wording(quote, normalized) and not _REPORT_REF.search(blob) and not is_absence(quote, normalized):
        return "external_fact"
    if document_ref or ref_state == "ambiguous" or _DOCUMENT.search(_scope_text(quote, normalized)):
        return "document_content"
    if is_absence(quote, normalized) and _DOC_NOUN.search(blob):
        return "document_content"
    return "external_fact"


def assign_status(
    fact_type: str,
    scope: str,
    missing: list[str],
    absence: bool,
    ref_state: str,
) -> tuple[str, str | None]:
    if scope == "text_analysis":
        return "skipped", "text_analysis is not sent to external retrieval in this version"
    if fact_type == "opinion":
        return "skipped", "opinion is not given a factual check"
    if fact_type == "prediction":
        return "skipped", "prediction is not given a factual check"
    if missing or ref_state == "ambiguous":
        return "needs_context", None
    if scope == "document_content" and absence and ref_state == "resolved":
        return "pending_document", "The complete document is required before an omission can be checked."
    return "ready", None


def relevant_gaps(quote: str, normalized: str, missing: list[str]) -> list[str]:
    """Keep only gaps in what this claim asserts. A wording claim is not a performance claim."""
    if not is_headline_wording(quote, normalized):
        return missing
    context_gap_needed = bool(headline_context_gaps(quote, normalized))
    return [
        item for item in missing
        if not _WORDING_GAP.search(item) and (item != _HEADLINE_GAP or context_gap_needed)
    ]


def is_headline_wording(quote: str, normalized: str) -> bool:
    return bool(_HEADLINE_ATTR.search(f"{quote} {normalized}"))


def is_absence(quote: str, normalized: str) -> bool:
    return bool(_ABSENCE.search(quote) or _ABSENCE.search(normalized))


def resolve_exam(article: str, offset: int, normalized: str) -> str:
    if not _EXAM_REF.search(normalized):
        return normalized
    names: list[str] = []
    for match in _EXAM_NAME.finditer(article[:offset]):
        if match.group(1) not in names:
            names.append(match.group(1))
    if len(names) != 1:
        return normalized
    return re.sub(r"\bthat exam\b", names[0], normalized, count=1, flags=re.IGNORECASE)


def resolve_document_ref(article: str, offset: int, quote: str, normalized: str) -> tuple[str | None, str]:
    """Bind a document phrase only when the input has one candidate. Otherwise do not guess."""
    blob = normalized
    items = _document_index(article)
    end = offset + len(quote)
    if _PRONOUN.match(quote.strip()) or _PRONOUN.match(normalized.strip()):
        paragraph = article.rfind("\n\n", 0, offset)
        start = 0 if paragraph < 0 else paragraph + 2
        return _one_name([item for item in items if item["intro"] and start <= item["start"] < offset])
    note_refs = [match.group(0) for match in _NOTE_REF.finditer(blob)]
    if note_refs:
        notes = [item for item in items if item["kind"] == "note" and item["intro"] and item["start"] < end]
        named = [item for item in notes if item["name"].lower() in blob.lower()]
        if named:
            return _one_name(named)
        if any(ref.lower() == "the note" for ref in note_refs):
            return _one_name(notes)
        return None, "ambiguous"
    if _REPORT_REF.search(blob):
        reports = [item for item in items if item["kind"] == "report" and item["intro"] and item["start"] < end]
        return _one_name(reports)
    if _DOC_REF.search(blob):
        return _one_name([item for item in items if item["intro"] and item["start"] < end])
    return None, "none"


def _document_index(article: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for match in _NAMED_REPORT.finditer(article):
        items.append({"kind": "report", "name": f"{match.group('name')} report", "start": match.start(), "intro": True})
    for match in _NAMED_NOTE.finditer(article):
        raw = re.sub(r"\s+", " ", match.group("name"))
        name = f"the {raw[4:]}" if raw.lower().startswith("the ") else raw
        items.append({"kind": "note", "name": name, "start": match.start(), "intro": name.lower() != "the note"})
    return items


def _one_name(items: list[dict[str, Any]]) -> tuple[str | None, str]:
    names: list[str] = []
    for item in items:
        if item["name"] not in names:
            names.append(item["name"])
    if len(names) == 1:
        return names[0], "resolved"
    return None, "ambiguous"


def needs_full_document(claim: dict[str, Any]) -> bool:
    if claim.get("status") == "pending_document":
        return True
    quote = str(claim.get("source_quote") or "")
    normalized = str(claim.get("normalized_claim") or "")
    scope = claim.get("verification_scope") or ""
    return scope == "document_content" and is_absence(quote, normalized) and bool(claim.get("document_ref"))


def _sentence(text: str) -> str:
    body = text.strip().rstrip(".")
    return body[:1].upper() + body[1:] + "."


def _needs_full_document(claim: dict[str, Any]) -> bool:
    return needs_full_document(claim)


def _price_clause(quote: str, normalized: str) -> str:
    match = _AND_ATTRIBUTION.match(quote.strip())
    if match and _words(match.group("b"))[:48] in _words(normalized):
        return match.group("b")
    return normalized


def _words(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def _number(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": f"c{index}", **claim} for index, claim in enumerate(claims, start=1)]


def _best_match(gold: dict[str, Any], predicted: list[dict[str, Any]], used: set[int]) -> int | None:
    quote = gold["source_quote"]
    for index, pred in enumerate(predicted):
        if index in used:
            continue
        other = pred.get("source_quote") or ""
        if quote in other or other in quote:
            return index
    required = [item.lower() for item in gold.get("required") or []]
    for index, pred in enumerate(predicted):
        if index in used or not required:
            continue
        blob = f"{pred.get('source_quote', '')} {pred.get('normalized_claim', '')}".lower()
        if all(item in blob for item in required):
            return index
    return None


def _meaning_problems(gold: dict[str, Any], pred: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if pred.get("category") != gold.get("category"):
        problems.append(f"category {pred.get('category')!r} instead of {gold.get('category')!r}")
    blob = str(pred.get("normalized_claim") or "").lower()
    for item in gold.get("required") or []:
        if item.lower() not in blob:
            problems.append(f"dropped limiter {item!r}")
    return problems


def _assess_category(category: Any) -> str:
    if category == "checkable_fact":
        return "factual"
    if category in {"opinion", "prediction", "needs_context"}:
        return category
    return "factual"


def _print_article(article_id: str, extracted: dict[str, Any], gold: list[dict[str, Any]] | None) -> None:
    print(f"\n== {article_id} ==")
    for claim in extracted["claims"]:
        ref = claim.get("document_ref") or "-"
        print(f"{claim['id']}  {claim['fact_type']}  {claim['verification_scope']}  {claim['status']}  {ref}")
        print(f"  quote: {claim['source_quote']}")
        print(f"  claim: {claim['normalized_claim']}")
        if claim.get("status_reason"):
            print(f"  reason: {claim['status_reason']}")
        if claim["missing_context"]:
            print(f"  missing: {', '.join(claim['missing_context'])}")
    for item in extracted["rejected"]:
        print(f"rejected: {item['errors']}")
    if gold is None:
        return
    aligned = align_to_gold(gold, extracted["claims"])
    for row in aligned["rows"]:
        detail = ",".join(row.get("problems") or [])
        print(f"review {row['gold_id']}: {row['kind']} {row.get('predicted_id') or ''} {detail}".rstrip())
    for extra in aligned["extras"]:
        print(f"review extra: {extra['predicted_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract claims from a passage and validate quotes.")
    parser.add_argument("--text", type=Path, help="One article file.")
    parser.add_argument("--sample", type=Path, default=SAMPLE / "gold.json")
    parser.add_argument("--out", type=Path, default=SAMPLE / "last_extract.json")
    parser.add_argument("--reprocess", action="store_true", help="Reapply claim rules to --out using --text. No model call.")
    parser.add_argument("--verify", type=Path, help="Manual evidence packet for one gold claim.")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-super-vl:free")
    parser.add_argument("--base-url", default="openrouter")
    args = parser.parse_args()
    if args.reprocess:
        if args.text is None:
            parser.error("--reprocess requires --text")
        text = args.text.read_text(encoding="utf-8")
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        extracted = {"claims": prepare_claims(text, saved["claims"]), "rejected": saved.get("rejected") or []}
        _print_article(args.text.stem, extracted, None)
        args.out.write_text(json.dumps(extracted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        return

    complete = default_complete(args.model, args.base_url)

    if args.text:
        text = args.text.read_text(encoding="utf-8")
        extracted = extract_claims(text, complete)
        _print_article(args.text.stem, extracted, None)
        args.out.write_text(json.dumps(extracted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        return

    articles = load_gold(args.sample)
    saved = []
    for article in articles:
        extracted = extract_claims(article["text"], complete)
        _print_article(article["id"], extracted, article["claims"])
        saved.append({"article_id": article["id"], "extraction": extracted, "alignment": align_to_gold(article["claims"], extracted["claims"])})

    if args.verify:
        packet = json.loads(args.verify.read_text(encoding="utf-8"))
        article = next(item for item in articles if item["id"] == packet["article_id"])
        claim = next(item for item in article["claims"] if item["id"] == packet["claim_id"])
        public = {key: claim[key] for key in ("id", "source_quote", "normalized_claim", "category", "missing_context")}
        verdict = verify_claim(public, packet, complete)
        print(f"\nverify {claim['id']}: {verdict['status']} {verdict['verdict']} {verdict['validation_status']}")
        print(f"  {verdict['explanation']}")
        saved.append({"verify": {"claim_id": claim["id"], "result": verdict}})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
