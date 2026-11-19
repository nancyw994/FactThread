"""Bounded, coordinate-preserving Lucy extraction diagnostic (no search/judge)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "pipeline"), str(ROOT / "web")]

import articles
import extract_claims
import llm_client

MODEL = "gemini-3.8-flash"
CAP = 8192
TIMEOUT_SECONDS = 120
MAX_503_RETRIES = 2
TARGET_CHARS = 2000
MIN_CHARS = 1500
MAX_CHARS = 2500


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Split flattened source at sentence ends, preserving every original character."""
    split_re = re.compile(r"(?<=[.!?])[\"’”\u2019)]*\s+(?=[A-Z“‘\"0-9])")
    month_abbrev = {"Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."}
    boundaries = []
    for match in split_re.finditer(text):
        before = text[:match.start()].rstrip()
        after = text[match.end():]
        if any(before.endswith(month) for month in month_abbrev) and re.match(r"\d", after):
            continue
        boundaries.append(match.end())
    ends = boundaries + [len(text)]
    ranges = []
    start = 0
    for end in ends:
        if end > start:
            ranges.append((start, end))
            start = end
    return ranges


def make_chunks(text: str) -> list[dict]:
    sentences = sentence_ranges(text)
    chunks = []
    first_sentence = 0
    while first_sentence < len(sentences):
        start = sentences[first_sentence][0]
        end = start
        cursor = first_sentence
        while cursor < len(sentences):
            candidate_end = sentences[cursor][1]
            size = candidate_end - start
            if size > MAX_CHARS and end > start:
                break
            if size > MAX_CHARS and end == start:
                raise ValueError(f"A single sentence exceeds {MAX_CHARS} characters")
            end = candidate_end
            cursor += 1
            if end - start >= TARGET_CHARS:
                break
        if end - start < MIN_CHARS and cursor < len(sentences):
            while cursor < len(sentences) and sentences[cursor][1] - start <= MAX_CHARS:
                end = sentences[cursor][1]
                cursor += 1
        chunk_text = text[start:end]
        chunks.append({
            "index": len(chunks) + 1,
            "start": start,
            "end": end,
            "chars": len(chunk_text),
            "sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            "text": chunk_text,
        })
        first_sentence = cursor
    if "".join(text[item["start"]:item["end"]] for item in chunks) != text:
        raise AssertionError("Chunk spans do not reconstruct the source exactly")
    return chunks


def add_run_metrics(results: dict, chunk_count: int) -> None:
    completed = sum(item.get("status") == "completed" for item in results["chunks"])
    attempted = len(results["chunks"])
    attempts = [a for item in results["chunks"] for a in item.get("attempts", [])]
    results["attempted_chunks"] = attempted
    results["completed_chunks"] = completed
    results["chunk_completion_rate"] = {"completed": completed, "total": chunk_count}
    results["total_elapsed_seconds"] = sum((a.get("latency_seconds") or 0) + (a.get("retry_delay_seconds") or 0)
                                           for a in attempts)
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(a.get("usage") or {}).get(key) for a in attempts]
        usable = [value for value in values if isinstance(value, (int, float))]
        usage[key] = sum(usable) if usable else None
    results["total_usage"] = usage
    results["reference_claim_coverage"] = "not_evaluable_no_completed_extraction"
    results["normalization_fidelity"] = "not_evaluable_no_completed_extraction"


def main() -> int:
    args = sys.argv[1:]
    mode = args[0] if args else "first"
    if mode not in {"plan", "first", "remaining"}:
        raise SystemExit("usage: chunk_lucy.py [plan|first|remaining] [run_dir]")

    llm_client.configure_providers(allow_openrouter_fallback=False, allow_paid=False)
    route = llm_client.provider_chain(MODEL)
    expected = [{"provider": "gemini", "model": MODEL, "base_url": llm_client.GEMINI_BASE_URL,
                 "key_name": "GEMINI_API_KEY", "models": [MODEL], "route_reason": "gemini_requested"}]
    if route != expected or llm_client.ALLOW_PAID or llm_client.ALLOW_OPENROUTER_FALLBACK:
        raise RuntimeError("Refusing provider/model/fee-policy change")
    _, key_status = llm_client.lookup_key("GEMINI_API_KEY")
    if key_status != "ok":
        raise RuntimeError(f"GEMINI_API_KEY is {key_status}; no model request was made")
    if articles.ARTICLE_MODEL != MODEL or articles.EXTRACTION_MAX_COMPLETION_TOKENS != CAP:
        raise RuntimeError("Model or output cap differs from the requested configuration")

    source = next(item for item in json.loads((BASE / "articles.json").read_text(encoding="utf-8"))
                  if item["id"] == "lucy")
    text = source["text"]
    full_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = make_chunks(text)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(args[1]) if len(args) > 1 else BASE / "after" / "chunked-lucy" / now
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    plan_path = run_dir / "chunk_plan.json"
    resume_existing = mode == "remaining" or (mode == "first" and len(args) > 1 and plan_path.is_file())
    if resume_existing:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["input_sha256"] != full_sha:
            raise RuntimeError("Source text changed; refusing to resume this chunk plan")
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        if mode == "remaining":
            first = results["chunks"][0]
            if first.get("status") != "completed":
                raise RuntimeError("First chunk did not complete; refusing to run later chunks")
    else:
        if run_dir.exists():
            raise RuntimeError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        plan = {
            "article_id": "lucy", "source_url": source["url"], "input_chars": len(text),
            "input_sha256": full_sha, "truncated": False, "provider": "gemini", "model": MODEL,
            "prompt_version": "sha256:" + hashlib.sha256(articles.PROMPT.read_bytes()).hexdigest(),
            "output_cap": CAP, "allow_paid": False, "allow_openrouter_fallback": False,
            "strategy": "sentence-boundary chunks from flattened one-paragraph article",
            "context_rule": "previous up to 500 chars is read-only; only quotes wholly inside core chunk are retained",
            "chunks": [{k: v for k, v in chunk.items() if k != "text"} for chunk in chunks],
        }
        write_json(plan_path, plan)
        old_path = BASE / "after" / "prompt-revision" / "20261118T203906.803306Z" / "lucy.json"
        if old_path.is_file():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            old["evaluation_classification"] = "run_failure"
            old["counts_as_extraction_omission"] = False
            old["failure_note"] = "Whole-article model request timed out before extraction returned; no omission judgment is made."
            write_json(old_path, old)
        results = {
            "article_id": "lucy", "input_chars": len(text), "input_sha256": full_sha,
            "provider": "gemini", "model": MODEL,
            "prompt_version": "sha256:" + hashlib.sha256(articles.PROMPT.read_bytes()).hexdigest(),
            "output_cap": CAP, "allow_paid": False, "allow_openrouter_fallback": False,
            "whole_article_baseline": {"status": "run_failure", "elapsed_seconds": 600.0867786658928,
                                       "http_status": None, "usage": None, "parsed_items": None,
                                       "counts_as_extraction_omission": False},
            "chunks": [], "status": "planned",
        }
        write_json(run_dir / "results.json", results)
        if mode == "plan":
            print(json.dumps({"run_dir": str(run_dir), "chunks": plan["chunks"]}, ensure_ascii=False, indent=2))
            return 0

    chunk_indexes = [0] if mode == "first" else list(range(1, len(chunks)))
    results_path = run_dir / "results.json"
    for index in chunk_indexes:
        chunk = chunks[index]
        previous_sentences = [span for span in sentence_ranges(text) if span[1] <= chunk["start"]]
        context_start = chunk["start"]
        for sentence_start, sentence_end in reversed(previous_sentences):
            if chunk["start"] - sentence_start > 500:
                break
            context_start = sentence_start
        context = text[context_start:chunk["start"]]
        core = text[chunk["start"]:chunk["end"]]
        model_input = context + core
        local_core_start = len(context)
        prompt_text = articles.PROMPT.read_text(encoding="utf-8")
        yaml_spec = yaml.safe_dump({"instructions": prompt_text}, sort_keys=False, allow_unicode=True)
        user_message = json.dumps({"article": model_input, "repair_errors": []}, ensure_ascii=False)
        messages = llm_client._messages(yaml_spec, user_message)
        chunk_record = {
            "chunk_index": index + 1, "core_start": chunk["start"], "core_end": chunk["end"],
            "core_chars": len(core), "context_start": context_start, "context_chars": len(context),
            "input_chars": len(model_input), "input_sha256": hashlib.sha256(model_input.encode("utf-8")).hexdigest(),
            "status": "running", "attempts": [], "accepted_claims": [], "rejected_claims": [],
        }
        if index == 0:
            results["chunks"] = [chunk_record]
        else:
            results["chunks"].append(chunk_record)
        results["status"] = "partial"
        write_json(results_path, results)
        original_client = llm_client._client

        def bounded_client(target: dict):
            from openai import OpenAI
            return OpenAI(api_key=target["api_key"], base_url=target["base_url"],
                          max_retries=0, timeout=TIMEOUT_SECONDS)

        llm_client._client = bounded_client
        response = None
        for attempt_index in range(MAX_503_RETRIES + 1):
            attempt = {"sequence": attempt_index + 1, "started_at": datetime.now(timezone.utc).isoformat(),
                       "timeout_seconds": TIMEOUT_SECONDS,
                       "request_parameters": {"model": MODEL, "temperature": 0.0, "response_format": articles.EXTRACTION_RESPONSE_FORMAT,
                                              "max_completion_tokens": CAP, "messages": messages}}
            chunk_record["attempts"].append(attempt)
            write_json(results_path, results)
            started = time.perf_counter()
            try:
                response = llm_client.ask_llm_usage(
                    {"instructions": prompt_text}, user_message, model=MODEL, temperature=0.0,
                    response_format=articles.EXTRACTION_RESPONSE_FORMAT, max_completion_tokens=CAP,
                )
                attempt.update({"status": "completed", "http_status": response.get("http_status"),
                                "finish_reason": response.get("finish_reason"), "usage": response.get("usage"),
                                "served_model": response.get("model"), "latency_seconds": response.get("elapsed_s")})
                break
            except Exception as exc:
                provider_attempts = getattr(exc, "attempts", []) or []
                provider_attempt = provider_attempts[-1] if provider_attempts else {}
                status = provider_attempt.get("http_status")
                elapsed = time.perf_counter() - started
                attempt.update({"status": "failed", "http_status": status,
                                "error_type": provider_attempt.get("exception_type") or type(exc).__name__,
                                "error_code": provider_attempt.get("error_code"),
                                "error_body": provider_attempt.get("error_body"),
                                "rate_limit_headers": provider_attempt.get("rate_limit_headers") or {},
                                "provider_attempts": provider_attempts,
                                "error": str(llm_client._redact_provider_value(str(exc)))[:1600],
                                "latency_seconds": elapsed})
                if status == 503 and attempt_index < MAX_503_RETRIES:
                    delay = min(2 ** (attempt_index + 1), 8)
                    attempt["retry_delay_seconds"] = delay
                    write_json(results_path, results)
                    time.sleep(delay)
                    continue
                chunk_record["status"] = "request_failed"
                chunk_record["failure_classification"] = "run_failure"
                chunk_record["counts_as_extraction_omission"] = False
                break
        llm_client._client = original_client

        if response is not None:
            raw = response.get("content") or ""
            raw_path = run_dir / f"chunk-{index + 1}-raw.txt"
            raw_path.write_text(raw, encoding="utf-8")
            chunk_record.update({"raw_reply_file": str(raw_path.relative_to(ROOT)), "raw_reply_chars": len(raw)})
            try:
                parsed = articles._parse_extract(raw)
            except Exception as exc:
                chunk_record["status"] = "parse_failed"
                chunk_record["parse_error"] = str(exc)
                parsed = None
            if parsed is not None:
                accepted, rejected = extract_claims._split(model_input, parsed)
                retained = []
                for claim in accepted:
                    ranges = extract_claims.find_quote_ranges(model_input, claim["source_quote"])
                    if len(ranges) != 1:
                        rejected.append({**claim, "errors": ["quote_position_ambiguous_in_chunk_input"]})
                        continue
                    local_start, local_end = ranges[0]
                    if local_start < local_core_start or local_end > len(model_input):
                        rejected.append({**claim, "errors": ["quote_in_read_only_context_or_crosses_core_boundary"]})
                        continue
                    claim = {**claim, "source_start": chunk["start"] + local_start - local_core_start,
                             "source_end": chunk["start"] + local_end - local_core_start}
                    if text[claim["source_start"]:claim["source_end"]] != claim["source_quote"]:
                        rejected.append({**claim, "errors": ["global_source_coordinate_mismatch"]})
                        continue
                    retained.append(claim)
                chunk_record["parsed_items"] = len(parsed.get("claims") or [])
                chunk_record["accepted_claims"] = retained
                chunk_record["rejected_claims"] = rejected
                chunk_record["parsed_items_accepted"] = len(retained)
                chunk_record["rejected_items"] = len(rejected)
                if response.get("finish_reason") in {"length", "max_tokens"}:
                    chunk_record["status"] = "truncated"
                elif retained:
                    chunk_record["status"] = "completed"
                else:
                    chunk_record["status"] = "validation_failed"
            if chunk_record["status"] == "completed":
                full_source_claims = []
                for prior in results["chunks"]:
                    full_source_claims.extend(prior.get("accepted_claims", []))
                deduped = []
                seen = set()
                for claim in full_source_claims:
                    key = (claim.get("source_start"), claim.get("source_end"), claim.get("normalized_claim"))
                    if key not in seen:
                        seen.add(key)
                        deduped.append(claim)
                results["accepted_claims"] = deduped
                results["status"] = "completed" if len(results["chunks"]) == len(chunks) else "partial"
        if chunk_record["status"] != "completed":
            results["status"] = "partial" if any(c.get("status") == "completed" for c in results["chunks"]) else "run_failure"
            results["stopped_after_chunk"] = index + 1
            add_run_metrics(results, len(chunks))
            write_json(results_path, results)
            break
        results["status"] = "completed" if len(results["chunks"]) == len(chunks) else "partial"
        add_run_metrics(results, len(chunks))
        write_json(results_path, results)

    print(json.dumps({"run_dir": str(run_dir), "status": results.get("status"),
                      "completed_chunks": sum(1 for c in results["chunks"] if c.get("status") == "completed"),
                      "total_chunks": len(chunks), "results_file": str(results_path.relative_to(ROOT))},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
