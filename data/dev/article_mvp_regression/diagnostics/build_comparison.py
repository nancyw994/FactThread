import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/dev/article_mvp_regression"


def canonical(url):
    parsed = urlsplit(url or "")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return host + (parsed.path.rstrip("/") or "/")


def detail(record, source_url):
    source_key = canonical(source_url)
    return {
        "id": record["id"],
        "claim": record.get("claim"),
        "queries": [item.get("query") if isinstance(item, dict) else item for item in record.get("queries") or []],
        "query_statuses": [item.get("status") for item in record.get("queries") or [] if isinstance(item, dict)],
        "search_status": record.get("search_status"),
        "candidate_pages": [
            {"url": item.get("url"), "title": item.get("title"), "query": item.get("query")}
            for item in record.get("candidates") or []
        ],
        "matched_passages": [
            {
                "url": item.get("url"),
                "is_input_page": canonical(item.get("url")) == source_key,
                "source_id": item.get("source_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "text": item.get("quote"),
            }
            for item in record.get("passages") or []
        ],
    }


def metrics(result, source_by_claim):
    rows = result["claims"]
    queries = [q for row in rows for q in row.get("queries") or [] if isinstance(q, dict)]
    passages = [p for row in rows for p in row.get("passages") or []]
    external_claims = set()
    for row in rows:
        source = canonical(source_by_claim[row["id"]])
        if any(canonical(p.get("url")) != source for p in row.get("passages") or []):
            external_claims.add(row["id"])
    return {
        "claims": len(rows),
        "search_ok_claims": sum(row.get("search_status") == "ok" for row in rows),
        "search_unavailable_claims": sum(row.get("search_status") == "search_unavailable" for row in rows),
        "queries": len(queries),
        "query_successes": sum(q.get("status") == "ok" for q in queries),
        "query_errors": sum(q.get("status") != "ok" for q in queries),
        "candidate_pages": sum(len(row.get("candidates") or []) for row in rows),
        "selected_pages": sum(len(row.get("selected") or []) for row in rows),
        "fetched_pages": sum(len(row.get("fetches") or []) for row in rows),
        "claims_with_passages": sum(bool(row.get("passages")) for row in rows),
        "passage_count": len(passages),
        "claims_with_external_page_passage": len(external_claims),
    }


before = json.loads((BASE / "before/retrieval.json").read_text())
after = json.loads((BASE / "after/retrieval.json").read_text())
source_by_id = {item["id"]: item["url"] for item in json.loads((BASE / "articles.json").read_text())}
gold = {item["id"]: item for item in json.loads((BASE / "gold_cases.json").read_text())["cases"]}
source_by_claim = {item["id"]: item.get("article_source_url") for item in gold.values()}
before_rows = {row["id"]: row for row in before["claims"]}
after_rows = {row["id"]: row for row in after["claims"]}
items = []
for item_id in gold:
    source = source_by_id[next(key for key in source_by_id if item_id.startswith(key + "_"))]
    items.append({
        "id": item_id,
        "article_source_url": source,
        "claim": gold[item_id]["claim"],
        "before": detail(before_rows[item_id], source),
        "after": detail(after_rows[item_id], source),
    })
data = {
    "definition": "Passage coverage means the pipeline returned a fetched paragraph for the claim. This does not mean that the paragraph supports the claim or is independent.",
    "before_metrics": metrics(before, source_by_claim),
    "after_metrics": metrics(after, source_by_claim),
    "claims": items,
}
(BASE / "retrieval_comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
