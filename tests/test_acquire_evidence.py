"""Local fixture evidence. No search API and no model calls."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import acquire_evidence as evidence  # noqa: E402
import extract_claims as claims  # noqa: E402


FIXTURES = ROOT / "data" / "dev" / "fixtures" / "harbor" / "sources.json"
CLAIMS = ROOT / "data" / "dev" / "extract" / "last_extract.json"


def _claim(claim_id: str, quote: str, **extra) -> dict:
    row = {
        "id": claim_id,
        "source_quote": quote,
        "normalized_claim": quote,
        "category": "checkable_fact",
        "missing_context": [],
        "ready_for_retrieval": True,
        "skip_reason": None,
        "verification_scope": "external_fact",
    }
    row.update(extra)
    return row


class HarborFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = evidence.load_fixtures(FIXTURES)
        self.by_id = {document["source_id"]: document for document in self.documents}
        article = (ROOT / "data/dev/extract/harbor.txt").read_text(encoding="utf-8")
        raw = json.loads(CLAIMS.read_text(encoding="utf-8"))["claims"]
        self.claims = claims.prepare_claims(article, raw)
        self.rows = {row["claim_id"]: row for row in evidence.acquire(self.claims, self.documents)["claims"]}

    def row(self, snippet: str) -> dict:
        claim = next(item for item in self.claims if snippet in item["source_quote"])
        return self.rows[claim["id"]]

    def test_fixtures_are_marked_fictional(self) -> None:
        kinds = {document["document_type"] for document in self.documents}
        self.assertEqual(kinds, {"report", "announcement", "headline"})
        for document in self.documents:
            self.assertTrue(document["fictional"])
            self.assertTrue(document["is_complete"])
            self.assertIn("FICTIONAL FIXTURE", document["body"])

    def test_score_and_baseline_locate_report_sentences(self) -> None:
        report = self.by_id["src_harbor_report"]["body"]
        for snippet, needle in (("The report says Model X scored 62%", "62%"), ("human baseline for that exam", "55%")):
            row = self.row(snippet)
            self.assertEqual(row["status"], "ready")
            self.assertEqual(row["source"]["source_id"], "src_harbor_report")
            self.assertEqual(row["document_ref"], "Harbor-Math report")
            self.assertIsNone(row["document"])
            passage = row["passages"][0]
            self.assertIn(needle, passage["quote"])
            self.assertTrue(evidence.quote_matches(report, passage["quote"], passage["start"], passage["end"]))
            self.assertNotIn("verdict", row)

    def test_report_omissions_receive_the_complete_report(self) -> None:
        report = self.by_id["src_harbor_report"]["body"]
        for snippet in (
            "does not report results on other reasoning tasks",
            "never names the system",
            "not a result in the report",
        ):
            row = self.row(snippet)
            self.assertEqual(row["status"], "ready", snippet)
            self.assertEqual(row["passages"], [])
            self.assertFalse(row["proves_absence"])
            document = row["document"]
            self.assertEqual(document["source_id"], "src_harbor_report")
            self.assertTrue(document["is_complete"])
            self.assertEqual(document["end"], len(report))
            self.assertTrue(evidence.quote_matches(report, document["quote"], document["start"], document["end"]))

    def test_note_claims_do_not_use_the_report(self) -> None:
        for snippet in ("does not say Model X leads", "does not say the $5 rate", "short on purpose"):
            row = self.row(snippet)
            self.assertEqual(row["status"], "source_not_found", snippet)
            self.assertIsNone(row["source"])
            self.assertIn("undecided", row["reason"])
            self.assertNotEqual((row.get("source") or {}).get("source_id"), "src_harbor_report")

    def test_blocked_claims_explain_why_they_are_not_checked(self) -> None:
        price = self.row("standard tier costs $5")
        self.assertEqual(price["status"], "needs_context")
        self.assertIsNone(price["source"])
        self.assertIn("product or service", price["reason"])
        self.assertEqual(self.row("I think Harbor-Math")["status"], "skipped")
        self.assertEqual(self.row("By 2027")["status"], "skipped")
        self.assertEqual(self.row("is the writer's forecast")["status"], "skipped")
        self.assertEqual(self.row("is a preference")["status"], "skipped")
        vague = self.row("no named baseline in the document")
        self.assertEqual(vague["status"], "needs_context")
        self.assertIn("which document", vague["reason"])

    def test_announcement_and_headline_are_separate_sources(self) -> None:
        posted = self.row("Harbor Lab posted")
        headlines = self.row("headlines called the gain")
        self.assertEqual(posted["status"], "ready")
        self.assertEqual(headlines["status"], "ready")
        self.assertEqual(posted["source"]["source_id"], "src_harbor_announcement")
        self.assertEqual(headlines["source"]["source_id"], "src_harbor_headlines")
        for row, source_id in ((posted, "src_harbor_announcement"), (headlines, "src_harbor_headlines")):
            body = self.by_id[source_id]["body"]
            passage = row["passages"][0]
            self.assertTrue(evidence.quote_matches(body, passage["quote"], passage["start"], passage["end"]))

    def test_shifted_span_does_not_match(self) -> None:
        body = self.by_id["src_harbor_report"]["body"]
        quote = "Model X scored 62% on Harbor-Math."
        start = body.find(quote)
        self.assertTrue(evidence.quote_matches(body, quote, start, start + len(quote)))
        self.assertFalse(evidence.quote_matches(body, quote, start, start + len(quote) - 1))
        self.assertFalse(evidence.quote_matches(body, quote + " ", start, start + len(quote)))


class MissingSourceTests(unittest.TestCase):
    def test_unknown_claim_is_not_a_refutation(self) -> None:
        documents = evidence.load_fixtures(FIXTURES)
        claim = _claim("x1", "Acme published a batch price of $3 per million tokens on May 2, 2024.")
        row = evidence.acquire_one(claim, documents)
        self.assertEqual(row["status"], "source_not_found")
        self.assertIsNone(row["source"])
        self.assertEqual(row["passages"], [])
        self.assertIn("undecided", row["reason"])
        self.assertNotIn("verdict", row)

    def test_incomplete_report_stays_pending_for_an_absence_claim(self) -> None:
        excerpt = {
            "source_id": "src_excerpt",
            "title": "Fictional Harbor-Math excerpt",
            "document_type": "report",
            "is_complete": False,
            "fictional": True,
            "body": "FICTIONAL FIXTURE.\n\nModel X scored 62% on Harbor-Math.",
        }
        excerpt["document_ref"] = "North-Math report"
        documents = [evidence._check_document(excerpt)]
        claim = _claim(
            "x2",
            "The report does not say Model X leads every reasoning test.",
            verification_scope="document_content",
            document_ref="North-Math report",
            status="pending_document",
        )
        row = evidence.acquire_one(claim, documents)
        self.assertEqual(row["status"], "pending_document")
        self.assertEqual(row["passages"], [])
        self.assertIsNone(row["document"])
        self.assertFalse(row["proves_absence"])
        self.assertEqual(row["source"]["is_complete"], False)

    def test_positive_claim_can_use_an_incomplete_excerpt(self) -> None:
        excerpt = evidence._check_document(
            {
                "source_id": "src_excerpt",
                "title": "Fictional Harbor-Math excerpt",
                "document_type": "report",
                "document_ref": "North-Math report",
                "is_complete": False,
                "fictional": True,
                "body": "FICTIONAL FIXTURE.\n\nModel X scored 62% on Harbor-Math.",
            }
        )
        claim = _claim(
            "x3",
            "The report says Model X scored 62% on Harbor-Math.",
            verification_scope="document_content",
            document_ref="North-Math report",
            status="ready",
        )
        row = evidence.acquire_one(claim, [excerpt])
        self.assertEqual(row["status"], "ready")
        self.assertIn("62%", row["passages"][0]["quote"])
        self.assertTrue(
            evidence.quote_matches(
                excerpt["body"],
                row["passages"][0]["quote"],
                row["passages"][0]["start"],
                row["passages"][0]["end"],
            )
        )

    def test_unmarked_fixture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evidence._check_document(
                {
                    "source_id": "src_real",
                    "title": "Unmarked page",
                    "document_type": "report",
                    "is_complete": True,
                    "fictional": False,
                    "body": "Model X scored 62% on Harbor-Math.",
                }
            )


if __name__ == "__main__":
    unittest.main()
