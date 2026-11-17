"""Quote checks and gold-sample alignment. No model calls."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import extract_claims as ex  # noqa: E402


ARTICLE = (
    "Harbor Lab posted its Harbor-Math report on March 4, 2024. "
    "Model X scored 62% on Harbor-Math, and the March 2024 standard tier costs $5 per million tokens. "
    "I think Harbor-Math is the only benchmark that matters this year."
)


def claim(quote: str, category: str = "checkable_fact", normalized: str | None = None) -> dict:
    return {
        "id": "c1",
        "source_quote": quote,
        "normalized_claim": normalized or quote,
        "category": category,
        "missing_context": [],
    }


class QuoteTests(unittest.TestCase):
    def test_gold_quotes_occur_once(self) -> None:
        for article in ex.load_gold():
            self.assertGreaterEqual(len(article["text"].split()), 180, article["id"])
            for item in article["claims"]:
                located = ex.locate_quote(article["text"], item["source_quote"])
                self.assertIsInstance(located, tuple, item["source_quote"])

    def test_missing_and_ambiguous_quotes_fail(self) -> None:
        self.assertIsInstance(ex.locate_quote(ARTICLE, "not in the article"), str)
        doubled = "Model X scored 62% on Harbor-Math. Model X scored 62% on Harbor-Math."
        self.assertIn("ambiguous", ex.locate_quote(doubled, "Model X scored 62% on Harbor-Math"))

    def test_retrieval_gate_is_separate_from_category(self) -> None:
        def complete(text, repair):
            fact = claim("Model X scored 62% on Harbor-Math")
            held = claim("Model X scored 62% on Harbor-Math")
            held["missing_context"] = ["which report states the score"]
            opinion = claim(
                "I think Harbor-Math is the only benchmark that matters this year.",
                "opinion",
            )
            opinion["ready_for_retrieval"] = True
            return {"claims": [fact, held, opinion]}

        rows = ex.extract_claims(ARTICLE, complete)["claims"]
        self.assertEqual(rows[0]["source_quote"], "Model X scored 62% on Harbor-Math")
        self.assertEqual(rows[0]["status"], "ready")
        self.assertEqual(rows[0]["fact_type"], "checkable_fact")
        self.assertEqual(rows[1]["fact_type"], "checkable_fact")
        self.assertEqual(rows[1]["status"], "needs_context")
        self.assertEqual(rows[2]["fact_type"], "opinion")
        self.assertEqual(rows[2]["status"], "skipped")

    def test_and_attribution_is_not_copied_onto_the_second_clause(self) -> None:
        quote = (
            "The report says Model X scored 62% on Harbor-Math, "
            "and the March 2024 standard tier costs $5 per million tokens."
        )
        article = quote + " Harbor Lab posted its Harbor-Math report on March 4, 2024."
        bad = claim(quote, normalized="The report says the March 2024 standard tier costs $5 per million tokens.")

        def complete(text, repair):
            return {"claims": [bad]}

        row = ex.extract_claims(article, complete)["claims"][0]
        self.assertNotIn("report says", row["normalized_claim"].lower())
        self.assertIn("$5", row["normalized_claim"])
        self.assertIn("which product or service the price applies to", row["missing_context"])
        self.assertEqual(row["status"], "needs_context")
        self.assertEqual(row["verification_scope"], "external_fact")
        self.assertIsNone(row["document_ref"])

    def test_unreported_claim_needs_the_full_document(self) -> None:
        item = claim(
            "The authors add that the paper does not report results on other reasoning tasks.",
            normalized="The authors add that the paper does not report results on other reasoning tasks.",
        )
        article = "Ada Lab posted its North-Math report. " + item["source_quote"]

        def complete(text, repair):
            return {"claims": [item]}

        row = ex.extract_claims(article, complete)["claims"][0]
        self.assertEqual(row["document_ref"], "North-Math report")
        self.assertEqual(row["verification_scope"], "document_content")
        self.assertEqual(row["status"], "pending_document")

        def should_not_run(packet, repair):
            raise AssertionError("excerpt is not the full document")

        verdict = ex.verify_claim(row, {"sources": [], "passages": []}, should_not_run)
        self.assertEqual(verdict["verdict"], "insufficient_evidence")

    def test_unnamed_baseline_in_the_document_is_not_retrieval_ready(self) -> None:
        item = claim(
            '"Twice as good" is a comparison with no named baseline in the document.',
            normalized='"Twice as good" is a comparison with no named baseline in the document.',
        )
        article = (
            "Ada Lab posted its North-Math report. The lab's note is separate. "
            + item["source_quote"]
        )

        def complete(text, repair):
            return {"claims": [item]}

        row = ex.extract_claims(article, complete)["claims"][0]
        self.assertEqual(row["status"], "needs_context")
        self.assertIsNone(row["document_ref"])
        self.assertIn("which document is referred to", row["missing_context"])

    def test_opinion_is_not_relabeled(self) -> None:
        errors = ex.check_claim(ARTICLE, claim("I think Harbor-Math is the only benchmark that matters this year.", "opinion"))
        self.assertEqual(errors, [])

    def test_repair_replaces_a_bad_quote(self) -> None:
        calls = {"n": 0}

        def complete(text, repair):
            calls["n"] += 1
            if repair is None:
                return {"claims": [claim("this quote was invented")]}
            return {"claims": [claim("Model X scored 62% on Harbor-Math")]}

        result = ex.extract_claims(ARTICLE, complete)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(result["claims"][0]["source_quote"], "Model X scored 62% on Harbor-Math")
        self.assertEqual(result["claims"][0]["id"], "c1")

    def test_alignment_flags_a_dropped_limiter(self) -> None:
        gold = [claim("Model X scored 62% on Harbor-Math")]
        gold[0]["required"] = ["62%", "Harbor-Math"]
        predicted = [claim("Model X scored 62% on Harbor-Math", normalized="Model X scored 62%.")]
        aligned = ex.align_to_gold(gold, predicted)
        self.assertEqual(aligned["rows"][0]["kind"], "meaning_changed")
        self.assertTrue(any("Harbor-Math" in item for item in aligned["rows"][0]["problems"]))


class RuleTests(unittest.TestCase):
    def test_two_reports_are_not_guessed(self) -> None:
        quote = "The report says the score is 10%."
        article = "Ada Lab posted its North-Math report. Bea Lab posted its South-Math report. " + quote

        def complete(text, repair):
            return {"claims": [claim(quote)]}

        row = ex.extract_claims(article, complete)["claims"][0]
        self.assertIsNone(row["document_ref"])
        self.assertEqual(row["status"], "needs_context")

    def test_headline_wording_does_not_require_the_comparison(self) -> None:
        quote = 'A few headlines called the gain "twice as good,"'
        article = "Ada Lab posted its North-Math report. " + quote
        item = claim(quote, "needs_context")
        item["missing_context"] = ["the system or baseline being compared"]

        def complete(text, repair):
            return {"claims": [item]}

        row = ex.extract_claims(article, complete)["claims"][0]
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["verification_scope"], "external_fact")
        self.assertEqual(row["missing_context"], [])

    def test_intention_is_not_an_opinion(self) -> None:
        quote = "The lab's note is short on purpose."

        def complete(text, repair):
            return {"claims": [claim(quote, "opinion")]}

        row = ex.extract_claims(quote, complete)["claims"][0]
        self.assertEqual(row["fact_type"], "intention")
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["document_ref"], "the lab's note")

    def test_best_is_an_opinion(self) -> None:
        quote = "I think the North-Math exam is the best one this year."

        def complete(text, repair):
            return {"claims": [claim(quote)]}

        row = ex.extract_claims(quote, complete)["claims"][0]
        self.assertEqual(row["fact_type"], "opinion")
        self.assertEqual(row["status"], "skipped")

    def test_ambiguous_causal_outcome_requires_target_metric_and_basis(self) -> None:
        quote = "These actions promoted development."

        def complete(text, repair):
            return {"claims": [claim(quote)]}

        row = ex.extract_claims(quote, complete)["claims"][0]
        self.assertEqual(row["status"], "needs_context")
        self.assertIn("需要明确因果关系所指的行动或变化", row["missing_context"])
        self.assertIn("需要说明因果结果的具体对象或衡量指标", row["missing_context"])
        self.assertIn("需要提供所述行动导致该结果的因果依据", row["missing_context"])

    def test_harbor_sample_matches_the_expected_roles(self) -> None:
        article = (ROOT / "data/dev/extract/harbor.txt").read_text(encoding="utf-8")
        raw = json.loads((ROOT / "data/dev/extract/last_extract.json").read_text(encoding="utf-8"))["claims"]
        rows = ex.prepare_claims(article, raw)

        def find(snippet: str) -> dict:
            matches = [row for row in rows if snippet in row["source_quote"] or snippet in row["normalized_claim"]]
            self.assertEqual(len(matches), 1, snippet)
            return matches[0]

        score = find("The report says Model X scored 62%")
        self.assertEqual(score["status"], "ready")
        self.assertEqual(score["document_ref"], "Harbor-Math report")
        self.assertIn("report says", score["normalized_claim"].lower())
        self.assertNotIn("$5", score["normalized_claim"])

        price = find("standard tier costs $5")
        self.assertEqual(price["status"], "needs_context")
        self.assertNotIn("report says", price["normalized_claim"].lower())
        self.assertIn("which product or service the price applies to", price["missing_context"])

        baseline = find("human baseline for that exam")
        self.assertEqual(baseline["document_ref"], "Harbor-Math report")
        self.assertEqual(baseline["status"], "ready")
        self.assertIn("Harbor-Math", baseline["normalized_claim"])
        self.assertNotIn("that exam", baseline["normalized_claim"].lower())

        unreported = find("does not report results on other reasoning tasks")
        self.assertEqual(unreported["status"], "pending_document")
        self.assertEqual(unreported["document_ref"], "Harbor-Math report")

        self.assertEqual(find("I think Harbor-Math")["status"], "skipped")
        self.assertEqual(find("By 2027")["fact_type"], "prediction")

        headline = find("headlines called the gain")
        self.assertEqual(headline["status"], "ready")
        self.assertEqual(headline["verification_scope"], "external_fact")
        self.assertEqual(headline["missing_context"], [])

        unnamed = find("never names the system")
        self.assertEqual(unnamed["status"], "pending_document")
        self.assertEqual(unnamed["document_ref"], "Harbor-Math report")

        purpose = find("short on purpose")
        self.assertEqual(purpose["fact_type"], "intention")
        self.assertEqual(purpose["status"], "ready")
        self.assertEqual(purpose["document_ref"], "the lab's note")

        self.assertEqual(find("does not say Model X leads")["document_ref"], "the lab's note")
        self.assertEqual(find("does not say the $5 rate")["document_ref"], "the lab's note")

        self.assertEqual(find("is the writer's forecast")["verification_scope"], "text_analysis")
        self.assertEqual(find("is the writer's forecast")["status"], "skipped")
        omitted = find("not a result in the report")
        self.assertEqual(omitted["status"], "pending_document")
        self.assertEqual(omitted["document_ref"], "Harbor-Math report")
        self.assertEqual(find("is a preference")["verification_scope"], "text_analysis")
        self.assertEqual(find("no named baseline in the document")["status"], "needs_context")
        self.assertIsNone(find("no named baseline in the document")["document_ref"])


class VerifyTests(unittest.TestCase):
    def test_verify_uses_supplied_evidence(self) -> None:
        def complete(packet, repair):
            self.assertEqual(packet["claim"], "Model X scored 62% on Harbor-Math.")
            self.assertEqual(packet["passages"][0]["id"], "ev_score")
            return {
                "verdict": "supported",
                "explanation": "The passage states the 62% Harbor-Math score.",
                "explanation_citations": ["ev_score"],
                "evidence": [
                    {
                        "id": "ev_score",
                        "source_id": "src_harbor",
                        "quote": "Model X scored 62% on Harbor-Math",
                        "relation": "entailment",
                    }
                ],
                "unresolved_questions": [],
                "needs_review": False,
            }

        evidence = {
            "evidence_as_of": "2026-11-15",
            "sources": [
                {
                    "id": "src_harbor",
                    "canonical_url": "https://example.org/harbor-2024",
                    "title": "Harbor-Math report",
                    "extraction_status": "ok",
                    "text": "The report says Model X scored 62% on Harbor-Math.",
                }
            ],
            "passages": [
                {"id": "ev_score", "source_id": "src_harbor", "quote": "Model X scored 62% on Harbor-Math"}
            ],
        }
        result = ex.verify_claim(claim("Model X scored 62% on Harbor-Math."), evidence, complete)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["verdict"], "supported")


if __name__ == "__main__":
    unittest.main()
