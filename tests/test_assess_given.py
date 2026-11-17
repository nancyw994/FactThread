"""Deterministic checks for evidence-given assessment. No model calls."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import assess_given as ag  # noqa: E402


HARBOR = (
    "The 2024 Harbor benchmark paper measured Model X on Harbor-Math. "
    "On Harbor-Math, Model X scored 62%, above the human baseline of 55% for that exam. "
    "The paper does not report results on other reasoning tasks."
)
SCORE = "On Harbor-Math, Model X scored 62%, above the human baseline of 55% for that exam."


def packet() -> dict:
    return {
        "claim": "The 2024 Harbor benchmark paper reports that Model X scored 62% on Harbor-Math.",
        "category": "factual",
        "claim_date": "2024-06-01",
        "evidence_as_of": "2026-11-15",
        "sources": [
            {
                "id": "src_harbor",
                "canonical_url": "https://example.org/harbor-2024",
                "title": "Harbor benchmark 2024",
                "published_at": "2024-03-01",
                "extraction_status": "ok",
                "text": HARBOR,
            }
        ],
        "passages": [{"id": "ev_harbor_score", "source_id": "src_harbor", "quote": SCORE}],
    }


def good_output() -> dict:
    return {
        "verdict": "supported",
        "explanation": "The stored passage states the 62% Harbor-Math score.",
        "explanation_citations": ["ev_harbor_score"],
        "evidence": [
            {
                "id": "ev_harbor_score",
                "source_id": "src_harbor",
                "quote": SCORE,
                "relation": "entailment",
                "canonical_url": "https://evil.example/invented",
            }
        ],
        "unresolved_questions": [],
        "needs_review": False,
    }


class ValidatePacketTests(unittest.TestCase):
    def test_sample_packets_resolve_and_hide_gold(self) -> None:
        cases = ag.load_sample(ROOT / "data" / "dev" / "evidence_given.json")
        self.assertGreaterEqual(len(cases), 6)
        for case in cases:
            packet = ag.packet_from_case(case)
            self.assertNotIn("gold", packet)
            self.assertNotIn("verdict", packet)
            self.assertEqual(ag.validate_packet(packet), [])

    def test_ambiguous_quote_is_rejected(self) -> None:
        item = packet()
        item["sources"][0]["text"] = "alpha beta alpha"
        item["passages"][0]["quote"] = "alpha"
        errors = ag.validate_packet(item)
        self.assertTrue(any("ambiguous" in error for error in errors))

    def test_offsets_must_match_quote(self) -> None:
        item = packet()
        start = HARBOR.index(SCORE)
        item["passages"][0]["start_offset"] = start
        item["passages"][0]["end_offset"] = start + len(SCORE)
        self.assertEqual(ag.validate_packet(item), [])
        item["passages"][0]["end_offset"] = start + 4
        self.assertTrue(any("offsets" in error for error in ag.validate_packet(item)))

    def test_non_http_source_is_rejected(self) -> None:
        item = packet()
        item["sources"][0]["canonical_url"] = "file:///tmp/secret"
        self.assertTrue(any("canonical_url" in error for error in ag.validate_packet(item)))


class ValidateAssessmentTests(unittest.TestCase):
    def test_exact_quote_passes(self) -> None:
        self.assertEqual(ag.validate_assessment(packet(), good_output()), [])

    def test_unknown_citation_fails(self) -> None:
        output = good_output()
        output["explanation_citations"] = ["ev_missing"]
        output["evidence"][0]["id"] = "ev_missing"
        errors = ag.validate_assessment(packet(), output)
        self.assertTrue(any("not in this run" in error for error in errors))

    def test_quote_mismatch_fails(self) -> None:
        output = good_output()
        output["evidence"][0]["quote"] = SCORE.replace("62%", "90%")
        errors = ag.validate_assessment(packet(), output)
        self.assertTrue(any("quote does not match" in error for error in errors))

    def test_old_verdict_labels_are_rejected(self) -> None:
        for label in ("contradicted", "misleading", "unverifiable", "partially_supported"):
            output = good_output()
            output["verdict"] = label
            errors = ag.validate_assessment(packet(), output)
            self.assertTrue(any("not a product label" in error for error in errors), label)

    def test_supported_requires_cited_entailment(self) -> None:
        output = good_output()
        output["evidence"][0]["relation"] = "neutral"
        errors = ag.validate_assessment(packet(), output)
        self.assertTrue(any("no cited entailment" in error for error in errors))


class AssessTests(unittest.TestCase):
    def test_publishable_result_uses_stored_source(self) -> None:
        calls = {"n": 0}

        def complete(item, repair):
            calls["n"] += 1
            self.assertIsNone(repair)
            return good_output()

        result = ag.assess(packet(), complete, run_id="run_ok")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["validation_status"], "passed")
        self.assertEqual(result["evidence"][0]["quote"], SCORE)
        self.assertEqual(result["sources"][0]["canonical_url"], "https://example.org/harbor-2024")
        self.assertNotIn("evil.example", json.dumps(result["sources"]))
        self.assertNotIn("evil.example", json.dumps(result["evidence"]))

    def test_repair_then_accept(self) -> None:
        seen: list[list[str] | None] = []

        def complete(item, repair):
            seen.append(repair)
            if repair is None:
                bad = good_output()
                bad["evidence"][0]["quote"] = "not the stored passage"
                return bad
            return good_output()

        result = ag.assess(packet(), complete, run_id="run_repair")
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[0])
        self.assertTrue(seen[1])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["debug"]["rejected"][0]["response"]["evidence"][0]["quote"], "not the stored passage")

    def test_failed_validation_does_not_keep_a_strong_verdict(self) -> None:
        def complete(item, repair):
            bad = good_output()
            bad["verdict"] = "supported"
            bad["evidence"][0]["quote"] = "invented passage"
            return bad

        result = ag.assess(packet(), complete, run_id="run_bad")
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["verdict"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["error_code"], "validation_failed")
        self.assertEqual(result["needs_review"], True)
        preserved = json.dumps(result["debug"]["rejected"])
        self.assertIn("supported", preserved)
        self.assertIn("invented passage", preserved)

    def test_model_exception_is_not_a_refutation(self) -> None:
        def complete(item, repair):
            raise TimeoutError("upstream timed out")

        result = ag.assess(packet(), complete, run_id="run_down")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "model_error")
        self.assertIsNone(result["verdict"])
        self.assertNotEqual(result["verdict"], "refuted")

    def test_opinion_and_prediction_skip_the_model(self) -> None:
        def complete(item, repair):
            raise AssertionError("model should not run")

        for category in ("opinion", "prediction"):
            item = packet()
            item["category"] = category
            result = ag.assess(item, complete, run_id=f"run_{category}")
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["verdict"])
            self.assertIn(category, result["explanation"])
            self.assertEqual(result["validation_status"], "passed")

    def test_invalid_packet_skips_the_model(self) -> None:
        def complete(item, repair):
            raise AssertionError("model should not run")

        item = packet()
        item["sources"][0]["extraction_status"] = "failed"
        result = ag.assess(item, complete, run_id="run_input")
        self.assertEqual(result["error_code"], "invalid_input")
        self.assertIsNone(result["verdict"])

    def test_model_view_hides_page_text_and_urls(self) -> None:
        view = ag._model_view(packet())
        encoded = json.dumps(view)
        self.assertIn(SCORE, encoded)
        self.assertNotIn(HARBOR, encoded)
        self.assertNotIn("canonical_url", encoded)
        self.assertNotIn("https://example.org", encoded)

    def test_sample_gold_stays_out_of_the_packet_copy(self) -> None:
        case = ag.load_sample(ROOT / "data" / "dev" / "evidence_given.json")[0]
        original = copy.deepcopy(case)
        ag.packet_from_case(case)
        self.assertEqual(case, original)


if __name__ == "__main__":
    unittest.main()
