"""Evidence judgments from local fixtures. No model calls and no confidence scores."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import acquire_evidence as evidence  # noqa: E402
import judge_evidence as judge  # noqa: E402


FIXTURES = ROOT / "data" / "dev" / "fixtures" / "harbor" / "sources.json"
EVIDENCE = ROOT / "data" / "dev" / "extract" / "last_evidence.json"


class HarborJudgmentTests(unittest.TestCase):
    def setUp(self) -> None:
        documents = evidence.load_fixtures(FIXTURES)
        self.bodies = {document["source_id"]: document["body"] for document in documents}
        rows = json.loads(EVIDENCE.read_text(encoding="utf-8"))["claims"]
        judged = judge.judge_evidence(rows, self.bodies)
        self.rows = {row["claim_id"]: row for row in judged["claims"]}
        self.contrast = {case["id"]: case for case in judged["contrast"]}

    def test_ready_score_is_about_the_report_not_the_model(self) -> None:
        row = self.rows["c2"]
        report = self.bodies["src_harbor_report"]
        self.assertEqual(row["verdict"], "supported")
        self.assertEqual(row["validation_status"], "passed")
        self.assertTrue(row["publish"])
        self.assertNotIn("confidence", row)
        ref = row["evidence_refs"][0]
        self.assertTrue(evidence.quote_matches(report, ref["quote"], ref["start"], ref["end"]))
        self.assertIn("62%", ref["quote"])
        self.assertTrue(any("capability" in item for item in row["limitations"]))

    def test_baseline_posting_and_headline_wording(self) -> None:
        self.assertEqual(self.rows["c1"]["verdict"], "supported")
        self.assertEqual(self.rows["c4"]["verdict"], "supported")
        headline = self.rows["c8"]
        self.assertEqual(headline["verdict"], "supported")
        self.assertTrue(any("wording" in item for item in headline["limitations"]))

    def test_keyword_absence_does_not_support_an_omission(self) -> None:
        report = self.bodies["src_harbor_report"]
        for claim_id in ("c5", "c9", "c14"):
            row = self.rows[claim_id]
            self.assertEqual(row["verdict"], "insufficient_evidence", claim_id)
            ref = row["evidence_refs"][0]
            self.assertTrue(ref["full_document"])
            self.assertEqual(ref["quote"], report)
            self.assertIn("missing keyword", row["explanation"].lower())
            self.assertNotIn("paraphrase", row["explanation"].lower())

    def test_authors_add_requires_attribution_and_content(self) -> None:
        row = self.rows["c5"]
        self.assertEqual(row["checks"]["attribution"], "failed")
        self.assertEqual(row["checks"]["content"], "insufficient_evidence")
        report = self.bodies["src_harbor_report"]
        score = "Model X scored 62% on Harbor-Math."
        start = report.find(score)
        authors = judge._ready_row(
            "x_authors",
            "The authors add that Model X scored 62% on Harbor-Math.",
            passages=[{"source_id": "src_harbor_report", "quote": score, "start": start, "end": start + len(score)}],
        )
        authors["source"]["document_type"] = "report"
        judged = judge.judge_one(authors, self.bodies)
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertEqual(judged["checks"]["content"], "passed")
        self.assertEqual(judged["checks"]["attribution"], "failed")

    def test_claims_that_are_not_ready_keep_their_status(self) -> None:
        expected = {
            "c3": "needs_context",
            "c6": "skipped",
            "c7": "skipped",
            "c10": "source_not_found",
            "c13": "skipped",
            "c16": "needs_context",
        }
        for claim_id, status in expected.items():
            row = self.rows[claim_id]
            self.assertIsNone(row["verdict"], claim_id)
            self.assertEqual(row["evidence_status"], status)
            self.assertFalse(row["publish"])

    def test_contrast_cases_match_their_expected_verdicts(self) -> None:
        expected = {
            "score_mismatch": "refuted",
            "date_missing": "insufficient_evidence",
            "other_task_present": "refuted",
            "incomplete_omission": "insufficient_evidence",
            "instruction_in_source": "refuted",
        }
        for case_id, verdict in expected.items():
            case = self.contrast[case_id]
            self.assertEqual(case["expected"], verdict)
            self.assertEqual(case["actual"], verdict)
            self.assertTrue(case["match"])

    def test_a_bad_offset_is_not_a_published_verdict(self) -> None:
        report = self.bodies["src_harbor_report"]
        quote = "Model X scored 62% on Harbor-Math."
        start = report.find(quote)
        row = judge._ready_row(
            "x_bad",
            "The report says Model X scored 62% on Harbor-Math.",
            passages=[{"source_id": "src_harbor_report", "quote": quote, "start": start, "end": start + len(quote) - 1}],
        )
        judged = judge.judge_one(row, self.bodies)
        self.assertIsNone(judged["verdict"])
        self.assertEqual(judged["validation_status"], "failed")
        self.assertFalse(judged["publish"])

    def test_a_short_document_does_not_prove_intention(self) -> None:
        body = "The note is four sentences long."
        row = judge._ready_row(
            "x_intent",
            "The lab's note is short on purpose.",
            source_id="src_note",
            passages=[{"source_id": "src_note", "quote": body, "start": 0, "end": len(body)}],
        )
        row["fact_type"] = "intention"
        judged = judge.judge_one(row, {"src_note": body})
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertTrue(any("intent" in item.lower() or "purpose" in item.lower() for item in judged["limitations"]))

    def test_method_is_a_rule_baseline_without_a_model(self) -> None:
        self.assertEqual(judge.METHOD["kind"], "fixed_rules")
        self.assertFalse(judge.METHOD["uses_model"])
        self.assertTrue(judge.METHOD["uses_keywords"])

    def test_eval_set_keeps_failures_instead_of_hiding_them(self) -> None:
        report = judge.evaluate_expected(ROOT / "data" / "dev" / "judge" / "eval_expected.json")
        failed = {item["id"] for item in report["failures"]}
        self.assertTrue(
            {"synonym_paraphrase", "indirect_comparison", "table_baseline", "year_mismatch", "model_version_mismatch"}
            <= failed
        )
        self.assertGreater(report["citation_validation"]["total"], 0)
        self.assertEqual(report["citation_validation"]["passed"], report["citation_validation"]["total"])
        self.assertEqual(report["false_support"], 2)
        self.assertEqual(
            {item["id"] for item in report["false_support_cases"]},
            {"table_baseline", "model_version_mismatch"},
        )
        self.assertIn("does not mean the verdict is correct", report["citation_validation_note"])


class LlmJudgeTests(unittest.TestCase):
    def _row(self) -> tuple[dict, dict]:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_llm",
            "The report says Model X scored 62% on Harbor-Math.",
            passages=[{"source_id": "src_llm", "quote": body, "start": 0, "end": len(body)}],
        )
        row["source"]["document_type"] = "report"
        return row, {"src_llm": body}

    def _complete(self, verdict: str = "supported"):
        def complete(system: str, user: str) -> str:
            packet = json.loads(user)
            return json.dumps(
                {
                    "verdict": verdict,
                    "explanation": "Judged from the supplied span only.",
                    "evidence_refs": packet["supplied_spans"],
                    "limitations": ["Checked the supplied evidence only."],
                }
            )

        return complete

    def test_missing_key_is_an_error_and_not_a_rule_result(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import llm_client

        original = llm_client.require_provider_key

        def missing(model: str = "") -> None:
            raise RuntimeError("GROQ_API_KEY is missing; OPENROUTER_API_KEY is missing")

        llm_client.require_provider_key = missing
        try:
            with self.assertRaises(RuntimeError) as caught:
                judge.judge_evidence([], {}, judge="llm")
        finally:
            llm_client.require_provider_key = original
        self.assertIn("GROQ_API_KEY", str(caught.exception))
        self.assertIn("OPENROUTER_API_KEY", str(caught.exception))
        self.assertIn("not substituted", str(caught.exception))

    def test_a_model_error_is_not_relabeled(self) -> None:
        row, bodies = self._row()

        def complete(system: str, user: str) -> str:
            raise RuntimeError("model unavailable")

        with self.assertRaises(RuntimeError):
            judge.judge_one(row, bodies, judge="llm", complete=complete, model="stub-model")

    def test_valid_citation_publishes_the_model_and_method(self) -> None:
        row, bodies = self._row()
        judged = judge.judge_evidence(
            [row],
            bodies,
            judge="llm",
            complete=self._complete(),
            model="stub-model",
        )
        self.assertEqual(judged["method"]["kind"], "llm")
        self.assertEqual(judged["method"]["model"], "stub-model")
        self.assertTrue(judged["method"]["uses_model"])
        saved = judged["claims"][0]
        self.assertEqual(saved["verdict"], "supported")
        self.assertTrue(saved["publish"])
        self.assertEqual(saved["validation_status"], "passed")
        self.assertIn("model version", saved["check_scope"] and judge._CHECKLIST)

    def test_bad_citation_is_not_published(self) -> None:
        row, bodies = self._row()
        calls = []

        def complete(system: str, user: str) -> str:
            calls.append(user)
            packet = json.loads(user.split("\n\n", 1)[0])
            span = dict(packet["supplied_spans"][0])
            span["end"] = span["start"]
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "A guess.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        judged = judge.judge_one(row, bodies, judge="llm", complete=complete, model="stub-model")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(judged["verdict"])
        self.assertIsNone(judged["explanation"])
        self.assertEqual(judged["validation_status"], "validation_failed")
        self.assertFalse(judged["publish"])
        self.assertEqual([item["purpose"] for item in judged["judge_requests"]], ["initial", "repair"])
        self.assertTrue(judged["judge_requests"][0]["validation_errors"])
        self.assertTrue(judged["judge_requests"][1]["validation_errors"])
        self.assertIn("A guess.", judged["judge_requests"][0]["raw_reply"])
        self.assertEqual(judged["judge_requests"][0]["model"], "stub-model")
        self.assertEqual(judged["judge_requests"][0]["cost_usd"], "unknown")
        self.assertIsInstance(judged["judge_requests"][0]["elapsed_s"], float)

    def test_omission_is_not_checked_when_the_document_does_not_fit(self) -> None:
        body = "Model X scored 62% on Harbor-Math. " * 20
        row = judge._ready_row(
            "x_omit",
            "The paper does not report results on other reasoning tasks.",
            source_id="src_long",
            document=judge._span("src_long", body, 0, len(body), full=True),
        )
        calls = []

        def complete(system: str, user: str) -> str:
            calls.append(user)
            return "{}"

        judged = judge.judge_one(
            row,
            {"src_long": body},
            judge="llm",
            complete=complete,
            model="stub-model",
            context_limit=40,
        )
        self.assertEqual(calls, [])
        self.assertIsNone(judged["verdict"])
        self.assertFalse(judged["publish"])
        self.assertTrue(judged["check_scope"]["truncated"])
        self.assertFalse(judged["check_scope"]["entered_context"])
        self.assertEqual(judged["check_scope"]["chars_sent"], 0)
        self.assertEqual(judged["check_scope"]["document_chars"], len(body))

    def test_a_complete_omission_document_enters_context_without_truncation(self) -> None:
        body = "The authors add that the paper does not report results on other reasoning tasks."
        row = judge._ready_row(
            "x_full",
            "The paper does not report results on other reasoning tasks.",
            source_id="src_full",
            document=judge._span("src_full", body, 0, len(body), full=True),
        )
        seen = []

        def complete(system: str, user: str) -> str:
            seen.append((system, user))
            packet = json.loads(user)
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The document states the omission.",
                    "evidence_refs": packet["supplied_spans"],
                    "limitations": [],
                }
            )

        judged = judge.judge_one(row, {"src_full": body}, judge="llm", complete=complete, model="stub-model")
        self.assertEqual(len(seen), 1)
        system, user = seen[0]
        self.assertIn(body, user)
        self.assertIn("model version", system)
        self.assertIn("as a refutation", system)
        self.assertIn("Matching numbers alone are not support", user)
        scope = json.loads(user)["check_scope"]
        self.assertFalse(scope["truncated"])
        self.assertTrue(scope["entered_context"])
        self.assertEqual(scope["chars_sent"], len(body))
        self.assertEqual(judged["verdict"], "supported")
        self.assertIn("no truncation", " ".join(judged["limitations"]))

    def test_both_judges_keep_the_same_expected_labels(self) -> None:
        path = ROOT / "data" / "dev" / "judge" / "eval_expected.json"
        before = path.read_text(encoding="utf-8")
        report = judge.compare_expected(path, complete=self._complete(), model="stub-model")
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertEqual(report["fixed_rules"]["false_support"], 2)
        self.assertEqual(report["llm"]["false_support"], 5)
        self.assertEqual(report["llm"]["model"], "stub-model")
        self.assertEqual(report["fixed_rules"]["citation_validation"]["passed"], 12)
        self.assertEqual(report["llm"]["citation_validation"]["passed"], 12)
        failed = {item["id"] for item in report["fixed_rules"]["failures"]}
        self.assertTrue(
            {"synonym_paraphrase", "indirect_comparison", "table_baseline", "year_mismatch", "model_version_mismatch"}
            <= failed
        )

    def test_unconfirmed_attribution_does_not_support_the_whole_claim(self) -> None:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_attr",
            "The authors add that Model X scored 62% on Harbor-Math.",
            passages=[{"source_id": "src_attr", "quote": body, "start": 0, "end": len(body)}],
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_attr": body}, judge="llm", complete=self._complete(), model="stub-model")
        self.assertEqual(judged["attribution_check"], "failed")
        self.assertEqual(judged["content_check"], "supported")
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertNotEqual(judged["verdict"], "supported")

    def test_source_metadata_confirms_report_attribution(self) -> None:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_meta",
            "The report says Model X scored 62% on Harbor-Math.",
            passages=[{"source_id": "src_meta", "quote": body, "start": 0, "end": len(body)}],
        )
        row["source"]["document_type"] = "report"

        def complete(system: str, user: str) -> str:
            packet = json.loads(user)
            self.assertEqual(packet["source"]["document_type"], "report")
            span = packet["supplied_spans"][0]
            return json.dumps(
                {
                    "verdict": "supported",
                    "attribution_check": "failed",
                    "content_check": "supported",
                    "explanation": "The model did not see the source type.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        judged = judge.judge_one(row, {"src_meta": body}, judge="llm", complete=complete, model="stub-model")
        self.assertEqual(judged["attribution_check"], "passed")
        self.assertEqual(judged["content_check"], "supported")
        self.assertEqual(judged["verdict"], "supported")

    def test_confirmed_attribution_restores_a_supportable_verdict(self) -> None:
        body = "Model X scored 62% on Harbor-Math. The human baseline for Harbor-Math is 55%."
        row = judge._ready_row(
            "x_restore",
            "The report says Model X scored higher than the human baseline on Harbor-Math.",
            passages=[{"source_id": "src_restore", "quote": body, "start": 0, "end": len(body)}],
        )
        row["source"]["document_type"] = "report"

        def complete(system: str, user: str) -> str:
            span = json.loads(user)["supplied_spans"][0]
            return json.dumps(
                {
                    "verdict": "insufficient_evidence",
                    "attribution_check": "failed",
                    "content_check": "supported",
                    "explanation": "The content matches, but the model did not confirm the speaker.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        judged = judge.judge_one(row, {"src_restore": body}, judge="llm", complete=complete, model="stub-model")
        self.assertEqual(judged["attribution_check"], "passed")
        self.assertEqual(judged["content_check"], "supported")
        self.assertEqual(judged["verdict"], "supported")

    def test_confirmed_report_attribution_can_support(self) -> None:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_report",
            "The report says Model X scored 62% on Harbor-Math.",
            passages=[{"source_id": "src_report", "quote": body, "start": 0, "end": len(body)}],
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_report": body}, judge="llm", complete=self._complete(), model="stub-model")
        self.assertEqual(judged["attribution_check"], "passed")
        self.assertEqual(judged["content_check"], "supported")
        self.assertEqual(judged["verdict"], "supported")

    def test_complete_metadata_is_not_described_as_an_excerpt(self) -> None:
        body = "The authors add that the paper does not report results on other reasoning tasks."
        row = judge._ready_row(
            "x_scope",
            "The paper does not report results on other reasoning tasks.",
            source_id="src_scope",
            document=judge._span("src_scope", body, 0, len(body), full=True),
        )

        def complete(system: str, user: str) -> str:
            packet = json.loads(user)
            self.assertIn("complete document", packet["document_scope"])
            self.assertNotIn("excerpt", packet["document_scope"])
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The document states it.",
                    "evidence_refs": packet["supplied_spans"],
                    "limitations": ["The supplied excerpt may not include the full document."],
                }
            )

        judged = judge.judge_one(row, {"src_scope": body}, judge="llm", complete=complete, model="stub-model")
        text = " ".join(judged["limitations"]).lower()
        self.assertIn("complete document", text)
        self.assertNotIn("excerpt", text)
        self.assertEqual(judged["document_scope"], judged["check_scope"]["document_scope"])

    def test_unknown_cost_stays_unknown(self) -> None:
        meter = judge.CallMeter()
        meter.add({"model": "openai/gpt-oss-120b", "provider": "groq", "cost_usd": "unknown", "elapsed_s": 1, "usage": {}})
        meter.add({"model": "openai/gpt-oss-120b", "provider": "groq", "cost_usd": 0, "elapsed_s": 1, "usage": {}})
        self.assertEqual(meter.summary()["cost_usd"], "unknown")
        reported = judge.CallMeter()
        reported.add({"model": "m", "provider": "openrouter", "cost_usd": 0, "elapsed_s": 1, "usage": {}})
        self.assertEqual(reported.summary()["cost_usd"], 0)
        self.assertEqual(meter.summary()["requests"][0]["cost_usd"], "unknown")
        self.assertNotEqual(meter.summary()["cost_usd"], 0)

    def test_a_rejected_quote_is_sent_again(self) -> None:
        row, bodies = self._row()
        meter = judge.CallMeter()
        seen = []

        def complete(system: str, user: str) -> str:
            seen.append((meter.purpose, user))
            meter.add(
                {
                    "provider": "groq",
                    "model": "openai/gpt-oss-120b",
                    "elapsed_s": 0.4,
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                    "cost_usd": "unknown",
                }
            )
            packet = json.loads(user.split("\n\nThe previous reply", 1)[0])
            span = dict(packet["supplied_spans"][0])
            if meter.purpose == "initial":
                span["end"] = span["start"]
            return json.dumps(
                {
                    "verdict": "supported",
                    "explanation": "The supplied sentence states it.",
                    "evidence_refs": [span],
                    "limitations": [],
                }
            )

        judged = judge.judge_one(row, bodies, judge="llm", complete=complete, model="stub-model", meter=meter)
        self.assertEqual([item[0] for item in seen], ["initial", "repair"])
        self.assertIn("previous reply", seen[1][1])
        self.assertEqual(judged["verdict"], "supported")
        self.assertEqual(judged["validation_status"], "passed")
        self.assertEqual(judged["judge_requests"][0]["purpose"], "initial")
        self.assertTrue(judged["judge_requests"][0]["validation_errors"])
        self.assertEqual(judged["judge_requests"][1]["purpose"], "repair")
        self.assertEqual(judged["judge_requests"][1]["validation_errors"], [])
        self.assertIn("The supplied sentence states it.", judged["judge_requests"][1]["raw_reply"])
        self.assertEqual(judged["judge_requests"][0]["provider"], "groq")
        self.assertEqual(judged["judge_requests"][0]["model"], "openai/gpt-oss-120b")
        self.assertEqual(judged["judge_requests"][0]["elapsed_s"], 0.4)
        self.assertEqual(judged["judge_requests"][0]["usage"]["prompt_tokens"], 12)
        self.assertEqual(judged["judge_requests"][0]["cost_usd"], "unknown")

    def test_model_version_review_is_recorded(self) -> None:
        data = json.loads((ROOT / "data" / "dev" / "judge" / "eval_expected.json").read_text(encoding="utf-8"))
        case = next(item for item in data["cases"] if item["id"] == "model_version_mismatch")
        self.assertEqual(case["expected"], "insufficient_evidence")
        self.assertIn("Model X-2", case["claim"])
        self.assertIn("Model X-1", case["body"])
        revision = case["revisions"][0]
        self.assertEqual(revision["previous_expected"], "refuted")
        self.assertEqual(revision["new_expected"], "insufficient_evidence")
        self.assertIn("missing", revision["review"])

    def test_a_different_document_speaker_is_not_confirmed(self) -> None:
        body = "Cedar scored 40% on Pike-Logic."
        row = judge._ready_row(
            "x_speaker",
            "The bulletin says Cedar scored 40% on Pike-Logic.",
            passages=[{"source_id": "src_speaker", "quote": body, "start": 0, "end": len(body)}],
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_speaker": body}, judge="llm", complete=self._complete(), model="stub-model")
        self.assertEqual(judged["attribution_check"], "failed")
        self.assertEqual(judged["verdict"], "insufficient_evidence")

    def test_silence_does_not_support_an_omission(self) -> None:
        body = "Cedar scored 40% on Pike-Logic. The tables stop at Pike-Logic."
        row = judge._ready_row(
            "x_silent",
            "The bulletin never names the baseline Cedar was compared with.",
            source_id="src_silent",
            document=judge._span("src_silent", body, 0, len(body), full=True),
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_silent": body}, judge="llm", complete=self._complete(), model="stub-model")
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertIn("comparison", judged["explanation"].lower())

    def test_a_complete_document_omission_does_not_need_an_absence_sentence(self) -> None:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_coverage",
            "The report does not report results on other reasoning tasks.",
            source_id="src_coverage",
            document=judge._span("src_coverage", body, 0, len(body), full=True),
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_coverage": body}, judge="llm", complete=self._complete(), model="stub-model")
        self.assertEqual(judged["verdict"], "supported")
        self.assertNotIn("itself state this omission", judged["explanation"].lower())

    def test_an_author_omission_is_not_supported_by_absence_alone(self) -> None:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_author_omit",
            "The authors add that the paper does not report results on other reasoning tasks.",
            source_id="src_author_omit",
            document=judge._span("src_author_omit", body, 0, len(body), full=True),
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_author_omit": body}, judge="llm", complete=self._complete(), model="stub-model")
        self.assertEqual(judged["attribution_check"], "failed")
        self.assertEqual(judged["verdict"], "insufficient_evidence")

    def test_rule_baseline_reports_uncovered_omission_scope(self) -> None:
        body = "Model X scored 62% on Harbor-Math."
        row = judge._ready_row(
            "x_uncovered",
            "The report does not report results on other reasoning tasks.",
            source_id="src_uncovered",
            document=judge._span("src_uncovered", body, 0, len(body), full=True),
        )
        judged = judge.judge_one(row, {"src_uncovered": body})
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        text = (judged["explanation"] + " " + " ".join(judged["limitations"])).lower()
        self.assertIn("synonym", text)
        self.assertIn("table", text)
        self.assertIn("appendices", text)

    def test_an_excerpt_omission_is_insufficient_without_a_model_call(self) -> None:
        body = "Cedar scored 40% on Pike-Logic."
        row = judge._ready_row(
            "x_excerpt",
            "The report never names another exam.",
            source_id="src_excerpt",
            complete=False,
            document=judge._span("src_excerpt", body, 0, len(body), full=False),
        )
        calls = []

        def complete(system: str, user: str) -> str:
            calls.append(user)
            return "{}"

        judged = judge.judge_one(row, {"src_excerpt": body}, judge="llm", complete=complete, model="stub-model")
        self.assertEqual(calls, [])
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertIn("excerpt", judged["explanation"].lower())

    def test_an_unlinked_baseline_does_not_refute_a_comparison_omission(self) -> None:
        body = "Model X scored 62% on Harbor-Math. The human baseline for Harbor-Math is 55%."
        row = judge._ready_row(
            "x_unlinked",
            "The report never names the system that phrase is comparing against.",
            source_id="src_unlinked",
            document=judge._span("src_unlinked", body, 0, len(body), full=True),
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_unlinked": body}, judge="llm", complete=self._complete("refuted"), model="stub-model")
        self.assertEqual(judged["verdict"], "insufficient_evidence")
        self.assertIn("link", judged["explanation"].lower())
        self.assertNotEqual(judged["verdict"], "refuted")

    def test_a_linked_baseline_can_still_refute_a_comparison_omission(self) -> None:
        body = "Model X was compared against Harbor Baseline and scored 62%."
        row = judge._ready_row(
            "x_linked",
            "The report never names the system that phrase is comparing against.",
            source_id="src_linked",
            document=judge._span("src_linked", body, 0, len(body), full=True),
        )
        row["source"]["document_type"] = "report"
        judged = judge.judge_one(row, {"src_linked": body}, judge="llm", complete=self._complete("refuted"), model="stub-model")
        self.assertEqual(judged["verdict"], "refuted")


if __name__ == "__main__":
    unittest.main()
