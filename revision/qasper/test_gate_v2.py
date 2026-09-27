"""Behavior tests for the QASPER Gate V2 policy."""

import json
import unittest

import pandas as pd

from revision.qasper.analyze_qasper_gate_v2_dev import (
    build_case_decisions,
    build_policy_metrics,
)
from revision.qasper import normalize_qasper_gate_v2_judgments as normalizer
from revision.qasper.gate_v2 import (
    aggregate_judgments,
    parse_gate_judgment,
)
from revision.qasper.normalize_qasper_gate_v2_judgments import (
    build_normalized_judgments,
    normalize_raw_response,
)


class GateV2Tests(unittest.TestCase):
    def test_parse_supported_claim_judgment(self) -> None:
        raw = json.dumps(
            {
                "overall_label": "SUPPORTED",
                "claims": [
                    {
                        "claim": "The method uses a CRF layer.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "SUPPORTED",
                        "evidence": "The model uses a CRF layer.",
                        "evidence_section": "Method",
                        "evidence_owner": "CURRENT_PAPER",
                        "attribution": "CURRENT_PAPER",
                        "issue_types": [],
                        "reason": "Directly supported.",
                    }
                ],
                "summary_reason": "Every material claim is supported.",
            }
        )

        parsed = parse_gate_judgment(raw)

        self.assertEqual(parsed["label"], "supported")
        self.assertEqual(parsed["unsupported_claim_count"], 0)
        self.assertEqual(parsed["uncertain_claim_count"], 0)
        self.assertFalse(parsed["label_repaired"])

    def test_claim_verdict_repairs_inconsistent_top_level_label(self) -> None:
        raw = "```json\n" + json.dumps(
            {
                "overall_label": "SUPPORTED",
                "claims": [
                    {
                        "claim": "The paper uses SVM.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "UNSUPPORTED",
                        "evidence": "No supporting evidence",
                        "evidence_section": "Related Work",
                        "evidence_owner": "PRIOR_WORK",
                        "attribution": "RELATED_WORK",
                        "issue_types": ["attribution_error"],
                        "reason": "SVM is attributed to prior work.",
                    }
                ],
                "summary_reason": "The attribution is unsupported.",
            }
        ) + "\n```"

        parsed = parse_gate_judgment(raw)

        self.assertEqual(parsed["label"], "unsupported")
        self.assertEqual(parsed["unsupported_claim_count"], 1)
        self.assertTrue(parsed["label_repaired"])

    def test_current_paper_claim_cannot_use_prior_work_evidence(self) -> None:
        raw = json.dumps(
            {
                "overall_label": "SUPPORTED",
                "claims": [
                    {
                        "claim": "The paper reports results on German data.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "SUPPORTED",
                        "evidence": "Kluge built a German Web corpus.",
                        "evidence_section": "Argumentation Mining",
                        "evidence_owner": "PRIOR_WORK",
                        "attribution": "RELATED_WORK",
                        "issue_types": [],
                        "reason": "The context mentions German data.",
                    }
                ],
                "summary_reason": "The answer is supported.",
            }
        )

        parsed = parse_gate_judgment(raw)

        claim = parsed["claims"][0]
        self.assertEqual(parsed["label"], "unsupported")
        self.assertEqual(claim["verdict"], "unsupported")
        self.assertIn("attribution_error", claim["issue_types"])
        self.assertTrue(parsed["label_repaired"])

    def test_aggregate_pre_registered_policy(self) -> None:
        self.assertEqual(
            aggregate_judgments(
                ["unsupported", "unsupported", "supported"]
            )["decision"],
            "REJECT",
        )
        self.assertEqual(
            aggregate_judgments(
                ["unsupported", "supported", "supported"]
            )["decision"],
            "WARNING",
        )
        self.assertEqual(
            aggregate_judgments(
                ["uncertain", "supported", "supported"]
            )["decision"],
            "WARNING",
        )
        self.assertEqual(
            aggregate_judgments(
                ["supported", "supported", "supported"]
            )["decision"],
            "ANSWER",
        )

    def test_build_case_decisions_and_policy_metrics(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "item_id": 1,
                    "author_label": "unsupported",
                    "gate_reject": False,
                },
                {
                    "item_id": 2,
                    "author_label": "supported",
                    "gate_reject": False,
                },
                {
                    "item_id": 3,
                    "author_label": "uncertain",
                    "gate_reject": True,
                },
            ]
        )
        judgments = pd.DataFrame(
            [
                {"item_id": 1, "judge": "deepseek", "label": "unsupported"},
                {"item_id": 1, "judge": "qwen", "label": "unsupported"},
                {"item_id": 1, "judge": "glm", "label": "supported"},
                {"item_id": 2, "judge": "deepseek", "label": "unsupported"},
                {"item_id": 2, "judge": "qwen", "label": "supported"},
                {"item_id": 2, "judge": "glm", "label": "supported"},
                {"item_id": 3, "judge": "deepseek", "label": "uncertain"},
                {"item_id": 3, "judge": "qwen", "label": "supported"},
                {"item_id": 3, "judge": "glm", "label": "supported"},
            ]
        )

        decisions = build_case_decisions(scored, judgments)
        self.assertEqual(
            decisions.set_index("item_id")["gate_v2_decision"].to_dict(),
            {1: "REJECT", 2: "WARNING", 3: "WARNING"},
        )

        metrics = build_policy_metrics(decisions).set_index("policy")
        self.assertEqual(
            int(metrics.loc["gate_v2_reject_only", "tp"]),
            1,
        )
        self.assertEqual(
            int(metrics.loc["gate_v2_intervention", "fp"]),
            1,
        )

    def test_offline_normalization_repairs_format_only_errors(self) -> None:
        raw = json.dumps(
            {
                "overall_label": "SUPPORTED",
                "claims": [
                    {
                        "claim": "The paper reports a 5 percent gain.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "SUPPORTED",
                        "evidence": "The gain is 5\\%.",
                        "evidence_section": "Results",
                        "evidence_owner": "CURRENT_PAPER",
                        "attribution": "Results",
                        "issue_types": [],
                        "reason": "The value and metric agree.",
                    }
                ],
                "summary_reason": "The answer is supported.",
            }
        ).replace("\\\\%", "\\%")

        parsed, repair_codes = normalize_raw_response(raw)

        self.assertEqual(parsed["label"], "supported")
        self.assertIn("invalid_json_escape", repair_codes)
        self.assertIn(
            "attribution_from_evidence_owner",
            repair_codes,
        )

    def test_offline_normalization_preserves_double_backslash_latex(self) -> None:
        raw = json.dumps(
            {
                "overall_label": "SUPPORTED",
                "claims": [
                    {
                        "claim": "The dataset includes ISA relations.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "SUPPORTED",
                        "evidence": (
                            "The paper uses ISA$^\\\\uparrow$ and "
                            "ISA$^\\\\downarrow$ probes."
                        ),
                        "evidence_section": (
                            "Example Generation @!START@\\\\textsc{gen}"
                        ),
                        "evidence_owner": "CURRENT_PAPER",
                        "attribution": "Dataset Probes and Construction",
                        "issue_types": [],
                        "reason": "The retrieved evidence lists the probes.",
                    }
                ],
                "summary_reason": "The answer is supported.",
            }
        )

        parsed, repair_codes = normalize_raw_response(raw)

        self.assertEqual(parsed["label"], "supported")
        self.assertNotIn("invalid_json_escape", repair_codes)
        self.assertIn(
            "attribution_from_evidence_owner",
            repair_codes,
        )

    def test_offline_normalization_rejects_truncated_json(self) -> None:
        raw = (
            '{"overall_label":"SUPPORTED","claims":['
            '{"claim":"unfinished"'
        )

        with self.assertRaises(ValueError):
            normalize_raw_response(raw)

    def test_offline_normalization_repairs_multi_value_owner(self) -> None:
        raw = json.dumps(
            {
                "overall_label": "UNSUPPORTED",
                "claims": [
                    {
                        "claim": "The claim cannot be verified.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "UNSUPPORTED",
                        "evidence": "Prior work and current paper text mix.",
                        "evidence_section": "Mixed Sections",
                        "evidence_owner": "PRIOR_WORK; CURRENT_PAPER",
                        "attribution": "BACKGROUND; CURRENT_PAPER",
                        "issue_types": ["evidence_missing"],
                        "reason": "The retrieved context is mixed.",
                    }
                ],
                "summary_reason": "The claim is not supported.",
            }
        )

        parsed, repair_codes = normalize_raw_response(raw)

        self.assertEqual(parsed["label"], "unsupported")
        self.assertEqual(parsed["claims"][0]["evidence_owner"], "mixed")
        self.assertEqual(parsed["claims"][0]["attribution"], "unresolved")
        self.assertIn("evidence_owner_multi_value", repair_codes)
        self.assertIn(
            "attribution_from_evidence_owner",
            repair_codes,
        )

    def test_retry_offline_repair_resolves_retry_parse_error(self) -> None:
        retry_response = json.dumps(
            {
                "overall_label": "SUPPORTED",
                "claims": [
                    {
                        "claim": "The paper reports a supported result.",
                        "claim_subject": "CURRENT_PAPER",
                        "verdict": "SUPPORTED",
                        "evidence": "The result is reported in the paper.",
                        "evidence_section": "Experiments and Results",
                        "evidence_owner": "CURRENT_PAPER",
                        "attribution": "Experiments and Results",
                        "issue_types": [],
                        "reason": "The retrieved section supports it.",
                    }
                ],
                "summary_reason": "The answer is supported.",
            }
        )
        base = {column: "" for column in normalizer.BASE_COLUMNS}
        base.update(
            {
                "item_id": 201,
                "audit_id": "audit-201",
                "judge": "qwen",
                "judge_model": "qwen-plus",
                "status": "parse_error",
                "raw_response": (
                    '{"overall_label":"SUPPORTED","claims":['
                    '{"claim":"unfinished"'
                ),
                "error_message": "truncated",
            }
        )
        retry = base.copy()
        retry.update(
            {
                "raw_response": retry_response,
                "error_message": "invalid attribution enum",
            }
        )
        normalized, report = build_normalized_judgments(
            pd.DataFrame([base]),
            pd.DataFrame([retry]),
            expected_rows=1,
        )

        row = normalized.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["label"], "supported")
        self.assertEqual(
            row["normalization_source"],
            "retry_offline_repair",
        )
        self.assertIn(
            "attribution_from_evidence_owner",
            row["repair_codes"],
        )
        self.assertEqual(
            report.iloc[0]["resolution"],
            "retry_offline_repair",
        )


if __name__ == "__main__":
    unittest.main()
