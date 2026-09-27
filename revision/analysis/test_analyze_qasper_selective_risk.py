"""Behavior tests for QASPER selective-risk analysis."""

import unittest

import pandas as pd

from revision.analysis.analyze_qasper_selective_risk import (
    best_answer_f1,
    build_risk_coverage_curve,
    compute_aurc,
    deterministic_stratified_split,
    normalize_answer,
    replay_gate,
)


class QasperSelectiveRiskTests(unittest.TestCase):
    def test_normalization_matches_official_qasper_behavior(self) -> None:
        self.assertEqual(normalize_answer("The, Model!"), "model")
        self.assertEqual(normalize_answer("  An answer  "), "answer")

    def test_best_answer_f1_uses_best_reference(self) -> None:
        score = best_answer_f1(
            "F-score; Kappa",
            ["Accuracy", "F-score; Kappa"],
        )
        self.assertEqual(score, 1.0)

    def test_split_is_deterministic_and_stratified(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "question_id": f"q-{index:03d}",
                    "is_unanswerable": int(index >= 240),
                }
                for index in range(300)
            ]
        )

        first = deterministic_stratified_split(rows, calibration_size=100)
        second = deterministic_stratified_split(rows, calibration_size=100)

        self.assertEqual(first["split"].tolist(), second["split"].tolist())
        calibration = first[first["split"] == "calibration"]
        test = first[first["split"] == "test"]
        self.assertEqual(len(calibration), 100)
        self.assertEqual(len(test), 200)
        self.assertEqual(int(calibration["is_unanswerable"].sum()), 20)
        self.assertEqual(int(test["is_unanswerable"].sum()), 40)

    def test_gate_replay_never_accepts_natural_abstentions(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "generated_answer": "Information not found.",
                    "trust_score": 9.0,
                },
                {
                    "generated_answer": "Supported answer.",
                    "trust_score": 6.0,
                },
                {
                    "generated_answer": "Low-score answer.",
                    "trust_score": 4.0,
                },
            ]
        )

        replayed = replay_gate(rows, threshold=5.0)

        self.assertEqual(replayed["gate_accepted"].tolist(), [0, 1, 0])
        self.assertEqual(replayed["abstention_source"].tolist(), [
            "generator",
            "answered",
            "gate",
        ])

    def test_risk_coverage_uses_fixed_generated_answers(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "generated_answer": "alpha",
                    "trust_score": 9.0,
                    "answer_f1": 1.0,
                },
                {
                    "generated_answer": "beta",
                    "trust_score": 6.0,
                    "answer_f1": 0.5,
                },
                {
                    "generated_answer": "gamma",
                    "trust_score": 2.0,
                    "answer_f1": 0.0,
                },
                {
                    "generated_answer": "Information not found.",
                    "trust_score": 10.0,
                    "answer_f1": 0.0,
                },
            ]
        )

        curve = build_risk_coverage_curve(rows, thresholds=[0.0, 5.0, 8.0])

        zero = curve[curve["threshold"] == 0.0].iloc[0]
        five = curve[curve["threshold"] == 5.0].iloc[0]
        eight = curve[curve["threshold"] == 8.0].iloc[0]
        self.assertEqual(int(zero["accepted_n"]), 3)
        self.assertAlmostEqual(float(zero["coverage"]), 0.75)
        self.assertAlmostEqual(float(zero["selective_risk"]), 0.5)
        self.assertEqual(int(five["accepted_n"]), 2)
        self.assertAlmostEqual(float(five["selective_risk"]), 0.25)
        self.assertEqual(int(eight["accepted_n"]), 1)
        self.assertAlmostEqual(float(eight["selective_risk"]), 0.0)

    def test_aurc_matches_known_fixture(self) -> None:
        curve = pd.DataFrame(
            {
                "coverage": [0.0, 0.5, 1.0],
                "selective_risk": [0.0, 0.2, 0.6],
            }
        )

        self.assertAlmostEqual(compute_aurc(curve), 0.25)


if __name__ == "__main__":
    unittest.main()
