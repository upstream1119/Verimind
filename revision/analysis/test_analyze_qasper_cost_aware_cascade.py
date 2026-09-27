"""Behavior tests for the QASPER cost-aware verification cascade."""

import unittest

import pandas as pd

from revision.analysis.analyze_qasper_cost_aware_cascade import (
    CascadePolicy,
    bootstrap_policy_comparison,
    deterministic_train_calibration_split,
    estimate_verifier_token_budget,
    evaluate_frozen_policy,
    route_cascade,
    search_cascade_policy,
)


class QasperCostAwareCascadeTests(unittest.TestCase):
    def test_cascade_routes_low_middle_and_high_scores(self) -> None:
        policy = CascadePolicy(
            risk_budget=0.50,
            cheap_low=0.30,
            cheap_high=0.70,
            verified_threshold=0.60,
        )

        low = route_cascade(0.20, 0.95, policy)
        middle_pass = route_cascade(0.50, 0.80, policy)
        middle_fail = route_cascade(0.50, 0.40, policy)
        high = route_cascade(0.80, 0.10, policy)

        self.assertEqual((low.decision, low.verifier_called), ("abstain", False))
        self.assertEqual(
            (middle_pass.decision, middle_pass.verifier_called),
            ("answer_verified", True),
        )
        self.assertEqual(
            (middle_fail.decision, middle_fail.verifier_called),
            ("abstain_verified", True),
        )
        self.assertEqual(
            (high.decision, high.verifier_called),
            ("answer_direct", False),
        )

    def test_development_split_is_deterministic_and_disjoint(self) -> None:
        rows = pd.DataFrame(
            {
                "question_id": [f"q-{index}" for index in range(40)],
                "is_unanswerable": [index % 5 == 0 for index in range(40)],
            }
        )

        first_train, first_calibration = deterministic_train_calibration_split(rows)
        second_train, second_calibration = deterministic_train_calibration_split(rows)

        self.assertEqual(first_train["question_id"].tolist(), second_train["question_id"].tolist())
        self.assertEqual(
            first_calibration["question_id"].tolist(),
            second_calibration["question_id"].tolist(),
        )
        self.assertFalse(
            set(first_train["question_id"]) & set(first_calibration["question_id"])
        )
        self.assertEqual(len(first_train) + len(first_calibration), len(rows))

    def test_policy_search_respects_risk_budget_and_prefers_coverage(self) -> None:
        candidates = pd.DataFrame(
            {
                "question_id": ["a", "b", "c", "d"],
                "cheap_score": [0.95, 0.80, 0.50, 0.20],
                "verified_score": [0.90, 0.75, 0.85, 0.10],
                "answer_f1": [1.0, 0.8, 0.7, 0.0],
                "estimated_verifier_tokens": [10, 10, 10, 10],
            }
        )

        policy, calibration_metrics = search_cascade_policy(
            candidates,
            total_questions=4,
            risk_budget=0.20,
        )
        evaluated, frozen_metrics = evaluate_frozen_policy(
            candidates,
            total_questions=4,
            policy=policy,
        )

        self.assertTrue(calibration_metrics["feasible"])
        self.assertLessEqual(frozen_metrics["selective_risk"], 0.20 + 1e-12)
        self.assertEqual(frozen_metrics["accepted_n"], 3)
        self.assertEqual(len(evaluated), len(candidates))

    def test_token_estimate_and_bootstrap_are_reproducible(self) -> None:
        tokens = estimate_verifier_token_budget("one two", "abcde", "answer")
        self.assertGreater(tokens, 0)
        evaluated = pd.DataFrame(
            {
                "question_id": ["a", "b", "c", "d"],
                "accepted": [True, True, False, False],
                "verifier_called": [True, False, True, False],
                "answer_f1": [1.0, 0.5, 0.0, 0.0],
                "estimated_verifier_tokens": [20, 20, 20, 20],
            }
        )

        first = bootstrap_policy_comparison(
            {"cascade": evaluated, "always_verify": evaluated},
            n_bootstrap=50,
            seed=20260918,
        )
        second = bootstrap_policy_comparison(
            {"cascade": evaluated, "always_verify": evaluated},
            n_bootstrap=50,
            seed=20260918,
        )
        pd.testing.assert_frame_equal(first, second)

    def test_infeasible_budget_returns_lowest_risk_policy(self) -> None:
        candidates = pd.DataFrame(
            {
                "question_id": ["a", "b", "c"],
                "cheap_score": [0.9, 0.5, 0.1],
                "verified_score": [0.9, 0.5, 0.1],
                "answer_f1": [0.4, 0.2, 0.0],
                "estimated_verifier_tokens": [10, 10, 10],
            }
        )

        _, metrics = search_cascade_policy(
            candidates,
            total_questions=3,
            risk_budget=0.10,
        )

        self.assertFalse(metrics["feasible"])
        self.assertAlmostEqual(metrics["selective_risk"], 0.60)


if __name__ == "__main__":
    unittest.main()
