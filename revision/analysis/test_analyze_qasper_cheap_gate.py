"""Behavior tests for the QASPER cheap-gate feasibility analysis."""

import unittest

import numpy as np
import pandas as pd

from revision.analysis.analyze_qasper_cheap_gate import (
    CHEAP_FEATURE_COLUMNS,
    answer_context_token_coverage,
    answer_number_support,
    assert_no_split_overlap,
    extract_deployment_features,
    fit_cheap_quality_model,
    predict_quality,
)


class QasperCheapGateTests(unittest.TestCase):
    def test_features_exclude_gold_and_llm_judge_columns(self) -> None:
        forbidden = {
            "gold_answer",
            "gold_answers_json",
            "evidence_recall",
            "pre_gate_faith_score",
            "trust_score",
            "raw_hallucination",
            "passed_hallucination",
        }

        self.assertTrue(forbidden.isdisjoint(CHEAP_FEATURE_COLUMNS))

    def test_answer_context_coverage(self) -> None:
        self.assertAlmostEqual(
            answer_context_token_coverage("Alpha beta", "Only alpha appears."),
            0.5,
        )
        self.assertEqual(answer_context_token_coverage("", "context"), 0.0)

    def test_numeric_support(self) -> None:
        self.assertAlmostEqual(
            answer_number_support("Scores were 12 and 15.", "The score was 12."),
            0.5,
        )
        self.assertEqual(answer_number_support("No numeric claim.", "context"), 0.0)

    def test_development_and_heldout_ids_do_not_overlap(self) -> None:
        development = pd.DataFrame({"question_id": ["q1", "q2"]})
        heldout = pd.DataFrame({"question_id": ["q3", "q4"]})
        assert_no_split_overlap(development, heldout)

        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_no_split_overlap(
                development,
                pd.DataFrame({"question_id": ["q2", "q5"]}),
            )

    def test_model_scores_are_finite_and_reproducible(self) -> None:
        rows = pd.DataFrame(
            [
                self._row(index, answer_f1=value)
                for index, value in enumerate(
                    [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
                )
            ]
        )
        features = extract_deployment_features(rows)
        first_model = fit_cheap_quality_model(features, rows["answer_f1"])
        second_model = fit_cheap_quality_model(features, rows["answer_f1"])
        first = predict_quality(first_model, features)
        second = predict_quality(second_model, features)

        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((0.0 <= first) & (first <= 1.0)).all())
        np.testing.assert_allclose(first, second)

    @staticmethod
    def _row(index: int, answer_f1: float) -> dict:
        return {
            "question_id": f"q-{index}",
            "question": "What score was reported?" if index % 2 else "Why?",
            "generated_answer": f"The reported score was {index}.",
            "retrieved_context": f"Results report score {index}.",
            "average_similarity": 0.50 + index * 0.03,
            "routed_granularity": "DETAIL" if index % 2 else "CONCEPT",
            "answer_f1": answer_f1,
        }


if __name__ == "__main__":
    unittest.main()
