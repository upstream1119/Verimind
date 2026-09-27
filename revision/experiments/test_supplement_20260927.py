"""Behavior checks for the supplemental evaluation protocol."""
import importlib.util
import pathlib
import unittest


MODULE = pathlib.Path(__file__).with_name("supplement_20260927.py")


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), "Supplemental metric implementation is missing")
        spec = importlib.util.spec_from_file_location("supplement", MODULE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_recall_deduplicates_gold_and_retrieved_units(self):
        result = self.m.ranking_metrics([{"a"}, {"a"}, set(), {"b"}], {"a", "b"})
        self.assertEqual(result, {"recall_at_3": 0.5, "recall_at_5": 1.0, "mrr_at_5": 1.0})

    def test_reciprocal_rank_is_rank_sensitive(self):
        self.assertEqual(self.m.ranking_metrics([set(), {"a"}], {"a"})["mrr_at_5"], 0.5)
        self.assertEqual(self.m.ranking_metrics([set()], {"a"})["recall_at_5"], 0.0)

    def test_no_gold_is_missing_not_zero(self):
        self.assertTrue(all(v is None for v in self.m.ranking_metrics([], set()).values()))

    def test_gate_ablation_preserves_natural_refusal(self):
        for policy in ["fixed_512", "multi", "full", "no_routing", "no_judge"]:
            self.assertEqual(self.m.decision("Information not found.", 10, 10, policy), "ABSTAIN")

    def test_gate_and_judge_ablations_are_distinct(self):
        self.assertEqual(self.m.decision("An answer.", 8, 1, "full"), "REJECT")
        self.assertEqual(self.m.decision("An answer.", 8, 1, "no_judge"), "ANSWER")
        self.assertEqual(self.m.decision("An answer.", 1, 1, "multi"), "ANSWER")

    def test_threshold_boundary_is_inclusive(self):
        self.assertEqual(self.m.decision("Answer.", 5, 5, "full"), "ANSWER")

    def test_refusal_confusion_counts_false_rejections(self):
        result = self.m.refusal_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual([result[x] for x in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 1])
        self.assertEqual(result["f1"], 0.5)

    def test_undefined_precision_is_not_perfect(self):
        self.assertIsNone(self.m.refusal_metrics([False], [False])["precision"])

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(self.m.percentile([1, 2, 3, 4], .95), 3.85)

    def test_judge_does_not_silently_default_invalid_response(self):
        with self.assertRaises(ValueError):
            self.m.parse_evaluation('{"supported": "yes", "correct": true}')
        with self.assertRaises(ValueError):
            self.m.parse_evaluation('not a JSON response')
        self.assertEqual(self.m.parse_evaluation('{"supported": false, "correct": true}')["supported"], False)

    def test_exact_evidence_normalization_is_not_bag_of_words(self):
        self.assertEqual(self.m.normalize_evidence("A \n B"), "ab")
        self.assertNotEqual(self.m.normalize_evidence("A B"), self.m.normalize_evidence("B A"))

    def test_answer_f1_retains_repeated_token_counts(self):
        self.assertAlmostEqual(self.m.answer_f1("red red blue", ["red blue"]), .8)

    def test_smoke_covers_routes_and_unanswerability(self):
        rows = [{"question": "What is this?", "dataset": "qasper", "is_unanswerable": False},
                {"question": "how many", "dataset": "qasper", "is_unanswerable": False},
                {"question": "why", "dataset": "qasper", "is_unanswerable": False},
                {"question": "Unknown?", "dataset": "qasper", "is_unanswerable": True}]
        self.assertEqual(len(self.m.smoke_sample(rows, 4)), 4)
        self.assertEqual({self.m.route_query(r["question"], "qasper") for r in self.m.smoke_sample(rows, 4)},
                         {"DETAIL", "CONCEPT", "CONTEXT"})

    def test_parent_reconstruction_preserves_overlap(self):
        import json
        class Collection:
            def get(self, include):
                return {"documents": ["abcdefgh", "fghijkl"], "metadatas": [
                    {"_node_content": json.dumps({"metadata": {"paper_id": "paper"},
                        "relationships": {"1": {"node_id": "parent"}}, "start_char_idx": start})}
                    for start in [0, 5]]}
        self.assertEqual(self.m.Runtime.parent_documents(Collection())["paper"]["parent"], "abcdefghijkl")


if __name__ == "__main__":
    unittest.main()
