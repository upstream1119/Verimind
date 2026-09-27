"""Ensure deployment timing excludes offline judges from baselines."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))


class LatencyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("latency_20260927.py")
        self.assertTrue(path.exists(), "Independent latency runner is missing")
        spec = importlib.util.spec_from_file_location("latency", path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.calls = []
        calls = self.calls
        class Embedding:
            def get_query_embedding(self, query):
                calls.append("embedding")
                return [1.0]
        class Collection:
            def query(self, **kwargs):
                calls.append(("retrieve", kwargs["n_results"]))
                return {"documents": [["Evidence."]], "distances": [[0.3]]}
        class Client:
            def __init__(self, provider):
                self.provider = provider
            def complete(self, prompt, **kwargs):
                calls.append(self.provider)
                return "A substantive answer." if self.provider == "glm" else "1"
        class Runtime:
            embedding = Embedding()
            collections = {("qasper", "CONCEPT"): Collection()}
            def clients(self):
                return {p: Client(p) for p in ("glm", "deepseek", "qwen")}
        self.runtime = Runtime()
        self.item = {"uid": "test", "dataset": "qasper", "question": "Question?", "paper_id": "paper"}

    def test_baseline_calls_no_judge(self):
        result = self.m.run_one(self.runtime, self.item, "fixed_512")
        self.assertEqual(result["decision"], "ANSWER")
        self.assertEqual(self.calls, ["embedding", ("retrieve", 3), "glm"])

    def test_full_calls_only_gate_judge(self):
        result = self.m.run_one(self.runtime, self.item, "full")
        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(self.calls, ["embedding", ("retrieve", 3), "glm", "deepseek"])
        self.assertGreaterEqual(result["latency_sec"], result["generation_sec"])


if __name__ == "__main__":
    unittest.main()
