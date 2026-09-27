"""Check whether gold-independent RRF tie order changes aggregate recall."""
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision/experiments"))
from supplement_20260927 import ranking_metrics


def main():
    records = [json.loads(x) for x in (OUT / "fusion_retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for order in itertools.permutations(("DETAIL", "CONCEPT", "CONTEXT")):
        totals = {("all", 3): [], ("all", 5): [], ("text_retrievable", 3): [], ("text_retrievable", 5): []}
        for record in records:
            if not record["gold_units_n"]:
                continue
            def preferred(node):
                return min(node["appearances"], key=lambda x: (x["route_rank"], order.index(x["route"])))
            ranked = sorted(record["ranked_canonical_parent_spans"],
                            key=lambda x: (-x["rrf_score"], order.index(preferred(x)["route"]),
                                           preferred(x)["route_rank"], x["node_id"]))
            metrics = ranking_metrics([set(x["gold_units"]) for x in ranked[:5]], set(range(record["gold_units_n"])))
            for k in (3, 5):
                totals["all", k].append(metrics[f"recall_at_{k}"])
                if record["text_retrievable"]:
                    totals["text_retrievable", k].append(metrics[f"recall_at_{k}"])
        for (subset, k), scores in totals.items():
            rows.append({"tie_order": "|".join(order), "subset": subset, "k": k,
                         "evaluable_n": len(scores), "mean_fusion_recall": sum(scores) / len(scores)})
    with (OUT / "fusion_tie_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 24
    for subset in ("all", "text_retrievable"):
        for k in (3, 5):
            scores = [r["mean_fusion_recall"] for r in rows if r["subset"] == subset and r["k"] == k]
            assert max(scores) - min(scores) < 1e-12
    print("six tie orders yield identical aggregate Recall@3/5 in both subsets")


if __name__ == "__main__":
    main()
