"""Offline QASPER router-regret analysis from frozen retrieval artifacts."""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "revision/artifacts/supplement_20260927_full"
ORACLE = ROOT / "revision/artifacts/oracle_granularity_20260927"
ROUTES = {"DETAIL": 128, "CONCEPT": 512, "CONTEXT": 1024}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    output = {"evaluable_n": len(rows)}
    for k in (3, 5):
        regrets = [r[f"regret_at{k}"] for r in rows]
        best = [set(r[f"best_granularities_at{k}"].split("|")) for r in rows]
        output[f"mean_regret_at{k}"] = statistics.mean(regrets)
        output[f"median_regret_at{k}"] = statistics.median(regrets)
        output[f"regret_positive_n_at{k}"] = sum(x > 1e-12 for x in regrets)
        output[f"regret_positive_rate_at{k}"] = output[f"regret_positive_n_at{k}"] / len(rows)
        output[f"router_optimal_n_at{k}"] = sum(r[f"router_optimal_at{k}"] for r in rows)
        output[f"router_optimal_rate_at{k}"] = output[f"router_optimal_n_at{k}"] / len(rows)
        output[f"router_unique_best_n_at{k}"] = sum(len(b) == 1 and r["router_choice"] in b for r, b in zip(rows, best))
        output[f"router_unique_best_rate_at{k}"] = output[f"router_unique_best_n_at{k}"] / len(rows)
        output[f"all_three_equal_n_at{k}"] = sum(len(b) == 3 for b in best)
        output[f"unique_best_counts_at{k}"] = {str(size): sum(b == {str(size)} for b in best) for size in ROUTES.values()}
        output[f"optimal_including_ties_counts_at{k}"] = {str(size): sum(str(size) in b for b in best) for size in ROUTES.values()}
        output[f"best_set_counts_at{k}"] = dict(Counter("|".join(sorted(b, key=int)) for b in best))
        output[f"regret_sum_at{k}"] = sum(regrets)
        output[f"top_10_regret_share_at{k}"] = sum(sorted(regrets, reverse=True)[:10]) / sum(regrets) if sum(regrets) else 0
    output["router_choice_counts"] = dict(Counter(r["router_choice"] for r in rows))
    return output


def main():
    items = {x["uid"]: x for x in json.loads((SOURCE / "items.json").read_text(encoding="utf-8")) if x["dataset"] == "qasper"}
    frozen = {x["uid"]: x for x in map(json.loads, (SOURCE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()) if x["dataset"] == "qasper"}
    oracle = {x["uid"]: x for x in map(json.loads, (ORACLE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(items) == len(frozen) == len(oracle) == 300 and set(items) == set(frozen) == set(oracle)
    records = []
    for uid in sorted(items):
        item, observed, exhaustive = items[uid], frozen[uid]["results"], oracle[uid]["results"]
        assert set(exhaustive) == set(ROUTES)
        assert exhaustive["CONCEPT"]["gold_units_n"] == observed["fixed_512"]["gold_units_n"]
        if exhaustive["CONCEPT"]["recall_at_3"] is None:
            continue
        chosen = observed["multi"]["route"]
        text_retrievable = observed["multi"]["gold_units_unmapped"] == 0
        row = {"question_id": uid, "question": item["question"],
               "question_type": "not_available", "router_rule_category": chosen,
               "router_choice": str(ROUTES[chosen]), "text_retrievable": text_retrievable,
               "gold_units_n": exhaustive["CONCEPT"]["gold_units_n"]}
        for k in (3, 5):
            scores = {route: exhaustive[route][f"recall_at_{k}"] for route in ROUTES}
            assert all(v is not None for v in scores.values())
            assert scores[chosen] == observed["multi"][f"recall_at_{k}"]
            assert scores["CONCEPT"] == observed["fixed_512"][f"recall_at_{k}"]
            top = max(scores.values())
            best = [route for route in ROUTES if abs(scores[route] - top) < 1e-12]
            for route, size in ROUTES.items():
                row[f"recall_{size}_at{k}"] = scores[route]
            row[f"router_recall_at{k}"] = scores[chosen]
            row[f"oracle_recall_at{k}"] = top
            row[f"regret_at{k}"] = top - scores[chosen]
            row[f"best_granularities_at{k}"] = "|".join(str(ROUTES[route]) for route in best)
            row[f"router_optimal_at{k}"] = chosen in best
            row[f"router_unique_best_at{k}"] = len(best) == 1 and chosen in best
        records.append(row)
    assert len(records) == 237
    write_csv(OUT / "router_regret_per_question.csv", records)
    summary = {"all": summarize(records), "text_retrievable": summarize([r for r in records if r["text_retrievable"]])}
    assert summary["text_retrievable"]["evaluable_n"] == 188
    (OUT / "router_regret_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    by_type = []
    for category in ROUTES:
        subset = [r for r in records if r["router_rule_category"] == category]
        stats = summarize(subset)
        by_type.append({"router_rule_category": category, "n": len(subset), "independent_question_type": "not_available",
                        "router_choice_distribution": json.dumps(stats["router_choice_counts"]),
                        "optimal_rate_at3": stats["router_optimal_rate_at3"],
                        "optimal_rate_at5": stats["router_optimal_rate_at5"],
                        "mean_regret_at3": stats["mean_regret_at3"], "mean_regret_at5": stats["mean_regret_at5"],
                        "best_set_counts_at3": json.dumps(stats["best_set_counts_at3"]),
                        "best_set_counts_at5": json.dumps(stats["best_set_counts_at5"])})
    write_csv(OUT / "router_regret_by_type.csv", by_type)
    print(json.dumps({"all_n": len(records), "text_n": summary["text_retrievable"]["evaluable_n"],
                      "mean_regret_at5": summary["all"]["mean_regret_at5"]}))


if __name__ == "__main__":
    main()
