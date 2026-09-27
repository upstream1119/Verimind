"""Offline route diagnostics on frozen QASPER candidate answers."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from offline_gate_diagnostics import auc_ap, read_rows, write_csv
from four_gate_analyses_20260927 import risk_curve

SOURCE = Path("revision/artifacts/supplement_20260927_full")
OUTPUT = Path("revision/artifacts/route_score_audit_20260927")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    retrieval = {r["uid"]: r for r in map(json.loads, (SOURCE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = [r for r in read_rows(SOURCE) if r["dataset"] == "qasper" and r["candidate"]]
    for row in rows:
        row["route"] = retrieval[row["sample_id"]]["results"]["multi"]["route"]
        row["safe"] = bool(row["supported"] and row["correct"])
    groups = {route: [r for r in rows if r["route"] == route] for route in ("DETAIL", "CONCEPT", "CONTEXT")}
    stratified = []
    for route, sample in [("ALL", rows), *groups.items()]:
        for target in ("supported", "safe"):
            y = [r[target] for r in sample]
            auc, ap = auc_ap([r["retrieval_score"] for r in sample], y)
            stratified.append(dict(route=route, target=target, n=len(sample), positive_n=sum(y), auroc=auc, auprc=ap))
    write_csv(OUTPUT / "route_retrieval_auc.csv", stratified)

    score_variants = []
    for method in ("raw", "percentile", "zscore", "empirical_cdf"):
        for row in rows:
            vals = sorted(r["retrieval_score"] for r in groups[row["route"]])
            v = row["retrieval_score"]
            below, equal = sum(x < v for x in vals), sum(x == v for x in vals)
            if method == "raw":
                normalized = v
            elif method == "percentile":
                normalized = (below + (equal - 1) / 2) / max(1, len(vals) - 1)
            elif method == "empirical_cdf":
                normalized = (below + equal / 2) / len(vals)
            else:
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
                normalized = .5 * (1 + math.erf(((v - mean) / std if std else 0) / math.sqrt(2)))
            score_variants.append(dict(method=method, sample_id=row["sample_id"], route=row["route"],
                                       retrieval_raw=v, retrieval_normalized=normalized,
                                       judge=row["judge_score"], trust=.4 * normalized + .6 * row["judge_score"],
                                       supported=row["supported"], correct=row["correct"], safe=row["safe"]))
    write_csv(OUTPUT / "normalized_scores.csv", score_variants)
    metrics = []
    by_id = {r["sample_id"]: r for r in rows}
    for method in ("raw", "percentile", "zscore", "empirical_cdf"):
        sample = [r for r in score_variants if r["method"] == method]
        for target in ("supported", "safe"):
            auc, ap = auc_ap([r["trust"] for r in sample], [r[target] for r in sample])
            metrics.append(dict(method=method, metric="AUROC", target=target, value=auc))
            metrics.append(dict(method=method, metric="AUPRC", target=target, value=ap))
        for target, fail in (("error", lambda r: not r["correct"]),
                             ("unsupported", lambda r: not r["supported"])):
            curve = risk_curve(sample, "trust", fail)
            metrics.append(dict(method=method, metric="AURC", target=target,
                                value=sum(p["risk"] for p in curve) / len(curve)))
    write_csv(OUTPUT / "normalization_metrics.csv", metrics)

    # Only fixed 512 and the routed size were queried; this is NOT a three-size oracle.
    comparison = []
    for k in (3, 5):
        eligible = [r for r in retrieval.values() if r["dataset"] == "qasper"
                    and r["results"]["fixed_512"][f"recall_at_{k}"] is not None]
        comparison.append(dict(k=k, n=len(eligible), fixed_512=sum(r["results"]["fixed_512"][f"recall_at_{k}"] for r in eligible)/len(eligible),
                               routed=sum(r["results"]["multi"][f"recall_at_{k}"] for r in eligible)/len(eligible),
                               best_observed=sum(max(r["results"]["fixed_512"][f"recall_at_{k}"],
                                                     r["results"]["multi"][f"recall_at_{k}"]) for r in eligible)/len(eligible)))
    write_csv(OUTPUT / "observed_route_oracle_lower_bound.csv", comparison)
    validation = dict(candidate_n=len(rows), route_counts={k: len(v) for k,v in groups.items()},
                      three_granularity_oracle_available=False,
                      queried_routes_per_question=dict(Counter(len({v["route"] for v in r["results"].values()})
                                                            for r in retrieval.values() if r["dataset"] == "qasper")))
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
