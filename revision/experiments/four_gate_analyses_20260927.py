"""Four offline gate analyses on frozen candidate answers and retrieval records."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from offline_gate_diagnostics import auc_ap, read_rows, write_csv
from supplement_20260927 import percentile


def risk_curve(rows, score, failure):
    """Expected prefix risk under random order within exact-score ties."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[score]].append(row)
    points = []
    seen = 0
    expected_errors = 0.0
    n = len(rows)
    for value in sorted(groups, reverse=True):
        group = groups[value]
        error_fraction = sum(failure(row) for row in group) / len(group)
        for within in range(1, len(group) + 1):
            prefix = seen + within
            risk = (expected_errors + within * error_fraction) / prefix
            points.append({"rank": prefix, "coverage": prefix / n, "score": value, "risk": risk})
        seen += len(group)
        expected_errors += sum(failure(row) for row in group)
    assert seen == n and len(points) == n
    return points


def risk_at_coverage(curve, fraction):
    index = max(1, min(len(curve), math.ceil(len(curve) * fraction)))
    return curve[index - 1]["risk"]


def analyze_group(label, sample, score_rows, aurc_rows, curve_rows):
    if not sample:
        return
    for target in ("supported", "safe"):
        for score in ("retrieval_score", "judge_score", "trust_score"):
            auroc, ap = auc_ap([row[score] for row in sample], [row[target] for row in sample])
            score_rows.append({"dataset": label, "target": target, "score": score, "n": len(sample),
                               "positive_n": sum(row[target] for row in sample), "auroc": auroc,
                               "auprc_ap": ap, "prevalence": sum(row[target] for row in sample) / len(sample)})
    for score in ("retrieval_score", "judge_score", "trust_score"):
        for risk_name, failure in (("error", lambda row: not row["correct"]),
                                   ("unsupported", lambda row: not row["supported"])):
            curve = risk_curve(sample, score, failure)
            aurc_rows.append({"dataset": label, "score": score, "risk": risk_name, "candidate_n": len(sample),
                              "aurc": sum(point["risk"] for point in curve) / len(curve),
                              "risk_at_60pct_candidates": risk_at_coverage(curve, .6),
                              "risk_at_70pct_candidates": risk_at_coverage(curve, .7),
                              "risk_at_80pct_candidates": risk_at_coverage(curve, .8),
                              "risk_at_90pct_candidates": risk_at_coverage(curve, .9),
                              "risk_at_100pct_candidates": curve[-1]["risk"]})
            for point in curve:
                curve_rows.append({"dataset": label, "score": score, "risk": risk_name,
                                   "rank": point["rank"], "coverage_of_candidates": point["coverage"],
                                   "coverage_of_all_questions": point["rank"] / ({"QASPER_all": 300,
                                       "QASPER_text_retrievable": 251, "SciFact+NEI": 514}[label]),
                                   "score_at_rank": point["score"], "selective_risk": point["risk"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.source)
    for row in rows:
        row["safe"] = bool(row["supported"] and row["correct"])
    retrievals = {r["uid"]: r for r in (json.loads(line) for line in
                  (args.source / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(retrievals) == 814
    qasper = [r for r in rows if r["dataset"] == "qasper"]
    scifact = [r for r in rows if r["dataset"] in ("scifact", "scifact_nei")]
    # 63 inaccessible chart-evidence units occur in 49 distinct questions.
    removed_ids = {r["sample_id"] for r in qasper if
                   retrievals[r["sample_id"]]["results"]["multi"]["gold_units_unmapped"] > 0}
    text_subset = [r for r in qasper if r["sample_id"] not in removed_ids]
    assert len(removed_ids) == 49 and len(text_subset) == 251
    assert sum(retrievals[uid]["results"]["multi"]["gold_units_unmapped"] for uid in removed_ids) == 63
    assert all(retrievals[r["sample_id"]]["results"]["fixed_512"]["gold_units_unmapped"] == 0 for r in text_subset)
    score_rows, aurc_rows, curve_rows = [], [], []
    for name, group in (("QASPER_all", qasper), ("SciFact+NEI", scifact),
                        ("QASPER_text_retrievable", text_subset)):
        analyze_group(name, [row for row in group if row["candidate"]], score_rows, aurc_rows, curve_rows)
    recall_rows = []
    for name, group in (("QASPER_all", qasper), ("QASPER_text_retrievable", text_subset)):
        for strategy in ("fixed_512", "multi"):
            relevant = [retrievals[r["sample_id"]]["results"][strategy] for r in group
                        if retrievals[r["sample_id"]]["results"][strategy]["gold_units_n"] > 0]
            recall_rows.append({"dataset": name, "strategy": strategy, "questions_n": len(group),
                                "evaluable_n": len(relevant), "recall_at_3": sum(r["recall_at_3"] for r in relevant)/len(relevant),
                                "recall_at_5": sum(r["recall_at_5"] for r in relevant)/len(relevant)})
    distributions, hist, unique = [], [], []
    for name, group in (("QASPER", qasper), ("SciFact+NEI", scifact)):
        for candidate_only in (False, True):
            selected = [r for r in group if r["candidate"]] if candidate_only else group
            values = [r["trust_score"] for r in selected]
            distributions.append({"dataset": name, "candidate_only": candidate_only, "n": len(values),
                                  "min": min(values), "p10": percentile(values,.1), "p25": percentile(values,.25),
                                  "median": percentile(values,.5), "p75": percentile(values,.75),
                                  "p90": percentile(values,.9), "max": max(values),
                                  "unique_values": len(set(values))})
            counts = Counter(min(19, int(value*20)) for value in values)
            for index in range(20):
                hist.append({"dataset": name, "candidate_only": candidate_only,
                             "bin_left": index/20, "bin_right": (index+1)/20,
                             "count": counts[index], "proportion": counts[index]/len(values)})
            for value, count in sorted(Counter(values).items()):
                unique.append({"dataset": name, "candidate_only": candidate_only,
                               "score": value, "count": count})
    boundary = []
    for name, group in (("QASPER", qasper), ("SciFact+NEI", scifact)):
        for row in group:
            if row["candidate"] and .3 <= row["trust_score"] < .4:
                boundary.append({"dataset": name, "sample_id": row["sample_id"],
                                 "retrieval_score": row["retrieval_score"], "judge_score": row["judge_score"],
                                 "trust_score": row["trust_score"], "supported": row["supported"],
                                 "correct": row["correct"]})
    for name, data in (("candidate_scores.csv",score_rows),("aurc.csv",aurc_rows),
                       ("risk_coverage.csv",curve_rows),("qasper_recall.csv",recall_rows),
                       ("trust_percentiles.csv",distributions),("trust_histogram.csv",hist),
                       ("trust_unique_values.csv",unique),("trust_030_040_cases.csv",boundary)):
        write_csv(args.output/name,data)
    (args.output/"qasper_removed_question_ids.txt").write_text("\n".join(sorted(removed_ids))+"\n",encoding="utf-8")
    # AURC invariance when all candidates have identical errors and exact tie handling.
    fixture = [{"score":1,"correct":True},{"score":1,"correct":False}]
    assert math.isclose(sum(p["risk"] for p in risk_curve(fixture,"score",lambda r:not r["correct"]))/2,.5)
    assert sum(r["dataset"]=="SciFact+NEI" for r in boundary)==145
    validation={"source_answer_rows":1628,"source_retrieval_rows":814,
                "candidate_qasper":sum(r["candidate"] for r in qasper),
                "candidate_scifact_nei":sum(r["candidate"] for r in scifact),
                "qasper_removed_questions":len(removed_ids),"qasper_removed_chart_evidence_units":63,
                "text_subset_questions":len(text_subset),"text_subset_candidates":sum(r["candidate"] for r in text_subset),
                "scifact_candidate_trust_030_040":145,"tests":"tie-aware AURC fixture and source integrity passed"}
    (args.output/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    print(json.dumps(validation))


if __name__=="__main__":
    main()
