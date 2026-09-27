"""Offline score and threshold audit of frozen paired answers; no model calls."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from supplement_20260927 import decision, is_refusal, percentile, refusal_metrics


def auc_ap(scores, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None, None
    groups = defaultdict(lambda: [0, 0])
    for score, label in zip(scores, labels):
        groups[score][0 if label else 1] += 1
    wins = 0.0
    negative_below = 0
    for score in sorted(groups):
        positive_here, negative_here = groups[score]
        wins += positive_here * (negative_below + negative_here / 2)
        negative_below += negative_here
    tp = fp = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        positive_here, negative_here = groups[score]
        tp += positive_here
        fp += negative_here
        ap += positive_here / positives * tp / (tp + fp)
    return wins / (positives * negatives), ap


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(source):
    items = {item["uid"]: item for item in json.loads((source / "items.json").read_text(encoding="utf-8"))}
    raw = [json.loads(line) for line in (source / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(items) == 814 and len(raw) == 1628
    assert len({(item["uid"], item["strategy"]) for item in raw}) == len(raw)
    rows = []
    for answer in raw:
        if answer["strategy"] != "multi":
            continue
        item = items[answer["uid"]]
        match = re.match(r"\s*(SUPPORT|CONTRADICT)\b", answer["answer"], re.I)
        correct = (answer["evaluation"]["correct"] if item["dataset"] == "qasper"
                   else bool(match and match.group(1).upper() in item["gold_labels"]))
        candidate = not is_refusal(answer["answer"])
        rows.append({"sample_id": item["uid"], "dataset": item["dataset"],
                     "answerable": not item["is_unanswerable"], "candidate": candidate,
                     "retrieval_score": answer["calibrated_retrieval"] / 10,
                     "judge_score": answer["faith_score"] / 10,
                     "trust_score": answer["trust_score"] / 10,
                     "supported": answer["evaluation"]["supported"], "correct": correct,
                     "safe": bool(not item["is_unanswerable"] and answer["evaluation"]["supported"] and correct),
                     "answer": answer["answer"]})
    assert Counter(row["dataset"] for row in rows) == {"qasper": 300, "scifact": 454, "scifact_nei": 60}
    return rows


def groups(rows):
    return {
        "QASPER": [r for r in rows if r["dataset"] == "qasper"],
        "SciFact+NEI": [r for r in rows if r["dataset"] in ("scifact", "scifact_nei")],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    metric_rows, sweep_rows, debug_rows, gate_rows, quantile_rows, distribution_rows = [], [], [], [], [], []
    for dataset, group in groups(rows).items():
        candidates = [r for r in group if r["candidate"]]
        for score, label, sample in (("retrieval_score", "answerable", group),
                                     ("judge_score", "supported", candidates),
                                     ("trust_score", "safe", candidates)):
            auc, ap = auc_ap([r[score] for r in sample], [r[label] for r in sample])
            metric_rows.append({"dataset": dataset, "score": score, "positive_label": label,
                                "n": len(sample), "positive_n": sum(r[label] for r in sample),
                                "auroc": auc, "auprc_average_precision": ap,
                                "prevalence": sum(r[label] for r in sample) / len(sample)})
        for score in ("retrieval_score", "judge_score", "trust_score"):
            auc, ap = auc_ap([r[score] for r in candidates], [r["safe"] for r in candidates])
            metric_rows.append({"dataset": dataset, "score": score, "positive_label": "safe_common_target",
                                "n": len(candidates), "positive_n": sum(r["safe"] for r in candidates),
                                "auroc": auc, "auprc_average_precision": ap,
                                "prevalence": sum(r["safe"] for r in candidates) / len(candidates)})
        retrieval_rejects = sum(r["candidate"] and r["retrieval_score"] < .5 for r in group)
        differs = sum(decision(r["answer"], 10*r["retrieval_score"], 10*r["judge_score"], "no_judge")
                      != decision(r["answer"], 10*r["retrieval_score"], 10*r["judge_score"], "multi") for r in group)
        gate_rows.append({"dataset": dataset, "n": len(group), "candidate_n": len(candidates),
                          "retrieval_threshold": .5, "retrieval_below_threshold_all": sum(r["retrieval_score"] < .5 for r in group),
                          "retrieval_gate_rejections": retrieval_rejects, "vs_no_gate_decision_differences": differs,
                          "score_min": min(r["retrieval_score"] for r in group),
                          "score_max": max(r["retrieval_score"] for r in group)})
        for step in range(1, 10):
            threshold = step / 10
            passed = [r for r in group if r["candidate"] and r["trust_score"] >= threshold]
            confusion = refusal_metrics([not r["answerable"] for r in group],
                                        [not (r["candidate"] and r["trust_score"] >= threshold) for r in group])
            sweep_rows.append({"dataset": dataset, "threshold": threshold, "n": len(group), "answered_n": len(passed),
                               "coverage": len(passed)/len(group),
                               "selective_accuracy": sum(r["correct"] for r in passed)/len(passed) if passed else None,
                               "unsupported_rate": sum(not r["supported"] for r in passed)/len(passed) if passed else None,
                               "refusal_precision": confusion["precision"], "refusal_recall": confusion["recall"],
                               "refusal_f1": confusion["f1"]})
            quantile = percentile([r["retrieval_score"] for r in group], threshold)
            quantile_rows.append({"dataset": dataset, "percentile": step*10, "threshold": quantile,
                                 "coverage": sum(r["candidate"] and r["retrieval_score"] >= quantile for r in group)/len(group)})
        for score, label, sample in (("retrieval_score", "answerable", group),
                                     ("judge_score", "supported", candidates),
                                     ("trust_score", "answerable", group),
                                     ("trust_score", "correct", candidates),
                                     ("trust_score", "supported", candidates)):
            for value in (False, True):
                values = [r[score] for r in sample if r[label] == value]
                distribution_rows.append({"dataset": dataset, "score": score, "group_by": label, "group_value": value,
                                          "n": len(values), "p10": percentile(values, .1), "median": percentile(values, .5),
                                          "p90": percentile(values, .9)})
        series = [r for r in sweep_rows if r["dataset"] == dataset]
        assert all(a["coverage"] >= b["coverage"] for a, b in zip(series, series[1:]))
    strata = (("qasper", True, 10), ("qasper", False, 10),
              ("scifact", True, 15), ("scifact_nei", False, 15))
    for dataset, answerable, count in strata:
        pool = [r for r in rows if r["dataset"] == dataset and r["answerable"] == answerable]
        chosen = sorted(pool, key=lambda r: hashlib.sha256(("gate-debug-20260927" + r["sample_id"]).encode()).hexdigest())[:count]
        for row in chosen:
            full = decision(row["answer"], 10*row["retrieval_score"], 10*row["judge_score"], "full")
            retrieval = decision(row["answer"], 10*row["retrieval_score"], 10*row["judge_score"], "no_judge")
            debug_rows.append({"sample_id": row["sample_id"], "dataset": row["dataset"], "answerable": row["answerable"],
                               "retrieval_score": row["retrieval_score"], "judge_score": row["judge_score"],
                               "trust_score": row["trust_score"], "retrieval_threshold": .5,
                               "judge_threshold": "not_used_separately", "fused_threshold": .5,
                               "retrieval_gate_pass": retrieval == "ANSWER", "judge_gate_pass": full == "ANSWER",
                               "final_gate_decision": full, "answered": full == "ANSWER", "refused": full != "ANSWER",
                               "supported": row["supported"] if row["candidate"] else "no_substantive_answer",
                               "correct": row["correct"] if row["candidate"] else "no_substantive_answer"})
    assert len(debug_rows) == 50
    for observed, expected in zip(auc_ap([.1,.4,.35,.8], [False,False,True,True]), (.75, 5/6)):
        assert math.isclose(observed, expected)
    for observed, expected in zip(auc_ap([1,1,1,1], [True,False,False,False]), (.5,.25)):
        assert math.isclose(observed, expected)
    for score, expected in ((10,"ANSWER"),(0,"REJECT"),(4.99999,"REJECT"),(5,"ANSWER"),(5.00001,"ANSWER")):
        assert decision("A factual answer.", score, 8, "no_judge") == expected
    for name, data in (("scores.csv", metric_rows),("threshold_sweep.csv", sweep_rows),
                       ("debug50.csv", debug_rows),("gate_activation.csv", gate_rows),
                       ("retrieval_quantiles.csv", quantile_rows),("distributions.csv", distribution_rows)):
        write_csv(args.output / name, data)
    summary = {"source": str(args.source.resolve()), "method": "offline frozen multi answers; no API calls",
               "rows": len(rows), "debug_rows": len(debug_rows), "metrics": len(metric_rows),
               "threshold_rows": len(sweep_rows), "tests": "AUROC/AP fixtures, boundary decisions, sweep monotonicity passed"}
    (args.output / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
