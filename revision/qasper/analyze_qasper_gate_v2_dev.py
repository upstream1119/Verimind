"""Analyze QASPER Gate V2 development or smoke-test judgments."""

import argparse
import json
import os
from typing import Dict, List

import pandas as pd

from revision.qasper.analyze_qasper_gate_paired_author_audit import (
    fisher_exact_two_sided,
    wilson_interval,
)
from revision.qasper.gate_v2 import (
    POLICY_VERSION,
    aggregate_judgments,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_SCORED = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_13_qasper_no_gating_complete_audit",
    "qasper_no_gating_149_scored.csv",
)
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_14_qasper_gate_v2_dev",
)
DEFAULT_JUDGMENTS = os.path.join(
    DEFAULT_INPUT_DIR,
    "gate_v2_raw_judgments.csv",
)
JUDGES = ["deepseek", "qwen", "glm"]


def build_case_decisions(
    scored: pd.DataFrame,
    judgments: pd.DataFrame,
) -> pd.DataFrame:
    required_scored = {"item_id", "author_label", "gate_reject"}
    required_judgments = {"item_id", "judge", "label"}
    missing_scored = required_scored - set(scored.columns)
    missing_judgments = required_judgments - set(judgments.columns)
    if missing_scored:
        raise ValueError(
            f"Scored data is missing columns: {sorted(missing_scored)}"
        )
    if missing_judgments:
        raise ValueError(
            "Judgments are missing columns: "
            f"{sorted(missing_judgments)}"
        )
    duplicate_count = int(
        judgments.duplicated(subset=["item_id", "judge"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Judgments contain {duplicate_count} duplicate item-judge rows"
        )
    if set(judgments["judge"]) - set(JUDGES):
        raise ValueError("Judgments contain an unexpected judge")

    label_pivot = judgments.pivot(
        index="item_id",
        columns="judge",
        values="label",
    )
    incomplete = label_pivot[JUDGES].isna().any(axis=1)
    if bool(incomplete.any()):
        raise ValueError(
            "Every analyzed item requires all three judge labels; "
            f"incomplete item IDs: "
            f"{label_pivot.index[incomplete].astype(int).tolist()}"
        )
    label_pivot = label_pivot.reset_index()
    decision_rows: List[Dict[str, object]] = []
    for row in label_pivot.itertuples(index=False):
        labels = [getattr(row, judge) for judge in JUDGES]
        aggregate = aggregate_judgments(labels)
        decision_rows.append(
            {
                "item_id": int(row.item_id),
                **{f"{judge}_label": getattr(row, judge) for judge in JUDGES},
                "gate_v2_decision": aggregate["decision"],
                "unsupported_votes": aggregate["unsupported_votes"],
                "uncertain_votes": aggregate["uncertain_votes"],
                "supported_votes": aggregate["supported_votes"],
                "policy_version": aggregate["policy_version"],
            }
        )
    decisions = pd.DataFrame(decision_rows)
    selected_scored = scored[
        scored["item_id"].isin(decisions["item_id"])
    ].copy()
    if len(selected_scored) != len(decisions):
        missing_ids = sorted(
            set(decisions["item_id"]) - set(selected_scored["item_id"])
        )
        raise ValueError(
            f"Judgment item IDs are absent from scored data: {missing_ids}"
        )
    details = selected_scored.merge(
        decisions,
        on="item_id",
        validate="one_to_one",
    )
    if {"claims_json", "summary_reason"}.issubset(judgments.columns):
        for judge in JUDGES:
            judge_details = judgments[judgments["judge"] == judge][
                ["item_id", "claims_json", "summary_reason"]
            ].rename(
                columns={
                    "claims_json": f"{judge}_claims_json",
                    "summary_reason": f"{judge}_summary_reason",
                }
            )
            details = details.merge(
                judge_details,
                on="item_id",
                validate="one_to_one",
            )
    return details.sort_values("item_id").reset_index(drop=True)


def _metric_row(
    decisions: pd.DataFrame,
    policy: str,
    intervention: pd.Series,
) -> Dict[str, object]:
    certain = decisions[
        decisions["author_label"].isin(["supported", "unsupported"])
    ].copy()
    policy_intervention = intervention.loc[certain.index].astype(bool)
    positive = certain["author_label"] == "unsupported"
    tp = int((policy_intervention & positive).sum())
    fp = int((policy_intervention & ~positive).sum())
    fn = int((~policy_intervention & positive).sum())
    tn = int((~policy_intervention & ~positive).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    false_positive_rate = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    coverage = float((~intervention.astype(bool)).mean())
    passed_risk = fn / (fn + tn) if fn + tn else float("nan")
    recall_low, recall_high = wilson_interval(tp, tp + fn)
    fpr_low, fpr_high = wilson_interval(fp, fp + tn)
    risk_low, risk_high = wilson_interval(fn, fn + tn)
    odds_ratio, fisher_p = fisher_exact_two_sided(tp, fp, fn, tn)
    return {
        "policy": policy,
        "evaluated_n": len(decisions),
        "certain_n": len(certain),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unsupported_recall": recall,
        "recall_ci_low": recall_low,
        "recall_ci_high": recall_high,
        "false_positive_rate": false_positive_rate,
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "precision": precision,
        "coverage_without_intervention": coverage,
        "passed_unsupported_risk": passed_risk,
        "passed_risk_ci_low": risk_low,
        "passed_risk_ci_high": risk_high,
        "rejection_enrichment_odds_ratio": odds_ratio,
        "rejection_enrichment_fisher_p": fisher_p,
    }


def build_policy_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    policy_masks = {
        "current_gate": decisions["gate_reject"].astype(bool),
        "deepseek_unsupported": decisions["deepseek_label"].eq(
            "unsupported"
        ),
        "qwen_unsupported": decisions["qwen_label"].eq("unsupported"),
        "glm_unsupported": decisions["glm_label"].eq("unsupported"),
        "gate_v2_reject_only": decisions["gate_v2_decision"].eq("REJECT"),
        "gate_v2_intervention": decisions["gate_v2_decision"].isin(
            ["REJECT", "WARNING"]
        ),
    }
    return pd.DataFrame(
        [
            _metric_row(decisions, policy, mask)
            for policy, mask in policy_masks.items()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default=DEFAULT_SCORED)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    scored = pd.read_csv(args.scored, keep_default_na=False)
    raw = pd.read_csv(args.judgments, keep_default_na=False)
    if "status" in raw.columns:
        unsuccessful = raw[raw["status"] != "ok"]
        if len(unsuccessful):
            raise ValueError(
                "All analyzed judgments must have status=ok; "
                f"found {unsuccessful['status'].value_counts().to_dict()}"
            )
    decisions = build_case_decisions(scored, raw)
    metrics = build_policy_metrics(decisions)
    disagreements = decisions[
        (decisions[JUDGES].nunique(axis=1) > 1)
        if set(JUDGES).issubset(decisions.columns)
        else (
            decisions[
                [f"{judge}_label" for judge in JUDGES]
            ].nunique(axis=1)
            > 1
        )
    ].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    decisions_path = os.path.join(
        args.output_dir,
        "gate_v2_case_decisions.csv",
    )
    metrics_path = os.path.join(
        args.output_dir,
        "gate_v2_policy_metrics.csv",
    )
    disagreements_path = os.path.join(
        args.output_dir,
        "gate_v2_disagreements.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "gate_v2_dev_summary.txt",
    )
    decisions.to_csv(decisions_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    disagreements.to_csv(
        disagreements_path,
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "QASPER Gate V2 development analysis",
        "===================================",
        "",
        f"Analyzed items: {len(decisions)}",
        f"Policy version: {POLICY_VERSION}",
        (
            "Author labels: "
            f"{decisions['author_label'].value_counts().to_dict()}"
        ),
        "",
    ]
    for row in metrics.itertuples():
        lines.append(
            f"{row.policy}: TP={row.tp}, FP={row.fp}, FN={row.fn}, "
            f"TN={row.tn}, recall={row.unsupported_recall:.2%}, "
            f"FPR={row.false_positive_rate:.2%}, "
            f"coverage={row.coverage_without_intervention:.2%}, "
            f"passed risk={row.passed_unsupported_risk:.2%}"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            (
                "This is development or smoke-test evidence on an audited "
                "development set. It cannot establish final Gate V2 "
                "performance."
            ),
            (
                "The pre-registered policy rejects with at least two "
                "unsupported votes and warns on one unsupported vote or any "
                "uncertain vote."
            ),
        ]
    )
    with open(summary_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nCase decisions: {decisions_path}")
    print(f"Policy metrics: {metrics_path}")
    print(f"Disagreements: {disagreements_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
