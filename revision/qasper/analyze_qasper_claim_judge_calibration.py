"""Evaluate claim-level judges against the author-conducted blind audit."""

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)
from analyze_qasper_author_audit import (  # noqa: E402
    compute_binary_metrics,
    normalize_label,
)
from run_qasper_claim_judge_calibration import (  # noqa: E402
    parse_claim_judgment,
)


DEFAULT_AUTHOR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_6_qasper_author_audit",
    "qasper_author_audit_50_completed.csv",
)
DEFAULT_JUDGMENTS = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_8_qasper_claim_judge_calibration",
    "qasper_claim_judge_results.csv",
)
DEFAULT_OUTPUT_DIR = os.path.dirname(DEFAULT_JUDGMENTS)
JUDGES = ["deepseek", "qwen", "glm"]
MIN_MAJORITY_RECALL = 0.75
MAX_MAJORITY_FALSE_POSITIVES = 3


def calibration_passes(
    true_positives: int,
    false_positives: int,
    author_positives: int,
) -> bool:
    required_true_positives = math.ceil(
        MIN_MAJORITY_RECALL * author_positives
    )
    return (
        true_positives >= required_true_positives
        and false_positives <= MAX_MAJORITY_FALSE_POSITIVES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    author = pd.read_csv(args.author, keep_default_na=False)
    judgments = pd.read_csv(args.judgments, keep_default_na=False)
    if len(author) != 50 or author["audit_id"].duplicated().any():
        raise ValueError("Author audit must contain 50 unique cases")
    author["author_label"] = author["author_label"].map(normalize_label)
    invalid_author = set(author["author_label"]) - {
        "supported",
        "unsupported",
        "uncertain",
    }
    if invalid_author:
        raise ValueError(
            f"Invalid author labels: {sorted(invalid_author)}"
        )

    expected = len(author) * len(JUDGES)
    if len(judgments) != expected:
        raise ValueError(
            f"Expected {expected} raw judgments, "
            f"found {len(judgments)}; "
            f"statuses={judgments['status'].value_counts().to_dict()}"
        )
    duplicate_records = int(
        judgments.duplicated(subset=["audit_id", "judge"]).sum()
    )
    if duplicate_records:
        raise ValueError(
            f"Found {duplicate_records} duplicate raw judgments"
        )
    if set(judgments["judge"]) != set(JUDGES):
        raise ValueError("Judgment file has unexpected or missing judges")

    normalized_rows: List[Dict[str, object]] = []
    for row in judgments.itertuples(index=False):
        parsed = parse_claim_judgment(str(row.raw_response))
        normalized_rows.append(
            {
                "audit_id": row.audit_id,
                "judge": row.judge,
                "judge_model": row.judge_model,
                "model_label": parsed["model_label"],
                "label": parsed["label"],
                "label_repaired": parsed["label_repaired"],
                "unsupported_event": int(
                    parsed["label"] == "unsupported"
                ),
                "claim_count": len(parsed["claims"]),
                "unsupported_claim_count": parsed[
                    "unsupported_claim_count"
                ],
                "claims_json": json.dumps(
                    parsed["claims"],
                    ensure_ascii=False,
                ),
                "summary_reason": parsed["summary_reason"],
                "original_status": row.status,
                "original_error_message": row.error_message,
                "raw_response": row.raw_response,
                "prompt_version": row.prompt_version,
                "context_chars": row.context_chars,
                "evaluated_at_utc": row.evaluated_at_utc,
            }
        )
    normalized = pd.DataFrame(normalized_rows)
    if (normalized["claim_count"].astype(int) < 1).any():
        raise ValueError("Every normalized judgment must contain a claim")

    label_pivot = normalized.pivot(
        index="audit_id",
        columns="judge",
        values="unsupported_event",
    ).reset_index()
    for judge in JUDGES:
        label_pivot[judge] = label_pivot[judge].astype(int)
    label_pivot["majority_vote"] = (
        label_pivot[JUDGES].sum(axis=1) >= 2
    ).astype(int)

    details = author.merge(
        label_pivot,
        on="audit_id",
        how="inner",
        validate="one_to_one",
    )
    for judge in JUDGES:
        judge_rows = normalized[normalized["judge"] == judge][
            [
                "audit_id",
                "claims_json",
                "summary_reason",
                "raw_response",
            ]
        ].rename(
            columns={
                "claims_json": f"{judge}_claims_json",
                "summary_reason": f"{judge}_summary_reason",
                "raw_response": f"{judge}_raw_response",
            }
        )
        details = details.merge(
            judge_rows,
            on="audit_id",
            how="inner",
            validate="one_to_one",
        )

    details["author_unsupported_event"] = details[
        "author_label"
    ].map({"supported": 0, "unsupported": 1, "uncertain": -1})
    certain = details[details["author_unsupported_event"] >= 0].copy()
    author_values = certain["author_unsupported_event"].astype(int).tolist()

    metric_rows: List[Dict[str, object]] = []
    for judge in JUDGES + ["majority_vote"]:
        metrics = compute_binary_metrics(
            author_values,
            certain[judge].astype(int).tolist(),
        )
        metric_rows.append({"judge": judge, **metrics})
    summary = pd.DataFrame(metric_rows)
    majority = summary[summary["judge"] == "majority_vote"].iloc[0]
    author_positives = int(sum(author_values))
    passed = calibration_passes(
        true_positives=int(majority["tp"]),
        false_positives=int(majority["fp"]),
        author_positives=author_positives,
    )

    certain["majority_disagrees_with_author"] = (
        certain["majority_vote"]
        != certain["author_unsupported_event"]
    )
    disagreements = certain[
        certain["majority_disagrees_with_author"]
    ].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(
        args.output_dir,
        "qasper_claim_judge_summary.csv",
    )
    normalized_csv = os.path.join(
        args.output_dir,
        "qasper_claim_judge_normalized.csv",
    )
    details_csv = os.path.join(
        args.output_dir,
        "qasper_claim_judge_scored.csv",
    )
    disagreements_csv = os.path.join(
        args.output_dir,
        "qasper_claim_judge_disagreements.csv",
    )
    decision_json = os.path.join(
        args.output_dir,
        "qasper_claim_judge_decision.json",
    )
    narrative_txt = os.path.join(
        args.output_dir,
        "qasper_claim_judge_summary.txt",
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    normalized.to_csv(
        normalized_csv,
        index=False,
        encoding="utf-8-sig",
    )
    details.to_csv(details_csv, index=False, encoding="utf-8-sig")
    disagreements.to_csv(
        disagreements_csv,
        index=False,
        encoding="utf-8-sig",
    )

    decision = {
        "decision": "PASS" if passed else "FAIL",
        "author_certain_n": len(certain),
        "author_unsupported_n": author_positives,
        "required_majority_true_positives": math.ceil(
            MIN_MAJORITY_RECALL * author_positives
        ),
        "maximum_majority_false_positives": (
            MAX_MAJORITY_FALSE_POSITIVES
        ),
        "observed_majority_true_positives": int(majority["tp"]),
        "observed_majority_false_positives": int(majority["fp"]),
        "observed_majority_recall": float(
            majority["recall_unsupported"]
        ),
        "observed_majority_agreement": float(majority["agreement"]),
        "observed_majority_kappa": float(majority["cohen_kappa"]),
        "normalized_repaired_labels": int(
            normalized["label_repaired"].sum()
        ),
        "full_802_run_authorized_by_rule": passed,
    }
    with open(decision_json, "w", encoding="utf-8") as output_file:
        json.dump(decision, output_file, ensure_ascii=False, indent=2)

    lines = [
        "QASPER claim-level judge calibration",
        "====================================",
        "",
        f"Decision: {decision['decision']}",
        f"Certain author labels: {len(certain)}",
        f"Author unsupported cases: {author_positives}",
        (
            "Pass rule: majority TP >= "
            f"{decision['required_majority_true_positives']} and FP <= "
            f"{decision['maximum_majority_false_positives']}"
        ),
        (
            "Observed majority: "
            f"TP={decision['observed_majority_true_positives']}, "
            f"FP={decision['observed_majority_false_positives']}, "
            f"recall={decision['observed_majority_recall']:.4f}, "
            f"agreement={decision['observed_majority_agreement']:.4f}, "
            f"kappa={decision['observed_majority_kappa']:.4f}"
        ),
        (
            "Normalized top-level/claim label conflicts: "
            f"{decision['normalized_repaired_labels']}"
        ),
        "",
    ]
    for row in summary.itertuples():
        lines.append(
            f"{row.judge}: agreement={row.agreement:.4f}, "
            f"kappa={row.cohen_kappa:.4f}, TP={row.tp}, "
            f"FP={row.fp}, FN={row.fn}, TN={row.tn}"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            (
                "The author-conducted blind audit is the reference for this "
                "calibration sample. It is not an independent expert study."
            ),
            (
                "The challenge-enriched sample cannot estimate population "
                "hallucination rates or compare system prevalence."
            ),
        ]
    )
    with open(narrative_txt, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nNormalized judgments: {normalized_csv}")
    print(f"\nSummary: {summary_csv}")
    print(f"Scored cases: {details_csv}")
    print(f"Disagreements: {disagreements_csv}")
    print(f"Decision: {decision_json}")
    print(f"Narrative: {narrative_txt}")


if __name__ == "__main__":
    main()
