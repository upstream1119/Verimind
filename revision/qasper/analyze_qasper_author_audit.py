"""Compare completed author labels with strict binary LLM judgments."""

import argparse
import os
from typing import Dict, List, Sequence

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_6_qasper_author_audit",
)
DEFAULT_BLIND = os.path.join(
    DEFAULT_INPUT_DIR,
    "qasper_author_audit_50_completed.csv",
)
DEFAULT_KEY = os.path.join(
    DEFAULT_INPUT_DIR,
    "qasper_author_audit_50_key.csv",
)
DEFAULT_JUDGMENTS = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_7_qasper_binary_judge_audit",
    "qasper_binary_judge_results.csv",
)
DEFAULT_OUTPUT_DIR = os.path.dirname(DEFAULT_JUDGMENTS)
JUDGES = ["deepseek", "qwen", "glm"]


def normalize_label(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "0": "supported",
        "1": "unsupported",
        "2": "uncertain",
        "supported": "supported",
        "unsupported": "unsupported",
        "uncertain": "uncertain",
    }
    return aliases.get(normalized, normalized)


def cohen_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_one = sum(left) / total
    right_one = sum(right) / total
    expected = (
        left_one * right_one
        + (1.0 - left_one) * (1.0 - right_one)
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return (observed - expected) / (1.0 - expected)


def compute_binary_metrics(
    author: Sequence[int],
    judge: Sequence[int],
) -> Dict[str, float]:
    if len(author) != len(judge) or not author:
        raise ValueError("Author and judge labels must be non-empty and paired")
    tp = sum(a == 1 and j == 1 for a, j in zip(author, judge))
    fp = sum(a == 0 and j == 1 for a, j in zip(author, judge))
    fn = sum(a == 1 and j == 0 for a, j in zip(author, judge))
    tn = sum(a == 0 and j == 0 for a, j in zip(author, judge))
    return {
        "n": len(author),
        "agreement": (tp + tn) / len(author),
        "cohen_kappa": cohen_kappa(author, judge),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_unsupported": (
            tp / (tp + fp) if tp + fp else float("nan")
        ),
        "recall_unsupported": (
            tp / (tp + fn) if tp + fn else float("nan")
        ),
        "specificity_supported": (
            tn / (tn + fp) if tn + fp else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", default=DEFAULT_BLIND)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    blind = pd.read_csv(args.blind, keep_default_na=False)
    key = pd.read_csv(args.key)
    judgments = pd.read_csv(args.judgments, keep_default_na=False)
    if len(blind) != 50 or len(key) != 50:
        raise ValueError(
            f"Expected 50 rows in each file; found {len(blind)} and {len(key)}"
        )
    if blind["audit_id"].duplicated().any():
        raise ValueError("Blind file contains duplicate audit_id values")
    if key["audit_id"].duplicated().any():
        raise ValueError("Key file contains duplicate audit_id values")

    blind["author_label"] = blind["author_label"].map(normalize_label)
    allowed = {"supported", "unsupported", "uncertain"}
    invalid = sorted(set(blind["author_label"]) - allowed)
    if invalid:
        raise ValueError(
            "Complete all labels with supported/unsupported/uncertain. "
            f"Invalid or blank values: {invalid}"
        )

    successful = judgments[judgments["status"] == "ok"].copy()
    duplicate_successes = int(
        successful.duplicated(subset=["audit_id", "judge"]).sum()
    )
    if duplicate_successes:
        raise ValueError(
            f"Found {duplicate_successes} duplicate successful judgments"
        )
    expected_records = len(blind) * len(JUDGES)
    if len(successful) != expected_records:
        status_counts = judgments["status"].value_counts().to_dict()
        raise ValueError(
            f"Expected {expected_records} successful judgments, found "
            f"{len(successful)}; statuses={status_counts}"
        )
    invalid_judges = set(successful["judge"]) - set(JUDGES)
    if invalid_judges:
        raise ValueError(f"Unexpected judges: {sorted(invalid_judges)}")
    invalid_labels = set(successful["binary_label"]) - {
        "supported",
        "unsupported",
    }
    if invalid_labels:
        raise ValueError(
            f"Invalid binary labels: {sorted(invalid_labels)}"
        )

    pivot = successful.pivot(
        index="audit_id",
        columns="judge",
        values="unsupported_event",
    ).reset_index()
    if set(JUDGES) - set(pivot.columns):
        raise ValueError("At least one expected judge column is missing")
    if pivot[JUDGES].isna().any().any():
        raise ValueError("At least one audit case lacks a judge label")
    for judge in JUDGES:
        pivot[judge] = pivot[judge].astype(int)
    pivot["majority_vote"] = (
        pivot[JUDGES].sum(axis=1) >= 2
    ).astype(int)

    scored = blind.merge(
        key,
        on="audit_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        pivot,
        on="audit_id",
        how="inner",
        validate="one_to_one",
    )
    scored["author_unsupported_event"] = scored[
        "author_label"
    ].map({"supported": 0, "unsupported": 1, "uncertain": -1})
    certain = scored[scored["author_unsupported_event"] >= 0].copy()
    author = certain["author_unsupported_event"].astype(int).tolist()
    metric_rows: List[Dict[str, object]] = []
    for judge in JUDGES + ["majority_vote"]:
        metrics = compute_binary_metrics(
            author,
            certain[judge].astype(int).tolist(),
        )
        metric_rows.append({"judge": judge, **metrics})
    summary = pd.DataFrame(metric_rows)

    certain["any_judge_disagreement"] = (
        certain[JUDGES + ["majority_vote"]]
        .ne(certain["author_unsupported_event"], axis=0)
        .any(axis=1)
    )
    disagreements = certain[certain["any_judge_disagreement"]].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(
        args.output_dir,
        "qasper_author_audit_50_scored.csv",
    )
    summary_csv_path = os.path.join(
        args.output_dir,
        "qasper_binary_judge_summary.csv",
    )
    disagreements_path = os.path.join(
        args.output_dir,
        "qasper_binary_judge_disagreements.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "qasper_binary_judge_summary.txt",
    )
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    summary.to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    disagreements.to_csv(
        disagreements_path,
        index=False,
        encoding="utf-8-sig",
    )

    lines: List[str] = [
        "QASPER strict binary-judge audit summary",
        "========================================",
        "",
        f"Total sampled cases: {len(scored)}",
        f"Certain author labels: {len(certain)}",
        f"Uncertain author labels: {len(scored) - len(certain)}",
        f"Cases with at least one judge disagreement: {len(disagreements)}",
        "",
    ]
    for row in summary.itertuples():
        lines.extend(
            [
                f"{row.judge}:",
                (
                    f"  agreement={row.agreement:.4f}; "
                    f"cohen_kappa={row.cohen_kappa:.4f}"
                ),
                (
                    f"  TP={row.tp}; FP={row.fp}; "
                    f"FN={row.fn}; TN={row.tn}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            (
                "The author conducted the blind review. The sample was "
                "challenge-enriched using legacy judge labels and must not "
                "be used to estimate the population hallucination rate."
            ),
            (
                "Agreement values diagnose the strict binary judging "
                "protocol; they do not establish independent expert "
                "validation."
            ),
        ]
    )
    with open(summary_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nScored audit: {scored_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Disagreements: {disagreements_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
