"""Aggregate frozen Gate V2 judgments on held-out QASPER answers."""

import argparse
import os
from typing import Dict, List

import pandas as pd

from revision.qasper.gate_v2 import POLICY_VERSION, aggregate_judgments


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
)
DEFAULT_CASES = os.path.join(
    DEFAULT_INPUT_DIR,
    "qasper_heldout_gate_v2_input.csv",
)
DEFAULT_JUDGMENTS = os.path.join(
    DEFAULT_INPUT_DIR,
    "gate_v2_heldout_normalized_judgments.csv",
)
JUDGES = ["deepseek", "qwen", "glm"]
EXPECTED_CASES = 96
EXPECTED_JUDGMENTS = EXPECTED_CASES * len(JUDGES)


def build_heldout_case_decisions(
    cases: pd.DataFrame,
    judgments: pd.DataFrame,
) -> pd.DataFrame:
    required_cases = {"item_id", "audit_id", "trust_score"}
    required_judgments = {"item_id", "judge", "label"}
    missing_cases = required_cases - set(cases.columns)
    missing_judgments = required_judgments - set(judgments.columns)
    if missing_cases:
        raise ValueError(f"Cases are missing: {sorted(missing_cases)}")
    if missing_judgments:
        raise ValueError(
            f"Judgments are missing: {sorted(missing_judgments)}"
        )
    if len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} cases, found {len(cases)}"
        )
    if len(judgments) != EXPECTED_JUDGMENTS:
        raise ValueError(
            f"Expected {EXPECTED_JUDGMENTS} judgments, "
            f"found {len(judgments)}"
        )
    if cases["item_id"].duplicated().any():
        raise ValueError("Cases contain duplicate item IDs")
    if judgments.duplicated(subset=["item_id", "judge"]).any():
        raise ValueError("Judgments contain duplicate item-judge rows")
    if set(judgments["judge"]) != set(JUDGES):
        raise ValueError("Judgments do not contain exactly the expected judges")
    if "status" in judgments.columns and set(judgments["status"]) != {"ok"}:
        raise ValueError("Every normalized judgment must have status=ok")

    pivot = judgments.pivot(
        index="item_id",
        columns="judge",
        values="label",
    )
    incomplete = pivot[JUDGES].isna().any(axis=1)
    if bool(incomplete.any()):
        raise ValueError(
            "Incomplete held-out judgments for item IDs: "
            f"{pivot.index[incomplete].astype(int).tolist()}"
        )
    decision_rows: List[Dict[str, object]] = []
    for row in pivot.reset_index().itertuples(index=False):
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
    merged = cases.merge(decisions, on="item_id", validate="one_to_one")
    merged["current_gate_reject"] = (
        merged["trust_score"].astype(float) < 5.0
    )
    return merged.sort_values("item_id").reset_index(drop=True)


def build_decision_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(decisions)
    for decision, count in (
        decisions["gate_v2_decision"].value_counts().sort_index().items()
    ):
        rows.append(
            {
                "metric": f"gate_v2_{decision.lower()}",
                "count": int(count),
                "rate": count / total if total else float("nan"),
            }
        )
    for judge in JUDGES:
        count = int(decisions[f"{judge}_label"].eq("unsupported").sum())
        rows.append(
            {
                "metric": f"{judge}_unsupported",
                "count": count,
                "rate": count / total if total else float("nan"),
            }
        )
    current_reject = int(decisions["current_gate_reject"].sum())
    rows.append(
        {
            "metric": "current_gate_reject",
            "count": current_reject,
            "rate": current_reject / total if total else float("nan"),
        }
    )
    rows.append(
        {
            "metric": "evaluated_answers",
            "count": total,
            "rate": 1.0,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases, keep_default_na=False)
    judgments = pd.read_csv(args.judgments, keep_default_na=False)
    decisions = build_heldout_case_decisions(cases, judgments)
    summary = build_decision_summary(decisions)
    disagreement_columns = [f"{judge}_label" for judge in JUDGES]
    disagreements = decisions[
        decisions[disagreement_columns].nunique(axis=1) > 1
    ].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    decisions_path = os.path.join(
        args.output_dir,
        "gate_v2_heldout_case_decisions.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "gate_v2_heldout_decision_summary.csv",
    )
    disagreements_path = os.path.join(
        args.output_dir,
        "gate_v2_heldout_disagreements.csv",
    )
    text_summary_path = os.path.join(
        args.output_dir,
        "gate_v2_heldout_summary.txt",
    )
    decisions.to_csv(decisions_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    disagreements.to_csv(
        disagreements_path,
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "QASPER Gate V2 held-out aggregation",
        "===================================",
        "",
        f"Evaluated answered held-out items: {len(decisions)}",
        f"Policy version: {POLICY_VERSION}",
        "",
        "Decision counts:",
        decisions["gate_v2_decision"].value_counts().sort_index().to_string(),
        "",
        "Judge label counts:",
    ]
    for judge in JUDGES:
        lines.append(
            f"{judge}: "
            + decisions[f"{judge}_label"].value_counts().sort_index().to_string()
        )
    lines.extend(
        [
            "",
            f"Disagreement rows: {len(disagreements)}",
            "Interpretation boundary:",
            (
                "These are frozen Gate V2 decisions on held-out generated "
                "answers. They are not final safety-performance metrics until "
                "author blind labels are collected for the same items."
            ),
        ]
    )
    with open(text_summary_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Case decisions: {decisions_path}")
    print(f"Decision summary: {summary_path}")
    print(f"Disagreements: {disagreements_path}")
    print(f"Summary: {text_summary_path}")


if __name__ == "__main__":
    main()
