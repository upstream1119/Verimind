"""Validate and summarize the QASPER cross-judge re-evaluation."""

import argparse
import itertools
import math
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_ORIGINAL = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_5_qasper_cross_judge",
    "qasper_cross_judge_results.csv",
)
DEFAULT_OUTPUT_DIR = os.path.dirname(DEFAULT_INPUT)
JUDGES = ["deepseek", "qwen", "glm"]


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


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


def fleiss_kappa(label_rows: Sequence[Sequence[int]]) -> float:
    if not label_rows:
        return float("nan")
    raters = len(label_rows[0])
    if raters < 2 or any(len(row) != raters for row in label_rows):
        raise ValueError("Each item must have the same number of judges")

    item_agreements: List[float] = []
    category_totals = [0, 0]
    for row in label_rows:
        counts = [row.count(0), row.count(1)]
        category_totals[0] += counts[0]
        category_totals[1] += counts[1]
        item_agreements.append(
            (
                counts[0] * counts[0]
                + counts[1] * counts[1]
                - raters
            )
            / (raters * (raters - 1))
        )

    observed = sum(item_agreements) / len(item_agreements)
    total_labels = len(label_rows) * raters
    proportions = [
        category_totals[0] / total_labels,
        category_totals[1] / total_labels,
    ]
    expected = sum(value * value for value in proportions)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return (observed - expected) / (1.0 - expected)


def validate_and_pivot(
    original: pd.DataFrame,
    judgments: pd.DataFrame,
) -> pd.DataFrame:
    accepted = original[original["accepted"] == 1][
        ["item_id", "strategy"]
    ].copy()
    accepted["item_id"] = accepted["item_id"].astype(int)
    accepted_keys = set(
        zip(accepted["item_id"], accepted["strategy"])
    )

    required = {
        "item_id",
        "strategy",
        "judge",
        "faith_score",
        "unsupported_event",
    }
    missing = required - set(judgments.columns)
    if missing:
        raise ValueError(f"Judgment file is missing: {sorted(missing)}")
    judgments = judgments.copy()
    judgments["item_id"] = judgments["item_id"].astype(int)
    judgments["unsupported_event"] = judgments[
        "unsupported_event"
    ].astype(int)

    invalid_judges = set(judgments["judge"]) - set(JUDGES)
    if invalid_judges:
        raise ValueError(f"Unexpected judges: {sorted(invalid_judges)}")
    duplicate_count = int(
        judgments.duplicated(
            subset=["item_id", "strategy", "judge"]
        ).sum()
    )
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate judgments")

    judgment_keys = set(
        zip(judgments["item_id"], judgments["strategy"])
    )
    extra_keys = judgment_keys - accepted_keys
    if extra_keys:
        raise ValueError(
            f"Found {len(extra_keys)} judgments for non-accepted answers"
        )

    expected_records = len(accepted_keys) * len(JUDGES)
    if len(judgments) != expected_records:
        raise ValueError(
            f"Incomplete run: expected {expected_records} records, "
            f"found {len(judgments)}"
        )

    pivot = judgments.pivot(
        index=["item_id", "strategy"],
        columns="judge",
        values="unsupported_event",
    ).reset_index()
    missing_columns = set(JUDGES) - set(pivot.columns)
    if missing_columns or pivot[JUDGES].isna().any().any():
        raise ValueError("At least one accepted answer lacks three judges")
    if set(zip(pivot["item_id"], pivot["strategy"])) != accepted_keys:
        raise ValueError("Judgment keys do not match accepted-answer keys")
    return pivot


def build_agreement(pivot: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for left, right in itertools.combinations(JUDGES, 2):
        left_values = pivot[left].astype(int).tolist()
        right_values = pivot[right].astype(int).tolist()
        rows.append(
            {
                "metric": "pairwise_cohen_kappa",
                "judges": f"{left}_vs_{right}",
                "n": len(pivot),
                "agreement": sum(
                    a == b
                    for a, b in zip(left_values, right_values)
                )
                / len(pivot),
                "kappa": cohen_kappa(left_values, right_values),
            }
        )
    label_rows = pivot[JUDGES].astype(int).values.tolist()
    rows.append(
        {
            "metric": "fleiss_kappa",
            "judges": "_".join(JUDGES),
            "n": len(pivot),
            "agreement": (
                pivot[JUDGES].nunique(axis=1).eq(1).mean()
            ),
            "kappa": fleiss_kappa(label_rows),
        }
    )
    return pd.DataFrame(rows)


def build_strategy_summary(pivot: pd.DataFrame) -> pd.DataFrame:
    data = pivot.copy()
    data["majority_vote"] = (
        data[JUDGES].sum(axis=1) >= 2
    ).astype(int)
    rows: List[Dict[str, object]] = []
    for strategy, group in data.groupby("strategy", sort=False):
        for judge in JUDGES + ["majority_vote"]:
            events = int(group[judge].sum())
            total = len(group)
            ci_low, ci_high = wilson_interval(events, total)
            rows.append(
                {
                    "strategy": strategy,
                    "judge": judge,
                    "accepted_n": total,
                    "unsupported_events_n": events,
                    "unsupported_rate": events / total,
                    "wilson_ci_low": ci_low,
                    "wilson_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    agreement: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    output_path: str,
) -> None:
    majority = strategy_summary[
        strategy_summary["judge"] == "majority_vote"
    ]
    lines = [
        "QASPER cross-judge robustness summary",
        "=====================================",
        "",
        "Agreement:",
    ]
    for row in agreement.itertuples():
        lines.append(
            f"- {row.judges}: agreement={row.agreement:.4f}, "
            f"kappa={row.kappa:.4f}, n={row.n}"
        )
    lines.extend(["", "Majority-vote unsupported-event rates:"])
    for row in majority.itertuples():
        lines.append(
            f"- {row.strategy}: {row.unsupported_events_n}/"
            f"{row.accepted_n} ({row.unsupported_rate:.2%}; "
            f"95% Wilson CI {row.wilson_ci_low:.2%}-"
            f"{row.wilson_ci_high:.2%})"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            (
                "This analysis tests judge robustness on accepted answers. "
                "It does not replace human validation or establish a "
                "universal hallucination rate."
            ),
        ]
    )
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default=DEFAULT_ORIGINAL)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    original = pd.read_csv(args.original)
    judgments = pd.read_csv(args.input)
    pivot = validate_and_pivot(original, judgments)
    agreement = build_agreement(pivot)
    strategy_summary = build_strategy_summary(pivot)

    agreement_path = os.path.join(
        args.output_dir,
        "qasper_cross_judge_agreement.csv",
    )
    strategy_path = os.path.join(
        args.output_dir,
        "qasper_cross_judge_strategy_summary.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "qasper_cross_judge_summary.txt",
    )
    agreement.to_csv(agreement_path, index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(
        strategy_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_summary(agreement, strategy_summary, summary_path)

    print(f"Validation: {len(pivot)} accepted answers, three judges each")
    print("\n" + agreement.round(4).to_string(index=False))
    print("\n" + strategy_summary.round(4).to_string(index=False))
    print(f"\nAgreement: {agreement_path}")
    print(f"Strategy summary: {strategy_path}")
    print(f"Narrative: {summary_path}")


if __name__ == "__main__":
    main()
