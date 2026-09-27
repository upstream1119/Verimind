"""Analyze the completed random QASPER author audit."""

import argparse
import itertools
import math
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_9_qasper_random_author_audit",
)
DEFAULT_BATCHES = [
    os.path.join(
        DEFAULT_INPUT_DIR,
        f"qasper_random_author_audit_batch_{batch_no:02d}.csv",
    )
    for batch_no in range(1, 4)
]
DEFAULT_KEY = os.path.join(
    DEFAULT_INPUT_DIR,
    "qasper_random_author_audit_150_key.csv",
)
STRATEGY_ORDER = ["Vanilla_RAG", "HyDE", "VeriMind_Full"]


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


def load_completed_batches(paths: Sequence[str]) -> pd.DataFrame:
    frames = [
        pd.read_csv(path, keep_default_na=False)
        for path in paths
    ]
    completed = pd.concat(frames, ignore_index=True)
    if len(completed) != 150:
        raise ValueError(
            f"Expected 150 completed rows, found {len(completed)}"
        )
    if completed["audit_id"].duplicated().any():
        raise ValueError("Completed batches contain duplicate audit_id values")
    completed["author_label"] = completed["author_label"].map(
        normalize_label
    )
    allowed = {"supported", "unsupported", "uncertain"}
    invalid = sorted(set(completed["author_label"]) - allowed)
    if invalid:
        raise ValueError(
            "Complete every label with supported/unsupported/uncertain. "
            f"Invalid or blank values: {invalid}"
        )
    return completed


def build_strategy_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        group = scored[scored["strategy"] == strategy]
        supported = int((group["author_label"] == "supported").sum())
        unsupported = int(
            (group["author_label"] == "unsupported").sum()
        )
        uncertain = int((group["author_label"] == "uncertain").sum())
        certain_n = supported + unsupported
        risk = unsupported / certain_n if certain_n else float("nan")
        ci_low, ci_high = wilson_interval(unsupported, certain_n)
        conservative_events = unsupported + uncertain
        conservative_risk = conservative_events / len(group)
        conservative_low, conservative_high = wilson_interval(
            conservative_events,
            len(group),
        )
        rows.append(
            {
                "strategy": strategy,
                "sample_n": len(group),
                "supported_n": supported,
                "unsupported_n": unsupported,
                "uncertain_n": uncertain,
                "certain_n": certain_n,
                "author_unsupported_rate": risk,
                "wilson_ci_low": ci_low,
                "wilson_ci_high": ci_high,
                "conservative_unsupported_n": conservative_events,
                "conservative_unsupported_rate": conservative_risk,
                "conservative_wilson_ci_low": conservative_low,
                "conservative_wilson_ci_high": conservative_high,
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_differences(summary: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("strategy")
    rows: List[Dict[str, object]] = []
    for left, right in itertools.combinations(STRATEGY_ORDER, 2):
        left_row = indexed.loc[left]
        right_row = indexed.loc[right]
        difference = (
            left_row["author_unsupported_rate"]
            - right_row["author_unsupported_rate"]
        )
        ci_low = (
            left_row["wilson_ci_low"]
            - right_row["wilson_ci_high"]
        )
        ci_high = (
            left_row["wilson_ci_high"]
            - right_row["wilson_ci_low"]
        )
        rows.append(
            {
                "comparison": f"{left} minus {right}",
                "risk_difference": difference,
                "newcombe_ci_low": ci_low,
                "newcombe_ci_high": ci_high,
                "interval_excludes_zero": bool(
                    ci_low > 0 or ci_high < 0
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-files",
        nargs="+",
        default=DEFAULT_BATCHES,
    )
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--output-dir", default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    completed = load_completed_batches(args.batch_files)
    key = pd.read_csv(args.key)
    if len(key) != 150 or key["audit_id"].duplicated().any():
        raise ValueError("Key must contain 150 unique audit IDs")
    if set(completed["audit_id"]) != set(key["audit_id"]):
        raise ValueError("Completed batches do not match the private key")
    scored = completed.merge(
        key,
        on="audit_id",
        how="inner",
        validate="one_to_one",
    )
    strategy_counts = scored.groupby("strategy").size().to_dict()
    if any(
        strategy_counts.get(strategy) != 50
        for strategy in STRATEGY_ORDER
    ):
        raise ValueError(
            f"Expected 50 cases per strategy, found {strategy_counts}"
        )

    summary = build_strategy_summary(scored)
    differences = build_pairwise_differences(summary)
    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_150_scored.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_summary.csv",
    )
    differences_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_pairwise_differences.csv",
    )
    narrative_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_summary.txt",
    )
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    differences.to_csv(
        differences_path,
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "QASPER random author-audit summary",
        "==================================",
        "",
    ]
    for row in summary.itertuples():
        lines.append(
            f"{row.strategy}: unsupported={row.unsupported_n}/"
            f"{row.certain_n} ({row.author_unsupported_rate:.2%}; "
            f"95% Wilson CI {row.wilson_ci_low:.2%}-"
            f"{row.wilson_ci_high:.2%}), uncertain={row.uncertain_n}"
        )
        lines.append(
            f"  conservative sensitivity: "
            f"{row.conservative_unsupported_n}/{row.sample_n} "
            f"({row.conservative_unsupported_rate:.2%}; "
            f"95% Wilson CI {row.conservative_wilson_ci_low:.2%}-"
            f"{row.conservative_wilson_ci_high:.2%})"
        )
    lines.extend(["", "System-level risk differences:"])
    for row in differences.itertuples():
        lines.append(
            f"- {row.comparison}: {row.risk_difference:.2%} "
            f"(Newcombe-style 95% CI {row.newcombe_ci_low:.2%} to "
            f"{row.newcombe_ci_high:.2%})"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            (
                "These are author-conducted blind labels on independent "
                "random samples of accepted answers, not independent expert "
                "annotations."
            ),
            (
                "Uncertain cases are excluded from the primary denominator "
                "and counted as unsupported in the conservative sensitivity "
                "analysis."
            ),
            (
                "With 50 cases per system, confidence intervals may remain "
                "wide; conclusions should emphasize large differences and "
                "avoid false precision."
            ),
            (
                "Systems were sampled separately, so the same question may "
                "occur for more than one system. Pairwise intervals use a "
                "separate-binomial approximation and are descriptive rather "
                "than a definitive paired significance test."
            ),
        ]
    )
    with open(narrative_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nScored cases: {scored_path}")
    print(f"Summary: {summary_path}")
    print(f"Differences: {differences_path}")
    print(f"Narrative: {narrative_path}")


if __name__ == "__main__":
    main()
