"""Analyze the completed QASPER gate-paired author audit."""

import argparse
import math
import os
from typing import Dict, List, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_EXISTING_COMPLETED = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_10_qasper_no_gating_author_audit",
    "qasper_no_gating_author_audit_50_completed.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_12_qasper_gate_paired_author_audit",
)
DEFAULT_REMAINING_COMPLETED = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_gate_paired_remaining_73_completed.csv",
)
DEFAULT_KEY = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_gate_paired_111_key.csv",
)
ALLOWED_LABELS = {"supported", "unsupported", "uncertain"}
EXPECTED_POOL_SIZE = 111


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


def fisher_exact_two_sided(
    a: int,
    b: int,
    c: int,
    d: int,
) -> Tuple[float, float]:
    row_1 = a + b
    row_2 = c + d
    col_1 = a + c
    total = row_1 + row_2
    if total == 0:
        return float("nan"), float("nan")

    denominator = math.comb(total, row_1)

    def probability(x: int) -> float:
        return (
            math.comb(col_1, x)
            * math.comb(total - col_1, row_1 - x)
            / denominator
        )

    minimum = max(0, row_1 - (total - col_1))
    maximum = min(row_1, col_1)
    observed = probability(a)
    p_value = sum(
        probability(x)
        for x in range(minimum, maximum + 1)
        if probability(x) <= observed + 1e-12
    )
    if b * c == 0:
        odds_ratio = (
            float("inf") if a * d > 0 else float("nan")
        )
    else:
        odds_ratio = (a * d) / (b * c)
    return odds_ratio, min(1.0, p_value)


def load_labels(path: str, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"audit_id", "author_label", "author_notes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing columns: {sorted(missing)}"
        )
    if frame["audit_id"].duplicated().any():
        raise ValueError(f"{source_name} contains duplicate audit IDs")
    frame["author_label"] = frame["author_label"].map(normalize_label)
    invalid = sorted(set(frame["author_label"]) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(
            f"{source_name} has blank or invalid labels: {invalid}"
        )
    return frame[
        ["audit_id", "author_label", "author_notes"]
    ].copy()


def combine_scored_pool(
    key: pd.DataFrame,
    existing: pd.DataFrame,
    remaining: pd.DataFrame,
) -> pd.DataFrame:
    if len(key) != EXPECTED_POOL_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_POOL_SIZE} key rows, found {len(key)}"
        )
    if key["pair_id"].duplicated().any():
        raise ValueError("Key contains duplicate pair IDs")
    if key["source_audit_id"].duplicated().any():
        raise ValueError("Key contains duplicate source audit IDs")

    labels = pd.concat([existing, remaining], ignore_index=True)
    if labels["audit_id"].duplicated().any():
        raise ValueError("Completed audit files overlap in audit IDs")
    expected_ids = set(key["source_audit_id"])
    available_ids = set(labels["audit_id"])
    missing = sorted(expected_ids - available_ids)
    if missing:
        raise ValueError(
            f"Completed files are missing {len(missing)} paired labels"
        )
    scored = key.merge(
        labels,
        left_on="source_audit_id",
        right_on="audit_id",
        how="left",
        validate="one_to_one",
    )
    return scored.drop(columns=["audit_id"])


def build_gate_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for full_accepted, gate_result in [
        (0, "rejected_by_full_gate"),
        (1, "accepted_by_full_gate"),
    ]:
        group = scored[scored["full_accepted"] == full_accepted]
        rows.append(
            {
                "gate_result": gate_result,
                "sample_n": len(group),
                "supported_n": int(
                    (group["author_label"] == "supported").sum()
                ),
                "unsupported_n": int(
                    (group["author_label"] == "unsupported").sum()
                ),
                "uncertain_n": int(
                    (group["author_label"] == "uncertain").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    scored: pd.DataFrame,
    gate_table: pd.DataFrame,
) -> pd.DataFrame:
    supported = int((scored["author_label"] == "supported").sum())
    unsupported = int((scored["author_label"] == "unsupported").sum())
    uncertain = int((scored["author_label"] == "uncertain").sum())
    certain = supported + unsupported
    rejected = gate_table.set_index("gate_result").loc[
        "rejected_by_full_gate"
    ]
    accepted = gate_table.set_index("gate_result").loc[
        "accepted_by_full_gate"
    ]

    unsupported_rejected = int(rejected["unsupported_n"])
    supported_rejected = int(rejected["supported_n"])
    unsupported_accepted = int(accepted["unsupported_n"])
    supported_accepted = int(accepted["supported_n"])
    accepted_certain = unsupported_accepted + supported_accepted

    metrics = [
        (
            "no_gating_unsupported_rate",
            unsupported,
            certain,
        ),
        (
            "full_gate_sensitivity_for_unsupported",
            unsupported_rejected,
            unsupported,
        ),
        (
            "full_gate_false_rejection_rate",
            supported_rejected,
            supported,
        ),
        (
            "full_passed_unsupported_rate",
            unsupported_accepted,
            accepted_certain,
        ),
        (
            "conservative_no_gating_risk",
            unsupported + uncertain,
            len(scored),
        ),
        (
            "conservative_full_passed_risk",
            unsupported_accepted + int(accepted["uncertain_n"]),
            int(accepted["sample_n"]),
        ),
    ]
    rows = []
    for metric, numerator, denominator in metrics:
        rate = numerator / denominator if denominator else float("nan")
        ci_low, ci_high = wilson_interval(numerator, denominator)
        rows.append(
            {
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "rate": rate,
                "wilson_ci_low": ci_low,
                "wilson_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing-completed",
        default=DEFAULT_EXISTING_COMPLETED,
    )
    parser.add_argument(
        "--remaining-completed",
        default=DEFAULT_REMAINING_COMPLETED,
    )
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    key = pd.read_csv(args.key, keep_default_na=False)
    existing = load_labels(args.existing_completed, "existing audit")
    remaining = load_labels(args.remaining_completed, "remaining audit")
    scored = combine_scored_pool(key, existing, remaining)
    gate_table = build_gate_table(scored)
    summary = build_summary(scored, gate_table)

    indexed = gate_table.set_index("gate_result")
    rejected = indexed.loc["rejected_by_full_gate"]
    accepted = indexed.loc["accepted_by_full_gate"]
    odds_ratio, p_value = fisher_exact_two_sided(
        int(rejected["unsupported_n"]),
        int(rejected["supported_n"]),
        int(accepted["unsupported_n"]),
        int(accepted["supported_n"]),
    )
    fisher = pd.DataFrame(
        [
            {
                "test": "unsupported_enrichment_among_full_rejections",
                "rejected_unsupported": int(rejected["unsupported_n"]),
                "rejected_supported": int(rejected["supported_n"]),
                "accepted_unsupported": int(accepted["unsupported_n"]),
                "accepted_supported": int(accepted["supported_n"]),
                "odds_ratio": odds_ratio,
                "fisher_two_sided_p": p_value,
            }
        ]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_111_scored.csv",
    )
    gate_table_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_gate_table.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_summary.csv",
    )
    fisher_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_fisher.csv",
    )
    narrative_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_summary.txt",
    )
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    gate_table.to_csv(
        gate_table_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fisher.to_csv(fisher_path, index=False, encoding="utf-8-sig")

    metric_rows = summary.set_index("metric")
    lines = [
        "QASPER gate-paired author-audit summary",
        "=======================================",
        "",
        (
            f"Paired pool: {len(scored)} cases with identical generated "
            "answers and retrieved contexts."
        ),
        (
            "No-Gating accepted all cases; the Full gate decision was hidden "
            "during author labeling."
        ),
        "",
    ]
    for metric in summary["metric"]:
        row = metric_rows.loc[metric]
        lines.append(
            f"{metric}: {int(row['numerator'])}/"
            f"{int(row['denominator'])} ({row['rate']:.2%}; "
            f"95% Wilson CI {row['wilson_ci_low']:.2%}-"
            f"{row['wilson_ci_high']:.2%})"
        )
    lines.extend(
        [
            "",
            (
                "Fisher exact test for unsupported-answer enrichment among "
                f"Full rejections: odds ratio={odds_ratio:.3g}, "
                f"two-sided p={p_value:.4f}."
            ),
            "",
            "Interpretation boundary:",
            (
                "This is a finite paired-pool diagnostic, not an independent "
                "population sample or a claim of zero residual risk."
            ),
            (
                "Primary rates exclude uncertain labels; conservative rates "
                "count uncertain labels as unsupported."
            ),
            (
                "Labels are author-conducted blind evidence audits, not "
                "independent expert annotations."
            ),
        ]
    )
    with open(narrative_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nScored pool: {scored_path}")
    print(f"Gate table: {gate_table_path}")
    print(f"Summary: {summary_path}")
    print(f"Fisher test: {fisher_path}")
    print(f"Narrative: {narrative_path}")


if __name__ == "__main__":
    main()
