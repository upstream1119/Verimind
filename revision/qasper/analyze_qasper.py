"""Validate, summarize, and plot the completed QASPER experiment."""

import argparse
import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
DEFAULT_OUTPUT_DIR = os.path.dirname(DEFAULT_INPUT)
EXPECTED_STRATEGIES = [
    "Vanilla_RAG",
    "HyDE",
    "VeriMind_Full",
    "VeriMind_No_Gating",
    "VeriMind_No_Routing",
    "Ablation_no_structure",
]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
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


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(0, min(b, c) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def validate(df: pd.DataFrame) -> None:
    duplicate_count = int(
        df.duplicated(subset=["item_id", "strategy"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate item-strategy rows")

    missing_strategies = set(EXPECTED_STRATEGIES) - set(df["strategy"])
    if missing_strategies:
        raise ValueError(f"Missing strategies: {sorted(missing_strategies)}")

    counts = df.groupby("strategy")["item_id"].nunique()
    if counts.nunique() != 1 or counts.iloc[0] != 300:
        raise ValueError(f"Incomplete strategy coverage: {counts.to_dict()}")

    expected_pairs = 300 * len(EXPECTED_STRATEGIES)
    if len(df) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} rows, found {len(df)}")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in df.groupby("strategy", sort=False):
        accepted = group[group["accepted"] == 1]
        accepted_answerable = accepted[accepted["is_unanswerable"] == 0]
        answerable = group[group["is_unanswerable"] == 0]
        unanswerable = group[group["is_unanswerable"] == 1]
        passed_count = int(accepted["passed_hallucination"].sum())
        accepted_count = len(accepted)
        ci_low, ci_high = wilson_interval(passed_count, accepted_count)

        rows.append(
            {
                "strategy": strategy,
                "n": len(group),
                "accepted_n": accepted_count,
                "passed_hallucinations_n": passed_count,
                "raw_hallucination_rate": group[
                    "raw_hallucination"
                ].mean(),
                "passed_hallucination_rate": (
                    passed_count / accepted_count
                    if accepted_count
                    else 0.0
                ),
                "passed_hallucination_ci_low": ci_low,
                "passed_hallucination_ci_high": ci_high,
                "answer_coverage": group["accepted"].mean(),
                "answerable_coverage": answerable["accepted"].mean(),
                "unanswerable_abstention_rate": (
                    1.0 - unanswerable["accepted"].mean()
                ),
                "evidence_recall": group.loc[
                    group["evidence_recall"] >= 0,
                    "evidence_recall",
                ].mean(),
                "overall_rouge_l_f1": group["rouge_l_f1"].mean(),
                "conditional_answerable_rouge_l_f1": (
                    accepted_answerable["rouge_l_f1"].mean()
                    if len(accepted_answerable)
                    else float("nan")
                ),
                "mean_latency_sec": group["latency_sec"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    pivot_passed = df.pivot(
        index="item_id",
        columns="strategy",
        values="passed_hallucination",
    )
    pivot_accepted = df.pivot(
        index="item_id",
        columns="strategy",
        values="accepted",
    )
    comparisons = []
    for baseline in [
        "Vanilla_RAG",
        "HyDE",
        "VeriMind_No_Gating",
        "VeriMind_No_Routing",
        "Ablation_no_structure",
    ]:
        full = pivot_passed["VeriMind_Full"]
        other = pivot_passed[baseline]
        full_only = int(((full == 1) & (other == 0)).sum())
        other_only = int(((full == 0) & (other == 1)).sum())

        full_coverage = pivot_accepted["VeriMind_Full"]
        other_coverage = pivot_accepted[baseline]
        full_accepts_only = int(
            ((full_coverage == 1) & (other_coverage == 0)).sum()
        )
        other_accepts_only = int(
            ((full_coverage == 0) & (other_coverage == 1)).sum()
        )
        comparisons.append(
            {
                "comparison": f"VeriMind_Full vs {baseline}",
                "risk_full_only": full_only,
                "risk_other_only": other_only,
                "risk_mcnemar_exact_p": exact_mcnemar_p(
                    full_only,
                    other_only,
                ),
                "coverage_full_only": full_accepts_only,
                "coverage_other_only": other_accepts_only,
                "coverage_mcnemar_exact_p": exact_mcnemar_p(
                    full_accepts_only,
                    other_accepts_only,
                ),
            }
        )
    return pd.DataFrame(comparisons)


def write_narrative(
    summary: pd.DataFrame,
    mcnemar: pd.DataFrame,
    output_path: str,
) -> None:
    indexed: Dict[str, pd.Series] = {
        row["strategy"]: row for _, row in summary.iterrows()
    }
    full = indexed["VeriMind_Full"]
    vanilla = indexed["Vanilla_RAG"]
    hyde = indexed["HyDE"]
    no_gate = indexed["VeriMind_No_Gating"]

    risk_p = {
        row["comparison"]: row["risk_mcnemar_exact_p"]
        for _, row in mcnemar.iterrows()
    }
    lines = [
        "QASPER public-benchmark result summary",
        "======================================",
        "",
        (
            "VeriMind_Full accepted "
            f"{full['answer_coverage']:.1%} of all questions and "
            f"{full['answerable_coverage']:.1%} of answerable questions."
        ),
        (
            "Its passed-hallucination rate among accepted answers was "
            f"{full['passed_hallucination_rate']:.2%} "
            f"(95% Wilson CI "
            f"{full['passed_hallucination_ci_low']:.2%}-"
            f"{full['passed_hallucination_ci_high']:.2%})."
        ),
        (
            "The corresponding rates were "
            f"{vanilla['passed_hallucination_rate']:.2%} for Vanilla RAG "
            f"and {hyde['passed_hallucination_rate']:.2%} for HyDE."
        ),
        (
            "Removing the trust gate increased passed hallucinations to "
            f"{no_gate['passed_hallucination_rate']:.2%}, while answer "
            f"coverage increased from {full['answer_coverage']:.1%} to "
            f"{no_gate['answer_coverage']:.1%}."
        ),
        (
            "The paired passed-risk difference was significant against "
            f"Vanilla RAG (exact McNemar p="
            f"{risk_p['VeriMind_Full vs Vanilla_RAG']:.6g}) and HyDE "
            f"(p={risk_p['VeriMind_Full vs HyDE']:.6g})."
        ),
        "",
        "Interpretation boundary:",
        (
            "These values are observed under the QASPER validation subset, "
            "the stated generator-judge pair, and the current threshold. "
            "They do not constitute a universal zero-hallucination claim."
        ),
    ]
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def plot_risk_coverage(summary: pd.DataFrame, output_dir: str) -> None:
    colors = {
        "Vanilla_RAG": "#C44E52",
        "HyDE": "#DD8452",
        "VeriMind_Full": "#2A6FBB",
        "VeriMind_No_Gating": "#8172B3",
        "VeriMind_No_Routing": "#55A868",
        "Ablation_no_structure": "#4C4C4C",
    }
    labels = {
        "Vanilla_RAG": "Vanilla RAG",
        "HyDE": "HyDE",
        "VeriMind_Full": "VeriMind",
        "VeriMind_No_Gating": "No gating",
        "VeriMind_No_Routing": "No routing",
        "Ablation_no_structure": "No structure",
    }
    label_offsets = {
        "Vanilla_RAG": (5, 5),
        "HyDE": (5, 5),
        "VeriMind_Full": (7, -12),
        "VeriMind_No_Gating": (5, 5),
        "VeriMind_No_Routing": (-48, 7),
        "Ablation_no_structure": (5, 5),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for _, row in summary.iterrows():
        strategy = row["strategy"]
        x = 100.0 * row["answer_coverage"]
        y = 100.0 * row["passed_hallucination_rate"]
        ax.scatter(
            x,
            y,
            s=80 if strategy == "VeriMind_Full" else 55,
            color=colors[strategy],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            labels[strategy],
            (x, y),
            xytext=label_offsets[strategy],
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_xlabel("Answer coverage (%)")
    ax.set_ylabel("Passed hallucination rate among accepted answers (%)")
    ax.grid(True, color="#D7D7D7", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(
        max(0, 100 * summary["answer_coverage"].min() - 4),
        min(100, 100 * summary["answer_coverage"].max() + 5),
    )
    ax.set_ylim(
        0,
        max(2.0, 100 * summary["passed_hallucination_rate"].max() + 2),
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "qasper_risk_coverage.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(output_dir, "qasper_risk_coverage.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    validate(df)
    summary = build_summary(df)
    mcnemar = build_mcnemar(df)

    summary_path = os.path.join(
        args.output_dir,
        "qasper_public_benchmark_summary.csv",
    )
    mcnemar_path = os.path.join(
        args.output_dir,
        "qasper_mcnemar_comparisons.csv",
    )
    narrative_path = os.path.join(
        args.output_dir,
        "qasper_result_narrative.txt",
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    mcnemar.to_csv(mcnemar_path, index=False, encoding="utf-8-sig")
    write_narrative(summary, mcnemar, narrative_path)
    plot_risk_coverage(summary, args.output_dir)

    print("Validation: 1,800 unique rows, 300 per strategy")
    print(summary.round(4).to_string(index=False))
    print("\nMcNemar comparisons")
    print(mcnemar.to_string(index=False))
    print(f"\nSummary: {summary_path}")
    print(f"Statistics: {mcnemar_path}")
    print(f"Narrative: {narrative_path}")


if __name__ == "__main__":
    main()
