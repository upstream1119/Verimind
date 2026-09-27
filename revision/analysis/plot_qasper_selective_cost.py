"""Create the publication figure for QASPER selective quality and cost."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANKING_PATH = (
    PROJECT_ROOT
    / "revision"
    / "artifacts"
    / "1_17_qasper_cheap_gate"
    / "cheap_gate_ranking_comparison.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "revision"
    / "artifacts"
    / "1_18_qasper_cost_aware_cascade"
    / "qasper_cost_aware_summary.csv"
)
BOOTSTRAP_PATH = (
    PROJECT_ROOT
    / "revision"
    / "artifacts"
    / "1_18_qasper_cost_aware_cascade"
    / "qasper_cost_aware_bootstrap.csv"
)
PAPER_OUTPUT = (
    PROJECT_ROOT
    / "revision"
    / "submission"
    / "VeriMind_KAIS_Overleaf"
    / "Fig13_selective_cost_tradeoff.pdf"
)
PREVIEW_OUTPUT = (
    PROJECT_ROOT
    / "revision"
    / "artifacts"
    / "1_18_qasper_cost_aware_cascade"
    / "Fig13_selective_cost_tradeoff_preview.png"
)

COLORS = {
    "cheap_score": "#0F4D92",
    "similarity_score": "#767676",
    "faithfulness_score": "#42949E",
    "legacy_trust_score": "#B64342",
    "No Gate": "#767676",
    "Cheap Only": "#42949E",
    "Always Verify": "#B64342",
    "Cost-Aware Cascade": "#0F4D92",
}
LABELS = {
    "cheap_score": "Cheap quality score",
    "similarity_score": "Retrieval similarity",
    "faithfulness_score": "LLM faithfulness",
    "legacy_trust_score": "Legacy trust score",
}


def _interval(
    bootstrap: pd.DataFrame,
    policy: str,
    metric: str,
) -> tuple[float, float, float]:
    row = bootstrap[
        (bootstrap["policy"] == f"0.55 {policy}")
        & (bootstrap["metric"] == metric)
    ].iloc[0]
    return float(row["mean"]), float(row["ci_low"]), float(row["ci_high"])


def main() -> None:
    ranking = pd.read_csv(RANKING_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)
    operating = summary[summary["risk_budget"] == 0.55].copy()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.45))

    ax = axes[0]
    marker_map = {
        "cheap_score": "o",
        "similarity_score": "s",
        "faithfulness_score": "^",
        "legacy_trust_score": "D",
    }
    annotation_offsets = {
        "cheap_score": (5, -13),
        "similarity_score": (5, 5),
        "faithfulness_score": (5, -13),
        "legacy_trust_score": (5, 5),
    }
    for row in ranking.itertuples(index=False):
        ax.scatter(
            row.spearman_with_answer_f1,
            row.aurc_observed,
            s=62,
            marker=marker_map[row.score],
            color=COLORS[row.score],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            LABELS[row.score],
            (row.spearman_with_answer_f1, row.aurc_observed),
            xytext=annotation_offsets[row.score],
            textcoords="offset points",
            fontsize=9.0,
        )
    ax.axvline(0.0, color="#CFCECE", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(-0.28, 0.52)
    ax.set_ylim(0.27, 0.415)
    ax.set_xlabel("Spearman correlation with Answer F1")
    ax.set_ylabel("Observed AURC (lower is better)")
    ax.set_title("(a) Held-out ranking quality", loc="left", fontweight="bold")
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.6, zorder=0)

    ax = axes[1]
    offsets = {
        "No Gate": (-8, 17),
        "Cheap Only": (12, -24),
        "Always Verify": (18, -32),
        "Cost-Aware Cascade": (12, 32),
    }
    horizontal_alignment = {
        "No Gate": "right",
        "Cheap Only": "left",
        "Always Verify": "left",
        "Cost-Aware Cascade": "left",
    }
    markers = {
        "No Gate": "s",
        "Cheap Only": "^",
        "Always Verify": "D",
        "Cost-Aware Cascade": "o",
    }
    for row in operating.itertuples(index=False):
        coverage_mean, coverage_low, coverage_high = _interval(
            bootstrap, row.policy, "coverage"
        )
        risk_mean, risk_low, risk_high = _interval(
            bootstrap, row.policy, "selective_risk"
        )
        ax.errorbar(
            row.coverage,
            row.selective_risk,
            xerr=[[row.coverage - coverage_low], [coverage_high - row.coverage]],
            yerr=[
                [max(0.0, row.selective_risk - risk_low)],
                [max(0.0, risk_high - row.selective_risk)],
            ],
            fmt=markers[row.policy],
            markersize=6.5 + 8.0 * row.verifier_call_rate,
            color=COLORS[row.policy],
            markeredgecolor="black",
            markeredgewidth=0.6,
            ecolor=COLORS[row.policy],
            elinewidth=0.9,
            capsize=2.3,
            alpha=0.95,
            zorder=3,
        )
        label = f"{row.policy}\nverify {row.verifier_call_rate:.1%}"
        ax.annotate(
            label,
            (row.coverage, row.selective_risk),
            xytext=offsets[row.policy],
            textcoords="offset points",
            fontsize=9.0,
            linespacing=1.05,
            ha=horizontal_alignment[row.policy],
        )
    ax.axhline(0.55, color="#272727", linewidth=0.9, linestyle="--")
    ax.set_xlim(0.0, 0.69)
    ax.set_ylim(0.30, 0.68)
    ax.set_xlabel("Overall answer coverage")
    ax.set_ylabel("Selective risk (lower is better)")
    ax.set_title("(b) Frozen policies at risk budget 0.55", loc="left", fontweight="bold")
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.6, zorder=0)

    fig.tight_layout(w_pad=2.4, pad=0.9)
    PAPER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUTPUT, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PREVIEW_OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {PAPER_OUTPUT}")
    print(f"Saved {PREVIEW_OUTPUT}")


if __name__ == "__main__":
    main()
