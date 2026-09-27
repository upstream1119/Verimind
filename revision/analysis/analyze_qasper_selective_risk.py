"""Analyze QASPER selective answering with dataset-native Answer F1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
from collections import Counter
from typing import Iterable, Sequence

import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_16_qasper_selective_risk",
)
SOURCE_STRATEGY = "VeriMind_No_Gating"
DEFAULT_THRESHOLD = 5.0
REFUSAL_FLAGS = (
    "information not found",
    "not found in the context",
    "cannot be determined",
    "unable to determine",
    "insufficient evidence",
    "rejected",
    "unanswerable",
)


def normalize_answer(text: str) -> str:
    """Apply the normalization used by the official QASPER evaluator."""

    lowered = (text or "").lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1(prediction: str, reference: str) -> float:
    """Compute the SQuAD-style token F1 used by QASPER."""

    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    matching = sum(common.values())
    if matching == 0:
        return 0.0
    precision = matching / len(prediction_tokens)
    recall = matching / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def best_answer_f1(prediction: str, references: Sequence[str]) -> float:
    """Return the maximum token F1 over the available reference answers."""

    valid_references = [str(reference) for reference in references if reference]
    if not valid_references:
        return 0.0
    return max(token_f1(prediction, reference) for reference in valid_references)


def is_refusal(answer: str) -> bool:
    """Identify explicit generator abstentions using the experiment protocol."""

    lowered = (answer or "").lower()
    return any(flag in lowered for flag in REFUSAL_FLAGS)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deterministic_stratified_split(
    rows: pd.DataFrame,
    calibration_size: int = 100,
) -> pd.DataFrame:
    """Create a deterministic split stratified by answerability."""

    required = {"question_id", "is_unanswerable"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if rows["question_id"].astype(str).duplicated().any():
        raise ValueError("question_id must be unique before splitting")
    if not 0 < calibration_size < len(rows):
        raise ValueError("calibration_size must be between zero and dataset size")

    result = rows.copy()
    result["question_id"] = result["question_id"].astype(str)
    result["is_unanswerable"] = pd.to_numeric(
        result["is_unanswerable"], errors="raise"
    ).astype(int)
    split_by_id: dict[str, str] = {}
    remaining = calibration_size
    groups = list(result.groupby("is_unanswerable", sort=True))
    for group_index, (_, group) in enumerate(groups):
        if group_index == len(groups) - 1:
            group_calibration_size = remaining
        else:
            proportion = len(group) / len(result)
            group_calibration_size = int(round(calibration_size * proportion))
            remaining -= group_calibration_size
        ranked_ids = sorted(
            group["question_id"].tolist(),
            key=lambda question_id: (_stable_hash(question_id), question_id),
        )
        calibration_ids = set(ranked_ids[:group_calibration_size])
        for question_id in ranked_ids:
            split_by_id[question_id] = (
                "calibration" if question_id in calibration_ids else "test"
            )

    result["split"] = result["question_id"].map(split_by_id)
    return result


def replay_gate(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Replay a trust threshold on fixed No-Gating generated answers."""

    required = {"generated_answer", "trust_score"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing gate columns: {sorted(missing)}")

    result = rows.copy()
    result["trust_score"] = pd.to_numeric(result["trust_score"], errors="raise")
    result["natural_refusal"] = result["generated_answer"].map(is_refusal)
    result["gate_accepted"] = (
        (~result["natural_refusal"]) & (result["trust_score"] >= threshold)
    ).astype(int)
    result["abstention_source"] = "answered"
    result.loc[result["natural_refusal"], "abstention_source"] = "generator"
    result.loc[
        (~result["natural_refusal"]) & (result["gate_accepted"] == 0),
        "abstention_source",
    ] = "gate"
    return result


def compute_operating_point(rows: pd.DataFrame, threshold: float) -> dict:
    """Summarize one selective-answering operating point."""

    replayed = replay_gate(rows, threshold)
    if "answer_f1" not in replayed.columns:
        raise ValueError("answer_f1 is required for selective-risk analysis")
    replayed["answer_f1"] = pd.to_numeric(replayed["answer_f1"], errors="raise")
    accepted = replayed[replayed["gate_accepted"] == 1]
    accepted_n = len(accepted)
    mean_answer_f1 = (
        float(accepted["answer_f1"].mean()) if accepted_n else float("nan")
    )
    return {
        "threshold": float(threshold),
        "n": len(replayed),
        "accepted_n": accepted_n,
        "coverage": accepted_n / len(replayed) if len(replayed) else 0.0,
        "mean_answer_f1": mean_answer_f1,
        "selective_risk": (
            1.0 - mean_answer_f1 if accepted_n else float("nan")
        ),
        "overall_answer_f1": (
            float(accepted["answer_f1"].sum()) / len(replayed)
            if len(replayed)
            else 0.0
        ),
        "generator_abstentions_n": int(
            (replayed["abstention_source"] == "generator").sum()
        ),
        "gate_abstentions_n": int(
            (replayed["abstention_source"] == "gate").sum()
        ),
    }


def build_risk_coverage_curve(
    rows: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate observed trust-score thresholds without regenerating answers."""

    if thresholds is None:
        observed = pd.to_numeric(rows["trust_score"], errors="raise")
        threshold_values = sorted(
            {0.0, DEFAULT_THRESHOLD, *observed.astype(float).tolist()}
        )
        threshold_values.append(float(observed.max()) + 1e-9)
    else:
        threshold_values = sorted({float(value) for value in thresholds})
    return pd.DataFrame(
        [compute_operating_point(rows, threshold) for threshold in threshold_values]
    )


def compute_aurc(curve: pd.DataFrame) -> float:
    """Integrate selective risk over the observed coverage interval."""

    clean = (
        curve[["coverage", "selective_risk"]]
        .dropna()
        .groupby("coverage", as_index=False)["selective_risk"]
        .min()
        .sort_values("coverage")
    )
    if len(clean) < 2:
        return 0.0
    area = 0.0
    points = list(clean.itertuples(index=False, name=None))
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        area += (right_x - left_x) * (left_y + right_y) / 2
    return float(area)


def _load_source_rows() -> pd.DataFrame:
    rows = pd.read_csv(INPUT_PATH)
    rows = rows[rows["strategy"] == SOURCE_STRATEGY].copy()
    if len(rows) != 300:
        raise ValueError(f"Expected 300 {SOURCE_STRATEGY} rows, found {len(rows)}")
    if rows["question_id"].astype(str).duplicated().any():
        raise ValueError("Expected one source row per QASPER question")
    rows["gold_answers"] = rows["gold_answers_json"].map(json.loads)
    rows["answer_f1"] = [
        best_answer_f1(prediction, references)
        for prediction, references in zip(
            rows["generated_answer"], rows["gold_answers"]
        )
    ]
    return deterministic_stratified_split(rows, calibration_size=100)


def _select_rows_at_threshold(curve: pd.DataFrame, threshold: float) -> pd.Series:
    exact = curve[curve["threshold"] == threshold]
    if exact.empty:
        raise ValueError(f"Threshold {threshold} is missing from curve")
    return exact.iloc[0]


def _build_summary(split_rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for split_name in ("calibration", "test"):
        subset = split_rows[split_rows["split"] == split_name]
        curve = build_risk_coverage_curve(subset)
        no_gate = _select_rows_at_threshold(curve, 0.0)
        default_gate = _select_rows_at_threshold(curve, DEFAULT_THRESHOLD)
        summaries.append(
            {
                "split": split_name,
                "n": len(subset),
                "aurc_observed": compute_aurc(curve),
                "no_gate_coverage": no_gate["coverage"],
                "no_gate_answer_f1": no_gate["mean_answer_f1"],
                "no_gate_selective_risk": no_gate["selective_risk"],
                "threshold_5_coverage": default_gate["coverage"],
                "threshold_5_answer_f1": default_gate["mean_answer_f1"],
                "threshold_5_selective_risk": default_gate["selective_risk"],
            }
        )
    return pd.DataFrame(summaries)


def _write_report(summary: pd.DataFrame, output_path: str) -> None:
    lines = [
        "# QASPER Selective-Risk Reanalysis",
        "",
        f"Source strategy: `{SOURCE_STRATEGY}`",
        "",
        "The analysis replays trust thresholds on fixed generated answers. "
        "Answer quality uses the official QASPER-style token F1; it does not "
        "use the LLM judge that produced the gate score.",
        "",
        "| Split | N | No-gate coverage | No-gate Answer F1 | "
        "Threshold-5 coverage | Threshold-5 Answer F1 | Observed AURC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.n} | {row.no_gate_coverage:.4f} | "
            f"{row.no_gate_answer_f1:.4f} | {row.threshold_5_coverage:.4f} | "
            f"{row.threshold_5_answer_f1:.4f} | {row.aurc_observed:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Answer F1 is a dataset-native utility measure, not a complete "
            "factual-support label. Human and independent-judge audits remain "
            "separate evidence sources.",
        ]
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as report_file:
        report_file.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = _load_source_rows()
    manifest_columns = [
        "item_id",
        "question_id",
        "paper_id",
        "is_unanswerable",
        "split",
    ]
    rows[manifest_columns].to_csv(
        os.path.join(OUTPUT_DIR, "qasper_split_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    curves = []
    for split_name in ("calibration", "test"):
        split_curve = build_risk_coverage_curve(rows[rows["split"] == split_name])
        split_curve.insert(0, "split", split_name)
        curves.append(split_curve)
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(
        os.path.join(OUTPUT_DIR, "qasper_risk_coverage_curve.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = _build_summary(rows)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "qasper_selective_risk_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(
        summary,
        os.path.join(OUTPUT_DIR, "qasper_selective_risk_report.md"),
    )
    print(summary.round(4).to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
