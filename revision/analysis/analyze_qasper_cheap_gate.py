"""Test whether cheap deployment features can rank QASPER answer quality."""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from revision.analysis.analyze_qasper_selective_risk import (
    best_answer_f1,
    compute_aurc,
    is_refusal,
)


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DEVELOPMENT_PATH = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
HELDOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
    "qasper_heldout_no_gating_results.csv",
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_17_qasper_cheap_gate",
)
SOURCE_STRATEGY = "VeriMind_No_Gating"

NUMERIC_FEATURES = (
    "average_similarity",
    "answer_context_token_coverage",
    "answer_number_support",
    "answer_length",
    "question_length",
)
CATEGORICAL_FEATURES = (
    "question_type",
    "routed_granularity",
)
CHEAP_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FORBIDDEN_MODEL_COLUMNS = {
    "gold_answer",
    "gold_answers_json",
    "evidence_recall",
    "pre_gate_faith_score",
    "trust_score",
    "raw_hallucination",
    "passed_hallucination",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?%?")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def answer_context_token_coverage(answer: str, context: str) -> float:
    """Return the fraction of unique answer tokens found in the context."""

    answer_tokens = set(_tokens(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(_tokens(context))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def answer_number_support(answer: str, context: str) -> float:
    """Return the fraction of answer numbers that also occur in the context."""

    answer_numbers = set(NUMBER_PATTERN.findall(answer or ""))
    if not answer_numbers:
        return 0.0
    context_numbers = set(NUMBER_PATTERN.findall(context or ""))
    return len(answer_numbers & context_numbers) / len(answer_numbers)


def classify_question(question: str) -> str:
    """Map a question to a small, deterministic lexical intent class."""

    lowered = (question or "").strip().lower()
    if lowered.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
        return "boolean"
    if any(term in lowered for term in ("how many", "how much", "what score", "what value")):
        return "numeric"
    if lowered.startswith(("why ", "how ")):
        return "explanatory"
    if lowered.startswith(("which ", "what ", "who ", "where ", "when ")):
        return "factoid"
    return "other"


def extract_deployment_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Build features available without gold labels or an LLM verifier."""

    required = {
        "question",
        "generated_answer",
        "retrieved_context",
        "average_similarity",
        "routed_granularity",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing feature source columns: {sorted(missing)}")
    if not FORBIDDEN_MODEL_COLUMNS.isdisjoint(CHEAP_FEATURE_COLUMNS):
        raise RuntimeError("Cheap feature specification includes forbidden columns")

    features = pd.DataFrame(index=rows.index)
    features["average_similarity"] = pd.to_numeric(
        rows["average_similarity"], errors="raise"
    )
    features["answer_context_token_coverage"] = [
        answer_context_token_coverage(answer, context)
        for answer, context in zip(
            rows["generated_answer"], rows["retrieved_context"]
        )
    ]
    features["answer_number_support"] = [
        answer_number_support(answer, context)
        for answer, context in zip(
            rows["generated_answer"], rows["retrieved_context"]
        )
    ]
    features["answer_length"] = rows["generated_answer"].map(
        lambda value: len(_tokens(value))
    )
    features["question_length"] = rows["question"].map(
        lambda value: len(_tokens(value))
    )
    features["question_type"] = rows["question"].map(classify_question)
    features["routed_granularity"] = rows["routed_granularity"].fillna("UNKNOWN")
    return features.loc[:, CHEAP_FEATURE_COLUMNS]


def assert_no_split_overlap(
    development_rows: pd.DataFrame,
    heldout_rows: pd.DataFrame,
) -> None:
    """Fail closed if development and held-out question IDs overlap."""

    development_ids = set(development_rows["question_id"].astype(str))
    heldout_ids = set(heldout_rows["question_id"].astype(str))
    overlap = development_ids & heldout_ids
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"Development/held-out question overlap: {examples}")


def fit_cheap_quality_model(
    features: pd.DataFrame,
    answer_f1: Iterable[float],
) -> Pipeline:
    """Fit a deterministic ridge model to continuous QASPER Answer F1."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "ridge",
                RidgeCV(
                    alphas=np.array([0.01, 0.1, 1.0, 10.0, 100.0]),
                    cv=5,
                    scoring="neg_mean_absolute_error",
                ),
            ),
        ]
    )
    model.fit(features, np.asarray(list(answer_f1), dtype=float))
    return model


def predict_quality(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return bounded quality scores for ranking selective answers."""

    return np.clip(np.asarray(model.predict(features), dtype=float), 0.0, 1.0)


def build_ranking_curve(
    rows: pd.DataFrame,
    score_column: str,
    total_questions: int,
) -> pd.DataFrame:
    """Build a prefix risk-coverage curve from a ranking score."""

    ranked = rows.sort_values(
        [score_column, "question_id"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["accepted_n"] = np.arange(1, len(ranked) + 1)
    ranked["coverage"] = ranked["accepted_n"] / total_questions
    ranked["mean_answer_f1"] = ranked["answer_f1"].expanding().mean()
    ranked["selective_risk"] = 1.0 - ranked["mean_answer_f1"]
    return ranked[
        [
            "accepted_n",
            "coverage",
            "mean_answer_f1",
            "selective_risk",
            score_column,
        ]
    ]


def summarize_ranking(
    rows: pd.DataFrame,
    score_column: str,
    total_questions: int,
) -> dict:
    """Summarize held-out ranking quality and observed AURC."""

    curve = build_ranking_curve(rows, score_column, total_questions)
    correlation = spearmanr(
        rows[score_column].astype(float),
        rows["answer_f1"].astype(float),
    ).statistic
    summary = {
        "score": score_column,
        "candidate_answers_n": len(rows),
        "maximum_coverage": len(rows) / total_questions,
        "spearman_with_answer_f1": float(correlation),
        "aurc_observed": compute_aurc(curve),
    }
    for fraction in (0.25, 0.50, 0.75, 1.00):
        accepted_n = max(1, int(round(len(rows) * fraction)))
        point = curve.iloc[accepted_n - 1]
        label = int(fraction * 100)
        summary[f"top_{label}_coverage"] = float(point["coverage"])
        summary[f"top_{label}_answer_f1"] = float(point["mean_answer_f1"])
    return summary


def _load_rows(path: str, expected_n: int, filter_strategy: bool) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if filter_strategy:
        rows = rows[rows["strategy"] == SOURCE_STRATEGY].copy()
    if len(rows) != expected_n:
        raise ValueError(f"Expected {expected_n} rows from {path}, found {len(rows)}")
    rows["question_id"] = rows["question_id"].astype(str)
    rows["gold_answers"] = rows["gold_answers_json"].map(json.loads)
    rows["answer_f1"] = [
        best_answer_f1(prediction, references)
        for prediction, references in zip(
            rows["generated_answer"], rows["gold_answers"]
        )
    ]
    return rows


def _natural_answers(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[~rows["generated_answer"].map(is_refusal)].copy()


def _write_report(summary: pd.DataFrame, output_path: str) -> None:
    lines = [
        "# QASPER Cheap-Gate Feasibility",
        "",
        "The cheap model is trained on the main 300-question set and evaluated "
        "once on the disjoint 150-question held-out set. It does not use gold "
        "answers, gold evidence, LLM-judge scores, or the existing trust score "
        "as input features.",
        "",
        "| Score | Spearman with Answer F1 | Observed AURC | "
        "Top-25% Answer F1 | Top-50% Answer F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.score} | {row.spearman_with_answer_f1:.4f} | "
            f"{row.aurc_observed:.4f} | {row.top_25_answer_f1:.4f} | "
            f"{row.top_50_answer_f1:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a feasibility test on existing outputs, not a final "
            "deployment policy. A cost-aware cascade should be constructed only "
            "if the cheap score generalizes on the held-out set.",
        ]
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as report_file:
        report_file.write("\n".join(lines) + "\n")


def main() -> None:
    development = _load_rows(DEVELOPMENT_PATH, 300, filter_strategy=True)
    heldout = _load_rows(HELDOUT_PATH, 150, filter_strategy=False)
    assert_no_split_overlap(development, heldout)

    development_answers = _natural_answers(development)
    heldout_answers = _natural_answers(heldout)
    development_features = extract_deployment_features(development_answers)
    heldout_features = extract_deployment_features(heldout_answers)
    model = fit_cheap_quality_model(
        development_features,
        development_answers["answer_f1"],
    )
    heldout_answers["cheap_score"] = predict_quality(model, heldout_features)
    heldout_answers["similarity_score"] = pd.to_numeric(
        heldout_answers["average_similarity"], errors="raise"
    )
    heldout_answers["faithfulness_score"] = pd.to_numeric(
        heldout_answers["pre_gate_faith_score"], errors="raise"
    ) / 10.0
    heldout_answers["legacy_trust_score"] = pd.to_numeric(
        heldout_answers["trust_score"], errors="raise"
    ) / 10.0

    score_columns = (
        "cheap_score",
        "similarity_score",
        "faithfulness_score",
        "legacy_trust_score",
    )
    summary = pd.DataFrame(
        [
            summarize_ranking(heldout_answers, score, total_questions=len(heldout))
            for score in score_columns
        ]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    heldout_output_columns = [
        "item_id",
        "question_id",
        "is_unanswerable",
        "answer_f1",
        *score_columns,
    ]
    heldout_answers[heldout_output_columns].to_csv(
        os.path.join(OUTPUT_DIR, "cheap_gate_heldout_scores.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "cheap_gate_ranking_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(
        summary,
        os.path.join(OUTPUT_DIR, "cheap_gate_report.md"),
    )
    print(summary.round(4).to_string(index=False))
    print(f"\nDevelopment answers: {len(development_answers)}/300")
    print(f"Held-out answers: {len(heldout_answers)}/150")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
