"""Evaluate a cost-aware verifier cascade on existing QASPER outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from revision.analysis.analyze_qasper_cheap_gate import (
    assert_no_split_overlap,
    extract_deployment_features,
    fit_cheap_quality_model,
    predict_quality,
)
from revision.analysis.analyze_qasper_selective_risk import (
    best_answer_f1,
    is_refusal,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
    "1_18_qasper_cost_aware_cascade",
)
SOURCE_STRATEGY = "VeriMind_No_Gating"
RISK_BUDGETS = (0.45, 0.50, 0.55, 0.60)


@dataclass(frozen=True)
class CascadePolicy:
    """Frozen thresholds selected without held-out labels."""

    risk_budget: float
    cheap_low: float
    cheap_high: float
    verified_threshold: float


@dataclass(frozen=True)
class RouteDecision:
    """Observable routing outcome for one candidate answer."""

    decision: str
    accepted: bool
    verifier_called: bool


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def deterministic_train_calibration_split(
    rows: pd.DataFrame,
    calibration_fraction: float = 1.0 / 3.0,
    seed: int = 20260918,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split development questions reproducibly, stratified by answerability."""

    required = {"question_id", "is_unanswerable"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")

    calibration_indices: list[int] = []
    for _, group in rows.groupby("is_unanswerable", sort=True):
        ranked = sorted(
            group.index,
            key=lambda index: (
                _stable_fraction(str(rows.at[index, "question_id"]), seed),
                str(rows.at[index, "question_id"]),
            ),
        )
        calibration_n = max(1, int(round(len(ranked) * calibration_fraction)))
        calibration_indices.extend(ranked[:calibration_n])

    calibration_set = set(calibration_indices)
    train = rows.loc[~rows.index.isin(calibration_set)].copy()
    calibration = rows.loc[rows.index.isin(calibration_set)].copy()
    return (
        train.sort_values("question_id").reset_index(drop=True),
        calibration.sort_values("question_id").reset_index(drop=True),
    )


def route_cascade(
    cheap_score: float,
    verified_score: float,
    policy: CascadePolicy,
) -> RouteDecision:
    """Route one answer through abstain, verify, or direct-answer paths."""

    if cheap_score < policy.cheap_low:
        return RouteDecision("abstain", False, False)
    if cheap_score >= policy.cheap_high:
        return RouteDecision("answer_direct", True, False)
    if verified_score >= policy.verified_threshold:
        return RouteDecision("answer_verified", True, True)
    return RouteDecision("abstain_verified", False, True)


def estimate_verifier_token_budget(question: str, context: str, answer: str) -> int:
    """Estimate verifier input tokens from characters; this is not measured usage."""

    characters = len(str(question)) + len(str(context)) + len(str(answer))
    return max(1, int(np.ceil(characters / 4.0)))


def _policy_metrics(
    evaluated: pd.DataFrame,
    total_questions: int,
) -> dict[str, float | int | bool]:
    accepted = evaluated[evaluated["accepted"]]
    accepted_n = len(accepted)
    mean_f1 = float(accepted["answer_f1"].mean()) if accepted_n else np.nan
    risk = 1.0 - mean_f1 if accepted_n else np.nan
    verifier_calls = int(evaluated["verifier_called"].sum())
    verifier_tokens = float(
        evaluated.loc[
            evaluated["verifier_called"], "estimated_verifier_tokens"
        ].sum()
    )
    return {
        "accepted_n": accepted_n,
        "coverage": accepted_n / total_questions,
        "mean_answer_f1": mean_f1,
        "selective_risk": risk,
        "verifier_calls": verifier_calls,
        "verifier_call_rate": verifier_calls / total_questions,
        "estimated_verifier_input_tokens": verifier_tokens,
    }


def evaluate_frozen_policy(
    candidates: pd.DataFrame,
    total_questions: int,
    policy: CascadePolicy,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Evaluate thresholds without fitting or selecting on evaluation labels."""

    evaluated = candidates.copy()
    routes = [
        route_cascade(cheap, verified, policy)
        for cheap, verified in zip(
            evaluated["cheap_score"], evaluated["verified_score"]
        )
    ]
    evaluated["decision"] = [route.decision for route in routes]
    evaluated["accepted"] = [route.accepted for route in routes]
    evaluated["verifier_called"] = [route.verifier_called for route in routes]
    metrics = _policy_metrics(evaluated, total_questions)
    metrics["feasible"] = bool(
        metrics["accepted_n"]
        and metrics["selective_risk"] <= policy.risk_budget + 1e-12
    )
    return evaluated, metrics


def _threshold_grid(values: pd.Series) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, 11)
    grid = np.unique(np.quantile(values.astype(float), quantiles)).tolist()
    return [float(value) for value in grid]


def search_cascade_policy(
    candidates: pd.DataFrame,
    total_questions: int,
    risk_budget: float,
) -> tuple[CascadePolicy, dict[str, float | int | bool]]:
    """Select the maximum-coverage feasible cascade on calibration data."""

    cheap_grid = _threshold_grid(candidates["cheap_score"])
    verified_grid = _threshold_grid(candidates["verified_score"])
    verified_grid.append(float(candidates["verified_score"].max()) + 1e-9)
    feasible: list[tuple[CascadePolicy, dict[str, float | int | bool]]] = []
    all_trials: list[tuple[CascadePolicy, dict[str, float | int | bool]]] = []
    for cheap_low in cheap_grid:
        for cheap_high in cheap_grid:
            if cheap_high < cheap_low:
                continue
            for verified_threshold in verified_grid:
                policy = CascadePolicy(
                    risk_budget=risk_budget,
                    cheap_low=cheap_low,
                    cheap_high=cheap_high,
                    verified_threshold=verified_threshold,
                )
                _, metrics = evaluate_frozen_policy(
                    candidates, total_questions, policy
                )
                all_trials.append((policy, metrics))
                if metrics["feasible"]:
                    feasible.append((policy, metrics))
    if feasible:
        policy, metrics = min(
            feasible,
            key=lambda item: (
                -float(item[1]["coverage"]),
                int(item[1]["verifier_calls"]),
                float(item[1]["selective_risk"]),
                item[0].cheap_low,
                item[0].cheap_high,
                item[0].verified_threshold,
            ),
        )
    else:
        policy, metrics = min(
            all_trials,
            key=lambda item: (
                float(item[1]["selective_risk"])
                if np.isfinite(item[1]["selective_risk"])
                else np.inf,
                -float(item[1]["coverage"]),
                int(item[1]["verifier_calls"]),
                item[0].cheap_low,
                item[0].cheap_high,
                item[0].verified_threshold,
            ),
        )
    return policy, dict(metrics)


def _evaluate_rank_threshold(
    candidates: pd.DataFrame,
    total_questions: int,
    score_column: str,
    threshold: float,
    verifier_all: bool,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    evaluated = candidates.copy()
    evaluated["accepted"] = evaluated[score_column] >= threshold
    evaluated["verifier_called"] = verifier_all
    evaluated["decision"] = np.where(evaluated["accepted"], "answer", "abstain")
    return evaluated, _policy_metrics(evaluated, total_questions)


def _search_rank_policy(
    candidates: pd.DataFrame,
    total_questions: int,
    score_column: str,
    risk_budget: float,
    verifier_all: bool,
) -> tuple[float, dict[str, float | int | bool]]:
    thresholds = _threshold_grid(candidates[score_column])
    thresholds.append(float(candidates[score_column].max()) + 1e-9)
    trials = []
    for threshold in thresholds:
        _, metrics = _evaluate_rank_threshold(
            candidates, total_questions, score_column, threshold, verifier_all
        )
        metrics["feasible"] = bool(
            metrics["accepted_n"]
            and metrics["selective_risk"] <= risk_budget + 1e-12
        )
        trials.append((threshold, metrics))
    feasible = [trial for trial in trials if trial[1]["feasible"]]
    if feasible:
        return min(
            feasible,
            key=lambda item: (
                -float(item[1]["coverage"]),
                float(item[1]["selective_risk"]),
                item[0],
            ),
        )
    return min(
        trials,
        key=lambda item: (
            float(item[1]["selective_risk"])
            if np.isfinite(item[1]["selective_risk"])
            else np.inf,
            -float(item[1]["coverage"]),
            item[0],
        ),
    )


def bootstrap_policy_comparison(
    evaluated_policies: dict[str, pd.DataFrame],
    n_bootstrap: int = 2000,
    seed: int = 20260918,
) -> pd.DataFrame:
    """Return percentile intervals for frozen held-out policy metrics."""

    names = list(evaluated_policies)
    if not names:
        raise ValueError("At least one evaluated policy is required")
    row_count = len(evaluated_policies[names[0]])
    if any(len(rows) != row_count for rows in evaluated_policies.values()):
        raise ValueError("Evaluated policy tables must have equal row counts")
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, row_count, size=row_count)
        for name, rows in evaluated_policies.items():
            sample = rows.iloc[sample_indices]
            metrics = _policy_metrics(sample, total_questions=row_count)
            records.append({"policy": name, **metrics})
    draws = pd.DataFrame(records)
    summaries = []
    for name, group in draws.groupby("policy", sort=False):
        for metric in (
            "coverage",
            "selective_risk",
            "verifier_call_rate",
            "estimated_verifier_input_tokens",
        ):
            values = group[metric].dropna().astype(float)
            summaries.append(
                {
                    "policy": name,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(values.quantile(0.025)),
                    "ci_high": float(values.quantile(0.975)),
                    "bootstrap_draws": n_bootstrap,
                    "seed": seed,
                }
            )
    return pd.DataFrame(summaries)


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
    rows["estimated_verifier_tokens"] = [
        estimate_verifier_token_budget(question, context, answer)
        for question, context, answer in zip(
            rows["question"], rows["retrieved_context"], rows["generated_answer"]
        )
    ]
    return rows


def fit_cheap_and_verified_models(
    train_answers: pd.DataFrame,
) -> tuple[object, RidgeCV]:
    """Fit cheap quality and verifier-assisted quality regressors."""

    cheap_model = fit_cheap_quality_model(
        extract_deployment_features(train_answers), train_answers["answer_f1"]
    )
    cheap_train = predict_quality(
        cheap_model, extract_deployment_features(train_answers)
    )
    verifier_train = pd.to_numeric(
        train_answers["pre_gate_faith_score"], errors="raise"
    ).to_numpy() / 10.0
    verified_model = RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0, 100.0]), cv=5)
    verified_model.fit(
        np.column_stack([cheap_train, verifier_train]),
        train_answers["answer_f1"].to_numpy(dtype=float),
    )
    return cheap_model, verified_model


def _score_candidates(
    rows: pd.DataFrame,
    cheap_model: object,
    verified_model: RidgeCV,
) -> pd.DataFrame:
    candidates = rows[~rows["generated_answer"].map(is_refusal)].copy()
    candidates["cheap_score"] = predict_quality(
        cheap_model, extract_deployment_features(candidates)
    )
    verifier = pd.to_numeric(
        candidates["pre_gate_faith_score"], errors="raise"
    ).to_numpy() / 10.0
    candidates["verified_score"] = np.clip(
        verified_model.predict(
            np.column_stack([candidates["cheap_score"].to_numpy(), verifier])
        ),
        0.0,
        1.0,
    )
    return candidates


def _expand_to_all_questions(
    all_rows: pd.DataFrame,
    candidate_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Add generator-level abstentions so coverage uses all questions."""

    columns = [
        "question_id",
        "accepted",
        "verifier_called",
        "decision",
    ]
    expanded = all_rows[
        ["question_id", "answer_f1", "estimated_verifier_tokens"]
    ].merge(candidate_evaluation[columns], on="question_id", how="left")
    expanded["accepted"] = expanded["accepted"].eq(True)
    expanded["verifier_called"] = expanded["verifier_called"].eq(True)
    expanded["decision"] = expanded["decision"].fillna("generator_abstain")
    return expanded


def _write_report(
    summary: pd.DataFrame,
    policies: pd.DataFrame,
    output_path: str,
) -> None:
    lines = [
        "# QASPER Cost-Aware Verification Cascade",
        "",
        "Thresholds were selected only on the deterministic calibration split. "
        "The disjoint 150-question held-out set was evaluated once with frozen "
        "policies. Answer quality is normalized QASPER Answer F1.",
        "",
        "Verifier input tokens are offline estimates based on total input "
        "characters divided by four. They are not measured API usage, latency, "
        "or monetary cost.",
        "",
        "## Held-out results",
        "",
        "| Risk budget | Policy | Feasible | Coverage | Answer F1 | Risk | "
        "Verifier call rate | Estimated tokens | Savings vs always verify |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.risk_budget:.2f} | {row.policy} | {row.feasible} | "
            f"{row.coverage:.4f} | {row.mean_answer_f1:.4f} | "
            f"{row.selective_risk:.4f} | {row.verifier_call_rate:.4f} | "
            f"{row.estimated_verifier_input_tokens:.0f} | "
            f"{row.token_savings_vs_always_verify:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Policy-selection boundary",
            "",
            "The held-out labels were not used to choose thresholds. A held-out "
            "row can therefore violate its nominal calibration risk budget; "
            "the `feasible` column reports the observed held-out outcome rather "
            "than retroactively changing the policy.",
            "",
            "## Frozen thresholds",
            "",
            policies.to_markdown(index=False),
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as report_file:
        report_file.write("\n".join(lines))


def main() -> None:
    development = _load_rows(DEVELOPMENT_PATH, 300, filter_strategy=True)
    heldout = _load_rows(HELDOUT_PATH, 150, filter_strategy=False)
    assert_no_split_overlap(development, heldout)
    train, calibration = deterministic_train_calibration_split(development)
    train_answers = train[~train["generated_answer"].map(is_refusal)].copy()
    cheap_model, verified_model = fit_cheap_and_verified_models(train_answers)
    calibration_candidates = _score_candidates(
        calibration, cheap_model, verified_model
    )
    heldout_candidates = _score_candidates(heldout, cheap_model, verified_model)

    summary_records = []
    policy_records = []
    decision_tables = []
    bootstrap_inputs: dict[str, pd.DataFrame] = {}
    for risk_budget in RISK_BUDGETS:
        cascade_policy, calibration_cascade_metrics = search_cascade_policy(
            calibration_candidates, len(calibration), risk_budget
        )
        cheap_threshold, calibration_cheap_metrics = _search_rank_policy(
            calibration_candidates,
            len(calibration),
            "cheap_score",
            risk_budget,
            verifier_all=False,
        )
        verified_threshold, calibration_verify_metrics = _search_rank_policy(
            calibration_candidates,
            len(calibration),
            "verified_score",
            risk_budget,
            verifier_all=True,
        )

        candidate_evaluations: dict[str, pd.DataFrame] = {}
        no_gate = heldout_candidates.copy()
        no_gate["accepted"] = True
        no_gate["verifier_called"] = False
        no_gate["decision"] = "answer"
        candidate_evaluations["No Gate"] = no_gate
        cheap_evaluation, _ = _evaluate_rank_threshold(
            heldout_candidates,
            len(heldout),
            "cheap_score",
            cheap_threshold,
            verifier_all=False,
        )
        candidate_evaluations["Cheap Only"] = cheap_evaluation
        verify_evaluation, _ = _evaluate_rank_threshold(
            heldout_candidates,
            len(heldout),
            "verified_score",
            verified_threshold,
            verifier_all=True,
        )
        candidate_evaluations["Always Verify"] = verify_evaluation
        cascade_evaluation, _ = evaluate_frozen_policy(
            heldout_candidates, len(heldout), cascade_policy
        )
        candidate_evaluations["Cost-Aware Cascade"] = cascade_evaluation

        expanded = {
            name: _expand_to_all_questions(heldout, evaluation)
            for name, evaluation in candidate_evaluations.items()
        }
        always_tokens = _policy_metrics(
            expanded["Always Verify"], len(heldout)
        )["estimated_verifier_input_tokens"]
        for name, evaluation in expanded.items():
            metrics = _policy_metrics(evaluation, len(heldout))
            metrics["feasible"] = bool(
                metrics["accepted_n"]
                and metrics["selective_risk"] <= risk_budget + 1e-12
            )
            tokens = float(metrics["estimated_verifier_input_tokens"])
            savings = 0.0 if always_tokens == 0 else 1.0 - tokens / always_tokens
            summary_records.append(
                {
                    "risk_budget": risk_budget,
                    "policy": name,
                    **metrics,
                    "token_savings_vs_always_verify": savings,
                }
            )
            labeled = evaluation.copy()
            labeled.insert(0, "risk_budget", risk_budget)
            labeled.insert(1, "policy", name)
            decision_tables.append(labeled)
            bootstrap_inputs[f"{risk_budget:.2f} {name}"] = evaluation

        policy_records.extend(
            [
                {
                    "risk_budget": risk_budget,
                    "policy": "Cheap Only",
                    "cheap_low": cheap_threshold,
                    "cheap_high": np.nan,
                    "verified_threshold": np.nan,
                    "calibration_feasible": calibration_cheap_metrics["feasible"],
                },
                {
                    "risk_budget": risk_budget,
                    "policy": "Always Verify",
                    "cheap_low": np.nan,
                    "cheap_high": np.nan,
                    "verified_threshold": verified_threshold,
                    "calibration_feasible": calibration_verify_metrics["feasible"],
                },
                {
                    "risk_budget": risk_budget,
                    "policy": "Cost-Aware Cascade",
                    "cheap_low": cascade_policy.cheap_low,
                    "cheap_high": cascade_policy.cheap_high,
                    "verified_threshold": cascade_policy.verified_threshold,
                    "calibration_feasible": calibration_cascade_metrics["feasible"],
                },
            ]
        )

    summary = pd.DataFrame(summary_records)
    policies = pd.DataFrame(policy_records)
    decisions = pd.concat(decision_tables, ignore_index=True)
    bootstrap = bootstrap_policy_comparison(
        bootstrap_inputs, n_bootstrap=2000, seed=20260918
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "qasper_cost_aware_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    policies.to_csv(
        os.path.join(OUTPUT_DIR, "qasper_frozen_policies.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    decisions.to_csv(
        os.path.join(OUTPUT_DIR, "qasper_heldout_policy_decisions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        os.path.join(OUTPUT_DIR, "qasper_cost_aware_bootstrap.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(
        summary,
        policies,
        os.path.join(OUTPUT_DIR, "qasper_cost_aware_report.md"),
    )
    print(summary.round(4).to_string(index=False))
    print(f"\nTrain questions: {len(train)}; calibration questions: {len(calibration)}")
    print(f"Train answers: {len(train_answers)}; held-out answers: {len(heldout_candidates)}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
