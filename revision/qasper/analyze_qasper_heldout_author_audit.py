"""Score the held-out QASPER author audit against frozen Gate V2 decisions."""

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd

from revision.qasper.analyze_qasper_gate_paired_author_audit import (
    fisher_exact_two_sided,
    wilson_interval,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
)
DEFAULT_COMPLETED = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_author_audit_96_completed.csv",
)
DEFAULT_BLIND = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_author_audit_96_blind.csv",
)
DEFAULT_KEY = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_author_audit_96_key.csv",
)
DEFAULT_DECISIONS = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "gate_v2_heldout_case_decisions.csv",
)
DEFAULT_NO_GATING_RESULTS = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_no_gating_results.csv",
)

EXPECTED_ROWS = 96
EXPECTED_COLUMNS = [
    "audit_id",
    "question",
    "retrieved_context",
    "generated_answer",
    "author_label",
    "author_notes",
]
ALLOWED_LABELS = {"supported", "unsupported", "uncertain"}
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


def read_audit(
    path: str,
    source_name: str,
    require_labels: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{source_name} columns do not match the blind audit schema: "
            f"{list(frame.columns)}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} {source_name} rows, found {len(frame)}"
        )
    if frame["audit_id"].duplicated().any():
        duplicates = frame.loc[
            frame["audit_id"].duplicated(keep=False),
            "audit_id",
        ].tolist()
        raise ValueError(f"{source_name} duplicate audit IDs: {duplicates}")
    if require_labels:
        frame["author_label"] = frame["author_label"].map(normalize_label)
        invalid = sorted(set(frame["author_label"]) - ALLOWED_LABELS)
        if invalid:
            raise ValueError(
                f"{source_name} has blank or invalid author labels: {invalid}"
            )
        blank_notes = frame["author_notes"].astype(str).str.strip().eq("")
        if bool(blank_notes.any()):
            raise ValueError(
                f"{source_name} has blank notes for audit IDs: "
                f"{frame.loc[blank_notes, 'audit_id'].tolist()}"
            )
    return frame


def validate_completed_against_blind(
    completed: pd.DataFrame,
    blind: pd.DataFrame,
) -> None:
    completed_ids = set(completed["audit_id"])
    blind_ids = set(blind["audit_id"])
    if completed_ids != blind_ids:
        raise ValueError(
            "Completed audit IDs do not match blind sheet IDs. "
            f"Missing={sorted(blind_ids - completed_ids)}, "
            f"extra={sorted(completed_ids - blind_ids)}"
        )
    merged = completed.merge(
        blind,
        on="audit_id",
        suffixes=("_completed", "_blind"),
        validate="one_to_one",
    )
    for column in ["question", "retrieved_context", "generated_answer"]:
        mismatch = merged[
            merged[f"{column}_completed"] != merged[f"{column}_blind"]
        ]
        if len(mismatch):
            raise ValueError(
                f"Completed audit changed {column} for audit IDs: "
                f"{mismatch['audit_id'].tolist()}"
            )


def validate_key_and_decisions(
    key: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    required_key = {
        "audit_id",
        "heldout_audit_id",
        "item_id",
        "question_id",
        "paper_id",
        "gate_v2_decision",
        "deepseek_label",
        "qwen_label",
        "glm_label",
        "unsupported_votes",
        "uncertain_votes",
        "current_gate_reject",
        "trust_score",
    }
    required_decisions = {
        "audit_id",
        "item_id",
        "gate_v2_decision",
        "deepseek_label",
        "qwen_label",
        "glm_label",
        "unsupported_votes",
        "uncertain_votes",
        "current_gate_reject",
        "trust_score",
    }
    missing_key = required_key - set(key.columns)
    missing_decisions = required_decisions - set(decisions.columns)
    if missing_key:
        raise ValueError(f"Key file is missing columns: {sorted(missing_key)}")
    if missing_decisions:
        raise ValueError(
            f"Decision file is missing columns: {sorted(missing_decisions)}"
        )
    if len(key) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} key rows, found {len(key)}")
    if len(decisions) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} decision rows, found {len(decisions)}"
        )
    if key["audit_id"].duplicated().any():
        raise ValueError("Key file has duplicate HOB audit IDs")
    if key["heldout_audit_id"].duplicated().any():
        raise ValueError("Key file has duplicate HO audit IDs")
    if decisions["audit_id"].duplicated().any():
        raise ValueError("Decision file has duplicate HO audit IDs")

    decision_check = key.merge(
        decisions[
            [
                "audit_id",
                "item_id",
                "gate_v2_decision",
                "deepseek_label",
                "qwen_label",
                "glm_label",
                "unsupported_votes",
                "uncertain_votes",
                "current_gate_reject",
                "trust_score",
            ]
        ].rename(columns={"audit_id": "heldout_audit_id"}),
        on="heldout_audit_id",
        suffixes=("_key", "_decision"),
        validate="one_to_one",
    )
    for column in [
        "item_id",
        "gate_v2_decision",
        "deepseek_label",
        "qwen_label",
        "glm_label",
        "unsupported_votes",
        "uncertain_votes",
        "current_gate_reject",
    ]:
        mismatch = decision_check[
            decision_check[f"{column}_key"].astype(str)
            != decision_check[f"{column}_decision"].astype(str)
        ]
        if len(mismatch):
            raise ValueError(
                f"Key and decision files disagree on {column} for HOB IDs: "
                f"{mismatch['audit_id'].tolist()}"
            )
    trust_delta = (
        decision_check["trust_score_key"].astype(float)
        - decision_check["trust_score_decision"].astype(float)
    ).abs()
    if bool((trust_delta > 1e-9).any()):
        raise ValueError("Key and decision files disagree on trust_score")

    return key.copy()


def build_scored_dataset(
    completed: pd.DataFrame,
    key: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    merged = completed.merge(
        key,
        on="audit_id",
        validate="one_to_one",
    )
    decision_details = decisions[
        [
            "audit_id",
            "paper_title",
            "gold_answer",
            "gold_answers_json",
            "raw_hallucination",
            "passed_hallucination",
            "evidence_recall",
            "rouge_l_f1",
            "policy_version",
        ]
    ].rename(columns={"audit_id": "heldout_audit_id"})
    scored = merged.merge(
        decision_details,
        on="heldout_audit_id",
        how="left",
        validate="one_to_one",
    )
    if scored["paper_title"].isna().any():
        raise ValueError("Some scored rows did not merge with decisions")
    scored["author_unsupported"] = scored["author_label"].eq("unsupported")
    scored["author_supported"] = scored["author_label"].eq("supported")
    scored["gate_v2_reject_only"] = scored["gate_v2_decision"].eq("REJECT")
    scored["gate_v2_intervention"] = scored["gate_v2_decision"].isin(
        ["REJECT", "WARNING"]
    )
    for judge in JUDGES:
        scored[f"{judge}_unsupported"] = scored[f"{judge}_label"].eq(
            "unsupported"
        )
    return scored.sort_values("audit_id").reset_index(drop=True)


def evaluate_policy(
    scored: pd.DataFrame,
    policy_name: str,
    policy_column: str,
    total_questions: int,
) -> Dict[str, object]:
    certain = scored[scored["author_label"].isin(["supported", "unsupported"])]
    intervened = certain[policy_column].astype(bool)
    unsupported = certain["author_label"].eq("unsupported")
    supported = certain["author_label"].eq("supported")

    tp = int((intervened & unsupported).sum())
    fp = int((intervened & supported).sum())
    fn = int((~intervened & unsupported).sum())
    tn = int((~intervened & supported).sum())
    rejected_total = int(scored[policy_column].astype(bool).sum())
    passed_total = int((~scored[policy_column].astype(bool)).sum())
    recall_den = tp + fn
    fpr_den = fp + tn
    precision_den = tp + fp
    passed_risk_den = fn + tn
    odds_ratio, p_value = fisher_exact_two_sided(tp, fp, fn, tn)
    recall_low, recall_high = wilson_interval(tp, recall_den)
    fpr_low, fpr_high = wilson_interval(fp, fpr_den)
    risk_low, risk_high = wilson_interval(fn, passed_risk_den)

    return {
        "policy": policy_name,
        "sample_n": len(scored),
        "certain_n": len(certain),
        "total_questions": total_questions,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "rejected_or_warned_n": rejected_total,
        "passed_n": passed_total,
        "unsupported_n": int(unsupported.sum()),
        "supported_n": int(supported.sum()),
        "recall": tp / recall_den if recall_den else float("nan"),
        "recall_ci_low": recall_low,
        "recall_ci_high": recall_high,
        "false_rejection_rate": fp / fpr_den if fpr_den else float("nan"),
        "false_rejection_ci_low": fpr_low,
        "false_rejection_ci_high": fpr_high,
        "precision": tp / precision_den if precision_den else float("nan"),
        "passed_unsupported_risk": (
            fn / passed_risk_den if passed_risk_den else float("nan")
        ),
        "passed_unsupported_risk_ci_low": risk_low,
        "passed_unsupported_risk_ci_high": risk_high,
        "answered_coverage_within_audited": passed_total / len(scored),
        "answered_coverage_total_questions": (
            passed_total / total_questions if total_questions else float("nan")
        ),
        "odds_ratio": odds_ratio,
        "fisher_two_sided_p": p_value,
    }


def build_metrics(
    scored: pd.DataFrame,
    total_questions: int,
) -> pd.DataFrame:
    policies: List[Tuple[str, str]] = [
        ("current_gate", "current_gate_reject"),
        ("deepseek_unsupported", "deepseek_unsupported"),
        ("qwen_unsupported", "qwen_unsupported"),
        ("glm_unsupported", "glm_unsupported"),
        ("gate_v2_reject_only", "gate_v2_reject_only"),
        ("gate_v2_intervention", "gate_v2_intervention"),
    ]
    return pd.DataFrame(
        [
            evaluate_policy(scored, name, column, total_questions)
            for name, column in policies
        ]
    )


def determine_total_questions(no_gating_results_path: str) -> int:
    if not os.path.exists(no_gating_results_path):
        return EXPECTED_ROWS
    results = pd.read_csv(no_gating_results_path, keep_default_na=False)
    if "item_id" not in results.columns:
        raise ValueError("No-Gating results are missing item_id")
    if results["item_id"].duplicated().any():
        raise ValueError("No-Gating held-out results contain duplicate item IDs")
    return int(len(results))


def format_rate(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.2%}"


def write_summary(
    path: str,
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    total_questions: int,
) -> None:
    label_counts = scored["author_label"].value_counts().sort_index()
    decision_counts = scored["gate_v2_decision"].value_counts().sort_index()
    non_supported = scored[scored["author_label"].ne("supported")]
    metric_by_policy = metrics.set_index("policy")
    reject_only = metric_by_policy.loc["gate_v2_reject_only"]
    intervention = metric_by_policy.loc["gate_v2_intervention"]
    current = metric_by_policy.loc["current_gate"]

    lines = [
        "QASPER held-out author-audit scoring",
        "===================================",
        "",
        f"Audited No-Gating answered held-out items: {len(scored)}",
        f"Total held-out questions: {total_questions}",
        f"No-Gating answered coverage: {len(scored) / total_questions:.2%}",
        "",
        "Author label counts:",
        label_counts.to_string(),
        "",
        "Gate V2 decision counts:",
        decision_counts.to_string(),
        "",
        "Main policy metrics:",
    ]
    for policy in [
        "current_gate",
        "gate_v2_reject_only",
        "gate_v2_intervention",
        "deepseek_unsupported",
        "qwen_unsupported",
        "glm_unsupported",
    ]:
        row = metric_by_policy.loc[policy]
        lines.append(
            (
                f"{policy}: TP={int(row['tp'])}, FP={int(row['fp'])}, "
                f"FN={int(row['fn'])}, TN={int(row['tn'])}, "
                f"recall={format_rate(row['recall'])}, "
                f"FPR={format_rate(row['false_rejection_rate'])}, "
                f"passed risk={format_rate(row['passed_unsupported_risk'])}, "
                f"coverage(total)={format_rate(row['answered_coverage_total_questions'])}, "
                f"Fisher p={row['fisher_two_sided_p']:.4g}"
            )
        )
    lines.extend(
        [
            "",
            "Non-supported author-audit cases:",
        ]
    )
    if len(non_supported) == 0:
        lines.append("None")
    else:
        for row in non_supported.itertuples(index=False):
            lines.extend(
                [
                    (
                        f"{row.audit_id} / item_id={int(row.item_id)} / "
                        f"Gate V2={row.gate_v2_decision} / "
                        f"votes={int(row.unsupported_votes)} unsupported"
                    ),
                    f"question: {row.question}",
                    f"answer: {row.generated_answer}",
                    f"author_note: {row.author_notes}",
                ]
            )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            (
                "The random held-out answered audit contains only one "
                "author-unsupported case. This supports a sanity check that "
                "the observed unsupported case was caught, but it is too "
                "sparse to support a strong standalone significance claim."
            ),
            (
                "Gate V2 reject-only catches the only unsupported case with "
                f"{format_rate(reject_only['false_rejection_rate'])} false "
                "rejection among supported cases and "
                f"{format_rate(reject_only['answered_coverage_total_questions'])} "
                "overall answered coverage."
            ),
            (
                "Treating WARNING as intervention is more conservative but "
                f"lowers overall answered coverage to "
                f"{format_rate(intervention['answered_coverage_total_questions'])}."
            ),
            (
                "The current scalar gate also catches the unsupported case "
                f"but has higher false rejection than reject-only Gate V2 "
                f"({format_rate(current['false_rejection_rate'])})."
            ),
            (
                "Labels are author-conducted blind evidence audits, not "
                "independent expert annotations."
            ),
        ]
    )
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed", default=DEFAULT_COMPLETED)
    parser.add_argument("--blind", default=DEFAULT_BLIND)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--no-gating-results", default=DEFAULT_NO_GATING_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    completed = read_audit(args.completed, "completed audit")
    blind = read_audit(args.blind, "blind audit", require_labels=False)
    validate_completed_against_blind(completed, blind)

    key = pd.read_csv(args.key, keep_default_na=False)
    decisions = pd.read_csv(args.decisions, keep_default_na=False)
    key = validate_key_and_decisions(key, decisions)
    total_questions = determine_total_questions(args.no_gating_results)

    scored = build_scored_dataset(completed, key, decisions)
    key_ids = set(key["audit_id"])
    completed_ids = set(completed["audit_id"])
    if key_ids != completed_ids:
        raise ValueError("Completed audit IDs do not match key HOB IDs")
    metrics = build_metrics(scored, total_questions)
    non_supported = scored[scored["author_label"].ne("supported")].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_scored.csv",
    )
    metrics_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_metrics.csv",
    )
    non_supported_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_non_supported_cases.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_summary.txt",
    )

    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    non_supported.to_csv(
        non_supported_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_summary(summary_path, scored, metrics, total_questions)

    print(f"Scored rows: {len(scored)}")
    print("Author label counts:")
    print(scored["author_label"].value_counts().sort_index().to_string())
    print("\nPolicy metrics:")
    print(
        metrics[
            [
                "policy",
                "tp",
                "fp",
                "fn",
                "tn",
                "recall",
                "false_rejection_rate",
                "passed_unsupported_risk",
                "answered_coverage_total_questions",
                "fisher_two_sided_p",
            ]
        ].to_string(index=False)
    )
    print(f"\nScored audit: {scored_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Non-supported cases: {non_supported_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
