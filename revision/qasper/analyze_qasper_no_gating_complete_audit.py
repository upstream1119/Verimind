"""Analyze the complete author audit of accepted QASPER No-Gating answers."""

import argparse
import math
import os
from typing import Dict, List, Tuple

import pandas as pd

from revision.qasper.analyze_qasper_gate_paired_author_audit import (
    fisher_exact_two_sided,
    wilson_interval,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_RESULTS = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
DEFAULT_OLD_COMPLETED = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_10_qasper_no_gating_author_audit",
    "qasper_no_gating_author_audit_50_completed.csv",
)
DEFAULT_OLD_KEY = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_10_qasper_no_gating_author_audit",
    "qasper_no_gating_author_audit_50_key.csv",
)
DEFAULT_PAIRED_COMPLETED = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_12_qasper_gate_paired_author_audit",
    "qasper_gate_paired_remaining_73_completed.csv",
)
DEFAULT_PAIRED_KEY = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_12_qasper_gate_paired_author_audit",
    "qasper_gate_paired_111_key.csv",
)
DEFAULT_REMAINING_COMPLETED = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_13_qasper_no_gating_complete_audit",
    "qasper_no_gating_remaining_26_completed.csv",
)
DEFAULT_REMAINING_KEY = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_13_qasper_no_gating_complete_audit",
    "qasper_no_gating_remaining_26_key.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_13_qasper_no_gating_complete_audit",
)
STRATEGY = "VeriMind_No_Gating"
GATE_THRESHOLD = 5.0
EXPECTED_SIZE = 149
ALLOWED_LABELS = {"supported", "unsupported", "uncertain"}


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


def load_completed(
    completed_path: str,
    key_path: str,
    key_audit_column: str,
    source_name: str,
) -> pd.DataFrame:
    completed = pd.read_csv(completed_path, keep_default_na=False)
    key = pd.read_csv(key_path, keep_default_na=False)
    required_completed = {
        "audit_id",
        "question",
        "retrieved_context",
        "generated_answer",
        "author_label",
        "author_notes",
    }
    required_key = {key_audit_column, "item_id"}
    missing_completed = required_completed - set(completed.columns)
    missing_key = required_key - set(key.columns)
    if missing_completed:
        raise ValueError(
            f"{source_name} completed file is missing columns: "
            f"{sorted(missing_completed)}"
        )
    if missing_key:
        raise ValueError(
            f"{source_name} key is missing columns: {sorted(missing_key)}"
        )
    if completed["audit_id"].duplicated().any():
        raise ValueError(f"{source_name} contains duplicate audit IDs")

    completed["author_label"] = completed["author_label"].map(
        normalize_label
    )
    invalid = sorted(set(completed["author_label"]) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(
            f"{source_name} contains blank or invalid labels: {invalid}"
        )
    blank_notes = completed["author_notes"].astype(str).str.strip().eq("")
    if bool(blank_notes.any()):
        raise ValueError(f"{source_name} contains blank author notes")

    mapping = key[[key_audit_column, "item_id"]].drop_duplicates()
    mapping = mapping.rename(columns={key_audit_column: "audit_id"})
    scored = completed.merge(
        mapping,
        on="audit_id",
        how="left",
        validate="one_to_one",
    )
    if scored["item_id"].isna().any():
        missing_ids = scored.loc[
            scored["item_id"].isna(),
            "audit_id",
        ].tolist()
        raise ValueError(
            f"{source_name} audit IDs are absent from the key: {missing_ids}"
        )
    scored["item_id"] = scored["item_id"].astype(int)
    scored["audit_source"] = source_name
    return scored[
        [
            "item_id",
            "audit_id",
            "audit_source",
            "question",
            "retrieved_context",
            "generated_answer",
            "author_label",
            "author_notes",
        ]
    ]


def build_scored_dataset(
    results: pd.DataFrame,
    audit_frames: List[pd.DataFrame],
) -> pd.DataFrame:
    audited = pd.concat(audit_frames, ignore_index=True)
    conflicts = (
        audited.groupby("item_id")["author_label"].nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(
            "Audit files disagree on labels for item IDs: "
            f"{conflicts.index.astype(int).tolist()}"
        )
    if audited["item_id"].duplicated().any():
        duplicates = audited.loc[
            audited["item_id"].duplicated(keep=False),
            "item_id",
        ].astype(int).tolist()
        raise ValueError(f"Audit files overlap on item IDs: {duplicates}")
    if len(audited) != EXPECTED_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_SIZE} unique audited rows, "
            f"found {len(audited)}"
        )

    required_results = {
        "item_id",
        "question_id",
        "paper_id",
        "paper_title",
        "question",
        "is_unanswerable",
        "strategy",
        "average_similarity",
        "calibrated_retrieval",
        "pre_gate_faith_score",
        "trust_score",
        "generated_answer",
        "retrieved_context",
    }
    missing = required_results - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    no_gating = results[
        (results["strategy"] == STRATEGY)
        & (results["accepted"] == 1)
    ].copy()
    if len(no_gating) != EXPECTED_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_SIZE} accepted No-Gating rows, "
            f"found {len(no_gating)}"
        )
    if no_gating["item_id"].duplicated().any():
        raise ValueError("Accepted No-Gating results contain duplicate items")

    content = audited.merge(
        no_gating[
            [
                "item_id",
                "question",
                "generated_answer",
                "retrieved_context",
            ]
        ],
        on="item_id",
        suffixes=("_audit", "_result"),
        validate="one_to_one",
    )
    for column in ["question", "generated_answer", "retrieved_context"]:
        changed = (
            content[f"{column}_audit"]
            != content[f"{column}_result"]
        )
        if bool(changed.any()):
            item_ids = content.loc[changed, "item_id"].astype(int).tolist()
            raise ValueError(
                f"Audited {column} differs from results for items: {item_ids}"
            )

    scored = no_gating.merge(
        audited[
            [
                "item_id",
                "audit_id",
                "audit_source",
                "author_label",
                "author_notes",
            ]
        ],
        on="item_id",
        validate="one_to_one",
    )
    scored["gate_reject"] = scored["trust_score"] < GATE_THRESHOLD
    scored["gate_decision"] = scored["gate_reject"].map(
        {True: "REJECT", False: "ANSWER"}
    )
    return scored.sort_values("item_id").reset_index(drop=True)


def exact_mcnemar_two_sided(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    probability = sum(
        math.comb(total, value)
        for value in range(lower + 1)
    ) / (2 ** total)
    return min(1.0, 2.0 * probability)


def build_confusion(scored: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for gate_reject, gate_result in [
        (True, "rejected_by_gate"),
        (False, "accepted_by_gate"),
    ]:
        group = scored[scored["gate_reject"] == gate_reject]
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


def rate_row(
    metric: str,
    numerator: int,
    denominator: int,
    notes: str,
) -> Dict[str, object]:
    estimate = numerator / denominator if denominator else float("nan")
    ci_low, ci_high = wilson_interval(numerator, denominator)
    return {
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "statistic": "",
        "p_value": "",
        "notes": notes,
    }


def build_metrics(
    scored: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    supported = scored["author_label"] == "supported"
    unsupported = scored["author_label"] == "unsupported"
    uncertain = scored["author_label"] == "uncertain"
    rejected = scored["gate_reject"]

    tp = int((rejected & unsupported).sum())
    fp = int((rejected & supported).sum())
    fn = int((~rejected & unsupported).sum())
    tn = int((~rejected & supported).sum())
    accepted_certain = fn + tn
    rejected_certain = tp + fp
    certain_n = int((~uncertain).sum())
    rejected_uncertain = int((rejected & uncertain).sum())
    accepted_uncertain = int((~rejected & uncertain).sum())

    rows = [
        rate_row(
            "no_gating_unsupported_rate",
            int(unsupported.sum()),
            certain_n,
            "Unsupported among all certain accepted No-Gating answers.",
        ),
        rate_row(
            "gate_sensitivity_for_unsupported",
            tp,
            int(unsupported.sum()),
            "Unsupported answers rejected by the current gate.",
        ),
        rate_row(
            "gate_false_rejection_rate",
            fp,
            int(supported.sum()),
            "Supported answers rejected by the current gate.",
        ),
        rate_row(
            "gate_precision_for_unsupported",
            tp,
            rejected_certain,
            "Unsupported among all certain gate rejections.",
        ),
        rate_row(
            "passed_unsupported_rate",
            fn,
            accepted_certain,
            "Unsupported among certain answers passed by the gate.",
        ),
        rate_row(
            "unsupported_exposure_after_gate",
            fn,
            certain_n,
            "Passed unsupported answers per all certain No-Gating answers.",
        ),
        rate_row(
            "conservative_no_gating_risk",
            int((unsupported | uncertain).sum()),
            len(scored),
            "Uncertain labels are counted as unsupported.",
        ),
        rate_row(
            "conservative_passed_risk",
            fn + accepted_uncertain,
            int((~rejected).sum()),
            "Uncertain passed labels are counted as unsupported.",
        ),
    ]

    odds_ratio, fisher_p = fisher_exact_two_sided(tp, fp, fn, tn)
    rows.append(
        {
            "metric": "fisher_rejection_enrichment",
            "numerator": "",
            "denominator": "",
            "estimate": "",
            "ci_low": "",
            "ci_high": "",
            "statistic": odds_ratio,
            "p_value": fisher_p,
            "notes": (
                "Two-sided Fisher exact test of unsupported enrichment "
                "among gate rejections."
            ),
        }
    )
    mcnemar_p = exact_mcnemar_two_sided(tp, 0)
    rows.append(
        {
            "metric": "mcnemar_unsupported_exposure",
            "numerator": "",
            "denominator": "",
            "estimate": "",
            "ci_low": "",
            "ci_high": "",
            "statistic": tp,
            "p_value": mcnemar_p,
            "notes": (
                "Exact two-sided McNemar test; the gate prevented "
                f"{tp} unsupported exposures and introduced none."
            ),
        }
    )
    counts = {
        "supported": int(supported.sum()),
        "unsupported": int(unsupported.sum()),
        "uncertain": int(uncertain.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "rejected_uncertain": rejected_uncertain,
        "accepted_uncertain": accepted_uncertain,
    }
    return pd.DataFrame(rows), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--old-completed", default=DEFAULT_OLD_COMPLETED)
    parser.add_argument("--old-key", default=DEFAULT_OLD_KEY)
    parser.add_argument(
        "--paired-completed",
        default=DEFAULT_PAIRED_COMPLETED,
    )
    parser.add_argument("--paired-key", default=DEFAULT_PAIRED_KEY)
    parser.add_argument(
        "--remaining-completed",
        default=DEFAULT_REMAINING_COMPLETED,
    )
    parser.add_argument("--remaining-key", default=DEFAULT_REMAINING_KEY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    audit_frames = [
        load_completed(
            args.old_completed,
            args.old_key,
            "audit_id",
            "initial_random_50",
        ),
        load_completed(
            args.paired_completed,
            args.paired_key,
            "source_audit_id",
            "gate_paired_remaining_73",
        ),
        load_completed(
            args.remaining_completed,
            args.remaining_key,
            "audit_id",
            "no_gating_remaining_26",
        ),
    ]
    results = pd.read_csv(args.results)
    scored = build_scored_dataset(results, audit_frames)
    confusion = build_confusion(scored)
    metrics, counts = build_metrics(scored)
    non_supported = scored[
        scored["author_label"].isin(["unsupported", "uncertain"])
    ].copy()

    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_149_scored.csv",
    )
    metrics_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_149_metrics.csv",
    )
    confusion_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_149_confusion.csv",
    )
    cases_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_non_supported_cases.csv",
    )
    summary_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_149_summary.txt",
    )
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    confusion.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    non_supported[
        [
            "item_id",
            "question_id",
            "paper_id",
            "question",
            "generated_answer",
            "retrieved_context",
            "author_label",
            "author_notes",
            "pre_gate_faith_score",
            "trust_score",
            "gate_decision",
            "audit_id",
            "audit_source",
        ]
    ].to_csv(cases_path, index=False, encoding="utf-8-sig")

    indexed = metrics.set_index("metric")

    def describe(metric: str) -> str:
        row = indexed.loc[metric]
        return (
            f"{int(row['numerator'])}/{int(row['denominator'])} "
            f"({row['estimate']:.2%}; 95% Wilson CI "
            f"{row['ci_low']:.2%}-{row['ci_high']:.2%})"
        )

    fisher = indexed.loc["fisher_rejection_enrichment"]
    mcnemar = indexed.loc["mcnemar_unsupported_exposure"]
    lines = [
        "QASPER complete No-Gating author-audit summary",
        "===============================================",
        "",
        f"Finite audited pool: {len(scored)} accepted No-Gating answers.",
        (
            f"Author labels: supported={counts['supported']}, "
            f"unsupported={counts['unsupported']}, "
            f"uncertain={counts['uncertain']}."
        ),
        (
            f"Current gate threshold replay: trust_score < "
            f"{GATE_THRESHOLD:.1f}."
        ),
        "",
        (
            "No-Gating unsupported rate: "
            f"{describe('no_gating_unsupported_rate')}."
        ),
        (
            "Gate sensitivity for unsupported answers: "
            f"{describe('gate_sensitivity_for_unsupported')}."
        ),
        (
            "Gate false-rejection rate: "
            f"{describe('gate_false_rejection_rate')}."
        ),
        (
            "Gate precision for unsupported answers: "
            f"{describe('gate_precision_for_unsupported')}."
        ),
        (
            "Passed unsupported rate: "
            f"{describe('passed_unsupported_rate')}."
        ),
        (
            "Conservative No-Gating risk: "
            f"{describe('conservative_no_gating_risk')}."
        ),
        (
            "Conservative passed risk: "
            f"{describe('conservative_passed_risk')}."
        ),
        "",
        (
            "Binary confusion on certain labels: "
            f"TP={counts['tp']}, FP={counts['fp']}, "
            f"FN={counts['fn']}, TN={counts['tn']}."
        ),
        (
            "Fisher exact rejection-enrichment test: "
            f"odds ratio={float(fisher['statistic']):.4f}, "
            f"two-sided p={float(fisher['p_value']):.4f}."
        ),
        (
            "Exact McNemar unsupported-exposure test: "
            f"two-sided p={float(mcnemar['p_value']):.4f}."
        ),
        "",
        "Interpretation boundary:",
        (
            "The complete 149-case audit is a finite author-conducted blind "
            "audit of accepted No-Gating answers, not an independent expert "
            "annotation study."
        ),
        (
            "This audited pool is now a development and error-analysis set. "
            "Any Gate V2 calibrated on these labels requires evaluation on "
            "a separate untouched test set."
        ),
        (
            "The current gate does not show a statistically significant "
            "reduction in author-audited unsupported-answer exposure."
        ),
    ]
    with open(summary_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nScored audit: {scored_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Confusion table: {confusion_path}")
    print(f"Non-supported cases: {cases_path}")
    print(f"Narrative: {summary_path}")


if __name__ == "__main__":
    main()
