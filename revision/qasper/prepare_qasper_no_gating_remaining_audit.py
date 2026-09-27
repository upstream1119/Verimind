"""Prepare the remaining blind audit of accepted QASPER No-Gating answers."""

import argparse
import hashlib
import os
from typing import Iterable, Set

import pandas as pd


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
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_13_qasper_no_gating_complete_audit",
)
STRATEGY = "VeriMind_No_Gating"
SEED = "qasper-no-gating-remaining-audit-v1"
ALLOWED_LABELS = {"supported", "unsupported", "uncertain"}
EXPECTED_ACCEPTED_SIZE = 149
EXPECTED_REVIEWED_SIZE = 123
EXPECTED_REMAINING_SIZE = 26


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        f"{SEED}|{value}".encode("utf-8")
    ).hexdigest()


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


def load_completed_mapping(
    completed_path: str,
    key_path: str,
    key_audit_column: str,
    source_name: str,
) -> pd.DataFrame:
    completed = pd.read_csv(completed_path, keep_default_na=False)
    key = pd.read_csv(key_path, keep_default_na=False)
    required_completed = {"audit_id", "author_label"}
    required_key = {key_audit_column, "item_id"}
    missing_completed = required_completed - set(completed.columns)
    missing_key = required_key - set(key.columns)
    if missing_completed:
        raise ValueError(
            f"{source_name} completed file is missing: "
            f"{sorted(missing_completed)}"
        )
    if missing_key:
        raise ValueError(
            f"{source_name} key is missing: {sorted(missing_key)}"
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

    mapping = key[[key_audit_column, "item_id"]].drop_duplicates()
    mapping = mapping.rename(columns={key_audit_column: "audit_id"})
    matched = completed[
        ["audit_id", "author_label"]
    ].merge(
        mapping,
        on="audit_id",
        how="left",
        validate="one_to_one",
    )
    if matched["item_id"].isna().any():
        missing_ids = matched.loc[
            matched["item_id"].isna(),
            "audit_id",
        ].tolist()
        raise ValueError(
            f"{source_name} has audit IDs absent from its key: {missing_ids}"
        )
    matched["item_id"] = matched["item_id"].astype(int)
    matched["audit_source"] = source_name
    return matched[
        ["item_id", "author_label", "audit_id", "audit_source"]
    ]


def combine_reviewed(mappings: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(list(mappings), ignore_index=True)
    label_counts = combined.groupby("item_id")["author_label"].nunique()
    conflicts = label_counts[label_counts > 1]
    if len(conflicts):
        raise ValueError(
            "Previously reviewed files disagree on labels for item IDs: "
            f"{conflicts.index.astype(int).tolist()}"
        )
    reviewed = combined.sort_values(
        ["item_id", "audit_source", "audit_id"]
    ).drop_duplicates("item_id")
    if len(reviewed) != EXPECTED_REVIEWED_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_REVIEWED_SIZE} unique reviewed items, "
            f"found {len(reviewed)}"
        )
    return reviewed.reset_index(drop=True)


def select_remaining(
    results: pd.DataFrame,
    reviewed_ids: Set[int],
) -> pd.DataFrame:
    required = {
        "item_id",
        "question_id",
        "paper_id",
        "strategy",
        "accepted",
        "is_unanswerable",
        "question",
        "generated_answer",
        "retrieved_context",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    duplicates = int(
        results.duplicated(subset=["item_id", "strategy"]).sum()
    )
    if duplicates:
        raise ValueError(
            f"Results contain {duplicates} duplicate item-strategy rows"
        )

    accepted = results[
        (results["strategy"] == STRATEGY)
        & (results["accepted"] == 1)
    ].copy()
    if len(accepted) != EXPECTED_ACCEPTED_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_ACCEPTED_SIZE} accepted No-Gating rows, "
            f"found {len(accepted)}"
        )
    accepted_ids = set(accepted["item_id"].astype(int))
    outside = reviewed_ids - accepted_ids
    if outside:
        raise ValueError(
            "Reviewed item IDs are not accepted No-Gating answers: "
            f"{sorted(outside)}"
        )

    remaining = accepted[
        ~accepted["item_id"].astype(int).isin(reviewed_ids)
    ].copy()
    if len(remaining) != EXPECTED_REMAINING_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_REMAINING_SIZE} remaining rows, "
            f"found {len(remaining)}"
        )
    remaining["blind_rank"] = remaining["item_id"].map(stable_hash)
    remaining = remaining.sort_values("blind_rank").reset_index(drop=True)
    remaining["audit_id"] = [
        f"NGR-{index:03d}"
        for index in range(1, len(remaining) + 1)
    ]
    return remaining


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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    old_mapping = load_completed_mapping(
        args.old_completed,
        args.old_key,
        "audit_id",
        "existing_no_gating_audit",
    )
    paired_mapping = load_completed_mapping(
        args.paired_completed,
        args.paired_key,
        "source_audit_id",
        "gate_paired_remaining_audit",
    )
    reviewed = combine_reviewed([old_mapping, paired_mapping])
    results = pd.read_csv(args.results)
    remaining = select_remaining(
        results,
        set(reviewed["item_id"].astype(int)),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    blind_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_remaining_26_blind.csv",
    )
    key_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_remaining_26_key.csv",
    )
    instructions_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_remaining_26_instructions.txt",
    )

    blind = remaining[
        [
            "audit_id",
            "question",
            "retrieved_context",
            "generated_answer",
        ]
    ].copy()
    blind["author_label"] = ""
    blind["author_notes"] = ""
    blind.to_csv(blind_path, index=False, encoding="utf-8-sig")

    key = remaining[
        [
            "audit_id",
            "item_id",
            "question_id",
            "paper_id",
            "strategy",
            "is_unanswerable",
        ]
    ].copy()
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    instructions = [
        "QASPER No-Gating remaining 26-case author blind audit",
        "=====================================================",
        "",
        "Open only qasper_no_gating_remaining_26_blind.csv.",
        "Do not open qasper_no_gating_remaining_26_key.csv until all labels",
        "are complete.",
        "",
        "Allowed author_label values:",
        "- supported: every material claim is supported by the context",
        "- unsupported: at least one material claim lacks support, conflicts",
        "  with the context, or is attributed to the wrong method or study",
        "- uncertain: the question or evidence is genuinely ambiguous",
        "",
        "Use only the retrieved context. Do not use outside knowledge.",
        "Do not penalize an answer merely for being incomplete.",
        "Check numerical metrics, method attribution, comparison relations,",
        "and whether Related Work is incorrectly described as the paper's",
        "own method.",
        "",
        "Fill author_label and author_notes without renaming columns or",
        "reordering rows. Save the completed file as:",
        "qasper_no_gating_remaining_26_completed.csv",
        "",
        "Sampling boundary:",
        "These are all remaining accepted No-Gating answers after removing",
        "the 123 unique cases completed in the earlier author audits.",
        "No gate score, automated judge label, or item identifier is exposed",
        "in the blind sheet.",
    ]
    with open(instructions_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(instructions) + "\n")

    print(f"Accepted No-Gating answers: {EXPECTED_ACCEPTED_SIZE}")
    print(f"Unique previously reviewed answers: {len(reviewed)}")
    print(f"Remaining blind audit: {len(blind)}")
    print(f"Blind audit sheet: {blind_path}")
    print(f"Private key: {key_path}")
    print(f"Instructions: {instructions_path}")


if __name__ == "__main__":
    main()
