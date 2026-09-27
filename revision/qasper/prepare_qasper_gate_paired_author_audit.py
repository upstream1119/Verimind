"""Prepare the remaining blind audit for the QASPER gate-paired pool."""

import argparse
import hashlib
import os
from typing import Set

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
DEFAULT_EXISTING_KEY = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_10_qasper_no_gating_author_audit",
    "qasper_no_gating_author_audit_50_key.csv",
)
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
FULL_STRATEGY = "VeriMind_Full"
NO_GATING_STRATEGY = "VeriMind_No_Gating"
SEED = "qasper-gate-paired-audit-v1"
EXPECTED_POOL_SIZE = 111
EXPECTED_EXISTING_SIZE = 38
EXPECTED_REMAINING_SIZE = 73


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        f"{SEED}|{value}".encode("utf-8")
    ).hexdigest()


def validate_source(source: pd.DataFrame) -> None:
    required = {
        "item_id",
        "question_id",
        "paper_id",
        "strategy",
        "accepted",
        "decision",
        "trust_score",
        "question",
        "generated_answer",
        "retrieved_context",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    relevant = source[
        source["strategy"].isin([FULL_STRATEGY, NO_GATING_STRATEGY])
    ]
    duplicates = int(
        relevant.duplicated(subset=["item_id", "strategy"]).sum()
    )
    if duplicates:
        raise ValueError(
            f"Results contain {duplicates} duplicate item-strategy rows"
        )


def build_gate_paired_pool(source: pd.DataFrame) -> pd.DataFrame:
    validate_source(source)
    columns = [
        "item_id",
        "question_id",
        "paper_id",
        "question",
        "generated_answer",
        "retrieved_context",
        "accepted",
        "decision",
        "trust_score",
    ]
    full = source[source["strategy"] == FULL_STRATEGY][columns].copy()
    no_gating = source[
        source["strategy"] == NO_GATING_STRATEGY
    ][columns].copy()
    paired = no_gating.merge(
        full,
        on="item_id",
        how="inner",
        suffixes=("_no_gating", "_full"),
        validate="one_to_one",
    )
    same_metadata = (
        paired["question_id_no_gating"].fillna("")
        == paired["question_id_full"].fillna("")
    ) & (
        paired["paper_id_no_gating"].fillna("")
        == paired["paper_id_full"].fillna("")
    )
    if not bool(same_metadata.all()):
        raise ValueError("Full and No-Gating metadata differ for paired items")

    same_input = (
        paired["generated_answer_no_gating"].fillna("")
        == paired["generated_answer_full"].fillna("")
    ) & (
        paired["retrieved_context_no_gating"].fillna("")
        == paired["retrieved_context_full"].fillna("")
    )
    pool = paired[
        (paired["accepted_no_gating"] == 1) & same_input
    ].copy()
    pool = pool.rename(
        columns={
            "question_id_no_gating": "question_id",
            "paper_id_no_gating": "paper_id",
            "question_no_gating": "question",
            "generated_answer_no_gating": "generated_answer",
            "retrieved_context_no_gating": "retrieved_context",
            "accepted_no_gating": "no_gating_accepted",
            "decision_no_gating": "no_gating_decision",
            "trust_score_no_gating": "no_gating_trust_score",
            "accepted_full": "full_accepted",
            "decision_full": "full_decision",
            "trust_score_full": "full_trust_score",
        }
    )
    keep = [
        "item_id",
        "question_id",
        "paper_id",
        "question",
        "generated_answer",
        "retrieved_context",
        "no_gating_accepted",
        "no_gating_decision",
        "no_gating_trust_score",
        "full_accepted",
        "full_decision",
        "full_trust_score",
    ]
    pool = pool[keep].sort_values("item_id").reset_index(drop=True)
    if len(pool) != EXPECTED_POOL_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_POOL_SIZE} gate-paired rows, "
            f"found {len(pool)}"
        )
    return pool


def load_existing_reviewed_ids(
    key_path: str,
    completed_path: str,
) -> pd.DataFrame:
    key = pd.read_csv(key_path, keep_default_na=False)
    completed = pd.read_csv(completed_path, keep_default_na=False)
    required_key = {"audit_id", "item_id"}
    required_completed = {"audit_id", "author_label"}
    if not required_key.issubset(key.columns):
        raise ValueError("Existing key is missing audit_id or item_id")
    if not required_completed.issubset(completed.columns):
        raise ValueError(
            "Existing completed audit is missing audit_id or author_label"
        )
    if key["audit_id"].duplicated().any():
        raise ValueError("Existing key contains duplicate audit IDs")
    if completed["audit_id"].duplicated().any():
        raise ValueError("Existing audit contains duplicate audit IDs")
    if set(key["audit_id"]) != set(completed["audit_id"]):
        raise ValueError("Existing completed audit does not match its key")
    blank = completed["author_label"].astype(str).str.strip().eq("")
    if bool(blank.any()):
        raise ValueError("Existing audit still contains blank labels")
    return key[["audit_id", "item_id"]].copy()


def prepare_audit(
    pool: pd.DataFrame,
    existing_key: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_ids: Set[int] = set(existing_key["item_id"].astype(int))
    pool_ids: Set[int] = set(pool["item_id"].astype(int))
    reviewed_ids = existing_ids & pool_ids
    if len(reviewed_ids) != EXPECTED_EXISTING_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_EXISTING_SIZE} already reviewed paired rows, "
            f"found {len(reviewed_ids)}"
        )

    remaining = pool[~pool["item_id"].isin(reviewed_ids)].copy()
    remaining["blind_rank"] = remaining["item_id"].map(stable_hash)
    remaining = remaining.sort_values("blind_rank").reset_index(drop=True)
    if len(remaining) != EXPECTED_REMAINING_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_REMAINING_SIZE} remaining rows, "
            f"found {len(remaining)}"
        )
    remaining["source_audit_id"] = [
        f"GP-{index:03d}"
        for index in range(1, len(remaining) + 1)
    ]

    existing_map = existing_key[
        existing_key["item_id"].astype(int).isin(reviewed_ids)
    ].copy()
    existing_map["item_id"] = existing_map["item_id"].astype(int)
    existing_map = existing_map.rename(
        columns={"audit_id": "source_audit_id"}
    )
    existing_map["audit_source"] = "existing_no_gating_audit"
    new_map = remaining[["source_audit_id", "item_id"]].copy()
    new_map["audit_source"] = "remaining_gate_paired_audit"
    audit_map = pd.concat([existing_map, new_map], ignore_index=True)

    key = pool.merge(
        audit_map,
        on="item_id",
        how="left",
        validate="one_to_one",
    ).sort_values("item_id").reset_index(drop=True)
    if key["source_audit_id"].isna().any():
        raise ValueError("Some paired rows do not have an audit mapping")
    key.insert(
        0,
        "pair_id",
        [f"PAIR-{index:03d}" for index in range(1, len(key) + 1)],
    )
    return remaining, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--existing-key", default=DEFAULT_EXISTING_KEY)
    parser.add_argument(
        "--existing-completed",
        default=DEFAULT_EXISTING_COMPLETED,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = pd.read_csv(args.results)
    pool = build_gate_paired_pool(source)
    existing_key = load_existing_reviewed_ids(
        args.existing_key,
        args.existing_completed,
    )
    remaining, key = prepare_audit(pool, existing_key)
    os.makedirs(args.output_dir, exist_ok=True)

    blind_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_remaining_73_blind.csv",
    )
    key_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_111_key.csv",
    )
    instructions_path = os.path.join(
        args.output_dir,
        "qasper_gate_paired_instructions.txt",
    )

    blind = remaining[
        [
            "source_audit_id",
            "question",
            "retrieved_context",
            "generated_answer",
        ]
    ].rename(columns={"source_audit_id": "audit_id"})
    blind["author_label"] = ""
    blind["author_notes"] = ""
    blind.to_csv(blind_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    instructions = [
        "QASPER gate-paired remaining 73-case author blind audit",
        "======================================================",
        "",
        "Open only qasper_gate_paired_remaining_73_blind.csv.",
        "Do not open qasper_gate_paired_111_key.csv before labeling ends.",
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
        "qasper_gate_paired_remaining_73_completed.csv",
        "",
        "Sampling boundary:",
        "All 111 paired cases have identical generated answers and retrieved",
        "contexts under Full and No-Gating, and No-Gating accepted them.",
        "The Full gate decision is hidden during author labeling.",
        "Thirty-eight cases were already labeled in the earlier random",
        "No-Gating audit; this sheet contains every remaining case.",
    ]
    with open(instructions_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(instructions) + "\n")

    print(f"Gate-paired pool: {len(key)}")
    print(
        "Already reviewed: "
        f"{int((key['audit_source'] == 'existing_no_gating_audit').sum())}"
    )
    print(f"Remaining blind audit: {len(blind)}")
    print(f"Blind audit sheet: {blind_path}")
    print(f"Private key: {key_path}")
    print(f"Instructions: {instructions_path}")


if __name__ == "__main__":
    main()
