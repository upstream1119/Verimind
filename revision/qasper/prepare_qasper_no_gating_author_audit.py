"""Prepare a deterministic random audit of accepted No-Gating answers."""

import argparse
import hashlib
import os

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
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_10_qasper_no_gating_author_audit",
)
STRATEGY = "VeriMind_No_Gating"
SEED = "qasper-no-gating-author-audit-v1"
SAMPLE_PLAN = {0: 47, 1: 3}


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        f"{SEED}|{value}".encode("utf-8")
    ).hexdigest()


def choose_random_rows(source: pd.DataFrame) -> pd.DataFrame:
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
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    duplicate_count = int(
        source.duplicated(subset=["item_id", "strategy"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Input contains {duplicate_count} duplicate item-strategy rows"
        )

    accepted = source[
        (source["strategy"] == STRATEGY)
        & (source["accepted"] == 1)
    ].copy()
    selected = []
    for is_unanswerable, quota in SAMPLE_PLAN.items():
        candidates = accepted[
            accepted["is_unanswerable"] == is_unanswerable
        ].copy()
        candidates["sample_rank"] = candidates["item_id"].map(
            lambda item_id: stable_hash(
                f"{is_unanswerable}|{int(item_id)}"
            )
        )
        candidates = candidates.sort_values("sample_rank")
        if len(candidates) < quota:
            raise ValueError(
                f"Not enough rows for is_unanswerable={is_unanswerable}: "
                f"need {quota}, found {len(candidates)}"
            )
        sample = candidates.head(quota).copy()
        sample["sampling_stratum"] = (
            "unanswerable" if is_unanswerable else "answerable"
        )
        selected.append(sample)

    audit = pd.concat(selected, ignore_index=True)
    if len(audit) != 50:
        raise ValueError(f"Expected 50 sampled rows, found {len(audit)}")
    audit["blind_rank"] = audit["item_id"].map(
        lambda item_id: stable_hash(f"blind|{int(item_id)}")
    )
    audit = audit.sort_values("blind_rank").reset_index(drop=True)
    audit["audit_id"] = [
        f"NG-{index:03d}"
        for index in range(1, len(audit) + 1)
    ]
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    audit = choose_random_rows(source)
    os.makedirs(args.output_dir, exist_ok=True)
    blind_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_author_audit_50_blind.csv",
    )
    key_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_author_audit_50_key.csv",
    )
    instructions_path = os.path.join(
        args.output_dir,
        "qasper_no_gating_author_audit_instructions.txt",
    )

    blind = audit[
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

    key = audit[
        [
            "audit_id",
            "item_id",
            "question_id",
            "paper_id",
            "strategy",
            "is_unanswerable",
            "sampling_stratum",
        ]
    ].copy()
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    instructions = [
        "QASPER No-Gating random 50-case author blind audit",
        "==================================================",
        "",
        "Open only qasper_no_gating_author_audit_50_blind.csv.",
        "Do not open the key file until all 50 labels are complete.",
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
        "reordering rows. Save as UTF-8 CSV.",
        "",
        "Sampling boundary:",
        "The 50 cases are randomly selected from accepted No-Gating answers.",
        "No automated judge label was used for sampling.",
    ]
    with open(instructions_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(instructions) + "\n")

    print(f"Blind audit sheet: {blind_path}")
    print(f"Private key: {key_path}")
    print(f"Instructions: {instructions_path}")
    print(
        key.groupby("sampling_stratum")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
