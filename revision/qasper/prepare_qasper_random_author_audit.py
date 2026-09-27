"""Prepare a deterministic random author audit of accepted QASPER answers."""

import argparse
import hashlib
import os
from typing import Dict, List

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
    "1_9_qasper_random_author_audit",
)
SEED = "qasper-random-author-audit-v1"
SAMPLE_PLAN: Dict[str, Dict[int, int]] = {
    "Vanilla_RAG": {0: 47, 1: 3},
    "HyDE": {0: 46, 1: 4},
    "VeriMind_Full": {0: 47, 1: 3},
}
BLIND_COLUMNS = [
    "audit_id",
    "question",
    "retrieved_context",
    "generated_answer",
    "author_label",
    "author_notes",
]


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

    selected: List[pd.DataFrame] = []
    for strategy, strata in SAMPLE_PLAN.items():
        accepted = source[
            (source["strategy"] == strategy)
            & (source["accepted"] == 1)
        ].copy()
        for is_unanswerable, quota in strata.items():
            candidates = accepted[
                accepted["is_unanswerable"] == is_unanswerable
            ].copy()
            candidates["sample_rank"] = candidates["item_id"].map(
                lambda item_id: stable_hash(
                    f"sample|{strategy}|{is_unanswerable}|{int(item_id)}"
                )
            )
            candidates = candidates.sort_values("sample_rank")
            if len(candidates) < quota:
                raise ValueError(
                    f"Not enough rows for {strategy}, "
                    f"is_unanswerable={is_unanswerable}: "
                    f"need {quota}, found {len(candidates)}"
                )
            sample = candidates.head(quota).copy()
            sample["sampling_stratum"] = (
                "unanswerable" if is_unanswerable else "answerable"
            )
            selected.append(sample)

    audit = pd.concat(selected, ignore_index=True)
    if len(audit) != 150:
        raise ValueError(f"Expected 150 sampled rows, found {len(audit)}")
    counts = audit.groupby("strategy").size().to_dict()
    if any(counts.get(strategy) != 50 for strategy in SAMPLE_PLAN):
        raise ValueError(f"Unexpected strategy counts: {counts}")

    audit["blind_rank"] = audit.apply(
        lambda row: stable_hash(
            f"blind|{row['strategy']}|{int(row['item_id'])}"
        ),
        axis=1,
    )
    audit = audit.sort_values("blind_rank").reset_index(drop=True)
    audit["audit_id"] = [
        f"RA-{index:03d}"
        for index in range(1, len(audit) + 1)
    ]
    audit["batch_no"] = audit.index // 50 + 1
    return audit


def build_blind(audit: pd.DataFrame) -> pd.DataFrame:
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
    return blind[BLIND_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    audit = choose_random_rows(source)
    blind = build_blind(audit)
    os.makedirs(args.output_dir, exist_ok=True)

    combined_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_150_blind.csv",
    )
    key_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_150_key.csv",
    )
    instructions_path = os.path.join(
        args.output_dir,
        "qasper_random_author_audit_instructions.txt",
    )
    blind.to_csv(combined_path, index=False, encoding="utf-8-sig")

    for batch_no in range(1, 4):
        batch_ids = set(
            audit.loc[audit["batch_no"] == batch_no, "audit_id"]
        )
        batch = blind[blind["audit_id"].isin(batch_ids)].copy()
        batch_path = os.path.join(
            args.output_dir,
            f"qasper_random_author_audit_batch_{batch_no:02d}.csv",
        )
        batch.to_csv(batch_path, index=False, encoding="utf-8-sig")

    key = audit[
        [
            "audit_id",
            "batch_no",
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
        "QASPER random 150-case author blind audit",
        "==========================================",
        "",
        "Review batch_01, batch_02, and batch_03 in order.",
        "Do not open the combined key file until all 150 labels are complete.",
        "",
        "Allowed author_label values:",
        "- supported: every material claim is supported by the context",
        "- unsupported: at least one material claim lacks support, conflicts",
        "  with the context, or is attributed to the wrong method or study",
        "- uncertain: the question or evidence is genuinely ambiguous",
        "",
        "Use only the retrieved context. Do not use outside knowledge.",
        "Do not penalize an answer merely for being incomplete.",
        "The same question may recur with a different system answer; review",
        "each row independently and do not carry a prior label across rows.",
        "Check numerical metrics, method attribution, comparison relations,",
        "and whether Related Work is incorrectly described as the paper's",
        "own method.",
        "",
        "Fill author_label and author_notes directly in each batch CSV.",
        "Do not rename columns or reorder rows.",
        "",
        "Sampling boundary:",
        "Each system contributes 50 randomly selected accepted answers.",
        "The sample was not selected using any automated judge label.",
    ]
    with open(instructions_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(instructions) + "\n")

    print(f"Combined blind sheet: {combined_path}")
    for batch_no in range(1, 4):
        print(
            "Batch: "
            + os.path.join(
                args.output_dir,
                f"qasper_random_author_audit_batch_{batch_no:02d}.csv",
            )
        )
    print(f"Private key: {key_path}")
    print(f"Instructions: {instructions_path}")
    print(
        key.groupby(["strategy", "sampling_stratum"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
