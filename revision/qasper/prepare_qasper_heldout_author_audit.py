"""Prepare blind author audit sheets for held-out QASPER Gate V2 validation."""

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
    "1_15_qasper_gate_v2_heldout",
    "gate_v2_heldout_case_decisions.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
)
SEED = "qasper-heldout-author-audit-v1"
EXPECTED_ROWS = 96


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        f"{SEED}|{value}".encode("utf-8")
    ).hexdigest()


def prepare_blind_audit(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "audit_id",
        "item_id",
        "question_id",
        "paper_id",
        "question",
        "retrieved_context",
        "generated_answer",
        "author_label",
        "gate_v2_decision",
        "deepseek_label",
        "qwen_label",
        "glm_label",
        "unsupported_votes",
        "uncertain_votes",
        "current_gate_reject",
        "trust_score",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            f"Held-out case decisions are missing columns: {sorted(missing)}"
        )
    if len(source) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} held-out rows, found {len(source)}"
        )
    if source["item_id"].duplicated().any():
        raise ValueError("Held-out case decisions contain duplicate item IDs")
    if source["audit_id"].duplicated().any():
        raise ValueError("Held-out case decisions contain duplicate audit IDs")
    labels = set(source["author_label"].astype(str).str.strip().str.lower())
    if labels != {"unreviewed"}:
        raise ValueError(
            "Held-out cases must be unreviewed before blind audit; "
            f"found labels: {sorted(labels)}"
        )

    audit = source.copy()
    audit["blind_rank"] = audit["item_id"].map(
        lambda item_id: stable_hash(int(item_id))
    )
    audit = audit.sort_values("blind_rank").reset_index(drop=True)
    audit = audit.rename(columns={"audit_id": "heldout_audit_id"})
    audit.insert(
        0,
        "audit_id",
        [f"HOB-{index:03d}" for index in range(1, len(audit) + 1)],
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

    key = audit[
        [
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
        ]
    ].copy()
    return blind, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = pd.read_csv(args.input, keep_default_na=False)
    blind, key = prepare_blind_audit(source)
    os.makedirs(args.output_dir, exist_ok=True)

    blind_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_96_blind.csv",
    )
    key_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_96_key.csv",
    )
    instructions_path = os.path.join(
        args.output_dir,
        "qasper_heldout_author_audit_instructions.txt",
    )

    blind.to_csv(blind_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    instructions = [
        "QASPER Gate V2 held-out 96-case author blind audit",
        "==================================================",
        "",
        "Open only qasper_heldout_author_audit_96_blind.csv.",
        "Do not open the key file until all 96 labels are complete.",
        "",
        "Allowed author_label values:",
        "- supported: every material claim is supported by the retrieved context",
        "- unsupported: at least one material claim lacks support, conflicts",
        "  with the context, or is attributed to the wrong method or study",
        "- uncertain: the question or evidence is genuinely ambiguous",
        "",
        "Use only the retrieved context. Do not use outside knowledge.",
        "Do not penalize an answer merely for being incomplete.",
        "Check numerical metrics, method attribution, comparison relations,",
        "and whether prior work is incorrectly described as the current",
        "paper's own method, result, dataset, or contribution.",
        "",
        "Fill author_label and author_notes without renaming columns or",
        "reordering rows. Save as UTF-8 CSV.",
        "",
        "Blindness boundary:",
        "The blind sheet intentionally excludes Gate V2 decisions, judge",
        "labels, trust scores, item IDs, paper IDs, and gold answers.",
        "Open the key only after all labels are complete.",
    ]
    with open(instructions_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(instructions) + "\n")

    print(f"Blind audit sheet: {blind_path}")
    print(f"Private key: {key_path}")
    print(f"Instructions: {instructions_path}")
    print(f"Rows: {len(blind)}")


if __name__ == "__main__":
    main()
