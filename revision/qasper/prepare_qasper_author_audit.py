"""Prepare a deterministic, challenge-enriched blind QASPER audit."""

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
    "1_6_qasper_author_audit",
)
SEED = "qasper-author-audit-v1"
SAMPLING_PLAN: Dict[str, Dict[str, int]] = {
    "Vanilla_RAG": {"unsupported": 8, "supported": 9},
    "HyDE": {"unsupported": 7, "supported": 9},
    "VeriMind_Full": {"unsupported": 1, "supported": 16},
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        f"{SEED}|{value}".encode("utf-8")
    ).hexdigest()


def choose_rows(source: pd.DataFrame) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    for strategy, quotas in SAMPLING_PLAN.items():
        strategy_rows = source[
            (source["accepted"] == 1)
            & (source["strategy"] == strategy)
        ].copy()
        strategy_rows["audit_rank"] = strategy_rows["item_id"].map(
            lambda item_id: stable_hash(f"{strategy}|{int(item_id)}")
        )
        for label_name, quota in quotas.items():
            event = int(label_name == "unsupported")
            candidates = strategy_rows[
                strategy_rows["passed_hallucination"] == event
            ].sort_values("audit_rank")
            if len(candidates) < quota:
                raise ValueError(
                    f"Not enough {label_name} rows for {strategy}: "
                    f"need {quota}, found {len(candidates)}"
                )
            sample = candidates.head(quota).copy()
            sample["sampling_stratum"] = label_name
            selected.append(sample)

    audit = pd.concat(selected, ignore_index=True)
    if len(audit) != 50:
        raise ValueError(f"Expected 50 audit rows, found {len(audit)}")
    audit["blind_rank"] = audit.apply(
        lambda row: stable_hash(
            f"blind|{row['strategy']}|{int(row['item_id'])}"
        ),
        axis=1,
    )
    audit = audit.sort_values("blind_rank").reset_index(drop=True)
    audit["audit_id"] = [
        f"QA-{index:03d}"
        for index in range(1, len(audit) + 1)
    ]
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    required = {
        "item_id",
        "question_id",
        "strategy",
        "accepted",
        "passed_hallucination",
        "pre_gate_faith_score",
        "question",
        "generated_answer",
        "retrieved_context",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    audit = choose_rows(source)
    os.makedirs(args.output_dir, exist_ok=True)
    blind_path = os.path.join(
        args.output_dir,
        "qasper_author_audit_50_blind.csv",
    )
    key_path = os.path.join(
        args.output_dir,
        "qasper_author_audit_50_key.csv",
    )
    instructions_path = os.path.join(
        args.output_dir,
        "qasper_author_audit_instructions.txt",
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
            "strategy",
            "sampling_stratum",
            "pre_gate_faith_score",
            "passed_hallucination",
        ]
    ].copy()
    key = key.rename(
        columns={
            "passed_hallucination": "deepseek_unsupported_event",
        }
    )
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    instructions = [
        "QASPER 50-case author blind audit",
        "=================================",
        "",
        "Open only qasper_author_audit_50_blind.csv while labeling.",
        "Do not open the key file until all 50 labels are complete.",
        "",
        "Allowed author_label values:",
        "- supported: every material claim is supported by the context",
        "- unsupported: at least one material claim lacks context support",
        "- uncertain: the evidence is genuinely ambiguous",
        "",
        "Judge faithfulness only. Do not use the gold answer or outside facts.",
        "Save the completed CSV without renaming columns.",
        "",
        "Important: this challenge-enriched sample validates label agreement;",
        "it must not be used to estimate the population hallucination rate.",
    ]
    with open(instructions_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(instructions) + "\n")

    print(f"Blind audit sheet: {blind_path}")
    print(f"Private key: {key_path}")
    print(f"Instructions: {instructions_path}")
    print(
        key.groupby(["strategy", "sampling_stratum"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
