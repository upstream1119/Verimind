"""Prepare accepted held-out QASPER answers for frozen Gate V2 judging."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
    "qasper_heldout_no_gating_results.csv",
)
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
    "qasper_heldout_gate_v2_input.csv",
)
DEFAULT_MANIFEST = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
    "qasper_heldout_gate_v2_input_manifest.json",
)
EXPECTED_ACCEPTED = 96


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_gate_v2_input(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "item_id",
        "question_id",
        "paper_id",
        "strategy",
        "accepted",
        "question",
        "generated_answer",
        "retrieved_context",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Held-out results are missing: {sorted(missing)}")
    selected = source[
        (source["strategy"] == "VeriMind_No_Gating")
        & (source["accepted"].astype(int) == 1)
    ].copy()
    if len(selected) != EXPECTED_ACCEPTED:
        raise ValueError(
            f"Expected {EXPECTED_ACCEPTED} accepted answers, "
            f"found {len(selected)}"
        )
    if selected["item_id"].duplicated().any():
        raise ValueError("Accepted held-out rows contain duplicate item IDs")
    selected = selected.sort_values("item_id").reset_index(drop=True)
    selected.insert(
        0,
        "audit_id",
        [f"HO-{index:03d}" for index in range(1, len(selected) + 1)],
    )
    selected["author_label"] = "unreviewed"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    source = pd.read_csv(args.input, keep_default_na=False)
    prepared = prepare_gate_v2_input(source)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    prepared.to_csv(args.output, index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": os.path.abspath(args.input),
        "input_sha256": file_sha256(args.input),
        "output_path": os.path.abspath(args.output),
        "expected_accepted": EXPECTED_ACCEPTED,
        "selected_n": len(prepared),
        "selection_rule": (
            "Keep VeriMind_No_Gating held-out rows with accepted == 1. "
            "Assign stable audit_id values HO-001..HO-096 and mark "
            "author_label as unreviewed."
        ),
        "selected_item_ids": [
            int(item_id) for item_id in prepared["item_id"]
        ],
        "selected_audit_ids": [
            str(audit_id) for audit_id in prepared["audit_id"]
        ],
    }
    with open(args.manifest, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)

    print(f"Gate V2 held-out input rows: {len(prepared)}")
    print(f"Output: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
