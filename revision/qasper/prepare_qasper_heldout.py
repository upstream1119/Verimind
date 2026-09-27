"""Prepare deterministic held-out QASPER questions for Gate V2 validation."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from datasets import load_dataset

from revision.qasper.prepare_qasper import (
    QASPER_VALIDATION_PARQUET,
    flatten_questions,
    stable_key,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_ORIGINAL_BENCH = os.path.join(BASE_DIR, "qasper_bench.json")
DEFAULT_INDEXED_CORPUS = os.path.join(BASE_DIR, "qasper_corpus.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_15_qasper_gate_v2_heldout",
)
DEFAULT_BENCH_OUTPUT = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_bench_150.json",
)
DEFAULT_MANIFEST_OUTPUT = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_selection_manifest.json",
)
DEFAULT_SUMMARY_OUTPUT = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "qasper_heldout_selection_summary.txt",
)
SEED = "qasper-gate-v2-heldout-v1"
DEFAULT_SAMPLE_SIZE = 150
HELDOUT_ITEM_ID_OFFSET = 1000


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_indexed_paper_ids(corpus_path: str) -> set:
    paper_ids = set()
    with open(corpus_path, "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if not line.strip():
                continue
            paper = json.loads(line)
            paper_ids.add(str(paper["paper_id"]))
    return paper_ids


def load_all_questions(cache_dir: str) -> List[Dict[str, object]]:
    dataset = load_dataset(
        "parquet",
        data_files={"validation": QASPER_VALIDATION_PARQUET},
        split="validation",
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
    )
    questions: List[Dict[str, object]] = []
    for row in dataset:
        questions.extend(flatten_questions(row))
    return questions


def select_heldout(
    all_questions: Iterable[Dict[str, object]],
    original_questions: Iterable[Dict[str, object]],
    indexed_paper_ids: set,
    sample_size: int,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    used_question_ids = {
        str(item["question_id"]) for item in original_questions
    }
    all_questions = list(all_questions)
    indexed_unused = [
        item
        for item in all_questions
        if str(item["paper_id"]) in indexed_paper_ids
        and str(item["question_id"]) not in used_question_ids
    ]
    candidates = [
        item for item in indexed_unused if not bool(item["is_unanswerable"])
    ]
    candidates = sorted(
        candidates,
        key=lambda item: stable_key(
            f"{SEED}|{item['question_id']}"
        ),
    )
    if len(candidates) < sample_size:
        raise ValueError(
            f"Need {sample_size} held-out answerable questions, "
            f"but only found {len(candidates)}"
        )
    selected = []
    for rank, item in enumerate(candidates[:sample_size], start=1):
        copied = dict(item)
        copied["item_id"] = HELDOUT_ITEM_ID_OFFSET + rank - 1
        copied["heldout_rank"] = rank
        copied["heldout_seed"] = SEED
        selected.append(copied)
    counts = {
        "all_questions": len(all_questions),
        "original_questions": len(used_question_ids),
        "indexed_paper_count": len(indexed_paper_ids),
        "indexed_unused_questions": len(indexed_unused),
        "answerable_candidates": len(candidates),
        "selected_questions": len(selected),
        "selected_papers": len({item["paper_id"] for item in selected}),
    }
    return selected, counts


def write_summary(
    path: str,
    selected: List[Dict[str, object]],
    counts: Dict[str, int],
) -> None:
    lines = [
        "QASPER Gate V2 held-out selection",
        "==================================",
        "",
        f"Seed: {SEED}",
        f"Selected questions: {counts['selected_questions']}",
        f"Selected papers: {counts['selected_papers']}",
        f"Candidate questions: {counts['answerable_candidates']}",
        f"Indexed unused questions: {counts['indexed_unused_questions']}",
        f"Original benchmark questions excluded: {counts['original_questions']}",
        "",
        "Interpretation boundary:",
        (
            "This is an answerable held-out question set selected from "
            "papers already present in the existing QASPER index. It is "
            "intended for frozen Gate V2 validation, not for further "
            "calibration."
        ),
        "",
        "First 10 selected question IDs:",
    ]
    lines.extend(
        f"{item['heldout_rank']:03d}: {item['question_id']}"
        for item in selected[:10]
    )
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--cache-dir", default=os.path.join(BASE_DIR, "raw"))
    parser.add_argument("--original-bench", default=DEFAULT_ORIGINAL_BENCH)
    parser.add_argument("--indexed-corpus", default=DEFAULT_INDEXED_CORPUS)
    parser.add_argument("--bench-output", default=DEFAULT_BENCH_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()

    original_questions = load_json(args.original_bench)
    if not isinstance(original_questions, list):
        raise ValueError("Original benchmark must be a list")
    indexed_paper_ids = load_indexed_paper_ids(args.indexed_corpus)
    all_questions = load_all_questions(args.cache_dir)
    selected, counts = select_heldout(
        all_questions,
        original_questions,
        indexed_paper_ids,
        args.sample_size,
    )

    os.makedirs(os.path.dirname(args.bench_output), exist_ok=True)
    with open(args.bench_output, "w", encoding="utf-8") as output_file:
        json.dump(selected, output_file, ensure_ascii=False, indent=2)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "sample_size": args.sample_size,
        "heldout_item_id_offset": HELDOUT_ITEM_ID_OFFSET,
        "selection_rule": (
            "Exclude original qasper_bench.json question IDs; keep "
            "answerable validation questions whose paper_id is present in "
            "the existing qasper_corpus.jsonl index; sort by fixed SHA-256 "
            "stable key and take the first sample_size."
        ),
        "counts": counts,
        "inputs": {
            "original_bench": os.path.abspath(args.original_bench),
            "original_bench_sha256": file_sha256(args.original_bench),
            "indexed_corpus": os.path.abspath(args.indexed_corpus),
            "indexed_corpus_sha256": file_sha256(args.indexed_corpus),
            "cache_dir": os.path.abspath(args.cache_dir),
            "qasper_validation_source": QASPER_VALIDATION_PARQUET,
        },
        "selected_question_ids": [
            str(item["question_id"]) for item in selected
        ],
        "selected_item_ids": [int(item["item_id"]) for item in selected],
    }
    with open(args.manifest_output, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
    write_summary(args.summary_output, selected, counts)

    print(f"Held-out questions selected: {len(selected)}")
    print(f"Candidate questions: {counts['answerable_candidates']}")
    print(f"Selected papers: {counts['selected_papers']}")
    print(f"Benchmark: {args.bench_output}")
    print(f"Manifest: {args.manifest_output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
