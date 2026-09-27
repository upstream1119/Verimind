"""Prepare a deterministic QASPER evaluation subset.

The script downloads the public validation parquet from Hugging Face, converts
the nested annotations into a compact benchmark, and writes one structured
document per paper for index construction.
"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
DEFAULT_BENCHMARK = os.path.join(BASE_DIR, "qasper_bench.json")
DEFAULT_CORPUS = os.path.join(BASE_DIR, "qasper_corpus.jsonl")
QASPER_VALIDATION_PARQUET = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    "refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet"
)


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique_nonempty(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        normalized = (value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def annotation_to_answer(annotation: Dict[str, Any]) -> str:
    if annotation.get("unanswerable"):
        return ""

    spans = unique_nonempty(annotation.get("extractive_spans") or [])
    if spans:
        return "; ".join(spans)

    free_form = (annotation.get("free_form_answer") or "").strip()
    if free_form:
        return free_form

    yes_no = annotation.get("yes_no")
    if yes_no is True:
        return "Yes"
    if yes_no is False:
        return "No"
    return ""


def flatten_paper(row: Dict[str, Any]) -> Dict[str, Any]:
    section_names = row["full_text"].get("section_name") or []
    section_paragraphs = row["full_text"].get("paragraphs") or []
    sections = []
    for section_index, paragraphs in enumerate(section_paragraphs):
        section_name = (
            section_names[section_index]
            if section_index < len(section_names)
            else f"Section {section_index + 1}"
        )
        sections.append(
            {
                "section_name": (section_name or "").strip(),
                "paragraphs": unique_nonempty(paragraphs or []),
            }
        )

    return {
        "paper_id": str(row["id"]),
        "title": (row.get("title") or "").strip(),
        "abstract": (row.get("abstract") or "").strip(),
        "sections": sections,
        "source": "QASPER validation",
    }


def flatten_questions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    qas = row["qas"]
    questions = qas.get("question") or []
    question_ids = qas.get("question_id") or []
    answer_groups = qas.get("answers") or []
    items = []

    for question_index, question in enumerate(questions):
        answer_group = (
            answer_groups[question_index]
            if question_index < len(answer_groups)
            else {}
        )
        annotations = answer_group.get("answer") or []
        references = unique_nonempty(
            annotation_to_answer(annotation) for annotation in annotations
        )
        is_unanswerable = bool(annotations) and all(
            annotation.get("unanswerable", False) for annotation in annotations
        )
        evidence = unique_nonempty(
            evidence_text
            for annotation in annotations
            for evidence_text in (annotation.get("evidence") or [])
        )
        highlighted_evidence = unique_nonempty(
            evidence_text
            for annotation in annotations
            for evidence_text in (annotation.get("highlighted_evidence") or [])
        )

        if is_unanswerable and not references:
            references = ["Unanswerable"]
        if not references:
            continue

        question_id = (
            str(question_ids[question_index])
            if question_index < len(question_ids)
            else f"{row['id']}-{question_index}"
        )
        items.append(
            {
                "question_id": question_id,
                "paper_id": str(row["id"]),
                "paper_title": (row.get("title") or "").strip(),
                "question": (question or "").strip(),
                "gold_answers": references,
                "gold_answer": references[0],
                "is_unanswerable": is_unanswerable,
                "evidence": evidence,
                "highlighted_evidence": highlighted_evidence,
                "source_split": "validation",
            }
        )
    return items


def select_subset(
    questions: List[Dict[str, Any]],
    sample_size: int,
    unanswerable_ratio: float,
) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(questions):
        selected = list(questions)
    else:
        unanswerable = sorted(
            (q for q in questions if q["is_unanswerable"]),
            key=lambda q: stable_key(q["question_id"]),
        )
        answerable = sorted(
            (q for q in questions if not q["is_unanswerable"]),
            key=lambda q: stable_key(q["question_id"]),
        )
        target_unanswerable = min(
            len(unanswerable),
            round(sample_size * unanswerable_ratio),
        )
        target_answerable = min(len(answerable), sample_size - target_unanswerable)
        remaining = sample_size - target_unanswerable - target_answerable
        if remaining > 0:
            extra_unanswerable = min(
                len(unanswerable) - target_unanswerable,
                remaining,
            )
            target_unanswerable += extra_unanswerable
            remaining -= extra_unanswerable
        if remaining > 0:
            target_answerable += min(
                len(answerable) - target_answerable,
                remaining,
            )
        selected = (
            answerable[:target_answerable]
            + unanswerable[:target_unanswerable]
        )

    selected.sort(key=lambda q: stable_key(q["question_id"]))
    for item_id, item in enumerate(selected):
        item["item_id"] = item_id
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--unanswerable-ratio", type=float, default=0.20)
    parser.add_argument("--benchmark-output", default=DEFAULT_BENCHMARK)
    parser.add_argument("--corpus-output", default=DEFAULT_CORPUS)
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    dataset = load_dataset(
        "parquet",
        data_files={"validation": QASPER_VALIDATION_PARQUET},
        split="validation",
        cache_dir=RAW_DIR,
    )

    corpus = []
    questions = []
    for row in dataset:
        corpus.append(flatten_paper(row))
        questions.extend(flatten_questions(row))

    selected = select_subset(
        questions,
        sample_size=args.sample_size,
        unanswerable_ratio=args.unanswerable_ratio,
    )

    selected_paper_ids = {item["paper_id"] for item in selected}
    selected_corpus = [
        paper for paper in corpus if paper["paper_id"] in selected_paper_ids
    ]

    with open(args.corpus_output, "w", encoding="utf-8") as corpus_file:
        for paper in selected_corpus:
            corpus_file.write(json.dumps(paper, ensure_ascii=False) + "\n")

    with open(args.benchmark_output, "w", encoding="utf-8") as benchmark_file:
        json.dump(selected, benchmark_file, ensure_ascii=False, indent=2)

    answerable_count = sum(not item["is_unanswerable"] for item in selected)
    unanswerable_count = len(selected) - answerable_count
    paper_count = len({item["paper_id"] for item in selected})
    print(
        f"QASPER papers written: {len(selected_corpus)} "
        f"of {len(corpus)} validation papers"
    )
    print(f"QASPER questions available: {len(questions)}")
    print(
        "Benchmark selected: "
        f"{len(selected)} questions across {paper_count} papers "
        f"({answerable_count} answerable, {unanswerable_count} unanswerable)"
    )
    print(f"Benchmark: {args.benchmark_output}")
    print(f"Corpus: {args.corpus_output}")


if __name__ == "__main__":
    main()
