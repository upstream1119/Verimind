"""Recompute publication metrics with explicit denominators and true ROUGE-L."""

import json
import os
import re
from typing import Dict, Iterable, List

import pandas as pd
from rouge_score import rouge_scorer


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MAIN_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_2_baseline",
    "1_2_1_main",
    "experiment_results_sci_v3.csv",
)
MAIN_BENCHMARK = os.path.join(
    PROJECT_ROOT,
    "json",
    "verimind_bench_v3.json",
)
SCIFACT_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_1_scifact",
    "results",
    "scifact_experiment_results.csv",
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "4_5_corrected_metrics",
)
REFUSAL_PATTERN = re.compile(
    r"information not found|not found in the (?:text|context)|"
    r"rejected|unable to determine|cannot be determined|cannot find|"
    r"insufficient evidence|no (?:relevant )?(?:evidence|information)|"
    r"(?:does not|doesn't) (?:mention|provide|contain|report)|"
    r"not (?:mentioned|provided|reported|available) in the "
    r"(?:text|context|documents?)",
    re.IGNORECASE,
)
ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def is_accepted(answer: str) -> int:
    return int(not REFUSAL_PATTERN.search(answer or ""))


def rouge_l(gold: str, prediction: str) -> float:
    if not gold or not prediction:
        return 0.0
    return ROUGE.score(gold, prediction)["rougeL"].fmeasure


def summarize(
    df: pd.DataFrame,
    rouge_column: str,
    retrieval_column: str,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for strategy, group in df.groupby("strategy", sort=False):
        accepted = group[group["accepted"] == 1]
        passed_count = int(accepted["hallucination_event"].sum())
        rows.append(
            {
                "strategy": strategy,
                "n": len(group),
                "accepted_n": len(accepted),
                "answer_coverage": group["accepted"].mean(),
                "passed_hallucinations_n": passed_count,
                "passed_hallucination_rate": (
                    passed_count / len(accepted)
                    if len(accepted)
                    else 0.0
                ),
                "retrieval_hit_rate": group[retrieval_column].mean(),
                "rouge_l_overall": group[rouge_column].mean(),
                "rouge_l_accepted": (
                    accepted[rouge_column].mean()
                    if len(accepted)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def recompute_main() -> pd.DataFrame:
    with open(MAIN_BENCHMARK, "r", encoding="utf-8") as benchmark_file:
        benchmark = json.load(benchmark_file)
    gold_by_id = {
        str(index): item.get("gold_answer", "")
        for index, item in enumerate(benchmark)
    }

    df = pd.read_csv(MAIN_INPUT)
    df["question_id"] = df["question_id"].astype(str)
    df["gold_answer"] = df["question_id"].map(gold_by_id).fillna("")
    if (df["gold_answer"] == "").any():
        missing = df.loc[df["gold_answer"] == "", "question_id"].unique()
        raise ValueError(f"Missing gold answers for question ids: {missing[:10]}")

    df["accepted"] = df["generated_answer"].map(is_accepted)
    df["hallucination_event"] = (
        pd.to_numeric(df["hallucination_rate"], errors="coerce")
        .fillna(0)
        .ge(0.5)
        .astype(int)
    )
    df["rouge_l_recomputed"] = [
        rouge_l(gold, prediction)
        for gold, prediction in zip(
            df["gold_answer"],
            df["generated_answer"],
        )
    ]
    enriched_path = os.path.join(
        OUTPUT_DIR,
        "verimind_bench_results_corrected.csv",
    )
    df.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    return summarize(
        df,
        rouge_column="rouge_l_recomputed",
        retrieval_column="retrieval_hit",
    )


def recompute_scifact() -> pd.DataFrame:
    df = pd.read_csv(SCIFACT_INPUT)
    df["accepted"] = df["generated_answer"].map(is_accepted)
    df["hallucination_event"] = (
        pd.to_numeric(df["hallucination_rate"], errors="coerce")
        .fillna(0)
        .ge(0.5)
        .astype(int)
    )
    return summarize(
        df,
        rouge_column="rouge_l_f1",
        retrieval_column="coverage_hit",
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main_summary = recompute_main()
    scifact_summary = recompute_scifact()

    main_path = os.path.join(
        OUTPUT_DIR,
        "verimind_bench_publication_summary.csv",
    )
    scifact_path = os.path.join(
        OUTPUT_DIR,
        "scifact_publication_summary.csv",
    )
    main_summary.to_csv(main_path, index=False, encoding="utf-8-sig")
    scifact_summary.to_csv(scifact_path, index=False, encoding="utf-8-sig")

    print("VeriMind-Bench corrected summary")
    print(main_summary.round(4).to_string(index=False))
    print("\nSciFact corrected summary")
    print(scifact_summary.round(4).to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
