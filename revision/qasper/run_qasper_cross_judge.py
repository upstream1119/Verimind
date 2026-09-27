"""Re-evaluate accepted QASPER answers with independent LLM judges."""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, Set, Tuple

import pandas as pd
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "revision", "cross_arch"))
from llm_clients import build_client  # noqa: E402


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_5_qasper_cross_judge",
    "qasper_cross_judge_results.csv",
)
MODEL_NAMES = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
    "glm": "glm-4.5-air",
}
OUTPUT_COLUMNS = [
    "item_id",
    "question_id",
    "strategy",
    "judge",
    "judge_model",
    "faith_score",
    "unsupported_event",
    "threshold",
    "source",
    "prompt_version",
    "evaluated_at_utc",
]


def load_completed(output_path: str) -> Set[Tuple[int, str, str]]:
    if not os.path.exists(output_path):
        return set()
    existing = pd.read_csv(output_path)
    required = {"item_id", "strategy", "judge"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(
            f"Existing output is missing columns: {sorted(missing)}"
        )
    return {
        (int(row.item_id), str(row.strategy), str(row.judge))
        for row in existing.itertuples()
    }


def append_result(output_path: str, result: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame([result], columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False,
        encoding="utf-8-sig",
    )


def make_result(
    row,
    judge: str,
    score: float,
    threshold: float,
    source: str,
) -> Dict[str, object]:
    return {
        "item_id": int(row.item_id),
        "question_id": row.question_id,
        "strategy": row.strategy,
        "judge": judge,
        "judge_model": MODEL_NAMES[judge],
        "faith_score": float(score),
        "unsupported_event": int(float(score) < threshold),
        "threshold": threshold,
        "source": source,
        "prompt_version": "faithfulness_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def selected_rows(
    input_path: str,
    limit: int,
) -> Iterable[object]:
    source = pd.read_csv(input_path)
    required = {
        "item_id",
        "question_id",
        "strategy",
        "accepted",
        "pre_gate_faith_score",
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

    accepted = source[source["accepted"] == 1].copy()
    accepted = accepted.sort_values(["item_id", "strategy"])
    if limit:
        accepted = accepted.head(limit)
    return accepted.itertuples(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["qwen", "glm"],
        default=["qwen", "glm"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of accepted answers for a smoke test.",
    )
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    rows = list(selected_rows(args.input, args.limit))
    completed = load_completed(args.output)
    clients = {
        provider: build_client(provider)
        for provider in args.providers
    }

    expected_new = len(rows) * (1 + len(args.providers))
    print(
        f"Accepted answers selected: {len(rows)}; "
        f"judge records in scope: {expected_new}"
    )
    print(f"Resuming with {len(completed)} completed records")

    written = 0
    failures = 0
    for row_index, row in enumerate(rows, start=1):
        deepseek_key = (int(row.item_id), row.strategy, "deepseek")
        if deepseek_key not in completed:
            result = make_result(
                row,
                judge="deepseek",
                score=float(row.pre_gate_faith_score),
                threshold=args.threshold,
                source="reused_original_run",
            )
            append_result(args.output, result)
            completed.add(deepseek_key)
            written += 1

        for provider in args.providers:
            key = (int(row.item_id), row.strategy, provider)
            if key in completed:
                continue
            try:
                score = clients[provider].score_faithfulness(
                    str(row.generated_answer),
                    str(row.retrieved_context),
                )
                result = make_result(
                    row,
                    judge=provider,
                    score=score,
                    threshold=args.threshold,
                    source="api_re_evaluation",
                )
                append_result(args.output, result)
                completed.add(key)
                written += 1
            except Exception as error:
                failures += 1
                print(
                    f"[ERROR] item={row.item_id} strategy={row.strategy} "
                    f"judge={provider}: {type(error).__name__}: {error}"
                )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        if row_index % 25 == 0 or row_index == len(rows):
            print(
                f"[{row_index}/{len(rows)}] written={written} "
                f"failures={failures} total_completed={len(completed)}"
            )

    print(f"Results: {args.output}")
    print(f"New records: {written}; API failures: {failures}")
    if failures:
        print("Re-run the same command to retry missing judge records.")


if __name__ == "__main__":
    main()
