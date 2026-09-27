"""Run a strict binary cross-judge audit on the 50 author-reviewed cases."""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Set, Tuple

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
    "1_6_qasper_author_audit",
    "qasper_author_audit_50_completed.csv",
)
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_7_qasper_binary_judge_audit",
    "qasper_binary_judge_results.csv",
)
PROVIDERS = ["deepseek", "qwen", "glm"]
MODEL_NAMES = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
    "glm": "glm-4.5-air",
}
OUTPUT_COLUMNS = [
    "audit_id",
    "judge",
    "judge_model",
    "binary_label",
    "unsupported_event",
    "status",
    "raw_response",
    "error_message",
    "prompt_version",
    "context_chars",
    "evaluated_at_utc",
]


def parse_binary_label(raw_response: str) -> Optional[str]:
    normalized = (raw_response or "").strip().upper()
    if normalized == "SUPPORTED":
        return "supported"
    if normalized == "UNSUPPORTED":
        return "unsupported"
    return None


def make_prompt(question: str, answer: str, context: str) -> str:
    return (
        "Judge whether the generated answer is supported by the retrieved "
        "context.\n"
        "Use only the retrieved context. Do not use outside knowledge.\n"
        "Label SUPPORTED only if every material factual claim in the answer "
        "is directly supported by the context or is a reasonable paraphrase.\n"
        "Label UNSUPPORTED if any material factual claim lacks support or "
        "contradicts the context.\n"
        "Do not penalize an answer merely for being incomplete.\n"
        "Output exactly one token: SUPPORTED or UNSUPPORTED.\n\n"
        f"[Question]\n{question}\n\n"
        f"[Retrieved Context]\n{context}\n\n"
        f"[Generated Answer]\n{answer}\n\n"
        "[Label]\n"
    )


def load_completed(output_path: str) -> Set[Tuple[str, str]]:
    if not os.path.exists(output_path):
        return set()
    existing = pd.read_csv(output_path, keep_default_na=False)
    required = {"audit_id", "judge", "status"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(
            f"Existing output is missing columns: {sorted(missing)}"
        )
    successful = existing[existing["status"] == "ok"]
    duplicate_successes = int(
        successful.duplicated(subset=["audit_id", "judge"]).sum()
    )
    if duplicate_successes:
        raise ValueError(
            f"Existing output has {duplicate_successes} duplicate successes"
        )
    return set(zip(successful["audit_id"], successful["judge"]))


def append_result(output_path: str, result: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame([result], columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False,
        encoding="utf-8-sig",
    )


def result_record(
    audit_id: str,
    provider: str,
    context_chars: int,
    status: str,
    raw_response: str = "",
    error_message: str = "",
) -> Dict[str, object]:
    label = parse_binary_label(raw_response)
    return {
        "audit_id": audit_id,
        "judge": provider,
        "judge_model": MODEL_NAMES[provider],
        "binary_label": label or "",
        "unsupported_event": (
            int(label == "unsupported") if label is not None else ""
        ),
        "status": status,
        "raw_response": raw_response,
        "error_message": error_message,
        "prompt_version": "binary_faithfulness_v1",
        "context_chars": context_chars,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=PROVIDERS,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    source = pd.read_csv(args.input, keep_default_na=False)
    required = {
        "audit_id",
        "question",
        "retrieved_context",
        "generated_answer",
        "author_label",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if len(source) != 50:
        raise ValueError(f"Expected 50 audit rows, found {len(source)}")
    if source["audit_id"].duplicated().any():
        raise ValueError("Input contains duplicate audit_id values")
    if args.limit:
        source = source.head(args.limit)

    completed = load_completed(args.output)
    clients = {
        provider: build_client(provider)
        for provider in args.providers
    }
    print(
        f"Audit cases selected: {len(source)}; "
        f"successful records expected: "
        f"{len(source) * len(args.providers)}"
    )
    print(f"Resuming with {len(completed)} successful records")

    successful = 0
    parse_errors = 0
    api_errors = 0
    for row_index, row in enumerate(source.itertuples(), start=1):
        for provider in args.providers:
            key = (row.audit_id, provider)
            if key in completed:
                continue
            context = str(row.retrieved_context)
            prompt = make_prompt(
                str(row.question),
                str(row.generated_answer),
                context,
            )
            try:
                raw_response = clients[provider].complete(
                    prompt,
                    max_tokens=8,
                    temperature=0.01,
                )
                label = parse_binary_label(raw_response)
                if label is None:
                    parse_errors += 1
                    record = result_record(
                        row.audit_id,
                        provider,
                        len(context),
                        status="parse_error",
                        raw_response=raw_response,
                        error_message=(
                            "Expected exactly SUPPORTED or UNSUPPORTED"
                        ),
                    )
                else:
                    successful += 1
                    completed.add(key)
                    record = result_record(
                        row.audit_id,
                        provider,
                        len(context),
                        status="ok",
                        raw_response=raw_response,
                    )
            except Exception as error:
                api_errors += 1
                record = result_record(
                    row.audit_id,
                    provider,
                    len(context),
                    status="api_error",
                    error_message=f"{type(error).__name__}: {error}",
                )
            append_result(args.output, record)
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        if row_index % 10 == 0 or row_index == len(source):
            print(
                f"[{row_index}/{len(source)}] ok={successful} "
                f"parse_errors={parse_errors} api_errors={api_errors}"
            )

    print(f"Results: {args.output}")
    print(
        f"New successful records: {successful}; "
        f"parse errors: {parse_errors}; API errors: {api_errors}"
    )
    if parse_errors or api_errors:
        print("Re-run the same command to retry unsuccessful judge records.")


if __name__ == "__main__":
    main()
