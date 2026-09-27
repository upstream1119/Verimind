"""Run attribution-aware Gate V2 judgments on QASPER development cases."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

import pandas as pd
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "revision", "cross_arch"))
from llm_clients import build_client  # noqa: E402

from revision.qasper.gate_v2 import (  # noqa: E402
    POLICY_VERSION,
    PROMPT_VERSION,
    make_gate_prompt,
    parse_gate_judgment,
)


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_13_qasper_no_gating_complete_audit",
    "qasper_no_gating_149_scored.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_14_qasper_gate_v2_dev",
)
DEFAULT_OUTPUT = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "gate_v2_raw_judgments.csv",
)
DEFAULT_CONFIG = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "gate_v2_config.json",
)
PROVIDERS = ["deepseek", "qwen", "glm"]
MODEL_NAMES = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
    "glm": "glm-4.5-air",
}
OUTPUT_COLUMNS = [
    "item_id",
    "audit_id",
    "judge",
    "judge_model",
    "model_label",
    "label",
    "label_repaired",
    "unsupported_event",
    "uncertain_event",
    "claim_count",
    "unsupported_claim_count",
    "uncertain_claim_count",
    "claims_json",
    "summary_reason",
    "status",
    "raw_response",
    "error_message",
    "prompt_version",
    "policy_version",
    "question_chars",
    "context_chars",
    "answer_chars",
    "prompt_chars",
    "evaluated_at_utc",
]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(
    source: pd.DataFrame,
    item_ids: List[int],
    limit: int,
) -> pd.DataFrame:
    required = {
        "item_id",
        "audit_id",
        "question",
        "generated_answer",
        "retrieved_context",
        "author_label",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if source["item_id"].duplicated().any():
        raise ValueError("Input contains duplicate item IDs")
    selected = source.copy()
    if item_ids:
        selected = selected[selected["item_id"].isin(item_ids)].copy()
        missing_ids = sorted(set(item_ids) - set(selected["item_id"]))
        if missing_ids:
            raise ValueError(f"Requested item IDs are absent: {missing_ids}")
    if limit:
        selected = selected.head(limit).copy()
    if not len(selected):
        raise ValueError("No development cases were selected")
    return selected.sort_values("item_id").reset_index(drop=True)


def load_completed(output_path: str) -> Set[Tuple[int, str]]:
    if not os.path.exists(output_path):
        return set()
    existing = pd.read_csv(output_path, keep_default_na=False)
    successful = existing[existing["status"] == "ok"]
    duplicate_successes = int(
        successful.duplicated(subset=["item_id", "judge"]).sum()
    )
    if duplicate_successes:
        raise ValueError(
            f"Existing output has {duplicate_successes} duplicate successes"
        )
    return {
        (int(row.item_id), str(row.judge))
        for row in successful.itertuples()
    }


def append_record(output_path: str, record: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame([record], columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False,
        encoding="utf-8-sig",
    )


def make_record(
    row,
    provider: str,
    prompt: str,
    status: str,
    raw_response: str = "",
    parsed: Dict[str, object] | None = None,
    error_message: str = "",
) -> Dict[str, object]:
    parsed = parsed or {}
    label = str(parsed.get("label", ""))
    claims = parsed.get("claims", [])
    return {
        "item_id": int(row.item_id),
        "audit_id": str(row.audit_id),
        "judge": provider,
        "judge_model": MODEL_NAMES[provider],
        "model_label": parsed.get("model_label", ""),
        "label": label,
        "label_repaired": parsed.get("label_repaired", ""),
        "unsupported_event": (
            int(label == "unsupported") if label else ""
        ),
        "uncertain_event": int(label == "uncertain") if label else "",
        "claim_count": len(claims) if claims else "",
        "unsupported_claim_count": parsed.get(
            "unsupported_claim_count",
            "",
        ),
        "uncertain_claim_count": parsed.get(
            "uncertain_claim_count",
            "",
        ),
        "claims_json": (
            json.dumps(claims, ensure_ascii=False) if claims else ""
        ),
        "summary_reason": parsed.get("summary_reason", ""),
        "status": status,
        "raw_response": raw_response,
        "error_message": error_message,
        "prompt_version": PROMPT_VERSION,
        "policy_version": POLICY_VERSION,
        "question_chars": len(str(row.question)),
        "context_chars": len(str(row.retrieved_context)),
        "answer_chars": len(str(row.generated_answer)),
        "prompt_chars": len(prompt),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_config(
    path: str,
    input_path: str,
    selected: pd.DataFrame,
    providers: List[str],
    dry_run: bool,
    max_tokens: int,
    temperature: float,
) -> None:
    config = {
        "input_path": os.path.abspath(input_path),
        "input_sha256": file_sha256(input_path),
        "selected_item_ids": [
            int(item_id) for item_id in selected["item_id"]
        ],
        "selected_n": len(selected),
        "providers": providers,
        "judge_models": {
            provider: MODEL_NAMES[provider]
            for provider in providers
        },
        "expected_judgments": len(selected) * len(providers),
        "dry_run": dry_run,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_retries_per_judgment": 1,
        "prompt_version": PROMPT_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(config, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=PROVIDERS,
    )
    parser.add_argument("--item-ids", nargs="*", type=int, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    source = pd.read_csv(args.input, keep_default_na=False)
    selected = select_cases(source, args.item_ids, args.limit)
    write_config(
        args.config,
        args.input,
        selected,
        args.providers,
        args.dry_run,
        args.max_tokens,
        args.temperature,
    )

    prompts = {
        int(row.item_id): make_gate_prompt(
            str(row.question),
            str(row.generated_answer),
            str(row.retrieved_context),
        )
        for row in selected.itertuples()
    }
    print(
        f"Selected cases: {len(selected)}; providers: {args.providers}; "
        f"expected judgments: {len(selected) * len(args.providers)}"
    )
    for row in selected.itertuples():
        print(
            f"item={int(row.item_id)} author={row.author_label} "
            f"prompt_chars={len(prompts[int(row.item_id)])}"
        )
    print(f"Config: {args.config}")
    if args.dry_run:
        print("Dry-run complete; no API clients were constructed.")
        return

    completed = load_completed(args.output)
    clients = {
        provider: build_client(provider, max_retries=1)
        for provider in args.providers
    }
    print(f"Resuming with {len(completed)} successful judgments")
    successful = 0
    parse_errors = 0
    api_errors = 0
    for row in selected.itertuples():
        prompt = prompts[int(row.item_id)]
        for provider in args.providers:
            key = (int(row.item_id), provider)
            if key in completed:
                continue
            raw_response = ""
            try:
                raw_response = clients[provider].complete(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                try:
                    parsed = parse_gate_judgment(raw_response)
                except (json.JSONDecodeError, ValueError) as error:
                    parse_errors += 1
                    record = make_record(
                        row,
                        provider,
                        prompt,
                        status="parse_error",
                        raw_response=raw_response,
                        error_message=f"{type(error).__name__}: {error}",
                    )
                else:
                    successful += 1
                    completed.add(key)
                    record = make_record(
                        row,
                        provider,
                        prompt,
                        status="ok",
                        raw_response=raw_response,
                        parsed=parsed,
                    )
            except Exception as error:
                api_errors += 1
                record = make_record(
                    row,
                    provider,
                    prompt,
                    status="api_error",
                    raw_response=raw_response,
                    error_message=f"{type(error).__name__}: {error}",
                )
            append_record(args.output, record)
            print(
                f"item={int(row.item_id)} judge={provider} "
                f"status={record['status']} label={record['label'] or '-'}"
            )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    print(f"Raw judgments: {args.output}")
    print(
        f"New successful={successful}; parse_errors={parse_errors}; "
        f"api_errors={api_errors}"
    )
    if parse_errors or api_errors:
        print(
            "Smoke test is incomplete. Do not rerun API calls automatically; "
            "inspect the recorded errors first."
        )


if __name__ == "__main__":
    main()
