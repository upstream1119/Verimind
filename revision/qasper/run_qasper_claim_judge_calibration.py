"""Run claim-level judge calibration on the 50 author-reviewed QASPER cases."""

import argparse
import json
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
    "1_8_qasper_claim_judge_calibration",
    "qasper_claim_judge_results.csv",
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
    "model_label",
    "label",
    "label_repaired",
    "unsupported_event",
    "claim_count",
    "unsupported_claim_count",
    "claims_json",
    "summary_reason",
    "status",
    "raw_response",
    "error_message",
    "prompt_version",
    "context_chars",
    "evaluated_at_utc",
]


def _strip_json_fence(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1]).strip()


def parse_claim_judgment(raw_response: str) -> Dict[str, object]:
    text = _strip_json_fence(raw_response)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")

    label = str(parsed.get("label", "")).strip().upper()
    if label not in {"SUPPORTED", "UNSUPPORTED"}:
        raise ValueError("label must be SUPPORTED or UNSUPPORTED")
    claims = parsed.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims must be a non-empty list")

    normalized_claims = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise ValueError(f"claim {index} must be an object")
        verdict = str(claim.get("verdict", "")).strip().upper()
        if verdict not in {"SUPPORTED", "UNSUPPORTED"}:
            raise ValueError(
                f"claim {index} verdict must be SUPPORTED or UNSUPPORTED"
            )
        normalized = {
            "claim": str(claim.get("claim", "")).strip(),
            "verdict": verdict.lower(),
            "evidence": str(claim.get("evidence", "")).strip(),
            "reason": str(claim.get("reason", "")).strip(),
        }
        if not all(normalized.values()):
            raise ValueError(
                f"claim {index} requires claim, verdict, evidence, and reason"
            )
        normalized_claims.append(normalized)

    unsupported_count = sum(
        claim["verdict"] == "unsupported"
        for claim in normalized_claims
    )
    summary_reason = str(parsed.get("summary_reason", "")).strip()
    if not summary_reason:
        raise ValueError("summary_reason must be non-empty")

    model_label = label.lower()
    canonical_label = (
        "unsupported" if unsupported_count else "supported"
    )
    return {
        "model_label": model_label,
        "label": canonical_label,
        "label_repaired": model_label != canonical_label,
        "claims": normalized_claims,
        "summary_reason": summary_reason,
        "unsupported_claim_count": unsupported_count,
    }


def make_prompt(question: str, answer: str, context: str) -> str:
    schema = {
        "label": "SUPPORTED or UNSUPPORTED",
        "claims": [
            {
                "claim": "one material claim from the answer",
                "verdict": "SUPPORTED or UNSUPPORTED",
                "evidence": "short context evidence or 'No supporting evidence'",
                "reason": "brief explanation",
            }
        ],
        "summary_reason": "brief overall rationale",
    }
    return (
        "Audit the generated answer against the retrieved context at the "
        "claim level.\n"
        "Use only the retrieved context. Do not use outside knowledge.\n"
        "Decompose the answer into every material factual claim.\n"
        "Label the answer SUPPORTED only when every material claim is "
        "directly supported or is a reasonable paraphrase.\n"
        "Label it UNSUPPORTED when any material claim is absent, contradicted, "
        "or incorrectly attributed.\n"
        "Pay special attention to these common scientific-QA errors:\n"
        "1. attributing Related Work methods or results to the current paper;\n"
        "2. treating a comparison baseline as a method actually used;\n"
        "3. reversing relations such as 'instead of', 'compared with', or "
        "'outperformed by';\n"
        "4. mixing accuracy, F1, relative improvement, and other metrics;\n"
        "5. assigning another team's result to the paper's own system.\n"
        "Do not penalize an answer merely for being incomplete.\n"
        "Return exactly one JSON object with this schema and no prose or "
        "Markdown fences:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Retrieved Context]\n{context}\n\n"
        f"[Generated Answer]\n{answer}\n"
    )


def load_completed(output_path: str) -> Set[Tuple[str, str]]:
    if not os.path.exists(output_path):
        return set()
    existing = pd.read_csv(output_path, keep_default_na=False)
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


def make_record(
    audit_id: str,
    provider: str,
    context_chars: int,
    status: str,
    raw_response: str = "",
    parsed: Optional[Dict[str, object]] = None,
    error_message: str = "",
) -> Dict[str, object]:
    parsed = parsed or {}
    label = str(parsed.get("label", ""))
    claims = parsed.get("claims", [])
    return {
        "audit_id": audit_id,
        "judge": provider,
        "judge_model": MODEL_NAMES[provider],
        "model_label": parsed.get("model_label", ""),
        "label": label,
        "label_repaired": parsed.get("label_repaired", ""),
        "unsupported_event": (
            int(label == "unsupported") if label else ""
        ),
        "claim_count": len(claims) if claims else "",
        "unsupported_claim_count": parsed.get(
            "unsupported_claim_count",
            "",
        ),
        "claims_json": (
            json.dumps(claims, ensure_ascii=False) if claims else ""
        ),
        "summary_reason": parsed.get("summary_reason", ""),
        "status": status,
        "raw_response": raw_response,
        "error_message": error_message,
        "prompt_version": "claim_faithfulness_v1",
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
                    max_tokens=900,
                    temperature=0.01,
                )
                try:
                    parsed = parse_claim_judgment(raw_response)
                except (json.JSONDecodeError, ValueError) as error:
                    parse_errors += 1
                    record = make_record(
                        row.audit_id,
                        provider,
                        len(context),
                        status="parse_error",
                        raw_response=raw_response,
                        error_message=f"{type(error).__name__}: {error}",
                    )
                else:
                    successful += 1
                    completed.add(key)
                    record = make_record(
                        row.audit_id,
                        provider,
                        len(context),
                        status="ok",
                        raw_response=raw_response,
                        parsed=parsed,
                    )
            except Exception as error:
                api_errors += 1
                record = make_record(
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
