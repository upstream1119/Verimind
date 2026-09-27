"""Normalize Gate V2 judgments without overwriting immutable raw responses."""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from revision.qasper.gate_v2 import (
    ATTRIBUTIONS,
    EVIDENCE_OWNERS,
    parse_gate_judgment,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_14_qasper_gate_v2_dev",
    "full_v1_1",
)
DEFAULT_RAW = os.path.join(
    DEFAULT_INPUT_DIR,
    "gate_v2_raw_judgments.csv",
)
DEFAULT_RETRY = os.path.join(
    DEFAULT_INPUT_DIR,
    "gate_v2_retry_judgments.csv",
)
DEFAULT_OUTPUT = os.path.join(
    DEFAULT_INPUT_DIR,
    "gate_v2_normalized_judgments.csv",
)
DEFAULT_REPORT = os.path.join(
    DEFAULT_INPUT_DIR,
    "gate_v2_normalization_report.csv",
)
EXPECTED_ROWS = 447
BASE_COLUMNS = [
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
OUTPUT_COLUMNS = BASE_COLUMNS + [
    "normalization_source",
    "repair_codes",
    "original_status",
    "original_error_message",
]


def normalize_evidence_owner(raw_owner: str) -> Tuple[str, List[str]]:
    owner_aliases = {
        "RELATED_WORK": "PRIOR_WORK",
        "BACKGROUND": "PRIOR_WORK",
        "PRIOR_STUDY": "PRIOR_WORK",
        "CURRENT": "CURRENT_PAPER",
    }
    original_owner = str(raw_owner).strip().upper()
    normalized_owner = owner_aliases.get(
        original_owner,
        original_owner,
    )
    if normalized_owner in EVIDENCE_OWNERS:
        codes = (
            ["evidence_owner_alias"]
            if normalized_owner != original_owner
            else []
        )
        return normalized_owner, codes

    parts = [
        owner_aliases.get(part.strip().upper(), part.strip().upper())
        for part in re.split(r"[;,/]", original_owner)
        if part.strip()
    ]
    if parts and all(part in EVIDENCE_OWNERS for part in parts):
        return "MIXED", ["evidence_owner_multi_value"]
    raise ValueError(f"Unrecoverable evidence_owner: {original_owner}")


def strip_json_fence(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_raw_response(
    raw_response: str,
) -> Tuple[Dict[str, object], List[str]]:
    repair_codes: List[str] = []
    text = strip_json_fence(raw_response)
    repaired_text = re.sub(
        r'(?<!\\)\\([^"\\/bfnrtu])',
        r"\1",
        text,
    )
    if repaired_text != text:
        repair_codes.append("invalid_json_escape")
    try:
        payload = json.loads(repaired_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Response is not complete recoverable JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON must be an object")

    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Every claim must be an object")
        normalized_owner, owner_repair_codes = normalize_evidence_owner(
            str(claim.get("evidence_owner", ""))
        )
        repair_codes.extend(owner_repair_codes)
        claim["evidence_owner"] = normalized_owner

        attribution = str(
            claim.get("attribution", "")
        ).strip().upper()
        if attribution not in ATTRIBUTIONS:
            attribution_by_owner = {
                "CURRENT_PAPER": "CURRENT_PAPER",
                "PRIOR_WORK": "RELATED_WORK",
                "MIXED": "UNRESOLVED",
                "UNRESOLVED": "UNRESOLVED",
            }
            claim["attribution"] = attribution_by_owner[
                normalized_owner
            ]
            repair_codes.append("attribution_from_evidence_owner")
    parsed = parse_gate_judgment(
        json.dumps(payload, ensure_ascii=False)
    )
    return parsed, list(dict.fromkeys(repair_codes))


def parsed_record(
    source_row: pd.Series,
    parsed: Dict[str, object],
    raw_response: str,
    normalization_source: str,
    repair_codes: List[str],
    original_status: str,
    original_error_message: str,
) -> Dict[str, object]:
    claims = parsed["claims"]
    label = str(parsed["label"])
    record = {
        column: source_row.get(column, "")
        for column in BASE_COLUMNS
    }
    record.update(
        {
            "model_label": parsed["model_label"],
            "label": label,
            "label_repaired": parsed["label_repaired"],
            "unsupported_event": int(label == "unsupported"),
            "uncertain_event": int(label == "uncertain"),
            "claim_count": len(claims),
            "unsupported_claim_count": parsed[
                "unsupported_claim_count"
            ],
            "uncertain_claim_count": parsed[
                "uncertain_claim_count"
            ],
            "claims_json": json.dumps(claims, ensure_ascii=False),
            "summary_reason": parsed["summary_reason"],
            "status": "ok",
            "raw_response": raw_response,
            "error_message": "",
            "normalization_source": normalization_source,
            "repair_codes": ";".join(repair_codes),
            "original_status": original_status,
            "original_error_message": original_error_message,
        }
    )
    return record


def build_normalized_judgments(
    raw: pd.DataFrame,
    retry: pd.DataFrame,
    expected_rows: int = EXPECTED_ROWS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = set(BASE_COLUMNS)
    missing_raw = required - set(raw.columns)
    if missing_raw:
        raise ValueError(
            f"Raw judgments are missing columns: {sorted(missing_raw)}"
        )
    if len(raw) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} raw rows, found {len(raw)}"
        )
    if raw.duplicated(subset=["item_id", "judge"]).any():
        raise ValueError("Raw judgments contain duplicate item-judge rows")

    if len(retry):
        missing_retry = required - set(retry.columns)
        if missing_retry:
            raise ValueError(
                "Retry judgments are missing columns: "
                f"{sorted(missing_retry)}"
            )
        if retry.duplicated(subset=["item_id", "judge"]).any():
            raise ValueError(
                "Retry judgments contain duplicate item-judge rows"
            )
    retry_index = {
        (int(row.item_id), str(row.judge)): pd.Series(row._asdict())
        for row in retry.itertuples(index=False)
    }

    normalized_rows: List[Dict[str, object]] = []
    report_rows: List[Dict[str, object]] = []
    unresolved: List[str] = []
    for raw_row in raw.to_dict("records"):
        row = pd.Series(raw_row)
        key = (int(row["item_id"]), str(row["judge"]))
        original_status = str(row["status"])
        original_error = str(row["error_message"])
        if original_status == "ok":
            record = {
                column: row.get(column, "")
                for column in BASE_COLUMNS
            }
            record.update(
                {
                    "normalization_source": "original_ok",
                    "repair_codes": "",
                    "original_status": original_status,
                    "original_error_message": original_error,
                }
            )
            normalized_rows.append(record)
            continue

        try:
            parsed, repair_codes = normalize_raw_response(
                str(row["raw_response"])
            )
        except ValueError:
            retry_row = retry_index.get(key)
            if retry_row is None:
                unresolved.append(f"{key[0]}:{key[1]}")
                continue
            if str(retry_row["status"]) == "ok":
                parsed = parse_gate_judgment(
                    str(retry_row["raw_response"])
                )
                resolution = "retry"
                codes = ["retry_truncated_response"]
            else:
                try:
                    parsed, retry_repair_codes = normalize_raw_response(
                        str(retry_row["raw_response"])
                    )
                except ValueError:
                    unresolved.append(f"{key[0]}:{key[1]}")
                    continue
                resolution = "retry_offline_repair"
                codes = [
                    "retry_truncated_response",
                    *retry_repair_codes,
                ]
            record = parsed_record(
                retry_row,
                parsed,
                str(retry_row["raw_response"]),
                resolution,
                codes,
                original_status,
                original_error,
            )
        else:
            record = parsed_record(
                row,
                parsed,
                str(row["raw_response"]),
                "offline_repair",
                repair_codes,
                original_status,
                original_error,
            )
            resolution = "offline_repair"
            codes = repair_codes
        normalized_rows.append(record)
        report_rows.append(
            {
                "item_id": key[0],
                "judge": key[1],
                "original_status": original_status,
                "original_error_message": original_error,
                "resolution": resolution,
                "repair_codes": ";".join(codes),
                "normalized_label": parsed["label"],
            }
        )

    if unresolved:
        raise ValueError(
            "Unresolved judgments require successful retry rows: "
            f"{unresolved}"
        )
    normalized = pd.DataFrame(
        normalized_rows,
        columns=OUTPUT_COLUMNS,
    )
    if len(normalized) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} normalized rows, "
            f"found {len(normalized)}"
        )
    if normalized.duplicated(subset=["item_id", "judge"]).any():
        raise ValueError(
            "Normalized judgments contain duplicate item-judge rows"
        )
    if set(normalized["status"]) != {"ok"}:
        raise ValueError("Every normalized judgment must have status=ok")
    report = pd.DataFrame(report_rows)
    return normalized, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--retry", default=DEFAULT_RETRY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw, keep_default_na=False)
    retry = (
        pd.read_csv(args.retry, keep_default_na=False)
        if os.path.exists(args.retry)
        else pd.DataFrame(columns=BASE_COLUMNS)
    )
    normalized, report = build_normalized_judgments(
        raw,
        retry,
        expected_rows=args.expected_rows,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    normalized.to_csv(args.output, index=False, encoding="utf-8-sig")
    report.to_csv(args.report, index=False, encoding="utf-8-sig")
    print(
        f"Normalized judgments: {len(normalized)}; "
        f"sources={normalized['normalization_source'].value_counts().to_dict()}"
    )
    print(f"Normalization report rows: {len(report)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
