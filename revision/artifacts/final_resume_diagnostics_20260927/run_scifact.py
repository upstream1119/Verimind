"""Trace frozen SciFact labels back to raw evaluator JSON, without relabeling."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "revision/artifacts/supplement_20260927_full"
sys.path.insert(0, str(ROOT / "revision/experiments"))
from supplement_20260927 import digest, is_refusal, parse_evaluation

# These patterns flag explicit *reason/boolean contradictions* for review. They never change a label.
EXPLICIT_INSUFFICIENT = re.compile(
    r"(?:evidence|context|information)\s+(?:is|was|remains)\s+insufficient"
    r"|insufficient\s+(?:retrieved\s+)?evidence"
    r"|not\s+(?:enough|sufficient)\s+(?:evidence|information|to\s+(?:verify|support|establish))"
    r"|no\s+(?:direct\s+)?evidence\s+(?:to|for|that)"
    r"|cannot\s+(?:be\s+)?(?:verify|support|establish)\b",
    re.I,
)
PARTIAL = re.compile(r"\bpartial(?:ly)?\b|\bambig(?:uous|uity)\b|\bnot directly\b", re.I)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    items = {x["uid"]: x for x in json.loads((SOURCE / "items.json").read_text(encoding="utf-8"))}
    retrievals = {x["uid"]: x for x in map(json.loads, (SOURCE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    raw_answers = list(map(json.loads, (SOURCE / "answers.jsonl").read_text(encoding="utf-8").splitlines()))
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    candidates = [x for x in raw_answers if x["strategy"] == "multi" and
                  x["dataset"] in ("scifact", "scifact_nei") and not is_refusal(x["answer"])]
    assert len(candidates) == 450 and len({x["uid"] for x in candidates}) == 450
    prior_review = {x["sample_id"]: x for x in json.loads((ROOT / "revision/artifacts/judge_zero_audit_20260927/provisional_review.json").read_text(encoding="utf-8"))}
    records = []
    for answer in sorted(candidates, key=lambda x: x["uid"]):
        uid = answer["uid"]
        item = items.get(uid)
        retrieval = retrievals.get(uid)
        context = "\n\n".join(n["text"] for n in retrieval["results"]["multi"]["nodes"][:3]) if retrieval else ""
        alignment = bool(item and retrieval and item["dataset"] == answer["dataset"] == retrieval["dataset"] and
                         digest(context) == answer["context_sha256"])
        parsed = parse_evaluation(answer["evaluation_raw"])
        raw_equal = parsed == answer["evaluation"]
        judge_raw = str(answer["gate_raw"])
        judge_parse_ok = bool(re.fullmatch(r"\s*(10|[0-9])\s*", judge_raw) and
                              int(judge_raw.strip()) == answer["faith_score"])
        supported = answer["evaluation"]["supported"]
        reason = str(answer["evaluation"].get("reason", ""))
        explicit_insufficient = bool(EXPLICIT_INSUFFICIENT.search(reason))
        flags = []
        if not alignment:
            flags.append("ID_ALIGNMENT_ERROR")
        if not raw_equal or not judge_parse_ok:
            flags.append("PARSER_ERROR")
        if supported and explicit_insufficient:
            flags.append("BOOLEAN_REASON_CONFLICT")
        if supported and not answer["evaluation"]["evidence_sufficient"]:
            flags.append("BOOLEAN_FIELD_CONFLICT")
        if (answer["faith_score"] >= 5) != supported:
            flags.append("JUDGE_EVALUATOR_CONFLICT")
        if PARTIAL.search(reason):
            flags.append("PARTIAL_SUPPORT_AMBIGUITY_CANDIDATE")
        if not flags:
            flags.append("NO_CONFLICT")
        reviewed = prior_review.get(uid)
        records.append({
            "sample_id": uid, "dataset": answer["dataset"], "claim": item["question"],
            "answer": answer["answer"], "retrieved_context": context,
            "judge_score": answer["faith_score"] / 10, "judge_raw_output": judge_raw,
            "judge_parsed_label": "SUPPORTED_PROXY" if answer["faith_score"] >= 5 else "UNSUPPORTED_PROXY",
            "judge_label_rule": "score >= 5; proxy for comparison, not an emitted categorical label",
            "evaluator_raw_output": answer["evaluation_raw"], "evaluator_reason": reason,
            "evaluator_evidence_sufficient_boolean": answer["evaluation"]["evidence_sufficient"],
            "evaluator_supported_boolean": supported, "derived_supported_boolean": parsed["supported"],
            "corrected_supported_label": "", "label_resolution": "unresolved" if flags != ["NO_CONFLICT"] else "original_preserved",
            "source_file": str(SOURCE / "answers.jsonl"), "source_experiment_version": manifest["protocol"],
            "source_split": "|".join(item.get("source_splits", [])), "timestamp_utc": answer["timestamp_utc"],
            "sample_id_alignment_ok": alignment, "context_sha256_ok": alignment,
            "evaluator_raw_parse_exact_match": raw_equal, "judge_raw_parse_exact_match": judge_parse_ok,
            "explicit_insufficient_reason_signal": explicit_insufficient,
            "conflict_type": "|".join(flags),
            "prior_30_case_provisional_category": reviewed["category"] if reviewed else "",
        })
    assert len(records) == 450
    candidate_by_id = {x["uid"]: x for x in candidates}
    write_csv(OUT / "scifact_label_trace.csv", records)
    conflicts = [r for r in records if r["conflict_type"] != "NO_CONFLICT"]
    write_csv(OUT / "scifact_conflicts.csv", conflicts)
    counts = Counter(flag for r in records for flag in r["conflict_type"].split("|"))
    summary = {
        "candidate_n": len(records), "conflict_union_n": len(conflicts), "flag_counts_overlapping": dict(counts),
        "judge_score_zero_n": sum(r["judge_score"] == 0 for r in records),
        "judge_score_zero_supported_true_n": sum(r["judge_score"] == 0 and r["evaluator_supported_boolean"] for r in records),
        "trust_030_040_judge_zero_n": sum(.3 <= candidate_by_id[r["sample_id"]]["trust_score"] / 10 < .4 and r["judge_score"] == 0 for r in records),
        "trust_030_040_judge_zero_supported_true_n": sum(.3 <= candidate_by_id[r["sample_id"]]["trust_score"] / 10 < .4 and r["judge_score"] == 0 and r["evaluator_supported_boolean"] for r in records),
        "supported_true_n": sum(r["evaluator_supported_boolean"] for r in records),
        "supported_true_evidence_sufficient_false_n": sum(r["evaluator_supported_boolean"] and not r["evaluator_evidence_sufficient_boolean"] for r in records),
        "raw_evaluator_parser_mismatches": sum(not r["evaluator_raw_parse_exact_match"] for r in records),
        "raw_judge_parser_mismatches": sum(not r["judge_raw_parse_exact_match"] for r in records),
        "id_or_context_alignment_errors": sum(not r["sample_id_alignment_ok"] for r in records),
        "uid_unique": len({r["sample_id"] for r in records}) == len(records),
        "source_protocol": manifest["protocol"],
        "source_models": {k: manifest[k] for k in ("generator", "gate_judge", "independent_evaluator")},
        "source_dates": dict(Counter(r["timestamp_utc"][:10] for r in records)),
        "source_splits": dict(Counter(r["source_split"] for r in records)),
        "confirmed_parser_bug": False, "confirmed_id_alignment_error": False,
        "confirmed_version_mismatch": False, "corrected_labels_generated": False,
        "warning": "Reason pattern and partial-support flags are review signals, not corrected ground truth. Judge label is a threshold proxy."
    }
    (OUT / "scifact_label_consistency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_n": len(records), "conflict_union_n": len(conflicts),
                      "flags": dict(counts), "parser_error_n": summary["raw_evaluator_parser_mismatches"]}))


if __name__ == "__main__":
    main()
