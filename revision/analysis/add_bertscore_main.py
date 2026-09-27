"""
add_bertscore_main.py

Task 2.1:
1) Add BERTScore F1 to the main experiment table.
2) Produce a Table-7-ready strategy summary.

Default input:
  revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv

Default outputs:
  revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv
  revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import types
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add BERTScore F1 to main experiment CSV.")
    parser.add_argument(
        "--input_csv",
        default="revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output_csv",
        default="revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv",
        help="Output CSV path with bert_score_f1 column.",
    )
    parser.add_argument(
        "--summary_csv",
        default="revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv",
        help="Output strategy summary CSV path.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="BERTScore batch size.")
    parser.add_argument("--limit", type=int, default=0, help="0 means full input.")
    parser.add_argument(
        "--resume",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Resume from existing output CSV if possible.",
    )
    parser.add_argument(
        "--allow_online",
        action="store_true",
        help="Allow HuggingFace online fetch if local cache is missing.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for BERTScore, e.g., cpu / cuda:0. Auto if omitted.",
    )
    return parser.parse_args()


def safe_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def read_csv_clean(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "rb") as f:
        data = f.read().replace(b"\x00", b"")
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"CSV header missing: {path}")
    return rows, list(reader.fieldnames)


def make_key(row: Dict[str, str]) -> str:
    return f"{row.get('question_id', '').strip()}::{row.get('strategy', '').strip()}"


def ensure_required_columns(fieldnames: List[str]) -> None:
    required = {"question_id", "strategy", "generated_answer", "judge_context"}
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")


def build_existing_index(output_csv: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(output_csv):
        return {}
    rows, _ = read_csv_clean(output_csv)
    by_key: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = make_key(row)
        if not key:
            continue
        score = row.get("bert_score_f1", "").strip()
        if score:
            by_key[key] = row
    return by_key


def write_header_if_needed(path: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def summarize_for_table7(output_csv: str, summary_csv: str) -> None:
    rows, _ = read_csv_clean(output_csv)
    by_strategy: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_strategy[r.get("strategy", "")].append(r)

    out_rows: List[Dict[str, str]] = []
    for strategy in sorted(by_strategy.keys()):
        g = by_strategy[strategy]

        def mean_col(col: str) -> float:
            vals = [safe_float(x.get(col, "")) for x in g]
            vals = [v for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else -1.0

        out_rows.append(
            {
                "strategy": strategy,
                "n_rows": str(len(g)),
                "hallucination_rate_mean": f"{mean_col('hallucination_rate'):.6f}",
                "faith_score_mean": f"{mean_col('faith_score'):.6f}",
                "f1_score_mean": f"{mean_col('f1_score'):.6f}",
                "bert_score_f1_mean": f"{mean_col('bert_score_f1'):.6f}",
                "exact_match_mean": f"{mean_col('exact_match'):.6f}",
                "coverage_mean": f"{mean_col('retrieval_hit'):.6f}",
            }
        )

    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "n_rows",
                "hallucination_rate_mean",
                "faith_score_mean",
                "f1_score_mean",
                "bert_score_f1_mean",
                "exact_match_mean",
                "coverage_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def inject_bertscore_safe_stubs() -> None:
    """
    bert_score imports plotting stacks at module import time.
    In some Windows envs PIL/matplotlib DLL loading may fail even though
    scoring itself does not need plotting. We inject minimal stubs so
    scoring can proceed.
    """
    if "matplotlib" not in sys.modules:
        mpl_mod = types.ModuleType("matplotlib")
        pyplot_mod = types.ModuleType("matplotlib.pyplot")

        def _noop(*args, **kwargs):  # noqa: ANN002, ANN003
            return None

        for name in [
            "subplots",
            "xlabel",
            "ylabel",
            "title",
            "setp",
            "savefig",
            "show",
        ]:
            setattr(pyplot_mod, name, _noop)

        mpl_mod.pyplot = pyplot_mod
        sys.modules["matplotlib"] = mpl_mod
        sys.modules["matplotlib.pyplot"] = pyplot_mod

    if "mpl_toolkits" not in sys.modules:
        mpt_mod = types.ModuleType("mpl_toolkits")
        axes_mod = types.ModuleType("mpl_toolkits.axes_grid1")

        def _dummy_make_axes_locatable(*args, **kwargs):  # noqa: ANN002, ANN003
            return None

        axes_mod.make_axes_locatable = _dummy_make_axes_locatable
        sys.modules["mpl_toolkits"] = mpt_mod
        sys.modules["mpl_toolkits.axes_grid1"] = axes_mod


def clear_module_cache(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def import_bertscorer():
    """
    Try normal import first, then retry with safe stubs for matplotlib stack.
    """
    try:
        from bert_score import BERTScorer  # type: ignore
        return BERTScorer
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        recoverable = ("_imaging" in msg) or ("matplotlib" in msg) or ("PIL" in msg)
        if not recoverable:
            raise
        print(f"[warn] bert_score import failed ({msg}), retrying with plotting stubs...")
        inject_bertscore_safe_stubs()
        clear_module_cache("bert_score")
        from bert_score import BERTScorer  # type: ignore
        return BERTScorer


def main() -> None:
    args = parse_args()
    resume = args.resume.lower() == "true"

    if not args.allow_online:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Import after environment setup.
    BERTScorer = import_bertscorer()

    input_rows, input_fields = read_csv_clean(args.input_csv)
    ensure_required_columns(input_fields)
    if args.limit > 0:
        input_rows = input_rows[: args.limit]

    output_fields = list(input_fields)
    if "bert_score_f1" not in output_fields:
        output_fields.append("bert_score_f1")

    existing = build_existing_index(args.output_csv) if resume else {}
    pending = [r for r in input_rows if make_key(r) not in existing]

    print(f"input_rows={len(input_rows)}")
    print(f"existing_rows={len(existing)}")
    print(f"pending_rows={len(pending)}")
    print(f"resume={resume}")
    print(f"batch_size={args.batch_size}")
    print(f"allow_online={args.allow_online}")

    if not resume and os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    write_header_if_needed(args.output_csv, output_fields)

    scorer = BERTScorer(lang="en", batch_size=args.batch_size, device=args.device)

    processed = 0
    total = len(pending)
    for i in range(0, total, args.batch_size):
        batch = pending[i : i + args.batch_size]
        cands: List[str] = []
        refs: List[str] = []
        score_targets: List[int] = []

        out_batch: List[Dict[str, str]] = []
        scores = [-1.0] * len(batch)

        for j, row in enumerate(batch):
            cand = (row.get("generated_answer") or "").strip()
            ref = (row.get("judge_context") or "").strip()
            if not cand or not ref:
                continue
            cands.append(cand)
            refs.append(ref)
            score_targets.append(j)

        if cands:
            _, _, f1 = scorer.score(cands, refs)
            f1_vals = [float(x) for x in f1.tolist()]
            for target_idx, score in zip(score_targets, f1_vals):
                scores[target_idx] = score

        for row, score in zip(batch, scores):
            new_row = dict(row)
            new_row["bert_score_f1"] = f"{score:.4f}" if score >= 0 else "-1.0"
            out_batch.append(new_row)

        append_rows(args.output_csv, output_fields, out_batch)
        processed += len(batch)
        print(f"progress={processed}/{total}")

    # If resume and all rows already existed, still ensure summary is regenerated.
    summarize_for_table7(args.output_csv, args.summary_csv)
    print(f"written_output={args.output_csv}")
    print(f"written_summary={args.summary_csv}")


if __name__ == "__main__":
    main()
