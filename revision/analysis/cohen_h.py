"""
cohen_h.py

Task 2.2:
Compute Cohen's h effect size of hallucination rate
for each baseline strategy vs VeriMind_Full.

Default input:
  revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv

Default outputs:
  revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv
  revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
from collections import defaultdict
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Cohen's h vs VeriMind_Full.")
    parser.add_argument(
        "--input_csv",
        default="revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv",
        help="Input CSV containing strategy and hallucination_rate columns.",
    )
    parser.add_argument(
        "--output_csv",
        default="revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_txt",
        default="revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt",
        help="Output TXT summary path.",
    )
    parser.add_argument(
        "--anchor_strategy",
        default="VeriMind_Full",
        help="Anchor strategy to compare against.",
    )
    return parser.parse_args()


def safe_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def read_csv_clean(path: str) -> List[Dict[str, str]]:
    with open(path, "rb") as f:
        data = f.read().replace(b"\x00", b"")
    text = data.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def cohen_h(p1: float, p2: float) -> float:
    # h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))
    # p in [0,1]
    p1 = min(1.0, max(0.0, p1))
    p2 = min(1.0, max(0.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def h_magnitude(abs_h: float) -> str:
    if abs_h < 0.2:
        return "negligible"
    if abs_h < 0.5:
        return "small"
    if abs_h < 0.8:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    rows = read_csv_clean(args.input_csv)
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_csv}")

    for col in ["strategy", "hallucination_rate"]:
        if col not in rows[0]:
            raise ValueError(f"Missing required column: {col}")

    by_strategy: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        s = (r.get("strategy") or "").strip()
        h = safe_float(r.get("hallucination_rate", ""))
        if not s or h is None:
            continue
        by_strategy[s].append(h)

    if args.anchor_strategy not in by_strategy or not by_strategy[args.anchor_strategy]:
        raise ValueError(f"Anchor strategy not found or empty: {args.anchor_strategy}")

    anchor_vals = by_strategy[args.anchor_strategy]
    p_anchor = sum(anchor_vals) / len(anchor_vals)
    n_anchor = len(anchor_vals)

    out_rows: List[Dict[str, str]] = []
    for strategy in sorted(by_strategy.keys()):
        if strategy == args.anchor_strategy:
            continue
        vals = by_strategy[strategy]
        if not vals:
            continue
        p_baseline = sum(vals) / len(vals)
        n_baseline = len(vals)

        h_signed = cohen_h(p_baseline, p_anchor)
        h_abs = abs(h_signed)
        delta = p_baseline - p_anchor
        direction = (
            "VeriMind lower hallucination"
            if delta > 0
            else "VeriMind higher hallucination"
            if delta < 0
            else "equal"
        )

        out_rows.append(
            {
                "anchor_strategy": args.anchor_strategy,
                "baseline_strategy": strategy,
                "n_anchor": str(n_anchor),
                "n_baseline": str(n_baseline),
                "hallucination_anchor": f"{p_anchor:.6f}",
                "hallucination_baseline": f"{p_baseline:.6f}",
                "delta_baseline_minus_anchor": f"{delta:.6f}",
                "cohen_h_signed": f"{h_signed:.6f}",
                "cohen_h_abs": f"{h_abs:.6f}",
                "magnitude": h_magnitude(h_abs),
                "direction": direction,
            }
        )

    out_rows.sort(key=lambda x: float(x["cohen_h_abs"]), reverse=True)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_strategy",
                "baseline_strategy",
                "n_anchor",
                "n_baseline",
                "hallucination_anchor",
                "hallucination_baseline",
                "delta_baseline_minus_anchor",
                "cohen_h_signed",
                "cohen_h_abs",
                "magnitude",
                "direction",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    lines: List[str] = []
    lines.append("Cohen's h vs VeriMind_Full (hallucination_rate)")
    lines.append(f"Input: {args.input_csv}")
    lines.append(f"Anchor strategy: {args.anchor_strategy}")
    lines.append(f"Anchor hallucination_rate mean: {p_anchor:.6f} (n={n_anchor})")
    lines.append("")
    lines.append("Top effects by |h|:")
    for row in out_rows:
        lines.append(
            f"- {row['baseline_strategy']}: h={row['cohen_h_signed']}, "
            f"|h|={row['cohen_h_abs']} ({row['magnitude']}), "
            f"delta={row['delta_baseline_minus_anchor']} ({row['direction']})"
        )

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"written_output_csv={args.output_csv}")
    print(f"written_output_txt={args.output_txt}")
    print(f"rows={len(out_rows)}")


if __name__ == "__main__":
    main()

