"""
threshold_sensitivity.py

Task 2.3:
Threshold sensitivity analysis for trust-gating in VeriMind.

Method:
- Anchor on observed `VeriMind_Full` results at current base threshold (default 5.0).
- Use paired rows of `VeriMind_Full` and `VeriMind_No_Gating` only for deltas:
  - if t < base: rows with trust in [t, base) are released from gating
    (replace metrics by `VeriMind_No_Gating` counterparts)
  - if t > base: rows with trust in [base, t) are newly gated
    (replace metrics by rejection template)
  - if t == base: identical to observed `VeriMind_Full`

Default input:
  revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv

Default outputs:
  revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv
  revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt
  revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sensitivity for VeriMind trust gating.")
    parser.add_argument(
        "--input_csv",
        default="revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output_csv",
        default="revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv",
        help="Output CSV (curve data).",
    )
    parser.add_argument(
        "--output_txt",
        default="revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt",
        help="Output TXT summary.",
    )
    parser.add_argument(
        "--output_svg",
        default="revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg",
        help="Output SVG curve chart.",
    )
    parser.add_argument("--threshold_min", type=float, default=0.0)
    parser.add_argument("--threshold_max", type=float, default=10.0)
    parser.add_argument("--threshold_step", type=float, default=0.5)
    parser.add_argument(
        "--base_threshold",
        type=float,
        default=5.0,
        help="Current deployed threshold used to identify observed gated rows.",
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


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def frange(start: float, stop: float, step: float) -> List[float]:
    out: List[float] = []
    x = start
    # numeric stability
    eps = step / 10.0
    while x <= stop + eps:
        out.append(round(x, 6))
        x += step
    return out


def build_reject_template(vm_full_rows: List[Dict[str, str]], base_threshold: float) -> Dict[str, float]:
    """
    Build rejection template from currently observed gated rows (trust < base_threshold).
    If missing, fallback to deterministic conservative defaults.
    """
    gated = [r for r in vm_full_rows if safe_float(r.get("trust_score", "")) is not None and float(r["trust_score"]) < base_threshold]

    if not gated:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "exact_match": 0.0,
            "faith_score": 10.0,
            "hallucination_rate": 0.0,
            "bert_score_f1": 0.0,
        }

    def m(col: str, default: float) -> float:
        vals = [safe_float(r.get(col, "")) for r in gated]
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else default

    return {
        "precision": m("precision", 0.0),
        "recall": m("recall", 0.0),
        "f1_score": m("f1_score", 0.0),
        "exact_match": m("exact_match", 0.0),
        "faith_score": m("faith_score", 10.0),
        "hallucination_rate": m("hallucination_rate", 0.0),
        "bert_score_f1": m("bert_score_f1", 0.0),
    }


def write_svg(rows: List[Dict[str, float]], out_svg: str) -> None:
    """
    Lightweight SVG line chart for:
    - hallucination_rate_mean (red)
    - f1_score_mean (blue)
    - accept_rate (green)
    """
    width, height = 980, 620
    ml, mr, mt, mb = 80, 40, 50, 80
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    xs = [r["threshold"] for r in rows]
    ys_h = [r["hallucination_rate_mean"] for r in rows]
    ys_f1 = [r["f1_score_mean"] for r in rows]
    ys_acc = [r["accept_rate"] for r in rows]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, 1.0

    def xp(x: float) -> float:
        if abs(x_max - x_min) < 1e-12:
            return ml + plot_w / 2
        return ml + (x - x_min) / (x_max - x_min) * plot_w

    def yp(y: float) -> float:
        return mt + (y_max - y) / (y_max - y_min) * plot_h

    def polyline(vals: List[float]) -> str:
        pts = " ".join(f"{xp(x):.2f},{yp(y):.2f}" for x, y in zip(xs, vals))
        return pts

    # ticks
    x_ticks = [x_min + i * (x_max - x_min) / 10 for i in range(11)]
    y_ticks = [i / 10 for i in range(11)]

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="Arial">Threshold Sensitivity (VeriMind)</text>')

    # axes
    lines.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="black" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="black" stroke-width="1.5"/>')

    for xt in x_ticks:
        x = xp(xt)
        lines.append(f'<line x1="{x:.2f}" y1="{mt+plot_h}" x2="{x:.2f}" y2="{mt+plot_h+6}" stroke="#666" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{mt+plot_h+24}" text-anchor="middle" font-size="11" font-family="Arial">{xt:.1f}</text>')
    for yt in y_ticks:
        y = yp(yt)
        lines.append(f'<line x1="{ml-6}" y1="{y:.2f}" x2="{ml}" y2="{y:.2f}" stroke="#666" stroke-width="1"/>')
        lines.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml+plot_w}" y2="{y:.2f}" stroke="#EEE" stroke-width="1"/>')
        lines.append(f'<text x="{ml-10}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{yt:.1f}</text>')

    lines.append(f'<text x="{ml+plot_w/2:.2f}" y="{height-24}" text-anchor="middle" font-size="14" font-family="Arial">Trust Threshold</text>')
    lines.append(f'<text x="24" y="{mt+plot_h/2:.2f}" transform="rotate(-90 24 {mt+plot_h/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial">Metric Value (0-1)</text>')

    # curves
    lines.append(f'<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{polyline(ys_h)}"/>')
    lines.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{polyline(ys_f1)}"/>')
    lines.append(f'<polyline fill="none" stroke="#2ca02c" stroke-width="2.5" points="{polyline(ys_acc)}"/>')

    # legend
    lx, ly = ml + 20, mt + 16
    legend = [
        ("#d62728", "hallucination_rate_mean"),
        ("#1f77b4", "f1_score_mean"),
        ("#2ca02c", "accept_rate"),
    ]
    for i, (c, t) in enumerate(legend):
        y = ly + i * 22
        lines.append(f'<line x1="{lx}" y1="{y}" x2="{lx+24}" y2="{y}" stroke="{c}" stroke-width="3"/>')
        lines.append(f'<text x="{lx+30}" y="{y+4}" font-size="12" font-family="Arial">{t}</text>')

    lines.append("</svg>")
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    rows = read_csv_clean(args.input_csv)
    if not rows:
        raise ValueError(f"No rows from input: {args.input_csv}")

    required = [
        "strategy",
        "question_id",
        "trust_score",
        "hallucination_rate",
        "faith_score",
        "precision",
        "recall",
        "f1_score",
        "exact_match",
        "bert_score_f1",
        "retrieval_hit",
    ]
    miss = [c for c in required if c not in rows[0]]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    vm_full = [r for r in rows if (r.get("strategy") or "").strip() == "VeriMind_Full"]
    vm_ng = [r for r in rows if (r.get("strategy") or "").strip() == "VeriMind_No_Gating"]
    if not vm_full or not vm_ng:
        raise ValueError("Need both VeriMind_Full and VeriMind_No_Gating rows.")

    idx_ng: Dict[str, Dict[str, str]] = {r["question_id"]: r for r in vm_ng}
    paired: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    for r in vm_full:
        ng = idx_ng.get(r["question_id"])
        if ng is None:
            continue
        paired.append((r, ng))
    if not paired:
        raise ValueError("No paired rows by question_id between VeriMind_Full and VeriMind_No_Gating.")

    reject_tpl = build_reject_template(vm_full, args.base_threshold)

    thresholds = frange(args.threshold_min, args.threshold_max, args.threshold_step)
    out_rows: List[Dict[str, float]] = []

    for t in thresholds:
        accepted = 0
        metrics = {
            "hallucination_rate": [],
            "faith_score": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "exact_match": [],
            "bert_score_f1": [],
            "retrieval_hit": [],
        }

        for full_row, ng_row in paired:
            trust = safe_float(full_row.get("trust_score", ""))
            if trust is None:
                continue

            # threshold-defined accept/gate counters
            if trust >= t:
                accepted += 1

            # baseline = observed VeriMind_Full at base threshold
            src = full_row

            # determine whether this row changes state due to threshold shift
            if t < args.base_threshold:
                # release previously gated rows
                if trust >= t and trust < args.base_threshold:
                    src = ng_row
            elif t > args.base_threshold:
                # newly gate previously accepted rows
                if trust >= args.base_threshold and trust < t:
                    src = None  # use rejection template

            if src is None:
                # newly gated row: use rejection template for response-dependent metrics
                metrics["hallucination_rate"].append(reject_tpl["hallucination_rate"])
                metrics["faith_score"].append(reject_tpl["faith_score"])
                metrics["precision"].append(reject_tpl["precision"])
                metrics["recall"].append(reject_tpl["recall"])
                metrics["f1_score"].append(reject_tpl["f1_score"])
                metrics["exact_match"].append(reject_tpl["exact_match"])
                metrics["bert_score_f1"].append(reject_tpl["bert_score_f1"])
                metrics["retrieval_hit"].append(safe_float(full_row.get("retrieval_hit", "")) or 0.0)
            else:
                metrics["hallucination_rate"].append(safe_float(src.get("hallucination_rate", "")) or 0.0)
                metrics["faith_score"].append(safe_float(src.get("faith_score", "")) or 0.0)
                metrics["precision"].append(safe_float(src.get("precision", "")) or 0.0)
                metrics["recall"].append(safe_float(src.get("recall", "")) or 0.0)
                metrics["f1_score"].append(safe_float(src.get("f1_score", "")) or 0.0)
                metrics["exact_match"].append(safe_float(src.get("exact_match", "")) or 0.0)
                metrics["bert_score_f1"].append(safe_float(src.get("bert_score_f1", "")) or 0.0)
                metrics["retrieval_hit"].append(safe_float(src.get("retrieval_hit", "")) or 0.0)

        n = len(metrics["hallucination_rate"])
        if n == 0:
            continue
        accept_rate = accepted / n
        gate_rate = 1.0 - accept_rate

        out_rows.append(
            {
                "threshold": t,
                "n_samples": float(n),
                "accept_count": float(accepted),
                "gate_count": float(n - accepted),
                "accept_rate": accept_rate,
                "gate_rate": gate_rate,
                "hallucination_rate_mean": mean(metrics["hallucination_rate"]),
                "faith_score_mean": mean(metrics["faith_score"]),
                "precision_mean": mean(metrics["precision"]),
                "recall_mean": mean(metrics["recall"]),
                "f1_score_mean": mean(metrics["f1_score"]),
                "exact_match_mean": mean(metrics["exact_match"]),
                "bert_score_f1_mean": mean(metrics["bert_score_f1"]),
                "coverage_mean": mean(metrics["retrieval_hit"]),
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "n_samples",
                "accept_count",
                "gate_count",
                "accept_rate",
                "gate_rate",
                "hallucination_rate_mean",
                "faith_score_mean",
                "precision_mean",
                "recall_mean",
                "f1_score_mean",
                "exact_match_mean",
                "bert_score_f1_mean",
                "coverage_mean",
            ],
        )
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})

    # key points
    def find_t(x: float) -> Dict[str, float] | None:
        for r in out_rows:
            if abs(r["threshold"] - x) < 1e-9:
                return r
        return None

    best_f1 = max(out_rows, key=lambda x: x["f1_score_mean"])
    best_hal = min(out_rows, key=lambda x: x["hallucination_rate_mean"])
    current = find_t(args.base_threshold)

    txt_lines: List[str] = []
    txt_lines.append("Threshold Sensitivity (VeriMind trust gating)")
    txt_lines.append(f"Input: {args.input_csv}")
    txt_lines.append(f"Paired samples: {len(paired)} (VeriMind_Full x VeriMind_No_Gating)")
    txt_lines.append(f"Threshold range: {args.threshold_min} .. {args.threshold_max} (step={args.threshold_step})")
    txt_lines.append("")
    txt_lines.append(f"Best F1 threshold: t={best_f1['threshold']:.2f}, f1={best_f1['f1_score_mean']:.6f}, hal={best_f1['hallucination_rate_mean']:.6f}, accept_rate={best_f1['accept_rate']:.6f}")
    txt_lines.append(f"Best Hallucination threshold: t={best_hal['threshold']:.2f}, hal={best_hal['hallucination_rate_mean']:.6f}, f1={best_hal['f1_score_mean']:.6f}, accept_rate={best_hal['accept_rate']:.6f}")
    if current is not None:
        txt_lines.append(
            f"Current threshold t={args.base_threshold:.2f}: "
            f"hal={current['hallucination_rate_mean']:.6f}, f1={current['f1_score_mean']:.6f}, "
            f"faith={current['faith_score_mean']:.6f}, accept_rate={current['accept_rate']:.6f}"
        )

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    write_svg(out_rows, args.output_svg)

    print(f"written_output_csv={args.output_csv}")
    print(f"written_output_txt={args.output_txt}")
    print(f"written_output_svg={args.output_svg}")
    print(f"rows={len(out_rows)}")


if __name__ == "__main__":
    main()
