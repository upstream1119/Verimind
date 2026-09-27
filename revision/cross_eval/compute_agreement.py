"""
compute_agreement.py
计算三评判器一致性：
1) 两两 Cohen's Kappa
2) 三方 Fleiss' Kappa（二值标签：hallucination_rate >= 0.5）
"""

import argparse
import os
from datetime import datetime
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DEFAULT = os.path.join(
    PROJECT_ROOT, "revision", "artifacts", "1_2_baseline", "1_2_2_cross_judge", "cross_judge_results.csv"
)
OUTPUT_CSV_DEFAULT = os.path.join(
    PROJECT_ROOT, "revision", "artifacts", "1_2_baseline", "1_2_3_agreement", "cross_judge_agreement.csv"
)
OUTPUT_TXT_DEFAULT = os.path.join(
    PROJECT_ROOT, "revision", "artifacts", "1_2_baseline", "1_2_3_agreement", "cross_judge_agreement.txt"
)
JUDGES = ["qwen", "glm", "deepseek"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cross-judge agreement (Cohen/Fleiss).")
    parser.add_argument("--input_csv", default=INPUT_DEFAULT, help="Input cross-judge CSV.")
    parser.add_argument("--output_csv", default=OUTPUT_CSV_DEFAULT, help="Output agreement CSV.")
    parser.add_argument("--output_txt", default=OUTPUT_TXT_DEFAULT, help="Output agreement summary TXT.")
    return parser.parse_args()


def to_binary_label(value: object) -> int:
    """hallucination_rate >= 0.5 -> 1 else 0"""
    try:
        return 1 if float(value) >= 0.5 else 0
    except (TypeError, ValueError):
        return -1


def cohen_kappa_binary(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b):
        raise ValueError("cohen_kappa_binary 输入长度不一致。")
    n = len(a)
    if n == 0:
        return float("nan")

    p0 = sum(1 for i in range(n) if a[i] == b[i]) / n
    pa1 = sum(1 for x in a if x == 1) / n
    pa0 = 1.0 - pa1
    pb1 = sum(1 for x in b if x == 1) / n
    pb0 = 1.0 - pb1
    pe = pa1 * pb1 + pa0 * pb0

    if pe >= 1.0:
        return 1.0 if p0 >= 1.0 else 0.0
    return (p0 - pe) / (1.0 - pe)


def fleiss_kappa_binary(count_rows: Iterable[Tuple[int, int]]) -> float:
    rows: List[Tuple[int, int]] = list(count_rows)
    if not rows:
        return float("nan")

    n = rows[0][0] + rows[0][1]
    if n <= 1:
        return float("nan")

    for n0, n1 in rows:
        if n0 + n1 != n:
            raise ValueError("fleiss_kappa_binary 输入行的评判人数不一致。")

    N = len(rows)
    p_bar = sum((n0 * (n0 - 1) + n1 * (n1 - 1)) / (n * (n - 1)) for n0, n1 in rows) / N
    p0 = sum(n0 for n0, _ in rows) / (N * n)
    p1 = 1.0 - p0
    p_e = p0 * p0 + p1 * p1

    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"输入文件不存在: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required = ["question_id", "strategy", "judge", "hallucination_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入 CSV 缺少必要列: {missing}")

    df["judge"] = df["judge"].astype(str).str.strip().str.lower()
    df = df[df["judge"].isin(JUDGES)].copy()
    if df.empty:
        raise ValueError("输入 CSV 中没有可用的 judge 记录（qwen/glm/deepseek）。")

    df["hal_bin"] = df["hallucination_rate"].apply(to_binary_label)
    df = df[df["hal_bin"] >= 0].copy()
    if df.empty:
        raise ValueError("输入 CSV 的 hallucination_rate 无可用数值。")

    # 断点续跑可能产生重复键，保留最后一条
    df = df.drop_duplicates(subset=["question_id", "strategy", "judge"], keep="last")

    pivot = (
        df.pivot(index=["question_id", "strategy"], columns="judge", values="hal_bin")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for judge in JUDGES:
        if judge not in pivot.columns:
            pivot[judge] = pd.NA

    total_items = len(pivot)
    triplet_df = pivot.dropna(subset=JUDGES).copy()
    complete_items = len(triplet_df)
    missing_items = total_items - complete_items
    triplet_coverage = (complete_items / total_items) if total_items else 0.0

    result_rows = []
    result_rows.append(
        {
            "metric": "triplet_coverage",
            "judge_pair": "qwen-glm-deepseek",
            "kappa": "",
            "n_samples": complete_items,
            "total_items": total_items,
            "missing_items": missing_items,
            "coverage": round(triplet_coverage, 6),
            "label_rule": "hallucination_rate >= 0.5 -> 1",
            "input_csv": os.path.relpath(args.input_csv, PROJECT_ROOT),
        }
    )

    for j1, j2 in combinations(JUDGES, 2):
        if complete_items == 0:
            kappa = float("nan")
        else:
            a = triplet_df[j1].astype(int).tolist()
            b = triplet_df[j2].astype(int).tolist()
            kappa = cohen_kappa_binary(a, b)
        result_rows.append(
            {
                "metric": "cohen_kappa",
                "judge_pair": f"{j1}-{j2}",
                "kappa": round(kappa, 6) if pd.notna(kappa) else "",
                "n_samples": complete_items,
                "total_items": total_items,
                "missing_items": missing_items,
                "coverage": round(triplet_coverage, 6),
                "label_rule": "hallucination_rate >= 0.5 -> 1",
                "input_csv": os.path.relpath(args.input_csv, PROJECT_ROOT),
            }
        )

    if complete_items == 0:
        fleiss = float("nan")
    else:
        counts = []
        for _, row in triplet_df.iterrows():
            n1 = int(row["qwen"]) + int(row["glm"]) + int(row["deepseek"])
            counts.append((3 - n1, n1))
        fleiss = fleiss_kappa_binary(counts)
    result_rows.append(
        {
            "metric": "fleiss_kappa",
            "judge_pair": "qwen-glm-deepseek",
            "kappa": round(fleiss, 6) if pd.notna(fleiss) else "",
            "n_samples": complete_items,
            "total_items": total_items,
            "missing_items": missing_items,
            "coverage": round(triplet_coverage, 6),
            "label_rule": "hallucination_rate >= 0.5 -> 1",
            "input_csv": os.path.relpath(args.input_csv, PROJECT_ROOT),
        }
    )

    out_df = pd.DataFrame(result_rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"Cross-Judge Agreement Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        f"Input: {args.input_csv}",
        f"Triplet coverage: {complete_items}/{total_items} = {triplet_coverage:.4f}",
        f"Missing triplets: {missing_items}",
    ]
    for _, row in out_df[out_df["metric"] == "cohen_kappa"].iterrows():
        summary_lines.append(f"Cohen's kappa [{row['judge_pair']}]: {row['kappa']}")
    summary_lines.append(
        f"Fleiss' kappa [qwen-glm-deepseek]: "
        f"{out_df[out_df['metric'] == 'fleiss_kappa']['kappa'].iloc[0]}"
    )

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"完成：{args.output_csv}")
    print(f"完成：{args.output_txt}")


if __name__ == "__main__":
    main()
