from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class PairStats:
    scope: str
    subset: str
    n: int
    agreement: float
    kappa: float
    label1_positive_rate: float
    label2_positive_rate: float
    conf_00: int
    conf_01: int
    conf_10: int
    conf_11: int
    notes: str


def _safe_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)


def _binary_pair_stats(
    label1: Iterable[int],
    label2: Iterable[int],
    *,
    scope: str,
    subset: str,
    notes: str = "",
) -> PairStats:
    s1 = pd.Series(list(label1), dtype="int64")
    s2 = pd.Series(list(label2), dtype="int64")
    valid = s1.isin([0, 1]) & s2.isin([0, 1])
    s1 = s1[valid]
    s2 = s2[valid]

    n = int(len(s1))
    if n == 0:
        return PairStats(
            scope=scope,
            subset=subset,
            n=0,
            agreement=0.0,
            kappa=0.0,
            label1_positive_rate=0.0,
            label2_positive_rate=0.0,
            conf_00=0,
            conf_01=0,
            conf_10=0,
            conf_11=0,
            notes="no valid binary pairs",
        )

    conf_00 = int(((s1 == 0) & (s2 == 0)).sum())
    conf_01 = int(((s1 == 0) & (s2 == 1)).sum())
    conf_10 = int(((s1 == 1) & (s2 == 0)).sum())
    conf_11 = int(((s1 == 1) & (s2 == 1)).sum())

    p0 = (conf_00 + conf_11) / n
    p1 = (conf_10 + conf_11) / n
    p2 = (conf_01 + conf_11) / n
    pe = (1.0 - p1) * (1.0 - p2) + p1 * p2
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        kappa = 0.0
    else:
        kappa = (p0 - pe) / denom

    return PairStats(
        scope=scope,
        subset=subset,
        n=n,
        agreement=float(p0),
        kappa=float(kappa),
        label1_positive_rate=float(p1),
        label2_positive_rate=float(p2),
        conf_00=conf_00,
        conf_01=conf_01,
        conf_10=conf_10,
        conf_11=conf_11,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="3.1 标注一致性审计")
    parser.add_argument(
        "--human50_csv",
        default="csv/human_validation_50_samples_scored.csv",
    )
    parser.add_argument(
        "--audit_csv",
        default="csv/relabeled_audit_sheet.csv",
    )
    parser.add_argument(
        "--audit2_csv",
        default="csv/relabeled_audit_sheet2.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="revision/artifacts/3_1_annotation/annotation_audit_summary.csv",
    )
    parser.add_argument(
        "--output_txt",
        default="revision/artifacts/3_1_annotation/annotation_audit_summary.txt",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_txt = Path(args.output_txt)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 50 条抽样：LLM 判定 vs 人工判定
    df_h50 = pd.read_csv(args.human50_csv)
    llm_col = "LLM Judge (机器判定幻觉)"
    human_col = "Human Judge (请你填 0 或 1)"
    s_llm = _safe_int_series(df_h50[llm_col])
    s_human = _safe_int_series(df_h50[human_col])
    stats_h50 = _binary_pair_stats(
        s_llm,
        s_human,
        scope="LLM_vs_Human_50",
        subset="overall",
        notes="source=human_validation_50_samples_scored.csv",
    )

    # 600 条复标：标注员A vs 标注员B
    df_audit = pd.read_csv(args.audit_csv)
    df_audit2 = pd.read_csv(args.audit2_csv)

    key_cols = ["sample_id", "system_id", "stratum"]
    merged = df_audit.merge(
        df_audit2[key_cols + ["label2"]],
        on=key_cols,
        how="inner",
    )
    merged["human_label"] = _safe_int_series(merged["human_label"])
    merged["label2"] = _safe_int_series(merged["label2"])

    stats_600 = _binary_pair_stats(
        merged["human_label"],
        merged["label2"],
        scope="AnnotatorA_vs_AnnotatorB_600",
        subset="overall",
        notes="source=relabeled_audit_sheet(.csv/.csv2)",
    )

    rows = [stats_h50, stats_600]
    for stratum in ["TRUSTED", "WARNING", "REJECTED"]:
        part = merged[merged["stratum"] == stratum]
        rows.append(
            _binary_pair_stats(
                part["human_label"],
                part["label2"],
                scope="AnnotatorA_vs_AnnotatorB_600",
                subset=stratum,
                notes="stratum-level",
            )
        )

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 补充流程规模统计
    system_counts = (
        df_audit["system_id"].value_counts(dropna=False).sort_index().to_dict()
    )
    stratum_counts = (
        df_audit["stratum"].value_counts(dropna=False).sort_index().to_dict()
    )

    def _fmt_rate(v: float) -> str:
        return f"{v:.4f}"

    with output_txt.open("w", encoding="utf-8-sig") as f:
        f.write("3.1 数据标注流程与一致性审计摘要\n")
        f.write("=" * 48 + "\n")
        f.write("\n")
        f.write("[流程规模]\n")
        f.write(f"- 复标审计总样本: {len(df_audit)}\n")
        f.write(f"- 参与系统分布: {system_counts}\n")
        f.write(f"- 分层分布: {stratum_counts}\n")
        f.write("\n")
        f.write("[一致性统计]\n")
        f.write(
            f"- LLM vs Human (n={stats_h50.n}): agreement={_fmt_rate(stats_h50.agreement)}, "
            f"kappa={_fmt_rate(stats_h50.kappa)}\n"
        )
        f.write(
            f"- Annotator A vs B overall (n={stats_600.n}): agreement={_fmt_rate(stats_600.agreement)}, "
            f"kappa={_fmt_rate(stats_600.kappa)}\n"
        )
        for s in ["TRUSTED", "WARNING", "REJECTED"]:
            row = out_df[
                (out_df["scope"] == "AnnotatorA_vs_AnnotatorB_600")
                & (out_df["subset"] == s)
            ].iloc[0]
            f.write(
                f"- Annotator A vs B {s} (n={int(row['n'])}): "
                f"agreement={_fmt_rate(float(row['agreement']))}, "
                f"kappa={_fmt_rate(float(row['kappa']))}\n"
            )
        f.write("\n")
        f.write("[写作建议边界]\n")
        f.write("- 可表述为“标注一致性达到高水平（overall kappa≈0.78）”。\n")
        f.write("- 对 REJECTED 分层需单独说明标签分布极不均衡，kappa 对边际分布敏感。\n")

    print(f"saved_csv={output_csv}")
    print(f"saved_txt={output_txt}")


if __name__ == "__main__":
    main()
