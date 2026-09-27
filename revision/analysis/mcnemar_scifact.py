"""
mcnemar_scifact.py
对 SciFact 实验结果运行 McNemar 检验，正式输出 p 值。

输入：csv/scifact_experiment_results.csv
输出：终端打印检验结果，同时保存至 revision/analysis/mcnemar_results.txt
"""
import os
import sys
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH     = os.path.join(PROJECT_ROOT, "csv", "scifact_experiment_results.csv")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "mcnemar_results.txt")

# ================== 读取数据 ==================
df = pd.read_csv(CSV_PATH)

# hallucination_rate: 1.0 = 有幻觉，0.0 = 无幻觉
# 转为 bool，方便构建列联表
def get_hal_series(strategy):
    s = df[df["strategy"] == strategy].sort_values("item_id")["hallucination_rate"]
    return (s.values > 0.5).astype(int)  # 1=hallucinated, 0=faithful

verimind = get_hal_series("VeriMind_Full")
vanilla  = get_hal_series("Vanilla_RAG")
hyde     = get_hal_series("HyDE")

assert len(verimind) == len(vanilla) == len(hyde), "条目数不匹配，请检查 CSV"

n = len(verimind)

# ================== McNemar 检验 ==================
def run_mcnemar(a_hal, b_hal, name_a, name_b):
    """
    列联表：
              B=0(ok)  B=1(hal)
    A=0(ok)    n00      n01
    A=1(hal)   n10      n11

    McNemar 关注的是 n01 和 n10（仅一方出错的条目）
    """
    n00 = ((a_hal == 0) & (b_hal == 0)).sum()
    n01 = ((a_hal == 0) & (b_hal == 1)).sum()
    n10 = ((a_hal == 1) & (b_hal == 0)).sum()
    n11 = ((a_hal == 1) & (b_hal == 1)).sum()

    table = [[n00, n01], [n10, n11]]
    # exact=True: 使用精确二项检验（样本量适中时更严格）
    result = mcnemar(table, exact=False, correction=True)  # chi2 with continuity correction

    hal_a = a_hal.mean()
    hal_b = b_hal.mean()

    line = (
        f"\n{'='*55}\n"
        f"McNemar 检验：{name_a} vs {name_b}\n"
        f"{'='*55}\n"
        f"  样本数 n = {n}\n"
        f"  幻觉率  {name_a}: {hal_a:.4f} ({hal_a*100:.2f}%)\n"
        f"  幻觉率  {name_b}: {hal_b:.4f} ({hal_b*100:.2f}%)\n"
        f"  列联表  n00={n00}  n01={n01}  n10={n10}  n11={n11}\n"
        f"  (n01: {name_a}正确但{name_b}错误; n10: {name_a}错误但{name_b}正确)\n"
        f"  统计量 χ² = {result.statistic:.4f}\n"
        f"  p 值      = {result.pvalue:.2e}\n"
        f"  结论: {'极显著差异 ***' if result.pvalue < 0.001 else '显著差异 **' if result.pvalue < 0.01 else '显著差异 *' if result.pvalue < 0.05 else '差异不显著'}\n"
    )
    return line

lines = []
lines.append(f"SciFact 外部基准 McNemar 检验报告")
lines.append(f"数据集：SciFact（{n} 条/策略，共 {n*3} 行）")
lines.append(f"生成器：GLM-4.5-Air  评判器：DeepSeek-V3.2\n")

lines.append(run_mcnemar(verimind, vanilla,  "VeriMind_Full", "Vanilla_RAG"))
lines.append(run_mcnemar(verimind, hyde,     "VeriMind_Full", "HyDE"))
lines.append(run_mcnemar(vanilla,  hyde,     "Vanilla_RAG",   "HyDE"))

# ================== 输出 ==================
report = "\n".join(lines)
print(report)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n结果已保存至 {OUT_PATH}")