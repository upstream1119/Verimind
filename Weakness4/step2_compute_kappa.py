"""
step2_compute_kappa.py
用法：python step2_compute_kappa.py
输入：blind_audit_sheet_to_annotate.csv（human_label 列已全部填写完毕）
输出：总体及分系统、分层 κ 值，可直接填入论文
"""
import csv
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

INPUT = "../csv/blind_audit_sheet_to_annotate.csv"

with open(INPUT, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# ── 过滤未填写行 ─────────────────────────────────────
valid = [r for r in rows if r["human_label"].strip() in ("0", "1")]
skipped = len(rows) - len(valid)
if skipped:
    print(f"⚠️  跳过 {skipped} 条未填写行")
print(f"有效样本数：{len(valid)}")

# ── 从实验结果推断 auto_label ────────────────────────
# hallucination_rate=1 → 有幻觉(1)，否则 → 0
# 此处从 experiment_results_sci_v3.csv 加载对应的 hallucination_rate
import os
EXP_FILE = "../csv/experiment_results_sci_v3.csv"

SYSTEM_TO_STRATEGY = {
    "VeriMind_Full": "VeriMind_Full",
    "Vanilla_RAG":   "Vanilla_RAG",
    "Self_RAG":      "Self-RAG",
    "HyDE":          "HyDE",
}

auto_label_idx = {}  # (strategy, question_id) -> hallucination_rate
if os.path.exists(EXP_FILE):
    with open(EXP_FILE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["strategy"], r["question_id"])
            auto_label_idx[key] = int(float(r.get("hallucination_rate", 0)))
    print(f"✅ 已加载实验结果：{EXP_FILE}")
else:
    print(f"⚠️  未找到 {EXP_FILE}，auto_label 将全部设为 0（仅凭 human_label 分析）")

human_labels = []
judge_labels = []
meta = []

for r in valid:
    h = int(r["human_label"])
    sys = r["system_id"]
    qid = r["question_id"]
    strat = SYSTEM_TO_STRATEGY.get(sys, sys)
    j = auto_label_idx.get((strat, qid), 0)
    human_labels.append(h)
    judge_labels.append(j)
    meta.append({"system": sys, "stratum": r["stratum"]})

# ── 总体结果 ─────────────────────────────────────────
agree = sum(j == h for j, h in zip(judge_labels, human_labels)) / len(valid)
try:
    kappa = cohen_kappa_score(judge_labels, human_labels)
except Exception as e:
    kappa = float("nan")
    print(f"⚠️  总体 κ 计算失败（可能方差为0）：{e}")

print()
print("=" * 50)
print(f"  总体 Agreement : {agree*100:.2f}%  (N={len(valid)})")
print(f"  总体 Cohen's κ : {kappa:.4f}")
print("=" * 50)

# ── 分系统结果 ────────────────────────────────────────
by_sys = defaultdict(lambda: {"j": [], "h": []})
for j, h, m in zip(judge_labels, human_labels, meta):
    by_sys[m["system"]]["j"].append(j)
    by_sys[m["system"]]["h"].append(h)

print("\n分系统结果（可填入 Table 9）：")
print(f"{'System':<20} {'Agreement':>10} {'κ':>8} {'N':>5}")
print("-" * 48)
for sys in ["VeriMind_Full", "Vanilla_RAG", "Self_RAG", "HyDE"]:
    d = by_sys.get(sys, {"j": [], "h": []})
    if not d["j"]:
        continue
    a = sum(x == y for x, y in zip(d["j"], d["h"])) / len(d["j"])
    try:
        k = cohen_kappa_score(d["j"], d["h"])
    except Exception:
        k = float("nan")
    print(f"{sys:<20} {a*100:>9.1f}% {k:>8.4f} {len(d['j']):>5}")

# ── 分层结果 ──────────────────────────────────────────
by_stratum = defaultdict(lambda: {"j": [], "h": []})
for j, h, m in zip(judge_labels, human_labels, meta):
    by_stratum[m["stratum"]]["j"].append(j)
    by_stratum[m["stratum"]]["h"].append(h)

print("\n分层结果：")
print(f"{'Stratum':<12} {'Agreement':>10} {'κ':>8} {'N':>5}")
print("-" * 40)
for stratum in ["TRUSTED", "WARNING", "REJECTED"]:
    d = by_stratum.get(stratum, {"j": [], "h": []})
    if not d["j"]:
        continue
    a = sum(x == y for x, y in zip(d["j"], d["h"])) / len(d["j"])
    try:
        k = cohen_kappa_score(d["j"], d["h"])
    except Exception:
        k = float("nan")
    print(f"{stratum:<12} {a*100:>9.1f}% {k:>8.4f} {len(d['j']):>5}")

print()
print("📋 论文填写提示：")
print(f"   Section 5.2 / Table 8：N={len(valid)}, Agreement={agree*100:.2f}%, κ={kappa:.4f}")
print(f"   VeriMind_Full 子集结果填入 Table 9 第一行")