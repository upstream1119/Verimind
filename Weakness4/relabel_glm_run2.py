"""
relabel_glm_run2.py
第二次独立标注（annotator 2）
输入: blind_audit_sheet_to_annotate.csv  （原始数据，不含任何标注结果）
输出: relabeled_audit_sheet2.csv              （只含 sample_id + label2）

跑完后把 relabeled_audit_sheet.csv（annotator1）和本文件输出合并计算 κ
"""
import csv
import os
import time
from collections import Counter
from zhipuai import ZhipuAI

API_KEY = os.environ.get("ZHIPUAI_API_KEY", "")
if not API_KEY:
    raise RuntimeError("ZHIPUAI_API_KEY is required")
INPUT   = "blind_audit_sheet_to_annotate.csv"   # 原始文件
OUTPUT  = "relabeled_audit_sheet2.csv"                # 本次输出

# ── 探测可用模型 ──────────────────────────────────────
client = ZhipuAI(api_key=API_KEY)

CANDIDATE_MODELS = [
    "glm-4.5-air", "glm-4-air", "glm-4-flash", "glm-4-plus", "glm-4",
]

def test_model(model_name):
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with digit 0 only."}],
            temperature=0.01,
            max_tokens=5,
        )
        content = resp.choices[0].message.content.strip()
        print(f"  [{model_name}] ✅ 响应: {content!r}")
        return len(content) > 0
    except Exception as e:
        print(f"  [{model_name}] ❌ {e}")
        return False

print("=== 探测可用模型 ===")
MODEL = None
for m in CANDIDATE_MODELS:
    if test_model(m):
        MODEL = m
        break
    time.sleep(1)

if MODEL is None:
    print("❌ 所有模型均不可用，请检查 API key")
    exit(1)

print(f"\n✅ 使用模型: {MODEL}\n")

# ── Prompt（与 run1 完全相同，保证公平对比）────────────
SYSTEM_PROMPT = """You are a hallucination auditor for academic RAG systems.

Given a [Question], [Context], and [Answer], output ONLY the digit 0 or 1:
- 0 = Answer is faithful (all claims are supported by the Context)
- 1 = Answer contains hallucination (introduces facts NOT in the Context)

Special rule: If the answer says "information not found" or refuses to answer, output 0.

You MUST output exactly one digit: 0 or 1. Nothing else."""

def annotate(question, context, answer):
    user_msg = f"[Question]: {question}\n\n[Context]: {context}\n\n[Answer]: {answer}"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.01,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            for ch in raw:
                if ch in ("0", "1"):
                    return int(ch)
            print(f"    ⚠️  意外输出: {raw!r}，重试...")
            time.sleep(1)
        except Exception as e:
            print(f"    ❌ 第{attempt+1}次异常: {e}")
            time.sleep(2)
    print("    ⚠️  三次失败，默认返回 0")
    return 0

# ── 读取原始数据 ──────────────────────────────────────
with open(INPUT, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# 全部 600 条都标注（REJECTED 的 answer 是固定拒绝语句，标注结果必然是 0，用于验证）
print(f"共 {len(rows)} 条，全部标注\n")

# ── 批量标注 ──────────────────────────────────────────
results = {}
for i, r in enumerate(rows, 1):
    sid = r["sample_id"]
    label = annotate(r["question"], r["context"], r["generated_answer"])
    results[sid] = label
    status = "🔴幻觉" if label == 1 else "✅忠实"
    print(f"[{i:3d}/600] sample={sid:4s} | {r['system_id']:15s} | {r['stratum']:8s} | {status}")
    time.sleep(0.5)

# ── 写出 annotator2 标注结果 ──────────────────────────
with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "stratum", "system_id", "label2"])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "sample_id": r["sample_id"],
            "stratum":   r["stratum"],
            "system_id": r["system_id"],
            "label2":    results[r["sample_id"]],
        })

print(f"\n✅ 已保存至 {OUTPUT}")
print(f"标注分布: {Counter(results.values())}")
print("\n下一步：把 relabeled_audit_sheet.csv 和 relabeled_audit_sheet2.csv 发给 Claude 计算 κ")
