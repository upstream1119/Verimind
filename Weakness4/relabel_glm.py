"""
relabel_glm_v2.py
修复版：先测试模型可用性，再批量标注
"""
import csv
import time
from collections import Counter
from zhipuai import ZhipuAI

API_KEY = "884369fcc8744d59b490c8d3989f7153.4Pr3tJoXMET2woB9"
INPUT   = "blind_audit_sheet_to_annotate.csv"
OUTPUT  = "relabeled_audit_sheet.csv"

client = ZhipuAI(api_key=API_KEY)

# ── Step 1: 探测可用模型 ──────────────────────────────
CANDIDATE_MODELS = [
    "glm-4.5-air",
    "glm-4-air",
    "glm-4-flash",
    "glm-4",
    "glm-4-plus",
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
        print(f"  [{model_name}] ❌ 失败: {e}")
        return False

print("=== 探测可用模型 ===")
MODEL = None
for m in CANDIDATE_MODELS:
    if test_model(m):
        MODEL = m
        break
    time.sleep(1)

if MODEL is None:
    print("\n❌ 所有模型均不可用，请检查 API key 或网络")
    exit(1)

print(f"\n✅ 使用模型: {MODEL}\n")

# ── Step 2: 标注 Prompt ───────────────────────────────
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
            # 提取第一个 0 或 1
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

# ── Step 3: 读取数据 ──────────────────────────────────
with open(INPUT, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

need = [r for r in rows if r["stratum"] in ("TRUSTED", "WARNING")]
skip = [r for r in rows if r["stratum"] == "REJECTED"]
print(f"需要标注: {len(need)} 条 | 跳过(REJECTED): {len(skip)} 条\n")

# ── Step 4: 批量标注 ──────────────────────────────────
results = {}
for i, r in enumerate(need, 1):
    sid = r["sample_id"]
    label = annotate(r["question"], r["context"], r["generated_answer"])
    results[sid] = label
    status = "🔴幻觉" if label == 1 else "✅忠实"
    print(f"[{i:3d}/{len(need)}] sample={sid:4s} | {r['system_id']:15s} | {r['stratum']:8s} | {status}")
    time.sleep(0.5)

# ── Step 5: 写出结果 ──────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        new_r = dict(r)
        if r["stratum"] == "REJECTED":
            new_r["human_label"] = "0"
        else:
            new_r["human_label"] = str(results.get(r["sample_id"], r["human_label"]))
        writer.writerow(new_r)

print(f"\n✅ 已保存至 {OUTPUT}")

# ── Step 6: 统计 ──────────────────────────────────────
new_labels = [results[r["sample_id"]] for r in need]
print(f"\n新标注分布: {Counter(new_labels)}")
print(f"TRUSTED+WARNING 幻觉率: {sum(new_labels)/len(new_labels)*100:.1f}%")

by_sys = {}
for r in need:
    sys = r["system_id"]
    label = results[r["sample_id"]]
    by_sys.setdefault(sys, []).append(label)

print("\n分系统幻觉率:")
for sys, labels in by_sys.items():
    print(f"  {sys:20s}: {sum(labels)/len(labels)*100:.1f}% ({sum(labels)}/{len(labels)})")