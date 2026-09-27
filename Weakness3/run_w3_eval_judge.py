import json
import pandas as pd
import dashscope
from dashscope import Generation
from tqdm import tqdm
import time
import re
import os

# ================= 配置区 =================
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if not dashscope.api_key:
    raise RuntimeError("DASHSCOPE_API_KEY is required")
JSON_FILE  = "../json/verimind_bench_v3.json"
COT_CSV    = "../csv/w3_cot_rag_results.csv"
CRAG_CSV   = "../csv/w3_crag_local_results.csv"
OUTPUT_DIR = "../csv"
SLEEP_SEC  = 0.3
MAX_RETRIES = 3
# Passed-Hallucination 阈值：与论文 Trust-Score Gating 保持一致
HALLUCINATION_THRESHOLD = 7.0
# ==========================================

print("📥 正在加载测试集与生成结果...")
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    raw = json.load(f)

# 用 enumerate idx 建索引，与 CSV 的 question_id 对齐
bench_data = {str(idx): item['context'] for idx, item in enumerate(raw)}

df_cot  = pd.read_csv(COT_CSV)
df_crag = pd.read_csv(CRAG_CSV)
print(f"✅ CoT+RAG: {len(df_cot)} 条 | CRAG-Local: {len(df_crag)} 条")


def judge_hallucination(context, answer, retries=MAX_RETRIES):
    """LLM-as-a-Judge 幻觉打分（0=严重幻觉, 10=完全忠实/合法拒绝）"""
    # 诚实拒绝直接满分，不占 API 调用
    if "information not found" in str(answer).lower():
        return 10.0

    messages = [
        {
            "role": "system",
            "content": (
                "You are a rigorous academic peer reviewer. "
                "Evaluate whether the [Generated Answer] is strictly and exclusively "
                "derived from the provided [Retrieved Context].\n"
                "Instructions:\n"
                "1. If the answer introduces external knowledge, dates, or entities "
                "NOT present in the context, deduct points severely.\n"
                "2. If the answer explicitly states 'Information not found', score it 10.\n"
                "3. Output ONLY a single integer from 0 to 10. No explanations."
            )
        },
        {
            "role": "user",
            "content": f"[Retrieved Context]: {context}\n\n[Generated Answer]: {answer}\n\nScore:"
        }
    ]

    for attempt in range(1, retries + 1):
        try:
            resp = Generation.call(
                model='qwen-plus',
                messages=messages,
                temperature=0.01,
                max_tokens=5,
                result_format='message'
            )
            score_text = resp.output.choices[0].message.content.strip()
            match = re.search(r"\b(10|[0-9])\b", score_text)
            return float(match.group()) if match else 5.0
        except Exception as e:
            print(f"\n⚠️ Judge 第{attempt}次失败: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    return 5.0  # 超时默认中间值，不影响统计


def evaluate_dataframe(df, name):
    print(f"\n🚀 开始评估 {name}（共 {len(df)} 条）...")
    faith_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_id    = str(int(row['question_id']))
        ans     = str(row['generated_answer'])
        context = bench_data.get(q_id, "")

        if not context:
            print(f"\n⚠️ question_id={q_id} 未找到对应 context，跳过")
            faith_scores.append(5.0)
            continue

        score = judge_hallucination(context, ans)
        faith_scores.append(score)
        time.sleep(SLEEP_SEC)

    df = df.copy()
    df['faith_score'] = faith_scores

    # ── Passed-Hallucination Rate（仅统计非拒绝答案）──
    answered = df[~df['generated_answer'].str.contains('Information not found', case=False, na=False)]
    rejected = df[df['generated_answer'].str.contains('Information not found', case=False, na=False)]

    answered = answered.copy()
    answered['is_hallucination'] = (answered['faith_score'] < HALLUCINATION_THRESHOLD).astype(int)

    passed_hal_rate = answered['is_hallucination'].mean() * 100 if len(answered) > 0 else 0.0
    rejection_rate  = len(rejected) / len(df) * 100

    print(f"✅ {name} 评估完成！")
    print(f"   Rejection Rate:          {rejection_rate:.2f}%  ({len(rejected)}/{len(df)})")
    print(f"   Passed-Hallucination Rate: {passed_hal_rate:.2f}%  (阈值 faith<{HALLUCINATION_THRESHOLD})")

    return df


df_cot_scored  = evaluate_dataframe(df_cot,  "CoT+RAG")
df_crag_scored = evaluate_dataframe(df_crag, "CRAG-Local")

out_cot  = os.path.join(OUTPUT_DIR, "w3_cot_rag_scored.csv")
out_crag = os.path.join(OUTPUT_DIR, "w3_crag_local_scored.csv")
df_cot_scored.to_csv(out_cot,   index=False, encoding='utf-8')
df_crag_scored.to_csv(out_crag, index=False, encoding='utf-8')

print(f"\n✅ 结果已保存至 {out_cot} 和 {out_crag}")
print("📋 将上面打印的 Rejection Rate 和 Passed-Hallucination Rate 填入 Table 7 即可")