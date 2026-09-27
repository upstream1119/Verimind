import json
import pandas as pd
import dashscope
from dashscope import Generation
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

# ================= 配置区 =================
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
INPUT_JSON  = "../json/verimind_bench_v3.json"
OUTPUT_CSV  = "../csv/w3_cot_rag_results.csv"
SLEEP_SEC   = 0.5   # 每题间隔，避免触发 QPS 限制
MAX_RETRIES = 3     # 单题最大重试次数
# ==========================================

print(f"📥 正在加载测试集 {INPUT_JSON} ...")
if not os.path.exists(INPUT_JSON):
    print(f"❌ 找不到测试集文件 {INPUT_JSON}，请检查路径！")
    exit()

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)

# 字段验证
sample = bench_data[0]
for field in ['question', 'context', 'gold_answer']:
    if field not in sample:
        print(f"❌ bench_data 中缺少字段 '{field}'，当前字段：{list(sample.keys())}")
        exit()

print(f"✅ 加载完成，共 {len(bench_data)} 条")
print("🚀 开始执行 CoT + RAG 基线评估...")

results = []

for idx, item in enumerate(tqdm(bench_data)):
    q_id    = idx                   # JSON 无 question_id，用列表索引代替
    query   = item['question']
    context = item['context']
    gold    = item['gold_answer']
    category   = item.get('category', '')
    difficulty = item.get('difficulty', '')

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict and highly accurate answering assistant. "
                "Answer the question based ONLY on the provided context. "
                "If the context does not contain sufficient information, "
                "you must output exactly: Information not found."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Please think step by step about how the context relates to "
                "the question, then provide your final answer."
            )
        }
    ]

    answer = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = Generation.call(
                model='qwen-plus',
                messages=messages,
                temperature=0.01,
                max_tokens=512,
                result_format='message'
            )
            answer = response.output.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"\n⚠️ 题目 ID {q_id} 第{attempt}次报错: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)
            else:
                answer = "Error during generation."

    results.append({
        'question_id':      q_id,
        'strategy':         'CoT_RAG',
        'category':         category,
        'difficulty':       difficulty,
        'generated_answer': answer,
        'gold_answer':      gold,
    })

    time.sleep(SLEEP_SEC)

df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\n✅ CoT+RAG 完成！共 {len(results)} 条，结果已保存至 {OUTPUT_CSV}")

# 快速检查
errors = df[df['generated_answer'] == 'Error during generation.']
if not errors.empty:
    print(f"⚠️  有 {len(errors)} 条生成失败，question_id: {errors['question_id'].tolist()}")
else:
    print("🎉 全部生成成功，无报错条目")
