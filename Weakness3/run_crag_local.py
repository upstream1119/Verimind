import json
import pandas as pd
import dashscope
from dashscope import Generation
import chromadb
from tqdm import tqdm
import time
import os

# ================= 核心配置区 =================
dashscope.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # ⚠️ 填入你的阿里云 API Key
CHROMA_DB_DIR = "./your_chroma_db_folder"  # ⚠️ 填入你本地 ChromaDB 文件夹的路径
COLLECTION_NAME = "verimind_corpus"  # ⚠️ 填入你的 Collection 名称
INPUT_JSON = "verimind_bench_v3.json"  # 测试集文件名
# ==============================================

print("🔌 正在连接本地 ChromaDB...")
if not os.path.exists(CHROMA_DB_DIR):
    print(f"❌ 找不到 ChromaDB 文件夹 {CHROMA_DB_DIR}，请检查路径！")
    exit()

# 初始化 ChromaDB 客户端
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# 默认使用 ChromaDB 自带的 embedding，如果你当时指定了特殊的 embedding_function，请在这里传入
collection = client.get_collection(name=COLLECTION_NAME)

print(f"📥 加载测试集 {INPUT_JSON}...")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)


# --- CRAG 核心功能函数 ---

def crag_evaluate(query, context):
    """评估器：判断初始检索结果是否足以回答问题"""
    prompt = f"""Evaluate if the provided context contains sufficient and highly relevant information to answer the query.
Query: {query}
Context: {context}
Constraint: Output ONLY a single float number between 0.0 (completely irrelevant) and 1.0 (perfectly answers the query). Do not output any other text."""
    try:
        resp = Generation.call(model='qwen-plus', prompt=prompt, temperature=0.01)
        score_text = resp.output.text.strip()
        # 提取数字
        import re
        match = re.search(r"0\.\d+|1\.0|0|1", score_text)
        score = float(match.group()) if match else 0.5
        return score
    except Exception as e:
        return 0.5  # 报错则默认给个中间值，触发保守重写


def crag_rewrite(query, failed_context):
    """重写器：当检索失败时，将原问题重写为更易于检索的子问题"""
    prompt = f"""The original query failed to retrieve sufficient information.
Original Query: {query}
Failed Context (Brief): {failed_context[:500]}...

Task: Rewrite the original query into 2 distinct, simpler keyword-based search queries to help find the missing information.
Constraint: Output ONLY the 2 queries, separated by a newline. No numbering or explanations."""
    try:
        resp = Generation.call(model='qwen-plus', prompt=prompt, temperature=0.3)
        queries = [q.strip() for q in resp.output.text.split('\n') if q.strip()]
        return queries[:2] if queries else [query]
    except:
        return [query]


def generate_final(query, context):
    """最终生成器"""
    prompt = f"""You are a strict and highly accurate answering assistant. 
Task: Answer the question based ONLY on the provided context. 
Constraint: If the context does not contain sufficient information, you must output "Information not found."

Context: 
{context}

Question: {query}
Answer:"""
    resp = Generation.call(model='qwen-plus', prompt=prompt, temperature=0.01)
    return resp.output.text.strip()


# --- 主循环 ---
results = []
print("🚀 开始执行 CRAG-Local 动态评估与二次检索...")

for item in tqdm(bench_data):
    q_id = item['question_id']
    query = item['query']
    initial_context = item['context']

    # Step 1: 评估初始 Context
    eval_score = crag_evaluate(query, initial_context)

    # Step 2: 动态路由与二次检索
    if eval_score >= 0.7:
        final_context = initial_context
        path = "CORRECT (Direct Generation)"
    else:
        # 触发本地二次检索
        rewritten_queries = crag_rewrite(query, initial_context)
        secondary_docs = []
        for rq in rewritten_queries:
            # 去本地 ChromaDB 搜
            try:
                res = collection.query(query_texts=[rq], n_results=2)
                if res['documents']:
                    secondary_docs.extend(res['documents'][0])
            except Exception as e:
                pass  # 忽略单次检索错误

        # 将二次检索的知识追加到上下文中
        final_context = initial_context + "\n\n[Supplemental Context from CRAG Rewrite]:\n" + "\n".join(
            list(set(secondary_docs)))
        path = "INCORRECT -> REWRITTEN & RE-RETRIEVED"

    # Step 3: 最终生成
    try:
        answer = generate_final(query, final_context)
    except Exception as e:
        print(f"\n⚠️ 题目 ID {q_id} 最终生成报错: {e}")
        answer = "Error during generation."
        time.sleep(2)

    results.append({
        'question_id': q_id,
        'strategy': 'CRAG_Local',
        'generated_answer': answer,
        'crag_score': eval_score,
        'retrieval_path': path
    })

# 保存结果
df_crag = pd.DataFrame(results)
df_crag.to_csv("w3_crag_local_results.csv", index=False)
print("\n✅ CRAG-Local 跑通！结果已保存至 w3_crag_local_results.csv")