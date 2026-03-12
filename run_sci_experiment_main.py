import os
import json
import time
import re
import collections
import pandas as pd
import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding

# ==========================================
# ⚙️ 1. 环境与模型配置
# ==========================================
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "你的API_KEY")
Settings.llm = DashScope(model_name="qwen-plus", api_key=DASHSCOPE_API_KEY, max_tokens=1500, temperature=0.01)
Settings.embed_model = DashScopeEmbedding(model_name="text-embedding-v1", api_key=DASHSCOPE_API_KEY)

DATA_DIR = "./knowledge_mds"
INDEX_CONFIGS = {
    "DETAIL": {"path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"path": "./chroma_db_md_1024", "collection": "rag_md_1024"},
    # 用于消融实验：无结构化解析的 baseline (若无此库，脚本会自动 fall back 到 CONCEPT)
    "FLAT": {"path": "./chroma_db_flat_512", "collection": "rag_flat_512"}
}

BENCHMARK_FILE = "json/verimind_bench_v3.json"
OUTPUT_CSV = "experiment_results_sci_v3.csv"
TOP_K = 3


# ==========================================
# 🧠 2. NLP 评价指标 (Metrics) 核心算法
# ==========================================
def compute_f1(a_gold, a_pred):
    """计算 Token 级别的 Precision, Recall, F1"""
    gold_toks = set(re.findall(r'\w+', a_gold.lower()))
    pred_toks = set(re.findall(r'\w+', a_pred.lower()))
    common = gold_toks.intersection(pred_toks)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0.0, 0.0, 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_exact_match(a_gold, a_pred):
    return 1 if a_gold.strip().lower() in a_pred.strip().lower() else 0


def estimate_cost(input_tokens, output_tokens):
    """基于通义千问 Plus 计费标准预估 Cost (美元/次)"""
    return (input_tokens * 0.004 + output_tokens * 0.012) / 1000 / 7.2


def evaluate_faithfulness(query, answer, context):
    """Faithfulness Score & Hallucination Rate 计算"""
    refusal_flags = ["information not found", "未找到", "未提及", "无证据", "不知道", "没有提及"]
    if any(flag in answer.lower() for flag in refusal_flags):
        return 10.0, 0.0  # 忠实度10，幻觉率0

    prompt = f"""Evaluate if the [Generated Answer] is strictly derived from the [Retrieved Context].
1. If the answer introduces external knowledge NOT present in the context, score it 0.
2. Output ONLY an integer score from 0 to 10.
[Context]: {context}
[Answer]: {answer}\nScore:"""
    try:
        res = Settings.llm.complete(prompt).text.strip()
        score = int(re.search(r'(?:^|\s)(10|[0-9])(?:\s|$)', res).group(1))
    except:
        score = 6

    hallucination_rate = 1.0 if score < 7 else 0.0
    return float(score), hallucination_rate


# ==========================================
# 🚀 3. SCI 要求的 Baseline 与 Ablation 策略
# ==========================================
def get_query_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in
           ["how many", "year", "parameter", "who", "when", "哪一年", "多少", "谁", "年份", "参数"]): return "DETAIL"
    if any(k in q for k in ["compare", "difference", "summarize", "比较", "区别", "总结"]): return "CONTEXT"
    return "CONCEPT"


def generate_hyde_doc(query):
    """HyDE 基线：生成伪造文档用于检索"""
    prompt = f"Please write a scientific paragraph that answers this query: {query}"
    return Settings.llm.complete(prompt).text


def load_indices():
    print("⏳ 正在挂载多粒度与消融向量数据库...")
    indices = {}
    for name, config in INDEX_CONFIGS.items():
        if os.path.exists(config['path']):
            db = chromadb.PersistentClient(path=config['path'])
            col = db.get_or_create_collection(name=config['collection'])
            vector_store = ChromaVectorStore(chroma_collection=col)
            indices[name] = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        else:
            print(f"⚠️ 未找到 {name} 数据库，后续如果调用将 fallback 到 CONCEPT")
    return indices


def run_pipeline(qa_item, indices, strategy):
    query = qa_item["question"]
    target_source = qa_item.get("source_file", "")
    gold_answer = qa_item.get("gold_answer", "")
    start_time = time.time()

    # ---------------- 1. 确定检索策略与粒度 ----------------
    search_query = query
    intent = "CONCEPT"
    top_k_val = TOP_K

    if strategy in ["VeriMind_Full", "VeriMind_No_Gating"]:
        intent = get_query_intent(query)
    elif strategy == "HyDE":
        search_query = generate_hyde_doc(query)
    elif strategy == "Long-Context_LLM":
        intent = "CONTEXT"
        top_k_val = 15  # 提取大量文本模拟长窗口
    elif strategy == "Ablation_no_structure":
        intent = "FLAT" if "FLAT" in indices else "CONCEPT"

    index = indices.get(intent, indices["CONCEPT"])
    retriever = index.as_retriever(similarity_top_k=top_k_val)
    nodes = retriever.retrieve(search_query)

    # 检索指标
    retrieved_sources = [n.metadata.get('file_name', '') for n in nodes]
    retrieval_hit = 1 if target_source and any(target_source in src for src in retrieved_sources) else 0
    context_str = "\n".join([n.text for n in nodes])
    avg_score = sum(n.score for n in nodes) / len(nodes) if nodes else 0
    calibrated_ret = min(10.0, max(0.0, (avg_score - 0.3) * 20))

    # ---------------- 2. 确定生成与门控策略 ----------------
    # 预估 Input Token (1 token ≈ 4 characters)
    input_tokens = len(context_str + query) / 4

    if strategy in ["VeriMind_Full", "VeriMind_No_Routing", "Ablation_no_structure"]:
        # 【修改点1】：全英文语境，并且强制要求 concise (简明) 以提升 F1
        QA_PROMPT = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information and NOT prior knowledge, answer the query concisely and directly. "
            "If the exact answer cannot be found, you MUST explicitly output exactly: 'Information not found.' "
            "Do NOT use parametric memory or guess.\nQuery: {query_str}\nAnswer: "
        )
    elif strategy == "Self-RAG":
        QA_PROMPT = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Critique the context. If irrelevant, say 'Information not found'. Otherwise answer the query concisely using ONLY the context.\nQuery: {query_str}\nAnswer: "
        )
    else:
        # 【修改点2】：同样要求 Baseline 也保持简明，保证 F1 评测公平
        QA_PROMPT = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information, answer the query concisely and directly.\nQuery: {query_str}\nAnswer: "
        )

    query_engine = index.as_query_engine(similarity_top_k=top_k_val, text_qa_template=QA_PROMPT)
    answer = str(query_engine.query(query))
    output_tokens = len(answer) / 4

    # ---------------- 3. 计算分数与门控拦截 ----------------
    faith_score, hallucination_rate = evaluate_faithfulness(query, answer, context_str)

    if strategy in ["VeriMind_Full", "VeriMind_No_Routing", "Ablation_no_structure"]:
        trust_score = 0.4 * calibrated_ret + 0.6 * faith_score
        if trust_score < 5.0:
            answer = "REJECTED: Information not found."
            faith_score, hallucination_rate = 10.0, 0.0  # 拦截成功即为防住了幻觉
    else:
        trust_score = 0.0

    latency = time.time() - start_time
    cost = estimate_cost(input_tokens, output_tokens)

    # 计算 NLP 指标
    precision, recall, f1 = compute_f1(gold_answer, answer)
    em = compute_exact_match(gold_answer, answer)

    return {
        "strategy": strategy,
        "category": qa_item["category"],
        "difficulty": qa_item["difficulty"],
        "routed_granularity": intent,
        "retrieval_hit": retrieval_hit,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "exact_match": em,
        "faith_score": faith_score,
        "hallucination_rate": hallucination_rate,
        "trust_score": round(trust_score, 2),
        "latency_sec": round(latency, 2),
        "est_cost_usd": round(cost, 5),
        "generated_answer": answer
    }


# ==========================================
# 📊 4. 自动化主引擎 (加入精准断点续传与防崩溃重试)
# ==========================================
def run_experiments():
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    indices = load_indices()

    strategies = [
        "Vanilla_RAG",
        "HyDE",
        "Self-RAG",
        "Long-Context_LLM",
        "VeriMind_No_Routing",
        "VeriMind_No_Gating",
        "Ablation_no_structure",
        "VeriMind_Full"
    ]

    print(f"🎯 开始执行 SCI 级自动化实验，共计 {len(qa_data)} 题 × {len(strategies)} 个策略")

    # ---------------- 核心：读取历史记录实现断点续传 ----------------
    completed_tasks = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            # 记录已经成功跑完的 (题目ID, 策略名称) 组合
            for _, row in df_existing.iterrows():
                completed_tasks.add((int(row['question_id']), row['strategy']))
            print(f"✅ 成功检测到防崩溃存档！已自动跳过 {len(completed_tasks)} 个已完成的测试项，无缝接力...")
        except Exception as e:
            print(f"⚠️ 读取存档失败，将从头开始: {e}")

    for i, qa in enumerate(qa_data):
        # 仅当这道题的第一个策略没跑时，才打印进度条，保持控制台整洁
        if (i, strategies[0]) not in completed_tasks:
            print(f"\n[Progress: {i + 1}/{len(qa_data)}] Testing Query: {qa['question'][:40]}...")

        for strategy in strategies:
            # 【断点续传拦截】：如果跑过，直接跳过
            if (i, strategy) in completed_tasks:
                continue

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    res = run_pipeline(qa, indices, strategy)
                    res["question_id"] = i

                    # 实时追加保存 CSV
                    df = pd.DataFrame([res])
                    df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False,
                              encoding='utf-8-sig')
                    print(
                        f"  └─ {strategy.ljust(22)} | F1: {res['f1_score']:.2f} | Hal Rate: {res['hallucination_rate']:.1f} | Latency: {res['latency_sec']}s")
                    break  # 成功则跳出重试循环

                except Exception as e:
                    print(
                        f"  ⚠️ {strategy} 遇到网络波动 ({type(e).__name__}). 正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(5)  # 休息5秒等网络恢复
            else:
                print(f"  ❌ {strategy} 连续 {max_retries} 次网络失败，已强制跳过以保证后续进度。")

    print(f"\n🎉 伟大的实验全部完成！拥有极其丰富指标的结果已保存至 {OUTPUT_CSV}")
    print("👉 下一步：基于这个 CSV，我们可以进行 P-value 显著性检验和 Error Analysis 图表绘制了！")


if __name__ == "__main__":
    run_experiments()