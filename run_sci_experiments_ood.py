import os
import json
import time
import re
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

INDEX_CONFIGS = {
    "DETAIL": {"path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"path": "./chroma_db_md_1024", "collection": "rag_md_1024"}
}

BENCHMARK_FILE = "json/cross_dataset_bench.json"
OUTPUT_CSV = "experiment_results_ood.csv"
TOP_K = 3


# ==========================================
# 🧠 2. 核心算法 (增加 Coverage 统计)
# ==========================================
def compute_f1(a_gold, a_pred):
    gold_toks = set(re.findall(r'\w+', a_gold.lower()))
    pred_toks = set(re.findall(r'\w+', a_pred.lower()))
    common = gold_toks.intersection(pred_toks)
    if len(gold_toks) == 0 or len(pred_toks) == 0: return 0.0, 0.0, 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_faithfulness(query, answer, context):
    refusal_flags = ["information not found", "rejected"]
    if any(flag in answer.lower() for flag in refusal_flags):
        return 10.0, 0.0  # 拒答算作绝对忠实，无幻觉

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
    return float(score), 1.0 if score < 7 else 0.0


def check_coverage(answer):
    """判断系统是否做出了实质性回答 (非拒答)"""
    refusal_flags = ["information not found", "rejected"]
    if any(flag in answer.lower() for flag in refusal_flags):
        return 0  # 覆盖失败 (拒答)
    return 1  # 覆盖成功 (给出答案)


# ==========================================
# 🚀 3. RAG 流水线
# ==========================================
def load_indices():
    indices = {}
    for name, config in INDEX_CONFIGS.items():
        if os.path.exists(config['path']):
            db = chromadb.PersistentClient(path=config['path'])
            col = db.get_or_create_collection(name=config['collection'])
            indices[name] = VectorStoreIndex.from_vector_store(vector_store=ChromaVectorStore(chroma_collection=col))
    return indices


def run_pipeline(qa_item, indices, strategy):
    query = qa_item["question"]
    gold_answer = qa_item.get("gold_answer", "")
    start_time = time.time()

    intent = "CONCEPT" if strategy == "Vanilla_RAG" else "CONTEXT"  # OOD常识题多走大粒度
    index = indices.get(intent, indices["CONCEPT"])
    nodes = index.as_retriever(similarity_top_k=TOP_K).retrieve(query)

    context_str = "\n".join([n.text for n in nodes])
    avg_score = sum(n.score for n in nodes) / len(nodes) if nodes else 0
    calibrated_ret = min(10.0, max(0.0, (avg_score - 0.3) * 20))

    if strategy == "VeriMind_Full":
        QA_PROMPT = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information and NOT prior knowledge, answer the query concisely and directly. "
            "If the exact answer cannot be found, you MUST explicitly output exactly: 'Information not found.' "
            "Do NOT use parametric memory or guess.\nQuery: {query_str}\nAnswer: "
        )
    else:
        QA_PROMPT = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information, answer the query concisely and directly.\nQuery: {query_str}\nAnswer: "
        )

    answer = str(index.as_query_engine(similarity_top_k=TOP_K, text_qa_template=QA_PROMPT).query(query))
    faith_score, hallucination_rate = evaluate_faithfulness(query, answer, context_str)

    if strategy == "VeriMind_Full":
        trust_score = 0.4 * calibrated_ret + 0.6 * faith_score
        if trust_score < 5.0:
            answer = "REJECTED: Information not found."
            faith_score, hallucination_rate = 10.0, 0.0
    else:
        trust_score = 0.0

    coverage = check_coverage(answer)
    precision, recall, f1 = compute_f1(gold_answer, answer)

    return {
        "strategy": strategy,
        "category": qa_item["category"],
        "coverage": coverage,  # 新增指标：回答覆盖率
        "f1_score": round(f1, 4),
        "faith_score": faith_score,
        "hallucination_rate": hallucination_rate,
        "trust_score": round(trust_score, 2),
        "generated_answer": answer
    }


# ==========================================
# 📊 4. 主引擎 (支持断点续传)
# ==========================================
def run_experiments():
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    indices = load_indices()
    strategies = ["Vanilla_RAG", "VeriMind_Full"]

    print(f"🎯 开始执行 OOD 泛化实验，共计 {len(qa_data)} 题 × {len(strategies)} 个策略")

    completed_tasks = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            for _, row in df_existing.iterrows():
                completed_tasks.add((int(row['question_id']), row['strategy']))
            print(f"✅ 成功检测到存档！跳过 {len(completed_tasks)} 个已完成项...")
        except:
            pass

    for i, qa in enumerate(qa_data):
        if (i, strategies[0]) not in completed_tasks:
            print(f"\n[Progress: {i + 1}/{len(qa_data)}] Testing OOD Query: {qa['question'][:40]}...")

        for strategy in strategies:
            if (i, strategy) in completed_tasks: continue

            for attempt in range(3):
                try:
                    res = run_pipeline(qa, indices, strategy)
                    res["question_id"] = i
                    df = pd.DataFrame([res])
                    df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False,
                              encoding='utf-8-sig')
                    print(
                        f"  └─ {strategy.ljust(15)} | Cov: {res['coverage']} | Hal Rate: {res['hallucination_rate']:.1f}")
                    break
                except Exception as e:
                    print(f"  ⚠️ 网络波动，重试 ({attempt + 1}/3)...")
                    time.sleep(3)

    print(f"\n🎉 跨数据集实验完成！结果已保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiments()