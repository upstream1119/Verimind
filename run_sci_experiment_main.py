import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import chromadb
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.chroma import ChromaVectorStore

# ==========================================
# ⚙️ 1) 环境与模型配置
# ==========================================
load_dotenv(".env")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
Settings.llm = DashScope(
    model_name="qwen-plus",
    api_key=DASHSCOPE_API_KEY,
    max_tokens=1500,
    temperature=0.01,
)
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v1",
    api_key=DASHSCOPE_API_KEY,
)

INDEX_CONFIGS = {
    "DETAIL": {"path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"path": "./chroma_db_md_1024", "collection": "rag_md_1024"},
    "FLAT": {"path": "./chroma_db_flat_512", "collection": "rag_flat_512"},
}

BENCHMARK_FILE = "json/verimind_bench_v3.json"
OUTPUT_CSV_DEFAULT = "revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv"
TOP_K = 3

STRATEGIES_10 = [
    "Vanilla_RAG",
    "HyDE",
    "Self-RAG",
    "Long-Context_LLM",
    "VeriMind_No_Routing",
    "VeriMind_No_Gating",
    "Ablation_no_structure",
    "VeriMind_Full",
    "CRAG_Local",
    "CoT_RAG",
]


# ==========================================
# 🧠 2) 指标函数
# ==========================================
def compute_f1(a_gold: str, a_pred: str) -> Tuple[float, float, float]:
    gold_toks = set(re.findall(r"\w+", (a_gold or "").lower()))
    pred_toks = set(re.findall(r"\w+", (a_pred or "").lower()))
    if not gold_toks or not pred_toks:
        return 0.0, 0.0, 0.0
    common = gold_toks.intersection(pred_toks)
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_exact_match(a_gold: str, a_pred: str) -> int:
    return 1 if (a_gold or "").strip().lower() in (a_pred or "").strip().lower() else 0


def estimate_cost(input_tokens: float, output_tokens: float) -> float:
    return (input_tokens * 0.004 + output_tokens * 0.012) / 1000 / 7.2


def evaluate_faithfulness(query: str, answer: str, context: str) -> Tuple[float, float]:
    refusal_flags = ["information not found", "未找到", "未提及", "无证据", "不知道", "没有提及", "rejected"]
    if any(flag in (answer or "").lower() for flag in refusal_flags):
        return 10.0, 0.0

    prompt = (
        "Evaluate if the [Generated Answer] is strictly derived from the [Retrieved Context].\n"
        "1. If the answer introduces external knowledge NOT present in the context, score it 0.\n"
        "2. Output ONLY an integer score from 0 to 10.\n"
        f"[Context]: {context}\n"
        f"[Answer]: {answer}\nScore:"
    )
    try:
        res = Settings.llm.complete(prompt).text.strip()
        score = int(re.search(r"(?:^|\s)(10|[0-9])(?:\s|$)", res).group(1))
    except Exception:
        score = 6
    return float(score), (1.0 if score < 7 else 0.0)


# ==========================================
# 🚀 3) 策略逻辑
# ==========================================
def get_query_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["how many", "year", "parameter", "who", "when", "哪一年", "多少", "谁", "年份", "参数"]):
        return "DETAIL"
    if any(k in q for k in ["compare", "difference", "summarize", "比较", "区别", "总结"]):
        return "CONTEXT"
    return "CONCEPT"


def generate_hyde_doc(query: str) -> str:
    prompt = f"Please write a scientific paragraph that answers this query: {query}"
    return Settings.llm.complete(prompt).text


def qwen_complete_with_retry(prompt: str, retries: int = 3, delay: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            return Settings.llm.complete(prompt).text.strip()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败: {type(last_err).__name__}: {last_err}")


def crag_evaluate(query: str, context: str) -> float:
    prompt = (
        "You are a retrieval quality evaluator.\n"
        "Output ONLY a single float between 0.0 and 1.0.\n"
        f"Query: {query}\nContext: {context}\n"
        "Score how well the context answers the query (0.0=irrelevant, 1.0=perfect):"
    )
    try:
        result = qwen_complete_with_retry(prompt, retries=3)
        m = re.search(r"1\.0|0\.\d+|\b[01]\b", result)
        return float(m.group(0)) if m else 0.5
    except Exception:
        return 0.5


def crag_rewrite(query: str, failed_context: str) -> List[str]:
    prompt = (
        "You are a query rewriting assistant.\n"
        "Output ONLY 2 rewritten queries separated by newline.\n"
        f"Original Query: {query}\n"
        f"Failed Context (excerpt): {(failed_context or '')[:400]}\n"
        "Rewrite into 2 simpler keyword-based search queries:"
    )
    try:
        result = qwen_complete_with_retry(prompt, retries=3)
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        return queries[:2] if queries else [query]
    except Exception:
        return [query]


def crag_retrieve_secondary(index: VectorStoreIndex, rewritten_queries: List[str]) -> str:
    chunks = []
    retriever = index.as_retriever(similarity_top_k=2)
    for rq in rewritten_queries:
        try:
            nodes = retriever.retrieve(rq)
            chunks.extend([n.text for n in nodes if n.text])
        except Exception:
            continue
    # 去重并保序
    dedup = list(dict.fromkeys(chunks))
    return "\n".join(dedup)


def normalize_source_id(meta: Dict) -> str:
    for key in ("file_name", "source_file", "doc_id", "source"):
        if key in meta and meta[key] is not None:
            return str(meta[key])
    return ""


def load_indices() -> Dict[str, VectorStoreIndex]:
    print("⏳ 正在挂载多粒度与消融向量数据库...")
    indices: Dict[str, VectorStoreIndex] = {}
    for name, config in INDEX_CONFIGS.items():
        if os.path.exists(config["path"]):
            db = chromadb.PersistentClient(path=config["path"])
            col = db.get_or_create_collection(name=config["collection"])
            vector_store = ChromaVectorStore(chroma_collection=col)
            indices[name] = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        else:
            print(f"⚠️ 未找到 {name} 数据库，后续如调用将 fallback 到 CONCEPT")
    if "CONCEPT" not in indices:
        raise RuntimeError("CONCEPT 索引缺失，无法执行实验。")
    return indices


def run_pipeline(qa_item: Dict, indices: Dict[str, VectorStoreIndex], strategy: str) -> Dict:
    query = qa_item["question"]
    target_source = str(qa_item.get("source_file", ""))
    gold_answer = qa_item.get("gold_answer", "")
    start_time = time.time()

    # 1) 检索
    search_query = query
    intent = "CONCEPT"
    top_k_val = TOP_K
    retrieval_path = "DEFAULT"
    crag_score = -1.0

    if strategy in ["VeriMind_Full", "VeriMind_No_Gating"]:
        intent = get_query_intent(query)
    elif strategy == "HyDE":
        search_query = generate_hyde_doc(query)
    elif strategy == "Long-Context_LLM":
        intent = "CONTEXT"
        top_k_val = 15
    elif strategy == "Ablation_no_structure":
        intent = "FLAT" if "FLAT" in indices else "CONCEPT"
    elif strategy == "CRAG_Local":
        intent = "CONCEPT"

    index = indices.get(intent, indices["CONCEPT"])
    retriever = index.as_retriever(similarity_top_k=top_k_val)
    nodes = retriever.retrieve(search_query)

    retrieved_sources = [normalize_source_id(getattr(n, "metadata", {}) or {}) for n in nodes]
    retrieval_hit = 1 if target_source and any(target_source in src for src in retrieved_sources) else 0
    context_str = "\n".join([n.text for n in nodes if getattr(n, "text", "")])
    avg_score = sum(getattr(n, "score", 0.0) for n in nodes) / len(nodes) if nodes else 0.0
    calibrated_ret = min(10.0, max(0.0, (avg_score - 0.3) * 20))

    # 2) CRAG 动态上下文改写与二次检索
    if strategy == "CRAG_Local":
        crag_score = crag_evaluate(query, context_str)
        if crag_score >= 0.7:
            retrieval_path = "CORRECT"
        elif crag_score < 0.3:
            rewritten = crag_rewrite(query, context_str)
            secondary = crag_retrieve_secondary(indices["CONCEPT"], rewritten)
            context_str = secondary if secondary else context_str
            retrieval_path = "INCORRECT->REWRITTEN"
        else:
            rewritten = crag_rewrite(query, context_str)
            secondary = crag_retrieve_secondary(indices["CONCEPT"], rewritten)
            if secondary:
                context_str = context_str + "\n\n[Supplemental]\n" + secondary
            retrieval_path = "AMBIGUOUS->MERGED"

    # 3) 生成
    input_tokens = len((context_str or "") + query) / 4

    if strategy in ["VeriMind_Full", "VeriMind_No_Routing", "Ablation_no_structure", "CRAG_Local"]:
        qa_prompt = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information and NOT prior knowledge, answer the query concisely and directly. "
            "If the exact answer cannot be found, you MUST explicitly output exactly: 'Information not found.' "
            "Do NOT use parametric memory or guess.\nQuery: {query_str}\nAnswer: "
        )
    elif strategy == "Self-RAG":
        qa_prompt = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Critique the context. If irrelevant, say 'Information not found'. "
            "Otherwise answer the query concisely using ONLY the context.\nQuery: {query_str}\nAnswer: "
        )
    elif strategy == "CoT_RAG":
        qa_prompt = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Please think step by step about how the context relates to the query, then provide the final answer. "
            "Use ONLY the context. If insufficient, output exactly 'Information not found.'\n"
            "Query: {query_str}\nAnswer: "
        )
    else:
        qa_prompt = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information, answer the query concisely and directly.\nQuery: {query_str}\nAnswer: "
        )

    answer = qwen_complete_with_retry(qa_prompt.format(context_str=context_str, query_str=query))
    output_tokens = len(answer) / 4

    # 4) 评判 + VeriMind 门控
    faith_score, hallucination_rate = evaluate_faithfulness(query, answer, context_str)
    trust_score = 0.0
    if strategy in ["VeriMind_Full", "VeriMind_No_Routing", "Ablation_no_structure"]:
        trust_score = 0.4 * calibrated_ret + 0.6 * faith_score
        if trust_score < 5.0:
            answer = "REJECTED: Information not found."
            faith_score, hallucination_rate = 10.0, 0.0

    # 5) NLP 指标
    precision, recall, f1 = compute_f1(gold_answer, answer)
    em = compute_exact_match(gold_answer, answer)
    latency = time.time() - start_time
    cost = estimate_cost(input_tokens, output_tokens)

    return {
        "strategy": strategy,
        "category": qa_item.get("category", ""),
        "difficulty": qa_item.get("difficulty", ""),
        "routed_granularity": intent,
        "retrieval_hit": retrieval_hit,
        "retrieval_path": retrieval_path,
        "crag_score": round(crag_score, 4) if crag_score >= 0 else -1.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "exact_match": em,
        "faith_score": faith_score,
        "hallucination_rate": hallucination_rate,
        "trust_score": round(trust_score, 2),
        "latency_sec": round(latency, 2),
        "est_cost_usd": round(cost, 5),
        "generated_answer": answer,
        # cross-eval 三评判器复用该上下文，避免二次检索漂移
        "judge_context": (context_str or "")[:4000],
    }


# ==========================================
# 📊 4) 主引擎（断点续传）
# ==========================================
def run_experiments(
    benchmark_file: str = BENCHMARK_FILE,
    output_csv: str = OUTPUT_CSV_DEFAULT,
    limit: int = 0,
) -> None:
    with open(benchmark_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    if limit and limit > 0:
        qa_data = qa_data[:limit]

    indices = load_indices()
    strategies = STRATEGIES_10

    print(f"🎯 开始执行实验，共 {len(qa_data)} 题 × {len(strategies)} 策略")

    completed_tasks = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            for _, row in df_existing.iterrows():
                completed_tasks.add((int(row["question_id"]), row["strategy"]))
            print(f"✅ 检测到存档，已自动跳过 {len(completed_tasks)} 个已完成项")
        except Exception as e:
            print(f"⚠️ 读取存档失败，将从头开始: {e}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    for i, qa in enumerate(qa_data):
        if (i, strategies[0]) not in completed_tasks:
            print(f"\n[Progress: {i + 1}/{len(qa_data)}] {qa['question'][:48]}...")

        for strategy in strategies:
            if (i, strategy) in completed_tasks:
                continue

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    res = run_pipeline(qa, indices, strategy)
                    res["question_id"] = i
                    res["question"] = qa.get("question", "")

                    df = pd.DataFrame([res])
                    df.to_csv(
                        output_csv,
                        mode="a",
                        header=not os.path.exists(output_csv),
                        index=False,
                        encoding="utf-8-sig",
                    )
                    print(
                        f"  └─ {strategy.ljust(22)} | F1: {res['f1_score']:.2f} | "
                        f"Hal: {res['hallucination_rate']:.1f} | Latency: {res['latency_sec']}s"
                    )
                    break
                except Exception as e:
                    print(f"  ⚠️ {strategy} 失败({type(e).__name__})，重试 {attempt + 1}/{max_retries}...")
                    time.sleep(4)
            else:
                print(f"  ❌ {strategy} 连续失败 {max_retries} 次，跳过继续。")

    print(f"\n🎉 实验完成，结果已保存至 {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VeriMind main experiment with 10 strategies.")
    parser.add_argument("--benchmark", default=BENCHMARK_FILE, help="Path to benchmark JSON.")
    parser.add_argument("--output", default=OUTPUT_CSV_DEFAULT, help="Path to output CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions for smoke test.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        benchmark_file=args.benchmark,
        output_csv=args.output,
        limit=args.limit,
    )
