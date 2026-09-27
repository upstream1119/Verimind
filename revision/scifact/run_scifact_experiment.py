"""
run_scifact_experiment.py
SciFact 外部基准验证实验（任务 1.3）

系统对比：Vanilla_RAG / HyDE / VeriMind_Full
生成器：GLM-4.5-Air  (model_id=glm-4.5-air，使用专属资源包)
评判器：DeepSeek-V3.2  (model_id=deepseek-chat)
指标：Hallucination Rate、Coverage、ROUGE-L F1、BERTScore
支持断点续传（CSV 追加模式）
"""
import os
import sys

# 必须在 transformers/bert_score import 之前设置，跳过 HuggingFace 网络检查，使用本地缓存
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import time
import re

import pandas as pd
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from rouge_score import rouge_scorer as rouge_scorer_mod
from zhipuai import ZhipuAI
from openai import OpenAI

# BERTScore 软导入（未安装时降级输出 -1.0）
# 使用 BERTScorer 类而非 score() 函数，避免每次调用重新加载模型
try:
    from bert_score import BERTScorer
    _bert_scorer = BERTScorer(lang="en")  # 只加载一次 roberta-large
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

sys.stdout.reconfigure(encoding="utf-8")

# 加载项目根目录的 .env（脚本位于 revision/scifact/，上两级为根）
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ================== 路径 & 常量 ==================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "json", "scifact_bench.json")
OUTPUT_CSV     = os.path.join(PROJECT_ROOT, "csv", "scifact_experiment_results.csv")

INDEX_CONFIGS = {
    "DETAIL":  {"path": os.path.join(BASE_DIR, "chroma_db_sf_128"),  "collection": "scifact_128"},
    "CONCEPT": {"path": os.path.join(BASE_DIR, "chroma_db_sf_512"),  "collection": "scifact_512"},
    "CONTEXT": {"path": os.path.join(BASE_DIR, "chroma_db_sf_1024"), "collection": "scifact_1024"},
}

TOP_K      = 3
STRATEGIES = ["Vanilla_RAG", "HyDE", "VeriMind_Full"]

# ================== API 客户端 ==================
ZHIPU_API_KEY     = os.environ.get("ZHIPU_API_KEY", "")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

glm_client      = ZhipuAI(api_key=ZHIPU_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Embedding 与建索引时保持完全一致
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v3",
    api_key=DASHSCOPE_API_KEY,
    embed_batch_size=10,
)

_rouge = rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)


# ================== 模型调用封装 ==================
def glm_generate(prompt: str, max_tokens: int = 512) -> str:
    """GLM-4.5-Air 生成答案（thinking 模式禁用，使用专属资源包）"""
    for attempt in range(3):
        try:
            resp = glm_client.chat.completions.create(
                model="glm-4.5-air",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.01,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
            else:
                raise


def deepseek_judge(answer: str, context: str) -> tuple:
    """DeepSeek-V3.2 评判幻觉，返回 (faith_score: float, hallucination_rate: float)"""
    refusal_flags = [
        "information not found", "not found", "cannot find",
        "no evidence", "unable to determine", "rejected",
    ]
    if any(f in answer.lower() for f in refusal_flags):
        return 10.0, 0.0  # 拒绝回答视为无幻觉

    prompt = (
        "Evaluate if the [Generated Answer] is strictly derived from the [Retrieved Context].\n"
        "Output ONLY an integer score from 0 to 10.\n"
        "10 = completely faithful; 0 = completely hallucinated.\n\n"
        f"[Context]: {context[:2000]}\n"
        f"[Answer]: {answer}\n"
        "Score:"
    )
    for attempt in range(3):
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\b(10|[0-9])\b', raw)
            score = int(m.group(1)) if m else 6
            return float(score), (1.0 if score < 7 else 0.0)
        except Exception:
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
    return 6.0, 0.0  # 降级：返回中性分


# ================== 指标计算 ==================
def compute_rouge_l(gold: str, pred: str) -> float:
    if not gold or not pred:
        return 0.0
    return round(_rouge.score(gold, pred)["rougeL"].fmeasure, 4)


def compute_bertscore(gold: str, pred: str) -> float:
    if not BERTSCORE_AVAILABLE:
        return -1.0
    try:
        _, _, F1 = _bert_scorer.score([pred], [gold])
        return round(F1[0].item(), 4)
    except Exception:
        return -1.0


# ================== 意图路由 ==================
def get_query_intent(query: str) -> str:
    """关键词意图路由，与主实验逻辑对齐"""
    q = query.lower()
    if any(k in q for k in ["how many", "rate", "percent", "number", "year",
                              "when", "who", "which gene", "which protein", "quantit"]):
        return "DETAIL"
    if any(k in q for k in ["compare", "difference", "relationship", "association",
                              "compared to", "summarize", "mechanism", "versus"]):
        return "CONTEXT"
    return "CONCEPT"


# ================== 索引加载 ==================
def load_indices() -> dict:
    print("加载 SciFact 多粒度索引...")
    indices = {}
    for name, cfg in INDEX_CONFIGS.items():
        db  = chromadb.PersistentClient(path=cfg["path"])
        col = db.get_or_create_collection(name=cfg["collection"])
        vs  = ChromaVectorStore(chroma_collection=col)
        indices[name] = VectorStoreIndex.from_vector_store(vector_store=vs)
        print(f"  {name}: {col.count()} 条向量")
    return indices


# ================== 单条实验管线 ==================
def run_pipeline(item: dict, indices: dict, strategy: str) -> dict:
    query       = item["question"]
    gold_answer = item.get("gold_answer", "")
    source_id   = str(item.get("source_file", ""))
    t0          = time.time()

    # ----- 1. 检索 -----
    search_query = query
    intent       = "CONCEPT"

    if strategy == "VeriMind_Full":
        intent = get_query_intent(query)
    elif strategy == "HyDE":
        hyde_prompt = (
            "Please write a concise scientific passage (2-3 sentences) "
            f"that answers the following query:\n{query}"
        )
        search_query = glm_generate(hyde_prompt, max_tokens=200)

    index     = indices.get(intent, indices["CONCEPT"])
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes     = retriever.retrieve(search_query)

    context_str    = "\n".join(n.text for n in nodes)
    coverage_hit   = 1 if source_id and any(
        n.metadata.get("doc_id", "") == source_id for n in nodes) else 0
    avg_sim        = sum(n.score for n in nodes) / len(nodes) if nodes else 0.0
    calibrated_ret = min(10.0, max(0.0, (avg_sim - 0.3) * 20))

    # ----- 2. 生成 -----
    if strategy in ("VeriMind_Full", "Vanilla_RAG"):
        qa_prompt = (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given ONLY the context information above, answer the following query "
            "concisely and directly. "
            "If the answer cannot be found in the context, output exactly: "
            "'Information not found.'\n"
            f"Query: {query}\nAnswer:"
        )
    else:  # HyDE
        qa_prompt = (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given the context information, answer the query concisely.\n"
            f"Query: {query}\nAnswer:"
        )

    answer = glm_generate(qa_prompt, max_tokens=400)

    # ----- 3. DeepSeek 评判 -----
    faith_score, hallucination_rate = deepseek_judge(answer, context_str)

    # ----- 4. VeriMind 门控拦截 -----
    trust_score = 0.0
    if strategy == "VeriMind_Full":
        trust_score = 0.4 * calibrated_ret + 0.6 * faith_score
        if trust_score < 5.0:
            answer = "REJECTED: Information not found."
            faith_score, hallucination_rate = 10.0, 0.0

    # ----- 5. 指标计算 -----
    rouge_l  = compute_rouge_l(gold_answer, answer)
    bert_f1  = compute_bertscore(gold_answer, answer)
    latency  = round(time.time() - t0, 2)

    return {
        "strategy":           strategy,
        "label":              item.get("difficulty", ""),   # SUPPORT / CONTRADICT
        "routed_granularity": intent,
        "coverage_hit":       coverage_hit,
        "faith_score":        faith_score,
        "hallucination_rate": hallucination_rate,
        "trust_score":        round(trust_score, 2),
        "rouge_l_f1":         rouge_l,
        "bert_score_f1":      bert_f1,
        "latency_sec":        latency,
        "generated_answer":   answer,
    }


# ================== 主循环 ==================
def run_experiments():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    indices = load_indices()

    # 断点续传：读取已完成的 (item_id, strategy) 组合
    completed = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_old = pd.read_csv(OUTPUT_CSV)
            for _, row in df_old.iterrows():
                completed.add((int(row["item_id"]), row["strategy"]))
            print(f"检测到存档，跳过 {len(completed)} 条已完成记录")
        except Exception as e:
            print(f"读取存档失败，从头开始: {e}")

    total   = len(qa_data) * len(STRATEGIES)
    pending = total - len(completed)
    print(f"共 {len(qa_data)} 条 × {len(STRATEGIES)} 策略 = {total} 任务，"
          f"已完成 {len(completed)}，剩余 {pending}")

    for i, item in enumerate(qa_data):
        for strategy in STRATEGIES:
            if (i, strategy) in completed:
                continue

            for attempt in range(3):
                try:
                    res = run_pipeline(item, indices, strategy)
                    res["item_id"]  = i
                    res["question"] = item["question"][:120]

                    df = pd.DataFrame([res])
                    df.to_csv(
                        OUTPUT_CSV, mode="a",
                        header=not os.path.exists(OUTPUT_CSV),
                        index=False, encoding="utf-8-sig",
                    )
                    print(
                        f"[{i+1:3d}/{len(qa_data)}] {strategy:<18} | "
                        f"hal={res['hallucination_rate']:.0f}  "
                        f"cov={res['coverage_hit']}  "
                        f"rouge={res['rouge_l_f1']:.3f}  "
                        f"({res['latency_sec']}s)"
                    )
                    break
                except Exception as e:
                    print(f"  重试 {attempt+1}/3 [{strategy}]: {type(e).__name__}: {e}")
                    time.sleep(5)
            else:
                print(f"  [跳过] {strategy} 连续3次失败，继续下一条")

    print(f"\n=== 实验完成，结果已保存至 {OUTPUT_CSV} ===")

    # 汇总统计
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print("\n--- 各系统指标汇总 ---")
        def safe_mean(x):
            valid = x[x >= 0]
            return valid.mean() if len(valid) > 0 else float("nan")

        summary = df.groupby("strategy").agg(
            hal_rate  =("hallucination_rate", "mean"),
            coverage  =("coverage_hit",       "mean"),
            rouge_l   =("rouge_l_f1",         "mean"),
            bert_f1   =("bert_score_f1",       safe_mean),
        ).round(4)
        print(summary.to_string())


if __name__ == "__main__":
    if not BERTSCORE_AVAILABLE:
        print("[警告] bert_score 未安装，BERTScore 列将输出 -1.0")
        print("[提示] 如需启用 BERTScore: pip install bert-score")
    run_experiments()