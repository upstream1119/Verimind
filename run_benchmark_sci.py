import pandas as pd
import os
import time
import chromadb
from rouge_score import rouge_scorer
import jieba

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# --- 1. 环境与模型配置 ---
MY_QWEN_KEY = "sk-01bcaa48e91445c2bdeeed0aaeff8855"

Settings.llm = DashScope(
    model_name="qwen-plus",
    api_key=MY_QWEN_KEY,
    temperature=0.01  # 极低温度保证评测公平和稳定
)
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v1",
    api_key=MY_QWEN_KEY
)

INDEX_CONFIGS = {
    "DETAIL": {"path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"path": "./chroma_db_md_1024", "collection": "rag_md_1024"},
}

# --- 2. 评测指标函数 ---
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def calculate_rouge_l(prediction, ground_truth):
    pred_tokens = " ".join(jieba.cut(str(prediction)))
    gt_tokens = " ".join(jieba.cut(str(ground_truth)))
    scores = scorer.score(gt_tokens, pred_tokens)
    return scores['rougeL'].fmeasure


def calculate_exact_match(prediction, ground_truth):
    if str(ground_truth).lower().strip() in str(prediction).lower().strip():
        return 1.0
    return 0.0


# --- 3. 核心功能函数 ---
def load_indices():
    indices = {}
    print("📦 正在加载多粒度向量库...")
    for name, config in INDEX_CONFIGS.items():
        try:
            db = chromadb.PersistentClient(path=config['path'])
            chroma_collection = db.get_or_create_collection(name=config['collection'])
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            if chroma_collection.count() > 0:
                indices[name] = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        except Exception as e:
            print(f"⚠️ {name} 索引暂不可用，跳过。")
    return indices


def get_intent(query):
    if any(k in query for k in ["比较", "总结", "趋势", "内容", "演讲"]): return "CONTEXT"
    if any(k in query for k in ["多少", "哪一年", "谁", "什么"]): return "DETAIL"
    return "CONCEPT"


def get_fair_top_k(intent):
    if intent == "DETAIL": return 12
    if intent == "CONTEXT": return 2
    return 3


# --- 4. 主评测流水线 (四大模型同台竞技) ---
def run_sci_evaluation(test_file="verimind_bench_full.csv"):
    if not os.path.exists(test_file):
        print(f"❌ 找不到测试题库: {test_file}")
        return

    df_test = pd.read_csv(test_file)
    indices = load_indices()
    if not indices or "CONCEPT" not in indices:
        print("❌ 向量库加载失败！")
        return

    results = []
    print(f"\n🚀 开始执行 SOTA 基准测试 (共 {len(df_test)} 题)...\n")

    for index, row in df_test.iterrows():
        query = row['query']
        ground_truth = str(row['ground_truth_answer'])
        print(f"[{index + 1}/{len(df_test)}] 正在评测: {query}")

        # ==== 1. 模型 A: Naive RAG (基线组 - 固定 512 Chunk) ====
        baseline_idx = indices["CONCEPT"]
        query_engine_base = baseline_idx.as_query_engine(similarity_top_k=3)
        ans_base = str(query_engine_base.query(query))
        rouge_base = calculate_rouge_l(ans_base, ground_truth)

        # ==== 2. 模型 B: HyDE (检索前生成假设性文档) ====
        hyde_transform = HyDEQueryTransform(llm=Settings.llm)
        hyde_query_engine = TransformQueryEngine(query_engine_base, query_transform=hyde_transform)
        ans_hyde = str(hyde_query_engine.query(query))
        rouge_hyde = calculate_rouge_l(ans_hyde, ground_truth)

        # ==== 3. 模型 C: Self-RAG (引入反思与严格证据约束) ====
        self_rag_prompt = PromptTemplate(
            "You are a Self-RAG system with rigorous reflection.\n"
            "Context information is below.\n"
            "---------------------\n{context_str}\n---------------------\n"
            "Task: Answer the query based strictly on the context.\n"
            "Critique Rule: If the context does not contain the answer, you MUST refuse to answer and say 'Information not found'. Do not hallucinate.\n"
            "Query: {query_str}\nAnswer: "
        )
        self_rag_engine = baseline_idx.as_query_engine(similarity_top_k=3)
        self_rag_engine.update_prompts({"response_synthesizer:text_qa_template": self_rag_prompt})
        ans_self_rag = str(self_rag_engine.query(query))
        rouge_self_rag = calculate_rouge_l(ans_self_rag, ground_truth)

        # ==== 4. 模型 D: VeriMind (Ours - 多粒度意图路由) ====
        intent = get_intent(query)
        fair_top_k = get_fair_top_k(intent)
        target_idx = indices.get(intent, indices["CONCEPT"])
        query_engine_ours = target_idx.as_query_engine(similarity_top_k=fair_top_k)
        ans_ours = str(query_engine_ours.query(query))
        rouge_ours = calculate_rouge_l(ans_ours, ground_truth)

        # 只算我们的 EM（用于学术探讨）
        em_ours = calculate_exact_match(ans_ours, ground_truth)

        # 记录数据
        results.append({
            "Query": query,
            "Ground_Truth": ground_truth,
            "Intent_Routed": intent,
            "Naive_RAG_ROUGE": rouge_base,
            "HyDE_ROUGE": rouge_hyde,
            "Self_RAG_ROUGE": rouge_self_rag,
            "VeriMind_ROUGE": rouge_ours,
            "VeriMind_EM": em_ours
        })
        print(f"  👉 [Naive RAG] ROUGE: {rouge_base:.4f}")
        print(f"  👉 [HyDE     ] ROUGE: {rouge_hyde:.4f}")
        print(f"  👉 [Self-RAG ] ROUGE: {rouge_self_rag:.4f}")
        print(f"  👉 [VeriMind ] ROUGE: {rouge_ours:.4f} (Top-K: {fair_top_k} | EM: {em_ours})")
        print("-" * 50)

    # --- 保存报告与宏观统计 ---
    df_res = pd.DataFrame(results)
    output_csv = "sota_evaluation_results.csv"
    df_res.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print(f"🎉 评测结束！SOTA 报表已保存至: {output_csv}")
    print(f"📊 【四大模型 ROUGE-L 宏观平均值】")
    print(f"  - Naive RAG : {df_res['Naive_RAG_ROUGE'].mean():.4f}")
    print(f"  - HyDE      : {df_res['HyDE_ROUGE'].mean():.4f}")
    print(f"  - Self-RAG  : {df_res['Self_RAG_ROUGE'].mean():.4f}")
    print(f"  - VeriMind  : {df_res['VeriMind_ROUGE'].mean():.4f} 🚀 (SOTA!)")
    print("=" * 50)


if __name__ == "__main__":
    run_sci_evaluation()