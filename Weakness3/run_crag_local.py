import json
import re
import pandas as pd
import dashscope
from dashscope import Generation, TextEmbedding
import chromadb
from tqdm import tqdm
import time
import os

# ================= 核心配置区 =================
dashscope.api_key = "sk-01bcaa48e91445c2bdeeed0aaeff8855"  # ⚠️ 填入你的阿里云 API Key
INPUT_JSON  = "../json/verimind_bench_v3.json"
OUTPUT_CSV  = "../csv/w3_crag_local_results.csv"
SLEEP_SEC   = 0.5
MAX_RETRIES = 3
CRAG_CORRECT_THRESHOLD   = 0.7   # 高于此分直接生成
CRAG_INCORRECT_THRESHOLD = 0.3   # 低于此分触发重写+二次检索
                                  # 两者之间为 AMBIGUOUS，合并上下文
# 三个 ChromaDB 库（二次检索默认用 512，必要时扩展到其他粒度）
CHROMA_CONFIGS = {
    "128":  {"dir": "../chroma_db_md_128",  "collection": "rag_md_128"},
    "512":  {"dir": "../chroma_db_md_512",  "collection": "rag_md_512"},
    "1024": {"dir": "../chroma_db_md_1024", "collection": "rag_md_1024"},
}
# ==============================================

print("🔌 正在连接本地 ChromaDB...")
chroma_collections = {}
for size, cfg in CHROMA_CONFIGS.items():
    if not os.path.exists(cfg["dir"]):
        print(f"❌ 找不到 ChromaDB 文件夹 {cfg['dir']}，请检查路径！")
        exit()
    client = chromadb.PersistentClient(path=cfg["dir"])
    chroma_collections[size] = client.get_collection(name=cfg["collection"])
    print(f"✅ 连接成功：{cfg['dir']} → collection '{cfg['collection']}'")

print(f"📥 加载测试集 {INPUT_JSON}...")
if not os.path.exists(INPUT_JSON):
    print(f"❌ 找不到测试集文件 {INPUT_JSON}，请检查路径！")
    exit()
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)
print(f"✅ 加载完成，共 {len(bench_data)} 条")


# ── CRAG 核心功能函数 ──────────────────────────────────

def call_llm(messages, temperature=0.01, max_tokens=512, retries=MAX_RETRIES):
    """统一的 LLM 调用入口，支持重试"""
    for attempt in range(1, retries + 1):
        try:
            resp = Generation.call(
                model='qwen-plus',
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format='message'
            )
            return resp.output.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n⚠️ LLM 调用第{attempt}次失败: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


def crag_evaluate(query, context):
    """评估初始检索结果是否足以回答问题，返回 [0,1] 分数"""
    messages = [
        {"role": "system", "content":
            "You are a retrieval quality evaluator. "
            "Output ONLY a single float between 0.0 and 1.0. No other text."},
        {"role": "user", "content":
            f"Query: {query}\nContext: {context}\n\n"
            "Score how well the context answers the query (0.0=irrelevant, 1.0=perfect):"}
    ]
    result = call_llm(messages, temperature=0.01, max_tokens=16)
    if result is None:
        return 0.5
    match = re.search(r"1\.0|0\.\d+|\b[01]\b", result)
    return float(match.group()) if match else 0.5


def crag_rewrite(query, failed_context):
    """将原问题重写为 2 个更易检索的子问题"""
    messages = [
        {"role": "system", "content":
            "You are a query rewriting assistant. "
            "Output ONLY 2 rewritten queries separated by a newline. No numbering or explanation."},
        {"role": "user", "content":
            f"Original Query: {query}\n"
            f"Failed Context (excerpt): {failed_context[:400]}\n\n"
            "Rewrite into 2 simpler keyword-based search queries:"}
    ]
    result = call_llm(messages, temperature=0.3, max_tokens=128)
    if result is None:
        return [query]
    queries = [q.strip() for q in result.split('\n') if q.strip()]
    return queries[:2] if queries else [query]


def embed_query(text):
    """用阿里云 text-embedding-v1 将文本转为 1536 维向量"""
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=[text]
    )
    return resp.output['embeddings'][0]['embedding']


def crag_retrieve_secondary(rewritten_queries):
    """用重写后的查询去 ChromaDB 做二次检索（手动 embed，避免维度冲突）"""
    docs = []
    for rq in rewritten_queries:
        try:
            vec = embed_query(rq)
            res = chroma_collections["512"].query(query_embeddings=[vec], n_results=2)
            if res['documents']:
                docs.extend(res['documents'][0])
        except Exception as e:
            print(f"\n⚠️ 二次检索报错: {e}")
    # 去重
    return list(dict.fromkeys(docs))


def generate_final(query, context):
    """最终生成，严格基于 context"""
    messages = [
        {"role": "system", "content":
            "You are a strict and highly accurate answering assistant. "
            "Answer the question based ONLY on the provided context. "
            "If the context does not contain sufficient information, "
            "output exactly: Information not found."},
        {"role": "user", "content":
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]
    result = call_llm(messages, temperature=0.01, max_tokens=512)
    return result if result else "Error during generation."


# ── 主循环 ────────────────────────────────────────────

results = []
print("🚀 开始执行 CRAG-Local 动态评估与二次检索...")

for idx, item in enumerate(tqdm(bench_data)):
    q_id           = idx                    # JSON 无 question_id，用索引代替
    query          = item['question']       # JSON 字段名是 question，不是 query
    initial_context = item['context']
    gold           = item['gold_answer']
    category       = item.get('category', '')
    difficulty     = item.get('difficulty', '')

    # Step 1: 评估初始 context 质量
    eval_score = crag_evaluate(query, initial_context)

    # Step 2: 三路路由
    if eval_score >= CRAG_CORRECT_THRESHOLD:
        final_context = initial_context
        path = "CORRECT"

    elif eval_score < CRAG_INCORRECT_THRESHOLD:
        # 完全重写+二次检索，替换原 context
        rewritten = crag_rewrite(query, initial_context)
        secondary = crag_retrieve_secondary(rewritten)
        final_context = "\n\n".join(secondary) if secondary else initial_context
        path = "INCORRECT->REWRITTEN"

    else:
        # AMBIGUOUS：合并原 context 与二次检索结果
        rewritten = crag_rewrite(query, initial_context)
        secondary = crag_retrieve_secondary(rewritten)
        supplemental = "\n".join(secondary)
        final_context = (initial_context + "\n\n[Supplemental]:\n" + supplemental
                         if supplemental else initial_context)
        path = "AMBIGUOUS->MERGED"

    # Step 3: 最终生成
    answer = generate_final(query, final_context)

    results.append({
        'question_id':      q_id,
        'strategy':         'CRAG_Local',
        'category':         category,
        'difficulty':       difficulty,
        'generated_answer': answer,
        'gold_answer':      gold,
        'crag_score':       eval_score,
        'retrieval_path':   path,
    })

    time.sleep(SLEEP_SEC)

# ── 保存与统计 ────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\n✅ CRAG-Local 完成！共 {len(results)} 条，结果已保存至 {OUTPUT_CSV}")

# 路由分布统计
from collections import Counter
path_dist = Counter(df['retrieval_path'])
print("\n路由分布：")
for p, n in path_dist.items():
    print(f"  {p}: {n} ({n/len(df)*100:.1f}%)")

rejected = df['generated_answer'].str.contains('Information not found').sum()
errors   = (df['generated_answer'] == 'Error during generation.').sum()
print(f"\n拒绝回答：{rejected} 条 ({rejected/len(df)*100:.1f}%)")
if errors:
    print(f"⚠️  生成失败：{errors} 条")
else:
    print("🎉 全部生成成功，无报错条目")