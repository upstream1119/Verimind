import streamlit as st
import os
import chromadb
import re
from pathlib import Path  # <--- 用于递归查找文件
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding

# --- 1. 配置区 ---
# 务必确保环境变量 DASHSCOPE_API_KEY 已设置
Settings.llm = DashScope(
    model_name="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    max_tokens=2048,
    temperature=0.1
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

DATA_DIR = "knowledge_mds"
INDEX_CONFIGS = {
    "DETAIL": {"chunk_size": 128, "path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"chunk_size": 512, "path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"chunk_size": 1024, "path": "./chroma_db_md_1024", "collection": "rag_md_1024"},
}

# --- 2. 核心功能函数 ---

@st.cache_resource
def initialize_index():
    """加载或构建索引 (支持递归读取子文件夹 + 自动分类标签)"""

    if not os.path.exists(DATA_DIR):
        st.error(f"❌ 错误：找不到 {DATA_DIR} 文件夹！")
        return None

    all_indices = {}

    # --- 🔥 Code Master 升级：递归读取所有子文件夹 ---
    # rglob("*.md") 会查找所有子目录下的 .md 文件
    md_files = list(Path(DATA_DIR).rglob("*.md"))

    if not md_files:
        st.error(f"❌ 错误：在 {DATA_DIR} 及其子文件夹中未找到任何 .md 文件！")
        return None

    # 预加载文档
    documents = []
    for file_path in md_files:
        try:
            # 读取内容
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # ✨ 自动提取分类信息 (文件夹名)
            # 例如: knowledge_mds/01_Agents/ReAct.md -> category = "01_Agents"
            category = file_path.parent.name

            # 创建文档对象，并打上标签
            doc = Document(
                text=text,
                metadata={
                    "file_name": file_path.name, # 文件名
                    "category": category         # 子文件夹名 (作为分类标签)
                }
            )
            documents.append(doc)
        except Exception as e:
            st.warning(f"跳过损坏文件 {file_path.name}: {e}")

    st.success(f"📂 已加载 {len(documents)} 篇论文 (涵盖 {len(set(d.metadata['category'] for d in documents))} 个分类领域)。")

    # --- 构建索引 ---
    for name, config in INDEX_CONFIGS.items():
        try:
            db = chromadb.PersistentClient(path=config['path'])
            chroma_collection = db.get_or_create_collection(
                name=config['collection'],
                metadata={"hnsw:space": "cosine"}
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 检查是否已存在数据
            if chroma_collection.count() > 0:
                st.success(f"✅ {name} 索引已就绪 (加载自本地)。")
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            else:
                st.info(f"🔨 正在构建 {name} 索引 (包含分类元数据)...")
                text_splitter = SentenceSplitter(chunk_size=config['chunk_size'], chunk_overlap=int(config['chunk_size']*0.1))

                # 使用标准构建 (如果网络好)
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    transformations=[text_splitter],
                    show_progress=True
                )
                st.success(f"🎉 {name} 索引构建完毕！")

            all_indices[name] = index

        except Exception as e:
            st.error(f"构建 {name} 索引失败: {e}")
            return None

    return all_indices


def get_query_intent(query: str) -> str:
    """
    🧠 意图分类器 (修复版)
    """
    prompt = (
        f"你是一个分类器。请分析用户问题：'{query}'。\n"
        "请仅从以下三个类别中选择一个最匹配的返回（不要解释，只返回单词）：\n"
        "1. DETAIL\n"
        "2. CONCEPT\n"
        "3. CONTEXT\n"
        "结果："
    )
    try:
        # 获取 LLM 回答
        response = Settings.llm.complete(prompt).text.strip().upper()

        # 🔥 Code Master 修复：使用关键词匹配，而不是全等匹配
        if "DETAIL" in response: return "DETAIL"
        if "CONTEXT" in response: return "CONTEXT"
        if "CONCEPT" in response: return "CONCEPT"

        return "CONCEPT"  # 兜底
    except:
        return "CONCEPT"

def evaluate_faithfulness(query, answer, context_text):
    """Trust-Score 审计员"""
    eval_prompt = (
        "你是一位公正的法官。请评估[回答]是否完全基于[参考上下文]生成。\n"
        "--------\n"
        f"[用户问题]: {query}\n"
        f"[参考上下文]: {context_text[:2000]}\n"
        f"[回答]: {answer}\n"
        "--------\n"
        "请打分（0-10），仅返回一个数字。"
    )
    try:
        score_str = Settings.llm.complete(eval_prompt).text.strip()
        match = re.search(r'\d+', score_str)
        return int(match.group()) if match else 5
    except: return 5

# --- 3. Streamlit 界面 ---

st.set_page_config(page_title="VeriMind RAG", layout="wide", page_icon="🛡️")

with st.sidebar:
    st.header("🛡️ VeriMind 系统")
    st.caption("基于 Markdown 知识库的高可信引擎")
    st.success("✅ 知识库格式: Markdown (.md)")
    st.info(f"📚 已加载文档库: {DATA_DIR}")

st.title("🛡️ VeriMind: 学术级防幻觉 RAG (Agentic 版)")
st.markdown("Powered by **Qwen-Plus** | **Adaptive Routing** | **Trust-Score**")

all_indices = initialize_index()

if all_indices:
    query = st.chat_input("请输入你的学术问题...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        intent = get_query_intent(query)
        # 简单映射逻辑
        target_intent = intent if intent in ["DETAIL", "CONTEXT"] else "CONCEPT"
        idx = all_indices.get(target_intent, all_indices["CONCEPT"])

        with st.spinner(f"🧠 意图识别: {intent} | 正在全库检索..."):
            query_engine = idx.as_query_engine(similarity_top_k=5)
            response = query_engine.query(query)

            # Trust-Score 计算
            raw_scores = [n.score for n in response.source_nodes if n.score]
            if raw_scores:
                raw_avg = sum(raw_scores) / len(raw_scores)
                calibrated_score = (raw_avg - 0.35) / (0.85 - 0.35) * 10
                avg_retrieval_score = max(0.0, min(10.0, calibrated_score))
            else:
                avg_retrieval_score = 0.0

            context_texts = [n.text for n in response.source_nodes]
            full_context = "\n".join(context_texts)
            faithfulness_score = evaluate_faithfulness(query, str(response), full_context)

            final_trust_score = (0.4 * avg_retrieval_score) + (0.6 * faithfulness_score)
            final_trust_score = min(10.0, final_trust_score)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**Qwen 回答:**\n\n{str(response)}")

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("📡 检索相关性", f"{avg_retrieval_score:.1f}")
            c2.metric("⚖️ 忠实度审计", f"{faithfulness_score}")
            c3.metric("🛡️ Trust-Score", f"{final_trust_score:.1f}",
                      delta="High Trust" if final_trust_score > 7.5 else "-Check Source")

            with st.expander("查看 Markdown 溯源 (含分类标签)"):
                for node in response.source_nodes:
                    # 🔥 显示分类标签
                    category_tag = node.metadata.get('category', 'Unknown')
                    st.markdown(f"**📄 {node.metadata.get('file_name')}** | 🏷️ `{category_tag}` | Score: {node.score:.4f}")
                    st.markdown("```markdown\n" + node.text[:500] + "\n```")