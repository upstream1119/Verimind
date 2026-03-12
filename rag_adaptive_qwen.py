import streamlit as st
import os
import shutil
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file.pymu_pdf import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding

# --- 1. 配置与初始化 (保持 Qwen 配置) ---
# 务必确保环境变量 DASHSCOPE_API_KEY 已设置
Settings.llm = DashScope(
    model_name="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    max_tokens=2048,  # <--- 关键修改：允许更长的输出 (默认通常较短)
    temperature=0.3   # <--- 可选建议：降低一点随机性，让学术回答更严谨
)

DATA_DIR = "knowledge_pdfs"

# 定义不同粒度的配置
INDEX_CONFIGS = {
    "DETAIL": {"chunk_size": 128, "path": "./chroma_db_qwen_128", "collection": "rag_qwen_128"},
    "CONCEPT": {"chunk_size": 512, "path": "./chroma_db_qwen_512", "collection": "rag_qwen_512"},
    "CONTEXT": {"chunk_size": 1024, "path": "./chroma_db_qwen_1024", "collection": "rag_qwen_1024"},
}


@st.cache_resource
def initialize_index():
    """
    CodeMaster 智能加载版：优先加载已有数据库，只有为空时才新建。
    """
    st.info("🤖 系统自检中：检查向量数据库状态...")

    # 检查 PDF 目录
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.error(f"错误：{DATA_DIR} 目录为空！")
        return None

    all_indices = {}
    documents_loaded = False
    documents = []

    for name, config in INDEX_CONFIGS.items():
        try:
            # 1. 尝试连接现有数据库
            db = chromadb.PersistentClient(path=config['path'])
            # 关键：使用 cosine 距离
            chroma_collection = db.get_or_create_collection(name=config['collection'],
                                                            metadata={"hnsw:space": "cosine"})
            count = chroma_collection.count()

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            if count > 0:
                # A. 如果库里有数据，直接加载 (0秒耗时)
                st.success(f"✅ 检测到 {name} 索引已存在 ({count} chunks)，直接加载。")
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            else:
                # B. 如果库是空的，说明需要构建 (首次运行或数据被删)
                st.warning(f"⚠️ {name} 索引为空，准备从 PDF 构建 (请耐心等待)...")

                # 延迟加载 PDF (只在真正需要构建时才读取，节省内存)
                if not documents_loaded:
                    reader = PyMuPDFReader()
                    for filename in os.listdir(DATA_DIR):
                        if filename.endswith(".pdf"):
                            filepath = os.path.join(DATA_DIR, filename)
                            docs = reader.load_data(file_path=filepath)
                            for doc in docs: doc.metadata = {"file_name": filename}
                            documents.extend(docs)
                    documents_loaded = True

                # 定义切分器
                text_splitter = SentenceSplitter(chunk_size=config['chunk_size'],
                                                 chunk_overlap=int(config['chunk_size'] * 0.1))
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # 构建索引
                index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context, transformations=[text_splitter], show_progress=True
                )
                st.success(f"🎉 {name} 索引构建完毕。")

            all_indices[name] = index

        except Exception as e:
            st.error(f"加载/构建 {name} 索引时出错: {e}")
            return None

    return all_indices


def get_query_intent(query: str) -> str:
    """
    🧠 意图分类器 (Classifier)
    使用 LLM 判断用户问题属于哪一类，从而选择最佳 Chunk Size。
    """
    prompt = (
        f"你是一个RAG系统的路由助手。请分析用户问题：'{query}'。\n"
        "请仅从以下三个类别中选择一个最匹配的返回（不要返回任何其他文字）：\n"
        "1. DETAIL (用户询问具体的参数、数值、日期、单一事实，需要小粒度切片)\n"
        "2. CONCEPT (用户询问定义、原理、概念解释，需要中等粒度切片)\n"
        "3. CONTEXT (用户询问比较、总结、大范围背景、跨段落关系，需要大粒度切片)\n"
        "结果："
    )
    try:
        # 直接调用 LLM 完成分类
        response = Settings.llm.complete(prompt)
        intent = response.text.strip().upper()
        # 简单清洗，防止 LLM 话痨
        if "DETAIL" in intent: return "DETAIL"
        if "CONTEXT" in intent: return "CONTEXT"
        return "CONCEPT"  # 默认为概念
    except:
        return "CONCEPT"  # 容错兜底


# --- Streamlit 界面 ---
st.set_page_config(page_title="Adaptive RAG v2.0", layout="wide")
st.title("🧠 自适应多粒度 RAG 系统 (Adaptive RAG)")
st.caption("Phase 2.2: Auto-Routing | Powered by Qwen-Plus")

with st.sidebar:
    st.header("⚙️ 调试面板")
    st.info("已连接 DashScope API")

# 初始化
all_indices = initialize_index()

if all_indices:
    query = st.text_input("请输入你的问题:", placeholder="例如: RAG 和 Fine-tuning 相比有什么优势？")

    if query:
        # --- 1. 意图识别阶段 (The Brain) ---
        with st.status("🧠 正在分析意图...", expanded=True) as status:
            st.write("正在调用意图分类器 (Classifier Model)...")
            intent = get_query_intent(query)

            # 动态选择索引
            selected_index = all_indices.get(intent, all_indices["CONCEPT"])
            chunk_size_info = INDEX_CONFIGS[intent]['chunk_size']

            st.markdown(f"### 🎯 判定意图: **{intent}**")
            st.markdown(f"⚡ 自动路由: 切换至 **{chunk_size_info} tokens** 粒度索引")
            status.update(label="意图分析完成", state="complete", expanded=True)

        # --- 2. 检索与生成阶段 (The Action) ---
        with st.spinner(f"正在 {intent} 索引中检索并生成回答..."):
            # 针对不同意图，微调 top_k (Context需要更多片段)
            top_k = 6 if intent == "CONTEXT" else 4

            query_engine = selected_index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)

            st.subheader("💡 Qwen 智能回答")
            st.markdown(str(response))

            st.divider()
            st.subheader(f"📄 知识追溯 (来自 {intent} 索引库)")

            if response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    file_name = node.metadata.get('file_name', 'Unknown')
                    score = node.score if node.score else 0.0
                    # 可视化相似度条
                    st.caption(f"Source {i + 1}: {file_name} | Score: {score:.4f}")
                    st.progress(min(score, 1.0))
                    with st.expander("点击查看原文片段"):
                        st.text(node.text)
            else:
                st.warning("未检索到高相关性文档。")