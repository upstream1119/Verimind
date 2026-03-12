import os
import time
import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from tqdm import tqdm  # 进度条库
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- 配置 ---
if "DASHSCOPE_API_KEY" not in os.environ:
    # 填入你的 Key
    os.environ["DASHSCOPE_API_KEY"] = "sk-你的Key"

Settings.embed_model = DashScopeEmbedding(model_name="text-embedding-v1")

DATA_DIR = "./knowledge_mds"
INDEX_CONFIGS = {
    "DETAIL": {"chunk_size": 128, "path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"chunk_size": 512, "path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"chunk_size": 1024, "path": "./chroma_db_md_1024", "collection": "rag_md_1024"},
}


def build_knowledge_base():
    print("🚀 Code Master 建库引擎启动...")

    # 1. 扫描文件
    md_files = list(Path(DATA_DIR).rglob("*.md"))
    print(f"📄 扫描到 {len(md_files)} 篇 Markdown 论文。")
    if not md_files: return

    # 2. 读取所有文档内容 (内存中)
    documents = []
    print("📖 正在预读取文件内容...")
    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            category = file_path.parent.name
            doc = Document(text=text, metadata={"file_name": file_path.name, "category": category})
            documents.append(doc)
        except Exception as e:
            print(f"❌ 跳过损坏文件: {file_path.name}")

    # 3. 分别构建三级索引
    for name, config in INDEX_CONFIGS.items():
        print(f"\n🔨 正在构建索引: {name} (Chunk Size: {config['chunk_size']})...")
        print("   (这可能需要几分钟，请不要关闭...)")

        try:
            # 初始化 Chroma
            db = chromadb.PersistentClient(path=config['path'])
            chroma_collection = db.get_or_create_collection(
                name=config['collection'],
                metadata={"hnsw:space": "cosine"}  # <--- 强制使用余弦相似度
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 如果已经有数据，先清空 (确保是全新的)
            # 注意：Chroma 的 delete 比较麻烦，这里我们假设你已经物理删除了文件夹
            # 如果没删，这个脚本会增量添加

            # 构建索引 (带进度显示)
            text_splitter = SentenceSplitter(chunk_size=config['chunk_size'], chunk_overlap=20)

            # 这里的 show_progress=True 会在终端显示进度条
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[text_splitter],
                show_progress=True
            )
            print(f"✅ {name} 索引构建完成！")

        except Exception as e:
            print(f"❌ 构建 {name} 失败: {e}")

    print("\n🎉🎉🎉 全部索引重建完毕！")
    print("👉 现在你可以运行 'streamlit run rag_app_final.py' 了，将会秒开！")


if __name__ == "__main__":
    build_knowledge_base()