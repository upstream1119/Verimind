"""
build_scifact_index.py
为 SciFact corpus 构建三粒度向量索引（128 / 512 / 1024）
Embedding: DashScope text-embedding-v3
存储: ChromaDB PersistentClient
支持分批写入 + 基于进度文件的断点续传
"""
import json
import os
import sys
import time
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv

# Windows GBK 兼容
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env（项目根目录）
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ================= 配置 =================
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "raw", "data", "corpus.jsonl")
BASE_DIR = os.path.dirname(__file__)

INDEX_CONFIGS = {
    "DETAIL":  {"chunk_size": 128,  "path": os.path.join(BASE_DIR, "chroma_db_sf_128"),  "collection": "scifact_128"},
    "CONCEPT": {"chunk_size": 512,  "path": os.path.join(BASE_DIR, "chroma_db_sf_512"),  "collection": "scifact_512"},
    "CONTEXT": {"chunk_size": 1024, "path": os.path.join(BASE_DIR, "chroma_db_sf_1024"), "collection": "scifact_1024"},
}
# ========================================

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v3",
    embed_batch_size=10,  # v3 API 限制每批最多 10 条
)


def load_corpus_as_documents():
    """将 SciFact corpus.jsonl 转为 LlamaIndex Document 列表"""
    documents = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title", "")
            abstract_sents = doc.get("abstract", [])
            abstract = " ".join(abstract_sents)
            text = f"Title: {title}\n\nAbstract: {abstract}"
            documents.append(Document(
                text=text,
                metadata={
                    "doc_id": str(doc["doc_id"]),
                    "title": title,
                    "source": "SciFact",
                }
            ))
    return documents


PROGRESS_FILE = os.path.join(BASE_DIR, "build_progress.json")
BATCH_SIZE = 200
MAX_RETRIES = 5


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def build_indices():
    print("=== SciFact 索引构建 ===")
    print(f"Embedding: text-embedding-v3")

    documents = load_corpus_as_documents()
    print(f"加载 corpus: {len(documents)} 篇文摘")

    if not documents:
        print("错误: corpus 为空，请先运行 download_scifact.py")
        return

    progress = load_progress()
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for name, config in INDEX_CONFIGS.items():
        idx_progress = progress.get(name, {"completed_batches": [], "done": False})

        if idx_progress.get("done"):
            print(f"\n--- {name} 索引已完成，跳过 ---")
            continue

        completed = set(idx_progress.get("completed_batches", []))
        print(f"\n--- 构建 {name} 索引 (chunk_size={config['chunk_size']}) ---")
        if completed:
            print(f"  续传：已完成批次 {sorted(completed)}")

        db = chromadb.PersistentClient(path=config["path"])
        chroma_collection = db.get_or_create_collection(
            name=config["collection"],
            metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        text_splitter = SentenceSplitter(chunk_size=config["chunk_size"], chunk_overlap=20)

        # 如果已有部分数据，从 vector store 恢复索引对象
        if completed:
            index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            index = None

        for batch_idx in range(total_batches):
            if batch_idx in completed:
                continue

            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(documents))
            batch_docs = documents[start:end]
            print(f"  批次 {batch_idx+1}/{total_batches} ({start}-{end-1})...", end=" ", flush=True)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if index is None:
                        # 第一批：创建索引
                        storage_context = StorageContext.from_defaults(vector_store=vector_store)
                        index = VectorStoreIndex.from_documents(
                            batch_docs,
                            storage_context=storage_context,
                            transformations=[text_splitter],
                            show_progress=False,
                        )
                    else:
                        nodes = text_splitter.get_nodes_from_documents(batch_docs)
                        index.insert_nodes(nodes)
                    print("OK", flush=True)
                    break
                except Exception as e:
                    print(f"重试 {attempt}/{MAX_RETRIES}: {e}", flush=True)
                    if attempt < MAX_RETRIES:
                        time.sleep(5 * attempt)
                    else:
                        print(f"失败 [{name}] 批次 {batch_idx+1}，保存进度后退出")
                        save_progress(progress)
                        raise

            # 每批成功后立即保存进度
            completed.add(batch_idx)
            idx_progress["completed_batches"] = sorted(completed)
            progress[name] = idx_progress
            save_progress(progress)

        idx_progress["done"] = True
        progress[name] = idx_progress
        save_progress(progress)
        print(f"完成: {config['path']} ({chroma_collection.count()} 条向量)")

    # 全部完成后删除进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("\n=== 全部索引构建完毕 ===")


if __name__ == "__main__":
    build_indices()