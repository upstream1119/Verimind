"""Build structured and flat QASPER Chroma indexes."""

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CORPUS_FILE = os.path.join(BASE_DIR, "qasper_corpus.jsonl")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

INDEX_CONFIGS = {
    "DETAIL": {
        "chunk_size": 128,
        "chunk_overlap": 0,
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_128"),
        "collection": "qasper_128",
        "structured": True,
    },
    "CONCEPT": {
        "chunk_size": 512,
        "chunk_overlap": 75,
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_512"),
        "collection": "qasper_512",
        "structured": True,
    },
    "CONTEXT": {
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_1024"),
        "collection": "qasper_1024",
        "structured": True,
    },
    "FLAT": {
        "chunk_size": 512,
        "chunk_overlap": 75,
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_flat_512"),
        "collection": "qasper_flat_512",
        "structured": False,
    },
}
INSERT_BATCH_SIZE = 100
MAX_RETRIES = 5


def load_corpus() -> List[dict]:
    papers = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if line.strip():
                papers.append(json.loads(line))
    return papers


def structured_documents(papers: Iterable[dict]) -> List[Document]:
    documents = []
    for paper in papers:
        prefix = f"Title: {paper['title']}\n"
        if paper.get("abstract"):
            documents.append(
                Document(
                    text=f"{prefix}Section: Abstract\n{paper['abstract']}",
                    metadata={
                        "paper_id": paper["paper_id"],
                    },
                )
            )
        for section in paper.get("sections", []):
            section_name = section.get("section_name") or "Untitled section"
            for paragraph_index, paragraph in enumerate(
                section.get("paragraphs", [])
            ):
                documents.append(
                    Document(
                        text=(
                            f"{prefix}Section: {section_name}\n"
                            f"{paragraph}"
                        ),
                        metadata={
                            "paper_id": paper["paper_id"],
                        },
                    )
                )
    return documents


def flat_documents(papers: Iterable[dict]) -> List[Document]:
    documents = []
    for paper in papers:
        pieces = [paper["title"], paper.get("abstract", "")]
        for section in paper.get("sections", []):
            pieces.append(section.get("section_name", ""))
            pieces.extend(section.get("paragraphs", []))
        documents.append(
            Document(
                text="\n\n".join(piece for piece in pieces if piece),
                metadata={
                    "paper_id": paper["paper_id"],
                },
            )
        )
    return documents


def build_index(name: str, config: Dict, papers: List[dict], rebuild: bool) -> None:
    os.makedirs(config["path"], exist_ok=True)
    client = chromadb.PersistentClient(path=config["path"])
    if rebuild:
        try:
            client.delete_collection(config["collection"])
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=config["collection"],
        metadata={"hnsw:space": "cosine"},
    )
    documents = (
        structured_documents(papers)
        if config["structured"]
        else flat_documents(papers)
    )
    splitter = SentenceSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    existing_count = collection.count()
    if existing_count > len(nodes):
        raise RuntimeError(
            f"{name}: index contains {existing_count} nodes but only "
            f"{len(nodes)} were generated. Re-run with --rebuild."
        )
    if existing_count == len(nodes):
        print(f"{name}: complete with {existing_count} nodes; skipped")
        return
    if existing_count:
        print(
            f"{name}: resuming at node {existing_count} of {len(nodes)}"
        )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    for batch_start in range(
        existing_count,
        len(nodes),
        INSERT_BATCH_SIZE,
    ):
        batch_end = min(batch_start + INSERT_BATCH_SIZE, len(nodes))
        batch = nodes[batch_start:batch_end]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                index.insert_nodes(batch)
                print(
                    f"{name}: inserted {batch_end}/{len(nodes)}",
                    flush=True,
                )
                break
            except Exception as error:
                if attempt == MAX_RETRIES:
                    raise
                delay = 5 * attempt
                print(
                    f"{name}: batch {batch_start}-{batch_end} failed "
                    f"({type(error).__name__}); retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
    print(f"{name}: built {collection.count()} nodes at {config['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        choices=[*INDEX_CONFIGS.keys(), "ALL"],
        default="ALL",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Missing {CORPUS_FILE}. Run prepare_qasper.py first."
        )

    sys.stdout.reconfigure(encoding="utf-8")
    Settings.embed_model = DashScopeEmbedding(
        model_name="text-embedding-v3",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        embed_batch_size=10,
    )
    papers = load_corpus()
    selected = INDEX_CONFIGS if args.index == "ALL" else {
        args.index: INDEX_CONFIGS[args.index]
    }
    print(f"Loaded {len(papers)} QASPER papers")
    for name, config in selected.items():
        build_index(name, config, papers, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
