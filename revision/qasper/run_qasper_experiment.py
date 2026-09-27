"""Run public-benchmark and core ablation experiments on QASPER."""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import chromadb
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rouge_score import rouge_scorer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "revision", "cross_arch"))
from llm_clients import build_client  # noqa: E402


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
BENCHMARK_FILE = os.path.join(BASE_DIR, "qasper_bench.json")
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "revision",
    "artifacts",
    "1_4_qasper",
    "qasper_experiment_results.csv",
)
TOP_K = 3
DEFAULT_STRATEGIES = [
    "Vanilla_RAG",
    "HyDE",
    "VeriMind_Full",
    "VeriMind_No_Gating",
    "VeriMind_No_Routing",
    "Ablation_no_structure",
]
INDEX_CONFIGS = {
    "DETAIL": {
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_128"),
        "collection": "qasper_128",
    },
    "CONCEPT": {
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_512"),
        "collection": "qasper_512",
    },
    "CONTEXT": {
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_1024"),
        "collection": "qasper_1024",
    },
    "FLAT": {
        "path": os.path.join(BASE_DIR, "chroma_db_qasper_flat_512"),
        "collection": "qasper_flat_512",
    },
}
REFUSAL_FLAGS = [
    "information not found",
    "not found in the context",
    "cannot be determined",
    "unable to determine",
    "insufficient evidence",
    "rejected",
    "unanswerable",
]
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def is_refusal(answer: str) -> bool:
    low = (answer or "").lower()
    return any(flag in low for flag in REFUSAL_FLAGS)


def get_query_intent(query: str) -> str:
    q = query.lower()
    if any(
        key in q
        for key in [
            "how many",
            "how much",
            "what score",
            "what value",
            "which dataset",
            "which model",
            "when",
            "where",
        ]
    ):
        return "DETAIL"
    if any(
        key in q
        for key in [
            "compare",
            "difference",
            "why",
            "how does",
            "relationship",
            "summarize",
            "what are the results",
        ]
    ):
        return "CONTEXT"
    return "CONCEPT"


def load_indices() -> Dict[str, VectorStoreIndex]:
    indices = {}
    for name, config in INDEX_CONFIGS.items():
        client = chromadb.PersistentClient(path=config["path"])
        collection = client.get_or_create_collection(config["collection"])
        if collection.count() == 0:
            raise RuntimeError(
                f"{name} index is empty. Run build_qasper_index.py first."
            )
        indices[name] = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=collection)
        )
        print(f"{name}: {collection.count()} nodes")
    return indices


def retrieve(
    index: VectorStoreIndex,
    query: str,
    paper_id: str,
    top_k: int = TOP_K,
) -> Tuple[str, float]:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="paper_id",
                operator=FilterOperator.EQ,
                value=str(paper_id),
            )
        ]
    )
    nodes = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    ).retrieve(query)
    context = "\n\n".join(node.text for node in nodes if node.text)
    average_similarity = (
        sum(float(node.score or 0.0) for node in nodes) / len(nodes)
        if nodes
        else 0.0
    )
    return context, average_similarity


def normalized_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def evidence_recall(context: str, evidence: Iterable[str]) -> float:
    evidence_items = [item for item in evidence if item and item.strip()]
    if not evidence_items:
        return -1.0
    context_tokens = normalized_tokens(context)
    matched = 0
    for evidence_item in evidence_items:
        evidence_tokens = normalized_tokens(evidence_item)
        if not evidence_tokens:
            continue
        recall = len(context_tokens & evidence_tokens) / len(evidence_tokens)
        if recall >= 0.70:
            matched += 1
    return matched / len(evidence_items)


def best_rouge_l(prediction: str, references: Sequence[str]) -> float:
    if not prediction or not references:
        return 0.0
    return max(
        _ROUGE.score(reference, prediction)["rougeL"].fmeasure
        for reference in references
    )


def make_answer_prompt(query: str, context: str) -> str:
    return (
        "Answer the research-paper question using ONLY the retrieved context.\n"
        "Give a concise direct answer. Do not add facts that are absent from "
        "the context. If the evidence is insufficient, output exactly: "
        "'Information not found.'\n\n"
        f"[Retrieved Context]\n{context}\n\n"
        f"[Question]\n{query}\n\n[Answer]\n"
    )


def run_pipeline(
    item: dict,
    strategy: str,
    indices: Dict[str, VectorStoreIndex],
    generator,
    judge,
) -> dict:
    start = time.time()
    query = item["question"]
    search_query = query
    routed_granularity = "CONCEPT"

    if strategy in ("VeriMind_Full", "VeriMind_No_Gating"):
        routed_granularity = get_query_intent(query)
    elif strategy == "Ablation_no_structure":
        routed_granularity = "FLAT"
    elif strategy == "HyDE":
        search_query = generator.complete(
            "Write a short hypothetical scientific passage that would answer "
            f"this question. Output only the passage:\n{query}",
            max_tokens=180,
            temperature=0.01,
        )

    context, average_similarity = retrieve(
        indices[routed_granularity],
        search_query,
        item["paper_id"],
    )
    generated_answer = generator.complete(
        make_answer_prompt(query, context),
        max_tokens=400,
        temperature=0.01,
    )

    pre_gate_faith = judge.score_faithfulness(generated_answer, context)
    raw_hallucination = int(
        not is_refusal(generated_answer) and pre_gate_faith < 7
    )
    calibrated_retrieval = min(
        10.0,
        max(0.0, (average_similarity - 0.3) * 20),
    )
    trust_score = (
        0.4 * calibrated_retrieval + 0.6 * pre_gate_faith
    )

    final_answer = generated_answer
    decision = "ANSWER"
    gating_enabled = strategy in (
        "VeriMind_Full",
        "VeriMind_No_Routing",
        "Ablation_no_structure",
    )
    if gating_enabled and trust_score < 5.0:
        final_answer = "REJECTED: Information not found."
        decision = "REJECT"
    elif is_refusal(final_answer):
        decision = "ABSTAIN"

    accepted = int(decision == "ANSWER")
    passed_hallucination = int(accepted and raw_hallucination)
    evidence_items = (
        item.get("highlighted_evidence")
        or item.get("evidence")
        or []
    )

    return {
        "item_id": item["item_id"],
        "question_id": item["question_id"],
        "paper_id": item["paper_id"],
        "paper_title": item["paper_title"],
        "question": query,
        "gold_answer": item["gold_answer"],
        "gold_answers_json": json.dumps(
            item["gold_answers"],
            ensure_ascii=False,
        ),
        "is_unanswerable": int(item["is_unanswerable"]),
        "strategy": strategy,
        "routed_granularity": routed_granularity,
        "average_similarity": round(average_similarity, 6),
        "calibrated_retrieval": round(calibrated_retrieval, 4),
        "pre_gate_faith_score": float(pre_gate_faith),
        "trust_score": round(trust_score, 4),
        "raw_hallucination": raw_hallucination,
        "passed_hallucination": passed_hallucination,
        "accepted": accepted,
        "decision": decision,
        "evidence_recall": round(
            evidence_recall(context, evidence_items),
            4,
        ),
        "rouge_l_f1": round(
            best_rouge_l(final_answer, item["gold_answers"]),
            4,
        ),
        "latency_sec": round(time.time() - start, 2),
        "generated_answer": generated_answer,
        "final_answer": final_answer,
        "retrieved_context": context,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in df.groupby("strategy"):
        accepted_count = int(group["accepted"].sum())
        answerable = group[group["is_unanswerable"] == 0]
        unanswerable = group[group["is_unanswerable"] == 1]
        rows.append(
            {
                "strategy": strategy,
                "n": len(group),
                "raw_hallucination_rate": group[
                    "raw_hallucination"
                ].mean(),
                "passed_hallucination_rate": (
                    group["passed_hallucination"].sum() / accepted_count
                    if accepted_count
                    else 0.0
                ),
                "answer_coverage": group["accepted"].mean(),
                "answerable_coverage": (
                    answerable["accepted"].mean()
                    if len(answerable)
                    else float("nan")
                ),
                "unanswerable_abstention_rate": (
                    1.0 - unanswerable["accepted"].mean()
                    if len(unanswerable)
                    else float("nan")
                ),
                "evidence_recall": group.loc[
                    group["evidence_recall"] >= 0,
                    "evidence_recall",
                ].mean(),
                "rouge_l_f1": group["rouge_l_f1"].mean(),
                "latency_sec": group["latency_sec"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark", default=BENCHMARK_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--generator", default="glm")
    parser.add_argument("--judge", default="deepseek")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=DEFAULT_STRATEGIES,
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    Settings.embed_model = DashScopeEmbedding(
        model_name="text-embedding-v3",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        embed_batch_size=10,
    )
    generator = build_client(args.generator)
    judge = build_client(args.judge)
    indices = load_indices()

    with open(args.benchmark, "r", encoding="utf-8") as benchmark_file:
        items = json.load(benchmark_file)
    if args.limit:
        items = items[: args.limit]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    completed = set()
    if os.path.exists(args.output):
        old = pd.read_csv(args.output)
        completed = {
            (int(row.item_id), row.strategy)
            for row in old.itertuples()
        }
        print(f"Resuming with {len(completed)} completed rows")

    total = len(items) * len(args.strategies)
    for item_index, item in enumerate(items, start=1):
        for strategy in args.strategies:
            key = (int(item["item_id"]), strategy)
            if key in completed:
                continue
            for attempt in range(1, 4):
                try:
                    result = run_pipeline(
                        item,
                        strategy,
                        indices,
                        generator,
                        judge,
                    )
                    pd.DataFrame([result]).to_csv(
                        args.output,
                        mode="a",
                        header=not os.path.exists(args.output),
                        index=False,
                        encoding="utf-8-sig",
                    )
                    print(
                        f"[{item_index}/{len(items)}] {strategy:<24} "
                        f"raw={result['raw_hallucination']} "
                        f"passed={result['passed_hallucination']} "
                        f"decision={result['decision']:<7} "
                        f"rouge={result['rouge_l_f1']:.3f}"
                    )
                    break
                except Exception as error:
                    print(
                        f"[ERROR {attempt}/3] item={item['item_id']} "
                        f"strategy={strategy}: "
                        f"{type(error).__name__}: {error}"
                    )
                    if attempt < 3:
                        time.sleep(5 * attempt)
            else:
                print(
                    f"[SKIP] item={item['item_id']} strategy={strategy} "
                    "failed after three attempts"
                )

    results = pd.read_csv(args.output)
    summary = summarize(results)
    summary_output = os.path.splitext(args.output)[0] + "_summary.csv"
    summary.to_csv(summary_output, index=False, encoding="utf-8-sig")
    print("\n" + summary.round(4).to_string(index=False))
    print(f"Results: {args.output}")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
