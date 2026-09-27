import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from llama_index.core import PromptTemplate, VectorStoreIndex

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from revision.cross_arch.llm_clients import LLMClient, build_client
from run_sci_experiment_main import (
    TOP_K,
    compute_exact_match,
    compute_f1,
    crag_evaluate,
    crag_retrieve_secondary,
    crag_rewrite,
    estimate_cost,
    generate_hyde_doc,
    get_query_intent,
    load_indices,
    normalize_source_id,
)

DEFAULT_INPUT_JSON = os.path.join(PROJECT_ROOT, "json", "verimind_bench_v3.json")
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "revision", "artifacts", "1_3_cross_arch", "cross_arch_run.csv")
DEFAULT_STRATEGIES = ["Vanilla_RAG", "HyDE", "VeriMind_Full"]


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    val = (v or "").strip().lower()
    if val in {"1", "true", "yes", "y"}:
        return True
    if val in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-architecture experiment with injectable generator/judge.")
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--generator", default="qwen", choices=["qwen", "glm", "deepseek"])
    parser.add_argument("--judge", default="qwen", choices=["qwen", "glm", "deepseek"])
    parser.add_argument("--generator_model", default="", help="Optional model override for generator provider.")
    parser.add_argument("--judge_model", default="", help="Optional model override for judge provider.")
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_prompt(strategy: str, context_str: str, query: str) -> str:
    if strategy in ["VeriMind_Full", "VeriMind_No_Routing", "Ablation_no_structure", "CRAG_Local"]:
        tpl = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information and NOT prior knowledge, answer the query concisely and directly. "
            "If the exact answer cannot be found, you MUST explicitly output exactly: 'Information not found.' "
            "Do NOT use parametric memory or guess.\nQuery: {query_str}\nAnswer: "
        )
    elif strategy == "Self-RAG":
        tpl = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Critique the context. If irrelevant, say 'Information not found'. "
            "Otherwise answer the query concisely using ONLY the context.\nQuery: {query_str}\nAnswer: "
        )
    elif strategy == "CoT_RAG":
        tpl = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Please think step by step about how the context relates to the query, then provide the final answer. "
            "Use ONLY the context. If insufficient, output exactly 'Information not found.'\n"
            "Query: {query_str}\nAnswer: "
        )
    else:
        tpl = PromptTemplate(
            "Context information is below.\n---------------------\n{context_str}\n---------------------\n"
            "Given the context information, answer the query concisely and directly.\nQuery: {query_str}\nAnswer: "
        )
    return tpl.format(context_str=context_str, query_str=query)


def run_pipeline_cross_arch(
    qa_item: Dict,
    question_id: int,
    indices: Dict[str, VectorStoreIndex],
    strategy: str,
    generator_name: str,
    judge_name: str,
    generator_client: Optional[LLMClient],
    judge_client: Optional[LLMClient],
    dry_run: bool = False,
) -> Dict:
    query = qa_item.get("question", "")
    target_source = str(qa_item.get("source_file", ""))
    gold_answer = qa_item.get("gold_answer", "")
    start_time = time.time()

    intent = "CONCEPT"
    retrieval_path = "DRY_RUN" if dry_run else "DEFAULT"
    retrieval_hit = 0
    crag_score = -1.0
    calibrated_ret = 0.0
    context_str = "[dry_run context]"
    answer = "[dry_run generated answer]"
    judge_status = "dry_run"

    if not dry_run:
        search_query = query
        top_k_val = TOP_K
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
        nodes = index.as_retriever(similarity_top_k=top_k_val).retrieve(search_query)
        retrieved_sources = [normalize_source_id(getattr(n, "metadata", {}) or {}) for n in nodes]
        retrieval_hit = 1 if target_source and any(target_source in src for src in retrieved_sources) else 0
        context_str = "\n".join([n.text for n in nodes if getattr(n, "text", "")])
        avg_score = sum(getattr(n, "score", 0.0) for n in nodes) / len(nodes) if nodes else 0.0
        calibrated_ret = min(10.0, max(0.0, (avg_score - 0.3) * 20))

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

        prompt = build_prompt(strategy, context_str, query)
        answer = generator_client.complete(prompt, max_tokens=1500, temperature=0.01)
        judge_score = judge_client.score_faithfulness(answer, context_str)
        faith_score = float(judge_score)
        hallucination_rate = 1.0 if judge_score < 7 else 0.0
        judge_status = "ok"
    else:
        faith_score = 10.0
        hallucination_rate = 0.0

    trust_score = 0.0
    if strategy in ["VeriMind_Full", "VeriMind_No_Routing", "Ablation_no_structure"]:
        trust_score = 0.4 * calibrated_ret + 0.6 * faith_score
        if trust_score < 5.0:
            answer = "REJECTED: Information not found."
            faith_score, hallucination_rate = 10.0, 0.0
            if judge_status == "ok":
                judge_status = "gated_reject"

    precision, recall, f1 = compute_f1(gold_answer, answer)
    em = compute_exact_match(gold_answer, answer)
    latency = time.time() - start_time
    cost = estimate_cost(len((context_str or "") + query) / 4, len(answer) / 4)

    return {
        "question_id": question_id,
        "question": query,
        "strategy": strategy,
        "generator": generator_name,
        "judge": judge_name,
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
        "judge_status": judge_status,
        "judge_context": (context_str or "")[:4000],
    }


def main() -> None:
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    with open(args.input_json, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.limit and args.limit > 0:
        qa_data = qa_data[: args.limit]

    indices = {} if args.dry_run else load_indices()
    generator_client = None if args.dry_run else build_client(
        args.generator,
        model_name_override=(args.generator_model or None),
    )
    judge_client = None if args.dry_run else build_client(
        args.judge,
        model_name_override=(args.judge_model or None),
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    completed = set()
    if args.resume and os.path.exists(args.output_csv):
        old = pd.read_csv(args.output_csv)
        if len(old) > 0:
            if "generator" not in old.columns or "judge" not in old.columns:
                raise ValueError("Existing CSV missing generator/judge columns; cannot resume safely.")
            gens = sorted(set(str(v) for v in old["generator"].dropna().unique()))
            judges = sorted(set(str(v) for v in old["judge"].dropna().unique()))
            if gens != [args.generator] or judges != [args.judge]:
                raise ValueError(
                    f"Resume blocked: existing CSV is for generator={gens}, judge={judges}, "
                    f"but current run is generator={args.generator}, judge={args.judge}."
                )
            for _, row in old.iterrows():
                completed.add((int(row["question_id"]), str(row["strategy"])))
            print(f"Resume enabled: skip {len(completed)} completed (question_id, strategy) items.")

    print(
        f"Start run: questions={len(qa_data)}, strategies={len(strategies)}, "
        f"generator={args.generator}, judge={args.judge}, "
        f"generator_model={args.generator_model or '(default)'}, "
        f"judge_model={args.judge_model or '(default)'}, dry_run={args.dry_run}"
    )

    for i, qa in enumerate(qa_data):
        print(f"[{i + 1}/{len(qa_data)}] {str(qa.get('question', ''))[:60]}...")
        for strategy in strategies:
            key = (i, strategy)
            if key in completed:
                continue

            for attempt in range(3):
                try:
                    out = run_pipeline_cross_arch(
                        qa_item=qa,
                        question_id=i,
                        indices=indices,
                        strategy=strategy,
                        generator_name=args.generator,
                        judge_name=args.judge,
                        generator_client=generator_client,
                        judge_client=judge_client,
                        dry_run=args.dry_run,
                    )
                    pd.DataFrame([out]).to_csv(
                        args.output_csv,
                        mode="a",
                        header=not os.path.exists(args.output_csv),
                        index=False,
                        encoding="utf-8-sig",
                        escapechar="\\",
                    )
                    print(
                        f"  └─ {strategy:<18} | faith={out['faith_score']:.1f} "
                        f"| hal={out['hallucination_rate']:.0f} | {out['judge_status']}"
                    )
                    break
                except Exception as err:
                    if attempt < 2:
                        print(f"  ⚠️ {strategy} failed ({type(err).__name__}), retry {attempt + 1}/3")
                        time.sleep(3 * (attempt + 1))
                    else:
                        print(f"  ❌ {strategy} failed after retries: {err}")

    print(f"Done. Output -> {args.output_csv}")


if __name__ == "__main__":
    main()
