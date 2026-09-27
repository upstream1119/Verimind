"""Export ranked retrieval evidence without generating answers."""
import argparse
import json
import os
from pathlib import Path

import run_qasper_experiment as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    items = json.loads(
        Path(runner.BENCHMARK_FILE).read_text(encoding="utf-8")
    )
    if args.limit is not None:
        items = items[:args.limit]
    runner.Settings.embed_model = runner.DashScopeEmbedding(
        model_name="text-embedding-v3",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        embed_batch_size=10,
    )
    indices = runner.load_indices()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for item in items:
            for strategy in ("fixed_512", "dynamic"):
                route = (
                    "CONCEPT" if strategy == "fixed_512"
                    else runner.get_query_intent(item["question"])
                )
                filters = runner.MetadataFilters(filters=[
                    runner.MetadataFilter(
                        key="paper_id",
                        operator=runner.FilterOperator.EQ,
                        value=str(item["paper_id"]),
                    )
                ])
                nodes = indices[route].as_retriever(
                    similarity_top_k=5, filters=filters
                ).retrieve(item["question"])
                record = {
                    "question_id": item["question_id"],
                    "strategy": strategy,
                    "route": route,
                    "nodes": [
                        {
                            "rank": rank,
                            "node_id": node.node.node_id,
                            "score": node.score,
                            "text": node.text,
                            "metadata": node.node.metadata,
                        }
                        for rank, node in enumerate(nodes, start=1)
                    ],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()


if __name__ == "__main__":
    main()
