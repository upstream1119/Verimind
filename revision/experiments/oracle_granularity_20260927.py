"""Complete QASPER three-granularity oracle; resumable, preserving raw query results."""
import json
import math
import sys
from pathlib import Path

from supplement_20260927 import ROOT, Runtime, normalize_evidence, ranking_metrics

SOURCE = ROOT / "revision/artifacts/supplement_20260927_full"
OUTPUT = ROOT / "revision/artifacts/oracle_granularity_20260927"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "retrieval.jsonl"
    existing = {r["uid"]: r for r in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines())} if raw_path.exists() else {}
    frozen = {r["uid"]: r for r in map(json.loads, (SOURCE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    items = [r for r in json.loads((SOURCE / "items.json").read_text(encoding="utf-8")) if r["dataset"] == "qasper"]
    runtime = Runtime()
    for number, item in enumerate(items, 1):
        if item["uid"] in existing:
            continue
        known = {r["route"]: r for r in frozen[item["uid"]]["results"].values()}
        vector = runtime.embedding.get_query_embedding(item["question"])
        results = {}
        for route in ("DETAIL", "CONCEPT", "CONTEXT"):
            if route in known:
                results[route] = {k: known[route][k] for k in ("recall_at_3", "recall_at_5", "gold_units_n", "gold_units_unmapped")}
                continue
            collection = runtime.collections["qasper", route]
            raw = collection.query(query_embeddings=[vector], n_results=5,
                                   include=["documents", "metadatas", "distances"],
                                   where={"paper_id": item["paper_id"]})
            parents = runtime.parents[route].get(item["paper_id"], {})
            evidence = list(dict.fromkeys(normalize_evidence(e) for e in item.get("evidence", []) if e.strip()))
            matches = {parent: {i for i, gold in enumerate(evidence) if gold in text} for parent, text in parents.items()}
            ranked = []
            for rank, (node_id, metadata, distance) in enumerate(zip(raw["ids"][0], raw["metadatas"][0], raw["distances"][0]), 1):
                parent = json.loads(metadata["_node_content"])["relationships"]["1"]["node_id"]
                ranked.append({"rank": rank, "node_id": node_id, "parent_id": parent,
                               "score": math.exp(-distance), "gold_units": sorted(matches.get(parent, set()))})
            metrics = ranking_metrics([set(x["gold_units"]) for x in ranked], set(range(len(evidence))))
            results[route] = {**metrics, "gold_units_n": len(evidence),
                              "gold_units_unmapped": len(set(range(len(evidence))) - set().union(*matches.values()) if matches else set(range(len(evidence)))),
                              "nodes": ranked}
        record = {"uid": item["uid"], "dataset": "qasper", "results": results}
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing[item["uid"]] = record
        if number % 25 == 0:
            print(f"completed {len(existing)}/{len(items)}", flush=True)
    assert len(existing) == len(items)
    summary = []
    for label, subset in (("all", items), ("text_retrievable", [i for i in items if frozen[i["uid"]]["results"]["multi"]["gold_units_unmapped"] == 0])):
        for k in (3, 5):
            records = [existing[i["uid"]]["results"] for i in subset if existing[i["uid"]]["results"]["CONCEPT"][f"recall_at_{k}"] is not None]
            summary.append({"subset": label, "k": k, "questions": len(subset), "evaluable": len(records),
                            **{route.lower(): sum(r[route][f"recall_at_{k}"] for r in records)/len(records) for route in ("DETAIL", "CONCEPT", "CONTEXT")},
                            "oracle": sum(max(r[route][f"recall_at_{k}"] for route in ("DETAIL", "CONCEPT", "CONTEXT")) for r in records)/len(records)})
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
