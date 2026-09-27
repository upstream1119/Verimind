"""Offline rank-only RRF, with original parent-evidence matching and no API calls."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision/experiments"))
from supplement_20260927 import Runtime, normalize_evidence, ranking_metrics

SOURCE = ROOT / "revision/artifacts/supplement_20260927_full"
ORACLE = ROOT / "revision/artifacts/oracle_granularity_20260927"
ROUTES = ("DETAIL", "CONCEPT", "CONTEXT")
ROUTE_NUM = {"DETAIL": 128, "CONCEPT": 512, "CONTEXT": 1024}
K_RRF = 60


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import chromadb
    from chromadb.config import Settings
    items = {x["uid"]: x for x in json.loads((SOURCE / "items.json").read_text(encoding="utf-8")) if x["dataset"] == "qasper"}
    frozen = {x["uid"]: x for x in map(json.loads, (SOURCE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()) if x["dataset"] == "qasper"}
    oracle = {x["uid"]: x for x in map(json.loads, (ORACLE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(items) == len(frozen) == len(oracle) == 300
    parents = {}
    for route in ROUTES:
        path = ROOT / f"revision/qasper/chroma_db_qasper_{ROUTE_NUM[route]}"
        collection = chromadb.PersistentClient(str(path), settings=Settings(anonymized_telemetry=False)).get_collection(f"qasper_{ROUTE_NUM[route]}")
        parents[route] = Runtime.parent_documents(collection)
        print(f"loaded parent evidence for {route}", flush=True)
    rows = []
    raw_path = OUT / "fusion_retrieval.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_out:
        for number, uid in enumerate(sorted(items), 1):
            item, orig, complete = items[uid], frozen[uid]["results"], oracle[uid]["results"]
            evidence = list(dict.fromkeys(normalize_evidence(e) for e in item.get("evidence", []) if e.strip()))
            gold = set(range(len(evidence)))
            score_groups = defaultdict(list)
            canonical_groups = defaultdict(list)
            route_metrics = {}
            for route in ROUTES:
                saved = next((v for v in orig.values() if v["route"] == route), None)
                nodes = saved["nodes"] if saved else complete[route]["nodes"]
                assert len(nodes) == 5
                matched = {pid: {i for i, gold_text in enumerate(evidence) if gold_text in text}
                           for pid, text in parents[route].get(item["paper_id"], {}).items()}
                ranked_units = []
                for rank, node in enumerate(nodes, 1):
                    assert node["rank"] == rank
                    units = matched.get(node["parent_id"], set())
                    ranked_units.append(units)
                    if "gold_units" in node:
                        assert units == set(node["gold_units"]), (uid, route, rank)
                    key = node["node_id"]  # Exact tracked ID only; no cross-granularity semantic merging.
                    score_groups[key].append({"route": route, "route_rank": rank, "parent_id": node["parent_id"],
                                              "gold_units": sorted(units)})
                    parent_text = parents[route][item["paper_id"]][node["parent_id"]]
                    canonical_key = item["paper_id"] + ":" + hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
                    canonical_groups[canonical_key].append({"route": route, "route_rank": rank,
                                                              "parent_id": node["parent_id"], "gold_units": sorted(units)})
                route_metrics[route] = ranking_metrics(ranked_units, gold)
                for k in (3, 5):
                    assert route_metrics[route][f"recall_at_{k}"] == complete[route][f"recall_at_{k}"], (uid, route, k)
            assert route_metrics["CONCEPT"]["recall_at_3"] == orig["fixed_512"]["recall_at_3"]
            assert route_metrics["CONCEPT"]["recall_at_5"] == orig["fixed_512"]["recall_at_5"]
            def fuse(groups):
                scored = []
                for key, appearances in groups.items():
                    # A source paragraph can have multiple child chunks in the same route.
                    by_route = {}
                    for appearance in appearances:
                        route = appearance["route"]
                        if route not in by_route or appearance["route_rank"] < by_route[route]["route_rank"]:
                            by_route[route] = appearance
                    score = sum(1 / (K_RRF + x["route_rank"]) for x in by_route.values())
                    preferred = min(by_route.values(), key=lambda x: (x["route_rank"], ROUTES.index(x["route"])))
                    assert all(x["gold_units"] == preferred["gold_units"] for x in appearances)
                    scored.append({"node_id": key, "rrf_score": score, "best_rank": preferred["route_rank"],
                                   "tie_route": preferred["route"], "appearances": appearances,
                                   "gold_units": preferred["gold_units"]})
                # Deterministic tie order is independent of gold.
                scored.sort(key=lambda x: (-x["rrf_score"], ROUTES.index(x["tie_route"]), x["best_rank"], x["node_id"]))
                return scored, ranking_metrics([set(x["gold_units"]) for x in scored[:5]], gold)
            scored, fusion_exact = fuse(score_groups)
            canonical_scored, fusion = fuse(canonical_groups)
            router = orig["multi"]["route"]
            subset = orig["multi"]["gold_units_unmapped"] == 0
            record = {"uid": uid, "router_choice": router, "text_retrievable": subset,
                      "gold_units_n": len(gold), "k_rrf": K_RRF,
                      "ranked_exact_node_ids": [{**{k: x[k] for k in ("node_id", "rrf_score", "appearances", "gold_units")}, "fusion_rank": rank}
                                                for rank, x in enumerate(scored, 1)],
                      "ranked_canonical_parent_spans": [{**{k: x[k] for k in ("node_id", "rrf_score", "appearances", "gold_units")}, "fusion_rank": rank}
                                                        for rank, x in enumerate(canonical_scored, 1)],
                      "fusion_exact_node_metrics": fusion_exact, "fusion_canonical_parent_metrics": fusion,
                      "route_metrics": route_metrics}
            raw_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            row = {"question_id": uid, "text_retrievable": subset, "router_choice": router,
                   "gold_units_n": len(gold), "unique_node_ids": len(scored),
                   "cross_route_exact_id_duplicates": 15-len(scored),
                   "unique_canonical_parent_spans": len(canonical_scored),
                   "canonical_parent_span_duplicate_chunks": 15-len(canonical_scored)}
            for k in (3, 5):
                for route in ROUTES:
                    row[f"recall_{ROUTE_NUM[route]}_at{k}"] = route_metrics[route][f"recall_at_{k}"]
                row[f"router_recall_at{k}"] = orig["multi"][f"recall_at_{k}"]
                row[f"fusion_recall_at{k}"] = fusion[f"recall_at_{k}"]
                row[f"fusion_exact_node_recall_at{k}"] = fusion_exact[f"recall_at_{k}"]
                row[f"oracle_recall_at{k}"] = max(route_metrics[route][f"recall_at_{k}"] for route in ROUTES) if gold else None
            rows.append(row)
            if number % 50 == 0:
                print(f"fused {number}/300", flush=True)
    write_csv(OUT / "fusion_per_question.csv", rows)
    summaries = []
    for label, sample in (("all", rows), ("text_retrievable", [r for r in rows if r["text_retrievable"]])):
        for k in (3, 5):
            valid = [r for r in sample if r[f"recall_512_at{k}"] is not None]
            summaries.append({"subset": label, "k": k, "questions": len(sample), "evaluable": len(valid),
                              **{name: sum(r[f"{name}_recall_at{k}"] for r in valid)/len(valid)
                                 for name in ("router", "fusion", "fusion_exact_node", "oracle")},
                              **{str(size): sum(r[f"recall_{size}_at{k}"] for r in valid)/len(valid)
                                 for size in ROUTE_NUM.values()}})
    write_csv(OUT / "fusion_summary.csv", summaries)
    (OUT / "fusion_summary.json").write_text(json.dumps({"k_rrf": K_RRF,
        "primary_dedup": "paper_id + SHA256(full normalized reconstructed source parent paragraph)",
        "secondary_dedup": "exact node_id only", "semantic_cross_granularity_dedup": False,
        "tie_order": [128, 512, 1024],
        "source_question_n": len(rows), "rows": summaries}, indent=2), encoding="utf-8")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
