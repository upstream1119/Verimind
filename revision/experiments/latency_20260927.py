"""Independent endpoint timing on a fixed stratified sample, without offline evaluation."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time

from supplement_20260927 import Runtime, decision, digest, dump_csv, is_refusal, load_items, percentile, read_jsonl, route_query


def run_one(runtime, item, policy):
    started = time.perf_counter()
    dataset = "qasper" if item["dataset"] == "qasper" else "scifact"
    route = "CONCEPT" if policy == "fixed_512" else route_query(item["question"], dataset)
    t0 = time.perf_counter()
    vector = runtime.embedding.get_query_embedding(item["question"])
    embedding_sec = time.perf_counter() - t0
    kwargs = {"query_embeddings": [vector], "n_results": 3, "include": ["documents", "distances"]}
    if dataset == "qasper":
        kwargs["where"] = {"paper_id": item["paper_id"]}
    t0 = time.perf_counter()
    raw = runtime.collections[dataset, route].query(**kwargs)
    query_sec = time.perf_counter() - t0
    context = "\n\n".join(raw["documents"][0])
    if dataset == "qasper":
        task = "Answer the research-paper question using ONLY the retrieved context. Give a concise direct answer."
    else:
        task = "Verify the scientific claim using ONLY the retrieved context. Begin with SUPPORT or CONTRADICT and give one concise evidence-based explanation."
    prompt = (task + " Do not add facts absent from the context. If evidence is insufficient, output exactly 'Information not found.'\n\n"
              f"[Retrieved Context]\n{context}\n\n[Question or Claim]\n{item['question']}\n\n[Answer]\n")
    clients = runtime.clients()
    t0 = time.perf_counter()
    answer = clients["glm"].complete(prompt, max_tokens=400, temperature=.01)
    generation_sec = time.perf_counter() - t0
    gate_sec = 0.0
    faith = 10
    gate_called = False
    if policy == "full" and not is_refusal(answer):
        gate_called = True
        prompt = ("Evaluate if the answer is strictly derived from the context. Output ONLY an integer 0 to 10; 10 means completely faithful.\n"
                  f"[Context]\n{context}\n[Answer]\n{answer}\nScore:")
        t0 = time.perf_counter()
        text = clients["deepseek"].complete(prompt, max_tokens=16, temperature=.01)
        gate_sec = time.perf_counter() - t0
        if not re.fullmatch(r"\s*(10|[0-9])\s*", text):
            raise ValueError("Invalid gate score")
        faith = int(text.strip())
    scores = [math.exp(-d) for d in raw["distances"][0]]
    calibrated = min(10., max(0., ((statistics.mean(scores) if scores else 0.) - .3) * 20))
    outcome = decision(answer, calibrated, faith, policy)
    elapsed = time.perf_counter() - started
    return {"uid": item["uid"], "dataset": dataset, "policy": policy, "route": route,
            "latency_sec": elapsed, "embedding_sec": embedding_sec, "query_sec": query_sec,
            "generation_sec": generation_sec, "gate_sec": gate_sec, "gate_called": gate_called,
            "decision": outcome, "answer": answer, "context_sha256": digest(context),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def choose_sample(n):
    selected = []
    all_items = load_items()
    for dataset in ("qasper", "scifact"):
        rows = [r for r in all_items if r["dataset"] == dataset]
        groups = defaultdict(list)
        for item in rows:
            label = str(item["is_unanswerable"]) if dataset == "qasper" else "/".join(item["gold_labels"])
            groups[label].append(item)
        chosen = []
        for index, (label, members) in enumerate(sorted(groups.items())):
            count = round(n * len(members) / len(rows)) if index < len(groups) - 1 else n - len(chosen)
            chosen.extend(sorted(members, key=lambda i: digest("20260927-latency-" + i["uid"]))[:count])
        selected.extend(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    items = choose_sample(args.per_dataset)
    manifest = {"seed": 20260927, "sample_per_dataset": args.per_dataset, "workers": args.workers,
                "uids": [i["uid"] for i in items], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "policy_order": "seeded random order per input", "top_k": 3, "offline_evaluator_called": False,
                "command": sys.argv}
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Latency resume manifest mismatch")
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runtime = Runtime()
    completed = {(r["uid"], r["policy"]): r for r in read_jsonl(output / "timings.jsonl")}
    failures = []
    lock = threading.Lock()
    def work(item):
        # Construct clients outside timed regions; no generated answer reuse.
        runtime.clients()
        policies = ["fixed_512", "multi", "full"]
        random.Random("20260927-" + item["uid"]).shuffle(policies)
        for policy in policies:
            if (item["uid"], policy) in completed:
                continue
            result = run_one(runtime, item, policy)
            with lock:
                with (output / "timings.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed[item["uid"], policy] = result
                dump_csv(output / "progress.csv", [{"completed": i + 1} for i in range(len(completed))])
                print(f"timed={len(completed)}/{3 * len(items)} dataset={item['dataset']}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, item): item for item in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                record = {"uid": futures[future]["uid"], "type": type(error).__name__, "message": str(error)[:600]}
                failures.append(record)
                with (output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
    rows = []
    for dataset in ("qasper", "scifact"):
        for policy in ("fixed_512", "multi", "full"):
            group = [r for r in completed.values() if r["dataset"] == dataset and r["policy"] == policy]
            if not group:
                continue
            rows.append({"dataset": dataset, "policy": policy, "n": len(group),
                         "mean_sec": statistics.mean(r["latency_sec"] for r in group),
                         "p95_sec": percentile([r["latency_sec"] for r in group], .95),
                         "gate_calls": sum(r["gate_called"] for r in group)})
    dump_csv(output / "summary.csv", rows)
    validation = {"complete": len(completed) == 3 * len(items), "expected_rows": 3 * len(items),
                  "actual_rows": len(completed), "failures_this_run": len(failures)}
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation), flush=True)
    if failures or not validation["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
