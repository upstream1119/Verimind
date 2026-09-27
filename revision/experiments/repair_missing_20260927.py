"""Retry only missing records with full API text diagnostics; frozen protocol unchanged."""
import argparse
import json
from pathlib import Path
import time

import supplement_20260927 as experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    items = json.loads((output / "items.json").read_text(encoding="utf-8"))
    retrievals = {r["uid"]: r for r in experiment.read_jsonl(output / "retrieval.jsonl")}
    answers = {(r["uid"], r["strategy"]): r for r in experiment.read_jsonl(output / "answers.jsonl")}
    runtime = experiment.Runtime()
    clients = runtime.clients()
    active = {}

    class LoggedClient:
        def __init__(self, provider, client):
            self.provider, self.client = provider, client

        def complete(self, prompt, **kwargs):
            # One missing evaluator record exhausted 300 tokens before closing JSON.
            # Keep model/prompt/temperature unchanged; record the larger completion cap.
            if self.provider == "qwen":
                kwargs["max_tokens"] = 1024
            start = time.perf_counter()
            response = self.client.complete(prompt, **kwargs)
            record = {**active, "provider": self.provider, "prompt": prompt, "parameters": kwargs,
                      "response": response, "seconds": time.perf_counter() - start}
            with (output / "retry_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return response

    runtime.local.clients = {p: LoggedClient(p, c) for p, c in clients.items()}
    for item in items:
        cache = {}
        for strategy in ("fixed_512", "multi"):
            retrieval = retrievals[item["uid"]]["results"][strategy]
            context_key = experiment.digest("\n\n".join(n["text"] for n in retrieval["nodes"][:3]))
            if (item["uid"], strategy) in answers:
                cache[context_key] = answers[item["uid"], strategy]
                continue
            active.update(uid=item["uid"], strategy=strategy)
            shared = context_key in cache
            answer = dict(cache[context_key]) if shared else runtime.generate_and_evaluate(item, retrieval)
            answer.update(uid=item["uid"], dataset=item["dataset"], strategy=strategy, shared_context_reuse=shared)
            with (output / "answers.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
            answers[item["uid"], strategy] = answer
            cache[context_key] = answer
            print(f"repaired={item['uid']} strategy={strategy}", flush=True)
    experiment.dump_csv(output / "progress.csv", [{"completed": i + 1} for i in range(len(items))])
    experiment.summarize(output)


if __name__ == "__main__":
    main()
