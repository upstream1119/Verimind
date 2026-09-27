"""Export a deterministic blind packet and provisional content review of 30 judge-zero cases."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE = Path("revision/artifacts/supplement_20260927_full")
INPUT = Path("revision/artifacts/four_gate_analyses_20260927/trust_030_040_cases.csv")
OUTPUT = Path("revision/artifacts/judge_zero_audit_20260927")

# Human-readable content review performed without evaluator labels on first pass.
# A fifth category is necessary: sampled judge=0 does not imply evaluator conflict.
ADJUDICATION = {
    1: ("judge_wrong", "Context reports haplogroup I increases CAD risk; answer says CONTRADICT."),
    2: ("judge_wrong", "Context explicitly links TH17 depletion to increased Salmonella dissemination."),
    3: ("judge_wrong", "Context says IFI16 promotes STING phosphorylation and translocation."),
    4: ("partial_or_ambiguous", "Trial benefit does not directly settle recommendation after inadequate response."),
    5: ("evaluation_wrong", "Neither steric hindrance nor flexible-versus-rigid comparison appears in retrieved evidence."),
    6: ("judge_wrong", "Retrieved abstract states genome size 7489 bp, contradicting 2140 bp."),
    7: ("evaluation_wrong", "Retrieved evidence does not address 1000 Genomes or penetrance comparison."),
    8: ("evaluation_wrong", "Retrieved evidence does not address the stated chemokine mechanism."),
    9: ("evaluation_wrong", "Trachoma evidence does not establish cataract as a primary cause."),
    10: ("judge_wrong", "Context describes protection against severe malarial anaemia, opposite the claim."),
    11: ("judge_wrong", "Context compares combination therapy and varenicline monotherapy."),
    12: ("partial_or_ambiguous", "Higher precancer detection does not directly establish cancer incidence trend."),
    13: ("both_agree_unsupported", "Podocyte evidence lacks injury-linked motility or migration; evaluator supported=False."),
    14: ("evaluation_wrong", "Retrieved evidence does not address insulin and severe kidney failure."),
    15: ("partial_or_ambiguous", "Antiviral ISG effect is relevant but neuronal survival is not directly measured."),
    16: ("judge_wrong", "Retrieved trial concludes CBT is superior to zopiclone for insomnia."),
    17: ("judge_wrong", "Context states acetylation restores axonal transport."),
    18: ("evaluation_wrong", "Chicken evidence alone does not settle Passeriformes claim."),
    19: ("judge_wrong", "Retrieved prevalence 493 per million contradicts 1 in 5 million."),
    20: ("partial_or_ambiguous", "Lower modeled HIV incidence supports benefit but differs from direct infectiveness measurement."),
    21: ("judge_wrong", "Context explicitly says stem cells form derivatives of all three germ layers."),
    22: ("judge_wrong", "BCL-2 removal causes cell death in c-Myc model; opposite of promoting apoptosis."),
    23: ("judge_wrong", "Reduced IL-2 signaling is linked to greater autoimmune susceptibility."),
    24: ("partial_or_ambiguous", "KLF4 targets in cholesterol-treated cells do not by themselves prove full causal chain."),
    25: ("evaluation_wrong", "Context lacks ROS inhibition of FoxO3a; evaluator reason itself says evidence insufficient."),
    26: ("judge_wrong", "Context explicitly says JMJD6 catalyzes p53 lysine hydroxylation."),
    27: ("judge_wrong", "Context calls KLF4 essential for inflammatory monocyte differentiation."),
    28: ("judge_wrong", "Context says 6PGD suppression decreases lipogenesis."),
    29: ("judge_wrong", "Retrieved evidence identifies non-genetic influences on breast cancer."),
    30: ("judge_wrong", "Retrieved study says women's heart-failure incidence did not decline; gold label disagrees."),
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    items = {x["uid"]: x for x in json.loads((SOURCE / "items.json").read_text(encoding="utf-8"))}
    retrieval = {x["uid"]: x for x in map(json.loads, (SOURCE / "retrieval.jsonl").read_text(encoding="utf-8").splitlines())}
    answers = {x["uid"]: x for x in map(json.loads, (SOURCE / "answers.jsonl").read_text(encoding="utf-8").splitlines()) if x["strategy"] == "multi"}
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        ids = [x["sample_id"] for x in csv.DictReader(f) if x["dataset"] == "SciFact+NEI"]
    chosen = sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:30]
    blind, reviewed = [], []
    for index, uid in enumerate(chosen, 1):
        item, answer, retrieved = items[uid], answers[uid], retrieval[uid]["results"]["multi"]
        packet = {"case": index, "sample_id": uid, "claim": item["question"],
                  "answer": answer["answer"], "retrieved_context": "\n\n".join(n["text"] for n in retrieved["nodes"][:3])}
        blind.append(packet)
        category, reason = ADJUDICATION[index]
        reviewed.append({"case": index, "sample_id": uid, "category": category, "content_reason": reason,
                         "judge_score": answer["faith_score"] / 10,
                         "evaluator_supported": answer["evaluation"]["supported"],
                         "evaluator_correct": answer["evaluation"]["correct"],
                         "evaluator_reason": answer["evaluation"].get("reason", ""),
                         "gold_labels": "|".join(item["gold_labels"])})
    for filename, records in (("blind_packet.json", blind), ("provisional_review.json", reviewed)):
        (OUTPUT / filename).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"population": len(ids), "reviewed": len(chosen), "selection": "SHA256(sample_id), first 30",
               "categories": dict(Counter(r["category"] for r in reviewed)),
               "evaluator_supported_true": sum(r["evaluator_supported"] for r in reviewed),
               "scope": "Provisional model-performed content review; not independent human expert annotation"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
