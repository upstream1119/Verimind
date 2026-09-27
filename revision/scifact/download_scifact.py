"""
download_scifact.py
SciFact 数据集下载 & 预处理脚本
将 SciFact claims + corpus 转为与 verimind_bench_v3.json 对齐的格式
输出: json/scifact_bench.json
"""
import json
import os
import sys
import urllib.request
import tarfile
from collections import Counter

# Windows GBK 兼容：强制 stdout 使用 utf-8
sys.stdout.reconfigure(encoding="utf-8")

# ================= 配置 =================
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
DATA_DIR = os.path.join(RAW_DIR, "data")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "json", "scifact_bench.json")
S3_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
# ========================================


def download_and_extract():
    """从 S3 下载 SciFact 原始数据并解压"""
    os.makedirs(RAW_DIR, exist_ok=True)
    tar_path = os.path.join(RAW_DIR, "data.tar.gz")

    if not os.path.exists(tar_path):
        print(f"📥 正在从 S3 下载 SciFact 数据集...")
        req = urllib.request.Request(S3_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            with open(tar_path, "wb") as f:
                f.write(data)
            print(f"✅ 下载完成: {len(data)} bytes")
    else:
        print(f"✅ 已存在: {tar_path}")

    if not os.path.exists(os.path.join(DATA_DIR, "corpus.jsonl")):
        print("📦 解压中...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(RAW_DIR)
        print("✅ 解压完成")
    else:
        print("✅ 数据已解压")


def load_corpus():
    """加载 corpus: doc_id -> {title, abstract(list of sentences)}"""
    corpus = {}
    with open(os.path.join(DATA_DIR, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["doc_id"]] = doc
    print(f"📚 加载 corpus: {len(corpus)} 篇文摘")
    return corpus


def load_claims():
    """加载 dev + train claims, dev 优先"""
    claims_dev = []
    with open(os.path.join(DATA_DIR, "claims_dev.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            c["_split"] = "dev"
            claims_dev.append(c)

    claims_train = []
    with open(os.path.join(DATA_DIR, "claims_train.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            c["_split"] = "train"
            claims_train.append(c)

    print(f"📋 加载 claims: dev={len(claims_dev)}, train={len(claims_train)}")
    return claims_dev, claims_train


def convert_to_bench(claims, corpus):
    """
    将有 evidence 的 claims 转为 verimind_bench 格式
    输出字段: question, gold_answer, context, category, difficulty, source_file
    """
    bench = []
    skipped = 0

    for claim in claims:
        if not claim["evidence"]:
            skipped += 1
            continue

        # 一条 claim 可能有多个 evidence doc，逐个展开
        for doc_id_str, evidence_sets in claim["evidence"].items():
            doc_id = int(doc_id_str)
            doc = corpus.get(doc_id)
            if doc is None:
                skipped += 1
                continue

            # 拼接 evidence 句子作为 gold_answer
            all_evidence_sents = []
            label = "SUPPORT"  # 默认
            for ev in evidence_sets:
                sents = [doc["abstract"][i] for i in ev["sentences"] if i < len(doc["abstract"])]
                all_evidence_sents.extend(sents)
                label = ev["label"]

            if not all_evidence_sents:
                skipped += 1
                continue

            # 全文摘作为 context（后续实验中会被检索替换）
            full_abstract = " ".join(doc["abstract"])

            bench.append({
                "question": claim["claim"],
                "gold_answer": " ".join(all_evidence_sents),
                "context": full_abstract,
                "category": "SciFact",
                "difficulty": label,  # SUPPORT 或 CONTRADICT
                "source_file": str(doc_id),
                "_claim_id": claim["id"],
                "_split": claim["_split"],
            })

    print(f"⚠️ 跳过 {skipped} 条无效数据")
    return bench


def main():
    # Step 1: 下载数据
    download_and_extract()

    # Step 2: 加载
    corpus = load_corpus()
    claims_dev, claims_train = load_claims()

    # Step 3: 转换（dev 优先，再补 train）
    bench_dev = convert_to_bench(claims_dev, corpus)
    bench_train = convert_to_bench(claims_train, corpus)

    print(f"\n有 evidence 的条目: dev={len(bench_dev)}, train={len(bench_train)}")

    # 优先用 dev，再从 train 补充到 ~500 条
    bench = bench_dev.copy()
    remaining = 500 - len(bench)
    if remaining > 0:
        bench.extend(bench_train[:remaining])
    print(f"最终选取: {len(bench)} 条")

    # Step 4: 统计
    label_dist = Counter(item["difficulty"] for item in bench)
    split_dist = Counter(item["_split"] for item in bench)
    print(f"标签分布: {dict(label_dist)}")
    print(f"来源分布: {dict(split_dist)}")

    # Step 5: 清理内部字段并保存
    for item in bench:
        del item["_claim_id"]
        del item["_split"]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存至 {OUTPUT_FILE}")
    print(f"   共 {len(bench)} 条，可直接用于 SciFact 实验")


if __name__ == "__main__":
    main()
