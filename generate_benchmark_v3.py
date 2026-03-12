import os
import json
import random
import re
import hashlib
from llama_index.llms.dashscope import DashScope

# =========================
# ⚙️ 配置
# =========================

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

MD_DIR = "./knowledge_mds"
OUTPUT_FILE = "json/verimind_bench_v3.json"

llm = DashScope(
    model_name="qwen-plus",
    api_key="sk-01bcaa48e91445c2bdeeed0aaeff8855",
    max_tokens=2500
)

TARGET_QUOTAS = {
    "Reasoning": 120,
    "Factual": 80,
    "Multi-hop": 60,
    "Adversarial": 40
}

collected_data = {
    "Reasoning": [],
    "Factual": [],
    "Multi-hop": [],
    "Adversarial": []
}

seen_questions = set()


# =========================
# 📚 Section Chunking
# =========================

def split_sections(md_text):

    pattern = r'(#+\s.*)'
    parts = re.split(pattern, md_text)

    sections = []

    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""

        text = header + body

        if len(text) > 800:
            sections.append(text)

    return sections


# =========================
# 📦 Extract chunk
# =========================

def extract_chunk(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    sections = split_sections(text)

    if not sections:
        return None

    section = random.choice(sections)

    if len(section) > 2000:
        start = random.randint(0, len(section) - 2000)
        section = section[start:start+2000]

    return section


# =========================
# 🧠 QA Generation
# =========================

def generate_qa_pairs(chunk, file_name, needed_categories):

    category_instructions = ""

    for cat in needed_categories:

        if cat == "Factual":
            category_instructions += "- Factual: Ask about a specific fact, metric, dataset, or method mentioned in the text.\n"

        if cat == "Reasoning":
            category_instructions += "- Reasoning: Ask a why/how question requiring logical reasoning from the text.\n"

        if cat == "Multi-hop":
            category_instructions += "- Multi-hop: Require combining information from multiple sentences.\n"

        if cat == "Adversarial":
            category_instructions += "- Adversarial: Ask about a dataset, metric, or author NOT mentioned in the text.\n"

    prompt = f"""
You are creating an academic QA benchmark.

Generate exactly {len(needed_categories)} QA pairs based on the text.

Categories required:
{needed_categories}

Instructions:
{category_instructions}

Rules:

- Use ONLY the provided text.
- For adversarial questions, ask about something not present.
- The answer must clearly state it is not in the text.
- Extract the exact answer span when possible.

Return valid JSON array.

Each item must contain:

category
difficulty
question
context
gold_answer
answer_span

TEXT:
{chunk}
"""

    try:

        response = llm.complete(prompt).text.strip()

        match = re.search(r'\[.*\]', response, re.S)

        if not match:
            return []

        return json.loads(match.group())

    except Exception as e:

        print("LLM error:", e)
        return []


# =========================
# 🔎 QA Validation
# =========================

def validate_qa(qa):

    question = qa.get("question", "").strip()

    if len(question) < 10:
        return False

    q_hash = hashlib.md5(question.encode()).hexdigest()

    if q_hash in seen_questions:
        return False

    seen_questions.add(q_hash)

    return True


# =========================
# 🎯 Remaining categories
# =========================

def get_remaining():

    return [
        c for c in TARGET_QUOTAS
        if len(collected_data[c]) < TARGET_QUOTAS[c]
    ]


# =========================
# 🚀 Main pipeline
# =========================

def build_benchmark():

    md_files = []

    for root, _, files in os.walk(MD_DIR):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))

    print("Found papers:", len(md_files))

    while True:

        needed = get_remaining()

        if not needed:
            break

        file = random.choice(md_files)

        chunk = extract_chunk(file)

        if not chunk:
            continue

        qas = generate_qa_pairs(chunk, os.path.basename(file), needed)

        for qa in qas:

            cat = qa.get("category")

            if cat not in needed:
                continue

            if not validate_qa(qa):
                continue

            qa["source_file"] = os.path.basename(file)

            collected_data[cat].append(qa)

            print(
                f"Collected {cat}:",
                len(collected_data[cat]),
                "/",
                TARGET_QUOTAS[cat]
            )

    final_data = []

    for v in collected_data.values():
        final_data.extend(v)

    random.shuffle(final_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print("Benchmark created:", len(final_data))


# =========================

if __name__ == "__main__":
    build_benchmark()