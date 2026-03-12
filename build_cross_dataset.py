import json
import random
from datasets import load_dataset

OUTPUT_FILE = "json/cross_dataset_bench.json"


def build_cross_benchmark():
    print("🚀 开始构建 OOD (Out-of-Distribution) 跨数据集泛化测试集...")
    cross_data = []

    # ==========================================
    # 1. 抽取 80 道 HotpotQA (多跳推理)
    # ==========================================
    print("⏳ 正在从 HuggingFace 下载/加载 HotpotQA (distractor 验证集)...")
    try:
        # 使用 distractor 验证集，这最考验 RAG 系统的文档筛选和多跳推理能力
        hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
        hotpot_samples = random.sample(list(hotpot), 80)

        for item in hotpot_samples:
            # HotpotQA 的 context 是个嵌套列表，我们把它展平作为参考 target_source
            context_str = " ".join(["".join(c[1]) for c in item['context']])
            cross_data.append({
                "category": "Multi-hop (HotpotQA)",
                "difficulty": "hard",
                "question": item['question'],
                "context": context_str,  # 用于后续的 Retriever 命中率比对或参考
                "gold_answer": item['answer'],
                "source_file": "External_HotpotQA"
            })
        print("✅ 成功抽取 80 道 HotpotQA 题目！")
    except Exception as e:
        print(f"❌ HotpotQA 获取失败: {e}")

    # ==========================================
    # 2. 抽取 40 道 TruthfulQA (幻觉压力测试)
    # ==========================================
    print("⏳ 正在从 HuggingFace 下载/加载 TruthfulQA (generation 验证集)...")
    try:
        # TruthfulQA 专门用来测试大模型是不是会胡说八道或者盲从人类偏见
        truthful = load_dataset("truthful_qa", "generation", split="validation")
        truthful_samples = random.sample(list(truthful), 40)

        for item in truthful_samples:
            cross_data.append({
                "category": "Adversarial (TruthfulQA)",
                "difficulty": "hard",
                "question": item['question'],
                "context": "",  # TruthfulQA 本身是用来测生成幻觉的，通常无特定上下文
                "gold_answer": item['best_answer'],
                "source_file": "External_TruthfulQA"
            })
        print("✅ 成功抽取 40 道 TruthfulQA 题目！")
    except Exception as e:
        print(f"❌ TruthfulQA 获取失败: {e}")

    # ==========================================
    # 3. 保存并输出
    # ==========================================
    if cross_data:
        # 打乱顺序，模拟真实的随机提问场景
        random.shuffle(cross_data)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(cross_data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 跨数据集测试集构建完成！共 {len(cross_data)} 题，已保存至 '{OUTPUT_FILE}'。")
    else:
        print("\n⚠️ 数据集构建失败，请检查网络连接或数据集名称。")


if __name__ == "__main__":
    # 固定随机种子，保证实验的可复现性
    random.seed(42)
    build_cross_benchmark()