import pandas as pd
import json
import random

CSV_FILE = 'csv/experiment_results_sci_v3.csv'
JSON_FILE = 'json/verimind_bench_v3.json'
OUTPUT_FILE = 'csv/human_validation_50_samples.csv'


def generate_validation_sheet():
    print("🚀 开始生成人类一致性校验 (Human Validation) 抽样表...")

    # 1. 加载数据
    df = pd.read_csv(CSV_FILE)
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        bench_data = json.load(f)

    # 2. 过滤掉拒答的题目 (专门挑那些大模型生成了实质性回答的案例)
    df_valid = df[~df['generated_answer'].str.contains('Information not found|REJECTED', case=False, na=False)]

    # 3. 随机抽取 50 个样本
    # 固定随机种子，保证每次抽出来的题一样
    sampled_df = df_valid.sample(n=50, random_state=42)

    # 4. 组装发给人类专家 (你) 的表格
    validation_records = []
    for _, row in sampled_df.iterrows():
        q_id = int(row['question_id'])
        # 从原始 JSON 中拿到问题和标准答案
        question = bench_data[q_id]['question']
        gold_answer = bench_data[q_id]['gold_answer']

        validation_records.append({
            "Question ID": q_id,
            "Model": row['strategy'],
            "Question": question,
            "Gold Answer (标准答案)": gold_answer,
            "Generated Answer (模型回答)": row['generated_answer'],
            "LLM Judge (机器判定幻觉)": int(row['hallucination_rate']),  # 1是幻觉，0是没幻觉
            "Human Judge (请你填 0 或 1)": ""  # 留空给你填！
        })

    output_df = pd.DataFrame(validation_records)
    # 保存为带 BOM 的 UTF-8 CSV，防止用 Excel 打开中文乱码
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"🎉 成功抽出 50 道高价值样本！已保存至 {OUTPUT_FILE}")
    print("👉 请用 Excel 打开该文件，在最后一列填入你的打分 (1表示有幻觉，0表示无幻觉)。")


if __name__ == "__main__":
    generate_validation_sheet()