import pandas as pd
import numpy as np

print("📥 正在加载并融合数据...")
# 1. 读取包含答案的实验结果
df_results = pd.read_csv('experiment_results_sci_v3.csv')

# 2. 读取你的原始测试集（必须包含 question_id, query 和 context）
# 【⚠️请修改】：根据你的实际文件名和格式修改这里 (比如 verimind_bench_v3.csv)
df_bench = pd.read_csv('verimind_bench_v3.csv')

# 3. 合并数据：把 context 拼接到实验结果中
df = pd.merge(df_results, df_bench[['question_id', 'query', 'context']], on='question_id', how='left')

# 4. 自动推断回答状态 (TRUSTED or REJECTED)
def determine_status(row):
    ans = str(row['generated_answer']).lower()
    # 如果答案里包含拒绝的字眼，则判定为 REJECTED
    if 'information not found' in ans or 'insufficient information' in ans:
        return 'REJECTED'
    return 'TRUSTED'

df['status'] = df.apply(determine_status, axis=1)

# 5. 为了涵盖对比基线，我们只从 Vanilla_RAG, Self-RAG 和 VeriMind_Full 中抽样
target_models = ['Vanilla_RAG', 'Self-RAG', 'VeriMind_Full']
df_filtered = df[df['strategy'].isin(target_models)]

print("🎲 开始执行 150 题分层双盲抽样...")
# 由于数据集中基本只有 TRUSTED 和 REJECTED，我们将抽样比例调整为各 75 题
strata_sizes = {
    'TRUSTED': 75,
    'REJECTED': 75
}

sampled_dfs = []
for status, size in strata_sizes.items():
    subset = df_filtered[df_filtered['status'] == status]
    if len(subset) > 0:
        sampled = subset.sample(n=size, replace=(len(subset) < size), random_state=42)
        sampled_dfs.append(sampled)

# 合并并彻底打乱顺序
final_sample = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# 6. 保存最终的交叉验证集
final_sample.to_csv('w1_cross_validation_set.csv', index=False)
print(f"✅ 成功生成验证集 w1_cross_validation_set.csv！各状态分布：\n{final_sample['status'].value_counts()}")