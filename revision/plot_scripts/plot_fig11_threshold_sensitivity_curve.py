import os
import pandas as pd
import matplotlib.pyplot as plt

# Input CSV
csv_path = r"D:\Desktop\peng\study\project\rag_project_domestic\revision\artifacts\2_3_threshold\threshold_sensitivity_curve.csv"

# Output PDF
output_dir = r"D:\Desktop\peng\study\project\rag_project_domestic\figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Fig11_threshold_sensitivity_curve.pdf")

# Read CSV
df = pd.read_csv(csv_path)
df = df.sort_values("threshold")

thresholds = df["threshold"]
hallucination_rate = df["hallucination_rate_mean"]
rouge_l_f1 = df["f1_score_mean"]
accept_rate = df["accept_rate"]

# Create figure
fig, ax1 = plt.subplots(figsize=(8.0, 4.8))

# Left axis
line1, = ax1.plot(
    thresholds, hallucination_rate,
    marker="o", linewidth=2, markersize=5,
    label="Hallucination Rate"
)
line2, = ax1.plot(
    thresholds, rouge_l_f1,
    marker="s", linewidth=2, markersize=5,
    label="ROUGE-L F1"
)

ax1.set_xlabel("Threshold")
ax1.set_ylabel("Hallucination Rate / ROUGE-L F1")

# 关键修改：把左轴范围调大
ax1.set_ylim(0.0, 0.25)
ax1.set_xlim(thresholds.min() - 0.1, thresholds.max() + 0.1)

# Current threshold marker：只保留竖线，不要文字
ax1.axvline(x=5.0, linestyle="--", linewidth=1)

# Right axis
ax2 = ax1.twinx()
line3, = ax2.plot(
    thresholds, accept_rate,
    marker="^", linewidth=2, markersize=6,
    label="Accept Rate"
)
ax2.set_ylabel("Accept Rate")

# 关键修改：右轴稍微放宽
ax2.set_ylim(0.0, 1.15)

# Legend: upper right
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True)

# Save
plt.tight_layout()
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"Saved to: {output_path}")