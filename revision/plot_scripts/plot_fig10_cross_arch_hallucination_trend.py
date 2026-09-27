import os
import matplotlib.pyplot as plt

# Data from E1-E5 cross-architecture results
settings = ["E1", "E2", "E3", "E4", "E5"]

vanilla = [0.3800, 0.4600, 0.7267, 0.6700, 0.7600]
hyde = [0.3533, 0.4133, 0.7067, 0.6667, 0.7700]
verimind = [0.0000, 0.0033, 0.0000, 0.0000, 0.0000]

# Output path
output_dir = r"D:\Desktop\peng\study\project\rag_project_domestic\figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Fig10_cross_arch_hallucination_trend.pdf")

# Create figure
plt.figure(figsize=(7.6, 4.8))

plt.plot(settings, vanilla, marker="o", linewidth=2, label="Vanilla_RAG")
plt.plot(settings, hyde, marker="o", linewidth=2, label="HyDE")
plt.plot(settings, verimind, marker="o", linewidth=2, label="VeriMind_Full")

# Labels
plt.xlabel("Cross-Architecture Setting")
plt.ylabel("Hallucination Rate")
plt.ylim(0, 0.82)

# Add legend
plt.legend()

# Save figure
plt.tight_layout()
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"Saved to: {output_path}")