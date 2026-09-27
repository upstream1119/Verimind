import os
import matplotlib.pyplot as plt

# Data from the finalized SciFact results table
systems = ["VeriMind_Full", "Vanilla_RAG", "HyDE"]
hallucination_rates = [0.0000, 0.3960, 0.5920]

# Output path
output_dir = r"D:\Desktop\peng\study\project\rag_project_domestic\figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Fig9_scifact_hallucination_bar.pdf")

# Create figure
plt.figure(figsize=(7.2, 4.8))
bars = plt.bar(systems, hallucination_rates)

# Labels
plt.ylabel("Hallucination Rate")
plt.xlabel("System")
plt.ylim(0, 0.65)

# Add value labels
for bar, value in zip(bars, hallucination_rates):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# Save figure
plt.tight_layout()
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"Saved to: {output_path}")