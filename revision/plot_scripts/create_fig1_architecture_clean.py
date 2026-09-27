"""Generate the Computing submission architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "revision" / "submission" / "Computing_2026" / "source" / "Fig1_architecture.pdf"


def box(ax, x, y, w, h, title, detail, face, edge="#30343B"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.35, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=16.5, fontweight="bold", linespacing=1.0, color="#17191D")
    ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center",
            fontsize=13.5, linespacing=1.12, color="#30343B")


def arrow(ax, start, end, label=None, y_offset=0.025):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.35, color="#3D424A", shrinkA=4, shrinkB=4,
    )
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + y_offset,
                label, ha="center", va="bottom", fontsize=13.5, color="#3D424A")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(10.0, 4.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_y, top_h, top_w = 0.66, 0.22, 0.205
    xs = [0.025, 0.275, 0.525, 0.775]
    top = [
        ("PARSE", "layout repair\nsemantic chunks", "#EDF1F4"),
        ("RETRIEVE", "intent routing\nBM25 + dense RRF", "#E8F2F8"),
        ("GENERATE", "evidence-bounded\ncandidate answer", "#EDF5EA"),
        ("SCORE", "runtime features\ncheap score $\\hat{u}_c$", "#FFF3DE"),
    ]
    for x, (title, detail, face) in zip(xs, top):
        box(ax, x, top_y, top_w, top_h, title, detail, face)
    for i in range(3):
        arrow(ax, (xs[i] + top_w, top_y + top_h / 2),
              (xs[i + 1], top_y + top_h / 2))

    decision_x, decision_y, decision_w, decision_h = 0.39, 0.40, 0.22, 0.13
    box(ax, decision_x, decision_y, decision_w, decision_h,
        "CALIBRATE",
        "freeze $\\tau_l$ and $\\tau_h$", "#F5F5F5")
    arrow(ax, (xs[-1] + top_w / 2, top_y),
          (decision_x + decision_w / 2, decision_y + decision_h))

    branch_y, branch_h, branch_w = 0.10, 0.18, 0.25
    branches = [
        (0.045, "LOW", "ABSTAIN", "#FBE9E7", "#B7463C"),
        (0.375, "UNCERTAIN", "VERIFY", "#FFF4D6", "#A66A00"),
        (0.705, "HIGH", "DIRECT ANSWER", "#E7F4EA", "#2C7A45"),
    ]
    for x, title, detail, face, edge in branches:
        box(ax, x, branch_y, branch_w, branch_h, title, detail, face, edge=edge)

    source = (decision_x + decision_w / 2, decision_y)
    for x, _, _, _, _ in branches:
        arrow(ax, source, (x + branch_w / 2, branch_y + branch_h), y_offset=0.0)

    ax.text(
        0.5, 0.035,
        "Verifier output is combined with the cheap score; uncertain candidates are released only when $\\hat{u}_v \\geq \\tau_v$.",
        ha="center", va="center", fontsize=12.8, color="#30343B",
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
