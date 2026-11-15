import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

output_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis("off")

nodes = [
    (9, "Data Scientist Agent", "Profile data and identify class imbalance"),
    (7, "Hypothesis Agent", "Propose the next model experiment"),
    (5, "ML Engineer Agent", "Train and evaluate the model"),
    (3, "Judge Agent", "Compare experiment results"),
    (1, "Final Evaluation Agent", "Evaluate selected model on unseen test data"),
]

for y, title, description in nodes:
    color = "lightgreen" if title == "Final Evaluation Agent" else "lightblue"

    box = FancyBboxPatch(
        (3, y), 6, 1.1,
        boxstyle="round,pad=0.12",
        linewidth=1.5,
        facecolor=color
    )
    ax.add_patch(box)

    ax.text(6, y + 0.72, title, ha="center",
            va="center", fontsize=12, fontweight="bold")
    ax.text(6, y + 0.32, description, ha="center",
            va="center", fontsize=9)

# Main workflow arrows
for y1, y2 in [(9, 8.1), (7, 6.1), (5, 4.1), (3, 2.1)]:
    ax.add_patch(FancyArrowPatch(
        (6, y1), (6, y2),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5
    ))

# Feedback loop from Judge back to Hypothesis
ax.add_patch(FancyArrowPatch(
    (9.1, 3.5), (9.1, 7.5),
    connectionstyle="arc3,rad=-0.55",
    arrowstyle="->",
    mutation_scale=16,
    linewidth=1.5
))

ax.text(11, 5.5, "More\nexperiments",
        ha="center", va="center", fontsize=10)

ax.text(1.4, 2.5, "Cycle limit\nreached",
        ha="center", va="center", fontsize=10)

ax.set_title("Financial Risk Agent Workflow",
             fontsize=17, pad=20)

plt.savefig(
    output_dir / "fraud_agent_graph.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.4
)

plt.close()
print("Complete agent graph saved successfully.")
