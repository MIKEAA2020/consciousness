#!/usr/bin/env python3
"""Figure for corpus doc 0: the corpus's own map (the five-act flow, the two
lines interleaved, the census factorial the map arrives at, and the formal
line's theorem ladder ending at the open MV cell). English labels,
constrained_layout, house palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# ---------------------------------------------------------------- (a) the flow
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
acts = [
    ("ACT I\nthe audits\n(docs 1-8)",
     "the question adjudicated,\nnot answered", GREEN),
    ("ACT II\nthe program\n(docs 9-10)",
     "the stalemate converted\ninto a falsifiable program", GREEN),
    ("ACT III\nthe instrument\n(docs 11-15)",
     "designer-blind audit;\nvariance floor measured", BLUE),
    ("ACT IV\nbuilds & wars\n(docs 16-49)",
     "entry, tradeoff broken,\npoison wars, certificates", BLUE),
    ("ACT V\nthe dial & census\n(docs 50-82)",
     "the arc, the golden mean,\nthe nesting", BLUE),
]
xw, y0, hh = 1.72, 3.1, 3.2
for i, (lab, sub, col) in enumerate(acts):
    x = 0.25 + i * (xw + 0.22)
    ax.add_patch(FancyBboxPatch(
        (x, y0), xw, hh, boxstyle="round,pad=0.10",
        fc="white", ec=col, lw=1.7))
    ax.text(x + xw / 2, y0 + hh - 0.62, lab, ha="center", va="center",
            fontsize=8.0, color=col, fontweight="bold")
    ax.text(x + xw / 2, y0 + 0.62, sub, ha="center", va="center",
            fontsize=6.7, color="#212121")
    if i < len(acts) - 1:
        ax.add_patch(FancyArrowPatch(
            (x + xw + 0.11, y0 + hh / 2), (x + xw + 0.22, y0 + hh / 2),
            arrowstyle="-|>", mutation_scale=11, color=GRAY, lw=1.4))
# the formal line as a parallel rail under the flow
ax.add_patch(FancyBboxPatch(
    (0.25, 0.7), 9.5, 1.15, boxstyle="round,pad=0.08",
    fc="#f4ecf7", ec=PURPLE, lw=1.5))
ax.text(5.0, 1.28, "THE FORMAL LINE, beside it all along - 18 documents (docs 24-80): ChuCost over the cost monoids, phases 0-15,",
        ha="center", va="center", fontsize=6.9, color=PURPLE)
ax.text(5.0, 0.98, "the unit-inclusive question closed in every Con form and OPEN in the MV home (the registrar's next order)",
        ha="center", va="center", fontsize=6.9, color=PURPLE)
for i in range(5):
    x = 0.25 + i * (xw + 0.22) + xw / 2
    ax.add_patch(FancyArrowPatch(
        (x, y0 - 0.06), (x, 1.95), arrowstyle="-|>", mutation_scale=8,
        color=PURPLE, lw=0.9, ls=(0, (3, 2))))
ax.text(0.25, 9.3, "the corpus, 82 documents - 10 philosophy / 54 measurement / 18 formal - 50 figures - 702 bitwise audits",
        fontsize=7.4, color="#212121", fontweight="bold")
ax.text(0.25, 8.75, "from a chat about consciousness to a measured world under a registered instrument",
        fontsize=6.9, color=GRAY)
ax.set_title("(a) THE MAP'S FLOW: five acts, one rail - the question transforms twice and never answers",
             fontsize=9.5)

# ------------------------------------------------- (b) the two lines interleaved
ax = axes[0, 1]
phil = list(range(1, 11))
emp = [d for d in range(11, 83) if d not in
       (24, 27, 32, 38, 42, 45, 46, 49, 50, 58, 62, 66, 69, 70, 74, 76,
        79, 80)]
for_ = [24, 27, 32, 38, 42, 45, 46, 49, 50, 58, 62, 66, 69, 70, 74, 76,
        79, 80]
ax.scatter(phil, [2.0] * len(phil), c=GREEN, marker="o", s=26, zorder=3,
           label=f"philosophy ({len(phil)})")
ax.scatter(emp, [2.0] * len(emp), c=BLUE, marker="o", s=26, zorder=3,
           label=f"empirical ({len(emp)})")
ax.scatter(for_, [1.1] * len(for_), c=PURPLE, marker="D", s=30, zorder=3,
           label=f"formal, interleaved ({len(for_)})")
for d, tag in [(1, "the critique"), (9, "the ladder"), (11, "M1"),
               (12, "A4"), (35, "claim table"), (51, "5th cond."),
               (67, "golden mean"), (78, "census"), (82, "the twin")]:
    ax.annotate(tag, (d, 2.42), fontsize=6.4, color="#212121",
                ha="center", rotation=38)
for d, tag in [(24, "ChuCost"), (38, "quantum"), (70, "unit-inc."),
               (80, "MV open")]:
    ax.annotate(tag, (d, 0.62), fontsize=6.4, color=PURPLE,
                ha="center", rotation=38)
ax.annotate("doc 0 (this map)", (0.9, 3.1), fontsize=6.8, color=RED,
            fontweight="bold")
ax.scatter([1], [3.0], c=RED, marker="*", s=90, zorder=4)
ax.set_xlim(-1.5, 85)
ax.set_ylim(0.1, 4.4)
ax.set_yticks([])
ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 82])
ax.tick_params(labelsize=7.5)
ax.set_xlabel("corpus doc number", fontsize=8)
ax.grid(axis="x", color="#dddddd", lw=0.6)
ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, 1.02),
          framealpha=0.9)
ax.set_title("(b) THE TWO LINES: the formal rail interleaved from doc 24 -\n"
             "one corpus, not two", fontsize=9.5)

# ------------------------------------------------------- (c) the census factorial
ax = axes[1, 0]
grid = np.array([[1, 6], [12, 0]])
labels = np.array([["twin  uniform 0.2484\n1 BURNED / 24",
                    "mirror  mixed 0.1238\n6 BURNED / 24"],
                   ["capacity  mixed 0.2505\n12 BURNED / 24",
                    "control  uniform 0.1291\n0 BURNED / 24"]])
im = ax.imshow(grid, cmap="OrRd", vmin=0, vmax=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i, j], ha="center", va="center",
                fontsize=8.2,
                color="white" if grid[i, j] >= 6 else "#212121")
ax.set_xticks([0, 1])
ax.set_xticklabels(["uniform (no gated mass)", "mixed (bulk + gated)"],
                   fontsize=8)
ax.set_yticks([0, 1])
ax.set_yticklabels(["realized ~0.25", "realized ~0.125"], fontsize=8)
ax.set_title("(c) WHERE THE MAP ARRIVES: the completed factorial - the burn\n"
             "sets nest by b*: {606} < the gate's six < the mix's twelve",
             fontsize=9.5)
fig.colorbar(im, ax=ax, shrink=0.82, label="BURNED of 24")
ax.annotate("doc 82, the twin", (0.02, 0.90), fontsize=7.4, color=BLUE,
            xycoords="axes fraction")

# ------------------------------------------------------ (d) the formal ladder
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
steps = [
    ("Theorem Z", "doc 70", "plain unit-inclusive\nREFUTED", RED),
    ("Theorem AA", "doc 74", "enriched\nREFUTED", RED),
    ("Theorem BB", "doc 76", "fiber-equivalent\nVACUOUS", ORANGE),
    ("Theorem CC", "doc 79", "fiber existence\nEMPTY", ORANGE),
    ("Theorem DD", "doc 80", "the MV cell\nOPEN", GREEN),
]
bw, bh, yb = 1.66, 3.3, 3.4
for i, (name, docref, verdict, col) in enumerate(steps):
    x = 0.35 + i * (bw + 0.18)
    ax.add_patch(FancyBboxPatch(
        (x, yb), bw, bh, boxstyle="round,pad=0.10",
        fc="white", ec=col, lw=1.8))
    ax.text(x + bw / 2, yb + bh - 0.62, name, ha="center", va="center",
            fontsize=8.4, color=col, fontweight="bold")
    ax.text(x + bw / 2, yb + bh - 1.28, docref, ha="center", va="center",
            fontsize=6.6, color=GRAY)
    ax.text(x + bw / 2, yb + 0.92, verdict, ha="center", va="center",
            fontsize=6.9, color="#212121")
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch(
            (x + bw + 0.02, yb + bh / 2), (x + bw + 0.18, yb + bh / 2),
            arrowstyle="-|>", mutation_scale=11, color=GRAY, lw=1.4))
ax.text(5.0, 8.6, "the unit-inclusive question's ladder - every Con form closed;",
        ha="center", fontsize=8.0, color="#212121", fontweight="bold")
ax.text(5.0, 8.05, "both kills are the quantum square's, and the square is absent in the MV home",
        ha="center", fontsize=7.2, color="#212121")
ax.text(5.0, 1.9, "the one live cell: the MV ABSTRACT route - fragments with big backward carriers,",
        ha="center", fontsize=7.4, color=GREEN)
ax.text(5.0, 1.35, "the observer SURVIVES there - the registrar's standing next order (doc 80, Part 4)",
        ha="center", fontsize=7.4, color=GREEN)
ax.add_patch(FancyArrowPatch(
    (8.6, 3.15), (7.4, 2.25), arrowstyle="-|>", mutation_scale=11,
    color=GREEN, lw=1.4, connectionstyle="arc3,rad=-0.25"))
ax.set_title("(d) THE FORMAL LINE'S FRONTIER: the theorem ladder to the open MV cell",
             fontsize=9.5)

fig.suptitle("Doc 0 - the corpus's own map: from the consciousness question to the census",
             fontsize=11.5, fontweight="bold")
out = f"{D}/doc0_figure1.png"
fig.savefig(out, dpi=170)
print("wrote", out)
