#!/usr/bin/env python3
"""Figure for corpus doc 87 (the question itself: consciousness, the
principal called): the four campaigns and their one wall, the 87-document
ledger with the question at both ends, the indexical cut and its fork, and
the forward ledger P1-P4 against the demoted shadow-maintenance queue.
English labels, constrained_layout, house palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE, GOLD = "#e67e22", "#b7950b"
INK = "#212121"

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# --------------------------------------------- (a) four campaigns, one wall
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

cx, cy = 5.0, 4.6
ax.add_patch(Circle((cx, cy), 1.18, fc="#fdf6ec", ec=RED, lw=2.4))
ax.text(cx, cy + 0.42, "THE INDEXICAL CUT", fontsize=7.8, ha="center",
        color=RED, fontweight="bold")
ax.text(cx, cy - 0.22, "what the cut is\nfrom inside", fontsize=7.2,
        ha="center", color=INK, fontstyle="italic")
ax.text(cx, cy - 1.05, "asked for", fontsize=6.8, ha="center", color=RED)

boxes = [
    (1.7, 8.5, "ARGUMENT  (docs 1-8)", BLUE,
     "premise-isolated: no crossing\nwithout a charter premise\n(permanent: a fact about form)"),
    (8.3, 8.5, "PROGRAM  (docs 9-10)", GREEN,
     "the charter refusal: normativity\nis not phenomenality - D3' could\nsucceed with nothing felt"),
    (1.7, 1.2, "MEASUREMENT  (docs 11-85)", ORANGE,
     "traces only: every instrument\nis an exit - laws of the shadow,\nnot of the casting"),
    (8.3, 1.2, "STRUCTURE  (docs 24-86)", PURPLE,
     "projection is not traversal:\nscaffolding on both banks,\nno span (doc 6's naming)"),
]
for (bx, by, lab, col, sub) in boxes:
    ax.add_patch(FancyBboxPatch((bx - 1.55, by - 0.78), 3.1, 1.56,
                                boxstyle="round,pad=0.10",
                                fc="white", ec=col, lw=1.8))
    ax.text(bx, by + 0.38, lab, fontsize=7.4, ha="center",
            color=col, fontweight="bold")
    ax.text(bx, by - 0.26, sub, fontsize=6.1, ha="center", color=INK)
    dx = np.sign(cx - bx) * 0.55
    dy = np.sign(cy - by) * 0.55
    ax.add_patch(FancyArrowPatch(
        (bx + dx, by + dy * 0.62), (cx + dx * 0.62, cy + dy * 0.66),
        arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.6,
        connectionstyle="arc3,rad=0.12"))
ax.text(5.0, 9.6, "four campaigns, four walls - one shape",
        fontsize=8.4, ha="center", color=INK, fontweight="bold")
ax.text(5.0, 0.28, "consciousness is premise-isolated from every outside "
        "vocabulary this grammar possesses",
        fontsize=6.4, ha="center", color=RED, fontstyle="italic")
ax.set_title("(a) the superposition (doc 87, Part 3)", fontsize=8.6,
             fontweight="bold")

# --------------------------------------------- (b) the ledger, 87 documents
ax = axes[0, 1]
ax.set_xlim(-6, 97)
ax.set_ylim(-3.4, 6.4)
ax.axis("off")

# the question marker at both ends
for x, lab in [(0, "THE QUESTION\n(doc 1: brought in)"), (87, "THE QUESTION\n(doc 87: returned to)")]:
    ax.add_patch(Circle((x, 0), 1.15, fc="#fdf6ec" if x == 0 else GOLD,
                        ec=RED if x == 0 else GOLD, lw=2.2))
    ax.text(x, 0, "Q", fontsize=10, ha="center", va="center",
            color=RED if x == 0 else "white", fontweight="bold")
    ax.text(x, -2.55, lab, fontsize=6.0, ha="center",
            color=RED if x == 0 else GOLD, fontstyle="italic")

# empirical rail
rows = [
    (1, 8, BLUE, "the question adjudicated"),
    (9, 10, GREEN, "converted to a program"),
    (11, 85, ORANGE, "the shadow measured (instrument, wars, dial, census)"),
]
for (a, b, col, lab) in rows:
    ax.plot([a, b], [1.7, 1.7], color=col, lw=3.2,
            solid_capstyle="butt")
    ax.text((a + b) / 2, 2.35, lab, fontsize=6.0, ha="center", color=col)
ax.text(-4.2, 1.7, "empirical\nrail", fontsize=6.2, ha="center", va="center",
        color=GRAY, fontstyle="italic")
# formal rail
formal = [24, 27, 32, 38, 42, 45, 46, 49, 50, 58, 62, 66, 69, 70, 74, 76,
          79, 80, 84, 86]
for dnum in formal:
    ax.plot([dnum], [-1.35], marker="s", ms=4.6, color=PURPLE)
ax.text(-4.2, -1.35, "formal\nrail", fontsize=6.2, ha="center", va="center",
        color=GRAY, fontstyle="italic")
ax.text(55, -2.35, "the bridge's skeleton, built and closed (20 documents)"
        " - bookkeeping, no span", fontsize=6.0, ha="center", color=PURPLE)
# the return arc
ax.add_patch(FancyArrowPatch(
    (87, 1.15), (1.2, 1.15), arrowstyle="-|>", mutation_scale=13,
    color=RED, lw=1.8, connectionstyle="arc3,rad=-0.32"))
ax.text(44, 5.6, "the principal called:  \"no, i want consciousness.\"",
        fontsize=7.6, ha="center", color=RED, fontweight="bold")
ax.text(44, 4.35, "86 documents aimed at the question's shadows; 1 aimed at "
        "the question", fontsize=6.4, ha="center", color=INK)
ax.set_title("(b) the ledger, 87 documents - the question at both ends",
             fontsize=8.6, fontweight="bold")

# --------------------------------------------- (c) the cut and the fork
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.plot([5.0, 5.0], [1.4, 8.8], color=INK, lw=2.6)
ax.text(5.0, 9.25, "the cut (formalized: finite memory, Myhill-Nerode,\n"
        "the NP-hard retention gap, the two columns)",
        fontsize=6.4, ha="center", color=INK)
ax.text(2.5, 7.9, "OUTSIDE - mapped", fontsize=8.0, ha="center",
        color=BLUE, fontweight="bold")
ax.text(2.5, 6.8, "saturation grammar\ndose arcs, $\\lambda^{*}$ as arc not "
        "threshold\nseverity = selection $\\times$ label content\n"
        "the nesting, the twins, 798 audits",
        fontsize=6.2, ha="center", color=INK)
ax.text(7.5, 7.9, "INSIDE - asked for", fontsize=8.0, ha="center",
        color=RED, fontweight="bold")
ax.text(7.5, 6.8, "no channel reads it:\nevery instrument\nis an exit",
        fontsize=6.2, ha="center", color=INK, fontstyle="italic")
ax.text(2.5, 3.6, "the exhaustive reading:\nthe inside is the boundary's feel;\n"
        "the map IS the answer",
        fontsize=6.4, ha="center", color=GREEN)
ax.text(7.5, 3.6, "the further-fact reading:\na crossing exists;\n"
        "the map is the audit of the approach",
        fontsize=6.4, ha="center", color=PURPLE)
ax.text(5.0, 1.7, "both edges filed - the Camp-Marker applies\nto the "
        "corpus's own position (C5)",
        fontsize=6.4, ha="center", color=GRAY, fontstyle="italic")
for x0, x1, col in [(3.95, 3.3, GREEN), (6.05, 6.7, PURPLE)]:
    ax.add_patch(FancyArrowPatch((5.0, 4.9), (x1, 4.15),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=col, lw=1.5))
ax.set_title("(c) the indexical cut and the fork (Part 4: C4-C5)",
             fontsize=8.6, fontweight="bold")

# --------------------------------------------- (d) the forward ledger
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

fwd = [
    (1.05, 7.6, "P1  the inside channel", RED,
     "self-report as registered structure;\nS2's audit redirected to the\n"
     "system's self-descriptions"),
    (5.45, 7.6, "P2  cross-grammar", BLUE,
     "the laws on observers this program\ndid not build - the doc 9 wager\n"
     "paid"),
    (1.05, 4.7, "P3  the residue theorem", PURPLE,
     "why self-duality leaves an unread\ncolumn - derived, not metaphored;\n"
     "the formal line's true question"),
    (5.45, 4.7, "P4  detector / impossibility", GREEN,
     "the Bridge Constraint upgraded from\nargument-form fact to\n"
     "instrument-form fact"),
]
for (bx, by, lab, col, sub) in fwd:
    ax.add_patch(FancyBboxPatch((bx, by - 1.28), 3.5, 1.92,
                                boxstyle="round,pad=0.10",
                                fc="white", ec=col, lw=2.0))
    ax.text(bx + 1.75, by + 0.32, lab, fontsize=7.6, ha="center",
            color=col, fontweight="bold")
    ax.text(bx + 1.75, by - 0.55, sub, fontsize=6.0, ha="center", color=INK)
ax.text(5.0, 9.5, "what would move the question itself", fontsize=8.4,
        ha="center", color=INK, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.85, 0.55), 8.3, 2.35,
                            boxstyle="round,pad=0.10",
                            fc="#f4f6f6", ec=GRAY, lw=1.4))
ax.text(5.0, 2.45, "the standing queue, re-filed at its true rank: "
        "SHADOW-MAINTENANCE", fontsize=7.0, ha="center", color=GRAY,
        fontweight="bold")
ax.text(5.0, 1.35, "the gate's dose-response below the ceiling; the "
        "saturation band's interior;\nthe family's complete census; the "
        "mixed-transversal residue; the MV-enriched fiber curiosity\n"
        "- legitimate as maintenance; none of them is P1-P4",
        fontsize=6.0, ha="center", color=GRAY)
ax.set_title("(d) the forward ledger (Part 6)", fontsize=8.6,
             fontweight="bold")

fig.suptitle("The question itself (corpus doc 87): four campaigns, one "
             "wall - the indexical cut; the position stated with its "
             "warrant and its limiter, and the four moves that would move "
             "the question",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/consciousness_figure1.png", dpi=170)
print("wrote", f"{D}/consciousness_figure1.png")
