#!/usr/bin/env python3
"""Figure for corpus doc 90 (the tribunal of nine): the four campaigns
applied to the field's nine theories - the tribunal matrix (bridge /
dark-system / observer / traverse-or-project), the filing (failures and
survivors), the reduction of the field to Theorem II's two completions,
and the bounded Russellian uniqueness verdict.
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

# --------------------------------------------- (a) the tribunal matrix
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.6)
ax.axis("off")

cols = [("bridge\nclaimed?", 3.35), ("dark-system\nobjection?", 5.25),
        ("reads an\nobserver?", 7.15), ("traverse or\nproject?", 9.05)]
for (lab, x) in cols:
    ax.text(x, 10.15, lab, fontsize=6.4, ha="center", va="center",
            color=INK, fontweight="bold")

# row: (name, [4 cells: (text, category)]), categories: f=fall, p=partial, s=survive, n=neutral
CAT = {"f": (RED, "#f9e3e0"), "p": (GOLD, "#fdf3dd"),
       "s": (GREEN, "#e8f4ec"), "n": (GRAY, "#eff1f1")}
rows = [
    ("physicalism", [("YES - P*b", "f"), ("no - begs", "f"), ("no", "n"),
                     ("claims traversal;\nprojects", "f")]),
    ("functionalism", [("YES - P*a/c", "f"), ("no - declines", "f"), ("no", "n"),
                       ("projects", "f")]),
    ("IIT", [("YES - P*c\nfar-bank stones", "f"),
             ("half: dark in,\nbright-dark out", "p"),
             ("no - analyst's\npartition", "n"),
             ("claims traversal;\nprojects", "f")]),
    ("GWT", [("YES - retreated\nto access", "p"), ("no - no\nreadership", "f"),
             ("no", "n"), ("projects;\nsays so", "p")]),
    ("higher-order", [("YES - P*a/c", "f"), ("no - generator", "f"),
                      ("no - but\ntried", "n"), ("projects", "f")]),
    ("illusionism", [("NO - deletion", "s"), ("yes - target\ndeleted", "p"),
                     ("no", "n"), ("far bank =\nartifact", "p")]),
    ("dualism", [("YES - P*c\nposited", "f"), ("by fiat", "f"), ("no", "n"),
                 ("projects -\nreified", "f")]),
    ("panpsychism", [("no micro /\nYES macro", "p"), ("inverts it", "p"),
                     ("no", "n"), ("inflates", "p")]),
    ("Russellian\nmonism (placed)", [("NO - placement", "s"),
                                     ("dissolves -\nTheorem II", "s"),
                                     ("no - never\nclaimed", "s"),
                                     ("places - two\nfaces, one base", "s")]),
]
y0, dy = 9.28, 1.0
for i, (name, cells) in enumerate(rows):
    yc = y0 - i * dy
    last = (i == len(rows) - 1)
    ax.text(1.05, yc, name, fontsize=6.6, ha="center", va="center",
            color=INK, fontweight="bold" if last else "normal")
    if last:
        ax.add_patch(FancyBboxPatch((0.18, yc - 0.44), 1.78, 0.88,
                                    boxstyle="round,pad=0.06",
                                    fc="#fdf6ec", ec=GOLD, lw=1.6))
        ax.text(1.05, yc, name, fontsize=6.6, ha="center", va="center",
                color=GOLD, fontweight="bold")
    for (text, cat), (_, x) in zip(cells, cols):
        fg, bg = CAT[cat]
        ax.add_patch(FancyBboxPatch((x - 0.90, yc - 0.44), 1.80, 0.88,
                                    boxstyle="round,pad=0.05",
                                    fc=bg, ec=fg, lw=1.3 if not last else 2.0))
        ax.text(x, yc, text, fontsize=5.4, ha="center", va="center", color=fg)
ax.text(5.0, 0.28, "the registrar's four questions = the four campaigns, one per wall;\n"
        "no grammar-internal criterion can rule a dark system out (Theorem II)",
        fontsize=6.0, ha="center", color=GRAY, fontstyle="italic")
ax.set_title("(a) the tribunal matrix (Part 1)", fontsize=8.6,
             fontweight="bold")

# --------------------------------------------- (b) the filing
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.25, 0.7), 4.4, 8.3,
                            boxstyle="round,pad=0.10", fc="white", ec=RED, lw=1.8))
ax.text(2.45, 8.55, "FAILURES - six counterfeit bridges", fontsize=7.4,
        ha="center", color=RED, fontweight="bold")
ax.text(2.45, 7.95, "by premise", fontsize=6.8, ha="center", color=INK,
        fontweight="bold")
ax.text(2.45, 7.30, "physicalism (P*b identity)\nfunctionalism (P*a/c realization)\n"
        "dualism (P*c pairing - interactionism\nfalsified on earned ground)",
        fontsize=6.0, ha="center", va="center", color=INK)
ax.text(2.45, 5.55, "by instrument", fontsize=6.8, ha="center", color=INK,
        fontweight="bold")
ax.text(2.45, 4.55, "IIT (Phi-identity)\nGWT (broadcast-identity)\nhigher-order (the generator)",
        fontsize=6.0, ha="center", va="center", color=INK)
ax.text(2.45, 2.85, "the instruments are real; the bridges\nare the instruments' names",
        fontsize=6.2, ha="center", va="center", color=RED, fontstyle="italic")
ax.text(2.45, 1.35, "GWT survives as access-law;\nIIT as integration science",
        fontsize=5.8, ha="center", va="center", color=GRAY)

ax.add_patch(FancyBboxPatch((5.1, 0.7), 4.65, 8.3,
                            boxstyle="round,pad=0.10", fc="white", ec=GREEN, lw=1.8))
ax.text(7.42, 8.55, "SURVIVORS", fontsize=7.4, ha="center", color=GREEN,
        fontweight="bold")
surv = [
    (7.05, "illusionism - by deletion", GOLD,
     "the exhaustivist completion of Theorem II;\nC1-protected; cost: the undischarged\nperformance; question returned to sender"),
    (4.55, "panpsychism - as position", ORANGE,
     "the founder's camp (marker up);\nproof refuted, position refutation-proof;\ncost: combination unposeable - the overshoot"),
]
for (by, lab, col, sub) in surv:
    ax.add_patch(FancyBboxPatch((5.35, by - 1.05), 4.15, 2.0,
                                boxstyle="round,pad=0.08", fc="#fdfdf9",
                                ec=col, lw=1.5))
    ax.text(7.42, by + 0.55, lab, fontsize=6.8, ha="center", color=col,
            fontweight="bold")
    ax.text(7.42, by - 0.30, sub, fontsize=5.8, ha="center", va="center",
            color=INK)
ax.add_patch(FancyBboxPatch((5.35, 0.95), 4.15, 1.95,
                            boxstyle="round,pad=0.08", fc="#e8f4ec",
                            ec=GREEN, lw=2.2))
ax.text(7.42, 2.42, "Russellian monism (placed) - SOLE SURVIVOR",
        fontsize=6.8, ha="center", color=GREEN, fontweight="bold")
ax.text(7.42, 1.55, "no bridge bought; dark-system dissolved on\n"
        "theorem-grade ground; the cut placed, not crossed;\n"
        "caps: no observer read, no crossing, one labeled jump",
        fontsize=5.8, ha="center", va="center", color=INK)
ax.set_title("(b) the filing (Part 2)", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (c) the reduction
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((2.0, 8.15), 6.0, 1.35,
                            boxstyle="round,pad=0.10", fc="#fdf6ec",
                            ec=PURPLE, lw=2.2))
ax.text(5.0, 9.12, "READING = EXTENSIONALIZING", fontsize=8.6, ha="center",
        color=PURPLE, fontweight="bold")
ax.text(5.0, 8.50, "Theorem II (doc 89): no grammar reads what its\n"
        "ontology quotiented away", fontsize=6.2, ha="center", color=INK)

ax.add_patch(FancyBboxPatch((0.35, 5.85), 4.3, 1.55,
                            boxstyle="round,pad=0.08", fc="#f9e3e0",
                            ec=RED, lw=1.6))
ax.text(2.5, 7.02, "six identify the complement\nwith quotient-side facts", fontsize=6.6,
        ha="center", color=RED, fontweight="bold")
ax.text(2.5, 6.25, "identity, role, Phi, broadcast, monitoring\n"
        "(physicalism, functionalism, IIT, GWT, HOT)",
        fontsize=5.6, ha="center", color=INK)
ax.add_patch(FancyBboxPatch((5.35, 5.85), 2.05, 1.55,
                            boxstyle="round,pad=0.08", fc="#f9e3e0",
                            ec=RED, lw=1.6))
ax.text(6.37, 7.02, "one bridges\nby charter", fontsize=6.6, ha="center",
        color=RED, fontweight="bold")
ax.text(6.37, 6.28, "dualism (P*c)", fontsize=5.8, ha="center", color=INK)
ax.add_patch(FancyBboxPatch((7.75, 5.85), 1.95, 1.55,
                            boxstyle="round,pad=0.08", fc="#fdf3dd",
                            ec=ORANGE, lw=1.6))
ax.text(8.72, 7.02, "one inflates to\nubiquity", fontsize=6.6, ha="center",
        color=ORANGE, fontweight="bold")
ax.text(8.72, 6.28, "panpsychism", fontsize=5.8, ha="center", color=INK)

for x0 in (2.5, 6.37, 8.72):
    ax.add_patch(FancyArrowPatch((x0, 5.80), (x0 if x0 > 5 else 3.6, 5.62),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=GRAY, lw=1.4))
ax.text(5.0, 5.18, "the forbidden reading: reading the complement\n"
        "from inside the quotient",
        fontsize=5.8, ha="center", va="center", color=GRAY, fontstyle="italic")

ax.add_patch(FancyBboxPatch((0.35, 3.05), 4.3, 1.85,
                            boxstyle="round,pad=0.08", fc="#e8f4ec",
                            ec=GREEN, lw=2.0))
ax.text(2.5, 4.50, "PLACEMENT - Russellian monism", fontsize=6.8,
        ha="center", color=GREEN, fontweight="bold")
ax.text(2.5, 3.70, "\"...and the complement is carried,\n"
        "unread, as the base's other face\"\nthe sole survivor",
        fontsize=6.0, ha="center", color=INK)
ax.add_patch(FancyBboxPatch((5.35, 3.05), 4.35, 1.85,
                            boxstyle="round,pad=0.08", fc="#fdf3dd",
                            ec=GOLD, lw=2.0))
ax.text(7.52, 4.50, "DELETION - illusionism", fontsize=6.8, ha="center",
        color=GOLD, fontweight="bold")
ax.text(7.52, 3.70, "\"...and the complement is empty\"\n"
        "survivor by deletion;\nthe fork's left edge",
        fontsize=6.0, ha="center", color=INK)

ax.add_patch(FancyBboxPatch((0.35, 0.65), 9.35, 1.95,
                            boxstyle="round,pad=0.10", fc="#f4f6f6",
                            ec=GRAY, lw=1.4))
ax.text(5.02, 2.15, "the sentence between them is undecidable in any grammar "
        "(Prop. 86)", fontsize=6.6, ha="center", color=INK, fontweight="bold")
ax.text(5.02, 1.25, "the field, run through the four campaigns, collapses onto the corpus's own\n"
        "open fork (C5): six violate the equation, one inflates it, two complete it -\n"
        "and the two completions are the two edges of the question the registrar asked",
        fontsize=5.9, ha="center", color=GRAY)
ax.set_title("(c) the reduction (Part 3)", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (d) the uniqueness verdict
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(5.0, 9.45, "IS RUSSELLIAN MONISM UNIQUE?", fontsize=8.4, ha="center",
        color=INK, fontweight="bold")
ax.text(5.0, 8.72, "YES - bounded three times; the bounds are as much of\n"
        "the verdict as the yes", fontsize=6.8, ha="center", color=GREEN,
        fontweight="bold")

ax.add_patch(FancyBboxPatch((0.30, 4.85), 9.35, 3.35,
                            boxstyle="round,pad=0.10", fc="#e8f4ec",
                            ec=GREEN, lw=1.8))
ax.text(5.0, 7.80, "the grounds", fontsize=7.0, ha="center", color=GREEN,
        fontweight="bold")
ax.text(0.62, 6.05, "1.  no bridge bought - the Bridge Constraint, the corpus's one\n"
        "     permanent result, idle for exactly one entry\n"
        "2.  the dark-system objection dissolved, not begged - the twin, as\n"
        "     specified, is a quotient class, and a quotient class is not a system\n"
        "3.  its central shape re-derived by the corpus's formalism, earned before\n"
        "     the convergence was disclosed (C2; no-right-adjoint; Thm II; Prop 86)\n"
        "4.  the residual located at one labeled jump from grammar to world -\n"
        "     C4's own warrant, re-filed at doc 89",
        fontsize=5.9, ha="left", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 2.35), 9.35, 2.20,
                            boxstyle="round,pad=0.10", fc="#f9e3e0",
                            ec=RED, lw=1.8))
ax.text(5.0, 4.20, "the caps", fontsize=7.0, ha="center", color=RED,
        fontweight="bold")
ax.text(0.62, 3.10, "1.  survival, not truth - the Camp-Marker governs; another camp, other\n"
        "     instruments, other survivors; no grammar settles between the two that stand\n"
        "2.  reads no observer - never claimed to; which is why it survives, and why it\n"
        "     delivers no consciousness: the map is not the crossing\n"
        "3.  the fork runs through its interior - epistemic wing vs identification wing",
        fontsize=5.9, ha="left", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 0.55), 9.35, 1.55,
                            boxstyle="round,pad=0.10", fc="#fdf6ec",
                            ec=GOLD, lw=1.6))
ax.text(5.02, 1.75, "unique because it is the shape of the wall", fontsize=7.0,
        ha="center", color=GOLD, fontweight="bold")
ax.text(5.02, 1.00, "the corpus cannot kill it without unsaying Theorem II, and cannot extract\n"
        "the crossing from it for the same reason - the crossing is still owed",
        fontsize=5.9, ha="center", color=INK)
ax.set_title("(d) the Russellian verdict (Part 4)", fontsize=8.6,
             fontweight="bold")

fig.suptitle("The tribunal of nine (corpus doc 90): the four campaigns against the "
             "field - the filing, the reduction of the field onto the fork, and the "
             "bounded uniqueness of Russellian monism",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/tribunal_figure1.png", dpi=170)
print("wrote", f"{D}/tribunal_figure1.png")
