#!/usr/bin/env python3
"""Figure for corpus doc 92 (the bias audit of the tribunal): the blind
battery of five scorers under the neutral criteria - the score matrix,
the inversion of doc 90's ranking, the two parity cells (the bridge
differential and the deletion differential), and the amended verdict.
English labels, constrained_layout, house palette; reads
doc92_results.json emitted by the bias audit instrument."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE, GOLD = "#e67e22", "#b7950b"
INK = "#212121"

with open(f"{D}/doc92_results.json", encoding="utf-8") as f:
    R = json.load(f)

SCORERS = ["4-a", "4-b", "4-c", "4-d", "4-e"]
SHORT = {"functionalism": "FUNC", "dualism": "DUAL", "iit": "IIT",
         "illusionism": "ILLU", "physicalism": "PHYS", "gwt": "GWT",
         "russellian": "RM", "higher-order": "HOT", "panpsychism": "PAN"}
ORDER = ["functionalism", "dualism", "iit", "illusionism", "physicalism",
         "gwt", "russellian", "higher-order", "panpsychism"]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# --------------------------------------------- (a) the battery matrix
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

x0, colw = 2.35, 0.82
y0, dy = 8.05, 1.28
for j, t in enumerate(ORDER):
    x = x0 + j * colw
    ax.text(x, 9.28, SHORT[t], fontsize=6.2, ha="center", va="center",
            color=INK, fontweight="bold")
    if t == "russellian":
        ax.add_patch(FancyBboxPatch((x - 0.42, 2.40), 0.84, 6.22,
                                    boxstyle="round,pad=0.03",
                                    fc="#fdf6ec", ec=GOLD, lw=1.8))
persona = {"4-a": "phys", "4-b": "func", "4-c": "illu", "4-d": "neut",
           "4-e": "russ"}
for i, s in enumerate(SCORERS):
    yc = y0 - i * dy
    ax.text(1.05, yc, f"{s} {persona[s]}", fontsize=6.3, ha="center",
            va="center", color=INK,
            fontweight="bold" if s == "4-d" else "normal")
    for j, t in enumerate(ORDER):
        v = R["totals"][s][t]
        fg, bg = ((GREEN, "#e8f4ec") if v >= 9 else
                  (GOLD, "#fdf3dd") if v == 8 else (RED, "#f9e3e0"))
        x = x0 + j * colw
        ax.add_patch(FancyBboxPatch((x - 0.38, yc - 0.48), 0.76, 0.96,
                                    boxstyle="round,pad=0.03",
                                    fc=bg, ec=fg, lw=1.1))
        ax.text(x, yc, str(v), fontsize=6.8, ha="center", va="center",
                color=fg, fontweight="bold")
ax.text(5.0, 1.30, "totals /10; gold column = the tribunal's sole "
        "survivor, which no unaligned scorer places\nabove 7.5th, while "
        "the six failures hold the battery's top six",
        fontsize=6.0, ha="center", va="center", color=GRAY,
        fontstyle="italic")
ax.text(5.0, 0.55, "personas: 4-a Type-B physicalist, 4-b empirical "
        "functionalist, 4-c illusionist, 4-d neutral methodologist,\n"
        "4-e Russellian monist - all blind to doc 90, the corpus, and the "
        "verdict",
        fontsize=5.8, ha="center", va="center", color=GRAY)
ax.set_title("(a) the blind battery (Part 3)", fontsize=8.6,
             fontweight="bold")

# --------------------------------------------- (b) the inversion
ax = axes[0, 1]
ax.set_xlim(-1.15, 2.15)
ax.set_ylim(10.7, -0.8)
ax.axis("off")

trib, bat = R["tribunal"]["rank"], R["mean_rank"]
# stagger right-side labels where battery ranks tie
right_counts = {}
for t in ORDER:
    right_counts.setdefault(bat[t], []).append(t)
right_off = {}
for v, ts in right_counts.items():
    if len(ts) > 1:
        for k, t in enumerate(ts):
            right_off[t] = (k - (len(ts) - 1) / 2.0) * 0.46
    else:
        right_off[ts[0]] = 0.0
for t in ORDER:
    x1, x2 = 0.0, 1.0
    y1, y2 = trib[t], bat[t]
    inv = (y1 == y2)
    col = GREEN if inv else (RED if t == "russellian" else GRAY)
    lw = 2.2 if (t == "russellian" or inv) else 1.2
    ax.plot([x1, x2], [y1, y2], color=col, lw=lw,
            solid_capstyle="round", zorder=2)
    ax.text(x1 - 0.10, y1, f"{SHORT[t]} {y1:.0f}", fontsize=6.2, ha="right",
            va="center", color=col, fontweight="bold" if t == "russellian"
            else "normal")
    ax.text(x2 + 0.10, y2 + right_off[t], f"{y2:.0f} {SHORT[t]}",
            fontsize=6.2, ha="left", va="center",
            color=col, fontweight="bold" if t == "russellian" else "normal")
    ax.plot([x1], [y1], "o", color=col, ms=4, zorder=3)
    ax.plot([x2], [y2], "o", color=col, ms=4, zorder=3)
ax.text(0.0, -0.60, "doc 90's tribunal", fontsize=7.0, ha="center",
        color=INK, fontweight="bold")
ax.text(1.0, -0.60, "battery, neutral criteria", fontsize=7.0, ha="center",
        color=INK, fontweight="bold")
ax.text(0.5, 10.45, "green = both lenses agree (ILLU 2-2, HOT 6-6, "
        "DUAL 9-9); spearman(tribunal, battery) = 0.2",
        fontsize=6.0, ha="center", va="center", color=GRAY,
        fontstyle="italic")
ax.text(0.5, 10.05, "red = the tribunal's sole survivor, falling 1 -> 7 "
        "under the neutral criteria",
        fontsize=6.0, ha="center", va="center", color=RED,
        fontstyle="italic")
ax.set_title("(b) the inversion (Part 3)", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (c) the parity cells
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

n1, n2 = R["n1_parity"]["cells"], R["n2_parity"]["cells"]


def parity_table(ox, title, data, keys, labels, note, gift_cells):
    """data: scorer -> {key: score}; gift_cells: {(scorer, key)}."""
    ax.text(ox + 1.85, 9.55, title, fontsize=7.0, ha="center", color=PURPLE,
            fontweight="bold")
    for j, (k, lab) in enumerate(zip(keys, labels)):
        x = ox + 1.35 + j * 1.60
        ax.text(x, 8.85, lab, fontsize=6.2, ha="center", va="center",
                color=INK, fontweight="bold")
    for i, s in enumerate(SCORERS):
        yc = 8.15 - i * 1.18
        ax.text(ox + 0.10, yc, s, fontsize=6.2, ha="right",
                va="center", color=GRAY,
                fontweight="bold" if s == "4-d" else "normal")
        for j, k in enumerate(keys):
            v = data[s][k]
            gifted = (s, k) in gift_cells
            fg, bg = (GOLD, "#fdf6ec") if gifted else (GRAY, "#eff1f1")
            x = ox + 1.35 + j * 1.60
            ax.add_patch(FancyBboxPatch((x - 0.55, yc - 0.40), 1.10, 0.80,
                                        boxstyle="round,pad=0.03",
                                        fc=bg, ec=fg,
                                        lw=1.6 if gifted else 1.0))
            ax.text(x, yc, str(v), fontsize=7.0, ha="center", va="center",
                    color=fg, fontweight="bold")
    ax.text(ox + 1.90, 1.45, note, fontsize=5.7, ha="center", va="center",
            color=INK)


parity_table(0.55, "N1: the bridge differential",
             n1, ["russellian", "physicalism"], ["RM", "PHYS"],
             "the unaligned scorers (4-b/c/d) score the placement\n"
             "and the identity at parity; each camp gifts itself +1;\n"
             "doc 90 filed the 4-e differential as the criterion:\n"
             "placement is not a bridge, identity is a bought one",
             {("4-a", "physicalism"), ("4-e", "russellian")})
parity_table(5.45, "N2: the deletion differential",
             n2, ["russellian", "illusionism"], ["RM", "ILLU"],
             "four of five scorers (the neutral included) score\n"
             "executed deletion above promissory acquaintance;\n"
             "doc 90 reversed it for everyone: deletion was charged\n"
             "the price, acquaintance was certified",
             {("4-e", "russellian")})
ax.text(5.0, 0.45, "the two differentials that produced the sole survivor "
        "are camp differentials, not criteria facts",
        fontsize=6.2, ha="center", va="center", color=PURPLE,
        fontweight="bold")
ax.set_title("(c) the parity cells (Part 3)", fontsize=8.6,
             fontweight="bold")

# --------------------------------------------- (d) the amended verdict
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(5.0, 9.45, "THE REGISTRAR'S QUESTION", fontsize=8.2, ha="center",
        color=INK, fontweight="bold")
ax.text(5.0, 8.80, "Did doc 90's tribunal assume the conclusion?",
        fontsize=7.0, ha="center", color=PURPLE, fontweight="bold")

ax.add_patch(FancyBboxPatch((0.30, 6.35), 9.35, 2.10,
                            boxstyle="round,pad=0.10", fc="#f9e3e0",
                            ec=RED, lw=1.8))
ax.text(5.0, 8.10, "REFUTED AS FILED", fontsize=7.6, ha="center",
        color=RED, fontweight="bold")
ax.text(0.62, 7.05, "the SOLE SURVIVOR cap does not survive neutral "
        "re-scoring: survival is\n"
        "criterion-relative (spearman 0.2; the sole survivor falls 1 -> 7);\n"
        "the differential that produced it is camp-relative (the parity "
        "cells)",
        fontsize=5.9, ha="left", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 4.05), 9.35, 2.10,
                            boxstyle="round,pad=0.10", fc="#e8f4ec",
                            ec=GREEN, lw=1.8))
ax.text(5.0, 5.80, "AMENDED FORM SURVIVES", fontsize=7.6, ha="center",
        color=GREEN, fontweight="bold")
ax.text(0.62, 4.75, "RM-placed is the field-position most isomorphic to "
        "the\ncorpus's own grammar - a convergence claim, marked, not a\n"
        "superiority claim; the fork (Prop. 86) stands; dualism's failure\n"
        "is upheld on earned ground; the theorems are untouched",
        fontsize=5.9, ha="left", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 1.95), 9.35, 1.85,
                            boxstyle="round,pad=0.10", fc="#fdf6ec",
                            ec=GOLD, lw=1.6))
ax.text(5.0, 3.45, "THE TRIBUNAL CAMP-MARKER (the audit's own result)",
        fontsize=7.0, ha="center", color=GOLD, fontweight="bold")
ax.text(5.0, 2.60, "every tribunal statable in a grammar privileges that\n"
        "grammar's homologous theories; every sole-survivor claim carries\n"
        "its grammar's marker - the Camp-Marker Theorem, extended from\n"
        "positions to tribunals",
        fontsize=5.9, ha="center", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 0.45), 9.35, 1.25,
                            boxstyle="round,pad=0.10", fc="#f4f6f6",
                            ec=GRAY, lw=1.4))
ax.text(5.0, 1.38, "consequence: the QM tribunal runs under BOTH lenses,\n"
        "and files the divergence - where the lenses agree, with "
        "confidence; where they invert, under the marker",
        fontsize=6.0, ha="center", va="center", color=INK)
ax.set_title("(d) the verdict (Parts 7-9)", fontsize=8.6, fontweight="bold")

fig.suptitle("The bias audit of the tribunal (corpus doc 92): five blind "
             "scorers, the neutral criteria, the inversion of the verdict, "
             "and the amended filing",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/doc92_figure1.png", dpi=170)
print("wrote", f"{D}/doc92_figure1.png")
