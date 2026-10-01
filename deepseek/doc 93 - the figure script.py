#!/usr/bin/env python3
"""Figure for corpus doc 93 (the quantum tribunal under the dual-lens
protocol): the blind battery of five scorers under the neutral five
criteria (Lens B) beside the corpus's four campaigns (Lens A) - the
battery matrix with the family columns and the unanimous column
marked, the divergence slopegraph from Lens A's filing order to the
battery's mean-rank order (spearman 0.533), the N1 dynamics/ontology
split and the persona ranges (evidence vs grammar, the doc 92 pattern
replicated), and the dual-lens verdict (agreements with confidence,
inversions under the Tribunal Camp-Marker, no sole-survivor cap).
English labels, constrained_layout, house palette; reads
doc93_results.json emitted by the dual-lens instrument."""
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

with open(f"{D}/doc93_results.json", encoding="utf-8") as f:
    R = json.load(f)

SCORERS = ["6-a", "6-b", "6-c", "6-d", "6-e"]
SHORT = {"copenhagen": "COPN", "wigner": "WIGN", "grw": "GRW",
         "everett": "EVER", "bohm": "BOHM", "qbism": "QBIS",
         "rqm": "RQM", "histories": "HIST", "many-minds": "MMIN"}
ORDER = ["copenhagen", "wigner", "grw", "everett", "bohm",
         "qbism", "rqm", "histories", "many-minds"]
persona = {"6-a": "everettian", "6-b": "bohmian", "6-c": "collapse",
           "6-d": "neutral", "6-e": "participatory"}

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# --------------------------------------------- (a) the battery matrix
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

x0, colw = 2.15, 0.86
y0, dy = 7.75, 1.18
for j, t in enumerate(ORDER):
    x = x0 + j * colw
    ax.text(x, 9.30, SHORT[t], fontsize=6.2, ha="center", va="center",
            color=INK, fontweight="bold")
    if t in ("rqm", "qbism"):
        ax.add_patch(FancyBboxPatch((x - 0.44, 2.32), 0.88, 6.28,
                                    boxstyle="round,pad=0.03",
                                    fc="#fdf6ec", ec=GOLD, lw=1.8))
    if t == "grw":
        ax.add_patch(FancyBboxPatch((x - 0.44, 2.32), 0.88, 6.28,
                                    boxstyle="round,pad=0.03",
                                    fc="#e8f4ec", ec=GREEN, lw=1.8))
# Lens A rank strip above the matrix
for j, t in enumerate(ORDER):
    x = x0 + j * colw
    ax.text(x, 8.62, f"LA{R['lens_a_rank'][t]}", fontsize=5.6,
            ha="center", va="center", color=PURPLE,
            fontweight="bold")
ax.text(0.62, 8.62, "lens A rank", fontsize=5.6, ha="right", va="center",
        color=PURPLE)
for i, s in enumerate(SCORERS):
    yc = y0 - i * dy
    ax.text(1.72, yc, f"{s} {persona[s]}", fontsize=6.0, ha="right",
            va="center", color=INK,
            fontweight="bold" if s == "6-d" else "normal")
    for j, t in enumerate(ORDER):
        v = R["totals"][s][t]
        fg, bg = ((GREEN, "#e8f4ec") if v >= 9 else
                  (GOLD, "#fdf3dd") if v >= 7 else
                  (ORANGE, "#fbe9dc") if v >= 5 else (RED, "#f9e3e0"))
        x = x0 + j * colw
        ax.add_patch(FancyBboxPatch((x - 0.40, yc - 0.46), 0.80, 0.92,
                                    boxstyle="round,pad=0.03",
                                    fc=bg, ec=fg, lw=1.1))
        ax.text(x, yc, str(v), fontsize=6.8, ha="center", va="center",
                color=fg, fontweight="bold")
# battery mean-rank strip below the matrix
for j, t in enumerate(ORDER):
    x = x0 + j * colw
    ax.text(x, 1.55, f"LB{R['battery_order_rank'][t]}", fontsize=5.6,
            ha="center", va="center", color=BLUE, fontweight="bold")
ax.text(0.62, 1.55, "lens B rank", fontsize=5.6, ha="right", va="center",
        color=BLUE)
ax.text(5.0, 0.85, "totals /10. gold columns = lens A's family (RQM, "
        "QBIS - the grammar's placements);\ngreen column = GRW, "
        "unanimous 9/9/9/9/9 - every scorer, every persona",
        fontsize=6.0, ha="center", va="center", color=GRAY,
        fontstyle="italic")
ax.text(5.0, 0.22, "all five blind to lens A, the corpus, and the "
        "verdict; theory order rotated per scorer",
        fontsize=5.8, ha="center", va="center", color=GRAY)
ax.set_title("(a) the blind battery (lens B): 5 scorers x 9 "
             "interpretations", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (b) the divergence
ax = axes[0, 1]
ax.set_xlim(-1.5, 2.65)
ax.set_ylim(10.4, -0.6)
ax.axis("off")

la, bo = R["lens_a_rank"], R["battery_order_rank"]
CLASS = {}
for t in ORDER:
    d = bo[t] - la[t]
    if abs(d) >= 3:
        CLASS[t] = RED if d > 0 else GREEN
    elif abs(d) == 2:
        CLASS[t] = GOLD
    else:
        CLASS[t] = GRAY
# stagger labels where ranks tie on a side
left_counts, right_counts = {}, {}
for t in ORDER:
    left_counts[la[t]] = left_counts.get(la[t], 0) + 1
    right_counts[bo[t]] = right_counts.get(bo[t], 0) + 1
seen_left, seen_right = {}, {}
for t in ORDER:
    x1, x2 = 0.0, 1.15
    y1, y2 = la[t], bo[t]
    k = seen_left.get(la[t], 0)
    seen_left[la[t]] = k + 1
    if left_counts[la[t]] > 1:
        x1 = 0.0 + 0.55 * (left_counts[la[t]] - 1 - k)
        y1 = la[t] + (0.34 if k % 2 == 0 else -0.34)
    k2 = seen_right.get(bo[t], 0)
    seen_right[bo[t]] = k2 + 1
    if right_counts[bo[t]] > 1:
        x2 = 1.15 - 0.55 * (right_counts[bo[t]] - 1 - k2)
        y2 = bo[t] + (0.34 if k2 % 2 == 0 else -0.34)
    ax.plot([x1, x2], [y1, y2], color=CLASS[t], lw=2.2, alpha=0.9,
            zorder=2)
ax.text(-1.35, 0.30, "lens A  (the four campaigns,\nthe corpus's filing "
        "order)", fontsize=6.4, ha="left", va="center", color=PURPLE)
ax.text(2.55, 0.30, "lens B  (the neutral five,\nthe battery's mean "
        "ranks)", fontsize=6.4, ha="right", va="center", color=BLUE)
for t in ORDER:
    d = bo[t] - la[t]
    lab = f"{SHORT[t]} {la[t]}" + ("" if d == 0 else
                                   f" \u2192 {bo[t]}")
    ax.text(-0.08, la[t], lab, fontsize=6.3, ha="right", va="center",
            color=CLASS[t], fontweight="bold" if abs(d) >= 3 else "normal")
    ax.text(1.23, bo[t], f"{bo[t]} {SHORT[t]}", fontsize=6.3, ha="left",
            va="center", color=CLASS[t], fontweight="bold" if abs(d) >= 3
            else "normal")
ax.text(0.575, -0.35, "spearman(lens A, lens B) = "
        f"{R['divergence']['spearman_lens_a_vs_battery']:.3f}",
        fontsize=6.6, ha="center", va="center", color=INK,
        fontweight="bold")
ax.text(0.575, 10.05, "red = the family falls (RQM 1\u21924, QBIS "
        "2\u21926) - the grammar's family does not sweep;\ngreen = GRW "
        "rises 6\u21921, unanimous - the grammar's bought bridge is the "
        "criteria's clearest physics;\ngray = agreement (EVER 3\u21923, "
        "WIGN 9\u21929, HIST 5\u21925); gold = mild (BOHM 4\u21922)",
        fontsize=5.6, ha="center", va="center", color=GRAY,
        fontstyle="italic")
ax.set_title("(b) the divergence: the two lenses on the same field",
             fontsize=8.6, fontweight="bold")

# --------------------------------------------- (c) the N1 split + ranges
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# left half: the N1 split cells (GRW vs RQM by scorer)
ax.text(2.35, 9.35, "the N1 split (unanimous)", fontsize=7.0, ha="center",
        va="center", color=INK, fontweight="bold")
ax.text(1.55, 8.62, "scorer", fontsize=6.0, ha="center", va="center",
        color=GRAY)
ax.text(2.85, 8.62, "GRW N1", fontsize=6.0, ha="center", va="center",
        color=GREEN, fontweight="bold")
ax.text(4.15, 8.62, "RQM N1", fontsize=6.0, ha="center", va="center",
        color=RED, fontweight="bold")
n1g = R["parity_and_gifts"]["n1_split"]["cells"]["grw"]
n1r = R["parity_and_gifts"]["n1_split"]["cells"]["rqm"]
for i, s in enumerate(SCORERS):
    yc = 7.85 - i * 0.92
    ax.text(1.55, yc, s, fontsize=6.2, ha="center", va="center", color=INK,
            fontweight="bold" if s == "6-d" else "normal")
    for x, v, fg, bg in ((2.85, n1g[s], GREEN, "#e8f4ec"),
                         (4.15, n1r[s], RED, "#f9e3e0")):
        ax.add_patch(FancyBboxPatch((x - 0.38, yc - 0.36), 0.76, 0.72,
                                    boxstyle="round,pad=0.03", fc=bg,
                                    ec=fg, lw=1.1))
        ax.text(x, yc, str(v), fontsize=6.8, ha="center", va="center",
                color=fg, fontweight="bold")
ax.text(2.85, 3.05, "the flash-law is 'entailed by the dynamics' -\n2 "
        "from every scorer;\nthe interaction-fact is 'a stipulated "
        "primitive' -\n1 from every scorer.\nthe rubric reads dynamics-"
        "first;\nlens A prices the same pair in reverse.\nthe "
        "differential is criterion-relative,\nmeasured twice, valence "
        "flipped", fontsize=5.7, ha="center", va="top", color=GRAY,
        fontstyle="italic")
ax.text(2.85, 0.62, "(the doc 92 N1 parity cell, mirrored)", fontsize=5.8,
        ha="center", va="center", color=GRAY)

# right half: persona ranges
ax.text(7.55, 9.35, "persona ranges (totals, max-min)", fontsize=7.0,
        ha="center", va="center", color=INK, fontweight="bold")
rng = R["range_total"]
srt = sorted(ORDER, key=lambda t: -rng[t])
for i, t in enumerate(srt):
    yc = 8.55 - i * 0.78
    v = rng[t]
    fg = GREEN if t == "grw" else (RED if t == "everett" else GRAY)
    ax.text(5.85, yc, SHORT[t], fontsize=6.2, ha="right", va="center",
            color=INK)
    ax.barh(yc, v * 1.05, left=6.05, height=0.44, color=fg, alpha=0.85)
    ax.text(6.05 + v * 1.05 + 0.18, yc, str(v), fontsize=6.4, ha="left",
            va="center", color=fg, fontweight="bold")
ax.text(7.55, 1.35, "spread collapses where evidence is (GRW: 9,9,9,9,9 "
        "- range 0, unanimous 1st for every scorer);\nspread opens "
        "where grammar is (EVERETT: 10,6,6,7,8 - range 4, the field's "
        "widest). doc 92 measured\nthe same pattern (RM range 3; the "
        "tested entries 1-2). measured twice now: it is a regularity, "
        "not an anecdote.",
        fontsize=5.7, ha="center", va="center", color=GRAY,
        fontstyle="italic")
ax.set_title("(c) the parity cells and the persona ranges: evidence vs "
             "grammar", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (d) the verdict
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

boxes = [
    (GREEN, "WHERE THE LENSES AGREE - FILE WITH CONFIDENCE",
     "EVERETT 3/3 - the placement both grammars certify (the index as\n"
     "observer column; classicality emergent-asymptotic).\n"
     "WIGNER 9/9 - the QM-dualism, upheld the way dualism was: the mind\n"
     "bought with no mechanism, no anomalies, N4 = 0 from every scorer.\n"
     "HISTORIES 5/5 - the mixed framework (decoherence law earned; the\n"
     "realm rule bought).  COPENHAGEN 7/8 - the cut priced low.  M-MINDS\n"
     "8/7 - the R4 patch, priced in both."),
    (RED, "WHERE THEY INVERT - FILED UNDER THE TRIBUNAL CAMP-MARKER",
     "GRW: lens A 6th \u2192 lens B unanimous 1st (9,9,9,9,9) - the\n"
     "grammar's clearest bought bridge is the criteria's clearest\n"
     "physics: the one live experimental program in the field.\n"
     "RQM 1\u21924, QBISM 2\u21926 - the grammar's family does not "
     "sweep.\nThe inversions are symmetric; each lens has its family;\n"
     "neither lens is the field's truth."),
    (GOLD, "NO SOLE-SURVIVOR CAP",
     "the protocol was designed so that this verdict cannot be what\n"
     "doc 90's was. the two-lens topology is the verdict. spearman\n"
     "0.533 - higher than doc 92's 0.2, and still not a fact about\n"
     "the field: the divergence is concentrated in exactly three\n"
     "cells, and all three are grammar-relative."),
    (PURPLE, "THE REDUCTION",
     "the QM field, run through both lenses, collapses onto C5:\n"
     "state-as-world (EVERETT, BOHM, GRW, HISTORIES - the residue\n"
     "placed at index / particle / flash / realm) vs state-as-relation\n"
     "(RQM, QBISM - the absolute deleted, the event kept). the\n"
     "measurement problem is the corpus's wall in physical dress: the\n"
     "non-derivability of the definite from the unitary is the\n"
     "non-derivability of the inside from the extensional."),
]
y = 9.55
for color, head, body in boxes:
    h = 0.62 + 0.335 * (body.count("\n") + 1)
    ax.add_patch(FancyBboxPatch((0.25, y - h), 9.5, h,
                                boxstyle="round,pad=0.08",
                                fc="#fafafa", ec=color, lw=1.5))
    ax.text(0.55, y - 0.33, head, fontsize=6.8, ha="left", va="center",
            color=color, fontweight="bold")
    ax.text(0.55, y - 0.72, body, fontsize=5.7, ha="left", va="top",
            color=INK, linespacing=1.45)
    y -= h + 0.42
ax.set_title("(d) the dual-lens verdict", fontsize=8.6, fontweight="bold")

fig.suptitle("Doc 93 - the quantum tribunal under the dual-lens "
             "protocol: two lenses, one field, the divergence filed",
             fontsize=9.6, fontweight="bold")

fig.savefig(f"{D}/doc93_figure1.png", dpi=170)
print("doc 93 figure 1 written:", f"{D}/doc93_figure1.png")
