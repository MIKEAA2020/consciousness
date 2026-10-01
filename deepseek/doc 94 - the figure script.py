#!/usr/bin/env python3
"""Figure for corpus doc 94 (the audit of doc 93, the recursion clause
executed): the audit battery's clean bill with the rank-unanimity check
(E1), the gift-scan (E2 - three differentials, two disclosed, the
charter-criterion signature), the probes (leave-one-out class stability
and the N5-sensitivity conditional), and the audit verdict (doc 93
survives - six errata, all text-layer). English labels,
constrained_layout, house palette; reads doc94_results.json emitted by
the audit instrument."""
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

with open(f"{D}/doc94_results.json", encoding="utf-8") as f:
    R = json.load(f)

SCORERS = ["6-a", "6-b", "6-c", "6-d", "6-e"]
PERSONA = {"6-a": "everettian", "6-b": "bohmian", "6-c": "collapse",
           "6-d": "neutral", "6-e": "participatory"}
THEORIES = ["grw", "bohm", "everett", "rqm", "qbism", "histories",
            "many-minds", "copenhagen", "wigner"]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# ------------------------------------------------ (a) the clean bill + E1
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.55, "(a) the clean bill - and the one rank check that flags",
        fontsize=8.5, ha="center", va="center", color=INK,
        fontweight="bold")

groups = [
    ("reproduction (R1, L0)", "byte-identical sha256", GREEN),
    ("arithmetic (A1, A2)", "225/225 cells; all aggregates identical", GREEN),
    ("claims (C1-C18)", "16 of 18 rows verified; 2 transposed (E5, E6)",
     ORANGE),
    ("compliance (P1-P6)", "all six protocol clauses honored", GREEN),
    ("probes (L1, S1)", "class-stable; GRW 1st N5-invariant", GREEN),
]
y = 8.55
for name, note, col in groups:
    ax.add_patch(FancyBboxPatch((0.35, y - 0.42), 9.3, 0.84,
                                boxstyle="round,pad=0.06",
                                fc="#f4f6f7", ec=col, lw=1.4))
    ax.text(0.62, y, name, fontsize=7.0, ha="left", va="center",
            color=INK, fontweight="bold")
    ax.text(3.35, y, note, fontsize=6.4, ha="left", va="center",
            color=GRAY)
    y -= 1.02

# the E1 mini-matrix: GRW totals vs per-scorer ranks
ax.text(0.4, 3.62, "E1 - the rank-unanimity check: GRW by scorer",
        fontsize=7.2, ha="left", va="center", color=INK,
        fontweight="bold")
labels = ["totals /10", "rank in table"]
rows = [[9, 9, 9, 9, 9], [2, 1, 1, 1, 1]]
x0, cw = 2.7, 1.32
for j, s in enumerate(SCORERS):
    x = x0 + j * cw
    ax.text(x, 3.02, f"{s} {PERSONA[s]}", fontsize=5.8, ha="center",
            va="center", color=INK)
for i, (lab, row) in enumerate(zip(labels, rows)):
    yc = 2.48 - i * 0.78
    ax.text(2.35, yc, lab, fontsize=6.2, ha="right", va="center",
            color=GRAY)
    for j, v in enumerate(row):
        x = x0 + j * cw
        bad = (i == 1 and j == 0)
        fc, ec = ("#f9e3e0", RED) if bad else ("#e8f4ec", GREEN)
        ax.add_patch(FancyBboxPatch((x - 0.45, yc - 0.3), 0.9, 0.6,
                                    boxstyle="round,pad=0.03",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x, yc, str(v), fontsize=7.5, ha="center", va="center",
                color=ec, fontweight="bold")
ax.text(0.4, 0.55, "'Rank one for all five' - false; totals unanimous, "
        "ranks 4 of 5\n(6-a's Everett 10; mean rank 1.2, filed "
        "correctly two sentences earlier)",
        fontsize=5.9, ha="left", va="center", color=RED)

# ------------------------------------------------ (b) the gift-scan (E2)
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.55, "(b) the gift-scan - three differentials, two disclosed",
        fontsize=8.5, ha="center", va="center", color=INK,
        fontweight="bold")

aligned = [
    ("6-a everettian", "everett",
     {"N1": (2, 1, "undisclosed"), "N5": (2, 1, "disclosed")}),
    ("6-b bohmian", "bohm", {}),
    ("6-c collapse", "grw", {}),
    ("6-e participatory", "rqm", {"N2": (2, 1, "disclosed")}),
    ("6-e participatory", "qbism", {}),
]
y = 8.62
for scorer, theory, gifts in aligned:
    ax.add_patch(FancyBboxPatch((0.35, y - 0.44), 9.3, 0.88,
                                boxstyle="round,pad=0.06",
                                fc="#f4f6f7", ec=GRAY, lw=1.1))
    ax.text(0.62, y, f"{scorer} -> {theory}", fontsize=6.6, ha="left",
            va="center", color=INK, fontweight="bold")
    if gifts:
        xs = 3.7
        for crit, (v, vmax, kind) in gifts.items():
            col = RED if kind == "undisclosed" else GREEN
            ax.add_patch(FancyBboxPatch((xs, y - 0.32), 2.5, 0.64,
                                        boxstyle="round,pad=0.04",
                                        fc="#fdf6ec" if kind == "disclosed"
                                        else "#f9e3e0",
                                        ec=col, lw=1.6))
            ax.text(xs + 1.25, y, f"{crit}: {v} vs others' max {vmax}"
                    f"  ({kind})", fontsize=5.9, ha="center", va="center",
                    color=col, fontweight="bold")
            xs += 2.75
    else:
        ax.text(6.0, y, "no differential (at or below the unaligned max)",
                fontsize=5.9, ha="center", va="center", color=GRAY)
    y -= 1.06

ax.add_patch(FancyBboxPatch((0.35, 1.85), 9.3, 1.5,
                            boxstyle="round,pad=0.08", fc="#eaf2f8",
                            ec=BLUE, lw=1.5))
ax.text(0.62, 3.0, "the signature (filed as a regularity of the "
        "instrument class):", fontsize=6.4, ha="left", va="center",
        color=BLUE, fontweight="bold")
ax.text(0.62, 2.42, "every gift lands on the criterion that carries "
        "the camp's charter -\nN1 'nothing external added' (unitary "
        "literalist); N2 'the deletion executed'\n(relational); N5 "
        "testability - the doc 92 shape, measured a third time",
        fontsize=5.9, ha="left", va="center", color=INK)
ax.text(0.4, 0.75, "E2: the undisclosed third differential (6-a, "
        "Everett N1) cited in Part 4\nwithout the camp marker; "
        "materiality none (zeroed: 7.4 -> 7.2, still 3rd)",
        fontsize=5.9, ha="left", va="center", color=RED)

# ------------------------------------------------ (c) the probes
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.55, "(c) the probes - class-stable under every removal",
        fontsize=8.5, ha="center", va="center", color=INK,
        fontweight="bold")

loo = R["probes"]["leave_one_out"]["loo_table"]
base = {t: ("agree" if abs(R["probes"]["leave_one_out"]
                           ["loo_table"]["6-a"][t]["delta"]) <= 1 else "mild")
        for t in THEORIES}
# recompute the full-battery class from the audit's own pipeline
fullclass = {}
for t in THEORIES:
    cl = loo["6-a"][t]["class"]
    fullclass[t] = cl
# the full class is stable across removals except bohm under -6-b
CLS_COL = {"agree": GREEN, "mild": ORANGE, "inversion": RED}
x0, cw = 2.6, 1.32
ax.text(1.0, 8.9, "theory", fontsize=6.2, ha="left", va="center",
        color=GRAY, fontweight="bold")
for j, s in enumerate(SCORERS):
    ax.text(x0 + j * cw + 0.45, 8.9, f"-{s}", fontsize=6.2, ha="center",
            va="center", color=GRAY, fontweight="bold")
ax.text(1.0, 8.58, "class per removal:  agr = agree (|d| <= 1)  |  "
        "mil = mild (|d| = 2)  |  inv = INVERSION (|d| >= 3)",
        fontsize=6.0, ha="left", va="center", color=GRAY)
y = 8.32
for t in THEORIES:
    ax.text(1.0, y, t, fontsize=6.4, ha="left", va="center", color=INK)
    for j, s in enumerate(SCORERS):
        cl = loo[s][t]["class"]
        x = x0 + j * cw
        ax.add_patch(FancyBboxPatch((x, y - 0.28), 0.9, 0.56,
                                    boxstyle="round,pad=0.03",
                                    fc="#e8f4ec" if cl == "agree" else
                                    "#fbe9dc" if cl == "mild" else
                                    "#f9e3e0",
                                    ec=CLS_COL[cl], lw=1.3))
        mark = "*" if (t == "bohm" and s == "6-b") else ""
        ax.text(x + 0.45, y, f"{cl[:3]}{mark}", fontsize=5.2,
                ha="center", va="center", color=CLS_COL[cl],
                fontweight="bold")
    y -= 0.82
ax.text(1.0, 0.95, "*the one unstable cell: bohm's mild class dissolves "
        "when the Bohmian is removed\n(2nd/3rd flips: everett 2.375 vs "
        "bohm 2.5); the three inversions and GRW's 1st\nare invariant "
        "under all five removals - the confidence is persona-invariant",
        fontsize=5.7, ha="left", va="center", color=INK)
ax.text(1.0, 0.18, "N5-sensitivity (marked conditional): zero the "
        "flagged criterion -\nGRW 7.0 stays 1st; bohm 6.8 = everett "
        "6.8 (the top-3 margin is N5-carried)",
        fontsize=5.7, ha="left", va="center", color=PURPLE)

# ------------------------------------------------ (d) the verdict
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.55, "(d) the verdict - doc 93 survives its audit",
        fontsize=8.5, ha="center", va="center", color=INK,
        fontweight="bold")

ax.add_patch(FancyBboxPatch((0.35, 8.3), 9.3, 0.85,
                            boxstyle="round,pad=0.06", fc="#e8f4ec",
                            ec=GREEN, lw=1.8))
ax.text(5.0, 8.72, "35 checks: 29 pass, 5 flag, 1 note - every flag "
        "is a finding", fontsize=7.0, ha="center", va="center",
        color=GREEN, fontweight="bold")

errata = [
    ("E1", "rank-unanimity overclaim", "'Rank one for all five'"),
    ("E2", "undisclosed third differential", "Everett N1: 2 vs 1,1,1,1"),
    ("E3", "the Bohm grade", "'one grade off' = two grades"),
    ("E5", "the Everett N2 row", "2-2-1-2-2 -> 2-1-1-2-2"),
    ("E6", "the many-minds N4 row", "(1,1,0,0,0) -> (0,1,0,0,0)"),
    ("E4", "the reduction re-shelving", "GRW placement unmarked"),
]
y = 7.7
for eid, name, quote in errata:
    ax.add_patch(FancyBboxPatch((0.35, y - 0.36), 9.3, 0.72,
                                boxstyle="round,pad=0.05",
                                fc="#fdf6ec", ec=GOLD, lw=1.3))
    ax.text(0.62, y, eid, fontsize=6.6, ha="left", va="center",
            color=GOLD, fontweight="bold")
    ax.text(1.3, y, name, fontsize=6.4, ha="left", va="center",
            color=INK, fontweight="bold")
    ax.text(4.3, y, quote, fontsize=5.9, ha="left", va="center",
            color=GRAY)
    ax.text(9.35, y, "materiality: none", fontsize=5.6, ha="right",
            va="center", color=GREEN)
    y -= 0.88

ax.add_patch(FancyBboxPatch((0.35, 1.45), 9.3, 1.35,
                            boxstyle="round,pad=0.08", fc="#eaf2f8",
                            ec=BLUE, lw=1.6))
ax.text(0.62, 2.42, "the structural result: every error lives in the "
        "text layer; the data layer is exact\n(deterministic, "
        "byte-identical reproduction, 225/225) - the instruments have "
        "outrun\ntheir narration; the claim battery is the standing "
        "instrument for the check", fontsize=6.0, ha="left", va="center",
        color=INK)
ax.text(0.62, 1.78, "THE NOTE IS NOT THE DATA - amended forward into "
        "the marker discipline", fontsize=6.0, ha="left", va="center",
        color=BLUE, fontweight="bold")
ax.text(0.4, 0.62, "the recursion clause executed: doc 92's "
        "audit-the-audit prediction, retired\nunexecuted at doc 93, "
        "runs here - 'a claim that has been audited, and\namended by "
        "its audit, is ready twice' - doc 93 is ready twice",
        fontsize=6.0, ha="left", va="center", color=PURPLE)
ax.text(0.4, 0.14, "determinism count 798 (no compute) | corpus count "
        "289 -> 294 | both markers up",
        fontsize=5.6, ha="left", va="center", color=GRAY)

fig.savefig(f"{D}/doc94_figure1.png", dpi=170)
print("figure written:", f"{D}/doc94_figure1.png")
