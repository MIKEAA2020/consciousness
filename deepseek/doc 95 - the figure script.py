#!/usr/bin/env python3
"""Figure for corpus doc 95 (the outside charge, adjudicated): the record
the charge meets (the timeline), the charge decomposed into its three
accusations against the record, the miss inventory with the mirror
finding, and the ruling on the registrar's either/or. English labels,
constrained_layout, house palette; reads the synced doc 92 battery
results for the numbers it cites (THE NOTE IS NOT THE DATA)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE, GOLD = "#e67e22", "#b7950b"
INK = "#212121"

# --- verify the cited numbers against the registered battery (doc 94 clause)
R = json.load(open("/home/z/my-project/scripts/bias_audit_results.json",
                   encoding="utf-8"))
assert R["mean_rank"]["russellian"] == 7.0
assert R["correlation"]["spearman_vs_mean"] == 0.2
assert R["inversion"]["rm_n5_by_scorer"] == {"4-a": 1, "4-b": 1, "4-c": 0,
                                             "4-d": 1, "4-e": 1}
assert R["range_total"]["russellian"] == 3
assert R["verdict"]["filing"].startswith("REFUTED AS FILED")

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# ------------------------------------------------ (a) the timeline
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

TL = [
    ("docs 1-6  -  the founding self-kills",
     "the registrar's own proof refuted (32 findings); the Russellian bridge P*d killed before any tribunal",
     GREEN, "#eafaf0"),
    ("docs 24-89  -  the instruments, built inward",
     "the Chu-cost formalism, Lemma 85, Theorem II, Prop 86 - proved without touching the field's nine theories",
     BLUE, "#eaf2f8"),
    ("doc 90  -  the tribunal of nine",
     "sole-survivor cap filed WITH the bounds; the Camp-Marker declared up in the Status header",
     PURPLE, "#f5eef8"),
    ("doc 92 (10:13Z)  -  the bias audit",
     "the cap REFUTED AS FILED: five blind scorers, de-theorized criteria, RM mean rank 7 of 9",
     ORANGE, "#fdf3e7"),
    ("doc 93 (11:06Z)  -  the corrected protocol, run",
     "the QM tribunal under both lenses; no cap issued; the GRW correction made binding",
     ORANGE, "#fdf3e7"),
    ("doc 94 (12:04Z)  -  the audit of the audit",
     "doc 93 survives 35 checks; byte-identical reproduction; the hygiene gate instituted",
     ORANGE, "#fdf3e7"),
    ("THE CHARGE ARRIVES  -  'doc 90 was biased'",
     "one sentence, no protocol, no record - post-refutation in the record's own timeline",
     RED, "#fdedec"),
]
ax.plot([0.55, 0.55], [0.55, 9.05], color=GRAY, lw=1.2, zorder=1)
y_top, bh, gap = 9.05, 1.02, 0.22
for i, (head, detail, ec, fc) in enumerate(TL):
    y = y_top - (i + 1) * (bh + gap) + gap
    ax.add_patch(FancyBboxPatch((0.85, y), 8.9, bh,
                                boxstyle="round,pad=0.06", fc=fc, ec=ec,
                                lw=1.5 if ec != RED else 1.8,
                                ls="-" if ec != RED else (0, (4, 2)),
                                zorder=2))
    ax.add_patch(Circle((0.55, y + bh / 2), 0.10, fc=ec, ec=ec, zorder=3))
    ax.text(1.05, y + bh * 0.68, head, fontsize=6.4, fontweight="bold",
            ha="left", va="center", color=ec, zorder=3)
    ax.text(1.05, y + bh * 0.28, detail, fontsize=5.4, ha="left",
            va="center", color=INK, zorder=3)
ax.set_title("(a) the record the charge meets (Part 0)", fontsize=8.6,
             fontweight="bold")

# ------------------------------------ (b) the charge, decomposed
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ROWS = [
    ("1 - CIRCULARITY", "the instruments were built to produce the verdict",
     "FAILS THE RECORD", RED, "#fdedec",
     ["the instruments pre-date the verdict by 84 documents; their first and",
      "longest application was against the corpus's own camp (P*d, docs 2-6);",
      "the theorems were proved in a program that never mentions the field.",
      "what stands instead: criterion-relativity at rubric grade, not",
      "circularity at instrument grade (doc 92, Part 7: A0 + 2 auxiliaries)"]),
    ("2 - CRITERION-RELATIVITY", "the rubric carried the grammar's shape",
     "TRUE - ALREADY FILED", ORANGE, "#fdf3e7",
     ["doc 92, pre-registered, five blind scorers: RM 1 -> 7 (mean rank 7.0;",
      "Spearman 0.2); the parity cells (unaligned 1/1, aligned self-gifts +1);",
      "the quietism premium (RM falsifiability 1,1,0,1,1 - the tribunal filed",
      "the same fact as its best answer); uniqueness partly by absorption.",
      "Numbers re-verified this session against the registered battery."]),
    ("3 - TACTICAL BIAS", "the corpus hid its steering",
     "REFUTED BY DISCLOSURE", BLUE, "#eaf2f8",
     ["the marker was declared up before the tribunal convened; the bounds",
      "were filed with the verdict; Part 4's first bound - 'another camp,",
      "running other instruments, would file other survivors' - is the",
      "refutation's method stated in advance, in the verdict's own document.",
      "doc 92: 'structural and disclosed, not tactical and hidden'"]),
]
y_top, bh = 9.15, 2.85
for i, (name, gloss, chip, ec, fc, lines) in enumerate(ROWS):
    y = y_top - (i + 1) * bh - i * 0.18
    ax.add_patch(FancyBboxPatch((0.25, y), 9.5, bh,
                                boxstyle="round,pad=0.06", fc=fc, ec=ec,
                                lw=1.5, zorder=2))
    ax.text(0.5, y + bh - 0.42, f"{name}  -  '{gloss}'", fontsize=6.3,
            fontweight="bold", ha="left", va="center", color=ec, zorder=3)
    ax.add_patch(FancyBboxPatch((7.35, y + bh - 0.62), 2.15, 0.42,
                                boxstyle="round,pad=0.04", fc="white",
                                ec=ec, lw=1.2, zorder=3))
    ax.text(8.42, y + bh - 0.41, chip, fontsize=5.3, fontweight="bold",
            ha="center", va="center", color=ec, zorder=4)
    for j, ln in enumerate(lines):
        ax.text(0.5, y + bh - 0.95 - j * 0.34, ln, fontsize=5.2,
                ha="left", va="center", color=INK, zorder=3)
ax.set_title("(b) the charge, decomposed (Parts 1-3)", fontsize=8.6,
             fontweight="bold")

# -------------------------- (c) the miss inventory and the mirror
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# left: what the reviewer read
ax.add_patch(FancyBboxPatch((0.25, 5.6), 4.5, 3.5,
                            boxstyle="round,pad=0.06", fc="#f4f6f6",
                            ec=GRAY, lw=1.4))
ax.text(2.5, 8.72, "WHAT THE REVIEWER READ", fontsize=6.6,
        fontweight="bold", ha="center", va="center", color=GRAY)
ax.text(0.5, 8.15, "doc 90, alone:", fontsize=6.0, fontweight="bold",
        ha="left", va="center", color=INK)
for j, ln in enumerate([
        "the cap - dominant, memorable",
        "the bounds - present, quieter",
        "the Camp-Marker - in the header",
        "a one-document sample of a",
        "94-document run"]):
    ax.text(0.75, 7.65 - j * 0.42, ln, fontsize=5.5, ha="left",
            va="center", color=INK)

# right: the record on both sides
ax.add_patch(FancyBboxPatch((5.05, 5.6), 4.7, 3.5,
                            boxstyle="round,pad=0.06", fc="#eaf2f8",
                            ec=BLUE, lw=1.4))
ax.text(7.4, 8.72, "THE RECORD, BOTH SIDES", fontsize=6.6,
        fontweight="bold", ha="center", va="center", color=BLUE)
ax.text(5.3, 8.15, "earlier:", fontsize=6.0, fontweight="bold",
        ha="left", va="center", color=INK)
for j, ln in enumerate([
        "docs 1-6 - the founding self-kills",
        "docs 24-89 - the instrument provenance"]):
    ax.text(5.55, 7.78 - j * 0.40, ln, fontsize=5.5, ha="left",
            va="center", color=INK)
ax.text(5.3, 6.83, "later:", fontsize=6.0, fontweight="bold",
        ha="left", va="center", color=INK)
for j, ln in enumerate([
        "doc 92 - the refutation, measured",
        "doc 93 - the protocol, corrected",
        "doc 94 - the audit, passed"]):
    ax.text(5.55, 6.46 - j * 0.40, ln, fontsize=5.5, ha="left",
            va="center", color=INK)

# the mirror box
ax.add_patch(FancyBboxPatch((0.25, 2.6), 9.5, 2.6,
                            boxstyle="round,pad=0.06", fc="#f5eef8",
                            ec=PURPLE, lw=1.5))
ax.text(5.0, 4.88, "THE MIRROR  (filed against itself, Part 4)",
        fontsize=6.6, fontweight="bold", ha="center", va="center",
        color=PURPLE)
for j, ln in enumerate([
        '"a verdict that flips with the rubric was never a fact about the',
        'field;  a charge that flips with the reading sample was never a',
        'fact about the run."   The one-sentence charge has the epistemic',
        "shape of the cap it indicts - a verdict stated above its bounds."]):
    ax.text(0.55, 4.38 - j * 0.40, ln, fontsize=5.5, ha="left",
            va="center", color=INK)

# the residue + new rule
ax.add_patch(FancyBboxPatch((0.25, 0.35), 9.5, 1.9,
                            boxstyle="round,pad=0.06", fc="#eafaf0",
                            ec=GREEN, lw=1.5))
ax.text(0.55, 1.95, "the residue that stands - and the rule it buys:",
        fontsize=6.0, fontweight="bold", ha="left", va="center",
        color=GREEN)
for j, ln in enumerate([
        "doc 90 read alone under-discloses relative to the corpus's own",
        "standards (an unaided reader lands on the true defect - the cap).",
        "NEW RULE: doc 90 travels with doc 92 - the co-citation rule."]):
    ax.text(0.55, 1.50 - j * 0.38, ln, fontsize=5.5, ha="left",
            va="center", color=INK)
ax.set_title("(c) the miss inventory, the mirror (Part 4)", fontsize=8.6,
             fontweight="bold")

# ------------------------------------------- (d) the ruling
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

QA = [
    ("was our run really biased?",
     "AT ONE LAYER, YES - the sole-survivor cap: criterion-relative, caught, measured and",
     "amended by the run itself (doc 92), before any outside word. AT EVERY OTHER LAYER, NO.",
     ORANGE, "#fdf3e7"),
    ("was it circular?",
     "NO - the provenance timeline is clean (instruments first, verdict 84 documents later);",
     "the two unstated auxiliaries located at axiom grade and priced; the theorems stand.",
     RED, "#fdedec"),
    ("did the other AI just miss our earlier work?",
     "YES, ON BOTH SIDES - the earlier disclosure trail (the self-kills, the marker, the",
     "bounds), and the later audit arc (docs 92-94, filed 10:13Z-12:04Z, before the charge).",
     BLUE, "#eaf2f8"),
]
y_top, bh = 9.15, 2.12
for i, (q, l1, l2, ec, fc) in enumerate(QA):
    y = y_top - (i + 1) * bh - i * 0.22
    ax.add_patch(FancyBboxPatch((0.25, y), 9.5, bh,
                                boxstyle="round,pad=0.06", fc=fc, ec=ec,
                                lw=1.5, zorder=2))
    ax.text(0.5, y + bh - 0.40, q, fontsize=6.6, fontweight="bold",
            ha="left", va="center", color=ec, zorder=3)
    ax.text(0.5, y + bh - 0.90, l1, fontsize=5.4, ha="left", va="center",
            color=INK, zorder=3)
    ax.text(0.5, y + bh - 1.35, l2, fontsize=5.4, ha="left", va="center",
            color=INK, zorder=3)

ax.add_patch(FancyBboxPatch((0.25, 0.30), 9.5, 1.55,
                            boxstyle="round,pad=0.06", fc="#f4f6f6",
                            ec=GRAY, lw=1.4))
for j, ln in enumerate([
        "the third-lens invitation stands: any reviewer, any family, under the published protocol -",
        "the corpus files the result under the markers whatever it says.  Both markers up;",
        "determinism chain 798;  P4 next (pre-registered, doc 91);  the crossing still owed."]):
    ax.text(5.0, 1.50 - j * 0.40, ln, fontsize=5.4, ha="center",
            va="center", color=INK)
ax.set_title("(d) the ruling (Parts 5-6)", fontsize=8.6, fontweight="bold")

fig.suptitle("The outside charge, adjudicated (corpus doc 95): doc 90's bias "
             "verdict against the filed record - confirmed at the cap, "
             "refuted at the run, missed on both sides",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/charge_figure1.png", dpi=170)
print("wrote", f"{D}/charge_figure1.png")
