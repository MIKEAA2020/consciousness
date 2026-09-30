#!/usr/bin/env python3
"""Figure for corpus doc 85 (the mode axis): the toxicity plane (realized
dose vs burn count, six worlds), the fill paths (crash vs absorption),
the channel's dispute traffic (catchable vs uncatchable, log scale), and
the six-world verdict map with the nesting. English labels,
constrained_layout, house palette."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"
S = "/home/z/my-project/scripts"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"

d = json.load(open(f"{S}/s17_modeaxis_results.json"))
t = json.load(open(f"{S}/s16_unif25_results.json"))
c78 = json.load(open(f"{S}/s14_anothermix_results.json"))
cls = d["classification"]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# --------------------------------------------------- (a) the toxicity plane
ax = axes[0, 0]
worlds = [
    ("control\n(uniform 0.13)", 0.1291, 0, GREEN, "doc 78"),
    ("twin\n(uniform 0.25)", 0.2484, 1, GREEN, "doc 82"),
    ("joint\n(blind, runner-up)", 0.0695, 1, ORANGE, "doc 85"),
    ("mirror\n(mixed 0.30)", 0.1238, 6, BLUE, "doc 78"),
    ("capacity\n(mixed 0.30)", 0.2505, 12, BLUE, "doc 73"),
    ("gate\n(blind, +1)", 0.0357, 22, RED, "doc 85"),
]
for lab, dose, burn, col, doc in worlds:
    ax.scatter(dose, burn, s=170, color=col, zorder=3, edgecolor="k",
               linewidth=0.7)
    dx, dy = 0.010, 0.45
    ha = "left"
    if "gate" in lab:
        dx, dy, ha = 0.012, -0.4, "left"
    if "joint" in lab:
        dy = -1.3
    ax.annotate(lab, (dose, burn), xytext=(dose + dx, burn + dy),
                fontsize=6.6, ha=ha, color=col, fontweight="bold")
ax.annotate("the lowest dose ever burned,\nthe widest burn ever measured",
            (0.0357, 22), xytext=(0.085, 19.2), fontsize=6.6,
            color=RED, fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
ax.set_xlabel("realized dose (rounds 7-14 mean)", fontsize=8)
ax.set_ylabel("seeds BURNED (of 24)", fontsize=8)
ax.set_xlim(-0.01, 0.34)
ax.set_ylim(-2.0, 25)
ax.set_title("(a) the toxicity plane: severity = selection x label content",
             fontsize=8.6, fontweight="bold")
ax.grid(alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=7)
ax.text(0.252, 3.6, "catchable:\nrandom selection", fontsize=6.4,
        color=GREEN, ha="center", fontstyle="italic")
ax.text(0.052, 8.0, "uncatchable:\nblind-targeted", fontsize=6.4,
        color=ORANGE, ha="center", fontstyle="italic")

# ------------------------------------------------------- (b) the fill paths
ax = axes[0, 1]
rr = range(7, 15)


def compF(arm, res):
    return [x for x in res["arms"][arm]["aggregate"]["comp_F"][6:14]]


gate_f = compF("S23_GATE30", d)
joint_f = compF("S23_JOINT30", d)
twin_f = compF("S22_UNIF25", t)
ax.plot(list(rr), gate_f, "-o", color=RED, ms=4, lw=1.8,
        label="gate (blind, +1 labels) - CRASH")
ax.plot(list(rr), joint_f, "-s", color=ORANGE, ms=4, lw=1.8,
        label="joint (blind, runner-up) - ABSORBED")
ax.plot(list(rr), twin_f, "-^", color=GREEN, ms=4, lw=1.4,
        label="twin (uniform 0.25, stored)")
ax.axhline(-0.02, color=RED, lw=1.0, ls="--", alpha=0.7)
ax.text(13.6, -0.030, "BURN floor -0.02", fontsize=6.4, color=RED,
        ha="right")
ax.axhline(0.02, color=GREEN, lw=0.9, ls=":", alpha=0.7)
ax.text(13.6, 0.028, "ON-COURSE floor +0.02", fontsize=6.4,
        color=GREEN, ha="right")
ax.set_xlabel("round", fontsize=8)
ax.set_ylabel("the fill (comp_F)", fontsize=8)
ax.set_title("(b) the fill: the world digests the runner-up,\n"
             "crashes on the alien label and never recovers",
             fontsize=8.6, fontweight="bold")
ax.legend(fontsize=6.6, loc="lower left")
ax.grid(alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=7)
ax.set_xticks(list(rr))

# ------------------------------------------- (c) the channel's dispute input
ax = axes[1, 0]
twin_dis = t["arms"]["S22_UNIF25"]["aggregate"]["n_dis_raw"][6:14]
gate_dis = d["arms"]["S23_GATE30"]["aggregate"]["n_dis_raw"][6:14]
joint_dis = d["arms"]["S23_JOINT30"]["aggregate"]["n_dis_raw"][6:14]
ax.semilogy(list(rr), [max(x, 0.02) for x in twin_dis], "-^", color=GREEN,
            ms=4, lw=1.4, label="twin: random flips (stored)")
ax.semilogy(list(rr), [max(x, 0.02) for x in gate_dis], "-o", color=RED,
            ms=4, lw=1.8, label="gate: blind flips")
ax.semilogy(list(rr), [max(x, 0.02) for x in joint_dis], "-s",
            color=ORANGE, ms=4, lw=1.8, label="joint: blind flips")
ax.set_xlabel("round", fontsize=8)
ax.set_ylabel("would-fire disputes per seed per round", fontsize=8)
ax.set_title("(c) the channel sees the random poison and repairs it;\n"
             "the blind-targeted poison it never sees (0.0-1.2/round)",
             fontsize=8.6, fontweight="bold")
ax.legend(fontsize=6.6)
ax.grid(alpha=0.25, linewidth=0.5, which="both")
ax.tick_params(labelsize=7)
ax.set_xticks(list(rr))
ax.text(0.055, 0.06, "d_recv = the flip mass in EVERY world:\n"
        "the panel disagrees with exactly the flips -\n"
        "only the random selection is CONFIDENT disagreement",
        transform=ax.transAxes, fontsize=6.3, color="#212121",
        va="bottom")

# ------------------------------------------------- (d) the six-world verdict
ax = axes[1, 1]
seeds = list(range(600, 624))
cols6 = ["capacity", "mirror", "control", "twin", "gate", "joint"]
vmap = cls["MX_six_world_map"]
CMAP = {"BURNED": RED, "CONSUMED": "#d5d8dc", "ON-COURSE": GREEN,
        "OVERFILLED": PURPLE, "MUM": "white", None: "white"}
for j, w in enumerate(cols6):
    for i, s in enumerate(seeds):
        v = vmap[str(s)][w]
        ax.add_patch(plt.Rectangle((j, i), 0.94, 0.94,
                                   facecolor=CMAP.get(v, "white"),
                                   edgecolor="#bdc3c7", linewidth=0.35))
    ax.text(j + 0.47, -0.85, w, fontsize=7.4, ha="center",
            fontweight="bold",
            color={"gate": RED, "joint": ORANGE}.get(w, "#212121"))
ax.set_xlim(-0.1, 6.0)
ax.set_ylim(-1.6, 24.2)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_yticks([i + 0.5 for i in range(24)])
ax.set_yticklabels(seeds, fontsize=5.6)
ax.set_ylabel("seed (600-623)", fontsize=8)
ax.set_title("(d) the six-world verdict map: the nesting - {606} inside\n"
             "the mirror's six inside the capacity's twelve inside the "
             "gate's twenty-two",
             fontsize=8.6, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=RED, label="BURNED"),
                   Patch(facecolor="#d5d8dc", label="CONSUMED"),
                   Patch(facecolor=GREEN, label="ON-COURSE")],
          fontsize=6.6, loc="lower right", ncol=3,
          bbox_to_anchor=(1.0, -0.145))
ax.text(3.0, 24.05, "the two gate survivors (602, 613) are the "
        "census's two lowest-b* seeds", fontsize=6.3, ha="center",
        color="#212121", fontstyle="italic")

fig.suptitle("The mode axis (corpus doc 85): the gate burns twenty-two "
             "at the lowest dose ever measured; the joint burns one - "
             "severity = selection x label content",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/modeaxis_figure1.png", dpi=170)
print("wrote", f"{D}/modeaxis_figure1.png")
