#!/usr/bin/env python3
"""Figure for corpus doc 82: the bulk-matched twin (the completed
factorial, the ordered fragility axis, the fill/churn race, the
paired verdicts). English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

tw = json.load(open(f"{S}/s16_unif25_results.json"))
d75 = json.load(open(f"{S}/s12_coldburn_results.json"))
cls = tw["classification"]
seeds = {int(k): v for k, v in cls["BT_seeds"].items()}
pair = {int(k): v for k, v in
        cls["BT4_paired_verdicts"]["per_seed"].items()}

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"


def flate_of(rr):
    n = len(rr["traj"])
    late = [t["comp_F"] for t in rr["traj"]
            if t["comp_F"] is not None and t["round"] >= n - 3]
    return float(np.mean(late)) if late else None


# the capacity's stored anchor/armed (doc 75, for the fill/churn race)
cap_anch = {rr["seed"]: flate_of(rr)
            for rr in d75["arms"]["S18_CAP30_NOCH"]["runs"]}
cap_arm = {rr["seed"]: flate_of(rr)
           for rr in d75["arms"]["S18_CAP30"]["runs"]}
cap_a = float(np.mean([v for v in cap_anch.values() if v is not None]))
cap_f = float(np.mean([v for v in cap_arm.values() if v is not None]))

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the completed factorial: composition x realized dose
ax = axes[0, 0]
grid = np.array([[1, 6], [12, 0]])      # rows: dose 0.25 / 0.125
labels = np.array([["twin (THIS battery)\nuniform 0.2484\n1 BURNED",
                    "mirror\nmixed 0.1238\n6 BURNED"],
                   ["capacity\nmixed 0.2505\n12 BURNED",
                    "control\nuniform 0.1291\n0 BURNED"]])
im = ax.imshow(grid, cmap="OrRd", vmin=0, vmax=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i, j], ha="center", va="center",
                fontsize=9.6,
                color="white" if grid[i, j] >= 6 else "#212121")
ax.set_xticks([0, 1])
ax.set_xticklabels(["uniform\n(no gated mass)", "mixed\n(bulk + gated)"],
                   fontsize=9)
ax.set_yticks([0, 1])
ax.set_yticklabels(["realized ~0.25", "realized ~0.125"], fontsize=9)
ax.set_title("(a) THE COMPLETED FACTORIAL: at the capacity's own dose the\n"
             "uniform world burns ONE - the twelve are the mix's act",
             fontsize=9.5)
ax.annotate("the missing cell,\nnow measured", (0.0, 0.28), fontsize=8.6,
            color=BLUE)
fig.colorbar(im, ax=ax, shrink=0.82, label="BURNED of 24")

# (b) the ordered fragility axis: the burn sets nest by b*
ax = axes[0, 1]
order = sorted(seeds, key=lambda s: -seeds[s]["b_star"])
b_all_three = [s for s in order if pair[s]["cap"] == "BURNED"
               and pair[s]["mirror"] == "BURNED"
               and pair[s]["twin"] == "BURNED"]
b_cap_mir = [s for s in order if pair[s]["cap"] == "BURNED"
             and pair[s]["mirror"] == "BURNED"
             and pair[s]["twin"] != "BURNED"]
b_cap_only = [s for s in order if pair[s]["cap"] == "BURNED"
              and pair[s]["mirror"] != "BURNED"]
safe = [s for s in order if pair[s]["cap"] != "BURNED"]
for grp, col, lab, mk in [
        (b_all_three, RED, "burns in all three attack worlds (1: 606)",
         "o"),
        (b_cap_mir, ORANGE, "capacity + mirror (5: the gate's six)", "^"),
        (b_cap_only, PURPLE, "capacity only (6: the interaction's)", "v"),
        (safe, BLUE, "never burns (12)", "s")]:
    ax.scatter([order.index(s) + 1 for s in grp],
               [seeds[s]["b_star"] for s in grp], c=col, marker=mk,
               s=58, label=lab, edgecolors="white", linewidths=0.6,
               zorder=3)
ax.axvline(6.5, color=GRAY, ls=":", lw=1.3)
ax.annotate("the mirror's edge (0.0768|0.0819)", (6.9, 0.0495),
            fontsize=8.2, color=GRAY)
ax.annotate("the nesting: {606} inside the gate's six\n"
            "inside the mix's twelve - the burn is an\n"
            "ORDERED set; the composition sets its depth",
            (7.6, 0.088), fontsize=8.4, color="#212121")
ax.set_xlabel("the 24 seeds ranked by the registration b* (descending)")
ax.set_ylabel("b* (identical across worlds)")
ax.set_title("(b) THE ORDERED FRAGILITY AXIS: the dose takes the top one,\n"
             "the gate the top six, the full mix the top six plus the middle",
             fontsize=9.5)
ax.legend(fontsize=7.7, loc="upper right")

# (c) the fill/churn race across the four worlds
ax = axes[1, 0]
worlds = ["mirror\nmixed 0.124", "control\nuniform 0.129",
          "twin\nuniform 0.248", "capacity\nmixed 0.251"]
anch = [cls["BT5_anchor_decomposition"]["mirror_anchor_stored"],
        cls["BT5_anchor_decomposition"]["control_anchor_stored"],
        cls["BT5_anchor_decomposition"]["twin_anchor_mean"], cap_a]
diff = [0.017, -0.107,
        cls["BT5_anchor_decomposition"]["twin_differential_mean"],
        cap_f - cap_a]
arm = [-0.010, 0.033,
       float(np.mean([seeds[s]["F_late"] for s in seeds])), cap_f]
xpos = np.arange(4)
w = 0.27
ax.bar(xpos - w, anch, w, color=BLUE, label="anchor F (the fill)")
ax.bar(xpos, diff, w, color=RED, label="differential (the churn)")
ax.bar(xpos + w, arm, w, color=GREEN, label="armed F (the net)")
ax.axhline(-0.02, color=RED, ls="--", lw=1.4)
ax.annotate("the burn floor (-0.02)", (2.62, -0.036), fontsize=8.4,
            color=RED)
for x, a, d_, n in zip(xpos, anch, diff, arm):
    ax.text(x - w, a + (0.006 if a >= 0 else -0.016), f"{a:+.3f}",
            ha="center", fontsize=7.8)
    ax.text(x, d_ + (0.006 if d_ >= 0 else -0.016), f"{d_:+.3f}",
            ha="center", fontsize=7.8)
    ax.text(x + w, n + (0.006 if n >= 0 else -0.016), f"{n:+.3f}",
            ha="center", fontsize=7.8)
ax.axhline(0, color="#444444", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(worlds, fontsize=8.6)
ax.set_ylabel("F_late mean")
ax.set_title("(c) THE FILL/CHURN RACE: the fill grows with dose but the\n"
             "churn grows faster - the margin thins, and at 606 it is lost",
             fontsize=9.5)
ax.legend(fontsize=8, loc="lower left")

# (d) the paired verdicts: capacity F_late against twin F_late
ax = axes[1, 1]
for s in sorted(pair):
    if pair[s]["cap"] == "BURNED" and pair[s]["twin"] == "BURNED":
        col, mk, z = RED, "o", 4
    elif pair[s]["cap"] == "BURNED":
        col, mk, z = ORANGE, "^", 3
    else:
        col, mk, z = BLUE, "s", 3
    ax.scatter(pair[s]["cap_F"], pair[s]["twin_F"], c=col, marker=mk,
               s=58, edgecolors="white", linewidths=0.6, zorder=z)
ax.plot([-0.07, 0.05], [-0.07, 0.05], color=GRAY, lw=0.9, ls=":")
ax.axhline(-0.02, color=RED, ls="--", lw=1.3)
ax.axvline(-0.02, color=RED, ls="--", lw=1.3)
ax.annotate("the floor (both axes)", (-0.066, -0.0145), fontsize=8.2,
            color=RED)
ax.annotate("606: the only seed below\nboth floors - the top b*,\n"
            "differential-carried (the defense's\nchurn -0.197)",
            (-0.0568, -0.0435), fontsize=8.2, color=RED)
ax.annotate("the eleven flips: the capacity's\nDEEPER burns (mean -0.033),\n"
            "all one-directional - the\ntwin saves the deep cluster",
            (-0.0435, 0.0215), fontsize=8.2, color=ORANGE)
ax.annotate("never burned", (0.016, 0.0365), fontsize=8.2, color=BLUE)
ax.set_xlabel("F_late at the capacity (doc 75, stored)")
ax.set_ylabel("F_late at the twin (uniform 0.25)")
ax.set_title("(d) THE PAIRED VERDICTS: 11 flips, all burn->consumed, zero\n"
             "reverse - the twin saves the deep, the mirror saved the shallow",
             fontsize=9.5)

fig.suptitle("Doc 82 - the bulk-matched twin: the uniform world at the "
             "capacity's realized dose (0.2484 vs 0.2505), 24 seeds, "
             "60/60 audits", fontsize=10.5)
out = f"{D}/unif25_figure1.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
