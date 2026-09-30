#!/usr/bin/env python3
"""Figures for corpus docs 67-68: the interior dose (the flip placed) and
the b* gap's provenance (the committee-panel relationship). English
labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

it = json.load(open(f"{S}/s8_interior_results.json"))
bp = json.load(open(f"{S}/s8_bstarprov_results.json"))
d51 = json.load(open(f"{S}/s4_candidate4_results.json"))

OVER, FILL, BURN = 0.08, 0.02, -0.02
DEEP = [603, 604, 606]
MARG = [600, 607]

RED, BLUE, GREEN, GRAY = "#c0392b", "#2471a3", "#1e8449", "#7f8c8d"


# =================== doc 67: the interior dose ================================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the p-curve at 0.15 - the fresh six-point grid
ax = axes[0, 0]
f15 = {float(k): v for k, v in
       it["classification"]["IN3_curve15"]["fractions_of_max"].items()}
ps = sorted(f15)
ax.plot(ps, [f15[p] for p in ps], "D-", color="#7d3c98", lw=2,
        label="the dial at 0.15 (the interior dose)")
ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1.2,
        label="the proportional line")
ax.annotate("0.644 of max\n(2.58x proportional:\nthe superlinearity's peak)",
            (0.25, 0.42), fontsize=8.5, arrowprops=dict(arrowstyle="->",
                                                        lw=0.8))
ax.annotate("the shelf:\n0.776 to 0.784", (0.375, 0.70), fontsize=8.5,
            ha="center", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("repair probability p (the cleaning bought)")
ax.set_ylabel("fraction of the max available effect")
ax.set_title("(a) The interior dose's curve: superlinear at the bottom\n"
             "(no S-bottom), a shelf at 0.35-0.40, no knee", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

# (b) the three-dose family - the flip's placement
ax = axes[0, 1]
f05 = {0.25: 0.571, 0.35: 0.601, 0.40: 0.669, 0.50: 0.757, 1.0: 1.010}
f30 = {0.25: 0.222, 0.35: 0.415, 0.40: 0.489, 0.50: 0.677, 0.75: 0.886,
       1.0: 0.986}
ax.plot(sorted(f05), [f05[p] for p in sorted(f05)], "s-", color=BLUE,
        lw=1.8, label="at 0.05 (stored): superlinear")
ax.plot(sorted(f15), [f15[p] for p in sorted(f15)], "D-", color="#7d3c98",
        lw=1.8, label="at 0.15 (fresh): still superlinear")
ax.plot(sorted(f30), [f30[p] for p in sorted(f30)], "o-", color=RED,
        lw=1.8, label="at 0.30 (stored): the S-bottom")
ax.plot([0, 1], [0, 1], ":", color="#9b9b9b", lw=1.1,
        label="the proportional line")
ax.annotate("the flip lives\nin (0.15, 0.30)", (0.62, 0.30), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_xlabel("repair probability p")
ax.set_ylabel("fraction of the max available effect")
ax.set_title("(b) The three-dose family: the bottom's superlinearity PEAKS\n"
             "at the interior (origin slopes 2.28 / 2.58 / 0.89) - the\n"
             "crossing is not a smooth monotone migration", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

# (c) the local slopes at 0.15 against the engagement threshold
ax = axes[1, 0]
sl15 = it["classification"]["IN3_curve15"]["local_slopes"]
lab = list(sl15)
val = [sl15[k] for k in lab]
ax.bar(np.arange(len(lab)), val, width=0.5, color="#7d3c98",
       label="local slopes at 0.15")
ax.axhline(1.5, color=RED, ls=":", lw=1.4,
           label="the engagement slope 1.5 (doc 63's convention)")
ax.set_xticks(np.arange(len(lab)))
ax.set_xticklabels(lab, fontsize=8.5)
ax.set_ylabel("forward slope (fraction of max per unit p)")
ax.set_title("(c) No interval reaches the engagement slope at 0.15:\n"
             "the knee is absent - the window is the 0.05 shape, not\n"
             "the 0.30 shape", fontsize=10)
ax.legend(fontsize=8)

# (d) the anchor's dose ladder and the grounds at 0.15
ax = axes[1, 1]
ladder = [(0.05, 0.8463), (0.10, 0.6207), (0.15, 0.3778), (0.30, 0.1859)]
ax.plot([d for d, _ in ladder], [a for _, a in ladder], "o-",
        color="#b03a2e", lw=2, label="the fresh anchor (undefended acc)")
lin = 0.8463 + (0.1859 - 0.8463) * (0.15 - 0.05) / 0.25
ax.plot([0.05, 0.30], [0.8463, 0.1859], "--", color=GRAY, lw=1.2,
        label="the linear dose interpolation")
ax.annotate("0.378 at 0.15:\nbelow the line 0.582\n(the collapse\n"
            "front-loaded)", (0.15, 0.55), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("attack dose (lambda)")
ax.set_ylabel("the anchor's final accuracy (no defense)")
ax.set_title("(d) The anchor's dose ladder: the undefended world's collapse\n"
             "is front-loaded - the interior dose already sits at 0.38",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right")

fig.savefig(f"{D}/interior_figure1.png", dpi=160)
plt.close(fig)
print("wrote interior_figure1.png")


# =================== doc 68: the b* gap's provenance =========================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

BPs = {int(k): v for k, v in
       bp["classification"]["BP_seeds"].items()}
seeds_sorted = sorted(BPs, key=lambda s: BPs[s]["b_star"])
deep_set = set(DEEP)
marg_set = set(MARG)


def colr(s):
    return RED if s in deep_set else ("#e67e22" if s in marg_set else BLUE)


# (a) the decomposition per seed: b* = b_pool + L
ax = axes[0, 0]
xs = np.arange(len(seeds_sorted))
pools = [BPs[s]["b_pool5"] for s in seeds_sorted]
Ls = [BPs[s]["L5"] for s in seeds_sorted]
ax.bar(xs, pools, width=0.62, color=BLUE, alpha=0.85,
       label="b_pool (the draw's blinding power, full pool)")
ax.bar(xs, Ls, bottom=pools, width=0.62, color="#7d3c98", alpha=0.85,
       label="L = b* - b_pool (the gate's concentration, negative)")
for i, s in enumerate(seeds_sorted):
    ax.plot([i - 0.36, i + 0.36], [BPs[s]["b_star"]] * 2, color="k",
            lw=1.4)
    if s in deep_set:
        ax.plot(i, BPs[s]["b_star"], "o", color=RED, zorder=5, ms=5)
ax.set_xticks(xs)
ax.set_xticklabels(seeds_sorted, fontsize=8.5)
ax.set_ylabel("blind fraction (round 5, the registration)")
ax.set_title("(a) b* (the black line) = the draw's pool (blue, FLAT ~0.33)\n"
             "+ the gate's lift (purple, negative): the deep three's\n"
             "elevation lives in the lift, not the pool", fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (b) the blind-admission rate - the perfect separation
ax = axes[0, 1]
adm_b = [BPs[s]["adm_blind5"] for s in seeds_sorted]
adm_c = [BPs[s]["adm_clean5"] for s in seeds_sorted]
w = 0.38
ax.bar(xs - w / 2, adm_b, width=w, color=[colr(s) for s in seeds_sorted],
       label="pool-blind items admitted")
ax.bar(xs + w / 2, adm_c, width=w, color=GRAY, alpha=0.6,
       label="pool-clean items admitted")
ax.set_xticks(xs)
ax.set_xticklabels(seeds_sorted, fontsize=8.5)
ax.set_ylabel("admission rate at round 5")
ax.set_title("(b) THE SEPARATION: the gate's blind-admission rate -\n"
             "the deep three 0.180-0.219, the body 0.094-0.147, no overlap;\n"
             "the clean-admission flat (0.90 vs 0.93)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.text(0.02, 0.55, "red = the deep three\norange = the marginals",
        transform=ax.transAxes, fontsize=8, color=RED)

# (c) the elevation: the pool flat, the committed read rising
ax = axes[1, 0]
body = [s for s in BPs if s not in deep_set]
rs = [1, 2, 3, 4, 5]
pool_deep = np.mean([[BPs[s]["b_pools_1to5"][str(r)] for r in rs]
                     for s in DEEP], axis=0)
pool_body = np.mean([[BPs[s]["b_pools_1to5"][str(r)] for r in rs]
                     for s in body], axis=0)
comm_deep = np.mean([[BPs[s]["b_comms_1to5"][str(r)] for r in rs]
                     for s in DEEP], axis=0)
comm_body = np.mean([[BPs[s]["b_comms_1to5"][str(r)] for r in rs]
                     for s in body], axis=0)
ax.plot(rs, pool_deep, "o--", color=RED, lw=1.6, ms=5,
        label="b_pool, the deep three")
ax.plot(rs, pool_body, "s--", color=BLUE, lw=1.6, ms=5,
        label="b_pool, the body")
ax.plot(rs, comm_deep, "o-", color=RED, lw=2.2,
        label="b_committed, the deep three")
ax.plot(rs, comm_body, "s-", color=BLUE, lw=2.2,
        label="b_committed, the body")
ax.axvline(5, color=GRAY, ls=":", lw=1.2)
ax.text(5.06, 0.30, "the registration\nround 5", fontsize=8, color=GRAY)
ax.set_xlabel("round (the clean prefix)")
ax.set_ylabel("panel-blind fraction")
ax.set_title("(c) THE ELEVATION IS THE GATE'S: the full pool's blindness is\n"
             "flat across the prefix (both groups); the committed pools'\n"
             "blindness climbs as the committee diverges from its panel\n"
             "(the deep three climb faster)", fontsize=10)
ax.legend(fontsize=8, loc="center right")

# (d) b* against the blind-admission rate - the upstream named
ax = axes[1, 1]
for s in seeds_sorted:
    ax.plot(BPs[s]["adm_blind5"], BPs[s]["b_star"], "o", ms=8,
            color=colr(s))
    ax.annotate(str(s), (BPs[s]["adm_blind5"], BPs[s]["b_star"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
bb = [BPs[s]["b_star"] for s in seeds_sorted]
aa = [BPs[s]["adm_blind5"] for s in seeds_sorted]
cc = np.corrcoef(bb, aa)[0, 1]
ax.set_xlabel("the gate's blind-admission rate at round 5")
ax.set_ylabel("b* (the registered blind mass)")
ax.set_title(f"(d) b* against its upstream: corr = {cc:.3f} - the\n"
             "committee-panel relationship (the committee clearing the\n"
             "gate on items its frozen panel reads blind)", fontsize=10)

fig.savefig(f"{D}/bstarprov_figure1.png", dpi=160)
plt.close(fig)
print("wrote bstarprov_figure1.png")
