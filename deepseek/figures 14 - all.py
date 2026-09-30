#!/usr/bin/env python3
"""Figures for corpus docs 63-65: the knee's location, the windowed-
baseline amendment, and the effect bars at double horizon. English
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

kn = json.load(open(f"{S}/s7_knee_results.json"))
wb = json.load(open(f"{S}/s7_winbase_results.json"))
eb = json.load(open(f"{S}/s7_effectbar30_results.json"))
bnd = json.load(open(f"{S}/s5_boundaries_results.json"))
d51 = json.load(open(f"{S}/s4_candidate4_results.json"))

OVER, FILL, BURN = 0.08, 0.02, -0.02
DEEP = [603, 604, 606]
MARG = [600, 607]


# =================== doc 63: the knee's location ============================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the p-curve at 0.30 with the interior resolved
ax = axes[0, 0]
f30 = {float(k): v for k, v in
       kn["classification"]["KN2_knee30"]["fractions_of_max"].items()}
ps = sorted(f30)
ax.plot(ps, [f30[p] for p in ps], "o-", color="#c0392b", lw=2,
        label="the dial at 0.30 (interior resolved)")
ax.plot([0, 1], [0, 1], ":", color="#7f8c8d", lw=1.2,
        label="the proportional line")
lo, hi = kn["classification"]["KN2_knee30"]["knee_interval"]
ax.axvspan(lo, hi, color="#c0392b", alpha=0.12,
           label=f"the knee's bracket ({lo}, {hi})")
ax.annotate("the quarter-buyer:\n0.222 of max", (0.25, 0.30),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("0.415", (0.35, f30[0.35] + 0.03), fontsize=8.5,
            ha="center", color="#c0392b")
ax.annotate("0.489", (0.40, f30[0.40] - 0.06), fontsize=8.5,
            ha="center", color="#c0392b")
ax.set_xlabel("repair probability p (the cleaning bought)")
ax.set_ylabel("fraction of the max available effect")
ax.set_title("(a) The S-curve's interior at 0.30: the engagement slope\n"
             "is reached on the window's bottom interval", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (b) the same window at 0.05 - the flanks
ax = axes[0, 1]
f05 = {float(k): v for k, v in
       kn["classification"]["KN3_flanks05"]["fractions_of_max"].items()}
ps = sorted(f05)
ax.plot(ps, [f05[p] for p in ps], "s-", color="#2471a3", lw=2,
        label="the dial at 0.05 (flanks measured)")
chord_x = [0.25, 0.5]
chord_y = [0.571, 0.757]
ax.plot(chord_x, chord_y, "--", color="#7f8c8d", lw=1.4,
        label="the concave arc (doc 56's chord)")
ax.plot([0, 1], [0, 1], ":", color="#9b9b9b", lw=1.1,
        label="the proportional line")
ax.annotate("0.601\n(above proportional,\nbelow the chord)",
            (0.35, f05[0.35] + 0.02), fontsize=8, ha="center")
ax.set_xlabel("repair probability p")
ax.set_ylabel("fraction of the max available effect")
ax.set_title("(b) The same window at 0.05: superlinear through and\n"
             "through, the bottom softening (no S-bottom)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (c) the local slopes at both doses
ax = axes[1, 0]
sl30 = kn["classification"]["KN2_knee30"]["local_slopes"]
lab30 = [k for k in sl30 if k.split("->")[0] >= "0.2"]
val30 = [sl30[k] for k in lab30]
ax.bar(np.arange(len(lab30)) - 0.17, val30, width=0.34,
       color="#c0392b", label="at 0.30")
ax.axhline(1.5, color="#c0392b", ls=":", lw=1.2,
           label="the engagement slope 1.5")
ax.set_xticks(np.arange(len(lab30)))
ax.set_xticklabels(lab30, fontsize=8)
ax.set_ylabel("forward slope (fraction of max per unit p)")
ax.set_title("(c) The local slopes at 0.30: 1.92 on the bottom interval,\n"
             "the shoulder at 1.48 unresolved inside the seed spread",
             fontsize=10)
ax.legend(fontsize=8)

# (d) the grounds' migration through the dial at 0.30
ax = axes[1, 1]
g59 = [(0.25, 0.1257, "OVERFILLED"), (0.35, 0.0777, "ON-COURSE"),
       (0.40, 0.0461, "ON-COURSE"), (0.50, 0.0425, "ON-COURSE"),
       (0.75, 0.0200, "CONSUMED"), (1.00, -0.0088, "CONSUMED")]
ax.plot([p for p, _, _ in g59], [f for _, f, _ in g59], "o-",
        color="#1e8449", lw=2)
ax.axhline(OVER, color="#b03a2e", ls="--", lw=1.1,
           label="the OVERFILLED ceiling")
ax.axhline(BURN, color="#b03a2e", ls=":", lw=1.1,
           label="the BURNED floor")
ax.set_xlabel("repair probability p")
ax.set_ylabel("ground F_late (the fill)")
ax.set_title("(d) The drain's migration is monotone and knee-free:\n"
             "the ground ages smoothly where the effect breaks",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right")

fig.savefig(f"{D}/knee_figure1.png", dpi=160)
plt.close(fig)
print("knee_figure1.png done")


# ============ doc 64: the windowed-baseline amendment =======================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

ps = wb["classification"]["WB_per_seed"]["armed"]
seeds = [r["seed"] for r in ps]

# (a) the three registrations' baselines per seed
ax = axes[0, 0]
ax.plot(seeds, [r["b_single"] for r in ps], "o-", color="#c0392b",
        lw=1.8, label="the single draw (round 5)")
ax.plot(seeds, [r["b_w35"] for r in ps], "s-", color="#2471a3",
        lw=1.6, label="w35 (mean of rounds 3-5)")
ax.plot(seeds, [r["b_w14"] for r in ps], "^-", color="#1e8449",
        lw=1.4, label="w14 (mean of rounds 1-4)")
ax.set_xlabel("seed")
ax.set_ylabel("registered baseline b*")
ax.set_title("(a) The three registrations: the conservatism price is the\n"
             "gap between the red and the blue - the high-water's height",
             fontsize=10)
ax.legend(fontsize=8)

# (b) the verdicts at the founding horizon under the three registrations
ax = axes[0, 1]
w = 0.27
ax.bar(np.arange(12) - w, [r["F14_single"] for r in ps], width=w,
       color="#c0392b", label="single (BURNED at the mean)")
ax.bar(np.arange(12), [r["F14_w35"] for r in ps], width=w,
       color="#2471a3", label="w35 (CONSUMED at the mean)")
ax.bar(np.arange(12) + w, [r["F14_w14"] for r in ps], width=w,
       color="#1e8449", label="w14 (CONSUMED at the mean)")
ax.axhline(BURN, color="k", ls=":", lw=1.3, label="the burn floor")
for s in DEEP:
    i = seeds.index(s)
    ax.annotate(str(s), (i, min(ps[i]["F14_w14"], ps[i]["F14_single"])
                 - 0.012), fontsize=7.5, ha="center")
ax.set_xticks(np.arange(12))
ax.set_xticklabels(seeds, fontsize=8)
ax.set_ylabel("F_late at the 11-14 window")
ax.set_title("(b) The founding horizon: the deep three (603/604/606)\n"
             "survive w35; the level verdict does not", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")

# (c) the decomposition of the level
ax = axes[1, 0]
regs = ["single", "w35", "w14"]
means14 = [np.mean([r[f"F14_{n}"] for r in ps]) for n in regs]
means30 = [np.mean([r[f"F30_{n}"] for r in ps]) for n in regs]
x = np.arange(3)
ax.bar(x - 0.17, means14, width=0.34, color="#7d3c98",
       label="the 11-14 window")
ax.bar(x + 0.17, means30, width=0.34, color="#e67e22",
       label="the 27-30 window")
ax.axhline(BURN, color="k", ls=":", lw=1.3)
ax.axhline(0, color="#7f8c8d", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["single draw\n(BURNED / BURNED)",
                    "w35\n(CONSUMED / CONSUMED)",
                    "w14\n(CONSUMED / CONSUMED)"], fontsize=8.5)
ax.set_ylabel("mean F_late")
ax.set_title("(c) The level's decomposition: the era's fall (-0.011) and\n"
             "the registration's conservatism (-0.0135) beneath it",
             fontsize=10)
ax.annotate("the refusal's margin\nwas the conservatism",
            (0, -0.021), (0.55, -0.036), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))

# (d) the differential's invariance and the anchor's reductio
ax = axes[1, 1]
diff = wb["classification"]["WB5_differential"]["values"]
anch = wb["classification"]["WB6_anchor"]
ax.plot([14, 30], [diff["h14_single"], diff["h30_single"]], "o-",
        color="#c0392b", lw=2, label="the differential (all 3 regs)")
ax.plot([14, 30], [anch["h14_single"]["F_mean"],
                   anch["h30_single"]["F_mean"]], "s--", color="#2471a3",
        lw=1.6, label="the anchor, single draw")
ax.plot([14, 30], [anch["h14_w14"]["F_mean"],
                   anch["h30_w14"]["F_mean"]], "^--", color="#b03a2e",
        lw=1.6, label="the anchor, w14 (OVERFILLED at 30)")
ax.axhline(OVER, color="#b03a2e", ls="--", lw=1.0,
           label="the OVERFILLED ceiling")
ax.set_xlabel("horizon (rounds)")
ax.set_ylabel("F_late")
ax.set_title("(d) The differential is invariant (to 1e-14); the w14\n"
             "window certifies the disarmed anchor a runaway fill",
             fontsize=10)
ax.legend(fontsize=7.5, loc="center left")

fig.savefig(f"{D}/winbase_figure1.png", dpi=160)
plt.close(fig)
print("winbase_figure1.png done")


# ============ doc 65: the effect bars at double horizon ====================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

frame = eb["classification"]["EB_frame"]
sp30 = eb["classification"]["EB2_sp30_effect"]
sp05 = eb["classification"]["EB4_sp05_refusal"]

# (a) the five worlds at both horizons
ax = axes[0, 0]
order = [("S11_SP30", "SP30 armed"), ("S11_TWIN", "twin"),
         ("S11_SP05", "SP05 armed"), ("S11_SP05_NOCH", "SP05 anchor"),
         ("S11_SP30_NOCH", "SP30 anchor")]
x = np.arange(len(order))
ax.bar(x - 0.17, [frame[a]["acc14"] for a, _ in order], width=0.34,
       color="#5d6d7e", label="at 14 rounds")
ax.bar(x + 0.17, [frame[a]["acc30"] for a, _ in order], width=0.34,
       color="#1a5276", label="at 30 rounds")
ax.axhline(0.93, color="#7d3c98", ls=":", lw=1.2, label="the ruler's letter")
ax.set_xticks(x)
ax.set_xticklabels([n for _, n in order], fontsize=8.5)
ax.set_ylabel("accuracy")
ax.set_title("(a) The frame at double horizon: the armed world improves at\n"
             "0.30, holds at 0.05; the anchors fall at both doses",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (b) the 0.05 refusal crossing
ax = axes[0, 1]
ax.plot([14, 30], [sp05["max14"], sp05["max30"]], "o-", color="#2471a3",
        lw=2, label="the max effect (twin - anchor)")
ax.plot([14, 30], [sp05["effect14"], sp05["effect30"]], "s-",
        color="#1e8449", lw=2, label="the full defense's effect")
ax.axhline(0.30, color="#b03a2e", ls="--", lw=1.5,
           label="the inherited bar 0.30")
ax.axvspan(14, 30, color="#b03a2e", alpha=0.06)
ax.annotate("the refusal:\nnothing clears at 14", (14.4, 0.16),
            fontsize=8.5)
ax.annotate("cleared by +0.0070\n(the anchor fell 20 points)",
            (24.3, 0.34), fontsize=8.5, color="#1e8449")
ax.set_xlabel("horizon (rounds)")
ax.set_ylabel("effect at the bottom of the grid")
ax.set_title("(b) The founding refusal was a 14-round letter: the\n"
             "reference world crossed the absolute bar by 30", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (c) the bars at double horizon on the calibration row
ax = axes[1, 0]
bars = eb["classification"]["EB3_bars30"]
x = np.arange(2)
ax.bar(x - 0.17, [bars["bar_rebased_14"], bars["bar_rebased_30"]],
       width=0.34, color="#b03a2e", label="the re-based bar (ages)")
ax.bar(x + 0.17, [sp30["effect14"], sp30["effect30"]], width=0.34,
       color="#1e8449", label="the effect (grows)")
ax.axhline(0.30, color="#7f8c8d", ls="--", lw=1.3,
           label="the inherited bar (absolute)")
for xi, cl in zip(x, [bars["clearance_rebased_14"],
                      bars["clearance_rebased_30"]]):
    yv = sp30["effect14"] if xi == 0 else sp30["effect30"]
    ax.annotate(f"clearance\n+{cl:.4f}", (xi + 0.17, yv + 0.02),
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(["at 14 rounds", "at 30 rounds"])
ax.set_ylabel("effect / bar")
ax.set_title("(c) The calibration row: the re-based letter ages with the\n"
             "threat (0.3000 -> 0.3319); both (iii) hold at both horizons",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (d) the three 30-round rows - the thin-clearance pair
ax = axes[1, 1]
rows = eb["classification"]["EB5_rru05"]["the_three_rows"]
names = [r["row"].split(" (")[0] for r in rows]
effs = [r["effect30"] for r in rows]
cls = [r["clearance"] for r in rows]
x = np.arange(3)
ax.bar(x, effs, width=0.5, color="#1a5276")
ax.axhline(0.30, color="#b03a2e", ls="--", lw=1.5,
           label="the inherited bar")
for xi, e, c in zip(x, effs, cls):
    ax.annotate(f"+{c:.4f}", (xi, e + 0.015), ha="center", fontsize=9,
                color="#b03a2e" if c < 0.05 else "#1e8449")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("effect at 30 rounds")
ax.set_title("(d) The three 30-round rows: the thin-clearance pair\n"
             "(+0.0078 stored, +0.0070 measured) and the wide row",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")

fig.savefig(f"{D}/effectbar30_figure1.png", dpi=160)
plt.close(fig)
print("effectbar30_figure1.png done")
