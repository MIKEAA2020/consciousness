#!/usr/bin/env python3
"""Figures for corpus docs 75 and 77: the cold-burn road's provenance
(the continuum at the floor, the drawn registration) and the crossing's
interior (the bracket halved, the fall back-loaded, the letter's razor,
the award's pocket). English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

cb = json.load(open(f"{S}/s12_coldburn_results.json"))
ci = json.load(open(f"{S}/s13_interior275_results.json"))

DEEP = [603, 604, 606]
RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"

# ============ doc 75: the cold-burn road's provenance =======================
seeds = cb["classification"]["CB_seeds"]
Ts = {int(s): v["T_reg"] for s, v in seeds.items()}
resid = cb["classification"]["CB3_height_path"]["per_seed_residual"]
persist5 = {int(s): v["persist_share5"] for s, v in seeds.items()}
flate = {int(s): v["F_late"] for s, v in seeds.items()}
latebr = {int(s): v["late_b_r"] for s, v in seeds.items()}
anchf = {int(s): v["anchor_F_late"] for s, v in seeds.items()}
cold_burn = [607, 615, 619]
burners = [600, 603, 604, 606, 607, 615, 616, 617, 619, 620, 622, 623]
hot_burn = [s for s in burners if s not in cold_burn]
cold_ok = [601, 602, 608, 611, 612, 613, 618, 621]
rest = [int(s) for s in seeds if int(s) not in burners + cold_ok]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the burn's depth against the registration - the continuum at the floor
ax = axes[0, 0]
for grp, col, lab, mk in [
        (cold_burn, BLUE, "cold-burn road (607/615/619)", "D"),
        (hot_burn, RED, "hot-burn road (9 seeds)", "o"),
        (cold_ok, GREEN, "cold, consumed (8 seeds)", "s"),
        (rest, GRAY, "body, consumed", "^")]:
    ax.scatter([seeds[str(s)]["b_star"] for s in grp],
               [flate[s] for s in grp], c=col, marker=mk, s=64,
               label=lab, edgecolors="white", linewidths=0.6, zorder=3)
ax.axhline(-0.02, color=RED, ls="--", lw=1.4)
ax.annotate("the burn floor (-0.02)", (0.096, -0.026), fontsize=8.5,
            color=RED)
ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.07)
xs = np.array([seeds[str(s)]["b_star"] for s in seeds])
ys = np.array([flate[int(s)] for s in seeds])
k, b0 = np.polyfit(xs, ys, 1)
xl = np.linspace(xs.min() - 0.004, xs.max() + 0.004, 50)
ax.plot(xl, k * xl + b0, color=GRAY, ls=":", lw=1.4, zorder=1)
ax.annotate("the cold burns: mid registrations under the universal fall\n"
            "(no separating column - the ranges touch at b* = 0.058-0.059)",
            (0.060, -0.049), fontsize=8.3, color=BLUE)
ax.set_xlabel("the registration b* (round-5 blind mass)")
ax.set_ylabel("F_late (the composition verdict)")
ax.set_title("(a) THE HEIGHT-PATH FORK -> CONTINUUM: burn depth against\n"
             "registration height - every stratum's range overlaps",
             fontsize=9.5)
ax.legend(fontsize=7.6, loc="lower left")

# (b) the filed candidate refuted - the registration's composition
ax = axes[0, 1]
labels = ["persistent share\nof b*'s blind mass", "drawn share\n(fresh blindness)"]
cbm = np.mean([persist5[s] for s in cold_burn])
okm = np.mean([persist5[s] for s in cold_ok])
hbm = np.mean([persist5[s] for s in hot_burn])
xpos = np.arange(2)
w = 0.25
ax.bar(xpos - w, [cbm * 100, (1 - cbm) * 100], w, color=BLUE,
       label="cold-burn (3)")
ax.bar(xpos, [okm * 100, (1 - okm) * 100], w, color=GREEN,
       label="cold-ok (8)")
ax.bar(xpos + w, [hbm * 100, (1 - hbm) * 100], w, color=RED,
       label="hot-burn (9)")
for x, v in zip(xpos - w, [cbm * 100, (1 - cbm) * 100]):
    ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.4)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("share of the registration's blind items (%)")
ax.set_ylim(0, 112)
ax.set_title("(b) THE FILED CANDIDATE REFUTED: ~95% of every registration's\n"
             "blind mass is FRESH-DRAW blindness; the persistent shared\n"
             "region (~45 items) contributes 2-4 items", fontsize=9.5)
ax.legend(fontsize=8, loc="upper center")

# (c) the anchor decomposition - the burn is assembled
ax = axes[1, 0]
for grp, col, lab, mk in [
        (cold_burn, BLUE, "cold-burn", "D"),
        (cold_ok, GREEN, "cold-ok", "s"),
        (hot_burn, RED, "hot-burn", "o")]:
    ax.scatter([anchf[s] for s in grp], [flate[s] for s in grp], c=col,
               marker=mk, s=64, label=lab, edgecolors="white",
               linewidths=0.6, zorder=3)
lim = np.array([-0.06, 0.10])
ax.plot(lim, lim, color=GRAY, ls=":", lw=1.3, zorder=1)
ax.annotate("y = x: no defense", (-0.045, -0.040), fontsize=8.2,
            color=GRAY, rotation=38)
ax.axhline(-0.02, color=RED, ls="--", lw=1.2)
for s in (607, 615, 619, 601, 618):
    ax.annotate(str(s), (anchf[s], flate[s]), fontsize=7.6,
                xytext=(5, 4), textcoords="offset points", color=GRAY)
ax.annotate("every world falls below its anchor\n"
            "(the differential is universal, -0.019 to -0.037);\n"
            "no cold burner's anchor burns - the shared course\n"
            "places the stratum, the fall crosses the floor",
            (-0.055, 0.030), fontsize=8.3, color=BLUE)
ax.set_xlabel("the anchor's course (undefended F_late)")
ax.set_ylabel("the armed world's F_late")
ax.set_title("(c) THE ANCHOR DECOMPOSITION AT 24: differential-carried at\n"
             "the criterion - assembled in truth (low course + universal fall)",
             fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")

# (d) the fall's companions - stratUM-flat exposures
ax = axes[1, 1]
cats = ["repairs\n(total)", "repairs\nwriting wrong\nlabels",
        "undisputed\nflips (all land\nblind)", "gate\nflips"]
def cummean(key, grp):
    return np.mean([seeds[str(s)]["cum"][key] for s in grp])
xpos = np.arange(len(cats))
for i, (grp, col, lab) in enumerate([
        (cold_burn, BLUE, "cold-burn (3)"),
        (cold_ok, GREEN, "cold-ok (8)"),
        (hot_burn, RED, "hot-burn (9)")]):
    vals = [cummean("rep", grp), cummean("rep_wrong", grp),
            cummean("flip_pass", grp), cummean("gate", grp)]
    ax.bar(xpos + (i - 1) * 0.25, vals, 0.25, color=col, label=lab)
    for x, v in zip(xpos + (i - 1) * 0.25, vals):
        if v < 400:
            ax.text(x, v + 12, f"{v:.0f}", ha="center", fontsize=7.4)
ax.set_xticks(xpos)
ax.set_xticklabels(cats, fontsize=8.4)
ax.set_ylabel("cumulative items, rounds 6-14 (mean per seed)")
ax.set_title("(d) THE FALL'S COMPANIONS ARE EVERYONE'S COMPANIONS: repairs,\n"
             "deference content (7.4%), and blind-landing poison - all\n"
             "stratum-flat; the cold road is not a different door",
             fontsize=9.5)
ax.legend(fontsize=8)

fig.suptitle("Doc 75 - the cold-burn road's provenance: the continuum at the floor",
             fontsize=12, fontweight="bold")
fig.savefig(f"{D}/coldburn_figure1.png", dpi=170)
plt.close(fig)

# ============ doc 77: the crossing's interior ================================
c = ci["classification"]
fam = {float(k): v for k, v in
       c["CI5_last_interval"]["origin_slopes_by_dose"].items()}
ladder = c["CI2_anchor"]["ladder"]
fr = {float(k): v for k, v in
      c["CI3_curve"]["fractions_of_max"].items()}

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the six-dose family with the interior decomposition
ax = axes[0, 0]
ds = sorted(fam)
ax.plot(ds, [fam[d] for d in ds], "o-", color=PURPLE, lw=2.2, ms=7,
        label="origin slope s(lambda)")
ax.axhline(1.0, color=GRAY, ls=":", lw=1.3,
           label="proportionality (the crossing)")
falls = c["CI5_last_interval"]["the_falls"]
for (x0, x1), share, lab in zip(
        [(0.15, 0.20), (0.20, 0.25), (0.25, 0.275), (0.275, 0.30)],
        [0.512, 0.324, falls["d3a_(0.25,0.275]"],
         falls["d3b_(0.275,0.30]"]],
        ["30%", "19%", "12%", "39%"]):
    xm, ym = (x0 + x1) / 2, (fam[x0] + fam[x1]) / 2
    ax.annotate(lab, (xm, ym), fontsize=9, ha="center",
                xytext=(0, 10), textcoords="offset points", color=RED)
ax.annotate("BACK-LOADED: 77% of the last\ninterval's fall is in its\n"
            "second half (0.275, 0.30]", (0.205, 1.02), fontsize=8.6,
            color=RED)
ax.annotate("lambda* in (0.275, 0.30]", (0.262, 1.55), fontsize=9,
            color=RED)
ax.set_xlabel("attack dose (lambda)")
ax.set_ylabel("the window-bottom's origin slope")
ax.set_title("(a) THE SIX-DOSE FAMILY: the bracket halved - the fall's\n"
             "shares 30/19/12/39, the last half-interval alone carrying\n"
             "more than any full interval except the first", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

# (b) the anchor ladder's interior - the dip and the stakes' peak
ax = axes[0, 1]
dl = [0.05, 0.10, 0.15, 0.20, 0.25, 0.275, 0.30]
ladder_keys = ["0.05", "0.10", "0.15", "0.20", "0.25", "0.275", "0.30"]
accs = [ladder[k] for k in ladder_keys]
ax.plot(dl, accs, "o-", color=BLUE, lw=2.2, ms=7,
        label="anchor accuracy (undefended)")
lin = 0.1752 + (0.1859 - 0.1752) * (0.275 - 0.25) / 0.05
ax.plot([0.25, 0.30], [0.1752, 0.1859], ls=":", color=GRAY, lw=1.4,
        label="the chord through (0.25, 0.30)")
ax.scatter([0.275], [ladder["0.275"]], c=RED, s=130, zorder=4, marker="*")
ax.annotate("0.1481 - BELOW the chord (0.1806)\nand below 0.30's 0.1859\n"
            "(directional: sd 0.049, one crash seed)",
            (0.275, 0.148), xytext=(0.028, 0.30), fontsize=8.2,
            color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax2 = ax.twinx()
mx = {0.05: 0.110, 0.15: 0.578, 0.20: 0.716, 0.25: 0.781,
      0.275: ladder and c["CI2_anchor"]["max_effect_275"], 0.30: 0.770}
ax2.plot(sorted(mx), [mx[d] for d in sorted(mx)], "s--", color=ORANGE,
         lw=1.6, ms=6, label="max effect (twin - anchor)")
ax2.set_ylabel("the max effect", color=ORANGE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax.annotate("the stakes peak INSIDE the last\ninterval: 0.8078, the grid's\n"
            "largest", (0.14, 0.62), fontsize=8.6, color=ORANGE)
ax.set_xlabel("attack dose (lambda)")
ax.set_ylabel("anchor accuracy")
ax.set_title("(b) THE LADDER'S INTERIOR: the anchor dips at 0.275 (inside\n"
             "the spread) while the defense's stakes set the grid's record",
             fontsize=9.5)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.6, loc="center right")

# (c) the curve at 0.275 with the letter's razor
ax = axes[1, 0]
ps = sorted(fr)
ax.plot(ps, [fr[p] for p in ps], "o-", color=PURPLE, lw=2.2, ms=7,
        label="fraction of max at lambda = 0.275")
ax.plot([0, 1], [0, 1], ls=":", color=GRAY, lw=1.3,
        label="proportionality")
ax.axhline(0.3896, color=RED, ls="--", lw=1.4,
           label="the re-based letter (0.3896)")
ax.scatter([0.25], [fr[0.25]], c=RED, s=140, zorder=4, marker="*")
ax.annotate("f(0.25) = 0.3856:\nthe letter DIES by 0.004\n"
            "(the razor - inside the ~0.03 spread)", (0.25, 0.386),
            xytext=(0.36, 0.18), fontsize=8.4, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.annotate("still ABOVE proportionality:\nthe crossing is NOT here -\n"
            "lambda* in (0.275, 0.30]", (0.62, 0.52), fontsize=8.6,
            color=GREEN)
ax.set_xlabel("the channel's cleaning probability p")
ax.set_ylabel("fraction of the max effect")
ax.set_title("(c) THE LETTER'S RAZOR AT 0.275: the coincidence SPLITS -\n"
             "the letter's end in (0.25, 0.275], the crossing in (0.275, 0.30]",
             fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

# (d) the award's pocket
ax = axes[1, 1]
doses = [0.10, 0.15, 0.20, 0.25, 0.275, 0.30]
status = [1, 1, 0, 0, 1, 1]
cols = [GREEN if s else RED for s in status]
ax.bar([str(d) for d in doses], status, color=cols, alpha=0.85,
       edgecolor="white")
for x, s, d in zip(range(len(doses)), status, doses):
    ax.text(x, s + 0.03, "AWARDED" if s else "FAILS (ii)", ha="center",
            fontsize=8.6, color=cols[x], fontweight="bold")
ax.axvspan(1.5, 3.5, color=RED, alpha=0.08)
ax.annotate("THE POCKET at (0.15, 0.25]: the honesty condition\n"
            "fails only here - bounded on both sides\n"
            "(doc 51's own census awarded SP30; doc 71's\n"
            '"0.15-specific" corrected)', (0.05, 0.52), fontsize=8.4,
            color=RED)
ax.set_ylim(0, 1.25)
ax.set_yticks([])
ax.set_xlabel("attack dose (lambda)")
ax.set_title("(d) THE AWARD'S POCKET: the five-condition award across the dial\n"
             "- a convention's shadow, not the world's", fontsize=9.5)

fig.suptitle("Doc 77 - the crossing's interior: the bracket halved, the fall back-loaded",
             fontsize=12, fontweight="bold")
fig.savefig(f"{D}/interior275_figure1.png", dpi=170)
plt.close(fig)

print("wrote coldburn_figure1.png and interior275_figure1.png")
