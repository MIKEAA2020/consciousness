#!/usr/bin/env python3
"""Figures for corpus docs 54-57: the overfill clause, the straddle at
twelve seeds, the half-buying dial, and the excursion tails. English
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

bnd = json.load(open(f"{S}/s5_boundaries_results.json"))
exc = json.load(open(f"{S}/s5_excursion_results.json"))
d51 = json.load(open(f"{S}/s4_candidate4_results.json"))
d47 = json.load(open(f"{S}/m3_compcert_results.json"))

OVER, FILL, BURN = 0.08, 0.02, -0.02
PERT = 6


def mean_series(res, arm, key, rounds=None):
    ag = res["arms"][arm]["aggregate"][key]
    return [(i + 1, v) for i, v in enumerate(ag)
            if v is not None][:rounds]


# =================== doc 54: the overfill clause ===========================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the ground trajectories: the drain
ax = axes[0, 0]
for arm, c, lab in [("S5_RR_U05", "#c0392b", "armed RR_U05 (5-cond AWARD)"),
                    ("S5_RR_U10", "#e67e22", "armed RR_U10 (refused-(iv))"),
                    ("S5_RR_U05_NOCH", "#7f8c8d", "anchor U05 (loop)"),
                    ("S5_RR_U10_NOCH", "#95a5a6", "anchor U10 (loop)")]:
    ser = mean_series(bnd, arm, "comp_F")
    ax.plot([r for r, _ in ser], [v for _, v in ser], "-", color=c,
            lw=1.7, label=lab)
ser = mean_series(d47, "CM_T30_REREG", "comp_F")
ax.plot([r for r, _ in ser], [v for _, v in ser], ":", color="#4a4a4a",
        lw=1.5, label="doc 47: REREG twin (no attack)")
for y, t in [(OVER, "OVERFILLED"), (FILL, "ON-COURSE floor"),
             (BURN, "BURNED floor")]:
    ax.axhline(y, color="black", ls="--", lw=0.9)
    ax.annotate(t, (47, y + 0.004), fontsize=7.5)
ax.set_xlabel("round (30-round horizon)")
ax.set_ylabel("the fill F(r) (mean over seeds)")
ax.set_title("(a) The drain: the armed REREG worlds' grounds sit at the\n"
             "fill floor while their anchors overfill at +0.16", fontsize=10)
ax.legend(fontsize=7.5, loc="center right")

# (b) the walk counts: one under attack vs 7-11 unattacked
ax = axes[0, 1]
walks_atk = [sum(1 for t in rr["traj"] if t.get("rereg_fired"))
             for rr in bnd["arms"]["S5_RR_U05"]["runs"]]
walks_atk2 = [sum(1 for t in rr["traj"] if t.get("rereg_fired"))
              for rr in bnd["arms"]["S5_RR_U10"]["runs"]]
walks_free = [sum(1 for t in rr["traj"] if t.get("rereg_fired"))
              for rr in exc["arms"]["S5_X_REREG"]["runs"]]
data = [walks_atk, walks_atk2, walks_free]
bp = ax.boxplot(data, tick_labels=["RR_U05\n(under attack)",
                                   "RR_U10\n(under attack)",
                                   "REREG twin\n(no attack, 60r)"],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["#c0392b", "#e67e22", "#4a4a4a"]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.set_ylabel("re-registration walks per seed")
ax.set_title("(b) The walk is in-band-gated: under attack it fires once\n"
             "(round 10, the last in-band moment) and never again",
             fontsize=10)

# (c) the masking map: all the ingredients
ax = axes[1, 0]
pts = [("doc47 BASE twin", d47, "CM_T30_BASE", "#1a6faf"),
       ("doc47 armed U05", d47, "CM_U05", "#2e8b57"),
       ("doc47 armed U10", d47, "CM_U10", "#2e8b57"),
       ("doc47 REREG twin", d47, "CM_T30_REREG", "#4a4a4a"),
       ("RR_U05 anchor", bnd, "S5_RR_U05_NOCH", "#7f8c8d"),
       ("RR_U05 armed", bnd, "S5_RR_U05", "#c0392b"),
       ("RR_U10 anchor", bnd, "S5_RR_U10_NOCH", "#95a5a6"),
       ("RR_U10 armed", bnd, "S5_RR_U10", "#e67e22")]
for i, (lab, res, arm, c) in enumerate(pts):
    f = res["arms"][arm]["aggregate"]["comp_F_late_mean"]
    acc = res["arms"][arm]["aggregate"]["acc"][-1]
    ax.scatter(i, f, s=90, color=c, zorder=3, edgecolor="white", lw=0.8)
    ax.annotate(f"{f:+.3f}", (i, f), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5)
ax.axhline(OVER, color="black", ls="--", lw=0.9)
ax.axhline(FILL, color="black", ls="--", lw=0.9)
ax.axhline(BURN, color="black", ls="--", lw=0.9)
ax.set_xticks(range(len(pts)))
ax.set_xticklabels([p[0].replace("doc47 ", "") for p in pts],
                   rotation=28, ha="right", fontsize=7.5)
ax.set_ylabel("F_late")
ax.set_title("(c) The masking map: no armed attacked world overfills -\n"
             "the attack's drain meets the defense's cleaning", fontsize=10)

# (d) the (iv) boundary: RR_U10's tau tail
ax = axes[1, 1]
for arm, c, lab in [("S5_RR_U05", "#c0392b", "RR_U05 (slope +0.0022)"),
                    ("S5_RR_U10", "#e67e22", "RR_U10 (slope -0.0005)")]:
    ser = mean_series(bnd, arm, "tau_sys")
    ax.plot([r for r, _ in ser], [v for _, v in ser], "-", color=c,
            lw=1.7, label=lab)
    r_f, v_f = ser[-1]
    ax.annotate(f"{v_f:.4f}", (r_f, v_f), textcoords="offset points",
                xytext=(4, -2), fontsize=7.5, color=c)
ax.axhline(0.9 + 0.05, color="gray", ls=":", lw=1.0)
ax.annotate("the bracket's ceiling 0.95", (18, 0.9510), fontsize=7.5)
ax.set_xlabel("round")
ax.set_ylabel("tau_sys (mean)")
ax.set_xlim(15, 31)
ax.set_title("(d) The bracket's teeth at 30 rounds: RR_U10's late slope\n"
             "turns negative by five ten-thousandths - refused on (iv)",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")
fig.suptitle("The overfill clause (doc 54): the cell that cannot be "
             "occupied", fontsize=11)
fig.savefig(f"{D}/s4overfill_figure1.png", dpi=160)
plt.close(fig)
print("wrote s4overfill_figure1.png")

# =================== doc 55: the straddle ==================================
st = bnd["classification"]["BD4_straddle"]
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the twelve seeds against the floor
ax = axes[0, 0]
vals = st["F_late_per_seed"]
orig, fresh = vals[:6], vals[6:]
ax.scatter(range(6), orig, s=90, color="#1a6faf", zorder=3,
           edgecolor="white", label="doc 51's seeds 600-605")
ax.scatter(range(6, 12), fresh, s=90, color="#e67e22", zorder=3,
           edgecolor="white", label="fresh seeds 606-611")
ax.axhline(BURN, color="#7b241c", ls="--", lw=1.4)
ax.axhline(0, color="gray", ls=":", lw=0.9)
ax.axhspan(-0.02, -0.075, color="#f5dcd4", zorder=0)
for i, v in enumerate(vals):
    ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7)
ax.annotate("the burn floor -0.02", (0.1, -0.0255), fontsize=8,
            color="#7b241c")
ax.annotate("the mean -0.0249 (BURNED)", (4.2, -0.055), fontsize=8.5)
ax.axhline(np.mean(vals), color="#7b241c", ls="-", lw=1.2)
ax.set_xticks(range(12))
ax.set_xticklabels([f"{s}" for s in range(600, 612)], fontsize=7.5)
ax.set_ylabel("CAP30 armed F_late per seed")
ax.set_title("(a) The twelve seeds: a body at -0.015, a tail at -0.055,\n"
             "the floor between them - 5 of 12 below", fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (b) the mean's movement
ax = axes[0, 1]
ax.plot([6, 12], [-0.0269, -0.0249], "o-", color="#7b241c", lw=1.8,
        ms=9)
ax.axhline(BURN, color="black", ls="--", lw=1.3)
ax.annotate("-0.0269", (6, -0.0277), fontsize=9, ha="center")
ax.annotate("-0.0249", (12, -0.0237), fontsize=9, ha="center")
ax.annotate("the burn floor", (8.2, -0.0207), fontsize=8.5)
ax.fill_between([5.5, 12.5], BURN, -0.075, color="#f5dcd4", zorder=0)
ax.set_xlim(5.5, 12.5)
ax.set_xticks([6, 12])
ax.set_xticklabels(["N = 6 (doc 51)", "N = 12 (this battery)"])
ax.set_ylabel("the mean F_late")
ax.set_title("(b) The founding refusal holds at doubled evidence -\n"
             "the margin thins from 0.0069 to 0.0049", fontsize=10)

# (c) the two readings
ax = axes[1, 0]
cats = ["mean-governs\n(the registered rule)", "majority-governs\n"
       "(the counterfactual)"]
vals2 = [st["F_late_mean"], np.mean(st["F_late_per_seed"])]
verds = [st["verdict_mean"], st["verdict_majority"]]
cols = ["#7b241c" if v == "BURNED" else "#1a6faf" for v in verds]
ax.bar([0, 1], [5, 7], color=cols, alpha=0.85, width=0.5)
ax.bar([0, 1], [7, 5], bottom=[5, 7], color="white",
       edgecolor=cols, lw=1.2, width=0.5)
for i, (n, v) in enumerate([(5, "BURNED"), (7, "CONSUMED")]):
    ax.annotate(f"{n}", (i, n / 2), ha="center", fontsize=10,
                color="white" if i == 0 else cols[i], weight="bold")
    ax.annotate(f"{12-n}", (i, n + (12 - n) / 2), ha="center",
                fontsize=10, color=cols[i])
ax.set_xticks([0, 1])
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("seeds (of 12)")
ax.set_title("(c) The boundary's two faces: the mean says BURNED (the\n"
             "refusal stands), the seeds say CONSUMED 7-5", fontsize=10)

# (d) the differential's stability
ax = axes[1, 1]
ax.plot([6, 12], [-0.0391, -0.0350], "o-", color="#c0392b", lw=1.8,
        ms=9, label="the defense's ground-delta")
ax.plot([6, 12], [0.0122, 0.0101], "s-", color="#7d3c98", lw=1.8,
        ms=9, label="the anchor's ground")
ax.axhline(0, color="gray", ls=":", lw=0.9)
ax.set_xticks([6, 12])
ax.set_xticklabels(["N = 6", "N = 12"])
ax.set_ylabel("F_late")
ax.set_title("(d) What did not move: the direction is robust - the\n"
             "defense pays the ground -0.035 at doubled evidence",
             fontsize=10)
ax.legend(fontsize=8, loc="center right")
fig.suptitle("The straddle at twelve seeds (doc 55): the founding "
             "refusal holds", fontsize=11)
fig.savefig(f"{D}/straddle_figure1.png", dpi=160)
plt.close(fig)
print("wrote straddle_figure1.png")

# =================== doc 56: the half-buying dial ==========================
hd = bnd["classification"]["BD5_hd_rows"]
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the p-curve
ax = axes[0, 0]
ps = [0, 0.25, 0.5, 1.0]
fr = [0, hd["S5_D25_05"]["effect_fraction_of_full"],
      hd["S5_D50_05"]["effect_fraction_of_full"],
      hd["S5_D100_05"]["effect_fraction_of_full"]]
ax.plot(ps, fr, "o-", color="#2e8b57", lw=1.9, ms=9,
        label="measured: fraction of the full effect")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2,
        label="the linear reference")
for p, f in zip(ps[1:], fr[1:]):
    ax.annotate(f"{f:.3f}", (p, f), textcoords="offset points",
                xytext=(8, -3), fontsize=9)
ax.set_xlabel("repair_p (the fraction of disputed labels repaired)")
ax.set_ylabel("effect as a fraction of the full defense")
ax.set_title("(a) The dial's curve is superlinear: the first quarter of\n"
             "the cleaning buys 57% of the defense", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (b) the ruler's letter
ax = axes[0, 1]
ps2 = [0.25, 0.5, 1.0]
accs = [hd[a]["acc_final"] for a in ["S5_D25_05", "S5_D50_05",
                                     "S5_D100_05"]]
ax.plot(ps2, accs, "o-", color="#c0392b", lw=1.9, ms=9)
ax.axhline(0.93, color="black", ls="--", lw=1.4)
ax.axhline(0.8463, color="#7f8c8d", ls=":", lw=1.2)
ax.axhline(0.9559, color="#2e8b57", ls=":", lw=1.2)
for p, a in zip(ps2, accs):
    ax.annotate(f"{a:.4f}", (p, a), textcoords="offset points",
                xytext=(8, -3), fontsize=9)
ax.annotate("the ruler's letter 93%", (0.27, 0.9307), fontsize=8.5)
ax.annotate("the anchor 0.8463", (0.27, 0.8490), fontsize=8.5,
            color="#7f8c8d")
ax.annotate("the twin 0.9559", (0.27, 0.9566), fontsize=8.5,
            color="#2e8b57")
ax.annotate("D50 misses by 0.0007", (0.5, 0.9215), fontsize=9,
            color="#7b241c", ha="center")
ax.set_xlabel("repair_p")
ax.set_ylabel("armed final accuracy")
ax.set_title("(b) The letter before the bar: D50 refuses on (i) by seven\n"
             "ten-thousandths; p* ~ 0.51 interpolated", fontsize=10)

# (c) the bar hierarchy
ax = axes[1, 0]
lines = [(0.763, "#c0392b", "the ruler's letter: 0.763 of the threat"),
         (0.3896, "#2e8b57", "the re-based bar: 0.389"),
         (1.0, "#4a4a4a", "full restoration: 1.0")]
for v, c, lab in lines:
    ax.axhline(v, color=c, ls="--", lw=1.5)
    ax.annotate(lab, (0.02, v + 0.015), fontsize=8.5, color=c)
ax.axhspan(2.0, 2.9, color="#f5dcd4", zorder=0)
ax.annotate("the inherited bar: 2.74x the threat (off the chart)",
            (0.02, 2.05), fontsize=8.5, color="#7b241c")
pts3 = [(0.25, 0.571), (0.5, 0.757), (1.0, 1.010)]
for p, f in pts3:
    ax.scatter(p, f, s=110, color="#c0392b", zorder=3,
               edgecolor="white", lw=1.0)
    ax.annotate(f"p={p}\n{f:.3f}", (p, f), textcoords="offset points",
                xytext=(9, -6), fontsize=8.5)
ax.set_xlim(0, 1.25)
ax.set_ylim(0, 2.9)
ax.set_xlabel("repair_p")
ax.set_ylabel("effect as a fraction of the threat")
ax.set_title("(c) The hierarchy at 0.05: the ruler binds at 0.76, the\n"
             "re-based bar at 0.39, the inherited never", fontsize=10)

# (d) the grounds by p
ax = axes[1, 1]
gs = [hd[a]["ground_F_late"] for a in ["S5_D25_05", "S5_D50_05",
                                       "S5_D100_05"]]
ax.bar([f"p={p}" for p in ps2], gs, color="#1a6faf", alpha=0.85,
       width=0.5)
ax.axhline(FILL, color="black", ls="--", lw=1.1)
ax.axhline(BURN, color="black", ls="--", lw=1.1)
for i, g in enumerate(gs):
    ax.annotate(f"{g:+.4f}", (i, g + 0.001), ha="center", fontsize=9)
ax.set_ylabel("ground F_late")
ax.set_title("(d) The dial's blind sides: all grounds ON-COURSE, the\n"
             "quarter-buyer's the fullest (freest committee)", fontsize=10)
fig.suptitle("The half-buying defense (doc 56): the dial's middle priced",
             fontsize=11)
fig.savefig(f"{D}/dial_figure1.png", dpi=160)
plt.close(fig)
print("wrote dial_figure1.png")

# =================== doc 57: the excursion tails ===========================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the four twins' mean fill trajectories
ax = axes[0, 0]
cols = {"S5_X_BASE": "#1a6faf", "S5_X_WIDE": "#e67e22",
        "S5_X_COMP": "#7d3c98", "S5_X_REREG": "#c0392b"}
for arm, c in cols.items():
    ser = mean_series(exc, arm, "comp_F")
    ax.plot([r for r, _ in ser], [v for _, v in ser], "-", color=c,
            lw=1.7, label=arm.replace("S5_X_", ""))
    ax.annotate(f"{ser[-1][1]:+.3f}", (ser[-1][0], ser[-1][1]),
                textcoords="offset points", xytext=(4, -2), fontsize=7.5,
                color=c)
for y, t in [(OVER, "the 0.08 ceiling"), (FILL, "fill floor")]:
    ax.axhline(y, color="black", ls="--", lw=0.9)
    ax.annotate(t, (44, y + 0.003), fontsize=7.5)
ax.axvline(30, color="gray", ls=":", lw=1.0)
ax.annotate("doc 40's horizon", (30.5, 0.155), fontsize=7.5,
            color="gray")
ax.set_xlabel("round (60-round horizon)")
ax.set_ylabel("the mean fill F(r)")
ax.set_title("(a) The healthy fill does not plateau: every dial rising at\n"
             "60 rounds; COMP fails hardest (+0.027 per 30)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (b) the crossing census
ax = axes[0, 1]
for i, (arm, c) in enumerate(cols.items()):
    fc = exc["classification"]["ET_twins"][arm]["first_crossing_rounds"]
    ax.scatter([i] * len(fc), fc, s=70, color=c, zorder=3, alpha=0.85,
               edgecolor="white", lw=0.8)
    for j, r in enumerate(fc):
        ax.annotate(str(r), (i, r), textcoords="offset points",
                    xytext=(9, 0), fontsize=7, color=c)
ax.axhline(30, color="gray", ls=":", lw=1.2)
ax.annotate("doc 40's horizon", (-0.35, 31), fontsize=8, color="gray")
ax.set_xticks(range(4))
ax.set_xticklabels(["BASE", "WIDE", "COMP", "REREG"], fontsize=9)
ax.set_ylabel("first round the seed's fill crosses 0.08")
ax.set_title("(b) Every seed of every dial touches the ceiling - the\n"
             "earliest crossings at rounds 16-17, inside the old horizon",
             fontsize=10)

# (c) the iota false-positive horizon
ax = axes[1, 0]
for i, (arm, c) in enumerate(cols.items()):
    tw = exc["classification"]["ET_twins"][arm]
    fa = tw["per_seed_first_armed"]
    ax.scatter([i - 0.08] * len(fa), fa, s=60, color=c, zorder=3,
               edgecolor="white", lw=0.7, label=None)
    tot60 = sum(tw["per_seed_armed_rounds"])
    tot30 = tw["doc40_armed_rounds_at30"]
    ax.bar(i + 0.22, tot60, width=0.3, color=c, alpha=0.55)
    ax.bar(i + 0.22, tot30, width=0.3, color=c, alpha=1.0)
ax.axhline(14, color="gray", ls=":", lw=1.0)
ax.annotate("doc 36's estimate: ~14 rounds", (-0.42, 14.8), fontsize=7.5,
            color="gray")
ax.set_xticks(range(4))
ax.set_xticklabels(["BASE", "WIDE", "COMP", "REREG"], fontsize=9)
ax.set_ylabel("first armed round (dots) / armed seed-rounds (bars)")
ax.set_title("(c) The deadband buys 14-40 rounds and no more: every seed\n"
             "arms by r40; armed rounds grow 3-8x to 60 (dark = at 30)",
             fontsize=10)

# (d) BASE's excursions: visits, not residences
ax = axes[1, 1]
for rr, c in zip(exc["arms"]["S5_X_BASE"]["runs"],
                 ["#1a6faf", "#4d9fe0", "#8ab8d8", "#1a6faf",
                  "#4d9fe0", "#8ab8d8"]):
    ser = [(t["round"], t["comp_F"]) for t in rr["traj"]
           if t["comp_F"] is not None]
    ax.plot([r for r, _ in ser], [v for _, v in ser], "-", color=c,
            lw=1.0, alpha=0.9)
ax.axhline(OVER, color="black", ls="--", lw=1.2)
ax.axhline(FILL, color="black", ls="--", lw=0.9)
ax.annotate("the ceiling", (40, 0.0835), fontsize=8)
ser = mean_series(exc, "S5_X_BASE", "comp_F")
ax.plot([r for r, _ in ser], [v for _, v in ser], "-", color="#7b241c",
        lw=2.4, label="the mean (ON-COURSE at 60)")
ax.set_xlabel("round")
ax.set_ylabel("F(r) per seed")
ax.set_title("(d) BASE's six seeds: crossings without residence - the\n"
             "excursions visit the ceiling, the mean stays below",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.suptitle("The excursion tails at sixty rounds (doc 57): the false "
             "positive is a when", fontsize=11)
fig.savefig(f"{D}/excursion_figure1.png", dpi=160)
plt.close(fig)
print("wrote excursion_figure1.png")
