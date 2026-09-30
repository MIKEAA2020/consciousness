#!/usr/bin/env python3
"""Figures for corpus docs 71-73: the finishing arm (the crossing
characterized), the temperature's own registration, and the 24-seed
census. English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

fin = json.load(open(f"{S}/s9_finishing_results.json"))
tr = json.load(open(f"{S}/s10_tempreg_results.json"))
se = json.load(open(f"{S}/s11_seeds24_results.json"))

DEEP = [603, 604, 606]
MARG = [600, 607]
RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"

# ============ doc 71: the finishing arm =====================================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the five-dose origin-slope family - the crossing's character
ax = axes[0, 0]
fam = {float(k): v for k, v in
       fin["classification"]["FA5_crossing"]["origin_slopes_by_dose"]
       .items()}
ds = sorted(fam)
ax.plot(ds, [fam[d] for d in ds], "o-", color=PURPLE, lw=2.2, ms=7,
        label="the origin slope s(lambda) = f(0.25)/0.25")
ax.axhline(1.0, color=GRAY, ls=":", lw=1.3,
           label="proportionality (s = 1: the crossing)")
fall = fin["classification"]["FA5_crossing"]["the_fall"]
for (x0, x1), share, lab in zip([(0.15, 0.20), (0.20, 0.25),
                                 (0.25, 0.30)],
                                [fall["shares"]["(0.15,0.20]"],
                                 fall["shares"]["(0.20,0.25]"],
                                 fall["shares"]["(0.25,0.30]"]],
                                ["30%", "19%", "50%"]):
    xm = (x0 + x1) / 2
    ym = (fam[x0] + fam[x1]) / 2
    ax.annotate(f"{lab}", (xm, ym), fontsize=9, ha="center",
                xytext=(0, 10), textcoords="offset points", color=RED)
ax.annotate("the peak\n(2.58 at 0.15)", (0.15, 2.35), fontsize=8.5,
            color=PURPLE)
ax.annotate("lambda* in (0.25, 0.30]", (0.262, 1.35), fontsize=9,
            color=RED)
ax.set_xlabel("attack dose (lambda)")
ax.set_ylabel("the window-bottom's origin slope")
ax.set_title("(a) THE CROSSING'S CHARACTER: an arc - rise to the 0.15 peak,\n"
             "a monotone spread fall (30/19/50%), the crossing in the last\n"
             "interval - NON-MONOTONE SMOOTH at the registered criterion",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (b) the fresh curves at 0.20 and 0.25 against the stored family
ax = axes[0, 1]
f05 = {0.25: 0.571, 0.35: 0.601, 0.40: 0.669, 0.50: 0.757, 1.0: 1.010}
f15 = {0.25: 0.644, 0.35: 0.776, 0.40: 0.784, 0.50: 0.860, 0.75: 0.949,
       1.0: 1.006}
f30 = {0.25: 0.222, 0.35: 0.415, 0.40: 0.489, 0.50: 0.677, 0.75: 0.886,
       1.0: 0.986}
f20 = {float(k): v for k, v in
       fin["classification"]["FA3_curves"]["0.2"]["fractions_of_max"]
       .items()}
f25 = {float(k): v for k, v in
       fin["classification"]["FA3_curves"]["0.25"]["fractions_of_max"]
       .items()}
ax.plot(sorted(f05), [f05[p] for p in sorted(f05)], "s-", color=BLUE,
        lw=1.4, ms=4, label="0.05 (stored)")
ax.plot(sorted(f15), [f15[p] for p in sorted(f15)], "D-", color=GREEN,
        lw=1.4, ms=4, label="0.15 (doc 67)")
ax.plot(sorted(f20), [f20[p] for p in sorted(f20)], "^-", color=ORANGE,
        lw=2.0, ms=6, label="0.20 (fresh)")
ax.plot(sorted(f25), [f25[p] for p in sorted(f25)], "v-", color=PURPLE,
        lw=2.0, ms=6, label="0.25 (fresh)")
ax.plot(sorted(f30), [f30[p] for p in sorted(f30)], "o-", color=RED,
        lw=1.4, ms=4, label="0.30 (stored)")
ax.plot([0, 1], [0, 1], ":", color="#9b9b9b", lw=1.1,
        label="proportional")
ax.annotate("the leading edge:\n0.35 below its chord", (0.33, 0.50),
            fontsize=8.5, color=ORANGE)
ax.set_xlabel("repair probability p")
ax.set_ylabel("fraction of the max available effect")
ax.set_title("(b) The five curves: the S-bottom's leading edge at 0.20,\n"
             "the knee present at both fresh doses, the dip at 0.25\n"
             "(texture, inside the spread)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right", ncol=2)

# (c) the knee's migration down-p
ax = axes[1, 0]
knee_data = [("0.05", None), ("0.15", None),
             ("0.20", fin["classification"]["FA4_knees"]["by_dose"]
              ["0.2"]["knee_interval"]),
             ("0.25", fin["classification"]["FA4_knees"]["by_dose"]
              ["0.25"]["knee_interval"]),
             ("0.30", (0.25, 0.35))]
xs, ys, labs = [], [], []
for i, (dose, knee) in enumerate(knee_data):
    if knee is None:
        ax.plot(i, 0.5, "x", color=GRAY, ms=9, mew=2)
        labs.append(f"{dose}\n(absent)")
    else:
        ax.plot(i, np.mean(knee), "o", color=RED, ms=8)
        ax.annotate(f"({knee[0]:.2f}, {knee[1]:.2f})",
                    (i, np.mean(knee)), fontsize=8,
                    xytext=(0, 9), textcoords="offset points",
                    ha="center")
        labs.append(f"{dose}")
        xs.append(i); ys.append(np.mean(knee))
ax.plot(xs, ys, "--", color=RED, lw=1.4, alpha=0.6)
ax.set_xticks(range(5))
ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("the knee's interval (midpoint, p-scale)")
ax.set_ylim(0.1, 0.55)
ax.set_title("(c) The knee's migration: absent at 0.05/0.15, present from\n"
             "0.20, walking down the window as the dose climbs - the\n"
             "upper interior wears the 0.30 shape early", fontsize=10)

# (d) the anchor ladder and the letter's budget sheet
ax = axes[1, 1]
ladder = fin["classification"]["FA2_anchors"]["ladder"]
pairs = sorted(((float(k), v) for k, v in ladder.items()))
ax.plot([p for p, _ in pairs], [v for _, v in pairs], "o-",
        color="#b03a2e", lw=2, label="the anchor (undefended acc)")
ax.annotate("saturated: the last step\na coin flip (+0.011, inside\n"
            "the per-seed spread)", (0.275, 0.30), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax2 = ax.twinx()
tuning = {"0.05": 0.110, "0.15": 0.578, "0.20": 0.716, "0.25": 0.781,
          "0.30": 0.770}
ax2.plot([float(k) for k in sorted(tuning, key=float)],
         [tuning[k] for k in sorted(tuning, key=float)], "s--",
         color=BLUE, lw=1.6, ms=5,
         label="the row's max effect")
ax2.set_ylabel("the max effect (twin - anchor)", color=BLUE)
ax.set_xlabel("attack dose (lambda)")
ax.set_ylabel("the anchor's final accuracy", color="#b03a2e")
ax.set_title("(d) The ladder saturates (fully collapsed by 0.25) while the\n"
             "max effect peaks at 0.25 - and the letter holds from p=0.25\n"
             "at both fresh doses (the collapse interval = the crossing's)",
             fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

fig.savefig(f"{D}/finishing_figure1.png", dpi=160)
plt.close(fig)
print("wrote finishing_figure1.png")

# ============ doc 72: the temperature's own registration ====================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

TRs = {int(k): v for k, v in tr["classification"]["TR_seeds"].items()}
seeds_sorted = sorted(TRs, key=lambda s: TRs[s]["T_reg"])
deep_set, marg_set = set(DEEP), set(MARG)


def colr(s):
    return RED if s in deep_set else ("#e67e22" if s in marg_set else BLUE)


# (a) T per seed - the registration, with the groups
ax = axes[0, 0]
xs = np.arange(len(seeds_sorted))
vals = [TRs[s]["T_reg"] for s in seeds_sorted]
ax.bar(xs, vals, width=0.62, color=[colr(s) for s in seeds_sorted],
       alpha=0.9)
body_max = max(TRs[s]["T_reg"] for s in TRs if s not in deep_set)
deep_min = min(TRs[s]["T_reg"] for s in DEEP)
ax.axhline(body_max, color=BLUE, ls=":", lw=1.3)
ax.axhline(deep_min, color=RED, ls=":", lw=1.3)
ax.annotate(f"the body's ceiling {body_max:.4f}", (0.02, body_max),
            xytext=(0, 4), textcoords="offset points", fontsize=8,
            color=BLUE)
ax.annotate(f"the deep floor {deep_min:.4f}", (6.4, deep_min),
            xytext=(0, 4), textcoords="offset points", fontsize=8,
            color=RED)
ax.annotate("margin 0.0045\n(a razor, not a wall)", (9.4, 0.092),
            fontsize=8.5, color="k")
ax.set_xticks(xs)
ax.set_xticklabels(seeds_sorted, fontsize=8.5)
ax.set_ylabel("T (rounds 1-4 mean blind-admission rate)")
ax.set_title("(a) THE REGISTRATION: T per seed - the deep three (red) hold\n"
             "the census's top three readings, no overlap at margin 0.0045;\n"
             "the marginals (orange) are NOT hot - cold burns exist",
             fontsize=10)

# (b) T against b* - the covariate
ax = axes[0, 1]
for s in seeds_sorted:
    ax.plot(TRs[s]["T_reg"], TRs[s]["b_star"], "o", ms=8, color=colr(s))
    ax.annotate(str(s), (TRs[s]["T_reg"], TRs[s]["b_star"]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
Ts = [TRs[s]["T_reg"] for s in seeds_sorted]
bs = [TRs[s]["b_star"] for s in seeds_sorted]
sl, ic = np.polyfit(Ts, bs, 1)
xx = np.linspace(min(Ts), max(Ts), 10)
ax.plot(xx, sl * xx + ic, "--", color=GRAY, lw=1.4,
        label=f"b* = {ic:.4f} + {sl:.3f}*T")
cc = np.corrcoef(Ts, bs)[0, 1]
ax.set_xlabel("T (registered at round 5, from rounds 1-4)")
ax.set_ylabel("b* (the registered blind mass)")
ax.set_title(f"(b) The covariate: corr = {cc:.3f}, 71.6% of b*'s variance\n"
             "absorbed; the deep elevation +0.042 -> +0.011 T-corrected;\n"
             "the bimodality's residual gap 0.0194 -> 0.0115 (scrambled)",
             fontsize=10)
ax.legend(fontsize=8, loc="lower right")

# (c) the climb paths - rounds 1-5 admission, deep vs body
ax = axes[1, 0]
rs = [1, 2, 3, 4, 5]
path_deep = np.mean([[TRs[s]["adm_path_1to5"][str(r)] for r in rs]
                     for s in DEEP], axis=0)
path_body = np.mean([[TRs[s]["adm_path_1to5"][str(r)] for r in rs]
                     for s in TRs if s not in DEEP + MARG], axis=0)
path_marg = np.mean([[TRs[s]["adm_path_1to5"][str(r)] for r in rs]
                     for s in MARG], axis=0)
ax.plot(rs, path_deep, "o-", color=RED, lw=2.2,
        label="the deep three (hot bases, fast climbs)")
ax.plot(rs, path_marg, "^-", color=ORANGE, lw=1.8,
        label="the marginals (cold bases, burning anyway)")
ax.plot(rs, path_body, "s-", color=BLUE, lw=2.0,
        label="the body")
ax.axvline(5, color=GRAY, ls=":", lw=1.2)
ax.text(5.06, 0.05, "the registration\nround 5", fontsize=8, color=GRAY)
ax.set_xlabel("round (the clean prefix; the poison arms at 6)")
ax.set_ylabel("the per-round blind-admission rate")
ax.set_title("(c) THE CLIMB: T is the base (rounds 1-4), the draw the climb\n"
             "(round 5) - the deep three climb +0.10 from hot bases, the\n"
             "marginals +0.08 from cold ones: two roads into the burn",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (d) the availability - T before the poison, the forecasting
ax = axes[1, 1]
for s in seeds_sorted:
    burned = TRs[s]["verdict"] == "BURNED"
    ax.plot(TRs[s]["T_reg"], TRs[s]["F_late"], "o", ms=8,
            color=(RED if burned else BLUE), alpha=0.85)
    ax.annotate(str(s), (TRs[s]["T_reg"], TRs[s]["F_late"]), fontsize=8,
                xytext=(4, -3), textcoords="offset points")
fl = [TRs[s]["F_late"] for s in seeds_sorted]
cc2 = np.corrcoef(Ts, fl)[0, 1]
ax.axhline(-0.02, color=GRAY, ls=":", lw=1.2)
ax.text(0.107, -0.017, "the BURN floor", fontsize=8, color=GRAY)
ax.set_xlabel("T (computable before the attack arrives)")
ax.set_ylabel("F_late (the verdict window)")
ax.set_title(f"(d) THE FORECAST: corr(T, F_late) = {cc2:.3f} - the registrar\n"
             "reads the thermometer in the healthy world; red = burns, and\n"
             "607 burns at the median's edge (the boundary is T's, blurred)",
             fontsize=10)

fig.savefig(f"{D}/tempreg_figure1.png", dpi=160)
plt.close(fig)
print("wrote tempreg_figure1.png")

# ============ doc 73: the 24-seed census ====================================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

SEs = {int(k): v for k, v in se["classification"]["SE_seeds"].items()}


def colr24(s):
    if s in DEEP:
        return RED
    if s in MARG:
        return ORANGE
    return BLUE if s < 612 else GREEN


# (a) the b* census - the sorted distribution, the moat filling
ax = axes[0, 0]
bs24 = sorted(SEs, key=lambda s: SEs[s]["b_star"])
xs = np.arange(len(bs24))
ax.bar(xs, [SEs[s]["b_star"] for s in bs24], width=0.7,
       color=[colr24(s) for s in bs24], alpha=0.9)
ax.axhline(0.0938, color=RED, ls=":", lw=1.3)
ax.text(0.3, 0.0955, "doc 60's deep edge (0.0938)", fontsize=8, color=RED)
ax.axhspan(0.0744, 0.0938, color=GRAY, alpha=0.12)
ax.text(9, 0.083, "doc 60's gap\n(0.0744, 0.0938)", fontsize=8,
        color="#555555", ha="center")
for i, s in enumerate(bs24):
    if 615 <= s <= 623:
        ax.annotate(f"{s}", (i, SEs[s]["b_star"]), fontsize=6.5,
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    color=GREEN)
ax.set_xticks(xs)
ax.set_xticklabels(bs24, fontsize=6.8, rotation=90)
ax.set_ylabel("b* (the registered blind mass)")
ax.set_title("(a) THE CENSUS AT 24: the moat FILLS - four fresh seeds (green\n"
             "labels) inside the old gap - but the upper mode does not grow:\n"
             "the deep three (red) stand alone, zero recruits", fontsize=10)

# (b) the admission separation at 24
ax = axes[0, 1]
adm_sorted = sorted(SEs, key=lambda s: SEs[s]["adm_blind5"])
xs = np.arange(len(adm_sorted))
ax.bar(xs, [SEs[s]["adm_blind5"] for s in adm_sorted], width=0.7,
       color=[colr24(s) for s in adm_sorted], alpha=0.9)
body_ceiling = max(SEs[s]["adm_blind5"] for s in SEs
                   if s not in DEEP)
deep_floor = min(SEs[s]["adm_blind5"] for s in DEEP)
ax.axhline(body_ceiling, color=GREEN, ls=":", lw=1.3)
ax.axhline(deep_floor, color=RED, ls=":", lw=1.3)
ax.annotate(f"body ceiling {body_ceiling:.4f} (the fresh 623)",
            (0.2, body_ceiling), xytext=(0, 4),
            textcoords="offset points", fontsize=8, color=GREEN)
ax.annotate(f"deep floor {deep_floor:.4f}", (14, deep_floor),
            xytext=(0, 4), textcoords="offset points", fontsize=8,
            color=RED)
ax.annotate("margin 0.0141\n(< half doc 68's 0.033:\nNARROWED)", (17.5, 0.19),
            fontsize=8.5, color="k")
ax.set_xticks(xs)
ax.set_xticklabels(adm_sorted, fontsize=6.8, rotation=90)
ax.set_ylabel("the round-5 blind-admission rate")
ax.set_title("(b) THE SEPARATION SURVIVES, THINNED: no overlap at 24, margin\n"
             "0.0141 against doc 68's 0.033 - the fresh 623 presses the\n"
             "deep floor; the core held, the edge softened", fontsize=10)

# (c) T against b* at 24 - the inversion and the correlation
ax = axes[1, 0]
for s in sorted(SEs):
    ax.plot(SEs[s]["T_reg"], SEs[s]["b_star"], "o", ms=7,
            color=colr24(s), alpha=0.9)
    if s in (623, 619, 614, 603):
        ax.annotate(f"{s}", (SEs[s]["T_reg"], SEs[s]["b_star"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
Ts24 = [SEs[s]["T_reg"] for s in SEs]
bs24v = [SEs[s]["b_star"] for s in SEs]
cc3 = np.corrcoef(Ts24, bs24v)[0, 1]
ax.axvspan(0.0740, 0.1253, color=RED, alpha=0.08)
ax.text(0.086, 0.048, "the deep T band\n(doc 72) - the fresh\n623 lands inside",
        fontsize=8, color=RED)
ax.set_xlabel("T (the registered temperature)")
ax.set_ylabel("b*")
ax.set_title(f"(c) THE TEMPERATURE AT 24: corr = {cc3:.3f} (holds), but the\n"
             "no-overlap INVERTS - 623 runs hot (T=0.108) and lands\n"
             "mid-shoulder: a covariate, no longer a classifier",
             fontsize=10)

# (d) the burn census at 24
ax = axes[1, 1]
verdicts = ["BURNED", "CONSUMED", "ON-COURSE", "OVERFILLED"]
counts = [sum(1 for s in SEs if SEs[s]["verdict"] == v) for v in verdicts]
colors_v = [RED, BLUE, GREEN, ORANGE]
ax.bar(range(4), counts, width=0.55, color=colors_v, alpha=0.9)
for i, c in enumerate(counts):
    ax.annotate(f"{c}", (i, c), fontsize=10, ha="center",
                xytext=(0, 3), textcoords="offset points")
ax.set_xticks(range(4))
ax.set_xticklabels(verdicts, fontsize=9)
ax.set_ylabel("seeds (of 24)")
ax.set_title("(d) THE BURN AT EXACTLY HALF: 12/24 BURNED (5 original + 7\n"
             "fresh), no seed ON-COURSE; 9 of 12 burning seeds above the\n"
             "body-median T - and 619 burns at the census's coldest T",
             fontsize=10)

fig.savefig(f"{D}/seeds24_figure1.png", dpi=160)
plt.close(fig)
print("wrote seeds24_figure1.png")
