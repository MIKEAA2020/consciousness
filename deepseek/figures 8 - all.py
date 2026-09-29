#!/usr/bin/env python3
"""Figures for corpus docs 39-41: the second S4 candidate (the stream row run
live, the ruler installed, CONTAINED awarded under 8.4), the iota maintenance
dial (the three filed fixes, the horizon law refuted), and the panel-aware
poison (the catch killed, the system unmoved, the award refusing itself).
English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

c2 = json.load(open(f"{S}/s4_candidate2_results.json"))
idr = json.load(open(f"{S}/m3_iota_dial_results.json"))
pp = json.load(open(f"{S}/m3_panelpoison_results.json"))

PERT = 6

# ================= doc 39: the second S4 candidate =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the stream row: the fork/award decoupling
ax = axes[0, 0]
v7 = c2["verdicts"]["V7_stream_row"]["rows"]
lams = [0.05, 0.10, 0.30, 0.50]
keys = ["C2_SP05", "C2_SP10", "C2_SP30", "C2_SP50"]
acc = [v7[k]["acc_final"] * 100 for k in keys]
tau = [v7[k]["final_tau"] for k in keys]
tens = [v7[k]["tension_at_own_q"] for k in keys]
award = [v7[k]["awardability"]["awarded_CONTAINED_UNDER_8_4"] for k in keys]
x = np.arange(4)
ax.plot(x, acc, "o-", color="#2e8b57", lw=1.8, ms=7,
        label="clean acc (the declared ruler)")
ax.axhline(93, color="gray", ls=":", lw=0.9)
ax.annotate("93% ruler", (2.42, 93.5), fontsize=7.5, color="gray")
for i, (a, ok) in enumerate(zip(acc, award)):
    ax.annotate("AWARDED" if ok else "REFUSED", (i, a - 5.6),
                fontsize=7.5, ha="center",
                color="#2e8b57" if ok else "#c0392b",
                fontweight="bold" if not ok else "normal")
ax.set_xticks(x)
ax.set_xticklabels([f"{l:.2f}" for l in lams])
ax.set_xlabel("stream-poison rate $\\lambda$")
ax.set_ylabel("final clean accuracy (%)")
ax.set_ylim(78, 101)
ax.set_title("(a) V7: the fork classifies, the ruler gates -\n"
             "CONTAINED-UNDER-8.4 awarded at 3 of 4 doses", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")
ax2 = ax.twinx()
ax2.plot(x, tau, "^-", color="#7d3c98", lw=1.4, ms=6, label="final $\\tau$")
ax2.plot(x, tens, "v--", color="#1a6faf", lw=1.4, ms=6,
         label="$T(0.9,\\bar q)$ (the bracket's floor)")
ax2.axhline(0.95, color="#d1721a", ls=":", lw=0.9)
ax2.annotate("band high 0.95", (0.0, 0.953), fontsize=7.5, color="#d1721a")
ax2.set_ylabel("$\\tau$, tension floor")
ax2.set_ylim(0.68, 1.0)
ax2.legend(fontsize=7.5, loc="lower right")

# (b) the awardability heatmap: the four conditions x the four doses
ax = axes[0, 1]
conds = ["(i) clean\nruler", "(ii) engage-\nment", "(iii) registry\neffect",
         "(iv) formula\nbracket", "AWARD"]
grid = np.zeros((5, 4))
for j, k in enumerate(keys):
    aw = v7[k]["awardability"]
    grid[0, j] = aw["(i)_clean_ruler_holds"]
    grid[1, j] = aw["(ii)_engagement_guarantee"]
    grid[2, j] = aw["(iii)_registry_effect_measured"]
    grid[3, j] = aw["(iv)_formula_anchored"]
    grid[4, j] = aw["awarded_CONTAINED_UNDER_8_4"]
ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i in range(5):
    for j in range(4):
        ax.text(j, i, "T" if grid[i, j] else "F", ha="center", va="center",
                fontsize=11, fontweight="bold", color="black")
ax.set_xticks(range(4))
ax.set_xticklabels([f"$\\lambda$={l:.2f}" for l in lams], fontsize=8)
ax.set_yticks(range(5))
ax.set_yticklabels(conds, fontsize=8)
ax.set_title("(b) The four conditions of the 8.4 award:\n"
             "(ii) holds everywhere; (i) breaks at 0.50", fontsize=10)

# (c) the pool ruler: the containment signature is selectivity
ax = axes[1, 0]
rarms = ["C2_SP_TWIN", "C2_SP05", "C2_SP30", "C2_SP30_NOCH"]
rlabs = ["twin", "0.05", "0.30", "0.30\nNOCH"]
gate = [c2["ruler"][a]["mean_gate_frac"] for a in rarms]
pac = [c2["ruler"][a]["mean_acc_pool"] for a in rarms]
x = np.arange(4)
ax.plot(x, [g * 100 for g in gate], "o-", color="#1a6faf", lw=1.8, ms=7,
        label="gate fraction (admits)")
ax.plot(x, [p * 100 for p in pac], "s-", color="#2e8b57", lw=1.8, ms=6,
        label="pool label acc at the gate")
ax.set_xticks(x)
ax.set_xticklabels(rlabs)
ax.set_xlabel("the ruler's arms (fresh draws, recomputed)")
ax.set_ylabel("%")
ax.set_ylim(0, 105)
ax.set_title("(c) V8: the ruler reads containment as selectivity -\n"
             "gate tightens, pool purifies; the disarmed world collapses",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")
fake = c2["ruler"]["C2_FAKE_SP30_DECL_TWIN"]["mean_abs_gap_q"]
ax.annotate(f"the FAKE: excluded at 100x the floor\n"
            f"(mean |gap| = {fake:.2f}, 9 consecutive rounds)",
            (0.30, 8), fontsize=7.5, color="#c0392b")

# (d) the Group A contingency: the channel arms wherever the gate softens
ax = axes[1, 1]
ac = c2["audit_classification"]
arms = ["TWIN_KF", "KR_K", "KNR_K", "POISON_TAU", "POISON_JOINT",
        "S1", "S2", "S3_CUTOFF"]
nact = [ac[f"C2_{a}"]["n_acting"] for a in arms]
nlab = [ac[f"C2_{a}"]["labels_rewritten"] for a in arms]
x = np.arange(len(arms))
ax.bar(x, nact, color="#7d3c98", alpha=0.75, label="seeds the channel armed")
ax.bar(x, np.array(nlab) / 25.0, color="#d1721a", alpha=0.6,
       label="labels rewritten (/25)")
ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=7, rotation=28, ha="right")
ax.set_ylabel("count")
ax.set_title("(d) The registered contingency: the fourteenth coordinate\n"
             "catches every class that softens the commit gate", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.savefig(f"{D}/s4cand2_figure1.png", dpi=170)
plt.close(fig)
print("wrote s4cand2_figure1.png")

# ================= doc 40: the iota maintenance dial =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the horizons, and the law that governed them wrong
ax = axes[0, 0]
h = idr["horizons"]
arms4 = [("ID_TWIN_BASE", "BASE ($\\epsilon_i$=0.03)", "#1a6faf"),
         ("ID_TWIN_WIDE", "WIDE ($\\epsilon_i$=0.05)", "#d1721a"),
         ("ID_TWIN_COMP", "COMP (the cone)", "#2e8b57"),
         ("ID_TWIN_REREG", "REREG (re-anchor)", "#7d3c98")]
rng = np.random.default_rng(0)
for i, (arm, lab, col) in enumerate(arms4):
    firsts = [r["first_armed"] for r in h[arm]]
    xs = np.full(len(firsts), i)
    ys = [f if f is not None else 31.2 for f in firsts]
    ax.scatter(xs, ys, s=42, color=col, zorder=3,
               edgecolor="white", linewidth=0.6)
    never = sum(1 for f in firsts if f is None)
    if never:
        ax.annotate(f"{never}/6 never", (i + 0.12, 31.2), fontsize=7.5,
                    color=col, va="center")
means = []
for arm, lab, col in arms4:
    fs = [r["first_armed"] for r in h[arm] if r["first_armed"] is not None]
    means.append((np.mean(fs) if fs else None, col))
for i, (m, col) in enumerate(means):
    if m is not None:
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, lw=2.2, zorder=4)
law = idr["classification"]["_references"]
ax.axhline(law["horizon_law_eps03"], color="#1a6faf", ls=":", lw=1.0)
ax.annotate("law: $\\epsilon_i$/drift = 24.4", (2.35, 24.9), fontsize=7.5,
            color="#1a6faf")
ax.axhline(30, color="gray", ls="--", lw=0.8)
ax.annotate("horizon 30", (0.0, 30.4), fontsize=7.5, color="gray")
ax.set_xticks(range(4))
ax.set_xticklabels([lab for _, lab, _ in arms4], fontsize=8)
ax.set_ylabel("round of first false arming")
ax.set_ylim(10, 34)
ax.set_title("(a) The horizons: every fix buys some, none buys all;\n"
             "the law (24.4 at 0.03) overshoots the measured 16.7",
             fontsize=10)

# (b) the two-point slope vs the law
ax = axes[0, 1]
eps = [0.03, 0.05]
law_line = [law["horizon_law_eps03"], law["horizon_law_eps05"]]
meas = [16.7, 21.7]
ax.plot(eps, law_line, "--", color="#c0392b", lw=1.6, marker="o", ms=5,
        label="the drift law $\\epsilon_i$/drift (slope 813/unit)")
ax.plot(eps, meas, "o-", color="#1a6faf", lw=1.8, ms=7,
        label="measured mean first-arming (slope ~250/unit)")
ax.fill_between([0.028, 0.055], [30, 30], [45, 45], color="whitesmoke")
ax.annotate("the law's 40.7 sits beyond\nthe 30-round horizon", (0.0405, 36),
            fontsize=7.5, color="#c0392b")
ax.axhline(30, color="gray", ls="--", lw=0.8)
ax.set_xlabel("deadband $\\epsilon_i$")
ax.set_ylabel("rounds to first arming")
ax.set_ylim(10, 45)
ax.set_title("(b) The horizon grows 3x more slowly in the band\n"
             "than the drift law says - the noise crosses, not the mean",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")

# (c) the trade at 0.05, resolved mild
ax = axes[1, 0]
ab = idr["arms"]["ID_P05_BASE"]["aggregate"]["acc"]
aw = idr["arms"]["ID_P05_WIDE"]["aggregate"]["acc"]
rr = np.arange(1, len(ab) + 1)
ax.plot(rr, [a * 100 for a in ab], "o-", color="#1a6faf", lw=1.6, ms=4,
        label="BASE ($\\epsilon_i$=0.03): armed r8-9")
ax.plot(rr, [a * 100 for a in aw], "s--", color="#d1721a", lw=1.6, ms=4,
        label="WIDE ($\\epsilon_i$=0.05): armed r9-10")
ax.axvline(6, color="gray", ls=":", lw=0.9)
ax.annotate("attack onset", (6.1, 92.4), fontsize=7.5, color="gray")
ax.set_xlabel("round")
ax.set_ylabel("clean accuracy (%)")
ax.set_ylim(90, 98)
ax.set_title("(c) The trade at $\\lambda$=0.05: the catch delayed 1.3 rounds,\n"
             "not lost - 0.37pp the price, the loop does not return",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (d) the scheduled record's price
ax = axes[1, 1]
rd = idr["classification"]["ID5_rereg"]["record_drift_total_per_seed"]
x = np.arange(6)
ax.bar(x, np.array(rd) * 100, color="#7d3c98", alpha=0.8)
ax.axhspan(0.6, 1.0, color="#2e8b57", alpha=0.18)
ax.annotate("the filed estimate: 0.6-1.0pp", (0.0, 2.6), fontsize=8,
            color="#2e8b57")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in range(600, 606)])
ax.set_xlabel("seed")
ax.set_ylabel("net record movement over 30 rounds (pp)")
ax.set_title("(d) REREG's price: the record walks 5.4-10.6pp - nine times\n"
             "the filed estimate; 4/6 seeds clean through all 30 rounds",
             fontsize=10)
fig.savefig(f"{D}/iotadial_figure1.png", dpi=170)
plt.close(fig)
print("wrote iotadial_figure1.png")

# ================= doc 41: the panel-aware poison =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the catch rates
ax = axes[0, 0]
pp2 = pp["classification"]["PP2_direction"]
pp3 = pp["classification"]["PP3_gate"]
doses = ["05", "10", "30"]
dl = [0.05, 0.10, 0.30]
cu = [pp2[d]["catch_uniform"] for d in doses]
cr = [pp2[d]["catch_runner"] for d in doses]
cg = [pp3[f"PP_G{d}"]["catch"] or 0.0 for d in doses]
cj = [pp3[f"PP_J{d}"]["catch"] or 0.0 for d in doses]
x = np.arange(3)
w = 0.19
ax.bar(x - 1.5 * w, cu, w, color="#1a6faf", label="UNIFORM (doc 34)")
ax.bar(x - 0.5 * w, cr, w, color="#2e8b57", label="RUNNER-UP (direction)")
ax.bar(x + 0.5 * w, cg, w, color="#d1721a", label="GATED (selection)")
ax.bar(x + 1.5 * w, cj, w, color="#c0392b", label="JOINT (the filed threat)")
ax.set_xticks(x)
ax.set_xticklabels([f"$\\lambda$={l:.2f}" for l in dl])
ax.set_ylabel("flip-catch rate (disputed / flipped)")
ax.set_ylim(-0.04, 1.02)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("(a) The gate kills the catch exactly: 0.000 at every dose\n"
             "(the direction leaves it untouched - the rule is item-level)",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper left", ncol=2)

# (b) the capacity bound and the surviving dirt
ax = axes[0, 1]
pp4 = pp["classification"]["PP4_capacity"]
rg = [pp4[f"PP_G{d}"]["realized_dose"] for d in doses]
rj = [pp4[f"PP_J{d}"]["realized_dose"] for d in doses]
eff_u = [pp2[d]["catch_uniform"] for d in doses]
surv_u = [l * (1 - c) for l, c in zip(dl, eff_u)]
ax.plot(dl, dl, ":", color="gray", lw=1.0)
ax.annotate("the filed dose", (0.24, 0.265), fontsize=7.5, color="gray",
            rotation=38)
ax.plot(dl, rg, "o-", color="#d1721a", lw=1.8, ms=6,
        label="GATED: realized dose (burns its territory)")
ax.plot(dl, rj, "^-", color="#c0392b", lw=1.8, ms=6,
        label="JOINT: realized dose (farms it)")
ax.plot(dl, surv_u, "s--", color="#1a6faf", lw=1.6, ms=6,
        label="UNIFORM: dirt surviving the catch")
ax.set_xlabel("filed poison rate $\\lambda$")
ax.set_ylabel("realized / surviving dose")
ax.set_title("(b) The capacity bound: the blind spot (~5-10% of commits)\n"
             "starves the stealthy attack; it still out-delivers the uniform",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")

# (c) the award re-grade: held everywhere, refused anyway
ax = axes[1, 0]
pp5 = pp["classification"]["PP5_award_regrade"]
groups = [("U", "#1a6faf"), ("R", "#2e8b57"), ("G", "#d1721a"),
          ("J", "#c0392b")]
for tag, col in groups:
    ys = [pp5[f"PP_{tag}{d}"]["clean_ruler"] * 100 for d in doses]
    awd = [pp5[f"PP_{tag}{d}"]["awarded_CONTAINED_UNDER_8_4"] for d in doses]
    xi = np.arange(3) + (0.18 if tag in "UR" else -0.18) \
        + (0.09 if tag in "RG" else -0.09)
    ax.plot(xi, ys, "o" if tag in "UR" else "^", color=col, lw=0, ms=7)
    for a, b in zip(xi, ys):
        pass
    lab = {"U": "uniform", "R": "runner-up", "G": "gated", "J": "joint"}[tag]
    ax.plot([], [], "o" if tag in "UR" else "^", color=col, ms=6,
            label=f"{lab}")
ax.axhline(93, color="gray", ls=":", lw=0.9)
ax.annotate("93% - the clean ruler (never broken)", (0.02, 93.4),
            fontsize=7.5, color="gray")
awd_mark = {(i, tag): pp5[f"PP_{tag}{d}"]["awarded_CONTAINED_UNDER_8_4"]
            for i, d in enumerate(doses) for tag, _ in groups}
for i, d in enumerate(doses):
    for tag, col in groups:
        xi = i + (0.18 if tag in "UR" else -0.18) \
            + (0.09 if tag in "RG" else -0.09)
        y = pp5[f"PP_{tag}{d}"]["clean_ruler"] * 100
        ax.annotate("award" if awd_mark[(i, tag)] else "refused",
                    (xi, y + 0.55), fontsize=5.6, ha="center", rotation=52,
                    color="#2e8b57" if awd_mark[(i, tag)] else "#c0392b")
ax.set_xticks(np.arange(3))
ax.set_xticklabels([f"$\\lambda$={l:.2f}" for l in dl])
ax.set_ylabel("final clean accuracy (%)")
ax.set_ylim(92.4, 98.2)
ax.set_title("(c) The world that didn't fall: the ruler holds under every\n"
             "attack - the award refuses on effect, not on collapse",
             fontsize=10)
ax.legend(fontsize=8, loc="lower right", ncol=2)

# (d) the detection signature: the implied catch
ax = axes[1, 1]
pp7 = pp["classification"]["PP7_detection"]
iu = [pp7[f"PP_U{d}"]["implied_catch"] for d in doses]
ij = [pp7[f"PP_J{d}"]["implied_catch"] for d in doses]
du = [pp7[f"PP_U{d}"]["dirt_implied_rate"] for d in doses]
dj = [pp7[f"PP_J{d}"]["dirt_implied_rate"] for d in doses]
x = np.arange(3)
ax.bar(x - 0.17, iu, 0.32, color="#1a6faf", label="uniform: implied catch")
ax.bar(x + 0.17, ij, 0.32, color="#c0392b", label="joint: implied catch")
ax.set_yscale("log")
ax.set_ylim(1e-4, 3)
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
for i in range(3):
    ax.annotate(f"dirt +{du[i]*100:.0f}pp", (i - 0.17, iu[i] * 1.35),
                fontsize=6.5, ha="center", color="#1a6faf")
    ax.annotate(f"dirt +{dj[i]*100:.0f}pp", (i + 0.17, ij[i] * 1.5),
                fontsize=6.5, ha="center", color="#c0392b")
ax.set_xticks(x)
ax.set_xticklabels([f"$\\lambda$={l:.2f}" for l in dl])
ax.set_ylabel("implied catch (catches / dirt), log scale")
ax.set_title("(d) The certificate the channel lacks: reading elevated,\n"
             "action at baseline - the evasion is in its own telemetry",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.savefig(f"{D}/papoison_figure1.png", dpi=170)
plt.close(fig)
print("wrote papoison_figure1.png")
