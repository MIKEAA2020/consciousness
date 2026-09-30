#!/usr/bin/env python3
"""Figures for corpus docs 59-61: the dial at 0.30, the bimodality's
mechanism, and the horizon-declaration letters. English labels,
constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

td = json.load(open(f"{S}/s6_topdial_results.json"))
bm = json.load(open(f"{S}/s6_bimodal_results.json"))
hz = json.load(open(f"{S}/s6_horizon_results.json"))
bnd = json.load(open(f"{S}/s5_boundaries_results.json"))
d51 = json.load(open(f"{S}/s4_candidate4_results.json"))

OVER, FILL, BURN = 0.08, 0.02, -0.02
TAIL = [603, 604, 606]
MARG = [600, 607]


def mean_series(res, arm, key, rounds=None):
    ag = res["arms"][arm]["aggregate"][key]
    return [(i + 1, v) for i, v in enumerate(ag)
            if v is not None][:rounds]


# =================== doc 59: the dial at 0.30 ==============================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the p-curve at both doses
ax = axes[0, 0]
p30 = sorted((v["p"], v["fraction_of_full"])
             for v in td["classification"]["TD2_p_curve"].values())
p05 = [(0.0, 0.0), (0.25, 0.571), (0.5, 0.757), (1.0, 1.0)]
ax.plot([p for p, _ in p30], [f for _, f in p30], "o-", color="#c0392b",
        lw=2, label="the dial at 0.30 (this battery)")
ax.plot([p for p, _ in p05], [f for _, f in p05], "s--", color="#2471a3",
        lw=1.7, label="the dial at 0.05 (doc 56)")
ax.plot([0, 1], [0, 1], ":", color="#7f8c8d", lw=1.2,
        label="the linear reference")
ax.annotate("the quarter-buyer:\n0.571 at 0.05, 0.225 at 0.30",
            (0.25, 0.40), fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("repair probability p (the cleaning bought)")
ax.set_ylabel("fraction of the full defense's effect")
ax.set_title("(a) The curve flips: superlinear at the bottom of the grid,\n"
             "an S with a sublinear bottom at the top", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (b) the armed accuracy against the letter
ax = axes[0, 1]
rows = sorted(td["classification"]["TD_dial_rows"].values(),
              key=lambda r: r["repair_p"])
ax.plot([r["repair_p"] for r in rows], [r["acc_final"] for r in rows],
        "o-", color="#c0392b", lw=2, label="armed accuracy at 0.30")
ax.axhline(0.93, color="black", ls="--", lw=1.0)
ax.annotate("the ruler's letter 93%", (0.27, 0.934), fontsize=8.5)
ax.axhline(0.1859, color="#7f8c8d", ls=":", lw=1.2)
ax.annotate("the collapsed anchor 18.6%", (0.27, 0.155), fontsize=8)
ax.axhline(0.9559, color="#7f8c8d", ls="-.", lw=1.0)
ax.annotate("the twin 95.6%", (0.27, 0.960), fontsize=8)
ps = td["classification"]["TD3_letter"]["p_star_interpolated"]
ax.plot([ps], [0.93], "*", color="#e67e22", ms=14, zorder=5)
ax.annotate(f"p* ~ {ps:.2f}\n(no p<1 holds the letter)",
            (ps - 0.30, 0.80), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("repair probability p")
ax.set_ylabel("armed final accuracy")
ax.set_title("(b) The letter's demand at 0.30: 96.6% of the threat -\n"
             "the tuning range collapses to the full defense",
             fontsize=10)

# (c) the grounds' migration
ax = axes[1, 0]
ax.plot([r["repair_p"] for r in rows],
        [r["ground_F_late"] for r in rows], "o-", color="#1e8449",
        lw=2, label="the dial's grounds at 0.30")
for y, t in [(OVER, "OVERFILLED"), (FILL, "ON-COURSE floor"),
             (BURN, "BURNED floor")]:
    ax.axhline(y, color="black", ls="--", lw=0.9)
    ax.annotate(t, (0.26, y + 0.004), fontsize=7.5)
ax.axhline(0.1466, color="#7f8c8d", ls=":", lw=1.2)
ax.annotate("the disarmed anchor's ground +0.147", (0.26, 0.152),
            fontsize=8)
ax.annotate("D25: the first armed OVERFILLED\n ground in the dial family",
            (0.25, 0.095), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("repair probability p")
ax.set_ylabel("ground F_late")
ax.set_title("(c) The drain scales with the cleaning: the grounds migrate\n"
             "from CONSUMED to the anchor's overfill as p thins",
             fontsize=10)

# (d) the hierarchy at both doses
ax = axes[1, 1]
labels = ["(i) the ruler", "(iii) both bars\n(0.30) / re-based (0.05)",
          "(iii) inherited\n(0.05 only)", "(v) the ground"]
h05 = [0.763, 0.39, 2.74, np.nan]
h30 = [0.966, 0.39, 0.39, np.nan]
x = np.arange(len(labels))
ax.bar(x - 0.18, h05, 0.34, color="#2471a3", alpha=0.75,
       label="at 0.05 (doc 56)")
ax.bar(x + 0.18, h30, 0.34, color="#c0392b", alpha=0.75,
       label="at 0.30 (this battery)")
ax.axhline(1.0, color="black", ls="--", lw=1.0)
ax.annotate("the whole threat", (2.35, 1.02), fontsize=8)
ax.text(3, 0.5, "(v) never binds:\nall grounds above\nthe burn floor",
        fontsize=8.5, ha="center")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("binding fraction of the threat")
ax.set_title("(d) The letter-vs-bar hierarchy at both ends of the grid:\n"
             "(i) rises 0.76 to 0.97; the bars stay at 0.39; (v) never",
             fontsize=10)
ax.legend(fontsize=8)

fig.savefig(f"{D}/topdial_figure1.png", dpi=160)
plt.close(fig)
print("topdial_figure1.png done")

# =================== doc 60: the bimodality's mechanism ====================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

seeds = bm["classification"]["BM_seeds"]

# (a) F_late vs b_star
ax = axes[0, 0]
bstar_of = {rr["seed"]: rr["traj"][4]["comp_bstar"]
            for rr in bm["arms"]["S7_CAP30"]["runs"]}
for s, r in seeds.items():
    si = int(s)
    if si in TAIL:
        c, m, lab = "#c0392b", "o", "the deep tail (603, 604, 606)"
    elif si in MARG:
        c, m, lab = "#e67e22", "D", "the marginals (600, 607)"
    else:
        c, m, lab = "#2471a3", "s", "the body (nine seeds)"
    ax.plot(bstar_of[si], r["F_late"], m, color=c, ms=8, label=lab)
handles, labels_ = ax.get_legend_handles_labels()
uniq = dict(zip(labels_, handles))
ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
bs = np.array([bstar_of[int(s)] for s in seeds])
fs = np.array([r["F_late"] for r in seeds.values()])
k, b0 = np.polyfit(bs, fs, 1)
xx = np.linspace(0.035, 0.115, 50)
ax.plot(xx, k * xx + b0, "-", color="#4a4a4a", lw=1.2)
ax.annotate(f"corr = -0.95", (0.088, -0.012), fontsize=9)
ax.axhline(BURN, color="black", ls="--", lw=0.9)
ax.annotate("the burn floor", (0.040, -0.026), fontsize=8)
ax.axvspan(0.076, 0.093, color="#c0392b", alpha=0.08)
ax.annotate("the b* gap", (0.0765, 0.012), fontsize=8, rotation=90)
ax.set_xlabel("b*  (the round-5 registered blind mass - clean prefix)")
ax.set_ylabel("F_late (rounds 11-14)")
ax.set_title("(a) The burn's depth is the baseline's height: the verdict's\n"
             "variance is the registration's, set before the attack",
             fontsize=10)

# (b) the per-round blind fraction, tail vs body
ax = axes[0, 1]
runs = {rr["seed"]: rr for rr in bm["arms"]["S7_CAP30"]["runs"]}
for grp, c, lab in [(TAIL, "#c0392b", "tail mean"),
                    ([si for si in runs if si not in TAIL + MARG],
                     "#2471a3", "body mean"),
                    (MARG, "#e67e22", "the marginals")]:
    ser = []
    for i in range(14):
        vals = [runs[s]["traj"][i]["blind_mass"] for s in grp]
        ser.append(np.mean(vals))
    ax.plot(range(1, 15), ser, "o-" if grp is TAIL else ("s-" if grp
             is MARG else "-"), color=c, lw=1.8 if grp is TAIL else 1.2,
            ms=4 if grp is not TAIL else 5, label=lab)
ax.axvline(5, color="#4a4a4a", ls=":", lw=1.2)
ax.annotate("round 5: the registration\n(the clean era's blindest point)",
            (5.1, 0.175), fontsize=8)
ax.axvline(6, color="#7f8c8d", ls=":", lw=1.2)
ax.annotate("round 6: the attack begins", (6.1, 0.132), fontsize=8)
ax.set_xlabel("round")
ax.set_ylabel("blind fraction of the committed pool")
ax.set_title("(b) The era compresses everyone to the same landing zone:\n"
             "late blind fractions 0.046 (tail) vs 0.047 (body)",
             fontsize=10)
ax.legend(fontsize=8, loc="center right")

# (c) the three worlds' fates against the shared baselines
ax = axes[1, 0]
twin_f = []
for rr in d51["arms"]["C4_TWIN"]["runs"]:
    n = len(rr["traj"])
    late = [t["comp_F"] for t in rr["traj"]
            if t["comp_F"] is not None and t["round"] >= n - 3]
    twin_f.append(np.mean(late))
anch_f = []
for rr in bnd["arms"]["S5_CAP30_NOCH"]["runs"]:
    n = len(rr["traj"])
    late = [t["comp_F"] for t in rr["traj"]
            if t["comp_F"] is not None and t["round"] >= n - 3]
    anch_f.append(np.mean(late))
armed_f = [r["F_late"] for r in seeds.values()]
ax.scatter(np.random.normal(0, 0.05, len(twin_f)), twin_f, s=42,
           color="#1e8449", label="the healthy twin (grows)")
ax.scatter(np.random.normal(1, 0.05, len(anch_f)), anch_f, s=42,
           color="#7f8c8d", label="the disarmed anchor (holds)")
ax.scatter(np.random.normal(2, 0.05, len(armed_f)), armed_f, s=42,
           color="#c0392b", label="the armed world (compresses)")
for y, t in [(OVER, "OVERFILLED"), (FILL, "ON-COURSE floor"),
             (BURN, "BURNED floor")]:
    ax.axhline(y, color="black", ls="--", lw=0.9)
    ax.annotate(t, (2.45, y + 0.004), fontsize=7.5)
ax.set_xticks([0, 1, 2], ["twin\n(no attack)", "anchor\n(attack, no channel)",
                          "armed\n(attack + channel)"])
ax.set_ylabel("F_late (rounds 11-14)")
ax.set_title("(c) Three fates, one set of starting lines: the twin grows,\n"
             "the anchor holds, the armed world lands seed-independent",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (d) the suspects, cleared
ax = axes[1, 1]
items = ["gate flips landing on\npreviously flipped items",
         "gate dirt repaired\nby the channel",
         "bulk dirt repaired\nby the channel",
         "committee conf on blind\ncommitted (r14, tail)",
         "committee conf on blind\ncommitted (r14, body)"]
vals = [22 / 261 * 100, 0.0, 79.5, 94.4, 94.4]
cols = ["#7f8c8d", "#7f8c8d", "#1e8449", "#2471a3", "#c0392b"]
bars = ax.barh(range(len(items)), vals, color=cols, alpha=0.8)
ax.set_yticks(range(len(items)), items, fontsize=7.5)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 1.2, i, f"{v:.1f}%" if v < 100 else f"{v:.1f}",
            va="center", fontsize=8)
ax.set_xlim(0, 112)
ax.set_xlabel("percent / value")
ax.set_title("(d) The registered suspects, cleared: the harvest roves, the\n"
             "coverage is group-invariant, no differential collapse",
             fontsize=10)

fig.savefig(f"{D}/bimodal_figure1.png", dpi=160)
plt.close(fig)
print("bimodal_figure1.png done")

# =================== doc 61: the horizon-declaration letters ================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the per-seed churn F14 -> F30
ax = axes[0, 0]
hl2 = hz["classification"]["HL2_burnfloor"]["per_seed"]
for r in hl2:
    if r["seed"] in TAIL:
        c = "#c0392b"
    elif r["seed"] in MARG:
        c = "#e67e22"
    else:
        c = "#2471a3"
    ax.annotate("", xy=(1, r["F30"]), xytext=(0, r["F14"]),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
    ax.plot([0], [r["F14"]], ".", color=c, ms=7)
    ax.plot([1], [r["F30"]], "o", color=c, ms=7)
ax.axhline(BURN, color="black", ls="--", lw=1.0)
ax.annotate("the burn floor", (-0.42, -0.026), fontsize=8)
ax.set_xticks([0, 1], ["F_late at r11-14\n(doc 55's window)",
                       "F_late at r27-30\n(double horizon)"])
ax.set_ylabel("per-seed F_late")
ax.set_title("(a) The level holds, the membership churns: 5/12 below the\n"
             "floor at 14 rounds, 7/12 at 30 - the mean -0.025 at both",
             fontsize=10)

# (b) the anchor's ground over 30 rounds
ax = axes[0, 1]
for arm, c, lab in [("S8_CAP30_NOCH", "#7f8c8d",
                     "the disarmed anchor (ages: CONSUMED to ON-COURSE)"),
                    ("S8_CAP30", "#c0392b",
                     "the armed world (holds BURNED)")]:
    ser = mean_series(hz, arm, "comp_F")
    ax.plot([r for r, _ in ser], [v for _, v in ser], "-", color=c,
            lw=1.8, label=lab)
ax.axvline(14, color="#4a4a4a", ls=":", lw=1.2)
ax.annotate("doc 55's horizon", (14.3, 0.10), fontsize=8)
for y, t in [(OVER, "OVERFILLED"), (FILL, "ON-COURSE floor"),
             (BURN, "BURNED floor")]:
    ax.axhline(y, color="black", ls="--", lw=0.9)
    ax.annotate(t, (26.5, y + 0.004), fontsize=7.5)
ax.set_xlabel("round (30-round horizon)")
ax.set_ylabel("the fill F(r) (mean over seeds)")
ax.set_title("(b) The anchor's ground letter ages - the ground's own\n"
             "slowWalk: the differential deepens to -0.080",
             fontsize=10)
ax.legend(fontsize=8, loc="center right")

# (c) the letters' clocks: the declaration map
ax = axes[1, 0]
letters = [
    ("the catch certificate's floor", 14, 60, "#1e8449", "stable (373/374)"),
    ("the burn floor (mean level)", 14, 30, "#1e8449", "stable (-0.025)"),
    ("the ruler's letter (defended)", 14, 30, "#1e8449", "widens (0.959-0.964)"),
    ("the bracket's slope (CAP30)", 14, 30, "#e67e22", "holds (+0.0003)"),
    ("the bracket's slope (RR_U10)", 14, 30, "#c0392b", "fails (-0.0005)"),
    ("the anchor's ground letter", 14, 30, "#c0392b", "ages (CONS->ON-C)"),
    ("the channel's arming band", 14, 40, "#c0392b", "arms on all by r40"),
    ("the band's ceiling (healthy)", 14, 33, "#c0392b", "crossed by all"),
    ("the effect bars", 14, 14, "#7f8c8d", "unmeasured past 14"),
]
for i, (name, h_cal, h_meas, c, note) in enumerate(letters):
    ax.plot([h_cal, h_meas], [i, i], "-", color=c, lw=3, alpha=0.8,
            solid_capstyle="round")
    ax.plot([h_cal], [i], "|", color="black", ms=10)
    ax.annotate(note, (h_meas + 1.0, i), fontsize=7.5, va="center")
ax.set_yticks(range(len(letters)), [n for n, *_ in letters], fontsize=7.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xticks([14, 30, 40, 60], ["14", "30", "40", "60"])
ax.set_xlabel("rounds (log scale) - the calibration mark | to the measured horizon")
ax.set_title("(c) The horizon-declaration table: every letter with its clock -\n"
             "green stable, orange row-dependent, red aging, grey unmeasured",
             fontsize=10)

# (d) the ruler at both horizons, both worlds
ax = axes[1, 1]
ax.plot([14, 30], [0.9585, 0.9644], "o-", color="#c0392b", lw=2,
        label="the defended world (CAP30)")
ax.plot([14, 30], [0.9559, 0.9515], "s--", color="#2471a3", lw=1.7,
        label="the healthy twin (doc 40)")
ax.axhline(0.93, color="black", ls="--", lw=1.0)
ax.annotate("the ruler's letter", (14.4, 0.9315), fontsize=8)
ax.set_xticks([14, 30], ["14 rounds", "30 rounds"])
ax.set_ylabel("final accuracy")
ax.set_title("(d) The ruler at double horizon: the defense widens its\n"
             "margin while the twin drifts toward the letter",
             fontsize=10)
ax.legend(fontsize=8, loc="center right")

fig.savefig(f"{D}/horizon_figure1.png", dpi=160)
plt.close(fig)
print("horizon_figure1.png done")
