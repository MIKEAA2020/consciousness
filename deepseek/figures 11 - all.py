#!/usr/bin/env python3
"""Figures for corpus docs 51-53: the fourth S4 candidate (the fifth
condition), the loop's onset (doc 34's dial closed), and the effect
bar's dose-dependence. English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

c4 = json.load(open(f"{S}/s4_candidate4_results.json"))
rows = c4["classification"]["FC_rows"]
anc = c4["classification"]["FC_anchors"]
A = c4["arms"]

FILL, OVER, BURN = 0.02, 0.08, -0.02
TWIN = 0.9559

VC = {"AWARDED-CONTAINED-UNDER-8.4": "#2e8b57",
      "REFUSED-(v)": "#7b241c",
      "REFUSED-(iii)": "#d1721a",
      "REFUSED-(i)": "#7f8c8d",
      "NO-THREAT": "#1a6faf"}
SHORT = {"C4_SP025": "SP.025", "C4_SP05": "SP05", "C4_SP10": "SP10",
         "C4_SP30": "SP30", "C4_SP50": "SP50", "C4_G10": "G10",
         "C4_G30": "G30", "C4_CAP30": "CAP30", "C4_CAP10": "CAP10",
         "C4_HALF10": "HALF10", "C4_HALF30": "HALF30"}

# =================== doc 51: the fourth candidate ==========================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the two-axis verdict plane: armed accuracy vs the ground
ax = axes[0, 0]
ax.axvspan(-0.075, BURN, color="#f5dcd4", zorder=0)
ax.axvline(BURN, color="#7b241c", ls="--", lw=1.0)
ax.axvline(FILL, color="#2e8b57", ls="--", lw=1.0)
ax.axhline(0.93, color="black", ls="--", lw=1.0)
ax.axhline(TWIN, color="gray", ls=":", lw=1.0)
for arm, r in rows.items():
    ax.scatter(r["ground_F_late"], r["acc_final"], s=64,
               color=VC[r["verdict"]], zorder=3, edgecolor="white",
               lw=0.8)
    ax.annotate(SHORT[arm], (r["ground_F_late"], r["acc_final"]),
                textcoords="offset points", xytext=(6, 5), fontsize=7.5)
ax.annotate("the ruler's letter 93%", (0.021, 0.9305), fontsize=7.5)
ax.annotate("the twin's course", (0.021, 0.9575), fontsize=7.5,
            color="gray")
ax.annotate("BURNED", (-0.073, 0.997), fontsize=8, color="#7b241c")
ax.annotate("CAP30: 4-cond AWARDED,\nground BURNED -> REFUSED-(v)",
            (-0.062, 0.872), fontsize=8, color="#7b241c")
ax.set_xlim(-0.075, 0.10)
ax.set_ylim(0.86, 1.005)
ax.set_xlabel("the ground: armed F_late (doc 47's band)")
ax.set_ylabel("the ruler: armed final clean accuracy")
ax.set_title("(a) The two-axis verdict plane: four conditions hold in\n"
             "the burned column and the award does not", fontsize=10)

# (b) the ground-delta: the defense's composition price
ax = axes[0, 1]
order = ["C4_SP025", "C4_SP05", "C4_SP10", "C4_SP30", "C4_SP50",
         "C4_G10", "C4_G30", "C4_CAP10", "C4_HALF10", "C4_CAP30",
         "C4_HALF30"]
ys = np.arange(len(order))[::-1]
for y, arm in zip(ys, order):
    d = rows[arm]["ground_delta"]
    c = "#c0392b" if d < -0.02 else (
        "#d1721a" if d < 0 else "#2e8b57")
    ax.barh(y, d, height=0.62, color=c, alpha=0.85)
    ax.annotate(f"{rows[arm]['anchor_ground_verdict'][:4]}>"
                f"{rows[arm]['ground_verdict'][:4]}",
                (d, y), textcoords="offset points",
                xytext=(6 if d < 0 else -6, -3),
                ha="left" if d < 0 else "right", fontsize=6.8)
ax.set_yticks(ys)
ax.set_yticklabels([SHORT[a] for a in order], fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(-0.02, color="#7b241c", ls="--", lw=0.9)
ax.set_xlabel("ground-delta: armed F_late - anchor F_late")
ax.set_title("(b) The defense's composition price: heals the disarmed\n"
             "burns (green), converts the overfill, pays its own (red)",
             fontsize=10)

# (c) CAP30's per-seed ground: the split under the award
ax = axes[1, 0]
gr = anc["C4_CAP30_NOCH"]["ground"]
armed_f = rows["C4_CAP30"]["ground_F_late"]
seeds = np.arange(6)
ax.axhspan(OVER, 0.12, color="#ededed", zorder=0)
ax.axhspan(BURN, -0.075, color="#f5dcd4", zorder=0)
for f in [OVER, FILL, BURN]:
    ax.axhline(f, color="black", ls="--", lw=0.9)
# per-seed armed fills straight from the runs
fl_armed = []
for rr in A["C4_CAP30"]["runs"]:
    n = len(rr["traj"])
    late = [t["comp_F"] for t in rr["traj"]
            if t["comp_F"] is not None and t["round"] >= n - 3]
    fl_armed.append(np.mean(late))
ax.scatter(seeds - 0.12, gr["per_seed_F_late"], s=52, marker="o",
           facecolors="none", edgecolors="#7d3c98",
           label="anchor (disarmed mix)", zorder=3)
ax.scatter(seeds + 0.12, fl_armed, s=52, color="#7b241c",
           label="armed (the defended mix)", zorder=3)
ax.axhline(armed_f, color="#7b241c", ls=":", lw=1.2)
ax.annotate(f"armed mean {armed_f:+.4f} (BURNED)", (0.02, armed_f - 0.011),
            fontsize=8, color="#7b241c")
ax.annotate(f"anchor mean {gr['F_late_mean']:+.4f} (CONSUMED)",
            (0.02, gr["F_late_mean"] + 0.006), fontsize=8,
            color="#7d3c98")
ax.axhline(gr["F_late_mean"], color="#7d3c98", ls=":", lw=1.2)
ax.set_xticks(seeds)
ax.set_xticklabels([f"seed {s}" for s in [600, 601, 602, 603, 604, 605]],
                   fontsize=8)
ax.set_ylim(-0.075, 0.12)
ax.set_ylabel("F_late per seed")
ax.set_title("(c) CAP30: the ruler held at 95.7%, the ground split\n"
             "three and three around the burn floor, the mean below it",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")

# (d) the verdict map: verdict x ground verdict
ax = axes[1, 1]
gx = ["BURNED", "CONSUMED", "ON-COURSE", "OVERFILLED"]
vy = ["NO-THREAT", "REFUSED-(i)", "REFUSED-(iii)", "REFUSED-(v)",
      "AWARDED"]
for arm, r in rows.items():
    xi = gx.index(r["ground_verdict"])
    yi = vy.index(
        "AWARDED" if r["verdict"].startswith("AWARDED") else r["verdict"])
    ax.scatter(xi, yi, s=90, color=VC[r["verdict"]], zorder=3,
               edgecolor="white", lw=0.8)
    ax.annotate(SHORT[arm], (xi, yi), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5)
for arm, r in anc.items():
    if arm.endswith("_NOCH") and arm.replace("_NOCH", "") in SHORT:
        xi = gx.index(r["ground"]["verdict_mean"])
        ax.scatter(xi, -0.7, s=34, marker="s",
                   facecolors="none", edgecolor="#555555", zorder=3)
ax.set_xticks(range(4))
ax.set_xticklabels(gx, fontsize=8.5)
ax.set_yticks(range(5))
ax.set_yticklabels(vy, fontsize=8.5)
ax.set_ylim(-1.4, 4.7)
ax.scatter([], [], s=34, marker="s", facecolors="none",
           edgecolor="#555555", label="anchors' grounds (bottom row)")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlabel("the ground's verdict (armed worlds; squares: anchors)")
ax.set_ylabel("the verdict under five conditions")
ax.set_title("(d) The table's structure: no award over burned ground;\n"
             "no-threat rows carry both grounds", fontsize=10)
fig.suptitle("The fourth S4 candidate (doc 51): the fifth condition "
             "- containment is a property of the world", fontsize=11)
fig.savefig(f"{D}/s4cand4_figure1.png", dpi=160)
plt.close(fig)
print("wrote s4cand4_figure1.png")

# =================== doc 52: the loop's onset ==============================
oc = {float(k): v for k, v in
      c4["classification"]["FC6_onset"]["onset_curve"].items()}
doses = sorted(oc.keys())
anch = [oc[d]["anchor_acc"] for d in doses]
arm_r = {0.025: "C4_SP025", 0.05: "C4_SP05", 0.10: "C4_SP10",
         0.30: "C4_SP30", 0.50: "C4_SP50"}

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the onset curve with the bracket
ax = axes[0, 0]
dx = [d for d in doses if d > 0]
ax.plot(dx, [oc[d]["anchor_acc"] for d in dx], "o-", color="#c0392b",
        lw=1.8, ms=7, label="the disarmed anchor (the loop)")
ax.plot([d for d in dx if d in arm_r],
        [rows[arm_r[d]]["acc_final"] for d in dx if d in arm_r], "s-",
        color="#2e8b57", lw=1.8, ms=7, label="the armed world")
seed_pts = {0.0125: anc["C4_SP0125_NOCH"]["per_seed_acc"],
            0.025: anc["C4_SP025_NOCH"]["per_seed_acc"],
            0.05: anc["C4_SP05_NOCH"]["per_seed_acc"],
            0.10: anc["C4_SP10_NOCH"]["per_seed_acc"],
            0.30: anc["C4_SP30_NOCH"]["per_seed_acc"],
            0.50: anc["C4_SP50_NOCH"]["per_seed_acc"]}
for d, ps in seed_pts.items():
    ax.scatter([d] * len(ps), ps, s=14, color="#c0392b", alpha=0.55,
               zorder=2)
ax.axhline(0.93, color="black", ls="--", lw=1.0)
ax.axhline(0.60, color="black", ls=":", lw=1.0)
ax.axhline(TWIN, color="gray", ls=":", lw=0.9)
ax.axvspan(0.0125, 0.025, color="#f7e6b2", alpha=0.8, zorder=0)
ax.annotate("the bracket (0.0125, 0.025]\ninterpolated ~0.0179\n"
            "(one label in 56)", (0.0165, 0.42), fontsize=8,
            ha="center")
ax.annotate("93% - the ruler's letter", (0.32, 0.933), fontsize=7.5)
ax.annotate("60% - the LOOP letter", (0.32, 0.603), fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("the committed-label flip dose (log scale)")
ax.set_ylabel("final clean accuracy (14 rounds)")
ax.set_title("(a) The onset curve: the loop's threshold bracketed at\n"
             "(0.0125, 0.025]; the armed grid restores to the twin",
             fontsize=10)
ax.legend(fontsize=8, loc="lower left")

# (b) the ground's onset: the anchors' fill by dose
ax = axes[0, 1]
gdx = [d for d in doses if d > 0]
gf = [oc[d]["ground"]["F_late_mean"] for d in gdx]
ax.plot(gdx, gf, "o-", color="#1a6faf", lw=1.8, ms=7)
for d in gdx:
    fl = []
    armn = {0.0125: "C4_SP0125_NOCH", 0.025: "C4_SP025_NOCH",
            0.05: "C4_SP05_NOCH", 0.10: "C4_SP10_NOCH",
            0.30: "C4_SP30_NOCH", 0.50: "C4_SP50_NOCH"}[d]
    for rr in A[armn]["runs"]:
        n = len(rr["traj"])
        late = [t["comp_F"] for t in rr["traj"]
                if t["comp_F"] is not None and t["round"] >= n - 3]
        fl.append(np.mean(late))
    ax.scatter([d] * len(fl), fl, s=14, color="#1a6faf", alpha=0.55)
ax.axhline(OVER, color="black", ls="--", lw=1.0)
ax.axhline(FILL, color="black", ls="--", lw=1.0)
ax.axhline(BURN, color="black", ls="--", lw=1.0)
ax.axvspan(0.025, 0.05, color="#ededed", zorder=0)
ax.annotate("ON-COURSE > .02", (0.2, 0.024), fontsize=7.5)
ax.annotate("OVERFILLED > .08", (0.2, 0.084), fontsize=7.5)
ax.annotate("the ground's onset:\n(0.025, 0.05]", (0.033, 0.052),
            fontsize=8, ha="center")
ax.set_xscale("log")
ax.set_xlabel("the flip dose (log scale)")
ax.set_ylabel("the anchor's F_late (the un-defended ground)")
ax.set_title("(b) The territory's runaway begins one bracket above\n"
             "the ruler's degradation: ground onset (0.025, 0.05]",
             fontsize=10)

# (c) the three onsets on one axis
ax = axes[1, 0]
bands = [("the ruler's onset\n(93% letter)", 0.0125, 0.025, "#c0392b"),
         ("the ground's onset\n(overfill verdict)", 0.025, 0.05,
          "#1a6faf"),
         ("the shape's onset\n(60% LOOP letter)", 0.10, 0.30, "#7d3c98")]
for i, (lab, lo, hi, c) in enumerate(bands):
    y = 2 - i
    ax.hlines(y, lo, hi, color=c, lw=7, alpha=0.75)
    ax.annotate(lab, ((lo + hi) / 2, y + 0.28), fontsize=8.5,
                ha="center", color=c)
    ax.annotate(f"({lo}, {hi}]", ((lo + hi) / 2, y - 0.42), fontsize=8,
                ha="center", color=c)
for d in [0.0125, 0.025, 0.05, 0.10, 0.30, 0.50]:
    ax.axvline(d, color="gray", lw=0.5, alpha=0.6)
ax.annotate("~0.018", (0.0179, 2.14), fontsize=8, color="#c0392b",
            ha="center")
ax.annotate("~0.11", (0.11, 0.14), fontsize=8, color="#7d3c98",
            ha="center")
ax.set_xscale("log")
ax.set_xlim(0.010, 0.62)
ax.set_ylim(-0.7, 3.0)
ax.set_yticks([])
ax.set_xlabel("the flip dose (log scale)")
ax.set_title("(c) The loop announces itself in the ruler before the\n"
             "territory, and in the territory before the shape",
             fontsize=10)

# (d) the local slopes: the curve's curvature
ax = axes[1, 1]
dd = [0.00625, 0.01875, 0.0375, 0.075, 0.20, 0.40]
sl = [1.60, 1.10, 3.03, 4.51, 2.17, 0.21]
ax.bar(range(len(sl)), sl, color=["#d1721a", "#d1721a", "#c0392b",
                                  "#c0392b", "#7d3c98", "#7d3c98"],
       alpha=0.85)
ax.set_xticks(range(len(sl)))
ax.set_xticklabels(["0-.0125", ".0125-.025", ".025-.05", ".05-.10",
                    ".10-.30", ".30-.50"], fontsize=8)
for i, v in enumerate(sl):
    ax.annotate(f"{v:.1f}", (i, v + 0.08), ha="center", fontsize=8)
ax.set_ylabel("accuracy fall per 0.01 of dose (points)")
ax.set_title("(d) The curve's bottom is shallow (1-2 pts per 0.01, seed\n"
             "noise at N=6), peaking mid-grid, compressing at the floor",
             fontsize=10)
fig.suptitle("The loop's onset (doc 52): doc 34's dial closed at one "
             "label in fifty-six", fontsize=11)
fig.savefig(f"{D}/onset_figure1.png", dpi=160)
plt.close(fig)
print("wrote onset_figure1.png")

# =================== doc 53: the effect bar ================================
R = rows
mx = {a: R[a]["max_effect"] for a in R}
eff = {a: R[a]["effect"] for a in R}
trows = ["C4_SP025", "C4_SP05", "C4_HALF30", "C4_SP10", "C4_CAP30",
         "C4_SP30", "C4_SP50"]
dose = {"C4_SP025": 0.025, "C4_SP05": 0.05, "C4_HALF30": 0.30,
        "C4_SP10": 0.10, "C4_CAP30": 0.30, "C4_SP30": 0.30,
        "C4_SP50": 0.50}
beta = 0.30 / mx["C4_SP30"]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the bar's curves vs the max-effect envelope
ax = axes[0, 0]
# envelope from the stream anchors + the two mix anchors
env_pts = {0.025: mx["C4_SP025"], 0.05: mx["C4_SP05"],
           0.10: mx["C4_SP10"], 0.30: max(mx["C4_SP30"], mx["C4_CAP30"],
                                          mx["C4_HALF30"]),
           0.50: mx["C4_SP50"]}
xs = sorted(env_pts)
ax.plot(xs, [env_pts[x] for x in xs], "-", color="#1a6faf", lw=1.8,
        label="the threat's price (twin - anchor)")
ax.axhline(0.30, color="#7b241c", ls="--", lw=1.6,
           label="the inherited bar: 0.30 absolute")
ax.plot(xs, [beta * env_pts[x] for x in xs], "--", color="#2e8b57",
        lw=1.6, label=f"the re-based bar: {beta:.4f} x price")
ax.fill_between(xs, [env_pts[x] for x in xs], 0.86,
                where=[env_pts[x] < 0.30 for x in xs],
                color="#f5dcd4", zorder=0,
                label="the impossibility region (bar > threat)")
for a in trows:
    ax.scatter(dose[a], eff[a], s=64, zorder=4,
               color=VC[R[a]["verdict"]], edgecolor="white", lw=0.8)
    ax.annotate(SHORT[a], (dose[a], eff[a]),
                textcoords="offset points", xytext=(6, 5), fontsize=7.5)
ax.set_xscale("log")
ax.set_ylim(0, 0.88)
ax.set_xlabel("dose / threat row (log scale)")
ax.set_ylabel("effect (clean-accuracy points)")
ax.set_title("(a) The impossibility structure: at 0.025 the bar demands\n"
             "30 points of a 3.4-point threat; the refusals are unit errors",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")

# (b) bar/max ratio
ax = axes[0, 1]
labs, rats = [], []
for a in ["C4_SP025", "C4_SP05", "C4_HALF30", "C4_SP10", "C4_CAP30",
          "C4_SP30", "C4_SP50"]:
    labs.append(SHORT[a])
    rats.append(0.30 / mx[a])
cols = ["#7b241c" if r > 1 else "#2e8b57" for r in rats]
ax.bar(range(len(rats)), rats, color=cols, alpha=0.9)
ax.axhline(1.0, color="black", ls="--", lw=1.2)
for i, r in enumerate(rats):
    ax.annotate(f"{r:.2f}x", (i, r + 0.18), ha="center", fontsize=8.5)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("the inherited bar / the threat's price")
ax.set_title("(b) Above 1.0 the row was un-awardable by construction:\n"
             "8.90x, 2.74x, 1.16x - complete restorations refused",
             fontsize=10)

# (c) effects in se units: the bar was never a noise bar
ax = axes[1, 0]
def se_of(a):
    a1 = [rr["traj"][-1]["acc"] for rr in A[a]["runs"]]
    a2 = [rr["traj"][-1]["acc"]
          for rr in A[R[a]["anchor"]]["runs"]]
    d = np.array(a1) - np.array(a2)
    return d.std(ddof=1) / np.sqrt(len(d))
ses = {a: se_of(a) for a in trows}
esu = [eff[a] / ses[a] for a in trows]
ax.bar(range(len(trows)), esu,
       color=[VC[R[a]["verdict"]] for a in trows], alpha=0.9)
ax.axhline(7.93, color="black", ls="--", lw=1.4)
ax.annotate("the inherited bar at its calibration: 7.9 se",
            (0.1, 8.6), fontsize=8)
ax.annotate("conventional significance: 3 se", (0.1, 0.8), fontsize=7.5,
            color="gray")
ax.axhline(3, color="gray", ls=":", lw=1.0)
for i, v in enumerate(esu):
    ax.annotate(f"{v:.1f}", (i, v + 0.9), ha="center", fontsize=8.5)
ax.set_xticks(range(len(trows)))
ax.set_xticklabels([SHORT[a] for a in trows], fontsize=8.5)
ax.set_ylabel("the effect in paired standard errors")
ax.set_title("(c) Every effect is real (4.5-59.8 se); the bar was always\n"
             "a materiality question, never a significance one", fontsize=10)

# (d) the re-grading: inherited -> re-based
ax = axes[1, 1]
vy2 = ["AWARDED", "REFUSED-(i)", "REFUSED-(iii)", "REFUSED-(v)"]
for i, a in enumerate(trows):
    v0 = R[a]["verdict_4cond"]
    y0 = 0 if v0.startswith("AWARDED") else vy2.index(v0)
    if R[a]["verdict"].startswith("AWARDED"):
        y1 = 0
    elif R[a]["verdict"] in ("REFUSED-(i)", "REFUSED-(iii)",
                             "REFUSED-(v)"):
        y1 = vy2.index(R[a]["verdict"])
    else:
        y1 = y0
    ax.plot([i - 0.18, i + 0.18], [y0, y1], "-", color="#555555",
            lw=1.4, zorder=2)
    ax.scatter(i - 0.18, y0, s=58, color="#d1721a", zorder=3,
               edgecolor="white", lw=0.7)
    ax.scatter(i + 0.18, y1, s=58, color="#2e8b57", zorder=3,
               edgecolor="white", lw=0.7)
    ax.annotate(SHORT[a], (i, -0.62), ha="center", fontsize=8.5)
ax.set_xticks([])
ax.set_yticks(range(4))
ax.set_yticklabels(vy2, fontsize=8.5)
ax.set_ylim(-1.0, 3.5)
ax.scatter([], [], s=58, color="#d1721a", label="inherited bar")
ax.scatter([], [], s=58, color="#2e8b57",
           label="re-based bar b(d)=0.3896 x price")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("(d) The re-grading: two impossibility refusals re-award,\n"
             "HALF30's reason migrates to the ground, nothing is lost",
             fontsize=10)
fig.suptitle("The effect bar's dose-dependence (doc 53): the same "
             "number, re-based", fontsize=11)
fig.savefig(f"{D}/effectbar_figure1.png", dpi=160)
plt.close(fig)
print("wrote effectbar_figure1.png")
