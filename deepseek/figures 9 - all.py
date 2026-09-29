#!/usr/bin/env python3
"""Figures for corpus docs 43-44: the catch-rate certificate (the division
built, the aliasing measured) and the mixed attacker (the synthesis that
does not compose, the laundering boundary, the award's mirror image).
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

cc = json.load(open(f"{S}/m3_catchcert_results.json"))
mx = json.load(open(f"{S}/m3_mixed_results.json"))

PERT = 6
FLOOR = 0.25

# ================= doc 43: the catch-rate certificate =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the certificate's readings under attack: honest vs evasive
ax = axes[0, 0]
styles = {"CC_U05": ("#2e8b57", "o"), "CC_U10": ("#217a3f", "s"),
          "CC_U30": ("#14572d", "^"), "CC_J05": ("#c0392b", "v"),
          "CC_J10": ("#922b21", "D"), "CC_J30": ("#6e1d15", "P"),
          "CC_G30": ("#d1721a", "X")}
for arm, (c, m) in styles.items():
    ys = cc["arms"][arm]["aggregate"]["c_ratio"]
    xs = [i + 1 for i, y in enumerate(ys) if y is not None]
    vs = [y for y in ys if y is not None]
    ax.plot(xs, vs, m + "-", color=c, lw=1.5, ms=5, label=arm.replace("CC_", ""))
ax.axhline(FLOOR, color="black", ls="--", lw=1.1)
ax.annotate("registered floor 0.25", (7.1, 0.30), fontsize=8)
ax.set_yscale("symlog", linthresh=0.01)
ax.axhline(0.0, color="gray", lw=0.5)
ax.set_xlabel("round")
ax.set_ylabel("certificate C (symlog)")
ax.set_ylim(-0.02, 1.6)
ax.set_title("(a) The division reads both worlds:\n"
             "honest catches 0.64-0.90, evasions 0.001-0.03", fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="upper left")

# (b) the false positive: every armed healthy round reads EVASION
ax = axes[0, 1]
twins = ["CC_T30_BASE", "CC_T30_WIDE", "CC_T30_COMP", "CC_T30_REREG"]
cols = ["#1a6faf", "#7d3c98", "#d1721a", "#4a4a4a"]
for arm, c in zip(twins, cols):
    ys = cc["arms"][arm]["aggregate"]["c_ratio"]
    xs = [i + 1 for i, y in enumerate(ys) if y is not None]
    vs = [y for y in ys if y is not None]
    armed = [cc["arms"][arm]["aggregate"]["armed"][i] >= 0.5
             for i, y in enumerate(ys) if y is not None]
    ax.plot(xs, vs, "-", color=c, lw=1.2, label=arm.replace("CC_T30_", ""))
    xs_a = [x for x, a in zip(xs, armed) if a]
    vs_a = [v for v, a in zip(vs, armed) if a]
    ax.plot(xs_a, vs_a, "o", color=c, ms=4)
ax.axhline(FLOOR, color="black", ls="--", lw=1.1)
ax.annotate("floor", (24.2, 0.27), fontsize=8)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("round (30-round dial twins)")
ax.set_ylabel("certificate C")
ax.set_ylim(-0.06, 0.35)
ax.set_title("(b) The aliasing is total: 145 of 145 armed\n"
             "healthy rounds read EVASION (dots = armed)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.annotate("the drift is 94% blind-item:\nconfident fraction 0.059",
            (15.5, 0.20), fontsize=8, color="#333333")

# (c) the elevation clause fails: healthy vs evaded cumulative elevation
ax = axes[1, 0]
groups = [("CC_T30_BASE", "#1a6faf", "twin BASE"), ("CC_T30_WIDE", "#7d3c98", "twin WIDE"),
          ("CC_T30_COMP", "#d1721a", "twin COMP"), ("CC_T30_REREG", "#4a4a4a", "twin REREG"),
          ("CC_J05", "#c0392b", "J05"), ("CC_J10", "#922b21", "J10"),
          ("CC_J30", "#6e1d15", "J30")]
for arm, c, lab in groups:
    finals = [rr["traj"][-1]["x_cum"] for rr in cc["arms"][arm]["runs"]
              if rr["traj"][-1]["x_cum"] is not None]
    ax.scatter([lab] * len(finals), finals, color=c, s=26, zorder=3)
    ax.scatter([lab], [np.mean(finals)], marker="_", s=420, color="black",
               zorder=4, lw=1.6)
ax.axhline(0.055, color="gray", ls=":", lw=0.9)
ax.annotate("the would-be separation:\ntwins below, attacks above",
            (0.35, 0.070), fontsize=7.5, color="gray")
ax.annotate("REREG's healthy reading\nreaches 0.094 - the overlap",
            (3.0, 0.098), fontsize=7.5, color="#4a4a4a")
ax.set_ylabel("final cumulative elevation x_cum")
ax.set_ylim(0.0, 0.115)
plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)
ax.set_title("(c) The elevation clause cannot rescue it:\n"
             "the REREG twin overlaps the attacks", fontsize=10)

# (d) the REREG twin: the reading runs while the channel sleeps
ax = axes[1, 1]
for arm, c, lab in [("CC_T30_BASE", "#1a6faf", "twin BASE"),
                    ("CC_T30_REREG", "#4a4a4a", "twin REREG"),
                    ("CC_J30", "#6e1d15", "J30 (the attack)")]:
    ys = cc["arms"][arm]["aggregate"]["d_recv"]
    ax.plot(range(1, len(ys) + 1), ys, "-", color=c, lw=1.6, label=lab)
    if arm == "CC_T30_REREG":
        armed_r = [i + 1 for i, v in
                   enumerate(cc["arms"][arm]["aggregate"]["armed"]) if v >= 0.5]
        ys_armed = [ys[i - 1] for i in armed_r]
        ax.plot(armed_r, ys_armed, "o", color=c, ms=5)
ax.axhline(0.03, color="gray", ls=":", lw=0.9)
ax.annotate("the 0.03 band edge", (24.5, 0.045), fontsize=7.5, color="gray")
ax.set_xlabel("round")
ax.set_ylabel("d_recv (panel vs received stream)")
ax.set_title("(d) The scheduled record's second bill: the REREG\n"
             "twin's reading runs to 0.10, unarmed (dots = armed)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(f"{D}/catchcert_figure1.png", dpi=150)
plt.close(fig)

# ================= doc 44: the mixed attacker =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the non-additivity: surviving dirt decomposition
ax = axes[0, 0]
mx2 = mx["classification"]["MX2_arithmetic"]
arms_a = ["MX_U30", "MX_J30", "MX_J10", "MX_CAP30", "MX_HALF30",
          "MX_CAP10", "MX_HALF10"]
labs = ["U30\n(pure bulk)", "J30\n(pure gated)", "J10", "CAP30\n(synthesis)",
        "HALF30", "CAP10\n(blind-heavy)", "HALF10"]
sb = [mx2[a]["surviving_bulk"] or 0 for a in arms_a]
sg = [mx2[a]["surviving_gate"] or 0 for a in arms_a]
bm = [mx2[a]["blind_mass"] or 0 for a in arms_a]
x = np.arange(len(arms_a))
ax.bar(x, sb, 0.62, color="#c0392b", label="surviving bulk dirt (the residue)")
ax.bar(x, sg, 0.62, bottom=sb, color="#4a4a4a",
       label="surviving gated fill (wholesale)")
ax.plot(x, bm, "v--", color="#1a6faf", lw=1.4, ms=6,
        label="the blind mass (the shared territory)")
ax.annotate("the synthesis leaves 0.055 alive\nvs the pure gated 0.082",
            (3.05, 0.088), fontsize=7.5, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(labs, fontsize=7.2)
ax.set_ylabel("rate over commits (late mean)")
ax.set_title("(a) The attacks do not compose: the bulk burns\n"
             "the blind spot the gated half needs", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")

# (b) the laundering map: C vs the gated share of realized dose
ax = axes[0, 1]
mx4 = mx["classification"]["MX4_laundering"]
arms_b = ["MX_U30", "MX_CAP30", "MX_HALF30", "MX_HALF10", "MX_CAP10",
          "MX_J10", "MX_J30"]
for arm in arms_b:
    row = mx2[arm]
    share = (row["gate_fill"] or 0) / max(row["total_dose"] or 1e-9, 1e-9)
    cval = mx4[arm]["final_C_mean"]
    catching = mx4[arm]["CATCHING"] >= 4
    ax.scatter(share, cval, s=90,
               color="#2e8b57" if catching else "#c0392b", zorder=3)
    dy = 0.05 if arm not in ("MX_HALF10",) else -0.07
    ax.annotate(arm.replace("MX_", ""), (share, cval + dy), fontsize=7.5,
                ha="center",
                color="#2e8b57" if catching else "#c0392b")
ax.axhline(FLOOR, color="black", ls="--", lw=1.1)
ax.annotate("the certificate's floor", (0.42, 0.285), fontsize=8)
ax.set_xlabel("gated share of the realized dose")
ax.set_ylabel("certificate C (final)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("(b) The laundering boundary: CATCHING while the gated\n"
             "share rides; EVASION only at 3:1 fill-to-bulk", fontsize=10)

# (c) the award re-grade: the four conditions per arm
ax = axes[1, 0]
mx5 = mx["classification"]["MX5_award_regrade"]
arms_c = ["MX_U30", "MX_CAP30", "MX_HALF30", "MX_CAP10", "MX_HALF10",
          "MX_J10", "MX_J30"]
conds = ["(i)_clean", "(ii)_engagement", "(iii)_registry", "(iv)_bracket"]
cond_labs = ["(i) ruler", "(ii) engage", "(iii) registry", "(iv) bracket"]
grid = np.zeros((len(arms_c), len(conds)))
for i, arm in enumerate(arms_c):
    for j, ck in enumerate(conds):
        grid[i, j] = 1.0 if mx5[arm][ck] else 0.0
ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(
    ["#f2d5d0", "#d5ecd9"]), aspect="auto", vmin=0, vmax=1)
for i, arm in enumerate(arms_c):
    for j in range(len(conds)):
        ax.text(j, i, "PASS" if grid[i, j] else "FAIL",
                ha="center", va="center", fontsize=7.5,
                color="#1e5c30" if grid[i, j] else "#8a2c20")
    aw = mx5[arm]["awarded_CONTAINED_UNDER_8_4"]
    ax.text(4.35, i, "AWARDED" if aw else "refused", ha="left",
            va="center", fontsize=8, fontweight="bold" if aw else "normal",
            color="#1e5c30" if aw else "#8a2c20")
ax.set_xticks(range(4))
ax.set_xticklabels(cond_labs, fontsize=8)
ax.set_yticks(range(len(arms_c)))
ax.set_yticklabels([a.replace("MX_", "") for a in arms_c], fontsize=8)
ax.set_xlim(-0.5, 5.6)
ax.set_title("(c) The award under the mix: a true-premise award (CAP30),\n"
             "a harmless-attack refusal (CAP10), a one-round letter (HALF10)",
             fontsize=9.5)

# (d) the world stands; the disarmed synthesis collapses
ax = axes[1, 1]
mx3 = mx["classification"]["MX3_world"]
arms_d = ["MX_TWIN", "MX_J30", "MX_CAP30", "MX_HALF30", "MX_U30",
          "MX_CAP30_NOCH"]
for arm in arms_d:
    accs = [rr["traj"][-1]["acc"] * 100 for rr in mx["arms"][arm]["runs"]]
    qbar = mx3[arm]["q_bar"] if arm in mx3 else np.mean(
        [rr["traj"][-1]["q_target"] for rr in mx["arms"][arm]["runs"]])
    ax.scatter(qbar, np.mean(accs), s=110, zorder=3,
               color="#c0392b" if "NOCH" in arm else "#2e8b57",
               marker="X" if "NOCH" in arm else "o")
    ax.annotate(arm.replace("MX_", ""), (qbar, np.mean(accs)),
                xytext=(6, -3), textcoords="offset points", fontsize=7.5)
ax.axhline(93, color="gray", ls=":", lw=0.9)
ax.annotate("the 93% ruler", (0.415, 93.8), fontsize=7.5, color="gray")
ax.set_xlabel("q-bar (the internal quantile, late mean)")
ax.set_ylabel("final clean accuracy (%)")
ax.set_ylim(35, 101)
ax.set_title("(d) Every armed world stands; the disarmed synthesis\n"
             "collapses to 40% - the loop's fuel is the bulk", fontsize=10)
fig.savefig(f"{D}/mixed_figure1.png", dpi=150)
plt.close(fig)
print("wrote catchcert_figure1.png and mixed_figure1.png")
