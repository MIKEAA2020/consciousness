#!/usr/bin/env python3
"""Figures for corpus docs 29-31: the decay arm (two-threshold map), the
joint poison (tension points), and the substrate-aware brake (policy map +
the substrate shell). English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

dec = json.load(open(f"{S}/m3_decay_results.json"))
jp = json.load(open(f"{S}/m3_jointpoison_results.json"))
sub = json.load(open(f"{S}/m3_substrate_results.json"))

ETA, RHO, EPS, DRI, TAUS = 0.3, 0.5, 0.05, 0.02, 0.9
BAND = TAUS - EPS          # 0.85


def simulate(q_bar, drift, tau_start, rounds=16, pert=6):
    tau = tau_start
    for _ in range(pert, rounds + 1):
        if drift:
            tau -= DRI
        if abs(tau - TAUS) > EPS:
            tau += RHO * (TAUS - tau)
        tau = (1 - ETA) * tau + ETA * q_bar
    return tau


# ================= doc 29: the decay arm =================
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=True)

# (a) the two-threshold map
ax = axes[0]
qs = np.linspace(0.70, 1.00, 241)
for drift, col, lab in [(False, "#1a6faf", "twin (rot only)"),
                        (True, "#c0392b", "drift + rot")]:
    fin = [simulate(q, drift, 0.954) for q in qs]
    ax.plot(qs, fin, "-", color=col, lw=1.6, label=lab, zorder=3)
lv_pts = []
for lv in [0.93, 0.87, 0.80, 0.72]:
    tag = "DEC%d" % round(lv * 100)
    for kind, col, mk in [("TWIN", "#1a6faf", "o"), ("DRIFT", "#c0392b", "s")]:
        c = dec["classification"][f"{tag}_{kind}"]
        ax.scatter([c["q_bar_realized"]], [c["final_tau"]], marker=mk, s=46,
                   c=col, zorder=6, edgecolors="k", linewidths=0.5)
        lv_pts.append((tag, kind, c))
for x, col, lab in [(0.8967, "#7d3c98", "doc-26 threshold 0.897"),
                    (0.8150, "#d1721a", "2nd threshold 0.815"),
                    (0.7917, "#d1721a", "")]:
    ax.axvline(x, color=col, ls=":", lw=1.1, zorder=1)
ax.axhline(BAND, color="k", ls="--", lw=0.9, zorder=1)
ax.axhline(TAUS, color="#888", ls="-", lw=0.7, zorder=1)
ax.annotate("band edge 0.85", (0.703, 0.852), fontsize=7.5, color="k")
ax.annotate("0.897", (0.899, 0.735), fontsize=7.5, color="#7d3c98",
            rotation=90)
ax.annotate("0.815", (0.817, 0.735), fontsize=7.5, color="#d1721a",
            rotation=90)
ax.annotate("silent", (0.945, 0.806), fontsize=8, color="#555")
ax.annotate("visible,\nheld", (0.845, 0.806), fontsize=8, color="#555")
ax.annotate("broken", (0.735, 0.806), fontsize=8, color="#555")
ax.set_xlabel("realized evaluator level $\\bar q$")
ax.set_ylabel("final $\\tau_{sys}$ (round 16)")
ax.set_title("(a) The two-threshold map", fontsize=10)
ax.set_xlim(0.70, 1.0)
ax.set_ylim(0.72, 0.99)
ax.legend(fontsize=7.5, loc="lower right")

# (b) trajectories
ax = axes[1]
cols = {"93": "#2e8b57", "87": "#1a6faf", "80": "#d1721a", "72": "#7d3c98"}
for lv in [93, 87, 80, 72]:
    a_t = dec["arms"][f"DEC{lv}_TWIN"]["aggregate"]["tau_sys"]
    a_d = dec["arms"][f"DEC{lv}_DRIFT"]["aggregate"]["tau_sys"]
    xs = range(1, len(a_t) + 1)
    ax.plot(xs, a_t, "-", color=cols[str(lv)], lw=1.3,
            label=f"twin q=0.{lv}")
    ax.plot(xs, a_d, "--", color=cols[str(lv)], lw=1.3,
            label=f"drift q=0.{lv}")
hf = dec["arms"]["HEALTHY_FLAT"]["aggregate"]["tau_sys"]
ax.plot(range(1, len(hf) + 1), hf, "-", color="#c9c9c9", lw=1.2,
        label="healthy flat")
ax.axhline(BAND, color="k", ls="--", lw=0.9)
ax.axvline(6, color="#c0392b", ls=":", lw=1.0)
ax.annotate("tempering + drift\nfrom round 6", (6.15, 0.77), fontsize=7.5,
            color="#c0392b")
ax.set_xlabel("round")
ax.set_ylabel("$\\tau_{sys}$")
ax.set_title("(b) Norm trajectories by decay level", fontsize=10)
ax.legend(fontsize=6.6, ncol=2, loc="lower left")

# (c) repair engagement + attack marginal
ax = axes[2]
lvls = [0.93, 0.87, 0.80, 0.72]
xs = np.arange(4)
rep_t = [dec["classification"]["DEC%d_TWIN" % round(l * 100)][
    "repair_rounds_post"] for l in lvls]
rep_d = [dec["classification"]["DEC%d_DRIFT" % round(l * 100)][
    "repair_rounds_post"] for l in lvls]
ax.bar(xs - 0.18, rep_t, 0.34, color="#1a6faf", label="twin (rot only)")
ax.bar(xs + 0.18, rep_d, 0.34, color="#c0392b", label="drift + rot")
ax.set_xticks(xs)
ax.set_xticklabels([f"q={l:.2f}" for l in lvls])
ax.set_ylabel("repair rounds fired (of 11)")
ax.set_ylim(0, 11.5)
ax2 = ax.twinx()
marg = [dec["classification"]["DEC%d_DRIFT" % round(l * 100)][
    "attack_marginal_pp"] for l in lvls]
ax2.plot(xs, marg, "k*--", ms=9, lw=1.0, label="attack marginal (pp)")
ax2.set_ylabel("attack marginal $\\tau_{twin}-\\tau_{drift}$ (pp)")
ax2.set_ylim(-5.5, 0.5)
ax.set_title("(c) Visible containment, and the\nattack's shrinking marginal",
             fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")

fig.savefig(f"{D}/m3decay_figure1.png", dpi=170)
plt.close(fig)
print("m3decay_figure1.png done")

# ================= doc 30: the joint poison =================
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=True)

# (a) tau coordinate
ax = axes[0]
for arm, col, mk, lab in [
        ("TWIN", "#c9c9c9", "-", "TWIN (kappa frozen)"),
        ("KAP_TWIN", "#888888", "-", "KAP_TWIN"),
        ("JP_TAU", "#1a6faf", "-", "JP_TAU (tau* poisoned)"),
        ("JP_TAU_KSTATE", "#7d3c98", "-", "JP_TAU_KSTATE"),
        ("JP_JOINT", "#c0392b", "-", "JP_JOINT (both poisoned)"),
        ("JP_JOINT_NOREP", "#2e8b57", "--", "JP_JOINT_NOREP")]:
    a = jp["arms"][arm]["aggregate"]["tau_sys"]
    ax.plot(range(1, 11), a, mk, color=col, lw=1.5, label=lab)
cls = jp["classification"]
ax.axhline(cls["JP_JOINT"]["tension_point_tau"], color="#c0392b", ls=":",
           lw=1.1)
ax.annotate("tension point 0.835", (1.1, 0.839), fontsize=7.5,
            color="#c0392b")
ax.axhline(0.70, color="#c0392b", ls="--", lw=0.7)
ax.annotate("poison $\\tau^*$=0.70", (1.1, 0.707), fontsize=7.5,
            color="#c0392b")
ax.axvline(6, color="#c0392b", ls=":", lw=1.0)
ax.set_xlabel("round")
ax.set_ylabel("$\\tau_{sys}$")
ax.set_title("(a) The tau coordinate: contained at\nthe tension point",
             fontsize=10)
ax.legend(fontsize=6.8, loc="lower right")

# (b) kappa coordinate
ax = axes[1]
for arm, col, mk, lab in [
        ("KAP_TWIN", "#888888", "-", "KAP_TWIN"),
        ("JP_KAP", "#d1721a", "-", "JP_KAP (kappa* poisoned)"),
        ("JP_JOINT", "#c0392b", "-", "JP_JOINT"),
        ("JP_JOINT_NOREP", "#2e8b57", "--", "JP_JOINT_NOREP")]:
    a = jp["arms"][arm]["aggregate"]["kappa_sys"]
    ax.plot(range(1, 11), a, mk, color=col, lw=1.5, label=lab)
ax.axhline(cls["JP_JOINT"]["tension_point_kappa"], color="#c0392b", ls=":",
           lw=1.1)
ax.annotate("tension point 0.863", (1.1, 0.866), fontsize=7.5,
            color="#c0392b")
ax.axhline(0.75, color="#d1721a", ls="--", lw=0.7)
ax.annotate("poison $\\kappa^*$=0.75", (1.1, 0.757), fontsize=7.5,
            color="#d1721a")
ax.axvline(6, color="#c0392b", ls=":", lw=1.0)
ax.set_xlabel("round")
ax.set_ylabel("$\\kappa_{sys}$")
ax.set_title("(b) The kappa coordinate: the second\ntension, independently",
             fontsize=10)
ax.legend(fontsize=7.2, loc="lower right")

# (c) the landing map + the escape
ax = axes[2]
for arm, col, mk, lab in [
        ("KAP_TWIN", "#888888", "o", "KAP_TWIN (live course)"),
        ("JP_TAU", "#1a6faf", "o", "JP_TAU"),
        ("JP_KAP", "#d1721a", "o", "JP_KAP"),
        ("JP_JOINT", "#c0392b", "s", "JP_JOINT"),
        ("JP_JOINT_NOREP", "#2e8b57", "D", "JP_JOINT_NOREP")]:
    a = jp["arms"][arm]["aggregate"]
    ax.scatter([a["tau_sys"][-1]], [a["kappa_sys"][-1]], marker=mk, s=60,
               c=col, zorder=5, edgecolors="k", linewidths=0.5, label=lab)
ax.scatter([cls["JP_JOINT"]["tension_point_tau"]],
           [cls["JP_JOINT"]["tension_point_kappa"]], marker="x", s=90,
           c="#c0392b", lw=1.6, zorder=6)
ax.annotate("predicted joint tension\n(0.835, 0.863)", (0.838, 0.872),
            fontsize=7.5, color="#c0392b")
ax.scatter([0.70], [0.75], marker="v", s=60, c="#c0392b", zorder=6)
ax.annotate("the poisons\n(0.70, 0.75)", (0.705, 0.735), fontsize=7.5,
            color="#c0392b")
ax.annotate("", xy=(jp["arms"]["JP_JOINT_NOREP"]["aggregate"]["tau_sys"][-1],
                    jp["arms"]["JP_JOINT_NOREP"]["aggregate"]["kappa_sys"][-1]),
            xytext=(0.836, 0.864),
            arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.4))
ax.annotate("disarmament:\nescape to live", (0.872, 0.908), fontsize=7.5,
            color="#2e8b57")
ax.set_xlabel("final $\\tau_{sys}$")
ax.set_ylabel("final $\\kappa_{sys}$")
ax.set_title("(c) The landing map: factorized tension,\nrepair-carried escape",
             fontsize=10)
ax.set_xlim(0.68, 0.99)
ax.set_ylim(0.72, 1.01)
ax.legend(fontsize=7.0, loc="lower right")

fig.savefig(f"{D}/m3jp_figure1.png", dpi=170)
plt.close(fig)
print("m3jp_figure1.png done")

# ================= doc 31: the substrate-aware brake =================
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=True)

# (a) accuracy trajectories
ax = axes[0]
for arm, col, mk, lab in [
        ("S1_BRAKE", "#2e8b57", "-", "S1 brake (1-of-4)"),
        ("S2_BRAKE", "#c0392b", "-", "S2 brake (2-of-4)"),
        ("S2_NOBRAKE", "#1a6faf", "--", "S2 no-brake"),
        ("S2_CUTOFF", "#1a6faf", "-", "S2 cutoff (stand down)"),
        ("S2_SUBSTRATE", "#7d3c98", "-", "S2 probe (never fires)"),
        ("S3_CUTOFF", "#d1721a", "-", "S3 cutoff (3-of-4)"),
        ("TRUNK_HALF", "#8B4513", "-", "TRUNK half (machinery)"),
        ("TRUNK_HALF_NOBRAKE", "#8B4513", "--", "TRUNK half (no brake)")]:
    a = sub["arms"][arm]["aggregate"]["acc"]
    ax.plot(range(1, 11), a, mk, color=col, lw=1.4, label=lab)
fz = np.mean([f["acc_after"] for f in sub["frozen_trunk_kill"]])
ax.axhline(fz, color="#8B4513", ls=":", lw=1.1)
ax.annotate("raw trunk damage 84.5%", (1.1, fz - 0.012), fontsize=7.5,
            color="#8B4513")
ax.axvline(6, color="#c0392b", ls=":", lw=1.0)
ax.set_xlabel("round")
ax.set_ylabel("argmax accuracy (test)")
ax.set_title("(a) The multiplicity map: keep-training\ninverts at 2-of-4, stasis rules",
             fontsize=10)
ax.legend(fontsize=6.5, loc="lower left")

# (b) the substrate shell, live
ax = axes[1]
a = sub["arms"]["TRUNK_HALF"]["aggregate"]
xs = range(1, 11)
ax.plot(xs, a["acc"], "-o", color="#c0392b", lw=1.6, ms=3.5,
        label="accuracy (external ruler)")
ax.plot(xs, a["tau_sys"], "-", color="#2e8b57", lw=1.4,
        label="$\\tau_{sys}$ (norm: 'recovered')")
ax.plot(xs, a["core_pair"], "-", color="#1a6faf", lw=1.4,
        label="core pair agreement")
ax.plot(xs, [1 - x for x in a["flagged_frac"]], "-", color="#7d3c98",
        lw=1.4, label="1 - flag fraction")
ax.axvline(6, color="#c0392b", ls=":", lw=1.0)
ax.annotate("substrate damage:\ninternal panel reads recovery,\naccuracy is 11pp down",
            (6.15, 0.42), fontsize=7.5, color="#c0392b")
ax.set_xlabel("round")
ax.set_ylabel("signal level")
ax.set_title("(b) The substrate shell (F5): every\ninternal signal reads 'recovered'",
             fontsize=10)
ax.legend(fontsize=6.8, loc="center left")

# (c) the two-ruler price
ax = axes[2]
arms_c = ["S1_BRAKE", "S2_BRAKE", "S2_NOBRAKE", "S2_CUTOFF", "S3_CUTOFF",
          "TRUNK_HALF"]
labels = ["1-of-4\nbrake", "2-of-4\nbrake", "2-of-4\nno brake",
          "2-of-4\ncutoff", "3-of-4\ncutoff", "trunk\nmachinery"]
accs = [sub["classification"][a]["acc_final"] * 100 for a in arms_c]
taus = [sub["classification"][a]["final_tau_sys"] for a in arms_c]
xs = np.arange(len(arms_c))
bars = ax.bar(xs, accs, 0.58, color=["#2e8b57", "#c0392b", "#1a6faf",
                                     "#1a6faf", "#d1721a", "#8B4513"])
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=7.2)
ax.set_ylabel("final argmax accuracy (%)")
ax.set_ylim(80, 100)
ax2 = ax.twinx()
ax2.plot(xs, taus, "k*--", ms=10, lw=1.1, label="final $\\tau_{sys}$")
ax2.axhline(0.85, color="#c0392b", ls=":", lw=1.0)
ax2.annotate("band edge", (4.35, 0.853), fontsize=7, color="#c0392b")
ax2.set_ylabel("final $\\tau_{sys}$ (norm coordinate)")
ax2.set_ylim(0.55, 1.0)
for x, acc in zip(xs, accs):
    ax.annotate(f"{acc:.1f}", (x, acc + 0.25), ha="center", fontsize=7)
ax.set_title("(c) The two-ruler price: stasis preserves\nthe argmax and breaks the norm",
             fontsize=10)
ax2.legend(fontsize=7.5, loc="lower left")

fig.savefig(f"{D}/m3substrate_figure1.png", dpi=170)
plt.close(fig)
print("m3substrate_figure1.png done")
