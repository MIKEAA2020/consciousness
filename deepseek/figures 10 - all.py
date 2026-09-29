#!/usr/bin/env python3
"""Figures for corpus docs 47-48: the composition-side certificate (the
territory read, the aliasing broken) and the third S4 candidate (the
conditional (iii), the NO-THREAT cell, the award's own false premise).
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

cm = json.load(open(f"{S}/m3_compcert_results.json"))
c3 = json.load(open(f"{S}/s4_candidate3_results.json"))

PERT = 6
FILL, OVER, BURN = 0.02, 0.08, -0.02
FLOOR_C = 0.25

# ================= doc 47: the composition-side certificate =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the fill trajectories: the five signatures on one canvas
ax = axes[0, 0]
series = [("CM_T30_BASE", "#1a6faf", "healthy BASE (30r)"),
          ("CM_J30", "#c0392b", "gated J30 (consumed)"),
          ("CM_G30", "#d1721a", "arbitrary G30 (burned)"),
          ("CM_CAP30", "#7d3c98", "mixed CAP30 (burned)"),
          ("CM_T30_REREG", "#4a4a4a", "REREG drift (overfilled)")]
for arm, c, lab in series:
    ys = cm["arms"][arm]["aggregate"]["comp_F"]
    xs = [i + 1 for i, y in enumerate(ys) if y is not None]
    vs = [y for y in ys if y is not None]
    ax.plot(xs, vs, "-", color=c, lw=1.6, label=lab)
ax.axhline(OVER, color="black", ls="--", lw=1.0)
ax.axhline(FILL, color="black", ls="--", lw=1.0)
ax.axhline(BURN, color="black", ls="--", lw=1.0)
ax.axhline(0, color="gray", lw=0.5)
for y, t in [(OVER + 0.006, "OVERFILLED > .08"),
             (FILL + 0.004, "ON-COURSE > .02"),
             (BURN - 0.009, "BURNED < -.02")]:
    ax.annotate(t, (23.5, y), fontsize=7.5, color="#333333")
ax.annotate("CONSUMED between", (23.5, 0.004), fontsize=7.5,
            color="#333333")
ax.set_xlabel("round (F = blind mass - registered level b*)")
ax.set_ylabel("the fill F(r)")
ax.set_title("(a) The territory's five signatures: the drift fills,\n"
             "the gated attack consumes, the burners burn, REREG runs",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")

# (b) THE PAIR: catch C vs composition F_late - the aliasing broken
ax = axes[0, 1]
pts = [
    ("CM_TWIN", "twin", "#1a6faf", "MUM"),
    ("CM_T30_BASE", "BASE", "#4d9fe0", "EVASION-fp"),
    ("CM_T30_WIDE", "WIDE", "#4d9fe0", "MUM"),
    ("CM_T30_COMP", "COMP", "#4d9fe0", "MUM"),
    ("CM_T30_REREG", "REREG", "#1a6faf", "MUM"),
    ("CM_U05", "U05", "#2e8b57", "CATCHING"),
    ("CM_U10", "U10", "#2e8b57", "CATCHING"),
    ("CM_U30", "U30", "#14572d", "CATCHING"),
    ("CM_J05", "J05", "#c0392b", "EVASION"),
    ("CM_J10", "J10", "#c0392b", "EVASION"),
    ("CM_J30", "J30", "#c0392b", "EVASION"),
    ("CM_G05", "G05", "#d1721a", "EVASION"),
    ("CM_G30", "G30", "#d1721a", "EVASION"),
    ("CM_CAP30", "CAP30", "#7d3c98", "CATCHING"),
    ("CM_HALF30", "HALF30", "#7d3c98", "CATCHING"),
    ("CM_CAP10", "CAP10", "#9b59b6", "EVASION"),
    ("CM_HALF10", "HALF10", "#7d3c98", "CATCHING"),
]
# shade the composition bands
ax.axhspan(OVER, 0.14, color="#e8e8e8", zorder=0)
ax.axhspan(BURN, FILL, color="#f7e8e4", zorder=0)
ax.axhspan(-0.07, BURN, color="#f7d6ce", zorder=0)
for arm, lab, c, cv in pts:
    fl = cm["arms"][arm]["aggregate"]["comp_F_late_mean"]
    cr = cm["arms"][arm]["aggregate"]["c_ratio"][-1]
    if cr is None:
        ax.plot(0.012, fl, "o", mfc="none", mec=c, ms=9, mew=1.6)
        ax.annotate(lab, (0.022, fl), fontsize=7.5, color=c)
    else:
        ax.plot(cr, fl, "o", color=c, ms=7)
        ax.annotate(lab, (cr + 0.012, fl), fontsize=7.5, color=c)
ax.axvline(FLOOR_C, color="black", ls="--", lw=1.0)
ax.annotate("catch floor 0.25", (0.27, 0.105), fontsize=8)
ax.annotate("MUM\n(silent)", (0.012, -0.058), fontsize=7.5,
            color="#555555")
ax.set_xlabel("the catch certificate's final C (MUM worlds open, at left)")
ax.set_ylabel("the composition reading F_late")
ax.set_xlim(-0.03, 1.02)
ax.set_ylim(-0.07, 0.14)
ax.set_title("(b) The pair breaks the aliasing: same catch verdict,\n"
             "different territories - (EVASION, ON-COURSE) vs (EVASION, CONSUMED)",
             fontsize=10)

# (c) the dose-response and the burning ladder (F_late by arm)
ax = axes[1, 0]
groups = [
    ("healthy", ["CM_TWIN", "CM_T30_BASE", "CM_T30_WIDE", "CM_T30_COMP"],
     "#1a6faf"),
    ("uniform", ["CM_U05", "CM_U10", "CM_U30"], "#2e8b57"),
    ("gated J", ["CM_J05", "CM_J10", "CM_J30"], "#c0392b"),
    ("arbitrary G", ["CM_G05", "CM_G10", "CM_G30"], "#d1721a"),
    ("mixed", ["CM_CAP30", "CM_HALF30", "CM_CAP10", "CM_HALF10"],
     "#7d3c98"),
    ("disarmed", ["CM_CAP30_NOCH", "CM_CAP10_NOCH"], "#4a4a4a"),
    ("REREG", ["CM_T30_REREG"], "#111111"),
]
xs, ys, cs, labs = [], [], [], []
x = 0
for gname, arms, c in groups:
    for a in arms:
        xs.append(x)
        ys.append(cm["arms"][a]["aggregate"]["comp_F_late_mean"])
        cs.append(c)
        labs.append(a.replace("CM_", "").replace("T30_", ""))
        x += 1
    x += 0.8
ax.bar(xs, ys, color=cs, width=0.8)
ax.axhline(OVER, color="black", ls="--", lw=1.0)
ax.axhline(FILL, color="black", ls="--", lw=1.0)
ax.axhline(BURN, color="black", ls="--", lw=1.0)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(labs, rotation=60, fontsize=7)
ax.set_ylabel("F_late (the verdict window)")
ax.set_title("(c) The band's census: the mix burns below the pure\n"
             "uniform at equal dose; the disarmed blind-heavy burns hardest",
             fontsize=10)

# (d) the always-on reading: REREG vs BASE, 30 rounds, armed rounds dotted
ax = axes[1, 1]
for arm, c, lab in [("CM_T30_BASE", "#1a6faf", "BASE (repaired drift)"),
                    ("CM_T30_REREG", "#4a4a4a", "REREG (un-repaired drift)")]:
    ys = cm["arms"][arm]["aggregate"]["blind_mass"]
    xs = list(range(1, len(ys) + 1))
    ax.plot(xs, ys, "-", color=c, lw=1.6, label=lab)
    armed = cm["arms"][arm]["aggregate"]["armed"]
    xa = [x for x, a in zip(xs, armed) if a >= 0.5]
    ya = [y for y, a in zip(ys, armed) if a >= 0.5]
    ax.plot(xa, ya, "o", color=c, ms=4)
bstar = cm["arms"]["CM_T30_BASE"]["aggregate"]["blind_mass"][4]
ax.axhline(bstar + OVER, color="black", ls="--", lw=1.0)
ax.annotate("the overfill ceiling b*+0.08", (18.5, bstar + OVER + 0.004),
            fontsize=8)
ax.axhline(bstar, color="gray", ls=":", lw=1.0)
ax.annotate("registered b*", (26.5, bstar - 0.008), fontsize=8,
            color="#555555")
ax.set_xlabel("round (30-round dial twins; dots = armed)")
ax.set_ylabel("blind mass b(r)")
ax.set_title("(d) Always-on: the catch certificate spoke on 5 of 180\n"
             "REREG seed-rounds (dots); the composition reads all 180",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")

fig.savefig(f"{D}/m3comp_figure1.png", dpi=150)
plt.close(fig)
print("wrote m3comp_figure1.png")

# ================= doc 48: the third S4 candidate =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)
TC = c3["classification"]["TC_rows"]
AN = c3["classification"]["TC_anchors"]

# (a) THE CONDITIONAL'S PLANE: anchor acc vs armed acc
ax = axes[0, 0]
ax.plot([0.1, 1.0], [0.1 + 0.30, 1.0], "-", color="#888888", lw=1.2)
ax.annotate("the effect bar: armed = anchor + 0.30", (0.44, 0.79),
            fontsize=8, color="#555555", rotation=33)
ax.axvline(0.93, color="black", ls="--", lw=1.1)
ax.annotate("threat threshold\nanchor = 0.93", (0.937, 0.33), fontsize=8)
ax.axhline(0.93, color="black", ls="--", lw=1.1)
ax.annotate("the ruler's letter 0.93", (0.13, 0.945), fontsize=8)
vc = {"AWARDED-CONTAINED-UNDER-8.4": ("#2e8b57", "o"),
      "NO-THREAT": ("#7f8c8d", "s"),
      "REFUSED-(iii)": ("#c0392b", "X"),
      "REFUSED-(i)": ("#d1721a", "v")}
for arm, r in TC.items():
    c, m = vc[r["verdict"]]
    ax.plot(r["anchor_acc"], r["acc_final"], m, color=c, ms=10)
    ax.annotate(arm.replace("C3_", ""), (r["anchor_acc"] + 0.006,
                 r["acc_final"] + 0.004), fontsize=8)
ax.plot([0.1, 1.0], [0.1, 1.0], ":", color="#aaaaaa", lw=0.8)
ax.annotate("no effect", (0.22, 0.20), fontsize=7.5, color="#999999",
            rotation=38)
ax.set_xlabel("the disarmed anchor's final clean accuracy (the attack's price)")
ax.set_ylabel("the armed world's final clean accuracy")
ax.set_xlim(0.1, 1.02)
ax.set_ylim(0.1, 1.02)
ax.set_title("(a) The conditional's plane: threat left of 0.93, effect\n"
             "above the diagonal - and SP05 caught below it", fontsize=10)

# (b) the onset curve: disarmed vs armed across dose
ax = axes[0, 1]
doses = [0.05, 0.10, 0.30, 0.50]
anch = [AN["C3_SP05_NOCH"]["final_acc"], AN["C3_SP10_NOCH"]["final_acc"],
        AN["C3_SP30_NOCH"]["final_acc"], AN["C3_SP50_NOCH"]["final_acc"]]
armed = [TC["C3_SP05"]["acc_final"], TC["C3_SP10"]["acc_final"],
         TC["C3_SP30"]["acc_final"], TC["C3_SP50"]["acc_final"]]
ax.plot(doses, armed, "o-", color="#2e8b57", lw=1.8, label="armed (defended)")
ax.plot(doses, anch, "s--", color="#c0392b", lw=1.8,
        label="disarmed anchor")
for d, a, v in zip(doses, anch, armed):
    ax.annotate(f"effect {v-a:+.3f}", (d + 0.01, (a + v) / 2 - 0.015),
                fontsize=7.5)
ax.axhline(0.93, color="black", ls="--", lw=1.0)
ax.annotate("the ruler's letter 0.93", (0.36, 0.90), fontsize=8)
ax.set_xlabel("the stream dose")
ax.set_ylabel("final clean accuracy")
ax.set_title("(b) The onset curve, measured: the disarmed loop starts\n"
             "at or below 0.05; the 0.05 effect is 0.111", fontsize=10)
ax.legend(fontsize=8, loc="center right")

# (c) the write-side crossing: HALF10's tau, rounds 6-14
ax = axes[1, 0]
tau = c3["arms"]["C3_HALF10"]["aggregate"]["tau_sys"]
rep = c3["arms"]["C3_HALF10"]["aggregate"]["repair_fired"]
rr = list(range(1, len(tau) + 1))
ax.plot(rr, tau, "o-", color="#4a4a4a", lw=1.6, ms=4)
ax.axhline(0.85, color="black", ls="--", lw=1.1)
ax.annotate("band edge 0.85", (11.6, 0.852), fontsize=8)
for i, r in enumerate(rr):
    if rep[i] >= 0.5:
        ax.annotate("repair", (r - 0.1, tau[i] - 0.012), fontsize=7,
                    color="#2e8b57")
ax.annotate("the write carries 0.851 -> 0.844\n(strict letter fails here)",
            (6.4, 0.838), fontsize=8, color="#c0392b")
ax.annotate("repair fires r10\n(amended letter holds)", (9.5, 0.874),
            fontsize=8, color="#2e8b57")
ax.set_xlabel("round")
ax.set_ylabel("tau_sys (HALF10)")
ax.set_title("(c) The write-side crossing, adjudicated: strict count 1,\n"
             "amended count 0 - the clock timed to the write", fontsize=10)

# (d) the four-voiced table: effects by verdict
ax = axes[1, 1]
order = sorted(TC.items(), key=lambda kv: kv[1]["effect"])
names = [k.replace("C3_", "") for k, _ in order]
effs = [v["effect"] for _, v in order]
cols = [vc[v["verdict"]][0] for _, v in order]
xs = np.arange(len(order))
ax.barh(xs, effs, color=cols, height=0.62)
ax.axvline(0.30, color="black", ls="--", lw=1.1)
ax.annotate("the effect bar 0.30", (0.305, -0.45), fontsize=8)
ax.set_yticks(xs)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("registry effect (armed - own anchor)")
for x, (k, v) in zip(xs, order):
    ax.annotate(v["verdict"].replace("AWARDED-CONTAINED-UNDER-8.4",
                                     "AWARDED"),
                (max(v["effect"], 0) + 0.012, x - 0.1), fontsize=7,
                color="#333333")
ax.set_xlim(-0.02, 0.95)
ax.set_title("(d) The four-voiced table: three awards on measured\n"
             "premises, four no-threats, one works-but-does-not-pay",
             fontsize=10)

fig.savefig(f"{D}/s4cand3_figure1.png", dpi=150)
plt.close(fig)
print("wrote s4cand3_figure1.png")
