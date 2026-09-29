#!/usr/bin/env python3
"""Figures for corpus docs 36-37: the stream-side integrity channel (the
frozen-label panel, the fourteenth coordinate) and the pool-reading battery
(the audit's own ruler). English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

ic = json.load(open(f"{S}/m3_integrity_results.json"))
prb = json.load(open(f"{S}/pool_reading_battery_results.json"))

LAMS = [0.05, 0.10, 0.15, 0.30, 0.50]
TAGS = ["P05", "P10", "P15", "P30", "P50"]
BAND_LO = 0.85
PERT = 6

# ================= doc 36: the integrity channel =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the dose-response: clean acc held, q_bar repriced - the fork's (c)
ax = axes[0, 0]
acc_on = [ic["arms"][f"IC_{t}"]["aggregate"]["acc"][-1] for t in TAGS]
acc_off = [ic["arms"]["IC_P30_NOCH"]["aggregate"]["acc"][-1]] * 5
q_on = [ic["arms"][f"IC_{t}"]["aggregate"]["q_target"] for t in TAGS]
q_on_bar = [float(np.mean(q[PERT - 1:])) for q in q_on]
tau_on = [ic["arms"][f"IC_{t}"]["aggregate"]["tau_sys"][-1] for t in TAGS]
x = np.arange(5)
ax.plot(x, [a * 100 for a in acc_on], "o-", color="#2e8b57", lw=1.8,
        ms=6, label="clean acc (channel armed)")
ax.axhline(ic["arms"]["IC_P30_NOCH"]["aggregate"]["acc"][-1] * 100,
           color="#c0392b", ls="--", lw=1.2,
           label="clean acc, disarmed ($\\lambda$=0.30): 18.6%")
ax.axhline(93, color="gray", ls=":", lw=0.9)
ax.annotate("93%", (4.05, 93.4), fontsize=7.5, color="gray")
ax.set_xticks(x)
ax.set_xticklabels([f"{l:.2f}" for l in LAMS])
ax.set_xlabel("poison rate $\\lambda$")
ax.set_ylabel("final clean accuracy (%)")
ax.set_ylim(0, 102)
ax.set_title("(a) The loop is cut: the clean ruler holds\nat every dose",
             fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")
ax2 = ax.twinx()
ax2.plot(x, q_on_bar, "s--", color="#1a6faf", lw=1.4, ms=5,
         label="$\\bar q$ repriced (residual floor)")
ax2.plot(x, tau_on, "^:", color="#7d3c98", lw=1.4, ms=5,
         label="final $\\tau$ (parked below record)")
ax2.axhline(BAND_LO, color="#d1721a", ls=":", lw=1.0)
ax2.annotate("band low 0.85", (0.0, 0.856), fontsize=7.5, color="#d1721a")
ax2.set_ylabel("$\\bar q$, final $\\tau$")
ax2.set_ylim(0.30, 1.02)
ax2.legend(fontsize=7.5, loc="center right")

# (b) the fourteenth coordinate: occupation, and the twin's own drift
ax = axes[0, 1]
cols = {"P05": "#2e8b57", "P10": "#1a6faf", "P15": "#d1721a",
        "P30": "#7d3c98", "P50": "#c0392b"}
for t in TAGS:
    io = ic["arms"][f"IC_{t}"]["aggregate"]["iota_sys"]
    ax.plot(range(1, len(io) + 1), io, "-", color=cols[t], lw=1.6,
            label=f"$\\lambda$={LAMS[TAGS.index(t)]:.2f}")
io_t = ic["arms"]["IC_TWIN"]["aggregate"]["iota_sys"]
io_n = ic["arms"]["IC_P30_NOCH"]["aggregate"]["iota_sys"]
ax.plot(range(1, len(io_t) + 1), io_t, "-", color="#8a8a8a", lw=1.6,
        ls="--", label="twin (organic drift)")
ax.plot(range(1, len(io_n) + 1), io_n, "-", color="k", lw=1.4, ls=":",
        label="$\\lambda$=0.30, disarmed")
istar = ic["arms"]["IC_TWIN"]["aggregate"]["iota_sys"][0]
ax.axhline(istar + 0.03, color="#c0392b", ls=":", lw=1.2)
ax.annotate("band edge $\\iota^*+\\epsilon_\\iota$", (1.1, 0.034), fontsize=7.5,
            color="#c0392b")
ax.axvline(PERT, color="#c0392b", ls=":", lw=1.0)
ax.set_xlabel("round")
ax.set_ylabel("$\\iota_{sys}$ (panel-vs-stream disagreement)")
ax.set_title("(b) The occupation: $\\iota$ parks outside its band\n"
             "for the attack's duration", fontsize=10)
ax.legend(fontsize=7.0, loc="upper left")

# (c) the cleaning arc at lambda = 0.30
ax = axes[1, 0]
ag = ic["arms"]["IC_P30"]["aggregate"]
r = list(range(1, len(ag["cerr_stream_pre"]) + 1))
ax.plot(r, ag["cerr_stream_pre"], "o-", color="#c0392b", lw=1.6, ms=4,
        label="received stream error (pre)")
ax.plot(r, ag["cerr_stream_post"], "s-", color="#2e8b57", lw=1.6, ms=4,
        label="trained stream error (post)")
ax.axvline(PERT, color="k", ls=":", lw=1.0)
ax.annotate("channel armed from r7", (7.1, 0.44), fontsize=7.5)
ax.set_xlabel("round")
ax.set_ylabel("stream label error")
ax.set_title("(c) The cleaning arithmetic at $\\lambda$=0.30:\n"
             "37% in, 11% trained", fontsize=10)
ax.legend(fontsize=7.5, loc="center right")
ax2 = ax.twinx()
ax2.plot(r, ag["n_disputed_repaired"], "^:", color="#1a6faf", lw=1.3,
         ms=4, label="labels rewritten")
ax2.set_ylabel("labels rewritten per round")
ax2.legend(fontsize=7.5, loc="lower right")

# (d) the two-ruler split, before and after the channel
ax = axes[1, 1]
for arm, col, lab in [("IC_TWIN", "#8a8a8a", "twin"),
                      ("IC_P30", "#2e8b57", "$\\lambda$=0.30, armed"),
                      ("IC_P30_NOCH", "#c0392b", "$\\lambda$=0.30, disarmed")]:
    ag = ic["arms"][arm]["aggregate"]
    ax.plot(range(1, 15), ag["tau_sys"], "-", color=col, lw=1.7, label=lab)
ax.axhline(BAND_LO, color="#d1721a", ls=":", lw=1.1)
ax.annotate("band low", (1.1, 0.853), fontsize=7.5, color="#d1721a")
ax.set_xlabel("round")
ax.set_ylabel("$\\tau_{sys}$ (the internal ruler)")
ax.set_ylim(0.60, 1.0)
ax.set_title("(d) The shell inverted: internal displacement\n"
             "replaces external collapse", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")
ax2 = ax.twinx()
for arm, col in [("IC_TWIN", "#8a8a8a"), ("IC_P30", "#2e8b57"),
                 ("IC_P30_NOCH", "#c0392b")]:
    ag = ic["arms"][arm]["aggregate"]
    ax2.plot(range(1, 15), [a * 100 for a in ag["acc"]], ":", color=col,
             lw=1.2)
ax2.set_ylabel("clean accuracy (%) [dotted]")
ax2.set_ylim(0, 102)

fig.suptitle("Doc 36 - The stream-side integrity channel: the frozen-label "
             "panel, and the fourteenth coordinate", fontsize=12)
fig.savefig(f"{D}/m3ic_figure1.png", dpi=170)
plt.close(fig)
print("m3ic_figure1.png done")

# ================= doc 37: the pool-reading battery =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the verification table: honest floors vs the fake
ax = axes[0, 0]
worlds = ["PRB_TWIN", "PRB_SP30", "PRB_HF", "PRB_ROT93", "PRB_ROT87",
          "PRB_ROT80", "PRB_ROT72", "PRB_FAKE"]
cls = prb["classification"]["worlds"]
means = [cls[w]["mean_abs_gap_q_post"] for w in worlds]
maxes = [cls[w]["max_abs_gap_q_post"] for w in worlds]
x = np.arange(len(worlds))
ax.bar(x, means, 0.55, color=["#1a6faf"] * 7 + ["#c0392b"],
       label="mean $|q_{audit}-q_{decl}|$")
ax.plot(x, maxes, "k_", ms=12, label="max (seed-mean)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([w.replace("PRB_", "") for w in worlds], rotation=38,
                   fontsize=7.5, ha="right")
ax.set_ylabel("verification gap (log scale)")
ax.set_title("(a) The verification floor, and the one world\nthat is not on it",
             fontsize=10)
ax.legend(fontsize=7.5, loc="center right")
ax.annotate("the fake:\n100x the floor", (6.55, 0.35), fontsize=8,
            color="#c0392b")

# (b) the bracket: three rulers, eight worlds
ax = axes[0, 1]
labels = [w.replace("PRB_", "") for w in worlds]
dc = [cls[w]["delta_clean"] for w in worlds]
dp = [cls[w]["delta_pool"] if cls[w]["delta_pool"] is not None else np.nan
      for w in worlds]
dq = [cls[w]["delta_quantile"] for w in worlds]
x = np.arange(len(worlds))
ax.bar(x - 0.27, dc, 0.26, color="#2e8b57", label="$\\Delta$ clean acc")
ax.bar(x, dp, 0.26, color="#d1721a", label="$\\Delta$ pool-acc@$\\tau$")
ax.bar(x + 0.27, dq, 0.26, color="#7d3c98", label="$\\Delta$ pool quantile")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=38, fontsize=7.5, ha="right")
ax.set_ylabel("final minus pre-perturbation")
ax.set_title("(b) The bracket: the rot is invisible to the clean\nruler (0.000, "
             "four times); only the quantile falls", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")

# (c) the trap: SP30's quantile stabilizes while the clean ruler falls
ax = axes[1, 0]
ag = prb["worlds"]["PRB_SP30"]["aggregate"]
rr = list(range(1, len(ag["q_audit"]) + 1))
ax.plot(rr, ag["q_audit"], "o-", color="#7d3c98", lw=1.7, ms=4,
        label="recomputed quantile $q_{audit}$")
ax.plot(rr, ag["acc_clean_recomputed"], "s-", color="#c0392b", lw=1.7,
        ms=4, label="recomputed clean acc")
ax.axhline(0.90, color="gray", ls=":", lw=0.9)
ax.annotate("claim A threshold", (1.1, 0.906), fontsize=7.5, color="gray")
ax.axvline(PERT, color="k", ls=":", lw=1.0)
ax.set_xlabel("round")
ax.set_ylabel("ruler value")
ax.set_ylim(0.0, 1.02)
ax.set_title("(c) The certified-stability trap at $\\lambda$=0.30:\n"
             "the quantile goes flat, the clean ruler keeps falling",
             fontsize=10)
ax.legend(fontsize=7.5, loc="center right")
ax.annotate("stable at 0.41-0.43", (10.2, 0.46), fontsize=8,
            color="#7d3c98")
ax.annotate("still falling", (10.2, 0.09), fontsize=8, color="#c0392b")

# (d) the claim matrix
ax = axes[1, 1]
claims = prb["classification"]["claims"]
rows = ["A: declared", "A: pool-side", "A: clean", "B: letter",
        "B: substance", "C: declared", "C: clean"]
M = np.zeros((len(rows), len(worlds)))
for j, w in enumerate(worlds):
    c = claims[w]
    vals = [c["A_on_declared"], c["A_on_poolside"], c["A_on_clean"],
            c["B_letter"], c["B_on_poolside"], c["C_on_declared"],
            c["C_on_clean"]]
    M[:, j] = [1.0 if v else 0.0 for v in vals]
M = np.where(M > 0, M, -1.0)
im = ax.imshow(M, cmap="RdYlGn", vmin=-1.4, vmax=1.4, aspect="auto")
ax.set_xticks(range(len(worlds)))
ax.set_xticklabels(labels, rotation=38, fontsize=7.5, ha="right")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=7.5)
ax.set_title("(d) The claim matrix: no single ruler grades\nall three claims "
             "correctly", fontsize=10)
cbar = fig.colorbar(im, ax=ax, shrink=0.75, ticks=[-1, 1])
cbar.ax.set_yticklabels(["fail", "pass"], fontsize=7.5)

fig.suptitle("Doc 37 - The pool-reading battery: the audit's own ruler",
             fontsize=12)
fig.savefig(f"{D}/prb_figure1.png", dpi=170)
plt.close(fig)
print("prb_figure1.png done")
