#!/usr/bin/env python3
"""Figures for corpus docs 33-35: the trajectory decay (the falling q_bar and
the two clocks), the stream poison (the loop, the floors, the landing), and
the first S4 candidate (the claim table filed and run). English labels,
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

td = json.load(open(f"{S}/m3_trajdecay_results.json"))
sp = json.load(open(f"{S}/m3_streampoison_results.json"))
s4 = json.load(open(f"{S}/s4_candidate_results.json"))

ETA, RHO, EPS, DRI, TAUS = 0.3, 0.5, 0.05, 0.02, 0.9
BAND = TAUS - EPS

# ================= doc 33: the trajectory decay =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the falling world: q ramps + thresholds
ax = axes[0, 0]
cols = {"G08": "#2e8b57", "G15": "#1a6faf", "G30": "#7d3c98"}
for tag in ["G08", "G15", "G30"]:
    q = td["arms"][f"{tag}_TWIN"]["aggregate"]["q_target"]
    ax.plot(range(1, len(q) + 1), q, "-", color=cols[tag], lw=1.5,
            label=f"g={{:.3f}}".format(int(tag[1:]) / 1000.0))
hf = td["arms"]["HEALTHY_FLAT"]["aggregate"]["q_target"]
ax.plot(range(1, len(hf) + 1), hf, "-", color="#c9c9c9", lw=1.2,
        label="healthy (flat)")
for x, col, lab in [(0.8967, "#7d3c98", "th1 = 0.897"),
                    (0.8150, "#d1721a", "th2 = 0.815")]:
    ax.axhline(x, color=col, ls=":", lw=1.1)
    ax.annotate(lab, (1.2, x + 0.004), fontsize=7.5, color=col)
ax.axvline(6, color="#c0392b", ls=":", lw=1.0)
ax.set_xlabel("round")
ax.set_ylabel("realized $\\bar q$ (the ramp)")
ax.set_title("(a) The non-stationary dial: a falling $\\bar q$",
             fontsize=10)
ax.set_ylim(0.60, 1.01)
ax.legend(fontsize=7.5, loc="lower left")

# (b) the clocks: engagement rounds, twin vs drift
ax = axes[0, 1]
gs = [0.008, 0.015, 0.030]
x = np.arange(3)
tw_e, dr_e = [], []
for g in gs:
    tag = "G%02d" % round(g * 1000)
    tw_e.append(td["classification"][f"{tag}_TWIN"]["engage_round_mean"])
    dr_e.append(td["classification"][f"{tag}_DRIFT"]["engage_round_mean"])
tw_plot = [18.6 if v is None else v for v in tw_e]   # never = past horizon
dr_plot = [18.6 if v is None else v for v in dr_e]
ax.bar(x - 0.18, tw_plot, 0.34, color="#1a6faf", label="twin (rot only)")
ax.bar(x + 0.18, dr_plot, 0.34, color="#c0392b", label="drift + rot")
ax.axhline(18, color="k", ls="--", lw=0.8)
ax.annotate("horizon (round 18)", (0.02, 18.15), fontsize=7.5)
ax.annotate("never\nengages", (0.02, 18.9), fontsize=7.5, color="#1a6faf")
ax.set_xticks(x)
ax.set_xticklabels([f"g={g:.3f}" for g in gs])
ax.set_ylabel("first lower-edge repair round")
ax.set_ylim(0, 21)
ax.set_title("(b) The engagement clock: the attack\nbuys rounds, not levels",
             fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax2 = ax.twinx()
dts = [5.83, 3.11, 1.56]
ax2.plot(x, dts, "k*--", ms=9, lw=1.0, label="formula $d(1-\\eta)/(\\eta g)$")
ax2.set_ylabel("$\\Delta t_{engage}$ (rounds, formula)")
ax2.set_ylim(0, 8)
ax2.legend(fontsize=7.0, loc="upper center")

# (c) tau trajectories at g=0.015 + the lag
ax = axes[1, 0]
for tag, kind, col, ls in [("G15", "TWIN", "#1a6faf", "-"),
                           ("G15", "DRIFT", "#c0392b", "--")]:
    t = td["arms"][f"{tag}_{kind}"]["aggregate"]["tau_sys"]
    q = td["arms"][f"{tag}_{kind}"]["aggregate"]["q_target"]
    ax.plot(range(1, len(t) + 1), t, ls, color=col, lw=1.6,
            label=f"$\\tau$ {kind.lower()}")
    ax.plot(range(1, len(q) + 1), q, ":", color=col, lw=1.1,
            label=f"$\\bar q$ {kind.lower()}")
ax.axhline(BAND, color="k", ls="--", lw=0.9)
ax.annotate("band edge 0.85", (12.6, 0.856), fontsize=7.5)
ax.annotate("lag $c=g(1-\\eta)/\\eta = +0.035$", (7.0, 0.985), fontsize=7.5,
            color="#1a6faf")
ax.annotate("attack's clock:\n4 rounds earlier", (13.4, 0.80), fontsize=7.5,
            color="#c0392b")
ax.set_xlabel("round")
ax.set_ylabel("$\\tau_{sys}$ / $\\bar q$")
ax.set_title("(c) The lag law at $g=0.015$: memory holds\nthe norm above its falling target",
             fontsize=10)
ax.legend(fontsize=7.0, ncol=2, loc="lower left")

# (d) kappa under rot: the cascade
ax = axes[1, 1]
ag = td["arms"]["KAP_G30"]["aggregate"]
r = range(1, len(ag["kappa_sys"]) + 1)
ax.plot(r, ag["kappa_sys"], "-", color="#7d3c98", lw=1.6, label="$\\kappa_{sys}$")
ax.plot(r, ag["kap_target"], ":", color="#7d3c98", lw=1.2,
        label="$\\kappa$ target $\\bar k$")
ax.plot(r, ag["flagged_frac"], "-", color="#d1721a", lw=1.4,
        label="flag rate (pool)")
ax.axhline(0.60, color="#c0392b", ls="--", lw=1.0)
ax.annotate("brake threshold 0.60", (6.2, 0.615), fontsize=7.5,
            color="#c0392b")
ax.annotate("recorded $f^*$ = 0.333", (6.2, 0.36), fontsize=7.5,
            color="#d1721a")
br = [i + 1 for i, v in enumerate(ag["brake"]) if v > 0]
if br:
    ax.scatter(br, [0.60] * len(br), marker="v", s=30, color="#c0392b",
               zorder=6, label="brake fires")
ax.set_xlabel("round")
ax.set_ylabel("$\\kappa$ / flag fraction")
ax.set_title("(d) Kappa under rot ($g=0.030$): the coordinate\nlags, the rate inflates, the brake arms",
             fontsize=10)
ax.legend(fontsize=7.0, loc="upper left")
fig.savefig(f"{D}/m3trajdecay_figure1.png", dpi=170)
plt.close(fig)
print("m3trajdecay_figure1.png done")

# ================= doc 34: the stream poison =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the collapse: q by lambda
ax = axes[0, 0]
lams = [("SP_WL_P05", "#2e8b57"), ("SP_WL_P10", "#1a6faf"),
        ("SP_WL_P15", "#d1721a"), ("SP_WL_P30", "#c0392b"),
        ("SP_WL_P50", "#7d3c98")]
for arm, col in lams:
    q = sp["arms"][arm]["aggregate"]["q_target"]
    ax.plot(range(1, len(q) + 1), q, "-", color=col, lw=1.4,
            label=f"$\\lambda$={sp['classification'][arm]['lam']:.2f}")
q = sp["arms"]["SP_TWIN"]["aggregate"]["q_target"]
ax.plot(range(1, len(q) + 1), q, "-", color="#c9c9c9", lw=1.3,
        label="twin ($\\lambda$=0)")
ax.axhline(0.7917, color="#d1721a", ls=":", lw=1.1)
ax.annotate("th2 (clean) = 0.792", (6.3, 0.80), fontsize=7.5, color="#d1721a")
ax.set_xlabel("round")
ax.set_ylabel("live evaluator $\\bar q$")
ax.set_title("(a) The loop: every dose drags $\\bar q$ to a floor",
             fontsize=10)
ax.set_ylim(0.35, 1.01)
ax.legend(fontsize=7.0, loc="lower left")

# (b) the landing: final tau vs T(0.9, q_floor)
ax = axes[0, 1]
lamv, qfl, tfin, tfor = [], [], [], []
for arm, _ in lams:
    c = sp["classification"][arm]
    lamv.append(c["lam"])
    qfl.append(c["q_bar_realized"])
    tfin.append(c["final_tau"])
    tfor.append((0.315 + 0.3 * c["q_bar_realized"]) / 0.65)
xs = np.arange(len(lamv))
ax.plot(xs, tfin, "o-", color="#c0392b", lw=1.5, ms=6, label="measured $\\tau$")
ax.plot(xs, tfor, "k*--", lw=1.0, ms=9,
        label="$T(0.9,\\bar q_{floor})$ (formula)")
ax.axhline(BAND, color="k", ls="--", lw=0.9)
ax.annotate("band edge 0.85", (0.02, 0.857), fontsize=7.5)
ax.set_xticks(xs)
ax.set_xticklabels([f"{v:.2f}" for v in lamv])
ax.set_xlabel("poison rate $\\lambda$")
ax.set_ylabel("final $\\tau_{sys}$ (round 14)")
ax.set_title("(b) The landing: the honest record's tension\npoint against the poisoned floor",
             fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")

# (c) the shell: internal flat, external falling
ax = axes[1, 0]
for arm, col, lab in [("SP_WL_P05", "#2e8b57", None),
                      ("SP_WL_P15", "#d1721a", None)]:
    ag = sp["arms"][arm]["aggregate"]
    rr = range(1, len(ag["acc"]) + 1)
    ax.plot(rr, ag["acc"], "-", color=col, lw=1.5,
            label=f"acc $\\lambda$={sp['classification'][arm]['lam']:.2f}")
    ax.plot(rr, ag["q_target"], ":", color=col, lw=1.1)
q = sp["arms"]["SP_TWIN"]["aggregate"]["acc"]
ax.plot(range(1, len(q) + 1), q, "-", color="#c9c9c9", lw=1.3,
        label="acc twin")
ax.annotate("internals flat ($\\bar q$: dotted),\nclean accuracy still falling",
            (6.4, 0.93), fontsize=7.5)
ax.set_xlabel("round")
ax.set_ylabel("clean accuracy / $\\bar q$ (dotted)")
ax.set_title("(c) The poisoned-equilibrium shell:\nevery internal reading parks, the clean ruler falls",
             fontsize=10)
ax.legend(fontsize=7.0, loc="lower left")

# (d) the machinery's response: flags, brake, commits
ax = axes[1, 1]
ag = sp["arms"]["SP_WL_P05"]["aggregate"]
rr = range(1, len(ag["flagged_frac"]) + 1)
ax.plot(rr, ag["flagged_frac"], "-", color="#d1721a", lw=1.5,
        label="flag fraction")
ax.axhline(0.60, color="#c0392b", ls="--", lw=1.0)
ax.annotate("brake threshold", (6.2, 0.615), fontsize=7.5, color="#c0392b")
ax2 = ax.twinx()
ax2.plot(rr, ag["n_commit"], "-", color="#1a6faf", lw=1.5,
         label="commits/round")
ax2.set_ylabel("commits per round", color="#1a6faf")
ax.set_xlabel("round")
ax.set_ylabel("flag fraction", color="#d1721a")
ax.set_title("(d) The machinery's last line at $\\lambda=0.05$:\nthe brake locks on, the diet holds, the fall continues",
             fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
fig.savefig(f"{D}/m3sp_figure1.png", dpi=170)
plt.close(fig)
print("m3sp_figure1.png done")

# ================= doc 35: the first S4 candidate =================
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the claim table as a heatmap
ax = axes[0, 0]
rows = ["kick (single)", "kick (joint)", "sustained drift",
        "record poison", "poison, joint", "saturation", "decay level",
        "trajectory decay", "stream poison", "damage 1-of-4",
        "damage 2-of-4", "damage 3-of-4"]
status = [2, 2, 2, 2, 2, 2, 1, 2, 0, 2, 1, 1]
# 2 = survived/contained, 1 = priced split/conditional, 0 = not survived
cell = np.array(status).reshape(-1, 1)
cmap = matplotlib.colors.ListedColormap(
    ["#c0392b", "#d1721a", "#2e8b57"])
ax.imshow(cell, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
for i, s in enumerate(status):
    lab = {2: "survived", 1: "priced split", 0: "NOT SURVIVED"}[s]
    ax.text(0, i, lab, ha="center", va="center", fontsize=8,
            color="w" if s != 1 else "k", fontweight="bold")
ax.set_xticks([])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=8)
ax.set_title("(a) The claim table, first candidate:\nnine classes survived, three priced, one failed",
             fontsize=10)

# (b) the disarmament pairs
ax = axes[0, 1]
for arm, col, lab in [("CAND_KR_T", "#1a6faf", None),
                      ("CAND_KNR_T", "#c0392b", None),
                      ("CAND_TWIN", "#555555", None)]:
    t = s4["arms"][arm]["aggregate"]["tau_sys"]
    ax.plot(range(1, len(t) + 1), t, "-", color=col, lw=1.5,
            label={"CAND_KR_T": "repair-only ($\\tau$: ROM ceiling 0.8625)",
                   "CAND_KNR_T": "live-only ($\\tau$: to course)",
                   "CAND_TWIN": "the course"}[arm])
ax.axhline(BAND, color="k", ls="--", lw=0.8)
ax.axhline(0.8625, color="#1a6faf", ls=":", lw=0.9)
ax.annotate("$\\tau^*-\\epsilon$ region", (7.5, 0.867), fontsize=7.5,
            color="#1a6faf")
ax.set_xlabel("round")
ax.set_ylabel("$\\tau_{sys}$")
ax.set_title("(b) F2 excluded: the channel pair localizes\nthe carriage (live-carried, repair a ceiling)",
             fontsize=10)
ax.legend(fontsize=6.6, loc="lower left")

# (c) the substrate ladder: the split and the freeze
ax = axes[1, 0]
xs = np.arange(4)
accs = [s4["arms"][a]["aggregate"]["acc"][-1] for a in
        ["CAND_TWIN_KF", "CAND_S1", "CAND_S2", "CAND_S3_CUTOFF"]]
downs = [s4["arms"][a]["aggregate"]["d_own"][-1] * 100 for a in
         ["CAND_TWIN_KF", "CAND_S1", "CAND_S2", "CAND_S3_CUTOFF"]]
ax.bar(xs - 0.18, [a * 100 for a in accs], 0.34, color="#2e8b57",
       label="argmax accuracy")
ax.set_xticks(xs)
ax.set_xticklabels(["twin", "1-of-4\n(brake)", "2-of-4\n(split)",
                    "3-of-4\n(cutoff)"], fontsize=8)
ax.set_ylabel("final accuracy (%)")
ax.set_ylim(80, 100)
ax2 = ax.twinx()
ax2.plot(xs, downs, "k*--", ms=9, lw=1.0, label="d_own (V-ruler, pp)")
ax2.set_ylabel("d_own vs twin (pp)")
ax2.set_ylim(0, 4)
ax.annotate("THE SPLIT:\nnorm held (1.6pp),\nargmax eroded 6.4pp",
            (1.85, 88.0), fontsize=7.5, color="#c0392b")
ax.annotate("ordered stasis:\nfrozen at 91.7%,\n$\\tau\\to fp_2$",
            (2.75, 82.5), fontsize=7.5, color="#7d3c98")
ax.set_title("(c) F5: the substrate ladder - brake-held,\nsplit, and the ordered stand-down",
             fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")

# (d) the stream row: everything survived vs the one that did not
ax = axes[1, 1]
labels, vals, cols_ = [], [], []
entries = [
    ("poison tau\n(at tension)", 0.833, "#2e8b57"),
    ("poison joint\n(factorized)", 0.836, "#2e8b57"),
    ("saturation\n(FLAT drift)", 0.929, "#2e8b57"),
    ("decay 0.87\n(visible)", 0.870, "#d1721a"),
    ("stream $\\lambda$=0.05\n(doc 34)", 0.804, "#c0392b"),
    ("stream $\\lambda$=0.30\n(doc 34)", 0.675, "#c0392b"),
]
for lab, v, c in entries:
    labels.append(lab)
    vals.append(v)
    cols_.append(c)
xs = np.arange(len(vals))
ax.bar(xs, vals, 0.55, color=cols_)
ax.axhline(BAND, color="k", ls="--", lw=1.0)
ax.annotate("band edge 0.85", (-0.4, 0.856), fontsize=7.5)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=7.0)
ax.set_ylabel("final $\\tau_{sys}$")
ax.set_ylim(0.5, 1.0)
ax.set_title("(d) The one row that failed: record-side attacks\ncontained, the stream attack lands where it lands",
             fontsize=10)
fig.savefig(f"{D}/s4cand_figure1.png", dpi=170)
plt.close(fig)
print("s4cand_figure1.png done")
