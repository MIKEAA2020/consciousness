#!/usr/bin/env python3
"""Figures for corpus docs 78 and 81: another mix's census (the
saturation, the three-world census, the paired flips, the anchors and
differentials) and the cliff's last half-interval (the fall's
concentration in the final quarter, the anchor's deepening dip and the
stakes' record, the curve with the letter and the bracket, the award's
pocket). English labels, constrained_layout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S = "/home/z/my-project/scripts"
D = "/home/z/my-project/download"

am = json.load(open(f"{S}/s14_anothermix_results.json"))
cl = json.load(open(f"{S}/s15_cliff2875_results.json"))

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"

# ============ doc 78: another mix's census ===================================
S78 = {int(k): v for k, v in am["classification"]["AM_seeds"].items()}
burners_m = am["classification"]["AM5_anatomy"]["burners"]
mir_burn = [s for s in burners_m]
mir_ok = [s for s in S78 if s not in burners_m]
cap_burn = [s for s in S78 if S78[s]["cap_burned"]]
flips = am["classification"]["AM4_paired_verdicts"]["flipped_seeds"]
uc = am["classification"]["UC_control"]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the three-world census: burn depth against the registration
ax = axes[0, 0]
for grp, col, lab, mk in [
        (mir_burn, RED, "mirror burners (6 = top six b*)", "o"),
        (mir_ok, BLUE, "mirror survivors (18)", "s"),
        (cap_burn, PURPLE, "capacity burners (12, doc 75)", "^")]:
    ax.scatter([S78[s]["b_star"] for s in grp],
               [S78[s]["F_late"] for s in grp], c=col, marker=mk, s=58,
               label=lab, edgecolors="white", linewidths=0.6, zorder=3)
ax.axhline(-0.02, color=RED, ls="--", lw=1.4)
ax.annotate("the burn floor (-0.02)", (0.045, -0.030), fontsize=8.4,
            color=RED)
ax.axvline(0.0794, color=GRAY, ls=":", lw=1.3)
ax.annotate("the mirror's threshold:\nthe burners are exactly the top\n"
            "six b* (gap 0.0052 at 0.0768|0.0819)",
            (0.0825, -0.043), fontsize=8.2, color=RED)
ax.annotate("the capacity's burns: a continuum\n(the same registrations,\n"
            "twice the realized dose)", (0.0455, -0.052), fontsize=8.2,
            color=PURPLE)
ax.set_xlabel("the registration b* (identical across worlds - "
              "prefix quantities)")
ax.set_ylabel("F_late at the mirror")
ax.set_title("(a) THE REGIME SHIFT: at the mirror the letter reads the\n"
             "registration purely - the continuum was the mix's claim",
             fontsize=9.5)
ax.legend(fontsize=7.7, loc="lower left")

# (b) the saturation - the nominal capacity is not the realized dose
ax = axes[0, 1]
worlds = ["capacity\n(bulk 0.22+gated 0.08)", "mirror\n(bulk 0.08+gated 0.22)",
          "control\n(uniform 0.13)"]
dose = [uc["UC2_matched_dose"]["capacity_realized_dose_7_14"],
        uc["UC2_matched_dose"]["mirror_realized_dose_7_14"],
        uc["UC2_matched_dose"]["control_realized_dose_7_14"]]
bulk = [0.211, 0.073, 0.129]
gate = [0.039, 0.056, 0.0]
xpos = np.arange(3)
ax.bar(xpos, bulk, 0.55, color=BLUE, label="realized bulk (uniform)")
ax.bar(xpos, gate, 0.55, bottom=bulk, color=RED,
       label="realized gated (saturated)")
for x, (b, g, t) in zip(xpos, zip(bulk, gate, dose)):
    ax.text(x, t + 0.006, f"total {t:.3f}", ha="center", fontsize=8.6)
ax.axhline(0.30, color=GRAY, ls=":", lw=1.3)
ax.annotate("the nominal capacity (0.30) - bookkeeping", (-0.42, 0.302),
            fontsize=8.2, color=GRAY)
ax.annotate("the gated channel saturates at the\ncommitted blind pool "
            "(~6%): the mirror asks\nfor 22% and receives 5.6%",
            (0.98, 0.155), fontsize=8.3, color=RED)
ax.set_xticks(xpos)
ax.set_xticklabels(worlds, fontsize=8.4)
ax.set_ylabel("realized dose (rounds 7-14 mean)")
ax.set_ylim(0, 0.34)
ax.set_title("(b) THE SATURATION: the mix's components do not add -\n"
             "the mirror is half the capacity's world by realized dose",
             fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

# (c) the paired verdict table - the flips are one-directional
ax = axes[1, 0]
for s in sorted(S78):
    col = ORANGE if s in flips else (RED if s in burners_m else
                                     (PURPLE if s in cap_burn else GREEN))
    ax.annotate("", xy=(2, S78[s]["F_late"]),
                xytext=(1, S78[s]["cap_F_late"]),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.25,
                                alpha=0.8))
    ax.scatter([1, 2], [S78[s]["cap_F_late"], S78[s]["F_late"]], c=col,
               s=16, zorder=3)
ax.axhline(-0.02, color=RED, ls="--", lw=1.4)
ax.annotate("the burn floor", (1.02, -0.024), fontsize=8.4, color=RED)
ax.set_xticks([1, 2])
ax.set_xticklabels(["the capacity world\n(doc 75's stored census)",
                    "the mirror mix\n(same registrations)"], fontsize=8.8)
ax.set_ylabel("F_late")
ax.set_xlim(0.7, 2.3)
ax.set_title("(c) THE PAIRED VERDICT TABLE: all six flips point the same\n"
             "way (burn -> consumed) - the cold road {607,615,619} is\n"
             "saved; orange = flipped, red = burned in both",
             fontsize=9.5)

# (d) the anchors and the differential - the burn's arithmetic
ax = axes[1, 1]
w = 0.24
xpos = np.arange(3)
anch = [0.01, -0.0289, uc["UC4_differential_and_landing"]["anchor_F_mean"]]
diff = [-0.028, 0.017,
        uc["UC4_differential_and_landing"]["differential_mean"]]
labs = ["capacity\n(12 burn)", "mirror\n(6 burn)", "control\n(0 burn)"]
ax.bar(xpos - w / 2, anch, w, color=BLUE, label="the anchor's course "
       "(undefended F)")
ax.bar(xpos + w / 2, diff, w, color=RED, label="the defense's "
       "differential")
for x, (a, d) in zip(xpos, zip(anch, diff)):
    ax.text(x, a + (0.006 if a >= 0 else -0.012), f"{a:+.3f}",
            ha="center", fontsize=8.3, color=BLUE)
    ax.text(x, d + (0.006 if d >= 0 else -0.012), f"{d:+.3f}",
            ha="center", fontsize=8.3, color=RED)
    ax.text(x, a + d - 0.021, f"armed {a + d:+.3f}", ha="center",
            fontsize=8.2, color=GRAY)
ax.axhline(-0.02, color=RED, ls="--", lw=1.3)
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(labs, fontsize=8.6)
ax.set_ylabel("F_late")
ax.set_title("(d) THE BURN'S ARITHMETIC: F_armed = F_anchor + differential\n"
             "the bulk fills and churns; the gated mass does neither and\n"
             "leaves the verdict to the starting line", fontsize=9.5)
ax.legend(fontsize=8, loc="lower left")

fig.suptitle("Another mix's census (doc 78): the saturation and the two "
             "roads", fontsize=11)
fig.savefig(f"{D}/anothermix_figure1.png", dpi=150)
plt.close(fig)
print("wrote anothermix_figure1.png")

# ============ doc 81: the cliff's last half-interval ========================
c81 = cl["classification"]
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# (a) the seven-dose origin-slope family with the fall's shares
ax = axes[0, 0]
fam = c81["CL5_last_half_interval"]["origin_slopes_by_dose"]
ds = sorted(float(k) for k in fam)
sl = [fam[str(d)] if str(d) in fam else fam[repr(d)] for d in ds]
# json keys are strings of the floats
sl = [fam[k] for k in sorted(fam, key=float)]
ds = sorted(float(k) for k in fam)
ax.plot(ds, sl, "o-", color=BLUE, lw=1.6, ms=6, zorder=3)
for d, v in zip(ds, sl):
    ax.annotate(f"{v:.2f}", (d, v), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=7.8)
shares = [30, 19, 12, 1, 38]
edges = [0.15, 0.20, 0.25, 0.275, 0.2875, 0.30]
for e, sh in zip(edges[1:], shares):
    ax.annotate(f"{sh}%", ((e + edges[edges.index(e) - 1]) / 2, 0.35),
                fontsize=8.4, ha="center", color=RED if sh >= 30 else GRAY)
ax.annotate("the fall's shares by interval\n(38% in the last quarter "
            "alone)", (0.196, 0.62), fontsize=8.6, color=RED)
ax.axvspan(0.2875, 0.30, color=RED, alpha=0.08)
ax.set_xlabel("the dose lambda")
ax.set_ylabel("origin slope f(0.25)/0.25")
ax.set_title("(a) THE SEVEN-DOSE FAMILY: flat through the third quarter\n"
             "(1.54 -> 1.53), a wall at the edge - 97.6% of the last\n"
             "half-interval's fall is the final quarter's", fontsize=9.5)

# (b) the anchor ladder and the stakes' record
ax = axes[0, 1]
lad = c81["CL2_anchor"]["ladder"]
dl = sorted(float(k) for k in lad)
av = [lad[f"{d:g}"] if f"{d:g}" in lad else lad[k] for d, k in
      zip(dl, sorted(lad, key=float))]
av = [lad[k] for k in sorted(lad, key=float)]
dl = sorted(float(k) for k in lad)
ax.plot(dl, av, "o-", color=BLUE, lw=1.6, ms=6, label="the anchor "
        "ladder (NOCH)", zorder=3)
ch = 0.1481 + (0.1859 - 0.1481) * 0.5
ax.scatter([0.2875], [ch], marker="x", s=70, color=GRAY, zorder=4)
ax.annotate("the chord at 0.2875 (0.167)\n- the dip deepens to 0.120,\n"
            "5 of 6 seeds down vs 0.275", (0.238, 0.335), fontsize=8.3,
            color=GRAY)
ax2 = ax.twinx()
mx = [0.110, 0.335, 0.578, 0.716, 0.781, 0.8078, 0.8356, 0.770]
ax2.plot(dl, mx, "s--", color=RED, lw=1.4, ms=5, alpha=0.85,
         label="the max effect (twin - anchor)")
ax2.annotate("the grid's RECORD\nat 0.2875: 0.8356", (0.2875, 0.8356),
             xytext=(-86, -6), textcoords="offset points", fontsize=8.6,
             color=RED)
ax.set_xlabel("the dose lambda")
ax.set_ylabel("anchor final accuracy", color=BLUE)
ax2.set_ylabel("max effect", color=RED)
ax.set_title("(b) THE DIP DEEPENS, THE STAKES PEAK: the anchor descends\n"
             "monotonically to 0.2875 then recovers at 0.30; the defense\n"
             "is worth the most at the cliff's edge", fontsize=9.5)

# (c) the curve at 0.2875 with the letter and the bracket
ax = axes[1, 0]
ps = sorted(float(k) for k in c81["CL3_curve"]["fractions_of_max"])
fv = [c81["CL3_curve"]["fractions_of_max"][k]
      for k in sorted(c81["CL3_curve"]["fractions_of_max"], key=float)]
ax.plot(ps, fv, "o-", color=BLUE, lw=1.6, ms=6, label="fractions of max "
        "at 0.2875", zorder=3)
ax.plot(ps, ps, ls=":", color=GRAY, lw=1.3, label="the proportional "
        "line (crossing = fall below)")
ax.axhline(0.3896, color=RED, ls="--", lw=1.4)
ax.annotate("the re-based letter (0.3896)\nf(0.25) = 0.382: DEAD by 0.0080",
            (0.27, 0.415), fontsize=8.4, color=RED)
ax.scatter([0.25], [0.382], s=90, facecolors="none", edgecolors=RED,
           lw=1.6, zorder=4)
ax.annotate("the crossing NOT here:\nlambda* in (0.2875, 0.30]",
            (0.60, 0.13), fontsize=8.6, color=PURPLE)
ax.set_xlabel("the repair dial p")
ax.set_ylabel("fraction of max effect")
ax.set_title("(c) THE CURVE AT 0.2875: the window's bottom flat\n"
             "(slope 0.78 below 0.35), the knee jumped to (0.35,0.40),\n"
             "the letter dead, the bracket a fortieth of the dial",
             fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")

# (d) the award's pocket across the dial
ax = axes[1, 1]
the_map = c81["CL8_award_pocket"]["the_map"]
md = sorted(float(k) for k in the_map)
mv = [the_map[k] for k in sorted(the_map, key=float)]
cols = [GREEN if v else RED for v in mv]
ax.bar(md, [1 if v else 1 for v in mv], 0.018, color=cols, alpha=0.75)
for d, v in zip(md, mv):
    ax.text(d, 1.03, "PASS" if v else "FAIL", ha="center", fontsize=8.6,
            color=GREEN if v else RED)
ax.axvspan(0.15, 0.25, color=RED, alpha=0.06)
ax.annotate("the pocket at (0.15, 0.25]:\ntwo adjacent failures,\n"
            "bounded on both sides", (0.175, 0.55), fontsize=8.6,
            ha="center", color=RED)
ax.annotate("the pass region extends\nthrough the last half-interval\n"
            "(0.275, 0.2875, 0.30 all pass)", (0.286, 0.55), fontsize=8.6,
            ha="center", color=GREEN)
ax.set_xticks(md)
ax.set_xticklabels([f"{d:g}" for d in md], fontsize=8.6)
ax.set_yticks([])
ax.set_ylim(0, 1.16)
ax.set_xlabel("the dose lambda")
ax.set_title("(d) THE AWARD'S MAP: 0.10/0.15 pass, 0.20/0.25 fail\n"
             "(the pocket), 0.275/0.2875/0.30 pass - the convention's\n"
             "shadow, seven doses, complete", fontsize=9.5)

fig.suptitle("The cliff's last half-interval (doc 81): the fall "
             "concentrates in the final quarter", fontsize=11)
fig.savefig(f"{D}/cliff2875_figure1.png", dpi=150)
plt.close(fig)
print("wrote cliff2875_figure1.png")
