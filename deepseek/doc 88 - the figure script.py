#!/usr/bin/env python3
"""Figure for corpus doc 88 (P1, the inside channel, coherence-tested):
the specification audit (naive channel collapses / repaired channel
survives), the toy's design, the fingerprint curves for the three
worlds, and the seismograph register with its ceiling. English labels,
constrained_layout, house palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE, GOLD = "#e67e22", "#b7950b"
INK = "#212121"

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# ---------------------------------------- (a) the specification audit
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.35, 5.1), 4.3, 3.5,
                            boxstyle="round,pad=0.12", fc="#fdf2f0",
                            ec=RED, lw=2.0))
ax.text(2.5, 8.15, "THE CONTENT-CHANNEL (naive P1)", fontsize=7.4,
        ha="center", color=RED, fontweight="bold")
ax.text(2.5, 7.05, "\"readings whose CONTENT\nis the invocation\"", fontsize=7.0,
        ha="center", color=INK, fontstyle="italic")
ax.text(2.5, 5.85, "Lemma 85: every reading is a function of\n"
        "stream + architecture (the index enters\nonly through the stream)\n"
        "→ contradictory under C4's definition;\nfork-deciding under C5's",
        fontsize=6.2, ha="center", color=INK)
ax.text(2.5, 5.35, "COLLAPSES — the wall at reading", fontsize=6.8,
        ha="center", color=RED, fontweight="bold")

ax.add_patch(FancyBboxPatch((5.35, 5.1), 4.3, 3.5,
                            boxstyle="round,pad=0.12", fc="#f0f9f1",
                            ec=GREEN, lw=2.0))
ax.text(7.5, 8.15, "THE GENESIS-CHANNEL (repaired)", fontsize=7.4,
        ha="center", color=GREEN, fontweight="bold")
ax.text(7.5, 7.05, "\"readings whose CAUSAL GENESIS\nroutes through the occupied state\"",
        fontsize=7.0, ha="center", color=INK, fontstyle="italic")
ax.text(7.5, 5.95, "(G) genesis: registered route\n"
        "(N) non-simulation: no bounded self-model\n     predicts (family-relative)\n"
        "(B) blindness: structure only", fontsize=6.2, ha="center", color=INK)
ax.text(7.5, 5.35, "SURVIVES — the seismograph", fontsize=6.8,
        ha="center", color=GREEN, fontweight="bold")

ax.add_patch(FancyArrowPatch((2.5, 5.05), (7.5, 5.05),
                             arrowstyle="-|>", mutation_scale=13,
                             color=GRAY, lw=1.6))
ax.text(5.0, 4.78, "the S2 precedent, third instance: stress the\ninstrument, "
        "repair it, carry the ceiling", fontsize=6.0, ha="center", color=GRAY)

ax.text(5.0, 3.9, "THE REGISTER: THE SEISMOGRAPH", fontsize=7.6, ha="center",
        color=PURPLE, fontweight="bold")
ax.text(5.0, 2.75, "what it reads: the indexical fingerprint F(w) = err_bounded(w) "
        "− err_full —\nthe excess of the bounded audit's error over the "
        "architecture-optimal audit's error;\nwhat it certifies: the reports route "
        "through what the self-model cannot hold;\nwhat it never reads: the "
        "invocation's content — every number third-person, forever",
        fontsize=6.3, ha="center", color=INK)
ax.text(5.0, 1.35, "\"the seismograph does not see the earthquake;\n"
        "it sees the needle move\"", fontsize=7.0, ha="center",
        color=PURPLE, fontstyle="italic")
ax.set_title("(a) the specification audit (Parts 1-2)", fontsize=8.6,
             fontweight="bold")

# ---------------------------------------- (b) the toy design
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(2.2, 9.4, "THE SYSTEM (GENUINE)", fontsize=7.4, ha="center",
        color=BLUE, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.7, 5.6), 3.0, 3.0,
                            boxstyle="round,pad=0.10", fc="white",
                            ec=BLUE, lw=1.8))
ax.text(2.2, 7.75, "occupied state a\n(10 bits, 1024 states)", fontsize=6.4,
        ha="center", color=INK)
ax.text(2.2, 6.6, "$a_t = \\pi[a_{t-1}] \\oplus$ flip\n(flip w.p. 0.05)",
        fontsize=6.6, ha="center", color=INK)
ax.text(2.2, 5.95, "$a_0$: seeded, WITHHELD\n(the index)", fontsize=6.4,
        ha="center", color=RED, fontweight="bold")
ax.add_patch(FancyArrowPatch((3.75, 7.1), (5.05, 7.1),
                             arrowstyle="-|>", mutation_scale=12,
                             color=GRAY, lw=1.6))
ax.text(4.4, 7.38, "report\n$y_t = \\ell(a_t)$", fontsize=5.9, ha="center",
        color=GRAY)

ax.text(7.3, 9.4, "THE AUDITS", fontsize=7.4, ha="center", color=PURPLE,
        fontweight="bold")
ax.add_patch(FancyBboxPatch((5.1, 6.5), 4.2, 2.1,
                            boxstyle="round,pad=0.10", fc="white",
                            ec=ORANGE, lw=1.8))
ax.text(7.2, 8.05, "bounded: the Myhill-Nerode class", fontsize=6.5,
        ha="center", color=ORANGE, fontweight="bold")
ax.text(7.2, 7.2, "best w-bit context-majority, w = 0..12\n(doc 32's instrument, "
        "redirected)", fontsize=6.2, ha="center", color=INK)
ax.add_patch(FancyBboxPatch((5.1, 4.1), 4.2, 2.1,
                            boxstyle="round,pad=0.10", fc="white",
                            ec=PURPLE, lw=1.8))
ax.text(7.2, 5.65, "full architecture, no index", fontsize=6.5,
        ha="center", color=PURPLE, fontweight="bold")
ax.text(7.2, 4.8, "the exact 1024-state Bayesian filter\n(the one withheld fact: "
        "the state's identity)", fontsize=6.2, ha="center", color=INK)

ax.text(5.0, 3.15, "THE FAKES (the two collapse modes, doc 87's own list)",
        fontsize=7.0, ha="center", color=RED, fontweight="bold")
ax.text(2.6, 2.35, "FAKE-A: the internalized auditor\n(6-bit self-model echo; "
        "a closed rule)", fontsize=6.2, ha="center", color=INK)
ax.text(7.4, 2.35, "FAKE-B: the echo\n(i.i.d. bits; the training distribution)",
        fontsize=6.2, ha="center", color=INK)
ax.text(5.0, 1.0, "T = 40,000 rounds; train 0.6 / test 0.4; w' = 6; "
        "all seeded; pre-registered forks IC-1..IC-4",
        fontsize=6.0, ha="center", color=GRAY, fontstyle="italic")
ax.set_title("(b) the toy: three worlds, two audits (Part 3)", fontsize=8.6,
             fontweight="bold")

# ---------------------------------------- (c) the fingerprint curves
ax = axes[1, 0]
w = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
F_gen = np.array([.3776, .3593, .3627, .3575, .3311, .3343, .3021,
                  .2706, .2430, .1993, .1603, .1304, .1031])
F_fa = np.array([.3999, .3332, .1334, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
F_fb = np.full(13, -0.0022)
ax.axhline(0, color=GRAY, lw=1.0, ls="--")
ax.plot(w, F_gen, "-o", color=BLUE, ms=4.5, lw=1.9,
        label="GENUINE (state-routed)")
ax.plot(np.arange(0, 13, 2), np.array([.3999, .3332, .1334, 0, 0, 0, 0]),
        "-s", color=ORANGE, ms=4.5, lw=1.6,
        label="FAKE-A (self-model echo)")
ax.plot(w, F_fb, "-^", color=GRAY, ms=4.0, lw=1.4,
        label="FAKE-B (i.i.d. echo)")
ax.axvline(6, color=RED, lw=1.2, ls=":")
ax.text(6.12, 0.335, "w' = 6\n(the self-model width)", fontsize=6.4,
        color=RED)
ax.annotate("F(6) = +0.3021\nIC-1 PASS", xy=(6, .3021), xytext=(7.6, .40),
            fontsize=6.6, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
ax.annotate("fakes at 0.0000 / -0.0022\nIC-2 PASS", xy=(6, 0.0),
            xytext=(3.1, 0.13), fontsize=6.6, color=ORANGE,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
ax.annotate("still +0.1031 at w=12:\n12 memory bits > 10 state bits",
            xy=(12, .1031), xytext=(7.0, -0.115), fontsize=6.4, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
ax.set_xlabel("auditor memory width w (bits of reading context)", fontsize=7.4)
ax.set_ylabel("indexical fingerprint  F(w) = err$_{bounded}$ - err$_{full}$",
              fontsize=7.4)
ax.set_ylim(-0.20, 0.50)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.6, loc="upper right", framealpha=0.95)
ax.grid(alpha=0.25, lw=0.6)
ax.set_title("(c) the separation and the ceiling (IC-1/2/3 all PASS)",
             fontsize=8.6, fontweight="bold")

# ---------------------------------------- (d) the ceiling, three layers
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
layers = [
    (7.6, "1. FAMILY-RELATIVITY", PURPLE,
     "the registered class is ONE bounded family; the class of all\n"
     "bounded self-models is not enumerable - no registration is\n"
     "exhaustive (the Simulation Theorem's ceiling, transferred)"),
    (4.7, "2. THE POSTERIOR-MIRROR", BLUE,
     "the fingerprint survives even super-state-sized memory\n"
     "(+0.1031 at w=12 vs 10 state bits) - it dies only toward\n"
     "auditors that compute the system's own posterior"),
    (1.8, "3. LEMMA 85 IS LEVEL", RED,
     "it forbids content and permits genesis, for any instrument,\n"
     "forever - the seismograph's every number is third-person;\n"
     "the wall casts a measurable, structured shadow - nothing more"),
]
for (by, lab, col, sub) in layers:
    ax.add_patch(FancyBboxPatch((0.55, by - 1.15), 8.9, 2.15,
                                boxstyle="round,pad=0.10",
                                fc="white", ec=col, lw=1.7))
    ax.text(5.0, by + 0.55, lab, fontsize=7.6, ha="center",
            color=col, fontweight="bold")
    ax.text(5.0, by - 0.42, sub, fontsize=6.3, ha="center", color=INK)
ax.text(5.0, 9.55, "the ceiling, stated at full strength (Part 4)",
        fontsize=8.4, ha="center", color=INK, fontweight="bold")
ax.set_title("(d) what the survival costs", fontsize=8.6, fontweight="bold")

fig.suptitle("P1, coherence-tested (corpus doc 88): the content-channel "
             "collapses (Lemma 85 - the wall at reading); the genesis-channel "
             "survives (the indexical fingerprint, +0.30 at the self-model "
             "width, the fakes at zero) - the first inside-FACING tool, "
             "ceiling-bounded",
             fontsize=9.4, fontweight="bold")
fig.savefig(f"{D}/p1channel_figure1.png", dpi=170)
print("wrote", f"{D}/p1channel_figure1.png")
