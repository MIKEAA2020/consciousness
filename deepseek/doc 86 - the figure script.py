#!/usr/bin/env python3
"""Figure for corpus doc 86 (the general-poset preparation coherence):
the diamond and its polygon equation, the dimension trichotomy of
Theorem HH, the CNT census of Proposition 84, and the white-noise
section of Theorem FF. English labels, constrained_layout, house
palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE = "#e67e22"

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# ------------------------------------------- (a) the diamond and its equation
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
# the chain, left: p -> q -> r, arbitrary states (free)
ax.text(2.1, 9.3, "THE CHAIN (doc 42's case): every family extends",
        fontsize=7.6, ha="center", color=GREEN, fontweight="bold")
chain = [(1.0, 6.8), (2.1, 5.3), (3.2, 3.8)]
for (x, y), lab, col in zip(chain, ["$\\sigma^{pq}$", "$\\sigma^{qr}$",
                                     "$\\sigma^{pr}=\\sigma^{pq}\\otimes\\sigma^{qr}$"],
                            [GREEN, GREEN, "#b03a2e"]):
    ax.add_patch(Circle((x, y), 0.34, fc="white", ec=col, lw=1.8))
    ax.text(x, y - 0.75, lab, fontsize=7.2, ha="center", color=col)
for i in range(2):
    ax.add_patch(FancyArrowPatch(
        (chain[i][0] + 0.30, chain[i][1] - 0.18),
        (chain[i + 1][0] - 0.30, chain[i + 1][1] + 0.18),
        arrowstyle="-|>", mutation_scale=11, color=GRAY, lw=1.4))
ax.text(2.1, 2.4, "unique maximal chain:\nthe polygon equations are vacuous\n"
        "(Theorem GG(c), the iff)", fontsize=6.6, ha="center",
        color="#212121")
# the diamond, right
ax.text(7.3, 9.3, "THE DIAMOND: the two paths must agree",
        fontsize=7.6, ha="center", color=RED, fontweight="bold")
P, Q1, Q2, R = (6.6, 7.6), (6.0, 5.9), (8.6, 5.9), (7.6, 4.2)
for (x, y), lab in [(P, "p"), (Q1, "$q_1$"), (Q2, "$q_2$"), (R, "r")]:
    ax.add_patch(Circle((x, y), 0.30, fc="white", ec=BLUE, lw=1.8))
    ax.text(x, y + 0.48, lab, fontsize=8.0, ha="center", color=BLUE)
for a, b in [(P, Q1), (Q1, R), (P, Q2), (Q2, R)]:
    ax.add_patch(FancyArrowPatch(
        (a[0] + 0.14 * (b[0] - a[0]), a[1] + 0.14 * (b[1] - a[1])),
        (b[0] - 0.16 * (b[0] - a[0]), b[1] - 0.16 * (b[1] - a[1])),
        arrowstyle="-|>", mutation_scale=10, color=GRAY, lw=1.3))
ax.text(5.15, 5.9, "$E_{pq_1}\\!\\otimes\\!E_{q_1r}$", fontsize=7.0,
        ha="center", color=ORANGE)
ax.text(9.55, 5.9, "$E_{pq_2}\\!\\otimes\\!E_{q_2r}$", fontsize=7.0,
        ha="center", color=PURPLE)
ax.text(7.6, 3.0, "the polygon equation:\n"
        "$\\sigma^{pq_1}\\!\\otimes\\sigma^{q_1r} = "
        "\\sigma^{pq_2}\\!\\otimes\\sigma^{q_2r}$\n"
        "= the COMMON PRODUCT STATES of the two\n"
        "factorizations of $E_{pr}$ (Theorem HH)",
        fontsize=6.6, ha="center", color="#212121")
ax.set_title("(a) the cell's shape: a space question, not an existence "
             "question", fontsize=8.6, fontweight="bold")

# ------------------------------------------- (b) the dimension trichotomy
ax = axes[0, 1]
n = np.linspace(1.6, 5.4, 200)
dims = 4 * n - 4
amb = n ** 2 - 1
ax.plot(n, dims, color=GREEN, lw=2.0,
        label="dimension sum of the two pure-product loci (square legs)")
ax.plot(n, amb, color=RED, lw=2.0,
        label="ambient projective dimension $ab-1$")
ax.fill_between(n, dims, amb, where=dims >= amb, color=GREEN, alpha=0.14)
ax.fill_between(n, dims, amb, where=dims < amb, color=RED, alpha=0.10)
ax.axvline(3.5, color=GRAY, ls="--", lw=1.0)
for x, lab, col in [(2, "every qubit diamond\n(all legs $M_2$):\n"
                    "pure sections ALWAYS", GREEN),
                    (3, "$(3,3)$:\ntight (8 = 8)", GREEN),
                    (4, "$(4,4)$:\navoidable (12 < 15)", RED)]:
    ax.scatter([x], [4 * x - 4], s=60, color=col, zorder=3,
               edgecolor="k", linewidth=0.6)
    ax.annotate(lab, (x, 4 * x - 4), xytext=(x + 0.12, 4 * x - 4 + 2.2),
                fontsize=6.4, color=col)
ax.scatter([4.0], [4 * 4 - 4 + 6.0], s=0)
ax.text(2.35, 21.5, "FORCED: every factorization pair meets\n"
        "(Theorem HH(ii))", fontsize=7.0, color=GREEN,
        fontweight="bold")
ax.text(3.62, 6.2, "AVOIDABLE:\nthe general pair\nadmits no pure\n"
        "common product\n(HH(iii))", fontsize=7.0, color=RED,
        fontweight="bold")
ax.text(4.05, 13.6, "resonance, noted:\nTheorem F's exceptional\n"
        "family $(4,2,7),(6,3,17)$\nlives at this same\n"
        "small-dimension edge", fontsize=6.0, color=PURPLE,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f4ecf7", ec=PURPLE,
                  lw=0.8))
ax.set_xlabel("square leg dimension n (a=b=c=d)", fontsize=8)
ax.set_ylabel("projective dimension", fontsize=8)
ax.set_title("(b) the trichotomy's boundary at 3|4 - a two-dimensional "
             "leg is\nalways forced; the kill needs legs 3x4 or larger",
             fontsize=8.6, fontweight="bold")
ax.legend(fontsize=6.4, loc="upper left")
ax.grid(alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=7)
ax.set_ylim(0, 26)

# ------------------------------------------- (c) the CNOT census
ax = axes[1, 0]
th_a = np.linspace(0, np.pi, 180)
th_b = np.linspace(0, np.pi, 180)
# common iff theta_a at a pole OR theta_b at the equator (phi-free)
TA, TB = np.meshgrid(th_a, th_b, indexing="ij")
common = ((TA < 0.03) | (TA > np.pi - 0.03)
          | (np.abs(TB - np.pi / 2) < 0.03)).astype(float)
common[:, :] = np.where(common > 0, 1.0, 0.0)
ax.imshow(common.T * 0.85 + 0.08, origin="lower", extent=[0, np.pi, 0,
           np.pi], cmap="Greys", vmin=0, vmax=1, aspect="auto")
ax.axhline(np.pi / 2, color=ORANGE, lw=1.6)
ax.axvline(0.0, color=BLUE, lw=1.6)
ax.axvline(np.pi, color=BLUE, lw=1.6)
ax.set_xlabel("$\\theta_a$ (the control's polar angle)", fontsize=8)
ax.set_ylabel("$\\theta_b$ (the target's polar angle)", fontsize=8)
ax.set_xticks([0, np.pi / 2, np.pi])
ax.set_xticklabels(["$0$\n(Z pole)", "$\\pi/2$", "$\\pi$\n(Z pole)"],
                   fontsize=7)
ax.set_yticks([0, np.pi / 2, np.pi])
ax.set_yticklabels(["$0$", "$\\pi/2$\n(X equator)", "$\\pi$"], fontsize=7)
ax.set_title("(c) Proposition 84, the CNT census: common pure products "
             "exactly on\nthe boundary - a at a Z pole or b at the X "
             "equator (4,008 of 82,944 probes)",
             fontsize=8.6, fontweight="bold")
ax.text(np.pi / 2, 0.12, "no common products:\nthe generic pair",
        fontsize=6.8, ha="center", color="#212121")
ax.text(np.pi / 2, np.pi - 0.14, "the escaping families sit on the axes "
        "the two\nfactorizations' symmetries share (Z control, X target)",
        fontsize=6.4, ha="center", color="#212121")

# ------------------------------------------- (d) the white-noise section
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5.0, 9.4, "THEOREM FF - the section that always exists",
        fontsize=8.0, ha="center", color=GREEN, fontweight="bold")
ks = [(2, 1.9), (3, 4.4), (6, 7.2)]
for k, x in ks:
    m = np.eye(k) / k
    ax.imshow(m, extent=[x, x + 1.7, 5.2, 6.9], cmap="Greys",
              vmin=0, vmax=1, interpolation="nearest")
    ax.text(x + 0.85, 4.75, f"$\\omega^{{(k={k})}}=I_k/k$",
            fontsize=7.4, ha="center", color=GREEN)
ax.text(2.75, 3.6, "$\\otimes$", fontsize=13, ha="center")
ax.text(5.25, 3.6, "$=$", fontsize=13, ha="center")
ax.text(5.0, 2.6, "the identity tensors, the traces multiply, the dims "
        "multiply:\n$\\omega^{pr}=\\omega^{pq}\\otimes\\omega^{qr}$ on "
        "EVERY poset, EVERY assignment\n(three lines; certified on 41 "
        "posets, 22 diamonds, to $10^{-12}$)",
        fontsize=6.8, ha="center", color="#212121")
ax.text(5.0, 1.1, "the fill that testifies nothing - and the only fill "
        "a branch can force\n(transversal diamonds: white on all four "
        "arrows, Theorem HH(iv))",
        fontsize=6.8, ha="center", color=RED, fontstyle="italic")
ax.set_title("(d) the always-section and its price", fontsize=8.6,
             fontweight="bold")

fig.suptitle("The preparation coherence on general posets (corpus doc "
             "86): existence closes at white noise; the diamond prices "
             "richness by the common products of its two factorizations",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/prepcell_figure1.png", dpi=170)
print("wrote", f"{D}/prepcell_figure1.png")
