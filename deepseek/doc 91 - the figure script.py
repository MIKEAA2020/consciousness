#!/usr/bin/env python3
"""Figure for corpus doc 91 (P4 pre-registered): the fork as filed (one
ledger item, two execution registers), the decision (extension - the
impossibility proof, six grounds), what "not both" closes (detector branch
closed, chassis deployment untouched, the single mode-flip door), and the
pre-registered target (the Instrument-Form Bridge Constraint with its caps).
English labels, constrained_layout, house palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = "/home/z/my-project/download"

RED, BLUE, GREEN, GRAY, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                  "#7f8c8d", "#7d3c98")
ORANGE, GOLD = "#e67e22", "#b7950b"
INK = "#212121"

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)

# --------------------------------------------- (a) the fork as filed
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((1.3, 8.55), 7.4, 1.15,
                            boxstyle="round,pad=0.10", fc="#fdf6ec",
                            ec=INK, lw=1.6))
ax.text(5.0, 9.35, "P4 as filed (doc 87):", fontsize=7.2, ha="center",
        color=INK, fontweight="bold")
ax.text(5.0, 8.85, "\"the further-fact detector, OR its impossibility proof\"",
        fontsize=6.8, ha="center", color=INK, fontstyle="italic")

ax.add_patch(FancyBboxPatch((0.35, 4.95), 4.35, 3.15,
                            boxstyle="round,pad=0.10", fc="#f9e3e0",
                            ec=RED, lw=1.8))
ax.text(2.52, 7.62, "the detector branch", fontsize=7.2, ha="center",
        color=RED, fontweight="bold")
ax.text(2.52, 7.02, "an INSTRUMENT program", fontsize=6.4, ha="center",
        color=INK, fontweight="bold")
ax.text(2.52, 6.15, "specified, built, run;\ncompute-class work;\nthe register the chassis\ndeployment occupies",
        fontsize=6.0, ha="center", va="center", color=INK)
ax.text(2.52, 5.35, "register: DEPLOYMENT", fontsize=6.4, ha="center",
        color=RED, fontweight="bold")

ax.add_patch(FancyBboxPatch((5.25, 4.95), 4.4, 3.15,
                            boxstyle="round,pad=0.10", fc="#e8f4ec",
                            ec=GREEN, lw=1.8))
ax.text(7.45, 7.62, "the impossibility branch", fontsize=7.2, ha="center",
        color=GREEN, fontweight="bold")
ax.text(7.45, 7.02, "a THEOREM program", fontsize=6.4, ha="center",
        color=INK, fontweight="bold")
ax.text(7.45, 6.15, "formal-line derivation;\nthe register Theorem II and\nProposition 86 occupy",
        fontsize=6.0, ha="center", va="center", color=INK)
ax.text(7.45, 5.35, "register: EXTENSION", fontsize=6.4, ha="center",
        color=GREEN, fontweight="bold")

ax.add_patch(FancyBboxPatch((0.35, 3.22), 9.3, 1.42,
                            boxstyle="round,pad=0.08", fc="#f4f6f6",
                            ec=GRAY, lw=1.4))
ax.text(5.0, 4.28, "one ledger item, two registers - a catch-all that cannot fail:",
        fontsize=6.4, ha="center", color=INK, fontweight="bold")
ax.text(5.0, 3.55, "if the detector is never built, the proof is still owed; if the proof\nstalls, the detector was the other way",
        fontsize=5.9, ha="center", color=GRAY, fontstyle="italic")

ax.text(5.0, 2.55, "the filed results bearing on the seam:", fontsize=6.4,
        ha="center", color=INK, fontweight="bold")
ax.text(5.0, 1.55, "Lemma 85 (doc 88): every external reading computes a function of\n"
        "stream + architecture   |   Theorem II (doc 89): readings isomorphism-invariant;\n"
        "the invocation is presentation-data   |   Prop. 86 (doc 89): both fork-worlds admit\n"
        "the grammar unchanged   |   the seismograph: survived only via a registered route",
        fontsize=5.6, ha="center", va="center", color=INK)
ax.set_title("(a) the fork as filed (Part 0)", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (b) the decision
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.30, 8.30), 9.35, 1.45,
                            boxstyle="round,pad=0.10", fc="#e8f4ec",
                            ec=GREEN, lw=2.4))
ax.text(5.0, 9.35, "P4 = EXTENSION", fontsize=9.4, ha="center",
        color=GREEN, fontweight="bold")
ax.text(5.0, 8.72, "the impossibility proof - the detector branch is closed by\n"
        "this pre-registration", fontsize=6.6, ha="center", color=INK)

ax.text(5.0, 7.85, "the six grounds:", fontsize=6.8, ha="center",
        color=INK, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.30, 0.55), 9.35, 7.05,
                            boxstyle="round,pad=0.10", fc="white",
                            ec=GRAY, lw=1.2))
ax.text(0.62, 4.05, "G1  the filed wall points: Lemma 85 + Theorem II + Prop. 86 make\n"
        "      the detector expected-dead in every closed register; saying it once, at\n"
        "      theorem grade, across all families, is a derivation - the extension register\n"
        "\n"
        "G2  the repair route is unavailable: the seismograph survived via a registered\n"
        "      causal route; the further fact is presentation-data - the repair would need\n"
        "      as its route the thing it is deployed to detect\n"
        "\n"
        "G3  the detector is fork-deciding by design (doc 90: it would break the\n"
        "      exhaustive reading's floor); Lemma 85's C5-horn forbids building\n"
        "      fork-deciders; the impossibility floors both survivors symmetrically\n"
        "\n"
        "G4  the ledger's bookkeeping: the chassis deployment occupies the compute\n"
        "      register on its own warrant; P4 is filed as the next FORMAL session\n"
        "\n"
        "G5  falsifiability: the disjunction cannot fail; the pin is a bet that can fail\n"
        "      constructively - the strongest failure the formal line could produce\n"
        "\n"
        "G6  the pattern precedent (S2, P1, P3): instrument run into a wall, wall filed\n"
        "      as a theorem - the detector is the content-channel at full generality",
        fontsize=5.7, ha="left", va="center", color=INK)
ax.set_title("(b) the decision (Part 1)", fontsize=8.6, fontweight="bold")

# --------------------------------------------- (c) what "not both" closes
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.30, 7.00), 4.45, 2.65,
                            boxstyle="round,pad=0.10", fc="#f9e3e0",
                            ec=RED, lw=1.8))
ax.text(2.52, 9.20, "CLOSED", fontsize=7.4, ha="center", color=RED,
        fontweight="bold")
ax.text(2.52, 8.55, "the detector branch", fontsize=6.6, ha="center",
        color=INK, fontweight="bold")
ax.text(2.52, 7.65, "no instrument built, specified-\nfor-construction, or run under P4;\n"
        "no compute, no chassis run, no\nregistered quantity aimed at the\nfork's sentence",
        fontsize=5.8, ha="center", va="center", color=INK)

ax.add_patch(FancyBboxPatch((5.20, 7.00), 4.5, 2.65,
                            boxstyle="round,pad=0.10", fc="#eaf2f8",
                            ec=BLUE, lw=1.8))
ax.text(7.45, 9.20, "UNTOUCHED", fontsize=7.4, ha="center", color=BLUE,
        fontweight="bold")
ax.text(7.45, 8.55, "the chassis deployment", fontsize=6.6, ha="center",
        color=INK, fontweight="bold")
ax.text(7.45, 7.65, "keeps its rank, warrant, register:\nthe next compute (P1's follow-\n"
        "through, the M-instruments'\nself-reports); not P4's vehicle,\npreamble, or consolation",
        fontsize=5.8, ha="center", va="center", color=INK)

ax.text(5.0, 6.45, "the mode-flip - pre-registered as the only door:",
        fontsize=6.6, ha="center", color=INK, fontweight="bold")

ax.add_patch(FancyBboxPatch((0.30, 4.20), 3.15, 1.85,
                            boxstyle="round,pad=0.10", fc="#e8f4ec",
                            ec=GREEN, lw=2.0))
ax.text(1.87, 5.58, "EXTENSION", fontsize=7.0, ha="center",
        color=GREEN, fontweight="bold")
ax.text(1.87, 5.10, "(now)", fontsize=6.2, ha="center", color=GREEN)
ax.text(1.87, 4.60, "the derivation is\nP4's execution", fontsize=5.7,
        ha="center", va="center", color=INK)

ax.add_patch(FancyBboxPatch((5.20, 4.75), 4.5, 1.30,
                            boxstyle="round,pad=0.08", fc="#fdf3dd",
                            ec=GOLD, lw=1.6))
ax.text(7.45, 5.70, "FF-1: the floor total", fontsize=6.4, ha="center",
        color=GOLD, fontweight="bold")
ax.text(7.45, 5.12, "the theorem files; P4 closes as extension", fontsize=5.8,
        ha="center", va="center", color=INK)

ax.add_patch(FancyBboxPatch((5.20, 3.15), 4.5, 1.30,
                            boxstyle="round,pad=0.08", fc="#f9e3e0",
                            ec=RED, lw=1.6))
ax.text(7.45, 4.10, "FF-2: the floor broken", fontsize=6.4, ha="center",
        color=RED, fontweight="bold")
ax.text(7.45, 3.52, "a constructive detector exhibited at a named seam",
        fontsize=5.8, ha="center", va="center", color=INK)

ax.add_patch(FancyArrowPatch((3.52, 5.28), (5.12, 5.35), arrowstyle="-|>",
                             mutation_scale=11, color=GOLD, lw=1.5))
ax.add_patch(FancyArrowPatch((3.52, 4.42), (5.12, 3.95), arrowstyle="-|>",
                             mutation_scale=11, color=RED, lw=1.5))
ax.add_patch(FancyArrowPatch((7.45, 3.05), (7.45, 2.88), arrowstyle="-|>",
                             mutation_scale=10, color=RED, lw=1.3))

ax.add_patch(FancyBboxPatch((5.20, 0.60), 4.5, 2.20,
                            boxstyle="round,pad=0.08", fc="#f4f6f6",
                            ec=RED, lw=1.4))
ax.text(7.45, 2.38, "the flip, by right only", fontsize=6.4, ha="center",
        color=RED, fontweight="bold")
ax.text(7.45, 1.48, "FF-2 flips the mode to deployment\nby the failure's own content: the\n"
        "exhibited instrument re-registered\nbehind the chassis deployment;\n"
        "amendment to doc 91 only;\nno second door",
        fontsize=5.5, ha="center", va="center", color=INK)

ax.text(1.90, 3.20, "at no instant is P4 both:", fontsize=6.4, ha="center",
        color=INK, fontweight="bold")
ax.text(1.90, 2.25, "extension now; deployment only\nif the extension's own content\n"
        "forces the flip; never both at once",
        fontsize=5.9, ha="center", va="center", color=INK)
ax.text(1.90, 1.20, "(the surprise branch is inside\nthe theorem program, not beside it)",
        fontsize=5.7, ha="center", va="center", color=GRAY, fontstyle="italic")
ax.set_title("(c) what \"not both\" closes (Part 2)", fontsize=8.6,
             fontweight="bold")

# --------------------------------------------- (d) the pre-registered target
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.55, 8.45), 8.9, 1.35,
                            boxstyle="round,pad=0.10", fc="#fdf6ec",
                            ec=PURPLE, lw=2.2))
ax.text(5.0, 9.42, "the target: the Instrument-Form Bridge Constraint",
        fontsize=7.6, ha="center", color=PURPLE, fontweight="bold")
ax.text(5.0, 8.80, "Lemma 85 extended across all registerable families, assembled with Theorem II and Proposition 86",
        fontsize=5.4, ha="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 4.80), 9.35, 3.45,
                            boxstyle="round,pad=0.10", fc="white",
                            ec=GRAY, lw=1.2))
ax.text(0.62, 6.52, "(i)  registerable family: specified procedures, extensional inputs\n"
        "       (stream, architecture), family-relative ceiling - the seismograph\n"
        "       is the FIRST INSTANCE, not the counterexample (its readouts are\n"
        "       third-person: doc 88 proved it); exclusion of it is pre-forbidden\n"
        "\n"
        "(ii)  the extension: every instrument in every registerable family\n"
        "       computes an extensional function of (stream, architecture) - the\n"
        "       clause where FF-2 can fire, failure mode named in advance\n"
        "\n"
        "(iii) the assembly: readout distributions invariant across the two fork-\n"
        "       worlds (Thm II + Prop. 86) -> no positive reading is a further fact\n"
        "       -> the exhaustive reading's instrument-form floor is TOTAL",
        fontsize=5.7, ha="left", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 3.00), 9.35, 1.55,
                            boxstyle="round,pad=0.08", fc="#e8f4ec",
                            ec=GREEN, lw=1.6))
ax.text(5.0, 4.15, "the upgrade chain completes", fontsize=6.6, ha="center",
        color=GREEN, fontweight="bold")
ax.text(5.0, 3.50, "argument-form fact (docs 2-6)  ->  instrument-form fact, one channel\n"
        "(doc 88)  ->  instrument-form THEOREM, all registerable families (P4)",
        fontsize=5.9, ha="center", va="center", color=INK)

ax.add_patch(FancyBboxPatch((0.30, 0.55), 9.35, 2.25,
                            boxstyle="round,pad=0.10", fc="#f9e3e0",
                            ec=RED, lw=1.6))
ax.text(5.0, 2.40, "the caps, pre-committed as binding non-claims", fontsize=6.6,
        ha="center", color=RED, fontweight="bold")
ax.text(5.0, 1.45, "a floor is not a verdict: \"no instrument detects the further fact\" is not\n"
        "\"there is no further fact\" (Prop. 86 governs); both tribunal survivors stand\n"
        "(doc 90 unchanged); C4's one labeled jump remains the only crossing;\n"
        "the Camp-Marker up for the derivation session",
        fontsize=5.8, ha="center", va="center", color=INK)
ax.set_title("(d) the pre-registered target (Part 3)", fontsize=8.6,
             fontweight="bold")

fig.suptitle("P4 pre-registered (corpus doc 91): the impossibility - extension, not\n"
             "deployment - the fork killed by order, the target and its two forks bound\n"
             "before execution",
             fontsize=9.6, fontweight="bold")
fig.savefig(f"{D}/p4prereg_figure1.png", dpi=170)
print("wrote", f"{D}/p4prereg_figure1.png")
