#!/usr/bin/env python3
"""Figure for corpus doc 89 (ChuCost Phase 17, the residue theorem: the
extensional quotient): the quotient itself, Theorem II's clauses, the
convergence of P1 and P3 on the one equation, and C4's warrant re-filed
as one labeled jump. English labels, constrained_layout, house
palette."""
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

# ---------------------------------------- (a) the quotient
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(5.0, 9.5, "the world's system", fontsize=7.8, ha="center",
        color=INK, fontweight="bold")
ax.add_patch(Circle((5.0, 7.4), 1.5, fc="#fef9e7", ec=GOLD, lw=2.4))
ax.text(5.0, 7.75, "the performance\n(the lived row)", fontsize=6.6,
        ha="center", color=GOLD, fontweight="bold")
ax.text(5.0, 6.9, "the invocation", fontsize=6.4, ha="center", color=GOLD,
        fontstyle="italic")

ax.add_patch(FancyArrowPatch((5.0, 5.75), (5.0, 4.35),
                             arrowstyle="-|>", mutation_scale=14,
                             color=RED, lw=2.2))
ax.text(6.6, 5.05, "FORMALIZE\n= QUOTIENT", fontsize=7.0, ha="center",
        color=RED, fontweight="bold")
ax.text(6.6, 4.35, "(to formalize is to\nquotient - Prop. 86's engine)",
        fontsize=5.9, ha="center", color=GRAY, fontstyle="italic")

ax.add_patch(FancyBboxPatch((1.3, 1.15), 7.4, 2.95,
                            boxstyle="round,pad=0.12", fc="white",
                            ec=BLUE, lw=2.0))
ax.text(5.0, 3.5, "the grammar's object  (A, X, r)", fontsize=7.4,
        ha="center", color=BLUE, fontweight="bold")
ax.text(5.0, 2.55, "the structured set: states x observations, the evaluation\n"
        "r: A x X -> K - the isomorphism class plus the interaction\n"
        "history is ALL any reading extracts (Theorem II(i))",
        fontsize=6.3, ha="center", color=INK)
ax.text(5.0, 1.55, "the pointing, the copy, the performance: exiled at the "
        "foundation", fontsize=6.3, ha="center", color=RED, fontstyle="italic")
ax.text(5.0, 0.45, "the invocation is UNPOSED, not unread - never a column "
        "of the object", fontsize=6.6, ha="center", color=RED,
        fontweight="bold")
ax.set_title("(a) the extensional quotient (Theorem II)", fontsize=8.6,
             fontweight="bold")

# ---------------------------------------- (b) Theorem II, clause by clause
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
clauses = [
    (7.35, "(i) every reading is isomorphism-invariant", BLUE,
     "readings are composites of morphisms and evaluations;\n"
     "morphisms preserve evaluations, composition is functorial\n"
     "- the value transports along every isomorphism"),
    (4.55, "(ii) the invocation is not object-data", PURPLE,
     "the pointing is presentation-data (symmetries exchange it);\n"
     "and the grammar has NO PERFORMANCE-TYPE - r(a,x) is\n"
     "extensional in both arguments, and no term forms the\n"
     "distinction 'tabulated vs performed'"),
    (1.75, "(iii) corollary - the unread column, stated", RED,
     "external readings cannot be AIMED at the invocation:\n"
     "the aimable targets are the isomorphism-invariant facts,\n"
     "and the invocation is a fact about the world's\n"
     "INSTANTIATION of the object - what the quotient removed"),
]
for (by, lab, col, sub) in clauses:
    ax.add_patch(FancyBboxPatch((0.5, by - 1.25), 9.0, 2.25,
                                boxstyle="round,pad=0.10",
                                fc="white", ec=col, lw=1.8))
    ax.text(5.0, by + 0.58, lab, fontsize=7.5, ha="center",
            color=col, fontweight="bold")
    ax.text(5.0, by - 0.38, sub, fontsize=6.2, ha="center", color=INK)
ax.text(5.0, 9.5, "THEOREM II - named by alphabetical succession after HH;\n"
        "the roman numeral for the two columns, filed as coincidence",
        fontsize=7.0, ha="center", color=INK, fontstyle="italic")
ax.set_title("(b) the wall as a property of the grammar", fontsize=8.6,
             fontweight="bold")

# ---------------------------------------- (c) the convergence
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.6, 5.3), 4.0, 2.6,
                            boxstyle="round,pad=0.12", fc="#f0f6fc",
                            ec=BLUE, lw=2.0))
ax.text(2.6, 7.4, "P1 (doc 88) - the instrument", fontsize=7.2,
        ha="center", color=BLUE, fontweight="bold")
ax.text(2.6, 6.25, "Lemma 85: every reading computes\na function of stream + "
        "architecture;\nthe index enters only through the stream",
        fontsize=6.3, ha="center", color=INK)
ax.add_patch(FancyBboxPatch((5.4, 5.3), 4.0, 2.6,
                            boxstyle="round,pad=0.12", fc="#f9f0fc",
                            ec=PURPLE, lw=2.0))
ax.text(7.4, 7.4, "P3 (doc 89) - the theorem", fontsize=7.2,
        ha="center", color=PURPLE, fontweight="bold")
ax.text(7.4, 6.25, "Theorem II: every reading is\nisomorphism-invariant; "
        "the invocation\nis not even poseable", fontsize=6.3,
        ha="center", color=INK)
ax.add_patch(FancyArrowPatch((4.7, 6.6), (5.3, 6.6),
                             arrowstyle="<|-|>", mutation_scale=12,
                             color=GRAY, lw=1.6))
ax.add_patch(FancyArrowPatch((2.6, 5.2), (5.0, 4.1),
                             arrowstyle="-|>", mutation_scale=12,
                             color=BLUE, lw=1.5))
ax.add_patch(FancyArrowPatch((7.4, 5.2), (5.0, 4.1),
                             arrowstyle="-|>", mutation_scale=12,
                             color=PURPLE, lw=1.5))
ax.add_patch(FancyBboxPatch((0.9, 2.35), 8.2, 1.6,
                            boxstyle="round,pad=0.14", fc="#fdf6ec",
                            ec=RED, lw=2.6))
ax.text(5.0, 3.4, "READING = EXTENSIONALIZING", fontsize=10.5, ha="center",
        color=RED, fontweight="bold")
ax.text(5.0, 2.75, "the wall's single formal name - one equation, two faces",
        fontsize=6.6, ha="center", color=INK)
ax.text(5.0, 1.55, "every campaign's wall is this equation in one register: "
        "premise-isolation (proof is\nextension-preserving); the charter "
        "refusal (to operationalize is to extensionalize); the third person\n"
        "(every reading a registration at the reader's boundary); "
        "projection-not-traversal (functoriality)", fontsize=6.1,
        ha="center", color=INK)
ax.text(5.0, 0.55, "the two coins (unit position, branch geometry) were "
        "properties of homes; the equation is the property of the MINT",
        fontsize=6.3, ha="center", color=GRAY, fontstyle="italic")
ax.set_title("(c) the convergence: one order, two strokes (Part 4)",
             fontsize=8.6, fontweight="bold")

# ---------------------------------------- (d) the warrant re-filed
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.7, 4.6), 8.6, 2.3,
                            boxstyle="round,pad=0.12", fc="#eef7ef",
                            ec=GREEN, lw=2.2))
ax.text(5.0, 6.35, "GRAMMAR-GRADE (proved)", fontsize=8.0, ha="center",
        color=GREEN, fontweight="bold")
ax.text(5.0, 5.35, "the invocation is unposed in any extensional grammar\n"
        "(Theorem II); no grammar decides the residue (Proposition 86)\n"
        "- doc 6's sentence at its highest generality: no grammar reads\n"
        "what its ontology quotiented away", fontsize=6.4, ha="center",
        color=INK)
ax.add_patch(FancyArrowPatch((5.0, 4.5), (5.0, 3.45),
                             arrowstyle="-|>", mutation_scale=13,
                             color=GOLD, lw=2.2))
ax.text(6.5, 3.95, "THE ONE\nLABELED JUMP", fontsize=7.0, ha="center",
        color=GOLD, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.7, 0.95), 8.6, 2.3,
                            boxstyle="round,pad=0.12", fc="#fdf2f0",
                            ec=RED, lw=2.2))
ax.text(5.0, 2.7, "INTERPRETATION (the fork, C5 - both edges filed)",
        fontsize=8.0, ha="center", color=RED, fontweight="bold")
ax.text(5.0, 1.7, "exhaustive: nothing more to the system than the quotient "
        "- the map IS the answer\nfurther-fact: the residue exists - the map "
        "is the complete audit of the approach", fontsize=6.4,
        ha="center", color=INK)
ax.text(5.0, 0.45, "where the superposition blurred four jumps, there is "
        "now ONE - isolated, named, priced", fontsize=6.4, ha="center",
        color=GRAY, fontstyle="italic")
ax.text(5.0, 9.4, "C4's warrant, re-filed by amendment", fontsize=8.4,
        ha="center", color=INK, fontweight="bold")
ax.text(5.0, 8.55, "both pre-registered branches fired, at separated "
        "levels: weak form derivable, strong form underivable",
        fontsize=6.6, ha="center", color=INK)
ax.set_title("(d) the position after Phase 17 (Part 5)", fontsize=8.6,
             fontweight="bold")

fig.suptitle("The residue theorem (corpus doc 89, ChuCost Phase 17): the "
             "extensional quotient - the wall proved as the grammar's type "
             "system, the residue carried by the interpretation, and the two "
             "strokes of the order converging on READING = EXTENSIONALIZING",
             fontsize=9.4, fontweight="bold")
fig.savefig(f"{D}/residue_figure1.png", dpi=170)
print("wrote", f"{D}/residue_figure1.png")
