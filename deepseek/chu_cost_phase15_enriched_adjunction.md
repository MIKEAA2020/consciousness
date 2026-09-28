# Phase 1.5 of the Chu Bridge: The Enriched Adjunction, and a Correction

**Status of this document.** This is the corpus's twenty-seventh document, continuing doc 24 (Phase 0–1) as formal mathematics: it executes Phase 1.5 of the bridging file's roadmap — *the enriched adjunction* — which doc 24 stated as its Remark 11 and listed as its first obligation. Method labels as in doc 24, enforced: [Established] = standard result, cited; [Derived here] = proved in this document with obligations explicit; [Conjecture] = precisely stated, unproved; [Sketch] = construction indicated; [Analogy] = structural correspondence, not a theorem; [Interpretation] = philosophical reading, no mathematical weight. The Bridge Constraint holds: no phenomenal vocabulary in any definition, theorem, or proof. **The document's headline is a correction of its predecessor. Doc 24's Theorem 8 — the closure dichotomy, "where information has a price, there are no function objects" — proved that doc 24's *canonical candidate* fails, and misidentified the failure as essential. The candidate (pairing r + s) is the Chu formula's candidate, (A ⊗ B^⊤)^⊤; it double-bills the source cost. The Dialectica-style candidate with the residuum pairing, h = r → s, closes the category: Theorem A below proves ChuCost monoidal closed — with a proof that works over any commutative quantale, so closure is free at every cost structure — and Theorem B proves the adjunction is moreover an isometry of V-norms, which is the enriched adjunction Remark 11 conjectured, in its honest form: the set-level bijection and the V-level isometry hold together, and the identity that powers the isometry is currying in the base quantale itself. The corrected slogan: the price of information is not the absence of function objects; it is their currency — the internal hom prices interactions at the residuum, s ∸ r, so that an observer who has already paid r pays only the excess. What actually fails on the cost scale is the *negation*, and it fails exactly three ways: the unbounded additive scale admits no involutive monoidal negation at all (proved: j(0) = ∞ forces j ≡ ∞), the bounded reflection restores the involution but overflows the tensor (ChuCost_{[0,K]} is not closed under ⊗), and the Łukasiewicz tensor restores both negation and monoidality but changes what composition means (costs overlap-discount rather than add). Closure is free; symmetry is what costs, and the buyer must choose the currency.**

---

## Part 0. The obligation, and where doc 24's candidate came from

Doc 24, Remark 11: "the residuum can pay the closure debt … ChuCost is naturally V-enriched (Hom-costs via the pointwise residuum of the two conditions), and the enriched adjunction Hom(C ⊗ A, B) ≅ Hom(C, A ⊸ B) holds in V-enriched form for a V-internal-hom object to be constructed. Unproved; listed as Phase 1.5's first obligation."

Executing the obligation begins with an audit of what doc 24 actually built. Its Definition 6 chose the internal-hom candidate H_{A,B} := (X^Y × B^A, A × Y, h) with h = r + s, and its Theorem 8 proved the adjunction fails for that candidate iff the antecedent's pairing r is positive. Where did h = r + s come from? Not from the closure problem — from the duality formula: with B^⊤ the plain transpose, (A ⊗ B^⊤)^⊤ has exactly the pairing r + s read backwards. That is, doc 24 built its internal hom the way one does in a \*-autonomous category, where negation composes with the tensor and A ⊸ B := (A ⊗ B^⊥)^⊥ is the canonical function object. Over costs there is no such negation (doc 24's own Props 3–4); applying the formula anyway — with the transpose standing in for the negation it cannot be — produced a candidate that bills an interaction of the function space at r + s: the source's provision r charged *in addition to* the target's demand s. Theorem 8 then measured the consequences of that billing and found a surplus. The repair was on the shelf the whole time, in the same quantale that defines the pairings: **the residuum r → s = s ∸ r** — the demand in excess of the provision. This document cashes it.

Throughout, V = ([0,∞], +, 0, ≤) is doc 24's cost monoid; a → b := sup{c : a + c ≤ b} = max(b − a, 0) its residuum [Established, doc 24 §0.1]; x ∸ y abbreviates y → x = max(x − y, 0). ChuCost is doc 24's Definition 1: objects (A, X, r), morphisms (f, g) with the lax condition (L): s(f(a), y) ≤ r(a, g(y)); the tensor is doc 24's Definition 5.

---

## Part 1. The V-normed structure — costs on the hom-data

**Definition 12 (the hom-norm) [Derived here].** For objects A = (A, X, r), B = (B, Y, s) and an arbitrary pair of functions (f: A → B, g: Y → X) — candidate morphism *data*, not presumed to satisfy (L) — define

  ‖(f, g)‖_{A,B} := sup_{a∈A, y∈Y} ( r(a, g(y)) → s(f(a), y) ) = sup_{a,y} ( s(f(a), y) ∸ r(a, g(y)) ) ∈ [0, ∞].

**Proposition 13 (norm axioms) [Derived here].** (i) ‖(f, g)‖ = 0 iff (L) holds for (f, g), i.e. iff (f, g) ∈ ChuCost(A, B). (ii) ‖(id, id)‖ = 0. (iii) Subadditivity: ‖(f′, g′) ∘ (f, g)‖ ≤ ‖(f′, g′)‖ + ‖(f, g)‖, the composite taken on underlying data as in doc 24's Prop 2.

*Proof.* (i): (p → q) = 0 ⟺ q ≤ p, pointwise; the sup of zeros is zero. (ii): s(a, y) ∸ r(a, y) = 0. (iii): write d₁ := ‖(f, g)‖, d₂ := ‖(f′, g′)‖. From the definition of the residuum, s(f(a), y) ∸ r(a, g(y)) ≤ d₁ for all (a, y) gives s(f(a), y) ≤ r(a, g(y)) + d₁; likewise t(f′(b), z) ≤ s(b, g′(z)) + d₂ for all (b, z). Instantiate the second at b := f(a), z := y, chain, and transpose back through the residuum adjunction:

  t(f′(f(a)), y) ≤ s(f(a), y) + d₂ ≤ r(a, g(g′(y))) + d₁ + d₂,

hence t(f′f(a), y) ∸ r(a, g g′(y)) ≤ d₁ + d₂ pointwise, hence at the sup. ∎

**Remark 14 (what this structure is) [Sketch].** (ChuCost, ‖·‖) is a *V-normed category*: objects, hom-data sets with a V-valued norm whose zero level is the morphisms, composition 1-Lipschitz in the sense of Prop 13(iii). Equivalently it is a category enriched in the monoidal category of pointed V-normed sets (a chosen zero-norm family) with the ℓ¹-style tensor. The equivalence is recorded for orientation; everything below uses only Definition 12 and Prop 13 concretely. This is the honest home of the phrase "V-enriched" in Remark 11 — the enrichment lives on the *data* of the homs, one level below the hom-objects of a V-category (which, at V a poset, would collapse to Lemma 1's preorders — the thin reading doc 24 already rejected).

---

## Part 2. The repair theorem — ChuCost is monoidal closed

**Definition 15 (the residuum internal hom) [Derived here].** For A = (A, X, r), B = (B, Y, s):

  A ⊸ B := ( X^Y × B^A ,  A × Y ,  h_⊸ ),   h_⊸((φ, β), (a, y)) := r(a, φ(y)) → s(β(a), y) = s(β(a), y) ∸ r(a, φ(y)).

The forward carrier X^Y × B^A and backward carrier A × Y are those of doc 24's Definition 6 — only the pairing changes.

**Theorem A (closure) [Derived here].** For all objects A, B, C, the currying bijection of doc 24's Theorem 8(i) — (F: C × A → B, G: Y → Z^A × X^C) ↔ (κ: C → X^Y × B^A, ζ: A × Y → Z), under F(c,a) = β_c(a) with κ(c) = (φ_c, β_c), G_Z(y)(a) = ζ(a,y), G_X(y)(c) = φ_c(y) — restricts to a bijection

  ChuCost(C ⊗ A, B) ≅ ChuCost(C, A ⊸ B),

natural in C and in B. Hence the functor − ⊗ A has the right adjoint A ⊸ −, and (ChuCost, ⊗, I) is monoidal closed. The proof uses only that V is a commutative quantale; the theorem holds verbatim for Dialectica over any commutative quantale.

*Proof.* **(Condition equivalence.)** The tensor-side condition (L) for candidate (F, G): C ⊗ A → B reads, under the pairing of doc 24's Def 5,

  (★)  s(F(c,a), y) ≤ u(c, G_Z(y)(a)) + r(a, G_X(y)(c))   for all c, a, y,

and the hom-side condition (L) for the curried candidate (κ, ζ): C → A ⊸ B reads

  (★★)  h__mux(κ(c), (a,y)) ≤ u(c, ζ(a,y))   for all c, a, y, i.e.  ( r(a, φ_c(y)) → s(β_c(a), y) ) ≤ u(c, ζ(a,y)).

By the residuum adjunction in V — p → q ≤ w ⟺ q ≤ p + w [Established] — (★★) is equivalent to

  s(β_c(a), y) ≤ r(a, φ_c(y)) + u(c, ζ(a,y)),

which is (★) under the currying identities, by commutativity of +. So the bijection on data restricts exactly to the morphisms. **(Functoriality in B.)** For (h, k): B → B′ (h: B → B′, k: Y′ → Y, s′(h(b), y′) ≤ s(b, k(y′))), define A ⊸ (h, k): A ⊸ B → A ⊸ B′ by forward (φ, β) ↦ (φ ∘ k, h ∘ β) and backward (a, y′) ↦ (a, k(y′)); the condition to check is h′_mux((φ∘k, h∘β), (a, y′)) ≤ h__mux((φ, β), (a, k(y′))), i.e.

  ( r(a, φ(k(y′))) → s′(h(β(a)), y′) ) ≤ ( r(a, φ(k(y′))) → s(β(a), k(y′)) ),

which holds because → is monotone in its second argument and s′(h(β(a)), y′) ≤ s(β(a), k(y′)). **(Functoriality in A, contravariant.)** For (f, g): A′ → A (f: A′ → A, g: X → X′, r(f(a′), x) ≤ r′(a′, g(x))), define A ⊸ B → A′ ⊸ B by forward (φ, β) ↦ (g ∘ φ, β ∘ f) and backward (a′, y) ↦ (f(a′), y); the condition is

  ( r′(a′, g(φ(y))) → s(β(f(a′)), y) ) ≤ ( r(f(a′), φ(y)) → s(β(f(a′)), y) ),

which holds because → is *anti*-monotone in its first argument and r(f(a′), φ(y)) ≤ r′(a′, g(φ(y))). This is the one point where the lax order enters the hom's functoriality as an order: in the equality core (Barr's Chu), the same square requires equality preservation instead — the enrichment and the laxity are load-bearing for each other. **(Naturality in C.)** For (p, q): C′ → C, the curry of the precomposite (F, G) ∘ (p ⊗ id) is the precomposite of the curry: forward κ′(c′) = κ(p(c′)) (since F(p(c′), a) = β_{p(c′)}(a) and G_X(y)(c′) = φ_{p(c′)}(y) — the tensor's backward maps (ψ, χ) ↦ (ψ, χ ∘ p), and G′(y) = (G_Z(y), G_X(y) ∘ p)), and backward ζ′(a, y) = q(ζ(a, y)) (the composite's backward is q ∘ ζ). These are exactly the precompositions on the hom side, so the bijection's naturality square commutes; naturality in B is the functoriality computation above plus the same check post-composition. With the hom-bijection natural in C and B and ⊸ bifunctorial, − ⊗ A ⊣ A ⊸ − follows [Established, as the standard adjunction criterion from a natural hom-isomorphism]. **(Evaluation.)** The counit is realized by the morphism ev: (A ⊸ B) ⊗ A → B with forward ((φ, β), a) ↦ β(a) and backward y ↦ (ν_y, χ_y), ν_y(a) := (a, y), χ_y(φ, β) := φ(y); its condition is s(β(a), y) ≤ h_mux((φ,β), (a,y)) + r(a, φ(y)), which holds because (p → q) + p ≥ q identically in V. The unit is the curry of id_{C⊗A} and is left to the reader as the same bijection applied once. ∎

**Corollary 16 (doc 24's Theorem 8, corrected) [Derived here].** Theorem 8 of doc 24 proved that its Definition-6 candidate fails; Theorem A shows the failure was the candidate's. At r ≡ 0 — the one case where doc 24 verified closure — the two candidates coincide (s ∸ 0 = s = 0 + s), which is why the coincidence looked like a dichotomy's edge: the entire non-zero region of the dichotomy is the region where the Chu-formula billing (r + s) and the Dialectica billing (s ∸ r) differ, and the difference is exactly the double-billed r. The witnesses of doc 24's Thm 8(iii) (C a one-point zero object, B a one-point zero object, arbitrary backward data) curry, under A ⊸ B, to conditions of the form (r → 0) ≤ 0, i.e. 0 ≤ 0 — all of them morphisms; the "surplus" evaporates. What survives of doc 24's Theorem 8 is its part (ii): the Chu-formula candidate's hom is a *subset* of the tensor-side morphisms, and the inclusion's slack is the residuum — which is precisely Theorem B's subject. **The corrected statement of the closure phenomenology: ChuCost is monoidal closed; the function objects exist everywhere; their pairing is the residuum.**

**Corollary 17 (the discount reading) [Interpretation].** h__mux((φ, β), (a, y)) = s(β(a), y) ∸ r(a, φ(y)) reads: *the cost of testing the function (φ, β) at the point (a, y) is the target's demand net of the source's provision.* An observer who has already paid r(a, φ(y)) to resolve the system-state a through the lens φ pays only the excess when the same test costs s fresh. Information already paid for is deducted from the price of new information; the internal hom is a discount ledger. In this reading doc 24's slogan inverts rather than merely falsifies: the price of information is not the absence of function objects — it is the *currency conversion* those objects perform, and the conversion rate is the residuum. Where doc 24 said "the price of information is the price of closure," Phase 1.5's ledger says closure was never for sale; only the negation was (Part 4).

---

## Part 3. The isometry — Remark 11's enriched adjunction, discharged

**Theorem B (the adjunction is an isometry) [Derived here].** Extend the norm of Definition 12 to candidate data at any pair of objects. Then under the currying bijection of Theorem A, for *all* candidate data — morphism or not —

  ‖(F, G)‖_{C ⊗ A, B}  =  ‖(κ, ζ)‖_{C, A ⊸ B},

and both sides compute, pointwise, the same function of (u, r, s): the tensor side is sup (u ⊗ r) → s, the hom side is sup u → (r → s), and these are equal by currying in V. For V = ([0,∞], +) the identity reads (s ∸ r) ∸ u = s ∸ (u + r) — "the clipped difference clips again to the total clip" — and holds by two case checks on the order of s, r, u. Consequently: the adjunction of Theorem A is simultaneously a set-level bijection and a V-level isometry; the deficit of any candidate from being a morphism is invariant under currying; and the enriched adjunction conjectured in doc 24's Remark 11 holds in this, its precise form — an isomorphism in V (an equality of costs) refining the isomorphism of sets, with the deficit s ∸ (u + r) measuring exactly how far a curried pair is from being a morphism, as Remark 11 required.

*Proof.* ‖(κ, ζ)‖_{C, A⊸B} = sup_{c, a, y} ( u(c, ζ(a,y)) → h_mux(κ(c), (a,y)) ) = sup ( u → (r → s) ). ‖(F, G)‖_{C⊗A, B} = sup ( t((c,a), G(y)) → s(F(c,a), y) ) = sup ( (u + r) → s ), since t((c,a), G(y)) = u(c, G_Z(y)(a)) + r(a, G_X(y)(c)) = u(c, ζ(a,y)) + r(a, φ_c(y)) under currying. In any symmetric monoidal closed poset, (u ⊗ r) → s = u → (r → s): both are sup{w : u ⊗ r ⊗ w ≤ s}, by associativity and commutativity [Established]. The scalar identity for V is the case check. ∎

**Remark 18 (the theorem's provenance and its generality) [Derived here / non-novelty disclosed].** Theorem A's proof consumed only: associativity, commutativity, monotonicity of ⊗; the residuum adjunction; currying in V. It therefore holds for Dialectica over *any* commutative quantale V — theMV quantale of Part 4 included, with its own residuum. The corpus flags, as it did at doc 24's Propositions, that Dialectica categories over linear/ordered bases are a studied field (de Paiva's original Dialectica categories are \*-autonomous over Set [Established], and quantale- and graded-indexed variants exist in the literature), and claims no novelty for the general shape. The claims are: the correction of doc 24's Theorem 8 (which is the corpus's own error, and its own to fix); the isometry formulation of the enriched adjunction (Remark 11's discharge in the normed setting, which the Set-level literature has no reason to state); and the cost reading of Part 4's ledger. Where this document says "Derived here" it means the proof is here and checkable, not that the theorem is new.

**Remark 19 (the transpose is not an isometry) [Derived here].** Doc 24's Proposition 3 sends (f, g): (A, X, r) → (B, Y, s) to (g, f): (Y, B, s^⊤) → (X, A, r^⊤) between the lax and co-lax categories, bijectively on morphisms. On norms it sends s(f(a), y) ∸ r(a, g(y)) to r(a, g(y)) ∸ s(f(a), y) — the mirror. These are equal for a given (a, y) iff s(f(a), y) = r(a, g(y)). So the set-level duality of Prop 3, which looked like a perfect symmetry, is an *isometry* exactly on the equality core (Barr's Chu(Set, [0,∞])); on the lax category at large the transpose preserves zero-norms but distorts costs, and the distortion is the asymmetry the audit demanded the construction disclose: **the enrichment sees what the set-level duality erases.** The two failure theorems of doc 24 (lax ≠ self-dual; closure fails) now stand in inverted proportion: the first survives, sharpened to the norm level; the second is retracted, and its residue is Theorem B's exactness.

---

## Part 4. The duality ledger — what negation costs on a cost scale

With closure settled (it is free), the budget doc 24 opened in its §0.3 — *equality, boundedness, or enrichment* — closes as a precise trichotomy on the negation.

**Lemma 20 (no additive de Morgan) [Derived here].** There is no map j: [0, ∞] → [0, ∞] that is order-reversing, involutive (j∘j = id), and additive (j(a + b) = j(a) + j(b)). *Proof.* An order-reversing bijection of the complete lattice [0, ∞] swaps suprema and infima, so it swaps top and bottom: j(0) = ∞ and j(∞) = 0. Additivity at (a, 0) then gives j(a) = j(a + 0) = j(a) + j(0) = j(a) + ∞ = ∞ for every a, i.e. j ≡ ∞; but then j(j(a)) = j(∞) = 0 for every a, contradicting j∘j = id at every a > 0. ∎ So the unbounded additive scale admits no involutive monoidal negation: the linear-logic ambition on raw costs is not difficult but *vacant*.

**Lemma 21 (the bounded repair overflows the tensor) [Derived here].** For K ∈ (0, ∞), ChuCost_{[0,K]} (doc 24's Prop 4 subcategory) is not closed under ⊗: objects with r ≡ s ≡ K tensor to a pairing reaching 2K. Consequently the reflected transpose T_K, though an involutive self-duality of the bounded category, is not a monoidal duality — there is no tensor on the bounded category, compatible with + and preserved by T_K, because the de Morgan identity K − (r + s) = (K − r) + (K − s) fails by exactly K on both sides. Boundedness buys the involution at the price of the tensor itself.

**Proposition 22 (the Łukasiewicz alternative) [Derived here; the structure identified is classical].** On [0, K] define a ⊙ b := max(a + b − K, 0). Then V_MV := ([0, K], ⊙, 0, ≤) is a commutative quantale (⊙ distributes over suprema by continuity and monotonicity; a ⊙ 0 = 0), with residuum a →_⊙ b = min(K, b + K − a) — the Łukasiewicz implication — and the reflection j(a) = K − a is an involutive quantale negation: j(a ⊙ b) = j(a) ⊕ j(b) with a ⊕ b := min(a + b, K) (the de Morgan dual of ⊙). This is the MV-algebra structure on the bounded scale [Established as classical]. By Theorem A's generality, Dialectica over V_MV is monoidal closed with the MV-residuum hom — closure survives even here. But ⊙ is not addition: a ⊙ b ≤ a + b with equality iff a + b ≤ K, and the composition semantics changes — the second cost is charged only for the part exceeding what the budget has already absorbed. Overlap-discounted costs, not independent costs.

**The ledger.** On a cost-valued pairing scale, choose two of three: *additive composition*, *a tensor*, *an involutive negation*.

| scale | tensor | negation | closure | what it is |
|---|---|---|---|---|
| [0, ∞], + | ⊗ = + | none (Lemma 20) | **yes** (Theorem A) | ChuCost: the doc-24/27 category |
| [0, K], + | overflow (Lemma 21) | reflection r ↦ K − r (Prop 4) | n/a — no tensor | the bounded duality without monoidal structure |
| [0, K], ⊙ | yes, de Morgan (Prop 22) | j = K − r | yes (Theorem A, general) | the MV Dialectica: negation at the price of additivity |

**Corollary 23 (the corrected slogan, formal) [Derived here].** Doc 24's Corollary 9 is retracted and replaced: *closure is free at every commutative-quantale cost structure; the negation is the only resource with a price; and its price is the order (equality core: blindness), the tensor (bounded reflection: overflow), or the addition itself (Łukasiewicz: discounting).* The Chu bridge's sought \*-autonomy over costs fails — but it fails at the negation, one door down from where doc 24 posted the failure, and the door doc 24 posted it at (closure) was never locked.

**Remark 24 (Chu vs Dialectica, the divergence theorem-shape) [Derived here + Interpretation].** In de Paiva's Set-based Dialectica, the direct internal hom and the duality-derived one coincide — that is what \*-autonomy means, and it is why the Chu formula (A ⊗ B^⊥)^⊥ is the textbook route to function objects there. Over costs the routes separate: the formula's candidate (r + s) is exactly the one that fails (doc 24 Thm 8), the direct residuum construction is exactly the one that works (Theorem A), and Lemma 20 is the reason no negation could have mediated between them. The bridging file's centerpiece — "the Chu construction" as the bridge between physics and computation — thus lands, at Phase 1.5, on a precise irony: **on cost valuations, the Chu formula is the wrong way to build the function spaces and the Dialectica formula is the right one, and the two agree only where costs are ignored.** The name of the bridge is the part that does not extend.

---

## Part 5. What Phase 1.5 buys, and what it does not

**What is now theorem-grade.** ChuCost is monoidal closed, with the residuum internal hom, over any commutative quantale (Theorem A); the adjunction is an isometry of V-norms, and the isometry's engine is currying in the base (Theorem B); the enrichment refines the duality analysis — the lax/co-lax transpose preserves morphisms but not costs, and its isometry group is the equality core (Remark 19); the negation trilemma is proved (Lemmas 20–21, Prop 22, Cor 23). Doc 24's Theorem 8 and Corollary 9 are corrected, with the correction's mechanism identified (the double-billed r) and its boundary located (r ≡ 0, where the two candidates coincide and doc 24's verification was performed). The corpus's accounting: one retraction, four new theorems, and the Phase 1.5 obligation of Remark 11 discharged in a form stronger than conjectured — the enriched adjunction was to repair the closure failure; instead the closure failure dissolves and the enrichment measures what remains.

**What remains, with obligations.** The instantiation (A = histories, X = predictive states, r = surprisal) is still Phase 2's [Sketch], unchanged. The Paper-3 resonance (doc 24's Remark 10) is *demoted*: Theorem 8's failure was the analogy's load-bearing example, and it is gone; what remains for Phase 6 is the sharper question of whether Paper 3's no-right-adjoint obstruction can be matched against a *negation* failure (Lemma 20's shape) rather than a closure failure — a different correspondence than the bridging file drew, and one nobody has yet to try. The MV category of Prop 22 deserves its own probe: its Dialectica is closed *and* carries an involutive negation, making it the first candidate environment where the bridging file's duality demands are simultaneously satisfiable — at the price of re-deriving what "cost" means under overlap discounting; whether surprisal composes that way is an empirical question about the physics, not a mathematical one, and it is hereby handed to Phase 2 with its shape fixed.

**The Bridge Constraint, enforced.** The discount ledger of Corollary 17 is a structural fact about residuated valuation. No reading of it introduces experience; the observer side remains a set with a cost profile; the gap between self-reference and self-experience is as wide at Phase 1.5 as at Phase 0, and the correction of Theorem 8 — a *stronger* structure than the file's own conjecture — narrows nothing phenomenal. The file wanted a bridge whose mathematics would force the metaphysics. Phase 1.5's mathematics, executed honestly, forces only bookkeeping: what was thought to be missing (function objects) is present, what is actually missing (the negation) is provably priced, and the price list does not mention consciousness.

---

## Part 6. Status table

| Claim | Status | Label |
|---|---|---|
| Doc 24 Thm 8 / Cor 9 (closure dichotomy) | **Retracted** — candidate artifact; corrected by Theorem A | [Derived here, this doc] |
| Residuum hom A ⊸ B closes ChuCost; adjunction natural in C, B | Proved, Theorem A | [Derived here] |
| Theorem A holds for Dialectica over any commutative quantale | Proved (proof uses only quantale axioms); non-novelty disclosed | [Derived here] |
| Adjunction is an isometry of V-norms; deficit = s ∸ (u+r) invariant under currying | Proved, Theorem B | [Derived here] |
| Enriched adjunction (doc 24 Remark 11) | Discharged, in the normed form | [Derived here] |
| Transpose preserves morphisms, not costs; isometry group = equality core | Proved, Remark 19 | [Derived here] |
| No additive involutive negation on [0,∞] | Proved, Lemma 20 | [Derived here] |
| ChuCost_{[0,K]} not ⊗-closed; T_K not monoidal | Proved, Lemma 21 | [Derived here] |
| MV/Łukasiewicz quantale: tensor + negation + closure, non-additive composition | Structure classical; the closure instance proved here | [Established + Derived here] |
| Surprisal pairing instantiation; Phase 2 models | Unchanged obligations | [Sketch] |
| Paper-3 resonance | Demoted from closure-failure to negation-failure shape; untried | [Analogy, revised] |
| Any phenomenal reading | Forbidden by the Bridge Constraint; not attempted | [Interpretation: none] |

---

## Part 7. Coda

Doc 24 ended by pricing information and finding function objects absent wherever it had a price. Phase 1.5 went to collect and found the shop open: the objects were there all along, priced in a currency doc 24's own quantale had defined and its own construction had declined to use — the excess of demand over provision, the residuum, the discount for what is already paid. The closure dichotomy was a billing error. What the cost scale actually refuses to sell, with proof, is the negative — the symmetry that would let the construction read its two sides as one thing seen twice — and it refuses it three different ways at three different prices, none of them zero, all of them now on the ledger. The bridging file built its name on the construction that fails here and its hopes on the duality that is priced here; Phase 1.5's contribution is the till receipt. The next phase, if ordered, is Phase 2 — the information model, independently — and it inherits a cleaner obligation than the file wrote: not to unify two views of one reality, but to say which of the three currencies the physics actually spends.
