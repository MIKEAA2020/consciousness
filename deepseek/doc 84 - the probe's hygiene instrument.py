#!/usr/bin/env python3
"""MV probe hygiene instrument (doc 84, Part 3): brute-force verification of the
top-column collapse mechanism on finite toy objects.

Mechanism to verify (the finite shadow of Theorem EE):
(M1) MV home ([0,K], od = max(0, r+s-K), unit cost K): for ANY object X with a
     top-cost column b0 (r_X(a,b0)=K for all a) and ANY object Y, EVERY function
     f: A_X -> A_Y makes (f, const_b0) a morphism X -> Y  [the collapse flood].
     Brute force: all |A_Y|^|A_X| functions pass.
(M2) Additive mirror ([0,inf), +, unit cost 0): the same collapse with any column
     b1 does NOT trivialize -- there exist f failing the condition (the additive
     home's unit sits at the bottom; no column dominates).
(M3) Uniqueness bookkeeping: in the MV toy, the top column is exactly the set of
     columns with r(a,b)=K for all a -- count them (the equivalence forces 1).

No cardinality claims are tested here (finite sets cannot see c vs 2^c);
the cardinality step of Theorem EE is ZFC arithmetic stated in the document.
"""
import itertools

K = 10  # the MV top cost

def od(r, s):
    """Lukasiewicz tensor on costs: max(0, r+s-K)."""
    return max(0, r + s - K)

# ---------------- MV toy world ----------------
# X = (A_X, B_X, r_X): A_X = {0,1,2}; B_X = {b0 (top), b1 (structure)}
A_X = [0, 1, 2]
B_X = ["b0", "b1"]
r_X = {("b0", a): K for a in A_X}          # the top column
r_X.update({("b1", a): (K - 2) if a != 2 else 1 for a in A_X})  # a structure column

# Y = (A_Y, B_Y, r_Y): same shape, different structure column
A_Y = [0, 1, 2]
B_Y = ["c0", "c1"]
r_Y = {("c0", a): K for a in A_Y}
r_Y.update({("c1", a): (a + 3) % K for a in A_Y})  # costs 3,4,5 -- some < K

def is_morphism_MV(f, g, r_src, r_tgt, A_src, B_tgt):
    """Lax condition (MV): r_tgt(f(a), b') <= r_src(a, g(b')) for all a, b'."""
    return all(r_tgt[(g[bp], f[a])] <= r_src[(g[bp], a)] if False else
               r_tgt[(bp, f[a])] <= r_src[(g[bp], a)]
               for a in A_src for bp in B_tgt)

# (M1) the collapse flood: g = const b0
flood = 0
total_f = 0
for f in itertools.product(A_Y, repeat=len(A_X)):   # all functions A_X -> A_Y
    total_f += 1
    fmap = {a: f[i] for i, a in enumerate(A_X)}
    g = {b: "b0" for b in B_Y}                       # the collapse
    if is_morphism_MV(fmap, g, r_X, r_Y, A_X, B_Y):
        flood += 1
print(f"(M1) MV collapse flood: {flood}/{total_f} functions f admissible with g=b0 "
      f"-> {'VERIFIED' if flood == total_f else 'FAILED'}")

# (M3) the top columns of X: exactly {b0}?
tops = [b for b in B_X if all(r_X[(b, a)] == K for a in A_X)]
print(f"(M3) top-cost columns of X: {tops} -> "
      f"{'VERIFIED (exactly one)' if tops == ['b0'] else 'FAILED'}")

# ---------------- Additive mirror toy ----------------
# Same carriers; costs in [0, inf); the unit's cost is 0 (bottom).
# Collapse with ANY fixed column b1: does every f pass?  (Should NOT.)
rX_add = {("b0", a): 0 for a in A_X}                 # bottom column (the additive unit's shape)
rX_add.update({("b1", a): 1 for a in A_X})           # a finite-cost column
rY_add = {("c0", a): 0 for a in A_Y}
rY_add.update({("c1", a): 5 for a in A_Y})           # target costs can EXCEED source columns

def is_morphism_add(f, g, r_src, r_tgt, A_src, B_tgt):
    return all(r_tgt[(bp, f[a])] <= r_src[(g[bp], a)]
               for a in A_src for bp in B_tgt)

for bcol in ["b0", "b1"]:
    ok = 0
    for f in itertools.product(A_Y, repeat=len(A_X)):
        fmap = {a: f[i] for i, a in enumerate(A_X)}
        g = {b: bcol for b in B_Y}
        if is_morphism_add(fmap, g, rX_add, rY_add, A_X, B_Y):
            ok += 1
    verdict = "NO FLOOD (some f fail)" if ok < total_f else "flood"
    print(f"(M2) additive collapse via g={bcol}: {ok}/{total_f} -> {verdict} "
          f"-> {'VERIFIED' if ok < total_f else 'FAILED'}")

# ---------------- od sanity (the discount direction) ----------------
checks = [(K, 0, 0), (K, K, K), (6, 6, 2), (3, 4, 0), (0, 0, 0)]
sanity = all(od(r, s) == max(0, r + s - K) for (r, s, _) in checks)
print(f"(od) Lukasiewicz discount od(r,s) <= r with equality iff s=K or r=0: "
      f"{all(od(r, s) <= r for r in range(K + 1) for s in range(K + 1))}, "
      f"od == r exactly when s == K or r == 0: "
      f"{all((od(r, s) == r) == (s == K or r == 0) for r in range(K + 1) for s in range(K + 1))}")
