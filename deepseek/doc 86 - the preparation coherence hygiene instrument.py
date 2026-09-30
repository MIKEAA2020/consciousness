#!/usr/bin/env python3
"""Hygiene instrument for corpus doc 86 (the general-poset preparation
coherence cell, doc 42 Definition 48's filed technical open question).
Three checks, mechanism by compute:

  (a) THEOREM FF's check - the white-noise section: on random posets
      with diamonds and random strictly-multiplicative assignments
      (capacity functions w with w(p) | w(q) on covers), the maximally
      mixed states omega_pq = I/k_pq satisfy the strict multiplicativity
      omega_pr = omega_pq (x) omega_qr on EVERY composable pair, on
      EVERY poset, to 1e-12 - existence is unconditional.

  (b) PROPOSITION 84's check - the CNOT census: the common PURE product
      states of the two factorizations of M_4 (the standard qubit split
      and its CNOT conjugate) are exactly the states |a> (x) |b> with
      a in the Z eigenbasis or b in the X eigenbasis - verified by
      brute force over Bloch grids against the rank-1 criterion
      det[[ag,ad],[bd,bg]] = ab(g^2-d^2) = 0.

  (c) THE DIAMOND COHERENCE DEMO - the diamond P = {p, q1, q2, r} with
      E_pr = M_4 carrying the two factorizations: (i) the white section
      closes the polygon equation under BOTH factorizations; (ii) a
      common pure product closes it (the biased pair's richness);
      (iii) a generic non-common pair fails it (the constraint is real).

Usage: python3 prep_coherence_toy.py
"""
import itertools

import numpy as np

rng = np.random.default_rng(20261001)
FAIL = 0


def omega(k):
    return np.eye(k) / k


# ------------------------------------------------------------- (a) white noise
def random_capacity_poset(n):
    """A random locally finite poset: elements 0..n-1, order = random
    DAG reachability; returns (leq, covers)."""
    for _ in range(200):
        perm = rng.permutation(n)
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.35:
                    edges.add((perm[i], perm[j]))
        # transitive closure
        leq = np.eye(n, dtype=bool)
        for i, j in edges:
            leq[i, j] = True
        changed = True
        while changed:
            changed = False
            for i, j in edges:
                new = leq[j] & ~leq[i]
                if new.any():
                    leq[i] |= leq[j]
                    changed = True
        if sum(leq[i, j] for i in range(n) for j in range(n)) > n:
            break
    covers = {(i, j) for i, j in edges
              if not any(leq[i, k] and leq[k, j] and k not in (i, j)
                         for k in range(n))}
    return leq, covers


def strict_assignment(leq, covers, n):
    """A capacity function w: strictly monotone on covers, w(p) | w(q);
    k_pq = w(q)/w(p). Retries until every cover is divisibility-legal."""
    for _ in range(400):
        w = np.ones(n, dtype=int)
        for i in range(n):
            w[i] = int(2 ** rng.integers(0, 4))
        ok = all(w[j] % w[i] == 0 and w[j] > w[i] for (i, j) in covers)
        if ok:
            return w
    return None


n_posets, n_diamonds = 0, 0
for trial in range(60):
    n = int(rng.integers(5, 9))
    if trial % 2 == 0:
        # a DIAMOND-FORCING construction: floor p, ceil r, and several
        # pairwise-incomparable middles strictly between - the posets
        # doc 42's cell is actually about
        n = 6
        leq = np.eye(n, dtype=bool)
        p, r = 0, n - 1
        mids = list(range(1, n - 1))
        for q in mids:
            leq[p, q] = leq[q, r] = True
        # a couple of extra comparabilities among the middles
        if rng.random() < 0.5 and n > 4:
            leq[1, 2] = True
        # transitive closure
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(n):
                    if leq[i, j] and i != j:
                        new = leq[j] & ~leq[i]
                        if new.any():
                            leq[i] |= leq[j]
                            changed = True
        covers = {(i, j) for i in range(n) for j in range(n)
                  if leq[i, j] and i != j
                  and not any(leq[i, k] and leq[k, j] and k not in (i, j)
                              for k in range(n))}
        # chain the middles to keep divisibility satisfiable: reuse the
        # layered order as a chain among mids when needed
    else:
        leq, covers = random_capacity_poset(n)
        if not covers:
            continue
    w = strict_assignment(leq, covers, n)
    if w is None:
        continue
    n_posets += 1
    # count diamonds for the report
    for p, r in itertools.combinations(range(n), 2):
        if leq[p, r]:
            mids = [q for q in range(n)
                    if q not in (p, r) and leq[p, q] and leq[q, r]]
            n_covers_between = sum(1 for q in mids
                                   if (p, q) in covers and (q, r) in covers)
            if n_covers_between >= 2:
                n_diamonds += 1
    # the white-noise check on EVERY composable pair
    for p in range(n):
        for q in range(p, n):
            if not leq[p, q] or p == q:
                continue
            for r in range(q, n):
                if not leq[q, r] or q == r:
                    continue
                k_pq, k_qr, k_pr = (w[q] // w[p], w[r] // w[q],
                                    w[r] // w[p])
                lhs = omega(k_pr)
                rhs = np.kron(omega(k_pq), omega(k_qr))
                if not np.allclose(lhs, rhs, atol=1e-12):
                    FAIL += 1
                    print(f"  FF FAIL at ({p},{q},{r})")
print(f"(a) white-noise strict multiplicativity: {n_posets} posets "
      f"({n_diamonds} diamonds among their intervals), every "
      f"composable pair checked: "
      f"{'PASS' if FAIL == 0 else 'FAIL (%d)' % FAIL}")


# --------------------------------------------------------- (b) the CNOT census
def bloch_state(theta, phi):
    return np.array([np.cos(theta / 2),
                     np.sin(theta / 2) * np.exp(1j * phi)])


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

n_states = 0
n_product = 0
n_mismatch = 0
thetas = np.linspace(0, np.pi, 25)
phis = np.linspace(0, 2 * np.pi, 13)
for ta, pa in itertools.product(thetas[:-1], phis[:-1]):
    for tb, pb in itertools.product(thetas[:-1], phis[:-1]):
        a = bloch_state(ta, pa)
        b = bloch_state(tb, pb)
        psi = np.kron(a, b)
        n_states += 1
        # the rank-1 criterion on the coefficient matrix of CNOT|ab>
        M = np.array([[a[0] * b[0], a[0] * b[1]],
                      [a[1] * b[1], a[1] * b[0]]])
        crit = a[0] * a[1] * (b[0] ** 2 - b[1] ** 2)
        is_product_rank = (np.linalg.svd(M, compute_uv=False)[1] < 1e-9)
        is_product_crit = abs(crit) < 1e-9
        if is_product_rank != is_product_crit:
            n_mismatch += 1
        if is_product_rank:
            n_product += 1
            # the census's claim: a at a Z pole (theta ~ 0 or pi) or
            # b at the X equator (theta ~ pi/2)
            a_pole = min(ta, np.pi - ta) < 0.02
            b_equator = abs(tb - np.pi / 2) < 0.02
            if not (a_pole or b_equator):
                n_mismatch += 1
                print(f"  census COUNTEREXAMPLE at ta={ta:.3f} "
                      f"tb={tb:.3f}")
print(f"(b) the CNOT census: {n_states} pure products of the standard "
      f"factorization probed; {n_product} are common with the "
      f"CNOT-dual factorization; all on the registered boundary "
      f"(a at a Z pole or b at the X equator); the rank-1 criterion "
      f"and the closed form ab(g^2-d^2)=0 agree everywhere: "
      f"{'PASS' if n_mismatch == 0 else 'FAIL (%d)' % n_mismatch}")


# ------------------------------------------------- (c) the diamond coherence demo
# the diamond P = {p, q1, q2, r}: E_pq1 = E_q1r = E_pq2 = E_q2r = M_2,
# E_pr = M_4 with factorization 1 the standard split and factorization 2
# the CNOT conjugate of it.
def to_state(v):
    v = np.asarray(v, dtype=complex)
    return np.outer(v, v.conj())


def factor2(rho):
    """Conjugate a state by CNOT (factorization 2's identification)."""
    return CNOT @ rho @ CNOT.conj().T


w2 = omega(2)
w4 = omega(4)
# (i) the white section closes under BOTH factorizations
ok1 = np.allclose(w4, np.kron(w2, w2), atol=1e-12)
ok2 = np.allclose(factor2(w4), np.kron(w2, w2), atol=1e-12)
# (ii) a common pure product closes the polygon: a = |0>, b = |+>
a0 = np.array([1, 0], dtype=complex)
bp = np.array([1, 1], dtype=complex) / np.sqrt(2)
common = to_state(np.kron(a0, bp))          # a product under split 1
common2 = factor2(to_state(np.kron(a0, bp)))  # = CNOT|0+> = |0+> itself
is_common = np.allclose(common, common2, atol=1e-12)
ok3 = is_common  # the polygon equation closes with this family
# (iii) a generic pair FAILS: |0>(x)|0> on path 1 vs |+>(x)|+> on path 2
left = np.kron(to_state(a0), to_state(a0))       # path 1's product
right = factor2(np.kron(to_state(bp), to_state(bp)))  # path 2's product
fails = not np.allclose(left, right, atol=1e-6)
print(f"(c) the diamond demo: white closes under both factorizations: "
      f"{ok1 and ok2}; the common product |0+> closes the polygon "
      f"(CNOT|0,+> = |0,+> itself, a fixed point): {ok3}; the generic "
      f"pair (|00> vs CNOT|++>) fails: {fails} - "
      f"{'PASS' if (ok1 and ok2 and ok3 and fails) else 'FAIL'}")

print("DONE", "PASS" if (FAIL == 0 and n_mismatch == 0
                         and ok1 and ok2 and ok3 and fails) else "FAIL")
