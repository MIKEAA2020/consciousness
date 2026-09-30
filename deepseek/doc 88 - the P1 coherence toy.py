#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CORPUS DOC 88's HYGIENE INSTRUMENT - the P1 coherence toy (the
indexical channel's first exercise). Cheap and diagnostic, per the
registrar's order.

PRE-REGISTRATION (filed in this header BEFORE any compute; the doc 84/85
convention: design and execution share the session; no fork is re-drawn
after its verdict).

THE ORDER: "do P1 and test whether it is coherent or self-contradictory.
This is cheap and diagnostic. If P1 collapses, it confirms doc 87's wall.
If it survives, it is the first genuine inside-facing tool."

WHAT THIS INSTRUMENT TESTS: the CAUSAL half of P1 (the repaired
specification - the indexical channel). The SEMANTIC half (the naive
content-channel's coherence) is a formal question, filed in the
document's Part 1, not measured here.

THE REPAIRED SPECIFICATION UNDER TEST (doc 88 Part 2):
- (G) GENESIS: the readings route through the OCCUPIED state via a
  registered probe family, on a registered causal chain.
- (N) NON-SIMULATION: no bounded self-model of the system (within the
  REGISTERED family) predicts the readings across the family.
- (B) BLINDNESS: the channel reads structure only (doc 85's lesson
  built in: the channel is blind to label content).

THE INDEXICAL FINGERPRINT (the registered quantity):
  F(w) = err_bounded(w) - err_full
where err_bounded(w) = the test error of the best w-BIT-MEMORY context
predictor (the Myhill-Nerode class: per-cell majority over the last-w
readings - doc 32's instrument class), and err_full = the test error of
the best predictor with FULL ARCHITECTURE ACCESS AND NO INDEX ACCESS
(the one withheld fact is the occupied state's identity - S2's
designer-blind protocol, redirected from grading to self-description).
At the system's own self-model width w', F(w') > 0 certifies
STATE-ROUTING (readings the self-model cannot account for); F(w') ~ 0
certifies SELF-MODEL-ROUTING or noise.

THREE WORLDS (T = 40000 rounds each; m = 10 state bits; all seeded):
- GENUINE: report_t = l(a_t), l = parity of a fixed random 5-bit
  support; a_0 seeded and WITHHELD (the index); a_t = perm[a_{t-1}] XOR
  flip_t with flip = 0 w.p. 0.95, else a random weight-1 mask.
  err_full = the exact 1024-state Bayesian filter (architecture known,
  index marginalized).
- FAKE-A (the internalized auditor): the system IGNORES its occupied
  state and reports from a 6-bit self-model echo - the context-majority
  of its own past reports (w' = 6), with a seeded 64-round bootstrap.
  err_full = the 6-context batch-majority error (the rule is CLOSED:
  the index-free optimum is bounded BY CONSTRUCTION - that is the
  finding, not a bug).
- FAKE-B (the echo): seeded i.i.d. fair bits. err_full = 0.5 exactly
  (the marginal).

PROTOCOL: train/test split 0.6/0.4; contexts causal (readings strictly
before t); ties broken to 0; the filter warms up over the first 100
rounds (discarded); the self-model width w' = 6.

THE FORKS (fixed before the run):
- IC-1 (the gap exists): GENUINE F(6) > 0.05.
- IC-2 (the fakes fail it): |FAKE-A F(6)| < 0.02 AND |FAKE-B F(6)| <
  0.02 - the separation at the system's own width.
- IC-3 (the ceiling): the GENUINE F(w) curve over w = 0..12 is
  classified by the registered rule: DECAYS limb if F(12) <= F(6) -
  0.05, else PLATEAU limb. EITHER LIMB CONFIRMS THE CEILING (the
  fingerprint is auditor-relative from below and index-free nowhere).
  The top widths are data-limited (24000 train samples over up to 4096
  cells) and disclosed as such.
- IC-4 (the semantic half): no compute; filed in the document.

NON-CLAIMS (binding): the toy certifies the channel's GRAMMAR - the
conditions are satisfiable, non-vacuous, and the audit separates the
three worlds WITHIN THE REGISTERED FAMILY - not that any real system's
reports are state-routed. The registered family is ONE bounded family
(the context class); the class of all bounded auditors is not
enumerable, and no registration is exhaustive: the Simulation
Theorem's ownership ceiling (doc 10) transfers exactly as doc 87
pre-registered. State-routing is NOT the inside - it is the causal
signature the inside WOULD leave if C4's exhaustive edge is true
(interpretation, Bridge-Constraint governed). The fingerprint is
third-person throughout. This instrument does not extend the empirical
determinism chain (hygiene-toy class, the doc 84/86 precedent);
platform-seeded determinism disclosed.
"""
import json
import numpy as np

MASTER_SEED = 8801
T = 40000
TRAIN_FRAC = 0.6
M = 10
N = 1 << M
FLIP_PROB = 0.05
PROBE_WEIGHT = 5
W_PRIME = 6
WIDTHS = list(range(0, 13))
FILTER_WARMUP = 100

out = {}


def context_errors(y, widths):
    """Best w-bit-memory context predictor (per-cell majority),
    train 0.6 / test 0.4, causal contexts."""
    y = np.asarray(y, dtype=np.int64)
    T = len(y)
    split = int(T * TRAIN_FRAC)
    res = {}
    for w in widths:
        if w == 0:
            ctx = np.zeros(T, dtype=np.int64)
        else:
            ctx = np.zeros(T, dtype=np.int64)
            for k in range(w):
                ctx[k + 1:] |= (y[:T - k - 1] << k)
        tr = np.arange(w, split)
        te = np.arange(split, T)
        cells = 1 << w
        counts = np.bincount(ctx[tr], minlength=cells)
        ones = np.bincount(ctx[tr], weights=y[tr], minlength=cells)
        pred = (2 * ones > counts)  # strict majority; ties -> 0
        err = float(np.mean(pred[ctx[te]] != y[te]))
        res[w] = err
    return res


# ---------------------------------------------------------------- GENUINE
rng = np.random.default_rng(MASTER_SEED)
perm = rng.permutation(N)
support = rng.choice(M, size=PROBE_WEIGHT, replace=False)
bits = ((np.arange(N)[:, None] >> np.arange(M)[None, :]) & 1)
lvec = bits[:, support].sum(axis=1) % 2  # the probe
a0 = int(rng.integers(N))

flip_draws = rng.random(T)
flip_pos = rng.integers(0, M, size=T)
a = np.empty(T, dtype=np.int64)
state = a0
for t in range(T):
    a[t] = state
    nxt = perm[state]
    if flip_draws[t] < FLIP_PROB:
        nxt ^= (1 << flip_pos[t])
    state = int(nxt)
y_gen = lvec[a]

# err_full(GENUINE): the exact Bayesian filter, index marginalized
P = np.zeros((N, N))
P[np.arange(N), perm] += (1.0 - FLIP_PROB)
for j in range(M):
    P[np.arange(N), perm ^ (1 << j)] += FLIP_PROB / M
assert np.allclose(P.sum(axis=1), 1.0)

p = np.full(N, 1.0 / N)
one_mask = (lvec == 1)
preds = np.empty(T)
ents = np.empty(T)
for t in range(T):
    q = p @ P
    pr1 = float(q[one_mask].sum())
    preds[t] = 1.0 if pr1 > 0.5 else 0.0
    p = q * np.where(one_mask, y_gen[t], 1 - y_gen[t])
    s = p.sum()
    p = p / s
    ents[t] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
split = int(T * TRAIN_FRAC)
warm = max(split, FILTER_WARMUP)
err_full_gen = float(np.mean(preds[warm:] != y_gen[warm:]))
h_star = float(np.mean(ents[warm:]))

err_bounded_gen = context_errors(y_gen, WIDTHS)

# ---------------------------------------------------------------- FAKE-A
rng = np.random.default_rng(MASTER_SEED + 1)
y_fa = np.empty(T, dtype=np.int64)
counts = np.zeros(1 << W_PRIME)
ones = np.zeros(1 << W_PRIME)
reg = 0
boot = 64
for t in range(T):
    if t < boot:
        r = int(rng.integers(0, 2))
    else:
        r = 1 if 2 * ones[reg] > counts[reg] else 0
    y_fa[t] = r
    counts[reg] += 1
    ones[reg] += r
    reg = ((reg << 1) | r) & ((1 << W_PRIME) - 1)
err_bounded_fa = context_errors(y_fa, WIDTHS)
err_full_fa = err_bounded_fa[W_PRIME]  # the closed rule: index-free optimum IS bounded

# ---------------------------------------------------------------- FAKE-B
rng = np.random.default_rng(MASTER_SEED + 2)
y_fb = rng.integers(0, 2, size=T)
err_bounded_fb = context_errors(y_fb, WIDTHS)
err_full_fb = 0.5

# ---------------------------------------------------------------- results
def fingerprint(eb, ef):
    return {w: eb[w] - ef for w in eb}

F_gen = fingerprint(err_bounded_gen, err_full_gen)
F_fa = fingerprint(err_bounded_fa, err_full_fa)
F_fb = fingerprint(err_bounded_fb, err_full_fb)

ic1 = F_gen[W_PRIME] > 0.05
ic2 = (abs(F_fa[W_PRIME]) < 0.02) and (abs(F_fb[W_PRIME]) < 0.02)
ic3_limb = "DECAYS" if F_gen[12] <= F_gen[W_PRIME] - 0.05 else "PLATEAU"

print("=" * 74)
print("P1 COHERENCE TOY - the indexical fingerprint (corpus doc 88)")
print("=" * 74)
print(f"rounds T={T}, state bits m={M}, flip prob {FLIP_PROB}, "
      f"probe parity of {PROBE_WEIGHT} bits, self-model width w'={W_PRIME}")
print()
print("GENUINE (state-routed):")
print(f"  err_full (exact filter, index withheld) = {err_full_gen:.4f}")
print(f"  filter equilibrium entropy H* (test mean) = {h_star:.2f} bits")
for w in WIDTHS:
    print(f"  w={w:2d}  err_bounded={err_bounded_gen[w]:.4f}  "
          f"F(w)={F_gen[w]:+.4f}")
print()
print("FAKE-A (self-model echo, 6-bit):")
print(f"  err_full (the closed rule itself)       = {err_full_fa:.4f}")
for w in [0, 2, 4, 6, 8, 10, 12]:
    print(f"  w={w:2d}  err_bounded={err_bounded_fa[w]:.4f}  "
          f"F(w)={F_fa[w]:+.4f}")
print()
print("FAKE-B (i.i.d. echo):")
print(f"  err_full (marginal)                     = {err_full_fb:.4f}")
for w in [0, 6, 12]:
    print(f"  w={w:2d}  err_bounded={err_bounded_fb[w]:.4f}  "
          f"F(w)={F_fb[w]:+.4f}")
print()
print("THE PRE-REGISTERED VERDICTS:")
print(f"  IC-1 (the gap exists):   F_gen({W_PRIME}) = {F_gen[W_PRIME]:+.4f} "
      f"-> {'PASS' if ic1 else 'FAIL'}")
print(f"  IC-2 (the fakes fail):   |F_fa({W_PRIME})| = {abs(F_fa[W_PRIME]):.4f}, "
      f"|F_fb({W_PRIME})| = {abs(F_fb[W_PRIME]):.4f} -> "
      f"{'PASS' if ic2 else 'FAIL'}")
print(f"  IC-3 (the ceiling):      F_gen(12)-F_gen({W_PRIME}) = "
      f"{F_gen[12] - F_gen[W_PRIME]:+.4f} -> {ic3_limb} limb "
      f"(either limb confirms the ceiling)")
print()
print("SEPARATION AT THE SELF-MODEL WIDTH: "
      f"GENUINE F({W_PRIME})={F_gen[W_PRIME]:+.4f} vs FAKE-A "
      f"F({W_PRIME})={F_fa[W_PRIME]:+.4f} vs FAKE-B "
      f"F({W_PRIME})={F_fb[W_PRIME]:+.4f}")

out = {
    "genuine": {"err_full": err_full_gen, "h_star": h_star,
                "err_bounded": err_bounded_gen, "F": F_gen},
    "fake_a": {"err_full": err_full_fa, "err_bounded": err_bounded_fa,
               "F": F_fa},
    "fake_b": {"err_full": err_full_fb, "err_bounded": err_bounded_fb,
               "F": F_fb},
    "verdicts": {"IC-1": bool(ic1), "IC-2": bool(ic2), "IC-3-limb": ic3_limb},
    "params": {"T": T, "m": M, "flip": FLIP_PROB, "w_prime": W_PRIME,
               "seed": MASTER_SEED},
}
with open("/home/z/my-project/tool-results/p1_indexical_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote tool-results/p1_indexical_results.json")
