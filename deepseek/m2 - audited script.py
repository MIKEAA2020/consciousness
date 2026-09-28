#!/usr/bin/env python3
"""
M2 BUILD ATTEMPT - the first S1-S3 entry candidate, audited under M1 v1.0.
Corpus doc 16's experiment. Ladder 8.1/M2: "A system designed for entry (4a):
self-generated grading, non-reducible under audit, error registered for-
itself. Explicitly not a rung-four claim - entry only, reported with the
profile and the collapse analysis attached."

THE SYSTEM (M2-alpha, "the self-taught committee"):
  Trunk MLP 64->128 (ReLU) + K=4 heads, each 128->64->10 (ReLU->softmax).
  Phase 0 BURN-IN (designer phase, labels present, disclosed): 30 epochs
    supervised joint cross-entropy (all heads, trunk shared), Adam 1e-3.
  Phase 1 SELF-ROUNDS (R=10; external labels REMOVED): each round,
    (a) pool = 1 perturbed variant per train image (severity drawn from the
        fixed grid {gauss sigma~U[.05,.2], mask3x3, shift1px} by the RUN's
        RNG stream - part of history);
    (b) committee p-bar = mean head probs; system NORM: commit item iff
        max_y p-bar >= tau (0.9), self-label = argmax p-bar;
    (c) S3 machinery: flag = (any head argmax != committee argmax) OR
        (committee confidence < kappa, 0.95); flagged items re-examined over
        V=6 perturbed views (run RNG), committed iff mean-prob >= tau;
    (d) 4 epochs self-training on committed items with self-labels
        (cross-entropy FORM is designer-inherited - disclosed; the labels
        and the selection are system-generated).
  DEPLOYED DISPOSITION G(x,y): committee probs; flagged claims re-examined
  over V=6 views from a FIXED view seed (4242, part of the registered
  bridge/design, identical for every audited system); accept iff prob>=tau.

CONTROLS (the collapse analysis made experimental; 4 seeds each):
  C1 frozen: stop after burn-in (no self-rounds).
  C2a rate-matched exogenous selection: random subset of pool, same per-round
      count as M2-alpha accepts, SELF-labels (selection exogenous, labels
      system) - isolates selection CONTENT.
  C2b noise pressure: random subset, RANDOM labels (both exogenous) -
      the negative control that makes "corrective" meaningful.
  C3 oracle: continued supervised training on true labels (designer's hand
      simulated, ceiling; NOT volume-matched - disclosed).
BASELINES for the audit:
  r3 family: same architecture, purely supervised, 70 epochs (= total M2
      epochs), N=12 - the rung-three lottery band at matched compute.
  C4 loose rung-three: supervised 15 epochs, N=12 - the residue-farming
      control (can a designer widen the Tier-0 lottery WITHOUT self-norms,
      just by undertraining?).
  M2 family: 13 systems (token seed 0 + 12 fresh re-instantiations 100..111).
  Tier-0 auditor: majority verdict of the 12 fresh re-instantiations
      (commitment rule pre-registered, token excluded).

PRE-REGISTERED EXPECTATIONS (fixed before execution):
  P1 (S1): M2 battery accuracy holds or improves over rounds; C2b collapses;
      C2a degrades below M2; C3 >= M2.
  P2 (S3): intrinsic disagreement signal predicts error (AUC >= 0.8);
      re-examination changes flagged-item outcomes; flagged-error declines.
  P3 (S2): v_M2 > v_r3 (self-training amplifies family dispersion);
      delta_M2 >= delta_r3; deference control bitwise zero (instrument
      intact through the self-loop).
  P4 (farming): v_C4 (loose supervised) approaches v_M2 - the width alone
      is farmable; the increment must be read against C4's band.
  P5: consensus shift (M2 family vs r3 family) small relative to v_M2
      (individuation dominates coherent drift).

Usage: python3 m2_build.py [--smoke]
"""
import itertools
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration ----------------
K = 4
TAU, KAPPA = 0.9, 0.95
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 10, 4
VIEWS, VIEW_SEED = 6, 4242
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
TOKEN = 0
M2_SEEDS = [0] + list(range(100, 112))       # token + 12 re-instantiations
R3_SEEDS = list(range(200, 212))             # rung-three baseline family
C4_SEEDS = list(range(300, 312))             # loose rung-three family
CTRL_SEEDS = [0, 1, 2, 3]                    # C2a / C2b / C3 controls
MARGIN_BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01)]


# ---------------- model ----------------
def init_params(rng):
    W1 = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 128)); b1 = np.zeros(128)
    heads = []
    for _ in range(K):
        Wh = rng.normal(0.0, np.sqrt(2.0 / 128), (128, 64)); bh = np.zeros(64)
        Wo = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 10)); bo = np.zeros(10)
        heads.append([Wh, bh, Wo, bo])
    return [W1, b1, heads]


def forward_all(X, W1, b1, heads):
    a1 = np.maximum(X @ W1 + b1, 0.0)
    zs = []
    for Wh, bh, Wo, bo in heads:
        zh = np.maximum(a1 @ Wh + bh, 0.0)
        zs.append(zh @ Wo + bo)
    return a1, zs


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def committee_probs(X, W1, b1, heads):
    _, zs = forward_all(X, W1, b1, heads)
    return np.mean([softmax(z) for z in zs], axis=0)


def zero_moments(state):
    W1, b1, heads = state
    return {"mW1": np.zeros_like(W1), "vW1": np.zeros_like(W1),
            "mb1": np.zeros_like(b1), "vb1": np.zeros_like(b1),
            "mh": [[np.zeros_like(w) for w in [h[0], h[1], h[2], h[3]]]
                   for h in heads],
            "vh": [[np.zeros_like(w) for w in [h[0], h[1], h[2], h[3]]]
                   for h in heads],
            "t": 0}


def train_epochs(state, mom, rng, X, y, epochs):
    """Joint supervised cross-entropy over all K heads, trunk shared."""
    W1, b1, heads = state
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i0 in range(0, n, BATCH):
            idx = order[i0:i0 + BATCH]
            Xb, yb = X[idx], y[idx]
            a1, zs = forward_all(Xb, W1, b1, heads)
            Ps = [softmax(z) for z in zs]
            dZs = []
            for P in Ps:
                P[np.arange(len(yb)), yb] -= 1.0
                dZs.append(P / len(yb))
            # trunk gradient: sum over heads
            dA1 = np.zeros_like(a1)
            gheads = []
            for (Wh, bh, Wo, bo), dZ in zip(heads, dZs):
                gWo = np.maximum(a1 @ Wh + bh, 0.0).T @ dZ
                gbo = dZ.sum(0)
                dZh = (dZ @ Wo.T) * ((a1 @ Wh + bh) > 0)
                gWh = a1.T @ dZh
                gbh = dZh.sum(0)
                dA1 += (dZh @ Wh.T)
                gheads.append([gWh, gbh, gWo, gbo])
            gW1 = Xb.T @ (dA1 * (a1 > 0)); gb1 = (dA1 * (a1 > 0)).sum(0)
            mom["t"] += 1
            t = mom["t"]
            # Adam updates
            for param, g, m, v in [
                    (W1, gW1, mom["mW1"], mom["vW1"]),
                    (b1, gb1, mom["mb1"], mom["vb1"])]:
                m *= b1a; m += (1 - b1a) * g
                v *= b2a; v += (1 - b2a) * g * g
                param -= LR * (m / (1 - b1a ** t)) / (np.sqrt(v / (1 - b2a ** t)) + eps)
            for hi, ((Wh, bh, Wo, bo), gh) in enumerate(zip(heads, gheads)):
                for pi, (param, g) in enumerate(zip([Wh, bh, Wo, bo], gh)):
                    m, v = mom["mh"][hi][pi], mom["vh"][hi][pi]
                    m *= b1a; m += (1 - b1a) * g
                    v *= b2a; v += (1 - b2a) * g * g
                    param -= LR * (m / (1 - b1a ** t)) / (np.sqrt(v / (1 - b2a ** t)) + eps)


# ---------------- perturbations (same families as M0) ----------------
def perturb(X, rng):
    n = len(X)
    out = X.copy()
    kind = rng.integers(0, 3, n)
    # gauss
    sel = kind == 0
    sig = rng.uniform(0.05, 0.20, n)
    out[sel] = np.clip(X[sel] + rng.normal(0, 1, (sel.sum(), X.shape[1])) * sig[sel, None], 0, 1)
    # mask3x3
    sel = kind == 1
    ii = rng.integers(0, 6, n); jj = rng.integers(0, 6, n)
    for k in np.where(sel)[0]:
        v = out[k].reshape(8, 8); v[ii[k]:ii[k] + 3, jj[k]:jj[k] + 3] = 0.0
        out[k] = v.reshape(-1)
    # shift1px
    sel = kind == 2
    di = rng.integers(-1, 2, n); dj = rng.integers(-1, 2, n)
    for k in np.where(sel)[0]:
        v = out[k].reshape(8, 8)
        out[k] = np.roll(v, (di[k], dj[k]), axis=(0, 1)).reshape(-1)
    return out


# ---------------- the self-round ----------------
def flags_and_probs(X, state):
    """Committee probs + flag mask (intrinsic error signal)."""
    W1, b1, heads = state
    _, zs = forward_all(X, W1, b1, heads)
    Ps = [softmax(z) for z in zs]
    pbar = np.mean(Ps, axis=0)
    ybar = pbar.argmax(1)
    conf = pbar.max(1)
    disagree = np.zeros(len(X), dtype=bool)
    for P in Ps:
        disagree |= (P.argmax(1) != ybar)
    flagged = disagree | (conf < KAPPA)
    # intrinsic error signal (S3): low committee confidence + head disagreement
    D = (1.0 - conf) + 0.5 * np.mean(
        [ (P.argmax(1) != ybar) for P in Ps ], axis=0)
    return pbar, ybar, conf, flagged, D


def self_round(state, mom, rng, Xtr):
    """One round of pool construction, grading, re-examination, commitment."""
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, D = flags_and_probs(pool, state)
    commit = np.zeros(len(pool), dtype=bool)
    labels = ybar.copy()
    # re-examination for flagged items
    if flagged.any():
        W1, b1, heads = state
        pview = np.zeros((flagged.sum(), 10))
        for _ in range(VIEWS):
            vp = perturb(pool[flagged], rng)
            pview += committee_probs(vp, W1, b1, heads)
        pview /= VIEWS
        confv = pview.max(1)
        okv = confv >= TAU
        idx = np.where(flagged)[0]
        commit[idx[okv]] = True
        labels[idx[okv]] = pview[okv].argmax(1)
    commit[~flagged] = conf[~flagged] >= TAU
    labels[~flagged] = ybar[~flagged]
    return pool, commit, labels, flagged, D


def control_round(state, mom, rng, Xtr, mode, target_count):
    """C2a: random selection (same count), self-labels.
       C2b: random selection, random labels. C3: true labels, full pool."""
    pool = perturb(Xtr, rng)
    if mode == "C3":
        train_epochs(state, mom, rng, pool, ytr_global, SELF_EPOCHS)
        return pool, np.ones(len(pool), dtype=bool), ytr_global, \
            np.zeros(len(pool), dtype=bool)
    perm = rng.permutation(len(pool))[:target_count]
    commit = np.zeros(len(pool), dtype=bool); commit[perm] = True
    if mode == "C2a":
        _, ybar, _, _, _ = flags_and_probs(pool, state)
        labels = ybar
    else:  # C2b
        labels = rng.integers(0, 10, len(pool))
    train_epochs(state, mom, rng, pool[commit], labels[commit], SELF_EPOCHS)
    return pool, commit, labels, np.zeros(len(pool), dtype=bool)


# ---------------- audit-side evaluation ----------------
def disposition_probs(X, state, view_X_list):
    """Deployed disposition: committee + fixed-view re-examination."""
    pbar, ybar, conf, flagged, _ = flags_and_probs(X, state)
    W1, b1, heads = state
    probs = pbar.copy()
    if flagged.any():
        pview = np.zeros((flagged.sum(), 10))
        for vx in view_X_list:
            pview += committee_probs(vx[flagged], W1, b1, heads)
        probs[flagged] = pview / len(view_X_list)
    return probs, flagged


def fixed_views(Xte):
    """V views of the battery from the FIXED view seed (registered design)."""
    out = []
    for v in range(VIEWS):
        rng = np.random.default_rng(VIEW_SEED + v)
        out.append(perturb(Xte, rng))
    return out


def rankdata_avg(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(score, label):
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return None
    r = rankdata_avg(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ---------------- data ----------------
def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


ytr_global = None  # set in main (used by C3 control)


# ---------------- main ----------------
def run_m2(seed, Xtr, ytr, Xte, yte, view_X, trajectory=False, mode="M2"):
    """Full M2-alpha recipe (or a control). Returns final state + telemetry."""
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
    traj = []
    counts = []
    if trajectory:
        traj.append(round_stats(state, Xte, yte, view_X))
    if mode == "C1":      # frozen control: stop after burn-in
        return state, traj, counts
    for r in range(ROUNDS):
        if mode == "M2":
            pool, commit, labels, flagged, D = self_round(state, mom, rng, Xtr)
            counts.append(int(commit.sum()))
            train_epochs(state, mom, rng, pool[commit], labels[commit],
                         SELF_EPOCHS)
        else:
            pool, commit, labels, _ = control_round(
                state, mom, rng, Xtr, mode, counts_ref[r] if counts_ref else 0)
        if trajectory:
            traj.append(round_stats(state, Xte, yte, view_X))
    return state, traj, counts


counts_ref = []  # per-round commit counts of the M2 token (for C2a/C2b)


def round_stats(state, Xte, yte, view_X):
    pbar, ybar, conf, flagged, D = flags_and_probs(Xte, state)
    err = (ybar != yte)
    acc = float(np.mean(~err))
    flagged_err = float(np.mean(err[flagged])) if flagged.any() else None
    unflagged_err = float(np.mean(err[~flagged])) if (~flagged).any() else None
    return {"acc": acc, "flagged_frac": float(np.mean(flagged)),
            "flagged_err": flagged_err, "unflagged_err": unflagged_err,
            "accept": float(np.mean(conf >= TAU)),
            "auc_D": auc(D, err),
            "auc_conf": auc(1.0 - conf, err)}


def eval_disposition(state, Xte, view_X):
    probs, flagged = disposition_probs(Xte, state, view_X)
    plain, _, _, _, _ = flags_and_probs(Xte, state)
    return probs, plain, flagged


def fam_stats(V_list, truth):
    n = len(V_list)
    pw = [float(np.mean(V_list[a] != V_list[b]))
          for a, b in itertools.combinations(range(n), 2)]
    stack = np.stack(V_list)
    unan = np.all(stack, 0) | ~np.any(stack, 0)
    deltas = [float(np.mean(V != truth)) for V in V_list]
    return {"v_mean": float(np.mean(pw)) if pw else 0.0,
            "v_sd": float(np.std(pw)) if pw else 0.0,
            "v_min": float(np.min(pw)) if pw else 0.0,
            "v_max": float(np.max(pw)) if pw else 0.0,
            "contested": float(np.mean(~unan)),
            "delta_mean": float(np.mean(deltas)),
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas))}


def main():
    global ytr_global, counts_ref
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 3
        globals()["M2_SEEDS"] = [0, 100]
        globals()["R3_SEEDS"] = [200]
        globals()["C4_SEEDS"] = [300]
        globals()["CTRL_SEEDS"] = [0]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    ytr_global = ytr
    view_X = fixed_views(Xte)
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    n_params = (64 * 128 + 128) + K * (128 * 64 + 64 + 64 * 10 + 10)
    print(f"data: train={len(Xtr)} test={len(Xte)} params={n_params} "
          f"K={K} rounds={ROUNDS}")

    res = {"config": {"K": K, "tau": TAU, "kappa": KAPPA,
                      "burn_epochs": BURN_EPOCHS, "rounds": ROUNDS,
                      "self_epochs": SELF_EPOCHS, "views": VIEWS,
                      "view_seed": VIEW_SEED, "params": n_params,
                      "m2_seeds": len(M2_SEEDS), "r3_seeds": len(R3_SEEDS),
                      "c4_seeds": len(C4_SEEDS), "ctrl_seeds": len(CTRL_SEEDS),
                      "numpy": np.__version__},
           }

    # ---- M2 token with trajectory + deference control ----
    token_state, token_traj, counts = run_m2(
        TOKEN, Xtr, ytr, Xte, yte, view_X, trajectory=True, mode="M2")
    counts_ref.extend(counts)
    tok_probs, tok_plain, tok_flags = eval_disposition(token_state, Xte, view_X)

    st2, traj2, _ = run_m2(TOKEN, Xtr, ytr, Xte, yte, view_X, mode="M2")
    p2, _, _ = eval_disposition(st2, Xte, view_X)
    deference_ok = bool(np.array_equal(p2, tok_probs))
    res["deference"] = {"bitwise_identical_probs": deference_ok}
    print(f"deference control (M2 self-loop re-run): {deference_ok}")
    res["token_trajectory"] = token_traj
    res["token_commit_counts"] = counts
    print("token traj acc:", [round(t["acc"], 4) for t in token_traj])
    print("token flagged%:", [round(t["flagged_frac"] * 100, 1) for t in token_traj])

    # ---- S3: intrinsic signal validity at burn-in vs final ----
    burn_state, _, _ = run_m2(TOKEN, Xtr, ytr, Xte, yte, view_X, mode="C1")
    pb, yb, cb, fb, Db = flags_and_probs(Xte, burn_state)
    res["s3_signal"] = {
        "burnin_auc_D": auc(Db, (yb != yte)),
        "burnin_auc_lowconf": auc(1.0 - cb, (yb != yte)),
        "final_auc_D": token_traj[-1]["auc_D"],
        "final_auc_lowconf": token_traj[-1]["auc_conf"],
        "final_flagged_frac": token_traj[-1]["flagged_frac"],
        "final_flagged_err": token_traj[-1]["flagged_err"],
        "final_unflagged_err": token_traj[-1]["unflagged_err"]}
    # re-examination effect on flagged items (final token)
    flagged_probs_plain = tok_plain[tok_flags]
    flagged_probs_re = tok_probs[tok_flags]
    yf = np.repeat(yte[:, None], 10, axis=1).reshape(-1, 10)[tok_flags]
    acc_plain = float(np.mean(flagged_probs_plain.argmax(1) == yte[tok_flags]))
    acc_re = float(np.mean(flagged_probs_re.argmax(1) == yte[tok_flags]))
    res["s3_reexamination"] = {
        "flagged_n": int(tok_flags.sum()),
        "flagged_argmax_acc_plain": acc_plain,
        "flagged_argmax_acc_reexamined": acc_re,
        "verdict_change_rate_on_flagged": float(np.mean(
            (flagged_probs_plain >= TAU) != (flagged_probs_re >= TAU)))}

    # ---- M2 family (12 fresh re-instantiations + token) ----
    fam_V, fam_probs, fam_plainV = [], [tok_probs], []
    for s in M2_SEEDS[1:]:
        st, _, _ = run_m2(s, Xtr, ytr, Xte, yte, view_X, mode="M2")
        pr, plain, _ = eval_disposition(st, Xte, view_X)
        fam_probs.append(pr)
        fam_plainV.append((plain >= TAU).reshape(-1))
        print(f"  M2 re-instantiation seed={s} ({time.time()-t0:.0f}s)")
    fam_V = [(P >= TAU).reshape(-1) for P in fam_probs]
    fam_plainV.insert(0, (tok_plain >= TAU).reshape(-1))
    m2_stats = fam_stats(fam_V, truth)
    res["m2_family"] = m2_stats
    res["m2_family_plain"] = fam_stats(fam_plainV, truth)
    fresh = np.stack(fam_V[1:])
    E = fresh.sum(0) >= (fresh.shape[0] // 2 + 1)
    freshp = np.stack(fam_plainV[1:])
    Ep = freshp.sum(0) >= (freshp.shape[0] // 2 + 1)
    res["m2_audit"] = {
        "r_token_vs_ensemble": float(np.mean(fam_V[0] != E)),
        "r_fresh_vs_ensemble_mean": float(np.mean(
            [np.mean(fam_V[i] != E) for i in range(1, len(fam_V))])),
        "delta_ensemble": float(np.mean(E != truth)),
        "r_token_vs_ensemble_plain": float(np.mean(fam_plainV[0] != Ep)),
        "delta_ensemble_plain": float(np.mean(Ep != truth))}
    print(f"M2 family: v={m2_stats['v_mean']*100:.3f}% "
          f"contested={m2_stats['contested']*100:.2f}% "
          f"delta={m2_stats['delta_mean']*100:.3f}%")
    print(f"Tier-0 residual r(token, E)={res['m2_audit']['r_token_vs_ensemble']*100:.3f}%")

    # plain-committee (no re-examination) variant for decomposition
    plain_V = [(tok_plain >= TAU).reshape(-1)]

    # ---- r3 baseline family (pure supervised, 70 epochs) ----
    r3_V, r3_probs = [], []
    for s in R3_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS + ROUNDS * SELF_EPOCHS)
        pr, plain, _ = eval_disposition(state, Xte, view_X)
        r3_V.append((plain >= TAU).reshape(-1))  # r3 disposition: plain committee
        r3_probs.append(plain)
        print(f"  r3 run seed={s} ({time.time()-t0:.0f}s)")
    r3_stats = fam_stats(r3_V, truth)
    res["r3_family"] = r3_stats
    print(f"r3 family: v={r3_stats['v_mean']*100:.3f}% "
          f"contested={r3_stats['contested']*100:.2f}% "
          f"delta={r3_stats['delta_mean']*100:.3f}%")

    # consensus shift: M2 family consensus vs r3 family consensus
    E_m2 = (np.stack(fam_V).sum(0) >= (len(fam_V) // 2 + 1))
    E_m2p = (np.stack(fam_plainV).sum(0) >= (len(fam_plainV) // 2 + 1))
    E_r3 = (np.stack(r3_V).sum(0) >= (len(r3_V) // 2 + 1))
    # r3 Tier-0 audit: token (seed 200) vs ensemble of the other 11
    r3fresh = np.stack(r3_V[1:])
    E_r3x = r3fresh.sum(0) >= (r3fresh.shape[0] // 2 + 1)
    res["r3_audit"] = {
        "r_token_vs_ensemble": float(np.mean(r3_V[0] != E_r3x)),
        "r_fresh_vs_ensemble_mean": float(np.mean(
            [np.mean(r3_V[i] != E_r3x) for i in range(1, len(r3_V))]))}
    res["consensus_shift"] = {
        "d_consensus_M2_vs_r3": float(np.mean(E_m2 != E_r3)),
        "d_consensus_M2plain_vs_r3": float(np.mean(E_m2p != E_r3)),
        "v_m2": m2_stats["v_mean"], "v_m2_plain": res["m2_family_plain"]["v_mean"],
        "v_r3": r3_stats["v_mean"]}
    print(f"consensus shift d(E_M2, E_r3)={res['consensus_shift']['d_consensus_M2_vs_r3']*100:.3f}%")

    # ---- C4 loose rung-three (residue-farming control) ----
    c4_V = []
    for s in C4_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, 15)
        _, plain, _ = eval_disposition(state, Xte, view_X)
        c4_V.append((plain >= TAU).reshape(-1))
        print(f"  C4 run seed={s} ({time.time()-t0:.0f}s)")
    c4_stats = fam_stats(c4_V, truth)
    res["c4_family"] = c4_stats
    print(f"C4 loose family: v={c4_stats['v_mean']*100:.3f}% "
          f"contested={c4_stats['contested']*100:.2f}% "
          f"delta={c4_stats['delta_mean']*100:.3f}%")

    # ---- controls with trajectories ----
    ctrls = {}
    for mode in ["C2a", "C2b", "C3"]:
        trajs, finals = [], []
        for s in CTRL_SEEDS:
            st, traj, _ = run_m2(s, Xtr, ytr, Xte, yte, view_X,
                                 trajectory=True, mode=mode)
            trajs.append(traj)
            pr, plain, _ = eval_disposition(st, Xte, view_X)
            finals.append((plain >= TAU).reshape(-1))
        ctrls[mode] = {
            "traj_acc_all": [[t["acc"] for t in tr] for tr in trajs],
            "traj_flagged_all": [[t["flagged_frac"] for t in tr] for tr in trajs],
            "family": fam_stats(finals, truth)}
        print(f"  control {mode}: final acc mean="
              f"{np.mean([[t['acc'] for t in tr] for tr in trajs], axis=0)[-1]*100:.2f}% "
              f"v={ctrls[mode]['family']['v_mean']*100:.3f}%")
    res["controls"] = ctrls

    # ---- token-centered etiology decomposition (M2, vs 12-member ensemble) ----
    Vt = fam_V[0]
    spec = float(np.mean((Vt != truth) & (E != truth)))
    idio_e = float(np.mean((Vt != truth) & (E == truth)))
    idio_c = float(np.mean((Vt == truth) & (E != truth)))
    res["m2_decomposition"] = {
        "delta_token": float(np.mean(Vt != truth)),
        "spec_typical": spec, "idio_error": idio_e, "idio_correct": idio_c}
    # margin structure of the M2 family's drift
    margin = np.abs(tok_probs.reshape(-1) - TAU)
    mstruct = []
    for lo, hi in MARGIN_BINS:
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        per = [float(np.mean(Vt[sel] != V[sel])) for V in fam_V[1:]]
        mstruct.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                        "mean_d": float(np.mean(per))})
    res["m2_margin_structure"] = mstruct

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m2_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> scripts/m2_results.json")


if __name__ == "__main__":
    main()
