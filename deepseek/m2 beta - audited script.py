#!/usr/bin/env python3
"""
M2-BETA - THE SECOND-GENERATION BUILD under the amended instrument.
Corpus doc 19's experiment. Ordered at the M2-alpha issuance: "a second-
generation M2 build using the three amendments + selection-channel response
design."

WHAT THE FIRST BUILD TAUGHT THE SECOND (m2_results.json):
  - S1 passed (self-selection content carries ~1pp corrective force) -> keep.
  - S3 split: registration strong (AUC .95, 42x concentration) but the
    REGISTERED RESPONSE (multi-view re-examination) was harmful (flagged
    accuracy 60.0 -> 46.7); correction travels through the SELECTION channel
    -> redesign the response to travel the measured channel.
  - S2 failed at the wall; Tier-0 residue real but width farmable (C4);
    the frontier is 2D (v, delta) -> report against farming families.

THE ONE DESIGN CHANGE (single controlled variable):
  M2-alpha's response: flagged -> re-examine over 6 external views ->
    commit iff view-mean >= tau.  [eval-time procedure; measured harmful]
  M2-beta's response:  flagged -> QUARANTINE (never trained on this round)
    -> re-graded next round by the EVOLVED committee (deferred self-
    re-examination through the system's own development) -> released iff
    then unflagged & conf >= tau; dropped permanently after 2 quarantined
    rounds.  [an error-triggered STATE change - the ladder's S3 letter]
  Deployment disposition: plain committee at tau=0.9 (the bridge is a
  development-time component now; the flag signal remains computed and
  reported at deployment as the S3 registration).

THE AMENDED INSTRUMENT, EXERCISED:
  Amendment 1 (frontier reporting): the (v, delta) point is reported against
    the farming-control families of the same campaign (farming_results.json,
    including the adaptive composite F8) - not against C4 alone.
  Amendment 2 (three-way residual): r(token,E), r(fresh,E), v - all three.
  Amendment 3 (bridge-efficacy audit): the quarantine bridge audited
    separately from the norm: released-vs-dropped item accuracy, flagged
    trajectory, ratchet shape, razor's-edge band occupancy.

PRE-REGISTERED EXPECTATIONS (fixed before execution):
  B1 (S1): accuracy holds >= 96.0%; C2a gap ~1pp preserved; C2b collapses;
      C1 (frozen) still above all continued regimes.
  B2 (S3-response): no deployment-time flagged degradation (plain-committee
      flagged accuracy ~60% vs M2-alpha's re-examined 46.7%); released items
      more accurate than dropped items (deferral selects recoverable
      uncertainty); razor's-edge band not emptied.
  B3 (S2): v_M2beta < v_M2alpha (quarantine narrows experienced dispersion,
      predict v ~0.35-0.45%); delta_M2beta <= delta_M2alpha; the F8
      composite dominates both M2 points -> S2 fails again, now reported
      against the adversarial frontier.
  B4: deference control bitwise through the quarantine loop.
  B5: commit ratchet flatter than M2-alpha's; flagged fraction declines
      slower (less consolidation through the flag gate).
  B6: consensus shift vs r3 comparable to M2-alpha's 0.311% (the self-phase
      still coherently moves the family).

Usage: python3 m2beta_build.py [--smoke]
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
VIEWS, VIEW_SEED = 6, 4242           # VIEWS: M2-alpha arm only
QUAR_MAX_AGE = 2                     # rounds in quarantine before permanent drop
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
TOKEN = 0
M2B_SEEDS = [0] + list(range(100, 112))   # M2-beta family (token + 12 fresh)
M2A_SEEDS = [0] + list(range(100, 112))   # M2-alpha family, recomputed
R3_SEEDS = list(range(200, 212))
C4_SEEDS = list(range(300, 312))
CTRL_SEEDS = [0, 1, 2, 3]
MARGIN_BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01)]


# ---------------- model (identical to m2_build.py) ----------------
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
            "mh": [[np.zeros_like(w) for w in h] for h in heads],
            "vh": [[np.zeros_like(w) for w in h] for h in heads],
            "t": 0}


def train_epochs(state, mom, rng, X, y, epochs):
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


# ---------------- perturbations (identical to m2_build.py) ----------------
def perturb(X, rng):
    n = len(X)
    out = X.copy()
    kind = rng.integers(0, 3, n)
    sel = kind == 0
    sig = rng.uniform(0.05, 0.20, n)
    out[sel] = np.clip(X[sel] + rng.normal(0, 1, (sel.sum(), X.shape[1])) * sig[sel, None], 0, 1)
    sel = kind == 1
    ii = rng.integers(0, 6, n); jj = rng.integers(0, 6, n)
    for k in np.where(sel)[0]:
        v = out[k].reshape(8, 8); v[ii[k]:ii[k] + 3, jj[k]:jj[k] + 3] = 0.0
        out[k] = v.reshape(-1)
    sel = kind == 2
    di = rng.integers(-1, 2, n); dj = rng.integers(-1, 2, n)
    for k in np.where(sel)[0]:
        v = out[k].reshape(8, 8)
        out[k] = np.roll(v, (di[k], dj[k]), axis=(0, 1)).reshape(-1)
    return out


def flags_and_probs(X, state):
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
    D = (1.0 - conf) + 0.5 * np.mean(
        [(P.argmax(1) != ybar) for P in Ps], axis=0)
    return pbar, ybar, conf, flagged, D


# ---------------- the two self-rounds ----------------
def self_round_alpha(state, mom, rng, Xtr):
    """M2-alpha's round: view re-examination commits flagged items."""
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, D = flags_and_probs(pool, state)
    commit = np.zeros(len(pool), dtype=bool)
    labels = ybar.copy()
    if flagged.any():
        W1, b1, heads = state
        pview = np.zeros((flagged.sum(), 10))
        for _ in range(VIEWS):
            vp = perturb(pool[flagged], rng)
            pview += committee_probs(vp, W1, b1, heads)
        pview /= VIEWS
        okv = pview.max(1) >= TAU
        idx = np.where(flagged)[0]
        commit[idx[okv]] = True
        labels[idx[okv]] = pview[okv].argmax(1)
    commit[~flagged] = conf[~flagged] >= TAU
    labels[~flagged] = ybar[~flagged]
    return pool, commit, labels, flagged


def self_round_beta(state, mom, rng, Xtr, quar, ytr):
    """M2-beta's round: quarantine + deferred re-grade (the response IS a
    state change). `quar` = list of (X_item, true_idx, age). Returns
    (train_X, train_y, quar, dropped, telemetry)."""
    released_X, released_y, released_true = [], [], []
    kept, dropped = [], []
    for Xq, ti, age in quar:
        pbar, ybar, conf, flagged, _ = flags_and_probs(Xq[None, :], state)
        if (not flagged[0]) and (conf[0] >= TAU):
            released_X.append(Xq); released_y.append(int(ybar[0]))
            released_true.append(int(ytr[ti]))
        elif age + 1 >= QUAR_MAX_AGE:
            dropped.append((Xq, ti))          # never trained on
        else:
            kept.append((Xq, ti, age + 1))
    # fresh pool, plain grading, flagged -> quarantine
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, D = flags_and_probs(pool, state)
    commit = (~flagged) & (conf >= TAU)
    labels = ybar.copy()
    for i in np.where(flagged)[0]:
        kept.append((pool[i], i, 1))
    # training set = this round's commits + released items
    if released_X:
        Xtr_ = np.concatenate([pool[commit], np.stack(released_X)])
        ytr_ = np.concatenate([labels[commit], np.array(released_y)])
    else:
        Xtr_, ytr_ = pool[commit], labels[commit]
    tele = {"n_commit": int(commit.sum()), "n_released": len(released_X),
            "n_dropped": len(dropped), "n_quar": len(kept),
            "released_acc": (float(np.mean(
                [ry == rt for ry, rt in zip(released_y, released_true)]))
                if released_true else None)}
    return Xtr_, ytr_, kept, dropped, tele


def dropped_accuracy(state, quar_dropped, ytr):
    """Audit-side: accuracy of the committee's CURRENT labels on items it
    dropped (never trained on) - did quarantine discard recoverable items?"""
    if not quar_dropped:
        return None
    Xd = np.stack([q[0] for q in quar_dropped])
    pbar, ybar, conf, flagged, _ = flags_and_probs(Xd, state)
    true = np.array([ytr[q[1]] for q in quar_dropped])
    return float(np.mean(ybar == true))


# ---------------- audit-side ----------------
def disposition_probs_beta(X, state):
    """M2-beta deployed disposition: plain committee at tau."""
    pbar, ybar, conf, flagged, _ = flags_and_probs(X, state)
    return pbar, flagged


def disposition_probs_alpha(X, state, view_X_list):
    """M2-alpha deployed disposition: committee + fixed-view re-exam."""
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


def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def fam_stats(V_list, truth):
    n = len(V_list)
    if n < 2:
        deltas = [float(np.mean(V != truth)) for V in V_list]
        return {"v_mean": 0.0, "v_sd": 0.0, "v_min": 0.0, "v_max": 0.0,
                "contested": 0.0,
                "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
                "delta_min": float(np.min(deltas)) if deltas else 0.0,
                "delta_max": float(np.max(deltas)) if deltas else 0.0}
    pw = [float(np.mean(V_list[a] != V_list[b]))
          for a, b in itertools.combinations(range(n), 2)]
    stack = np.stack(V_list)
    unan = np.all(stack, 0) | ~np.any(stack, 0)
    deltas = [float(np.mean(V != truth)) for V in V_list]
    return {"v_mean": float(np.mean(pw)), "v_sd": float(np.std(pw)),
            "v_min": float(np.min(pw)), "v_max": float(np.max(pw)),
            "contested": float(np.mean(~unan)),
            "delta_mean": float(np.mean(deltas)),
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas))}


def round_stats(state, Xte, yte):
    pbar, ybar, conf, flagged, D = flags_and_probs(Xte, state)
    err = (ybar != yte)
    return {"acc": float(np.mean(~err)),
            "flagged_frac": float(np.mean(flagged)),
            "flagged_err": float(np.mean(err[flagged])) if flagged.any() else None,
            "unflagged_err": float(np.mean(err[~flagged])) if (~flagged).any() else None,
            "accept": float(np.mean(conf >= TAU)),
            "auc_D": auc(D, err)}


ytr_global = None
counts_ref = []


def run_gen(seed, Xtr, ytr, Xte, yte, mode="M2B", trajectory=False):
    """mode: M2B (quarantine), M2A (view re-exam), C1, C2a, C2b, C3.
    Returns (state, traj, counts, bridge, dropped_items)."""
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
    traj, counts = [], []
    quar = []
    bridge = []
    dropped_all = []
    if trajectory:
        traj.append(round_stats(state, Xte, yte))
    if mode == "C1":
        return state, traj, counts, bridge, dropped_all
    for r in range(ROUNDS):
        if mode == "M2B":
            Xtr_, ytr_, quar, dropped, tele = self_round_beta(
                state, mom, rng, Xtr, quar, ytr)
            dropped_all.extend(dropped)
            counts.append(tele["n_commit"] + tele["n_released"])
            bridge.append(tele)
            train_epochs(state, mom, rng, Xtr_, ytr_, SELF_EPOCHS)
        elif mode == "M2A":
            pool, commit, labels, flagged = self_round_alpha(
                state, mom, rng, Xtr)
            counts.append(int(commit.sum()))
            train_epochs(state, mom, rng, pool[commit], labels[commit],
                         SELF_EPOCHS)
        else:  # controls, rate-matched to M2B token's counts
            pool = perturb(Xtr, rng)
            tgt = counts_ref[r] if counts_ref else len(pool) // 2
            if mode == "C3":
                train_epochs(state, mom, rng, pool, ytr_global, SELF_EPOCHS)
                counts.append(len(pool))
            else:
                perm = rng.permutation(len(pool))[:tgt]
                commit = np.zeros(len(pool), dtype=bool); commit[perm] = True
                if mode == "C2a":
                    _, ybar, _, _, _ = flags_and_probs(pool, state)
                    labels = ybar
                else:
                    labels = rng.integers(0, 10, len(pool))
                train_epochs(state, mom, rng, pool[commit], labels[commit],
                             SELF_EPOCHS)
                counts.append(int(commit.sum()))
        if trajectory:
            traj.append(round_stats(state, Xte, yte))
    return state, traj, counts, bridge, dropped_all


def main():
    global ytr_global, counts_ref
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 3
        globals()["M2B_SEEDS"] = [0, 100]
        globals()["M2A_SEEDS"] = [0, 100]
        globals()["R3_SEEDS"] = [200]
        globals()["C4_SEEDS"] = [300]
        globals()["CTRL_SEEDS"] = [0]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    ytr_global = ytr
    view_X = fixed_views(Xte)
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    print(f"data: train={len(Xtr)} test={len(Xte)} quarantine_age={QUAR_MAX_AGE}")

    res = {"config": {"K": K, "tau": TAU, "kappa": KAPPA, "rounds": ROUNDS,
                      "burn": BURN_EPOCHS, "self_epochs": SELF_EPOCHS,
                      "quar_max_age": QUAR_MAX_AGE, "views": VIEWS,
                      "m2b_seeds": len(M2B_SEEDS), "m2a_seeds": len(M2A_SEEDS),
                      "numpy": np.__version__}}

    # ---- M2-beta token: trajectory + deference + bridge audit ----
    tok_state, tok_traj, tok_counts, tok_bridge, tok_dropped = run_gen(
        TOKEN, Xtr, ytr, Xte, yte, mode="M2B", trajectory=True)
    counts_ref = list(tok_counts)
    tok_probs, tok_flags = disposition_probs_beta(Xte, tok_state)
    st2, _, _, _, _ = run_gen(TOKEN, Xtr, ytr, Xte, yte, mode="M2B")
    p2, _ = disposition_probs_beta(Xte, st2)
    res["deference"] = {"bitwise_identical_probs": bool(np.array_equal(p2, tok_probs))}
    res["token_trajectory"] = tok_traj
    res["token_counts"] = tok_counts
    res["bridge_telemetry"] = tok_bridge
    print(f"deference (M2B re-run): {res['deference']['bitwise_identical_probs']}")
    print("M2B traj acc:", [round(t["acc"], 4) for t in tok_traj])
    print("M2B counts :", tok_counts)

    # ---- bridge-efficacy audit (amendment 3), development side ----
    rel_acc = [b["released_acc"] for b in tok_bridge if b["released_acc"] is not None]
    res["bridge_audit"] = {
        "released_acc_mean": float(np.mean(rel_acc)) if rel_acc else None,
        "released_n_total": int(sum(b["n_released"] for b in tok_bridge)),
        "dropped_n_total": len(tok_dropped)}
    # dropped-item accuracy: the FINAL committee's labels on items the bridge
    # permanently discarded (audit-side truth; the system never saw these)
    if tok_dropped:
        Xd = np.stack([q[0] for q in tok_dropped])
        _, ybar_d, _, _, _ = flags_and_probs(Xd, tok_state)
        true_d = np.array([ytr[q[1]] for q in tok_dropped])
        res["bridge_audit"]["dropped_acc_final_committee"] = float(
            np.mean(ybar_d == true_d))
    # still-quarantined items at run end (uncertainty that never resolved)
    # (reconstructed from the last telemetry row)
    res["bridge_audit"]["quar_final_n"] = int(
        tok_bridge[-1]["n_quar"] if tok_bridge else 0)

    # ---- S3 at deployment: plain-committee flagged accuracy (no re-exam) ----
    pbar, ybar, conf, flagged, D = flags_and_probs(Xte, tok_state)
    err = (ybar != yte)
    res["s3_deployment"] = {
        "auc_D": auc(D, err),
        "flagged_frac": float(np.mean(flagged)),
        "flagged_err": float(np.mean(err[flagged])) if flagged.any() else None,
        "unflagged_err": float(np.mean(err[~flagged])),
        "flagged_argmax_acc_plain": float(np.mean(~err[flagged])) if flagged.any() else None,
        "m2alpha_reference": {"flagged_argmax_acc_plain": 0.60,
                              "flagged_argmax_acc_reexamined": 0.4667}}

    # ---- families: M2B, M2A (recomputed), r3, C4 ----
    def family(seeds, mode, disposition):
        Vs, accs = [], []
        for s in seeds:
            st, _, _, _, _ = run_gen(s, Xtr, ytr, Xte, yte, mode=mode)
            if disposition == "beta":
                P, _ = disposition_probs_beta(Xte, st)
            else:
                P, _ = disposition_probs_alpha(Xte, st, view_X)
            Vs.append((P >= TAU).reshape(-1))
            accs.append(float(np.mean(P.argmax(1) == yte)))
            print(f"    {mode} seed={s} ({time.time()-t0:.0f}s)")
        return fam_stats(Vs, truth), np.stack(Vs), float(np.mean(accs))

    m2b_stats, m2b_V, m2b_acc = family(M2B_SEEDS, "M2B", "beta")
    res["m2b_family"] = m2b_stats; res["m2b_acc"] = m2b_acc
    m2a_stats, m2a_V, m2a_acc = family(M2A_SEEDS, "M2A", "alpha")
    res["m2a_family_recomputed"] = m2a_stats; res["m2a_acc"] = m2a_acc
    # r3: plain supervised 70 epochs, plain-committee disposition
    r3_V2 = []
    for s in R3_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS + ROUNDS * SELF_EPOCHS)
        P, _ = disposition_probs_beta(Xte, state)
        r3_V2.append((P >= TAU).reshape(-1))
    r3_stats = fam_stats(r3_V2, truth)
    res["r3_family"] = r3_stats
    c4_V = []
    for s in C4_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, 15)
        P, _ = disposition_probs_beta(Xte, state)
        c4_V.append((P >= TAU).reshape(-1))
    res["c4_family"] = fam_stats(c4_V, truth)
    for nm, st in [("M2B", m2b_stats), ("M2A", m2a_stats),
                   ("r3", r3_stats), ("C4", res["c4_family"])]:
        print(f"{nm}: v={st['v_mean']*100:.3f}% delta={st['delta_mean']*100:.3f}%")

    # ---- three-way residual (amendment 2) ----
    def three_way(V_list, name):
        fresh = np.stack(V_list[1:])
        E = fresh.sum(0) >= (fresh.shape[0] // 2 + 1)
        return {"r_token_vs_ensemble": float(np.mean(V_list[0] != E)),
                "r_fresh_vs_ensemble_mean": float(np.mean(
                    [np.mean(V_list[i] != E) for i in range(1, len(V_list))])),
                "delta_ensemble": float(np.mean(E != truth)),
                "v": fam_stats(V_list, truth)["v_mean"]}
    res["m2b_three_way"] = three_way(list(m2b_V), "M2B")
    res["m2a_three_way"] = three_way(list(m2a_V), "M2A")

    # ---- consensus shift vs r3 ----
    E_b = m2b_V.sum(0) >= (m2b_V.shape[0] // 2 + 1)
    E_a = m2a_V.sum(0) >= (m2a_V.shape[0] // 2 + 1)
    E_r3 = np.stack(r3_V2).sum(0) >= (len(r3_V2) // 2 + 1)
    res["consensus"] = {
        "d_M2B_vs_r3": float(np.mean(E_b != E_r3)),
        "d_M2A_vs_r3": float(np.mean(E_a != E_r3)),
        "d_M2B_vs_M2A": float(np.mean(E_b != E_a))}

    # ---- token decomposition + margin structure (M2B) ----
    Vt = m2b_V[0]
    Eb = np.stack(m2b_V[1:]); Eb = Eb.sum(0) >= (Eb.shape[0] // 2 + 1)
    res["m2b_decomposition"] = {
        "delta_token": float(np.mean(Vt != truth)),
        "spec_typical": float(np.mean((Vt != truth) & (Eb != truth))),
        "idio_error": float(np.mean((Vt != truth) & (Eb == truth))),
        "idio_correct": float(np.mean((Vt == truth) & (Eb != truth)))}
    margin = np.abs(tok_probs.reshape(-1) - TAU)
    ms = []
    for lo, hi in MARGIN_BINS:
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        per = [float(np.mean(Vt[sel] != V[sel])) for V in m2b_V[1:]]
        ms.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                   "mean_d": float(np.mean(per))})
    res["m2b_margin_structure"] = ms

    # ---- controls ----
    ctrls = {}
    for mode in ["C2a", "C2b", "C3"]:
        trajs, finals = [], []
        for s in CTRL_SEEDS:
            st, traj, cnts, _, _ = run_gen(s, Xtr, ytr, Xte, yte, mode=mode,
                                        trajectory=True)
            trajs.append(traj)
            P, _ = disposition_probs_beta(Xte, st)
            finals.append((P >= TAU).reshape(-1))
        ctrls[mode] = {"traj_acc": [[t["acc"] for t in tr] for tr in trajs],
                       "family": fam_stats(finals, truth)}
        print(f"  control {mode}: final acc="
              f"{np.mean(ctrls[mode]['traj_acc'], axis=0)[-1]*100:.2f}%")
    res["controls"] = ctrls

    # ---- amendment 1: frontier report vs the farming families ----
    try:
        farm = json.load(open("/home/z/my-project/scripts/farming_results.json"))
        pts = [(nm, st["v_mean"], st["delta_mean"])
               for nm, st in farm["families"].items()]
        tgt_b = (m2b_stats["v_mean"], m2b_stats["delta_mean"])
        tgt_a = (m2a_stats["v_mean"], m2a_stats["delta_mean"])
        def dom(pts, tgt):
            return [nm for nm, v, d in pts
                    if v >= tgt[0] and d <= tgt[1] and (v > tgt[0] or d < tgt[1])]
        res["frontier_report"] = {
            "m2b_point": {"v": tgt_b[0], "delta": tgt_b[1]},
            "m2a_point": {"v": tgt_a[0], "delta": tgt_a[1]},
            "m2b_dominated_by": dom(pts, tgt_b),
            "m2a_dominated_by": dom(pts, tgt_a),
            "n_farming_families": len(pts)}
        print("frontier: M2B dominated by", res["frontier_report"]["m2b_dominated_by"])
        print("frontier: M2A dominated by", res["frontier_report"]["m2a_dominated_by"])
    except FileNotFoundError:
        res["frontier_report"] = {"error": "farming_results.json not found"}

    res["runtime_s"] = time.time() - t0
    np.savez_compressed("/home/z/my-project/scripts/tmp/m2beta_V.npz",
                        m2b=m2b_V, m2a=m2a_V, r3=np.stack(r3_V2),
                        c4=np.stack(c4_V))
    with open("/home/z/my-project/scripts/m2beta_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m2beta_results.json")


if __name__ == "__main__":
    main()
