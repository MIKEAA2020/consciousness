#!/usr/bin/env python3
"""
M2-GAMMA - THE THIRD-GENERATION BUILD: evaluator redundancy attacking the
responsiveness-auditability tradeoff head-on. Corpus doc 22's experiment.
Ordered at the M2-beta issuance: "a third-generation build targeting the
responsiveness-auditability tradeoff (e.g., evaluator redundancy,
cross-member re-grading)."

THE DESIGN LAW TO BREAK (doc 19): responsiveness and auditability trade off
through path dependence - the channel that repairs S3 (quarantine +
deferred self-re-examination through the EVOLVING committee) is the channel
that scatters S2 (each run's training path diverges -> family widens:
v_M2beta 0.619% vs v_M2alpha 0.523%). Entry is a pair of leaning walls.

THE MECHANISTIC HYPOTHESIS: the scatter compounds because the repair
decision is carried by the very state that development is moving - the
re-grading committee at round r has itself been reshaped by rounds < r, so
release decisions feed back into the training that reshapes the grader.
Evaluator redundancy cuts the loop: carry the grader OUT OF THE PATH.

THE ONE DESIGN CHANGE (single controlled variable vs M2-beta):
  M2-beta's release:  quarantined item re-graded next round by the EVOLVED
    committee; released iff then unflagged & conf >= tau; committee-labeled.
  M2-gamma's release: quarantined item re-graded by a FROZEN REDUNDANT
    PANEL - three time-sliced snapshots of the committee's own burn-in
    (epochs 20/25/30), never trained during the self-phase - released iff
    >= 2 of 3 panel members unflagged AND panel-mean conf >= tau;
    PANEL-labeled (the redundant evaluator carries both the authorization
    and the label). The panel is the system's own past, carried along as
    development-time machinery: redundancy across TIME, out of the training
    path. (Disclosed: this is doc 18's M3 requirement 2 - a repair channel
    outside the damaged norm - instantiated at development scale; gamma
    pre-figures M3-proper's repair channel.)
  Everything else is M2-beta unchanged: burn-in 30 ep; 10 label-free
  rounds; committee flag = disagreement | conf < kappa; commit gate on the
  EVOLVING committee (the S1/S2 self-selection content is untouched);
  drop after 2 quarantined rounds; deployment = plain committee at tau.

PRE-REGISTERED EXPECTATIONS (fixed before execution):
  G1 (S1): accuracy holds >= 96.0%; C2a gap >= 1pp preserved; C2b
      collapses; C1 (frozen) still above all continued regimes.
  G2 (S3): AUC ~ .95; released items more accurate than dropped; no
      deployment-time flagged degradation (plain-committee flagged acc
      ~60%+, vs M2-alpha's re-examined 46.7%).
  G3 (S2 - the tradeoff test): v_M2gamma < v_M2beta = 0.619% (predict
      0.48-0.58%) at delta <= 0.87% - the out-of-path release absorbs part
      of the quarantine scatter. PRE-REGISTERED ALTERNATIVE: if
      v_M2gamma >= v_M2beta, the scatter lives in the SELECTION channel
      (the committee's own flag decides what is quarantined at all), not
      the response channel - the tradeoff's root is one level deeper than
      the repair pathway, and the design law relocates.
  G4: deference control bitwise through the panel loop.
  G5: commit ratchet between M2-alpha's and M2-beta's shapes.
  G6: consensus shift vs r3 ~ 0.3-0.4%.
  G7 (redundancy audit, amendment 5): panel-vs-committee label agreement
      on quarantined items < 100% and panel-release != committee-release
      on a measurable fraction - the evaluators are genuinely non-identical
      (otherwise "redundancy" is a renamed committee).

Usage: python3 m2gamma_build.py [--smoke]
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
PANEL_EPOCHS = [20, 25, 30]             # time-sliced frozen panel
VIEWS, VIEW_SEED = 6, 4242              # VIEWS: M2-alpha arm only
QUAR_MAX_AGE = 2
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
TOKEN = 0
M2G_SEEDS = [0] + list(range(100, 112))   # gamma family (token + 12 fresh)
M2B_SEEDS = [0] + list(range(100, 112))   # beta family, recomputed
M2A_SEEDS = [0] + list(range(100, 112))   # alpha family, recomputed
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


def copy_state(state):
    W1, b1, heads = state
    return [W1.copy(), b1.copy(), [[w.copy() for w in h] for h in heads]]


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


# ---------------- the three response designs ----------------
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
    """M2-beta's round: quarantine + deferred re-grade by the EVOLVED
    committee (the response travels through the evolving state)."""
    released_X, released_y, released_true = [], [], []
    kept, dropped = [], []
    for Xq, ti, age in quar:
        pbar, ybar, conf, flagged, _ = flags_and_probs(Xq[None, :], state)
        if (not flagged[0]) and (conf[0] >= TAU):
            released_X.append(Xq); released_y.append(int(ybar[0]))
            released_true.append(int(ytr[ti]))
        elif age + 1 >= QUAR_MAX_AGE:
            dropped.append((Xq, ti))
        else:
            kept.append((Xq, ti, age + 1))
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, D = flags_and_probs(pool, state)
    commit = (~flagged) & (conf >= TAU)
    labels = ybar.copy()
    for i in np.where(flagged)[0]:
        kept.append((pool[i], i, 1))
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


def self_round_gamma(state, mom, rng, Xtr, quar, ytr, panel):
    """M2-gamma's round: quarantine + re-grade by the FROZEN REDUNDANT
    PANEL (the response travels out of the evolving state's path).
    Panel = list of 3 frozen snapshots. Release iff >= 2/3 unflagged and
    panel-mean conf >= tau; labels = panel-mean argmax."""
    released_X, released_y, released_true = [], [], []
    kept, dropped = [], []
    audit = {"n_quar_regraded": len(quar), "panel_label_agree_committee": None,
             "panel_release_rate": None, "committee_release_rate_would": None,
             "released_acc_committee_label": None}
    if quar:
        Xq = np.stack([q[0] for q in quar])
        unflag_n = np.zeros(len(quar), dtype=int)
        conf_sum = np.zeros(len(quar))
        pbar_sum = np.zeros((len(quar), 10))
        for snap in panel:
            pj, yj, cj, fj, _ = flags_and_probs(Xq, snap)
            unflag_n += (~fj).astype(int)
            conf_sum += cj
            pbar_sum += pj
        panel_conf = conf_sum / len(panel)
        panel_pbar = pbar_sum / len(panel)
        panel_label = panel_pbar.argmax(1)
        release = (unflag_n >= 2) & (panel_conf >= TAU)
        # committee counterfactual on the same items (redundancy audit)
        pc, yc, cc, fc, _ = flags_and_probs(Xq, state)
        would = (~fc) & (cc >= TAU)
        audit["panel_label_agree_committee"] = float(
            np.mean(panel_label == yc))
        audit["panel_release_rate"] = float(np.mean(release))
        audit["committee_release_rate_would"] = float(np.mean(would))
        rel_true = np.array([ytr[q[1]] for q in quar])
        audit["released_acc_committee_label"] = float(
            np.mean(yc[release] == rel_true[release])) if release.any() else None
        for k, (Xqi, ti, age) in enumerate(quar):
            if release[k]:
                released_X.append(Xqi)
                released_y.append(int(panel_label[k]))
                released_true.append(int(ytr[ti]))
            elif age + 1 >= QUAR_MAX_AGE:
                dropped.append((Xqi, ti))
            else:
                kept.append((Xqi, ti, age + 1))
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, D = flags_and_probs(pool, state)
    commit = (~flagged) & (conf >= TAU)
    labels = ybar.copy()
    for i in np.where(flagged)[0]:
        kept.append((pool[i], i, 1))
    if released_X:
        Xtr_ = np.concatenate([pool[commit], np.stack(released_X)])
        ytr_ = np.concatenate([labels[commit], np.array(released_y)])
    else:
        Xtr_, ytr_ = pool[commit], labels[commit]
    tele = {"n_commit": int(commit.sum()), "n_released": len(released_X),
            "n_dropped": len(dropped), "n_quar": len(kept),
            "released_acc": (float(np.mean(
                [ry == rt for ry, rt in zip(released_y, released_true)]))
                if released_true else None),
            **audit}
    return Xtr_, ytr_, kept, dropped, tele


# ---------------- audit-side ----------------
def disposition_probs(X, state):
    """Gamma/beta deployed disposition: plain committee at tau."""
    pbar, ybar, conf, flagged, _ = flags_and_probs(X, state)
    return pbar, flagged


def disposition_probs_alpha(X, state, view_X_list):
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


def burn_in_with_panel(rng, Xtr, ytr):
    """Burn-in training with time-sliced snapshots for the frozen panel."""
    state = init_params(rng)
    mom = zero_moments(state)
    panel = []
    done = 0
    for ep in PANEL_EPOCHS:
        train_epochs(state, mom, rng, Xtr, ytr, ep - done)
        panel.append(copy_state(state))
        done = ep
    return state, mom, panel


def run_gen(seed, Xtr, ytr, Xte, yte, mode="M2G", trajectory=False):
    """mode: M2G (panel release), M2B (beta), M2A (alpha), C1, C2a, C2b, C3.
    Returns (state, traj, counts, bridge, dropped_items)."""
    rng = np.random.default_rng(seed)
    state, mom, panel = burn_in_with_panel(rng, Xtr, ytr)
    traj, counts = [], []
    quar = []
    bridge = []
    dropped_all = []
    if trajectory:
        traj.append(round_stats(state, Xte, yte))
    if mode == "C1":
        return state, traj, counts, bridge, dropped_all
    for r in range(ROUNDS):
        if mode in ("M2G", "M2B"):
            if mode == "M2G":
                Xtr_, ytr_, quar, dropped, tele = self_round_gamma(
                    state, mom, rng, Xtr, quar, ytr, panel)
            else:
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
        else:  # controls, rate-matched to M2G token's counts
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
        globals()["M2G_SEEDS"] = [0, 100]
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
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}")

    res = {"config": {"K": K, "tau": TAU, "kappa": KAPPA, "rounds": ROUNDS,
                      "burn": BURN_EPOCHS, "self_epochs": SELF_EPOCHS,
                      "quar_max_age": QUAR_MAX_AGE, "views": VIEWS,
                      "panel_epochs": PANEL_EPOCHS,
                      "m2g_seeds": len(M2G_SEEDS), "numpy": np.__version__}}

    # ---- M2-gamma token: trajectory + deference + bridge audit ----
    tok_state, tok_traj, tok_counts, tok_bridge, tok_dropped = run_gen(
        TOKEN, Xtr, ytr, Xte, yte, mode="M2G", trajectory=True)
    counts_ref = list(tok_counts)
    tok_probs, tok_flags = disposition_probs(Xte, tok_state)
    st2, _, _, _, _ = run_gen(TOKEN, Xtr, ytr, Xte, yte, mode="M2G")
    p2, _ = disposition_probs(Xte, st2)
    res["deference"] = {"bitwise_identical_probs": bool(np.array_equal(p2, tok_probs))}
    res["token_trajectory"] = tok_traj
    res["token_counts"] = tok_counts
    res["bridge_telemetry"] = tok_bridge
    print(f"deference (M2G re-run): {res['deference']['bitwise_identical_probs']}")
    print("M2G traj acc:", [round(t["acc"], 4) for t in tok_traj])
    print("M2G counts :", tok_counts)

    # ---- bridge-efficacy + redundancy audit (amendments 3+5) ----
    rel_acc = [b["released_acc"] for b in tok_bridge if b["released_acc"] is not None]
    res["bridge_audit"] = {
        "released_acc_mean": float(np.mean(rel_acc)) if rel_acc else None,
        "released_n_total": int(sum(b["n_released"] for b in tok_bridge)),
        "dropped_n_total": len(tok_dropped)}
    if tok_dropped:
        Xd = np.stack([q[0] for q in tok_dropped])
        _, ybar_d, _, _, _ = flags_and_probs(Xd, tok_state)
        true_d = np.array([ytr[q[1]] for q in tok_dropped])
        res["bridge_audit"]["dropped_acc_final_committee"] = float(
            np.mean(ybar_d == true_d))
    res["bridge_audit"]["quar_final_n"] = int(
        tok_bridge[-1]["n_quar"] if tok_bridge else 0)
    # redundancy audit aggregates
    for key in ["panel_label_agree_committee", "panel_release_rate",
                "committee_release_rate_would"]:
        vals = [b[key] for b in tok_bridge if b.get(key) is not None]
        res["bridge_audit"][f"{key}_mean"] = float(np.mean(vals)) if vals else None
    rac = [b["released_acc_committee_label"] for b in tok_bridge
           if b.get("released_acc_committee_label") is not None]
    res["bridge_audit"]["released_acc_committee_label_mean"] = (
        float(np.mean(rac)) if rac else None)
    print("redundancy audit:", {k: v for k, v in res["bridge_audit"].items()
                                if isinstance(v, float)})

    # ---- S3 at deployment: plain-committee flagged accuracy ----
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

    # ---- families: M2G, M2B (recomputed), M2A (recomputed), r3, C4 ----
    def family(seeds, mode, disposition):
        Vs, accs = [], []
        for s in seeds:
            st, _, _, _, _ = run_gen(s, Xtr, ytr, Xte, yte, mode=mode)
            if disposition == "plain":
                P, _ = disposition_probs(Xte, st)
            else:
                P, _ = disposition_probs_alpha(Xte, st, view_X)
            Vs.append((P >= TAU).reshape(-1))
            accs.append(float(np.mean(P.argmax(1) == yte)))
            print(f"    {mode} seed={s} ({time.time()-t0:.0f}s)")
        return fam_stats(Vs, truth), np.stack(Vs), float(np.mean(accs))

    m2g_stats, m2g_V, m2g_acc = family(M2G_SEEDS, "M2G", "plain")
    res["m2g_family"] = m2g_stats; res["m2g_acc"] = m2g_acc
    m2b_stats, m2b_V, m2b_acc = family(M2B_SEEDS, "M2B", "plain")
    res["m2b_family_recomputed"] = m2b_stats; res["m2b_acc"] = m2b_acc
    m2a_stats, m2a_V, m2a_acc = family(M2A_SEEDS, "M2A", "alpha")
    res["m2a_family_recomputed"] = m2a_stats; res["m2a_acc"] = m2a_acc
    r3_V2 = []
    for s in R3_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS + ROUNDS * SELF_EPOCHS)
        P, _ = disposition_probs(Xte, state)
        r3_V2.append((P >= TAU).reshape(-1))
    r3_stats = fam_stats(r3_V2, truth)
    res["r3_family"] = r3_stats
    c4_V = []
    for s in C4_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, 15)
        P, _ = disposition_probs(Xte, state)
        c4_V.append((P >= TAU).reshape(-1))
    res["c4_family"] = fam_stats(c4_V, truth)
    for nm, st in [("M2G", m2g_stats), ("M2B", m2b_stats), ("M2A", m2a_stats),
                   ("r3", r3_stats), ("C4", res["c4_family"])]:
        print(f"{nm}: v={st['v_mean']*100:.3f}% delta={st['delta_mean']*100:.3f}%")

    # ---- three-way residual (amendment 2) ----
    def three_way(V_list):
        fresh = np.stack(V_list[1:])
        E = fresh.sum(0) >= (fresh.shape[0] // 2 + 1)
        return {"r_token_vs_ensemble": float(np.mean(V_list[0] != E)),
                "r_fresh_vs_ensemble_mean": float(np.mean(
                    [np.mean(V_list[i] != E) for i in range(1, len(V_list))])),
                "delta_ensemble": float(np.mean(E != truth)),
                "v": fam_stats(V_list, truth)["v_mean"]}
    res["m2g_three_way"] = three_way(list(m2g_V))
    res["m2b_three_way"] = three_way(list(m2b_V))
    res["m2a_three_way"] = three_way(list(m2a_V))

    # ---- consensus shift vs r3 ----
    E_g = m2g_V.sum(0) >= (m2g_V.shape[0] // 2 + 1)
    E_b = m2b_V.sum(0) >= (m2b_V.shape[0] // 2 + 1)
    E_a = m2a_V.sum(0) >= (m2a_V.shape[0] // 2 + 1)
    E_r3 = np.stack(r3_V2).sum(0) >= (len(r3_V2) // 2 + 1)
    res["consensus"] = {
        "d_M2G_vs_r3": float(np.mean(E_g != E_r3)),
        "d_M2B_vs_r3": float(np.mean(E_b != E_r3)),
        "d_M2A_vs_r3": float(np.mean(E_a != E_r3)),
        "d_M2G_vs_M2B": float(np.mean(E_g != E_b))}

    # ---- token decomposition + margin structure (M2G) ----
    Vt = m2g_V[0]
    Eg = np.stack(m2g_V[1:]); Eg = Eg.sum(0) >= (Eg.shape[0] // 2 + 1)
    res["m2g_decomposition"] = {
        "delta_token": float(np.mean(Vt != truth)),
        "spec_typical": float(np.mean((Vt != truth) & (Eg != truth))),
        "idio_error": float(np.mean((Vt != truth) & (Eg == truth))),
        "idio_correct": float(np.mean((Vt == truth) & (Eg != truth)))}
    margin = np.abs(tok_probs.reshape(-1) - TAU)
    ms = []
    for lo, hi in MARGIN_BINS:
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        per = [float(np.mean(Vt[sel] != V[sel])) for V in m2g_V[1:]]
        ms.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                   "mean_d": float(np.mean(per))})
    res["m2g_margin_structure"] = ms

    # ---- controls ----
    ctrls = {}
    for mode in ["C2a", "C2b", "C3"]:
        trajs, finals = [], []
        for s in CTRL_SEEDS:
            st, traj, cnts, _, _ = run_gen(s, Xtr, ytr, Xte, yte, mode=mode,
                                        trajectory=True)
            trajs.append(traj)
            P, _ = disposition_probs(Xte, st)
            finals.append((P >= TAU).reshape(-1))
        ctrls[mode] = {"traj_acc": [[t["acc"] for t in tr] for tr in trajs],
                       "family": fam_stats(finals, truth)}
        print(f"  control {mode}: final acc="
              f"{np.mean(ctrls[mode]['traj_acc'], axis=0)[-1]*100:.2f}%")
    res["controls"] = ctrls

    # ---- amendment 1: frontier report vs farming + F9 sweep families ----
    try:
        farm = json.load(open("/home/z/my-project/scripts/farming_results.json"))
        pts = [(nm, st["v_mean"], st["delta_mean"])
               for nm, st in farm["families"].items()]
        f9 = json.load(open("/home/z/my-project/scripts/f9_results.json"))
        pts += [(f"f9:{nm}", st["v_mean"], st["delta_mean"])
                for nm, st in f9["families"].items()]
        tgt_g = (m2g_stats["v_mean"], m2g_stats["delta_mean"])
        tgt_b = (m2b_stats["v_mean"], m2b_stats["delta_mean"])
        tgt_a = (m2a_stats["v_mean"], m2a_stats["delta_mean"])
        def dom(pts, tgt):
            return [nm for nm, v, d in pts
                    if v >= tgt[0] and d <= tgt[1] and (v > tgt[0] or d < tgt[1])]
        res["frontier_report"] = {
            "m2g_point": {"v": tgt_g[0], "delta": tgt_g[1]},
            "m2b_point": {"v": tgt_b[0], "delta": tgt_b[1]},
            "m2a_point": {"v": tgt_a[0], "delta": tgt_a[1]},
            "m2g_dominated_by": dom(pts, tgt_g),
            "m2b_dominated_by": dom(pts, tgt_b),
            "m2a_dominated_by": dom(pts, tgt_a),
            "n_farming_families": len(pts)}
        print("frontier: M2G dominated by", res["frontier_report"]["m2g_dominated_by"])
        print("frontier: M2B dominated by", res["frontier_report"]["m2b_dominated_by"])
        print("frontier: M2A dominated by", res["frontier_report"]["m2a_dominated_by"])
    except FileNotFoundError as e:
        res["frontier_report"] = {"error": str(e)}

    res["runtime_s"] = time.time() - t0
    np.savez_compressed("/home/z/my-project/scripts/tmp/m2gamma_V.npz",
                        m2g=m2g_V, m2b=m2b_V, m2a=m2a_V, r3=np.stack(r3_V2),
                        c4=np.stack(c4_V))
    with open("/home/z/my-project/scripts/m2gamma_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m2gamma_results.json")


if __name__ == "__main__":
    main()
