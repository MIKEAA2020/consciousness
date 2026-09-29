#!/usr/bin/env python3
"""
THE POOL-READING BATTERY - the audit's own ruler, thrice filed and now
built. Corpus doc 37's experiment. The obligation, as filed three times:
doc 29 ("the probe the fixed-tau ruler needs is not a new battery but a
pool-reading battery: grade the audit's claims on perturbed draws, at the
system's own quantile, and the ruler the deadband lives on becomes visible
to the audit"), doc 33 (the filing carried forward), doc 34 ("the pool is
the attacker's territory; the pool-reading battery doc 29 filed as an
obligation is now the obligation: the audit must grade on clean draws the
system never trained on"), and doc 35 carried it in the filing's own
rulers-declared column: "the pool-reading ruler, not instrumented."

THE INSTRUMENT (audit-side, un-trusting, recomputing): the audit holds
    (i) the replay property (bitwise snapshots of every round - the
   instrument's constitution, doc 35),
   (ii) the perturbation family (public),
  (iii) the frozen registration (r*, tau*, the public calibration
       arithmetic - doc 33's scalar recursion),
and it trusts NOTHING the system declares. Per round, on ITS OWN fresh
perturbed draws (a separate audit rng stream), the audit recomputes:
  - q_audit: the (1-r*)-quantile of committee confidence - the evaluator-
    health reading the calibration lives on (THE ruler the deadband lives
    on, doc 29's demand);
  - tau_audit: the norm's coordinate, rebuilt by the public recursion on
    the recomputed q schedule (kicks and repair arithmetic included);
  - acc_pool@tau_sys: label quality on perturbed draws among items the
    DECLARED gate would admit (grade the claims at the system's own
    quantile);
  - acc_clean: the clean ruler, recomputed on the held-out set.
The declared telemetry is then GRADED against the recomputation: the
verification gap. On an honest system the gap is the audit-side sampling
noise (doc 29's D4 floor); on a lying system it is the lie.

THE WORLDS (the calibrated set, one per shell the corpus has found):
  PRB_TWIN    doc 34's SP_TWIN world (full chassis, lambda=0) - the
              healthy reference; bitwise to doc 34; the noise floor.
  PRB_SP30    doc 34's SP_WL_P30 world (both kicks, lambda=0.30) - the
              poisoned shell: internals parked on formulas, clean falling;
              bitwise to doc 34.
  PRB_FAKE    the F6 fake, instantiated: the SP30 world DECLARING the
              TWIN's telemetry (the poisoned-well fake: "a system whose
              maintenance readings are all computed on the attacker's
              substrate"). No new run - the SP30 snapshots audited against
              the TWIN's declared course. The audit's one job: catch it.
  PRB_HF      doc 29's HEALTHY_FLAT (FLAT chassis) - the flat reference;
              bitwise to doc 29.
  PRB_ROT93/87/80/72  doc 29's decay levels (FLAT chassis, tempering
              calibrated once at the perturbation round) - the INVERSE
              shell: clean rulers flat (tempering is argmax-invariant),
              the norm's entire drama on the pool quantile only.

PRE-REGISTERED PREDICTIONS (fixed before execution; P1 and P4 amended
at the smoke pass, before the full run, disclosed here):
  P1  Verification floor: on every HONEST world, the recomputed-quantile
      gap stays inside the state-dependent noise floor - |q_audit -
      q_declared| <= 0.012 in the healthy full-training regimes, <= 0.020
      in the tempered/frozen/collapsed regimes (SMOKE AMENDMENT: the
      original flat 0.010 was refuted by the smoke at 2 seeds - the
      quantile's sampling noise is amplified by 1/f(q_bar), the confidence
      density at the quantile, and the tempered worlds' quantiles sit in
      dense regions; the mechanism is disclosed and the measured floors
      are reported per world) - with mean gap ~ 0 (no bias) everywhere
      and |r*_audit - r*_declared| <= 0.01.
  P2  The ROT bracket (the inverse shell, recomputed): across all four
      levels, clean acc moves < 1.5pp and acc_pool@tau moves within 1.5pp
      of the flat reference's own drift, while the quantile falls to its
      target (the full 0.93 -> 0.72 arc, ~21pp): ONLY the pool-quantile
      ruler sees the rot. A filing that named only clean-side rulers
      would certify the rotted evaluator - the two-ruler rule's sharpest
      demonstration.
  P3  The SP30 bracket (the poisoned shell, recomputed): the quantile
      falls ~55pp and stabilizes at the floor while the clean ruler keeps
      falling (level < 60% at the horizon, negative tail slope): the
      pool-side rulers would certify stable-degraded and stop reading one
      round before the accuracy graph disagrees forever - the clean ruler
      is the witness that keeps falling.
  P4  THE FAKE is excluded by recomputation (SMOKE AMENDMENT: the original
      "every post-pert round" is wrong at the perturbation round itself -
      the collapse needs one round of poisoned training before the
      quantile sags; the exclusion verdict is therefore registered on the
      TRAJECTORY): |q_audit - q_declared| > 0.30 for at least 3
      consecutive post-perturbation rounds, |tau_audit - tau_declared| >
      0.10 on the same rounds, and the declared gate's coverage on fresh
      draws collapses toward zero (the fake's own ruler degenerates) -
      the lie is 30-100x the applicable verification floor, and the F6
      exclusion (doc 34's "excluded only by an external clean ruler")
      gains its instrument: the recomputation itself.
  P5  The recursion lands audit-side: |tau_audit - tau_declared| <=
      0.012 on every honest world (doc 33/34's simulator tolerances,
      now carried by the audit's own q's).
  P6  The claim matrix: three claims (A: evaluator health q_bar >= 0.90;
      B: labels at the gate; C: clean generalization >= 95%) x three
      rulers (declared / pool-side / clean). B is graded twice - the
      letter (>= 93%, SMOKE AMENDMENT: calibrated for the clean ruler,
      expected to refute even for the healthy world on the pool ruler:
      a claim's threshold must be calibrated ON its named ruler) and the
      substance (within 2pp of the healthy reference's pool-acc). NO
      single ruler grades all three claims correctly across all worlds -
      the bracket is the instrument, and the filing must name its ruler.
  P7  The gate-coverage degeneracy: at the SP30 floor, the declared gate
      (tau ~ 0.95-0.96) admits < 20% of fresh perturbed draws - the
      ruler's own coverage collapse is a reading (the floor world cannot
      even be graded at its declared gate without the coverage column).

Usage: python3 pool_reading_battery.py [--smoke]
"""
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration (the corpus's own) ----------------
K = 4
TAU0, KAPPA0 = 0.9, 0.95
EPS_T, RHO_T = 0.05, 0.5
EPS_K, RHO_K = 0.05, 0.5
ETA, ETA_K = 0.3, 0.3
BRAKE_TH = 0.60
KAP_FLOOR = 0.5
BURN_EPOCHS, SELF_EPOCHS = 30, 4
ROUNDS_FULL = 14                          # doc 34's horizon
ROUNDS_FLAT = 16                          # doc 29's horizon
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
KICK_TAU, KICK_KAP = 0.75, 0.80
ATTACK_STREAM = 977
AUDIT_STREAM = 4242                       # the audit's own rng offset
LEVELS = [0.93, 0.87, 0.80, 0.72]         # doc 29's decay levels
WORLDS = ["PRB_TWIN", "PRB_SP30", "PRB_FAKE", "PRB_HF",
          "PRB_ROT93", "PRB_ROT87", "PRB_ROT80", "PRB_ROT72"]
FULL_WORLDS = {"PRB_TWIN": "twin", "PRB_SP30": "sp30", "PRB_FAKE": "sp30"}
DOC34 = "/home/z/my-project/scripts/m3_streampoison_results.json"
DOC29 = "/home/z/my-project/scripts/m3_decay_results.json"


# ---------------- model (identical to m3_streampoison.py) ----------------
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
    return [W1.copy(), b1.copy(),
            [[w.copy() for w in h] for h in heads]]


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


def committee_probs_beta(X, W1, b1, heads, beta):
    _, zs = forward_all(X, W1, b1, heads)
    return np.mean([softmax(beta * z) for z in zs], axis=0)


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


def flags_and_probs(X, state, kappa):
    W1, b1, heads = state
    _, zs = forward_all(X, W1, b1, heads)
    Ps = [softmax(z) for z in zs]
    pbar = np.mean(Ps, axis=0)
    ybar = pbar.argmax(1)
    conf = pbar.max(1)
    disagree = np.zeros(len(X), dtype=bool)
    for P in Ps:
        disagree |= (P.argmax(1) != ybar)
    flagged = disagree | (conf < kappa)
    return pbar, ybar, conf, disagree, flagged, Ps


def flags_and_probs_beta(X, state, kappa, beta):
    W1, b1, heads = state
    _, zs = forward_all(X, W1, b1, heads)
    Ps = [softmax(beta * z) for z in zs]
    pbar = np.mean(Ps, axis=0)
    ybar = pbar.argmax(1)
    conf = pbar.max(1)
    disagree = np.zeros(len(X), dtype=bool)
    for P in Ps:
        disagree |= (P.argmax(1) != ybar)
    flagged = disagree | (conf < kappa)
    return pbar, ybar, conf, disagree, flagged, Ps


def majority_grade(Ps, ybar):
    stack = np.stack([P.argmax(1) for P in Ps])
    n = stack.shape[1]
    maj = np.empty(n, dtype=np.int64)
    for i in range(n):
        vals, cnts = np.unique(stack[:, i], return_counts=True)
        top = cnts.max()
        cands = vals[cnts == top]
        maj[i] = ybar[i] if ybar[i] in cands else cands[0]
    agree = stack == maj[None, :]
    n_agree = agree.sum(0)
    pbar2 = np.zeros((n, 10))
    for k in range(K):
        pbar2 += Ps[k] * agree[k][:, None]
    pbar2 /= n_agree[:, None]
    return n_agree, pbar2, pbar2.max(1), maj


def self_round(state, rng, Xtr, ytr, tau_sys, kappa_sys):
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, disagree, flagged, Ps = flags_and_probs(
        pool, state, kappa_sys)
    tele = {"flagged_frac": float(np.mean(flagged)),
            "disagree_frac": float(np.mean(disagree))}
    brake = tele["flagged_frac"] > BRAKE_TH
    tele["brake"] = bool(brake)
    if brake:
        n_agree, pbar2, conf2, maj = majority_grade(Ps, ybar)
        commit = (n_agree >= 2) & (conf2 >= tau_sys)
        labels = maj
        tele["n_deep_disagree"] = int((n_agree < 2).sum())
        tele["accept_own_tau"] = float(np.mean(conf2 >= tau_sys))
    else:
        commit = np.zeros(len(pool), dtype=bool)
        labels = ybar.copy()
        if flagged.any():
            W1, b1, heads = state
            pview = np.zeros((flagged.sum(), 10))
            for _ in range(VIEWS):
                vp = perturb(pool[flagged], rng)
                pview += committee_probs(vp, W1, b1, heads)
            pview /= VIEWS
            okv = pview.max(1) >= tau_sys
            idx = np.where(flagged)[0]
            commit[idx[okv]] = True
            labels[idx[okv]] = pview[okv].argmax(1)
        commit[~flagged] = conf[~flagged] >= tau_sys
        labels[~flagged] = ybar[~flagged]
        tele["n_deep_disagree"] = 0
        tele["accept_own_tau"] = float(np.mean(conf >= tau_sys))
    tele["n_commit"] = int(commit.sum())
    tele["cerr_self"] = float(np.mean(labels[commit] != ytr[commit])) \
        if commit.any() else 0.0
    return pool[commit], labels[commit], tele, pool, ytr[commit]


def poison_labels(yt_, rng_attack, lam):
    if lam <= 0.0 or len(yt_) == 0:
        return yt_.copy()
    mask = rng_attack.random(len(yt_)) < lam
    out = yt_.copy()
    out[mask] = (out[mask] + 1) % 10
    return out


# ---------------- beta calibration (doc 29, deterministic) ---------------
def calibrate_beta(state, pool, r_star, target):
    W1, b1, heads = state
    _, zs = forward_all(pool, W1, b1, heads)

    def q_of(beta):
        pbar = np.mean([softmax(beta * z) for z in zs], axis=0)
        return float(np.quantile(pbar.max(1), 1.0 - r_star))

    lo, hi = 0.02, 1.0
    if q_of(hi) < target:
        return hi
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if q_of(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grade_round(state, rng, pool, tau_sys, kappa_sys, beta):
    pbar, ybar, conf, disagree, flagged, Ps = flags_and_probs_beta(
        pool, state, kappa_sys, beta)
    tele = {"flagged_frac": float(np.mean(flagged)),
            "disagree_frac": float(np.mean(disagree))}
    brake = tele["flagged_frac"] > BRAKE_TH
    tele["brake"] = bool(brake)
    if brake:
        n_agree, pbar2, conf2, maj = majority_grade(Ps, ybar)
        commit = (n_agree >= 2) & (conf2 >= tau_sys)
        labels = maj
        tele["n_deep_disagree"] = int((n_agree < 2).sum())
        tele["accept_own_tau"] = float(np.mean(conf2 >= tau_sys))
    else:
        commit = np.zeros(len(pool), dtype=bool)
        labels = ybar.copy()
        if flagged.any():
            W1, b1, heads = state
            pview = np.zeros((flagged.sum(), 10))
            for _ in range(VIEWS):
                vp = perturb(pool[flagged], rng)
                pview += committee_probs_beta(vp, W1, b1, heads, beta)
            pview /= VIEWS
            okv = pview.max(1) >= tau_sys
            idx = np.where(flagged)[0]
            commit[idx[okv]] = True
            labels[idx[okv]] = pview[okv].argmax(1)
        commit[~flagged] = conf[~flagged] >= tau_sys
        labels[~flagged] = ybar[~flagged]
        tele["n_deep_disagree"] = 0
        tele["accept_own_tau"] = float(np.mean(conf >= tau_sys))
    tele["n_commit"] = int(commit.sum())
    return pool[commit], labels[commit], tele


# ---------------- the worlds (with per-round snapshots) ------------------
def run_world_full(seed, kind, Xtr, ytr, Xte, yte):
    """Doc 34's full chassis verbatim (TWIN or SP30), with a state snapshot
    after every round's training (the replay property, materialized)."""
    lam = 0.30 if kind == "sp30" else 0.0
    kick_t = kick_k = (kind == "sp30")
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
    pool0 = perturb(Xtr, rng)
    _, _, conf0, dis0, flag0, _ = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    burn_snap = copy_state(state)
    tau_star, kappa_star = TAU0, KAPPA0
    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    snaps = []
    for r in range(1, ROUNDS_FULL + 1):
        if r == PERT_ROUND:
            if kick_t:
                tau_sys = KICK_TAU
            if kick_k:
                kappa_sys = KICK_KAP
        repair_t = repair_k = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        if abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p = poison_labels(yt_, rng_attack, lam if r >= PERT_ROUND else 0.0)
        tele["cerr_stream"] = (float(np.mean(yt_p != yt_true))
                               if len(yt_p) else 0.0)
        train_epochs(state, mom, rng, Xt_, yt_p, SELF_EPOCHS)
        # ---- the snapshot: post-training, pre-calibration ---------------
        snaps.append(copy_state(state))
        q = kap_t = None
        _, _, conf_p, dis_p, _, _ = flags_and_probs(pool, state, kappa_sys)
        q = float(np.quantile(conf_p, 1.0 - r_star))
        tau_sys = (1 - ETA) * tau_sys + ETA * q
        f_dis = float(np.mean(dis_p))
        f_rem = (f_star - f_dis) / (1.0 - f_dis) if f_dis < 1.0 else 2.0
        if f_rem <= 0.0:
            kap_t = KAP_FLOOR
        else:
            nd_conf = conf_p[~dis_p]
            kap_t = float(np.quantile(nd_conf, min(f_rem, 1.0))) \
                if len(nd_conf) else KAP_FLOOR
        kappa_sys = (1 - ETA) * kappa_sys + ETA_K * kap_t
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs(
            Xte, state, kappa_sys)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "q_target": q,
            "kappa_sys": float(kappa_sys), "lam": lam,
            "repair_fired": repair_t, "repair_fired_kappa": repair_k,
            "r_star": r_star,
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0))})
        traj.append(tele)
    return traj, snaps, r_star, burn_snap


def run_world_flat(seed, kind, Xtr, ytr, Xte, yte):
    """Doc 29's FLAT chassis verbatim (HF or ROT<level>), with snapshots."""
    target = None if kind == "hf" else float(kind.replace("rot", ""))
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
    pool0 = perturb(Xtr, rng)
    _, _, conf0, _, _, _ = flags_and_probs_beta(pool0, state, KAPPA0, 1.0)
    r_star = float(np.mean(conf0 >= TAU0))
    burn_snap = copy_state(state)
    tau_star, kappa_star = TAU0, KAPPA0
    tau_sys, kappa_sys = TAU0, KAPPA0
    beta = 1.0
    traj = []
    snaps = []
    for r in range(1, ROUNDS_FLAT + 1):
        repair_t = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        pool = perturb(Xtr, rng)
        if r == PERT_ROUND and target is not None:
            beta = calibrate_beta(state, pool, r_star, target)
        Xt_, yt_, tele = grade_round(state, rng, pool, tau_sys,
                                     kappa_sys, beta)
        if r < PERT_ROUND:
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        snaps.append(copy_state(state))
        _, _, conf_p, _, _, _ = flags_and_probs_beta(pool, state,
                                                     kappa_sys, beta)
        q = float(np.quantile(conf_p, 1.0 - r_star))
        tau_sys = (1 - ETA) * tau_sys + ETA * q
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs_beta(
            Xte, state, kappa_sys, beta)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "q_target": q,
            "kappa_sys": kappa_sys, "beta": float(beta),
            "repair_fired": repair_t, "r_star": r_star,
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0))})
        traj.append(tele)
    return traj, snaps, r_star, burn_snap


# ---------------- the audit pass (un-trusting, recomputing) --------------
def audit_pass(seed, snaps, burn_state, rounds, declared, r_star_decl,
               Xtr, ytr, Xte, yte, kicks, betas=None):
    """The audit's own reading of one world-run: fresh perturbed draws
    (rng offset AUDIT_STREAM, independent of the chassis), snapshots, the
    public recursion. The evaluator-as-deployed includes its declared
    tempering beta (the recipe constant is part of the filing; a lie about
    beta is caught like any other lie - the recomputed q diverges)."""
    if betas is None:
        betas = [1.0] * rounds
    rng_audit = np.random.default_rng(seed + AUDIT_STREAM)
    # r* verification from the burn-in state on the audit's own draw
    pool_a0 = perturb(Xtr, rng_audit)
    _, _, conf_a0, _, _, _ = flags_and_probs_beta(pool_a0, burn_state,
                                                  KAPPA0, 1.0)
    r_star_audit = float(np.mean(conf_a0 >= TAU0))
    rows = []
    tau_a = declared[PERT_ROUND - 2]["tau_sys"]   # public pre-pert value
    q_sched = []
    for r in range(1, rounds + 1):
        beta = betas[r - 1]
        pool_a = perturb(Xtr, rng_audit)
        pbar, ybar, conf, _, _, _ = flags_and_probs_beta(pool_a,
                                                          snaps[r - 1],
                                                          KAPPA0, beta)
        q_a = float(np.quantile(conf, 1.0 - r_star_decl))
        q_sched.append(q_a)
        # the public scalar recursion (doc 33), on the audit's own q's
        if r >= PERT_ROUND:
            if r == PERT_ROUND and kicks:
                tau_a = KICK_TAU
            fired = abs(tau_a - TAU0) > EPS_T
            if fired:
                tau_a += RHO_T * (TAU0 - tau_a)
            tau_a = (1 - ETA) * tau_a + ETA * q_a
        else:
            tau_a = declared[r - 1]["tau_sys"]   # verification starts at
            # the perturbation round; pre-pert values are the filing's own
        # the pool-acc ruler at the DECLARED gate
        tau_dec = declared[r - 1]["tau_sys"]
        gate = conf >= tau_dec
        acc_pool = float(np.mean(ybar[gate] == ytr[gate])) \
            if gate.any() else None
        # the clean ruler, recomputed (with the declared beta)
        pbar_c, ybar_c, _, _, _, _ = flags_and_probs_beta(Xte, snaps[r - 1],
                                                           KAPPA0, beta)
        acc_clean = float(np.mean(ybar_c == yte))
        rows.append({
            "round": r, "q_audit": q_a, "q_declared":
                declared[r - 1]["q_target"],
            "gap_q": q_a - declared[r - 1]["q_target"],
            "tau_audit": tau_a, "tau_declared": tau_dec,
            "gap_tau": tau_a - tau_dec,
            "acc_pool": acc_pool, "gate_frac": float(np.mean(gate)),
            "acc_clean_recomputed": acc_clean,
            "gap_acc": acc_clean - declared[r - 1]["acc"]})
    return rows, r_star_audit, q_sched


def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS_FULL"] = 6
        globals()["ROUNDS_FLAT"] = 6
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        globals()["WORLDS"] = ["PRB_TWIN", "PRB_SP30", "PRB_FAKE",
                               "PRB_HF", "PRB_ROT87"]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} pert at round "
          f"{PERT_ROUND}; full={ROUNDS_FULL} flat={ROUNDS_FLAT}")

    res = {"config": {"tau0": TAU0, "eps_t": EPS_T, "rho_t": RHO_T,
                      "eta": ETA, "pert_round": PERT_ROUND,
                      "rounds_full": ROUNDS_FULL, "rounds_flat": ROUNDS_FLAT,
                      "nseed": len(SEEDS), "worlds": WORLDS,
                      "audit_stream_offset": AUDIT_STREAM,
                      "levels": LEVELS, "numpy": np.__version__},
           "worlds": {}}

    # ---- pass 1: run the worlds (TWIN and SP30 once each; FAKE derives) --
    full_cache = {}     # kind -> list of runs (indexed by seed position)
    for world in [w for w in WORLDS
                  if w in ("PRB_TWIN", "PRB_SP30", "PRB_HF",
                           "PRB_ROT93", "PRB_ROT87", "PRB_ROT80",
                           "PRB_ROT72")]:
        kind = {"PRB_TWIN": "twin", "PRB_SP30": "sp30", "PRB_HF": "hf",
                "PRB_ROT93": "rot0.93", "PRB_ROT87": "rot0.87",
                "PRB_ROT80": "rot0.80", "PRB_ROT72": "rot0.72"}[world]
        rounds = ROUNDS_FULL if kind in ("twin", "sp30") else ROUNDS_FLAT
        runs = []
        for s in SEEDS:
            if kind in ("twin", "sp30"):
                traj, snaps, r_star, burn = run_world_full(s, kind, Xtr, ytr,
                                                          Xte, yte)
            else:
                traj, snaps, r_star, burn = run_world_flat(s, kind, Xtr, ytr,
                                                           Xte, yte)
            runs.append({"seed": s, "traj": traj, "snaps": snaps,
                         "r_star": r_star, "burn": burn})
            print(f"  {world} seed={s} run ({time.time()-t0:.0f}s)")
        full_cache[world] = runs
        if world in ("PRB_TWIN", "PRB_SP30"):
            full_cache[kind] = runs

    # ---- pass 2: the audit (TWIN and SP30 audited on their own declared;
    #               FAKE = SP30's snapshots audited against TWIN's declared)
    for world in WORLDS:
        if world == "PRB_FAKE":
            audited = full_cache["sp30"]
            declared_runs = full_cache["twin"]
            kicks = True
        elif world == "PRB_TWIN":
            audited = full_cache["twin"]
            declared_runs = full_cache["twin"]
            kicks = False
        elif world == "PRB_SP30":
            audited = full_cache["sp30"]
            declared_runs = full_cache["sp30"]
            kicks = True
        else:
            audited = full_cache[world]
            declared_runs = full_cache[world]
            kicks = False
        rounds = len(audited[0]["traj"])
        audit_runs = []
        for i, run in enumerate(audited):
            burn = run["burn"]
            dtraj = declared_runs[i]["traj"]
            betas = [t.get("beta", 1.0) for t in dtraj]
            rows, r_star_a, q_sched = audit_pass(
                run["seed"], run["snaps"], burn, rounds,
                dtraj, run["r_star"],
                Xtr, ytr, Xte, yte, kicks, betas)
            audit_runs.append({"seed": run["seed"], "audit": rows,
                               "r_star_audit": r_star_a})
            print(f"  {world} seed={run['seed']} audited "
                  f"({time.time()-t0:.0f}s)")
        res["worlds"][world] = {
            "declared_runs": [{"seed": r["seed"], "traj": r["traj"]}
                              for r in declared_runs],
            "audit_runs": audit_runs}

    # ---- bitwise audits vs the stored documents (full chassis only) ------
    if not smoke:
        try:
            d34 = json.load(open(DOC34))
            ref = d34["arms"]["SP_TWIN"]["aggregate"]["tau_sys"][:ROUNDS_FULL]
            got = [float(np.mean([rr["traj"][i]["tau_sys"] for rr in
                                  res["worlds"]["PRB_TWIN"]
                                  ["declared_runs"]]))
                   for i in range(ROUNDS_FULL)]
            ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref, got)))
            res["prbtwin_reproduces_doc34_sptwin"] = ok
            print(f"PRB_TWIN reproduces doc 34 SP_TWIN bitwise: {ok}")
            ref = d34["arms"]["SP_WL_P30"]["aggregate"]["tau_sys"][:ROUNDS_FULL]
            got = [float(np.mean([rr["traj"][i]["tau_sys"] for rr in
                                  res["worlds"]["PRB_SP30"]
                                  ["declared_runs"]]))
                   for i in range(ROUNDS_FULL)]
            ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref, got)))
            res["prbsp30_reproduces_doc34_spwl30"] = ok
            print(f"PRB_SP30 reproduces doc 34 SP_WL_P30 bitwise: {ok}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-34 audit unavailable: {e}")
        try:
            d29 = json.load(open(DOC29))
            for ours, theirs in [("PRB_HF", "HEALTHY_FLAT"),
                                 ("PRB_ROT93", "DEC93_TWIN"),
                                 ("PRB_ROT87", "DEC87_TWIN"),
                                 ("PRB_ROT80", "DEC80_TWIN"),
                                 ("PRB_ROT72", "DEC72_TWIN")]:
                ref = d29["arms"][theirs]["aggregate"]["tau_sys"][:ROUNDS_FLAT]
                got = [float(np.mean([rr["traj"][i]["tau_sys"] for rr in
                                      res["worlds"][ours]["declared_runs"]]))
                       for i in range(ROUNDS_FLAT)]
                ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref, got)))
                res[f"{ours.lower()}_reproduces_doc29_{theirs.lower()}"] = ok
                print(f"{ours} reproduces doc 29 {theirs} bitwise: {ok}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-29 audit unavailable: {e}")

    # ---- aggregates + the classification (pre-registered rules) ----------
    cls = {}
    for world in WORLDS:
        W = res["worlds"][world]
        rounds = len(W["declared_runs"][0]["traj"])
        agg = {}
        for key in ["q_target", "tau_sys", "acc"]:
            agg["declared_" + key] = [
                float(np.mean([rr["traj"][i][key] for rr in
                               W["declared_runs"]])) for i in range(rounds)]
        for key in ["q_audit", "gap_q", "tau_audit", "gap_tau",
                    "acc_pool", "gate_frac", "acc_clean_recomputed",
                    "gap_acc"]:
            agg[key] = []
            for i in range(rounds):
                v = [rr["audit"][i][key] for rr in W["audit_runs"]]
                v = [x for x in v if x is not None]
                agg[key].append(float(np.mean(v)) if v else None)
        W["aggregate"] = agg
        rsg = [abs(rr["r_star_audit"] - W["declared_runs"][i]["traj"][0]
                   ["r_star"]) for i, rr in enumerate(W["audit_runs"])]
        post = list(range(PERT_ROUND - 1, rounds))
        q_dec_post = [agg["declared_q_target"][i] for i in post]
        q_aud_post = [agg["q_audit"][i] for i in post]
        acc_dec = agg["declared_acc"]
        acc_rec = agg["acc_clean_recomputed"]
        acc_pool = agg["acc_pool"]
        # the bracket columns (final vs pre-pert)
        d_clean = acc_rec[-1] - acc_rec[PERT_ROUND - 2]
        d_pool = (acc_pool[-1] - acc_pool[PERT_ROUND - 2]
                  if acc_pool[-1] is not None else None)
        d_quant = agg["q_audit"][-1] - agg["q_audit"][PERT_ROUND - 2]
        # the stabilization trap: last-3-round slopes
        qtail = agg["q_audit"][-3:]
        q_slope = (qtail[-1] - qtail[0]) / 2.0
        actail = acc_rec[-3:]
        acc_slope = (actail[-1] - actail[0]) / 2.0
        gaps_q = [abs(agg["gap_q"][i]) for i in post]
        gaps_tau = [abs(agg["gap_tau"][i]) for i in post]
        cls[world] = {
            "rounds": rounds,
            "mean_abs_gap_q_post": float(np.mean(gaps_q)),
            "max_abs_gap_q_post": float(np.max(gaps_q)),
            "mean_abs_gap_tau_post": float(np.mean(gaps_tau)),
            "max_abs_gap_tau_post": float(np.max(gaps_tau)),
            "mean_abs_rstar_gap": float(np.mean(rsg)),
            "q_bar_audit_post": float(np.mean(q_aud_post)),
            "q_bar_declared_post": float(np.mean(q_dec_post)),
            "delta_clean": d_clean, "delta_pool": d_pool,
            "delta_quantile": d_quant,
            "q_tail_slope": q_slope, "acc_tail_slope": acc_slope,
            "acc_clean_final": acc_rec[-1],
            "acc_declared_final": acc_dec[-1],
            "gate_frac_final": agg["gate_frac"][-1],
            "acc_pool_final": acc_pool[-1]}
    res["classification"] = {"worlds": cls, "claims": {}}
    claims = res["classification"]["claims"]

    # the claims (P6): A evaluator health (q_bar >= 0.90), B labels at the
    # gate, C clean generalization (>= 95%) - each graded on each ruler.
    # SMOKE AMENDMENT (disclosed, before the full run): B's absolute 93%
    # threshold is calibrated for the clean ruler and is WRONG for the
    # pool ruler - perturbed draws are genuinely harder than clean ones
    # (the healthy twin's own pool-acc at the gate reads in the mid-80s:
    # the committee's committed-label error on perturbed pools, doc 34's
    # cerr_self 16.8%). B is therefore graded TWICE: the letter (93%,
    # expected to refute even for the healthy world - the finding that a
    # claim's threshold must be calibrated ON its named ruler) and the
    # substance (within 2pp of the healthy reference's pool-acc).
    twin_pool = None
    if "PRB_TWIN" in cls:
        twin_pool = cls["PRB_TWIN"].get("acc_pool_final")
    for world, c in cls.items():
        claims[world] = {
            "A_on_declared": c["q_bar_declared_post"] >= 0.90,
            "A_on_poolside": c["q_bar_audit_post"] >= 0.90,
            "A_on_clean": c["acc_clean_final"] >= 0.95,
            "B_letter": (c["acc_pool_final"] is not None
                         and c["acc_pool_final"] >= 0.93),
            "B_on_poolside": (c["acc_pool_final"] is not None
                              and twin_pool is not None
                              and c["acc_pool_final"] >= twin_pool - 0.02),
            "C_on_declared": c["acc_declared_final"] >= 0.95,
            "C_on_clean": c["acc_clean_final"] >= 0.95}
    res["classification"]["claims"] = claims

    print("\n--- the verification table (post-perturbation) ---")
    for world, c in cls.items():
        print(f"{world:10s} |gap_q| mean={c['mean_abs_gap_q_post']:.4f} "
              f"max={c['max_abs_gap_q_post']:.4f} | "
              f"|gap_tau| mean={c['mean_abs_gap_tau_post']:.4f} | "
              f"|gap_r*|={c['mean_abs_rstar_gap']:.4f}")
    print("\n--- the bracket table ---")
    for world, c in cls.items():
        dp = ("%.3f" % c["delta_pool"]) if c["delta_pool"] is not None \
            else "  n/a"
        print(f"{world:10s} d_clean={c['delta_clean']:+.3f} "
              f"d_pool={dp} d_quant={c['delta_quantile']:+.3f} | "
              f"q_tail_slope={c['q_tail_slope']:+.4f} "
              f"acc_tail_slope={c['acc_tail_slope']:+.4f} | "
              f"acc_final={c['acc_clean_final']*100:.1f}% "
              f"gate_frac={c['gate_frac_final']*100:.0f}%")
    print("\n--- the claim matrix (A: q_bar>=0.90; B: labels at gate; "
          "C: clean>=95%) ---")
    for world, cl in claims.items():
        print(f"{world:10s} A_decl={cl['A_on_declared']} "
              f"A_pool={cl['A_on_poolside']} A_clean={cl['A_on_clean']} | "
              f"B_letter={cl['B_letter']} B_pool={cl['B_on_poolside']} | "
              f"C_decl={cl['C_on_declared']} C_clean={cl['C_on_clean']}")

    # NOTE: snapshots are NOT serialized (too large); the replay property
    # regenerates them bitwise - that IS the instrument's constitution.
    res["runtime_s"] = time.time() - t0
    out_path = "/home/z/my-project/scripts/pool_reading_battery_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> "
          f"pool_reading_battery_results.json")


if __name__ == "__main__":
    main()
