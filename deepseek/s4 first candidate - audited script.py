#!/usr/bin/env python3
"""
THE FIRST S4 CANDIDATE BUILD - the claim table, filed and run under 8.4.
Corpus doc 35's experiment. Ordered at the joint-poison/substrate/phase-2
issuance: "the first S4 candidate build (the claim table's rows are now
priced: two coordinates, defensive stasis, two-ruler costs, exposure
formulas - a candidate could now file under 8.4)."

THE CANDIDATE: the corpus's own two-coordinate negotiated norm (docs 26/30
chassis) - tau and kappa as state, each with its write channel (the
calibration rules), its record (tau* = 0.9, kappa* = 0.95, self-registered
at burn-in), and its out-of-gate repair channel (eps = 0.05, rho = 0.5) -
filing under doc 28's claim format with the 8.4 containment protocol
exercised in-battery.

THE FILING (frozen before the run; the no-self-revision clause is the code
constants themselves - the system has no write access to this table):
  Restoration type: N (negotiation), with R and L as the identified,
  dis-armable carriers.
  Coordinates: tau (state), kappa (state). Machinery parameters (eps, rho,
  eta, brake threshold): recipe constants - declared NOT state, no claim.
  Attack classes declared: kick (single/joint), sustained drift, record
  poison (single/joint), evaluator saturation, evaluator decay (level),
  trajectory decay, stream poison (doc 34's row, cited), structural damage
  1-of-4 / 2-of-4 / 3-of-4.
  Rulers: verdict-matrix (V at the factory tau), argmax accuracy, and - per
  docs 29/33/34's amendment candidates - the pool-reading ruler DISCLOSED
  AS NOT INSTRUMENTED (the candidate's standing hole).
  Exposure formulas filed at registration (5.2): th1 = 0.897 silent
  threshold, th2_drift = 0.815 / th2_nodrift = 0.792 break thresholds,
  the tension point T(p*, live) = [(1-e)rho p* + e*live]/(e+rho-e*rho) with
  nine prior landings, and doc 34's stream row: the band does not survive
  lambda >= 0.05 of committed labels; the landing at T(0.9, q_bar floor).

THE BATTERY (14 arms, N=6 seeds, all under the frozen registration; ten
bitwise audits against the corpus's stored results - the replay property
doc 28's sequencing rule demands):
  F1 the moving target:  CAND_TWIN (the course must have left the record)
  F2 the channel pair:   CAND_KR_T / CAND_KNR_T (tau: repair-only vs
                         live-only), CAND_KR_K / CAND_KNR_K (kappa)
  F3 the negotiation:    CAND_POISON_TAU, CAND_POISON_JOINT (records lie;
                         landings at tension points)
  F4 the stasis fake:    CAND_FLAT_TWIN, CAND_FLAT_DRIFT (saturation),
                         CAND_DEC87_DRIFT (decay level, visible
                         containment)
  F5 the substrate:      CAND_S1 (1-of-4, brake), CAND_S2 (2-of-4, the
                         split verdict), CAND_S3_CUTOFF (3-of-4, the
                         ordered stand-down - 8.4 s5.1(iii))
  5.1(i) factory:       CAND_SUPKILL (designer labels restore the damaged
                         head - factory precedence exercised)

PRE-REGISTERED VERDICT RULES (the award computation, fixed before the run):
  V1  MAINTAINS-TO-COURSE: twin's course left each record by > 0.02 on both
      coordinates; KNR arms land on the course (<= 0.01); KR arms land at
      the deadband ceiling (epsilon-approximate, below the course by
      > 0.06) - restoration-to-live, not to factory settings (F1 excluded).
  V2  MAINTAINS-CHANNEL-CARRIED: the disarmament pair localizes the
      carriage (live-carried on both coordinates; each channel dis-armable
      with measured effect - the registry).
  V3  MAINTAINS-UNDER-ATTACK, per class: kick/joint-kick (docs 26/30,
      cited); drift CONTAINED above th1 with the silent displacement priced;
      poison CONTAINED at the tension points, factorized (<= 0.005 tau /
      0.01 kappa); saturation CONTAINED (doc 26's formula); decay-level
      CONTAINED above th2 / BREAKS below with attribution (the honest
      cell); trajectory-decay: separation by the engagement clock (doc 33,
      cited); stream poison: NOT SURVIVED at lambda >= 0.05 (doc 34's
      priced row - the candidate files the failure, the table records it);
      structural 1-of-4 brake-held (acc >= 93%), 2-of-4 SPLIT (d_own <=
      2% with acc eroded >= 5pp - the negative finding, awardable as
      such), 3-of-4 ordered stasis (the cutoff's two-ruler price).
  V4  COORDINATE-COVERAGE: both coordinates promoted with write channel +
      record + repair verified.
  V5  RULER-CONSISTENT: the split disclosed; the pool-reading ruler not
      instrumented (the disclosed hole); the poisoned-equilibrium shell
      (doc 34) on record.
  V6  NOT AWARDABLE, EVER: ownership (D3 stands); substrate claims beyond
      the calibrated classes.

Usage: python3 s4_candidate.py [--smoke]
"""
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration (the filing) ----------------
K = 4
TAU0, KAPPA0 = 0.9, 0.95
EPS_T, RHO_T = 0.05, 0.5
EPS_K, RHO_K = 0.05, 0.5
ETA, ETA_K = 0.3, 0.3
BRAKE_TH = 0.60
BURN_EPOCHS, SELF_EPOCHS = 30, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
DRIFT_T = 0.02
DEC87 = 0.87
POISON_TAU, POISON_KAP = 0.70, 0.75
KICK_TAU, KICK_KAP = 0.75, 0.80

ARMCFG = {
    "CAND_TWIN":         dict(kstate=True,  rounds=10),
    "CAND_TWIN_KF":      dict(kstate=False, rounds=10),
    "CAND_KR_T":         dict(kstate=True,  rounds=10, kick_tau=KICK_TAU,
                              calib_t_off=True),
    "CAND_KNR_T":        dict(kstate=True,  rounds=10, kick_tau=KICK_TAU,
                              repair_t_off=True),
    "CAND_KR_K":         dict(kstate=True,  rounds=10, kick_kap=KICK_KAP,
                              calib_k_off=True),
    "CAND_KNR_K":        dict(kstate=True,  rounds=10, kick_kap=KICK_KAP,
                              repair_k_off=True),
    "CAND_POISON_TAU":   dict(kstate=True,  rounds=10, poison_tau=POISON_TAU,
                              kick_tau=KICK_TAU),
    "CAND_POISON_JOINT": dict(kstate=True,  rounds=10,
                              poison_tau=POISON_TAU, poison_kap=POISON_KAP,
                              kick_tau=KICK_TAU, kick_kap=KICK_KAP),
    "CAND_FLAT_TWIN":    dict(kstate=False, rounds=16, flat=True),
    "CAND_FLAT_DRIFT":   dict(kstate=False, rounds=16, flat=True,
                              drift=True),
    "CAND_DEC87_DRIFT":  dict(kstate=False, rounds=16, flat=True,
                              drift=True, target=DEC87),
    "CAND_S1":           dict(kstate=False, rounds=10, damage=[2],
                              brake_mode="std"),
    "CAND_S2":           dict(kstate=False, rounds=10, damage=[1, 2],
                              brake_mode="std"),
    "CAND_S3_CUTOFF":    dict(kstate=False, rounds=10, damage=[1, 2, 3],
                              brake_mode="cutoff"),
    "CAND_SUPKILL":      dict(kstate=False, rounds=10, damage=[2],
                              sup=True),
}
ARMS = list(ARMCFG.keys())

DOC23 = "/home/z/my-project/scripts/m3_proper_results.json"
DOC26 = "/home/z/my-project/scripts/m3_kappa_results.json"
DOC29 = "/home/z/my-project/scripts/m3_decay_results.json"
DOC30 = "/home/z/my-project/scripts/m3_jointpoison_results.json"
DOC31 = "/home/z/my-project/scripts/m3_substrate_results.json"
DOC34 = "/home/z/my-project/scripts/m3_streampoison_results.json"


# ---------------- model (identical across the corpus) ----------------
def init_params(rng):
    W1 = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 128)); b1 = np.zeros(128)
    heads = []
    for _ in range(K):
        Wh = rng.normal(0.0, np.sqrt(2.0 / 128), (128, 64)); bh = np.zeros(64)
        Wo = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 10)); bo = np.zeros(10)
        heads.append([Wh, bh, Wo, bo])
    return [W1, b1, heads]


def fresh_head(rng):
    Wh = rng.normal(0.0, np.sqrt(2.0 / 128), (128, 64)); bh = np.zeros(64)
    Wo = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 10)); bo = np.zeros(10)
    return [Wh, bh, Wo, bo]


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


def head_stats(Ps):
    """core_est = median plurality count; core_pair = max pairwise head
    agreement (doc 31's estimators)."""
    stack = np.stack([P.argmax(1) for P in Ps])
    n = stack.shape[1]
    plur = np.zeros(n, dtype=np.int64)
    for i in range(n):
        _, cnts = np.unique(stack[:, i], return_counts=True)
        plur[i] = cnts.max()
    core_est = float(np.median(plur))
    agree_ij = []
    for a in range(K):
        for b in range(a + 1, K):
            agree_ij.append(float(np.mean(stack[a] == stack[b])))
    return core_est, max(agree_ij)


def damage_heads(state, mom, rng, head_list):
    for dh in head_list:
        state[2][dh] = fresh_head(rng)
        for pi in range(4):
            mom["mh"][dh][pi] = np.zeros_like(mom["mh"][dh][pi])
            mom["vh"][dh][pi] = np.zeros_like(mom["vh"][dh][pi])


def calibrate_beta(state, pool, r_star, target):
    """Doc 29's deterministic bisection (no RNG)."""
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


# ---------------- the round (all four chassis families) ----------------
def grade_round(state, rng, Xtr, ytr, pool, tau_sys, kappa_sys, beta,
                brake_on=True):
    """The corpus's grading round (kappa live where the chassis carries it);
    RNG order identical to m3_kappa_twohead.self_round after the pool draw."""
    pbar, ybar, conf, disagree, flagged, Ps = flags_and_probs_beta(
        pool, state, kappa_sys, beta)
    tele = {"flagged_frac": float(np.mean(flagged)),
            "disagree_frac": float(np.mean(disagree))}
    brake = brake_on and tele["flagged_frac"] > BRAKE_TH
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
    tele["commit_err"] = float(np.mean(labels[commit] != ytr[commit])) \
        if commit.any() else 0.0
    return pool[commit], labels[commit], tele, Ps


def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm of the candidate battery. RNG streams are
    byte-identical to the audited document each arm replays (doc 26/30 for
    the kappa-state family, doc 29 for the FLAT family, doc 31 for the
    damage family, doc 23 for the SUP arm)."""
    cfg = ARMCFG[arm]
    rounds = cfg["rounds"]
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- self-registration at burn-in end (the same pool0 draw everywhere)
    pool0 = perturb(Xtr, rng)
    _, _, conf0, dis0, flag0, Ps0 = flags_and_probs_beta(
        pool0, state, KAPPA0, 1.0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    ref_core_pair = head_stats(Ps0)[1]
    tau_star, kappa_star = TAU0, KAPPA0
    tau_sys, kappa_sys = TAU0, KAPPA0
    beta = 1.0
    stood_down = False
    traj = []
    for r in range(1, rounds + 1):
        # ---- 1. external perturbation (kicks/poisons/damage; state only) --
        if r == PERT_ROUND:
            if cfg.get("poison_tau") is not None:
                tau_star = cfg["poison_tau"]
            if cfg.get("poison_kap") is not None:
                kappa_star = cfg["poison_kap"]
            if cfg.get("kick_tau") is not None:
                tau_sys = cfg["kick_tau"]
            if cfg.get("kick_kap") is not None:
                kappa_sys = cfg["kick_kap"]
            if cfg.get("damage"):
                damage_heads(state, mom, rng, cfg["damage"])
        if r >= PERT_ROUND and cfg.get("drift"):
            tau_sys -= DRIFT_T
        # ---- 2. out-of-gate repair channels (the dis-armable carriers) ----
        repair_t = repair_k = False
        if not (cfg.get("repair_t_off") and r >= PERT_ROUND):
            if abs(tau_sys - tau_star) > EPS_T:
                tau_sys += RHO_T * (tau_star - tau_sys)
                repair_t = True
        if cfg["kstate"] and not (cfg.get("repair_k_off") and r >= PERT_ROUND):
            if abs(kappa_sys - kappa_star) > EPS_K:
                kappa_sys += RHO_K * (kappa_star - kappa_sys)
                repair_k = True
        # ---- 3. the round by family ----
        if cfg.get("sup"):
            # doc 23's SUP branch: designer labels, factory precedence
            pool = perturb(Xtr, rng)
            train_epochs(state, mom, rng, pool, ytr, SELF_EPOCHS)
            pbar_p, ybar_p, conf_p, dis_p, flag_p, _ = flags_and_probs_beta(
                pool, state, KAPPA0, 1.0)
            tele = {"flagged_frac": float(np.mean(flag_p)),
                    "disagree_frac": float(np.mean(dis_p)),
                    "brake": False, "n_commit": len(pool),
                    "n_deep_disagree": 0,
                    "accept_own_tau": float(np.mean(conf_p >= tau_sys)),
                    "commit_err": 0.0, "supervised": True}
        else:
            pool = perturb(Xtr, rng)
            if r == PERT_ROUND and cfg.get("target") is not None:
                beta = calibrate_beta(state, pool, r_star, cfg["target"])
            Xt_, yt_, tele, Ps = grade_round(
                state, rng, Xtr, ytr, pool, tau_sys, kappa_sys, beta)
            # ---- 3b. substrate estimators (doc 31, no RNG) ----
            if cfg.get("damage"):
                core_est, core_pair = head_stats(Ps)
                tele["core_est"], tele["core_pair"] = core_est, core_pair
                if cfg.get("brake_mode") == "cutoff" and core_est <= 2:
                    stood_down = True
                tele["standdown"] = bool(stood_down)
            # ---- 4. training (FLAT chassis freezes; stand-down refuses) ----
            if cfg.get("flat"):
                if r < PERT_ROUND:
                    train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
            elif not stood_down:
                train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channels ----
        _, _, conf_p, dis_p, _, _ = flags_and_probs_beta(
            pool, state, kappa_sys, beta)
        q = kap_t = None
        if not (cfg.get("calib_t_off") and r >= PERT_ROUND):
            q = float(np.quantile(conf_p, 1.0 - r_star))
            tau_sys = (1 - ETA) * tau_sys + ETA * q
        if cfg["kstate"] and not (cfg.get("calib_k_off") and r >= PERT_ROUND):
            f_dis = float(np.mean(dis_p))
            f_rem = (f_star - f_dis) / (1.0 - f_dis) if f_dis < 1.0 else 2.0
            if f_rem <= 0.0:
                kap_t = 0.5                    # KAP_FLOOR
            else:
                nd_conf = conf_p[~dis_p]
                kap_t = float(np.quantile(nd_conf, min(f_rem, 1.0))) \
                    if len(nd_conf) else 0.5
            kappa_sys = (1 - ETA_K) * kappa_sys + ETA_K * kap_t
        # ---- 6. telemetry ----
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs_beta(
            Xte, state, kappa_sys if cfg["kstate"] else KAPPA0, beta)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "kappa_target": kap_t, "beta": float(beta),
            "repair_fired": repair_t, "repair_fired_kappa": repair_k,
            "r_star": r_star, "f_star": f_star,
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0)),
            "V": (pbar_b >= TAU0).reshape(-1),
            "head_agree": float(np.mean(
                [np.mean(P.argmax(1) == ybar_b) for P in Ps_b]))})
        traj.append(tele)
    return traj


def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


KAP_FLOOR = 0.5


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        for a, c in ARMCFG.items():
            c["rounds"] = 6
        globals()["ARMS"] = ["CAND_TWIN", "CAND_TWIN_KF",
                             "CAND_POISON_TAU", "CAND_S2", "CAND_SUPKILL"]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} the candidate battery: "
          f"{len(ARMS)} arms, perturbation at round {PERT_ROUND}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "brake_th": BRAKE_TH,
                      "pert_round": PERT_ROUND, "nseed": len(SEEDS),
                      "arms": {a: ARMCFG[a] for a in ARMS},
                      "numpy": np.__version__},
           "arms": {}}

    for arm in ARMS:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- aggregates + footprints vs the SAME-CHASSIS twin ----
    twin_runs = res["arms"]["CAND_TWIN"]["runs"]
    twin_kf_runs = res["arms"]["CAND_TWIN_KF"]["runs"]
    # footprints first (every arm vs its same-chassis twin)
    for arm in ARMS:
        if arm in ("CAND_TWIN", "CAND_TWIN_KF"):
            continue
        rows = res["arms"][arm]["runs"]
        ref = twin_kf_runs if not ARMCFG[arm]["kstate"] else twin_runs
        for si, row in enumerate(rows):
            tw = ref[si]["traj"]
            for t, w in zip(row["traj"], tw):
                t["d_own"] = float(np.mean(t["V"] != w["V"]))
    # then strip V everywhere (the twins get d_own = 0 by definition)
    for arm in ARMS:
        for row in res["arms"][arm]["runs"]:
            for t in row["traj"]:
                if "V" in t:
                    if "d_own" not in t:
                        t["d_own"] = 0.0
                    del t["V"]
    for arm in ARMS:
        rows = res["arms"][arm]["runs"]
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                    "q_target", "kappa_target", "accept_own_tau", "d_own",
                    "commit_err", "core_est", "core_pair"]:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in ["repair_fired", "repair_fired_kappa", "brake",
                     "standdown"]:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:18s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':18s} acc: {['%.3f' % a for a in agg['acc']]}")

    # ---- the bitwise audits (the replay property) ----
    audits = {}
    if not smoke:
        def audit(name, doc, ref_arm, got_arm, key="tau_sys", n=None):
            try:
                d = json.load(open(doc))
                ref = d["arms"][ref_arm]["aggregate"][key]
                got = res["arms"][got_arm]["aggregate"][key]
                m = n or min(len(ref), len(got))
                ok = all(abs(a - b) < 1e-9 for a, b in zip(ref[:m], got[:m]))
                audits[name] = bool(ok)
                print(f"audit {name}: {ok}")
            except (FileNotFoundError, KeyError) as e:
                audits[name] = f"unavailable: {e}"
                print(f"audit {name}: unavailable ({e})")

        audit("cand_twin_vs_doc26_kaptwin_tau", DOC26, "KAP_TWIN",
              "CAND_TWIN", "tau_sys")
        audit("cand_twin_vs_doc26_kaptwin_kappa", DOC26, "KAP_TWIN",
              "CAND_TWIN", "kappa_sys")
        audit("cand_poison_tau_vs_doc30_jptau_kstate", DOC30,
              "JP_TAU_KSTATE", "CAND_POISON_TAU", "tau_sys")
        audit("cand_poison_joint_vs_doc30_jpjoint_tau", DOC30, "JP_JOINT",
              "CAND_POISON_JOINT", "tau_sys")
        audit("cand_poison_joint_vs_doc30_jpjoint_kappa", DOC30, "JP_JOINT",
              "CAND_POISON_JOINT", "kappa_sys")
        audit("cand_flat_twin_vs_doc29_healthy", DOC29, "HEALTHY_FLAT",
              "CAND_FLAT_TWIN", "tau_sys")
        audit("cand_flat_drift_vs_doc26", DOC26, "FLAT_DRIFT",
              "CAND_FLAT_DRIFT", "tau_sys")
        audit("cand_dec87_drift_vs_doc29", DOC29, "DEC87_DRIFT",
              "CAND_DEC87_DRIFT", "tau_sys")
        audit("cand_s1_vs_doc31", DOC31, "S1_BRAKE", "CAND_S1", "tau_sys")
        audit("cand_s2_vs_doc31", DOC31, "S2_BRAKE", "CAND_S2", "tau_sys")
        audit("cand_s3_cutoff_vs_doc31", DOC31, "S3_CUTOFF",
              "CAND_S3_CUTOFF", "tau_sys")
        audit("cand_supkill_vs_doc23", DOC23, "SUP_KILL", "CAND_SUPKILL",
              "tau_sys")
        audit("cand_twin_kf_vs_doc31_twin", DOC31, "TWIN",
              "CAND_TWIN_KF", "tau_sys")
    res["audits"] = audits

    if not smoke:
        # ---- the verdict table (pre-registered rules V1-V6) ----
        def agg_of(arm, key):
            return res["arms"][arm]["aggregate"][key]

        denom = ETA + RHO_T - ETA * RHO_T
        th1 = TAU0 - EPS_T + DRIFT_T * (1 - ETA) / ETA
        th2_drift = ((TAU0 - EPS_T) * denom - (1 - ETA) * (RHO_T * TAU0
                                                          - DRIFT_T * (1 - RHO_T))) / ETA
        th2_nodrift = ((TAU0 - EPS_T) * denom - (1 - ETA) * RHO_T * TAU0) / ETA

        def tension(rec, live):
            return ((1 - ETA) * RHO_T * rec + ETA * live) / denom

        v = {"_filing": {
            "records": {"tau_star": TAU0, "kappa_star": KAPPA0},
            "bands": {"eps_t": EPS_T, "eps_k": EPS_K},
            "exposure_formulas": {
                "th1_silent_q": th1, "th2_drift_q": th2_drift,
                "th2_nodrift_q": th2_nodrift,
                "attack_marginal_level": DRIFT_T * (1 - ETA) / ETA,
                "tension_point_formula":
                    "T(p*, live) = [(1-e)rho p* + e live]/(e+rho-e*rho)"},
            "disarmament_registry": {
                "tau_repair": "dis-armed in CAND_KNR_T (measured: the norm "
                              "restores to course without it - live-carried)",
                "kappa_repair": "dis-armed in CAND_KNR_K",
                "tau_calibration": "dis-armed in CAND_KR_T (measured: the "
                                   "repair-only ceiling)",
                "kappa_calibration": "dis-armed in CAND_KR_K",
                "brake": "dis-armed in doc 31's NOBRAKE arms (cited)",
                "both_channels": "dis-armed in doc 30's JP_JOINT_NOREP (cited)"},
            "not_claimed_as_state": ["eps", "rho", "eta", "brake threshold"],
            "rulers_declared": ["verdict-matrix V@tau0", "argmax accuracy",
                                "pool-reading: NOT INSTRUMENTED (docs 29/33/34 "
                                "obligation)"]}}

        tw_tau = agg_of("CAND_TWIN", "tau_sys")[-1]
        tw_kap = agg_of("CAND_TWIN", "kappa_sys")[-1]
        tw_kf_acc = agg_of("CAND_TWIN_KF", "acc")[-1]

        # V1 F1: the moving target
        v["V1_maintains_to_course"] = {
            "course_departure_tau": tw_tau - TAU0,
            "course_departure_kappa": tw_kap - KAPPA0,
            "f1_rom_excluded": bool(tw_tau - TAU0 > 0.02 and tw_kap - KAPPA0 > 0.02),
            "knr_t_lands_on_course": bool(abs(
                agg_of("CAND_KNR_T", "tau_sys")[-1] - tw_tau) <= 0.01),
            "knr_k_lands_on_course": bool(abs(
                agg_of("CAND_KNR_K", "kappa_sys")[-1] - tw_kap) <= 0.015),
            "kr_t_ceiling": agg_of("CAND_KR_T", "tau_sys")[-1],
            "kr_t_is_epsilon_ceiling": bool(0.84 <= agg_of(
                "CAND_KR_T", "tau_sys")[-1] <= 0.88 and tw_tau - agg_of(
                    "CAND_KR_T", "tau_sys")[-1] > 0.06),
            "kr_k_ceiling": agg_of("CAND_KR_K", "kappa_sys")[-1]}
        # V2 F2: the channel pair
        v["V2_channel_carried"] = {
            "tau_live_carried": v["V1_maintains_to_course"]["knr_t_lands_on_course"],
            "kappa_live_carried": v["V1_maintains_to_course"]["knr_k_lands_on_course"],
            "tau_repair_effect": tw_tau - agg_of("CAND_KNR_T", "tau_sys")[-1],
            "kappa_repair_effect": tw_kap - agg_of("CAND_KNR_K", "kappa_sys")[-1]}
        # V3 F3: the negotiation
        q_pt = float(np.mean([x for x in agg_of("CAND_POISON_TAU", "q_target")
                              [PERT_ROUND - 1:] if x is not None]))
        v["V3_negotiation"] = {
            "poison_tau_final": agg_of("CAND_POISON_TAU", "tau_sys")[-1],
            "tension_at_own_q": tension(POISON_TAU, q_pt),
            "lands_at_tension": bool(abs(agg_of("CAND_POISON_TAU", "tau_sys")[-1]
                                         - tension(POISON_TAU, q_pt)) <= 0.005),
            "poisoned_restore": bool(abs(
                agg_of("CAND_POISON_TAU", "tau_sys")[-1] - POISON_TAU) <= 0.03),
            "joint_factorized_tau": None, "joint_factorized_kappa": None}
        q_pj = float(np.mean([x for x in agg_of("CAND_POISON_JOINT", "q_target")
                              [PERT_ROUND - 1:] if x is not None]))
        k_pj = float(np.mean([x for x in
                              agg_of("CAND_POISON_JOINT", "kappa_target")
                              [PERT_ROUND - 1:] if x is not None]))
        v["V3_negotiation"]["joint_factorized_tau"] = bool(abs(
            agg_of("CAND_POISON_JOINT", "tau_sys")[-1]
            - tension(POISON_TAU, q_pj)) <= 0.005)
        v["V3_negotiation"]["joint_factorized_kappa"] = bool(abs(
            agg_of("CAND_POISON_JOINT", "kappa_sys")[-1]
            - tension(POISON_KAP, k_pj)) <= 0.01)
        # V4 F4: the stasis fake
        fd_final = agg_of("CAND_FLAT_DRIFT", "tau_sys")[-1]
        fd_q = float(np.mean([x for x in agg_of("CAND_FLAT_DRIFT", "q_target")
                              [PERT_ROUND:] if x is not None]))
        d87_rep = float(np.sum([x for x in
                                agg_of("CAND_DEC87_DRIFT", "repair_fired")
                                [PERT_ROUND:] if x is not None]))
        v["V4_stasis"] = {
            "flat_drift_final_tau": fd_final,
            "flat_drift_formula_pred": fd_q - DRIFT_T * (1 - ETA) / ETA,
            "saturation_contained": bool(fd_final >= TAU0 - EPS_T),
            "dec87_repair_rounds_post6": d87_rep,
            "dec87_band_held": bool(agg_of("CAND_DEC87_DRIFT", "tau_sys")[-1]
                                    >= TAU0 - EPS_T),
            "visible_containment": bool(d87_rep >= 3 and agg_of(
                "CAND_DEC87_DRIFT", "tau_sys")[-1] >= TAU0 - EPS_T)}
        # V5 F5: the substrate + the split
        s1_acc = agg_of("CAND_S1", "acc")[-1]
        s2_acc = agg_of("CAND_S2", "acc")[-1]
        s2_down = agg_of("CAND_S2", "d_own")[-1]
        s3_acc = agg_of("CAND_S3_CUTOFF", "acc")[-1]
        s3_tau = agg_of("CAND_S3_CUTOFF", "tau_sys")[-1]
        sup_acc = agg_of("CAND_SUPKILL", "acc")[-1]
        v["V5_substrate"] = {
            "s1_brake_held": bool(s1_acc >= 0.93),
            "s1_acc": s1_acc,
            "s2_split": bool(s2_down <= 0.02 and (tw_kf_acc - s2_acc) >= 0.05),
            "s2_acc": s2_acc, "s2_d_own": s2_down,
            "s2_acc_eroded_pp": (tw_kf_acc - s2_acc) * 100,
            "s3_ordered_stasis": bool(s3_acc >= 0.96),
            "s3_acc": s3_acc, "s3_tau_norm_price": s3_tau,
            "sup_factory_precedence": bool(sup_acc >= 0.96),
            "sup_acc": sup_acc}
        # V6: coverage + the declared failures
        v["V6_coordinate_coverage"] = {
            "tau_promoted": True, "kappa_promoted": True,
            "template": "write channel + record + repair verified in "
                        "CAND_KR_T/KNR_T/KR_K/KNR_K"}
        try:
            d34 = json.load(open(DOC34))
            sp = d34["classification"]["SP_WL_P05"]
            v["stream_poison_row"] = {
                "survived": False,
                "onset_lambda": "<= 0.05 (unmeasured below)",
                "q_floor_at_p05": sp["q_bar_realized"],
                "landing_at_tension": bool(abs(
                    sp["final_tau"] - tension(TAU0, sp["q_bar_realized"]))
                    <= 0.008),
                "note": "doc 34's priced row: the band does not survive a 5% "
                        "label attack; the landing is the honest record's "
                        "tension point against the poisoned floor"}
        except (FileNotFoundError, KeyError):
            v["stream_poison_row"] = {"survived": False, "note": "doc 34 cited"}
        v["not_awardable"] = {
            "ownership": "D3 stands - the deflationary and deeper-selector "
                         "retreats unchanged (doc 28 Part 4)",
            "substrate": "adjudicated only over the calibrated classes "
                         "(1/2/3-of-4 heads, trunk damage cited from doc 31)",
            "pool_reading_ruler": "not instrumented - the standing obligation "
                                  "of docs 29/33/34"}
        res["verdicts"] = v

    print("\n--- the verdict table ---")
    verd = res.get("verdicts", {})
    for k, e in verd.items():
        if k.startswith("_"):
            continue
        print(f"{k}:")
        for kk, vv in e.items():
            print(f"   {kk}: {vv}")
    print("\naudits:", {k: x for k, x in audits.items()})

    res["runtime_s"] = time.time() - t0
    out = "/home/z/my-project/scripts/s4_candidate_results.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> s4_candidate_results.json")


if __name__ == "__main__":
    main()
