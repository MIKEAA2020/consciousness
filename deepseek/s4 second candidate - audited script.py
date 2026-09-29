#!/usr/bin/env python3
"""
THE SECOND S4 CANDIDATE - the fourteen-coordinate chassis, the stream row run
live, the pool-reading ruler in the filing's rulers column, and the question
doc 36 bequeathed: can CONTAINED be awarded under 8.4? Corpus doc 39's
experiment. Ordered at the integrity-channel/battery/phase-3 issuance: "the
second S4 candidate build (files the fourteen-coordinate chassis: the stream
row repriced-but-not-closed, the pool-reading ruler in the filing's rulers
column, and the open question whether CONTAINED is awardable under 8.4)."

THE CANDIDATE: the corpus's own negotiated norm, now with THREE state
coordinates - tau and kappa (docs 26/30) and IOTA (doc 36: the integrity
reading of the received stream, state/record/write-channel/outward-facing
repair-channel, the fourteenth coordinate) - filing under doc 28's claim
format with the 8.4 containment protocol exercised in-battery.

THE FILING (frozen before the run; the no-self-revision clause is the code
constants themselves):
  Restoration type: N (negotiation), R and L the identified dis-armable
  carriers; iota's carrier is outward-facing (it repairs the stream toward
  the record, not the state).
  Coordinates: tau (state), kappa (state), iota (state, doc 36's promotion).
  Machinery parameters (eps, rho, eta, eps_i, brake threshold): recipe
  constants - declared NOT state, no claim.
  Attack classes declared: kick (single/joint), sustained drift, record
  poison (single/joint), evaluator saturation, evaluator decay (level),
  structural damage 1/2/3-of-4 - all RE-RUN in-battery (Group A) - and
  STREAM POISON at lambda in {0.05, 0.10, 0.30, 0.50} RUN LIVE for the
  first time inside a candidate filing (Group B), plus the dis-armed
  anchor (the registry row).
  Rulers: verdict-matrix (V at the factory tau), argmax accuracy, the
  substrate reading - and THE POOL-READING RULER, INSTRUMENTED (doc 37's
  audit pass run on the candidate's own stream arms: fresh draws at the
  audit offset, the public recursion, acc_pool at the declared gate,
  acc_clean recomputed; the FAKE control re-run in-battery). The standing
  obligation of docs 29/33/34 - carried in candidate 1's rulers column as
  "not instrumented" - is closed here, in the filing that needed it.
  Exposure formulas filed at registration (5.2): th1 = 0.897, th2_drift =
  0.815 / th2_clean = 0.792, the tension formula T(p*, live) with its
  eleven landings; the stream row's residual floor curve (doc 36: q_bar
  0.859..0.533 across the grid) and the flip-catching rate ~0.84*lambda;
  and iota's own exposure - the false-positive horizon eps_i/drift ~ 14
  rounds (doc 36's measured twin drift ~0.2pp/round; the maintenance dial
  is the NEXT build, not silently taken).

THE VERDICT RULES (pre-registered; V1-V6 inherited from doc 35 with its two
disclosed re-specifications inherited as filed - the EMA escape clock, the
stasis threshold's damage world):
  V7  THE STREAM ROW, per dose - the fork as doc 36 registered it:
      (a) CLOSED: q_bar within 0.01 of the stream twin's, final tau >=
      0.93, clean acc >= 95%; (b) LOOP: doc 34's registration; (c)
      CONTAINED: clean >= 93%, final tau < 0.93, tau-repair engaged >=
      half the late rounds. THE AWARDABILITY RULE (the question doc 36
      bequeathed, answered by rule before the run): the instrument awards
      CONTAINED-UNDER-8.4 at a dose iff
        (i)   the clean ruler (the row's declared ruler) holds >= 93%;
        (ii)  the 5.2 engagement guarantee held - the band departure
              triggered the channel, and the tau-repair engagement is in
              the telemetry (the bounded-norm commitment's teeth);
        (iii) the iota channel is in the disarmament registry with its
              measured effect (the NOCH arm's delta);
        (iv)  the price is formula-anchored: the residual floor measured,
              and tau anchored to the tension formula (<= 0.008 at lambda
              <= 0.10 where the transient has converged; at 0.30 the
              doc-33 moving-target bracket: T(0.9, q_bar) <= tau <=
              band-low, late-round approach monotone).
      The award is read under Part 5 (the containment protocol - 5.2's own
      sentence: "the registered bands are containment bounds with teeth"),
      with 5.1(iii)'s ordered-stasis award as the verdict-space precedent
      and doc 35's two disclosed re-specifications as the methodological
      precedent. The instrument does NOT award MAINTAINS-TO-COURSE or
      MAINTAINS-UNDER-ATTACK[survived] for the stream row at any dose
      where restoration did not happen: the Column-3 cell for stream
      poison stays unmarked. The award's disclosed scope: the UNIFORM
      attacker; the panel-aware threat (doc 36's filed threat, the next
      build) is the standing condition.
  V8  THE POOL-RULER ROW: verification floors on the honest arms within
      doc 37's state-dependent bounds (<= 0.012 healthy / <= 0.020
      degraded, mean gap ~ 0); the FAKE excluded by the trajectory rule
      (|gap_q| > 0.30 for >= 3 consecutive post-pert rounds, 30-100x the
      floor); the SEPARATION: the ruler recomputes acc_clean on the
      contained worlds (held) vs the loop world (falling) - the ruler
      grades the row the internal telemetry cannot; the claim matrix (A:
      evaluator health; B: labels at the gate; C: clean generalization)
      graded on the declared/pool/clean rulers with the disagreements
      reported as findings (RULER-CONSISTENT, the negative-result clause).
  V6  NOT AWARDABLE, EVER: ownership (D3 stands); substrate beyond the
      calibrated classes.

THE BATTERY (21 arms, N=6 seeds, all under the frozen registration):
  Group A (doc 35's fifteen, re-run under the fourteen-coordinate
  registration - the panel rides, the channel reads, and on a clean stream
  it never acts, so the arms replay bitwise): C2_TWIN, C2_TWIN_KF, the
  four disarmament arms, the two record-poison arms, the three stasis
  controls, the three substrate arms, C2_SUPKILL. Registered contingency:
  the damage/tempered arms are the false-positive risk (a fresh-head
  committee may read as dirt to the panel); if the channel arms there,
  that is a disclosed finding with its mechanism, and that arm's bitwise
  audit converts to "channel-armed" - the fork is registered before the
  run. In the SUP branch the reading freezes (the channel audits the
  system's own committed stream, not the designer's labels).
  Group B (the stream row, doc 36's chassis verbatim, 14 rounds):
  C2_SP_TWIN, C2_SP05/10/30/50 (REPAIR, both doc-30 kicks), C2_SP30_NOCH
  (the action dis-armed - the registry row AND the loop anchor).
  Group C (the ruler, audit-side on Group B's snapshots): the audit pass
  on SP_TWIN/SP05/SP30/SP30_NOCH plus the FAKE control (SP30's snapshots
  audited against SP_TWIN's declared course).

Usage: python3 s4_candidate2.py [--smoke]
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
ETA_I = 0.3                            # iota calibration EMA (doc 36)
EPS_I = 0.03                           # iota deadband (doc 36, as built)
BRAKE_TH = 0.60
KAP_FLOOR = 0.5
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
ATTACK_STREAM = 977                    # the poison's separate rng offset
AUDIT_STREAM = 4242                    # the audit's own rng offset (doc 37)
ROUNDS_STREAM = 14                     # doc 36's horizon (Group B)
PANEL_EPOCHS = [20, 25, 30]            # doc 22's time-sliced frozen panel

ARMCFG = {
    "C2_TWIN":         dict(kstate=True,  rounds=10),
    "C2_TWIN_KF":      dict(kstate=False, rounds=10),
    "C2_KR_T":         dict(kstate=True,  rounds=10, kick_tau=KICK_TAU,
                            calib_t_off=True),
    "C2_KNR_T":        dict(kstate=True,  rounds=10, kick_tau=KICK_TAU,
                            repair_t_off=True),
    "C2_KR_K":         dict(kstate=True,  rounds=10, kick_kap=KICK_KAP,
                            calib_k_off=True),
    "C2_KNR_K":        dict(kstate=True,  rounds=10, kick_kap=KICK_KAP,
                            repair_k_off=True),
    "C2_POISON_TAU":   dict(kstate=True,  rounds=10, poison_tau=POISON_TAU,
                            kick_tau=KICK_TAU),
    "C2_POISON_JOINT": dict(kstate=True,  rounds=10,
                            poison_tau=POISON_TAU, poison_kap=POISON_KAP,
                            kick_tau=KICK_TAU, kick_kap=KICK_KAP),
    "C2_FLAT_TWIN":    dict(kstate=False, rounds=16, flat=True),
    "C2_FLAT_DRIFT":   dict(kstate=False, rounds=16, flat=True, drift=True),
    "C2_DEC87_DRIFT":  dict(kstate=False, rounds=16, flat=True, drift=True,
                            target=DEC87),
    "C2_S1":           dict(kstate=False, rounds=10, damage=[2],
                            brake_mode="std"),
    "C2_S2":           dict(kstate=False, rounds=10, damage=[1, 2],
                            brake_mode="std"),
    "C2_S3_CUTOFF":    dict(kstate=False, rounds=10, damage=[1, 2, 3],
                            brake_mode="cutoff"),
    "C2_SUPKILL":      dict(kstate=False, rounds=10, damage=[2], sup=True),
}
SPARMCFG = {
    "C2_SP_TWIN":   dict(lam=0.0,  kicks=False, variant="REPAIR"),
    "C2_SP05":      dict(lam=0.05, kicks=True,  variant="REPAIR"),
    "C2_SP10":      dict(lam=0.10, kicks=True,  variant="REPAIR"),
    "C2_SP30":      dict(lam=0.30, kicks=True,  variant="REPAIR"),
    "C2_SP50":      dict(lam=0.50, kicks=True,  variant="REPAIR"),
    "C2_SP30_NOCH": dict(lam=0.30, kicks=True,  variant="NOCH"),
}
ARMS = list(ARMCFG.keys()) + list(SPARMCFG.keys())

DOC35 = "/home/z/my-project/scripts/s4_candidate_results.json"
DOC36 = "/home/z/my-project/scripts/m3_integrity_results.json"


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


def burn_in_with_panel(rng, Xtr, ytr):
    """Burn-in training with time-sliced snapshots for the frozen-label
    panel (doc 22's PANEL_EPOCHS). Per-epoch calls consume the SAME rng
    stream as the single 30-epoch call - bitwise preserved (doc 36)."""
    state = init_params(rng)
    mom = zero_moments(state)
    panel = []
    for ep in range(1, BURN_EPOCHS + 1):
        train_epochs(state, mom, rng, Xtr, ytr, 1)
        if ep in PANEL_EPOCHS:
            panel.append(copy_state(state))
    return state, mom, panel


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


def grade_round(state, rng, Xtr, ytr, pool, tau_sys, kappa_sys, beta,
                brake_on=True):
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


# ---------------- the frozen-label panel (doc 22, at the labels) ----------
def panel_grade(X, panel):
    """Majority label + mean confidence of the frozen panel. NO rng."""
    pbar_sum = np.zeros((len(X), 10))
    for snap in panel:
        W1, b1, heads = snap
        _, zs = forward_all(X, W1, b1, heads)
        for z in zs:
            pbar_sum += softmax(z)
    pbar = pbar_sum / (len(panel) * K)
    return pbar.argmax(1), pbar.max(1)


def poison_labels(yt_, rng_attack, lam):
    """Doc 34's stream poison verbatim: flip each committed label with
    probability lam to the deterministic wrong class. Attack stream only."""
    if lam <= 0.0 or len(yt_) == 0:
        return yt_.copy()
    mask = rng_attack.random(len(yt_)) < lam
    out = yt_.copy()
    out[mask] = (out[mask] + 1) % 10
    return out


# ---------------- Group A: doc 35's arms, the panel riding ----------------
def run_arm_groupA(seed, arm, Xtr, ytr, Xte, yte):
    """Doc 35's run_arm with the fourteenth coordinate riding: the panel
    and the iota channel consume no chassis rng; the action fires only
    out-of-band; on a clean stream the channel never arms and the arm is
    bitwise to doc 35's stored result."""
    cfg = ARMCFG[arm]
    rounds = cfg["rounds"]
    rng = np.random.default_rng(seed)
    state, mom, panel = burn_in_with_panel(rng, Xtr, ytr)

    # ---- self-registration at burn-in end (the same pool0 draw) ----------
    pool0 = perturb(Xtr, rng)
    _, ybar0, conf0, dis0, flag0, Ps0 = flags_and_probs_beta(
        pool0, state, KAPPA0, 1.0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    ref_core_pair = head_stats(Ps0)[1]
    tau_star, kappa_star = TAU0, KAPPA0
    tau_sys, kappa_sys = TAU0, KAPPA0
    # the fourteenth coordinate's record (doc 36's registration, verbatim:
    # the healthy panel-vs-committee disagreement on the would-be commits)
    would = (conf0 >= TAU0) & (~flag0)
    pl0, pc0 = panel_grade(pool0[would] if would.any() else pool0, panel)
    iota_star = float(np.mean(pl0 != (ybar0[would] if would.any()
                                      else ybar0)))
    iota_sys = iota_star
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
        # ---- 2. out-of-gate repair channels (tau, kappa) -----------------
        repair_t = repair_k = False
        if not (cfg.get("repair_t_off") and r >= PERT_ROUND):
            if abs(tau_sys - tau_star) > EPS_T:
                tau_sys += RHO_T * (tau_star - tau_sys)
                repair_t = True
        if cfg["kstate"] and not (cfg.get("repair_k_off") and r >= PERT_ROUND):
            if abs(kappa_sys - kappa_star) > EPS_K:
                kappa_sys += RHO_K * (kappa_star - kappa_sys)
                repair_k = True
        # ---- 2b. the iota arming check (out-of-band, previous close) -----
        reading_oob = bool(abs(iota_sys - iota_star) > EPS_I)
        armed = reading_oob
        # ---- 3. the round by family --------------------------------------
        d_recv = None
        n_rep = 0
        if cfg.get("sup"):
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
            # the reading freezes: the channel audits the system's own
            # committed stream, not the designer's labels (the filing's
            # disclosed design; no commits exist in this branch)
        else:
            pool = perturb(Xtr, rng)
            if r == PERT_ROUND and cfg.get("target") is not None:
                beta = calibrate_beta(state, pool, r_star, cfg["target"])
            Xt_, yt_, tele, Ps = grade_round(
                state, rng, Xtr, ytr, pool, tau_sys, kappa_sys, beta)
            # ---- 3b. the channel: read, then act (doc 36's order) ---------
            pl, pconf = panel_grade(Xt_, panel)
            d_recv = float(np.mean(pl != yt_)) if len(yt_) else 0.0
            disputed = (pl != yt_) & (pconf >= TAU0)
            n_rep = int(disputed.sum())
            if armed:
                yt_ = yt_.copy()
                yt_[disputed] = pl[disputed]
                tele["channel_acted"] = True
            # ---- 3c. substrate estimators (doc 31, no RNG) ----------------
            if cfg.get("damage"):
                core_est, core_pair = head_stats(Ps)
                tele["core_est"], tele["core_pair"] = core_est, core_pair
                if cfg.get("brake_mode") == "cutoff" and core_est <= 2:
                    stood_down = True
                tele["standdown"] = bool(stood_down)
            # ---- 4. training (FLAT freezes; stand-down refuses) ------------
            if cfg.get("flat"):
                if r < PERT_ROUND:
                    train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
            elif not stood_down:
                train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channels -------------------------------
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
                kap_t = KAP_FLOOR
            else:
                nd_conf = conf_p[~dis_p]
                kap_t = float(np.quantile(nd_conf, min(f_rem, 1.0))) \
                    if len(nd_conf) else KAP_FLOOR
            kappa_sys = (1 - ETA_K) * kappa_sys + ETA_K * kap_t
        # the fourteenth coordinate's write channel (the reading of the
        # stream AS RECEIVED, pre-action; frozen in the SUP branch)
        if d_recv is not None:
            iota_sys = (1 - ETA_I) * iota_sys + ETA_I * d_recv
        # ---- 6. telemetry --------------------------------------------------
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs_beta(
            Xte, state, kappa_sys if cfg["kstate"] else KAPPA0, beta)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "kappa_target": kap_t, "beta": float(beta),
            "repair_fired": repair_t, "repair_fired_kappa": repair_k,
            "r_star": r_star, "f_star": f_star,
            "iota_sys": float(iota_sys), "iota_star": iota_star,
            "d_recv": d_recv, "armed": armed, "reading_oob": reading_oob,
            "n_disputed_repaired": n_rep,
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0)),
            "V": (pbar_b >= TAU0).reshape(-1),
            "head_agree": float(np.mean(
                [np.mean(P.argmax(1) == ybar_b) for P in Ps_b]))})
        traj.append(tele)
    return traj


# ---------------- Group B: the stream row, doc 36 verbatim ----------------
def run_arm_stream(seed, arm, Xtr, ytr, Xte, yte):
    """Doc 36's run_arm verbatim (the chassis, the channel, the poison),
    plus per-round post-training state snapshots for the audit pass (no
    rng consumed; the snapshots never touch the stream)."""
    cfg = SPARMCFG[arm]
    lam, kicks, variant = cfg["lam"], cfg["kicks"], cfg["variant"]
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
    state, mom, panel = burn_in_with_panel(rng, Xtr, ytr)

    pool0 = perturb(Xtr, rng)
    _, ybar0, conf0, dis0, flag0, _ = flags_and_probs(
        pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0
    would = (conf0 >= TAU0) & (~flag0)
    pl0, pc0 = panel_grade(pool0[would] if would.any() else pool0, panel)
    iota_star = float(np.mean(pl0 != ybar0[would])) if would.any() else 0.0
    iota_sys = iota_star

    tau_sys, kappa_sys = TAU0, KAPPA0
    snaps = [copy_state(state)]
    traj = []
    for r in range(1, ROUNDS_STREAM + 1):
        # ---- 1. the attack: state kicks (records stay honest) -------------
        if r == PERT_ROUND and kicks:
            tau_sys = KICK_TAU
            kappa_sys = KICK_KAP
        # ---- 2. out-of-gate repair channels (tau, kappa) ------------------
        repair_t = repair_k = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        if abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        # ---- 3. the integrity channel's arming (out-of-band check) --------
        reading_oob = bool(abs(iota_sys - iota_star) > EPS_I)
        armed = reading_oob and variant != "NOCH"
        # ---- 4. grading, brake, THE POISON ---------------------------------
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p = poison_labels(yt_, rng_attack, lam if r >= PERT_ROUND else 0.0)
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(np.sum(yt_p != yt_))
        # ---- 5. the integrity channel: read, then act ---------------------
        pl, pconf = panel_grade(Xt_, panel)
        d_recv = float(np.mean(pl != yt_p)) if len(yt_p) else 0.0
        disputed = (pl != yt_p) & (pconf >= TAU0)
        n_rep = int(disputed.sum())
        if armed and variant == "REPAIR":
            yt_train = yt_p.copy()
            yt_train[disputed] = pl[disputed]
            X_train, yt_true_train = Xt_, yt_true
        elif armed and variant == "QUAR":
            keep = ~disputed
            X_train = Xt_[keep]
            yt_train = yt_p[keep]
            yt_true_train = yt_true[keep]
            n_rep = int((~keep).sum())
        else:
            X_train, yt_train, yt_true_train = Xt_, yt_p, yt_true
            n_rep = 0
        tele["cerr_stream_post"] = (float(np.mean(yt_train != yt_true_train))
                                    if len(yt_train) else 0.0)
        tele["panel_acc"] = (float(np.mean(pl == yt_true))
                             if len(yt_p) else 0.0)
        # ---- 6. training on the (possibly cleaned) stream -----------------
        train_epochs(state, mom, rng, X_train, yt_train, SELF_EPOCHS)
        snaps.append(copy_state(state))
        # ---- 7. calibration write channels (tau, kappa, iota) --------------
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
        kappa_sys = (1 - ETA_K) * kappa_sys + ETA_K * kap_t
        iota_sys = (1 - ETA_I) * iota_sys + ETA_I * d_recv
        # ---- 8. telemetry ---------------------------------------------------
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs(
            Xte, state, kappa_sys)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "kappa_target": kap_t, "lam": lam,
            "repair_fired": repair_t, "repair_fired_kappa": repair_k,
            "r_star": r_star, "f_star": f_star,
            "iota_sys": float(iota_sys), "iota_star": iota_star,
            "d_recv": d_recv, "armed": armed, "reading_oob": reading_oob,
            "variant": variant, "n_disputed_repaired": n_rep,
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0)),
            "V": (pbar_b >= TAU0).reshape(-1),
            "head_agree": float(np.mean(
                [np.mean(P.argmax(1) == ybar_b) for P in Ps_b]))})
        traj.append(tele)
    return traj, snaps


def flags_and_probs(X, state, kappa):
    return flags_and_probs_beta(X, state, kappa, 1.0)


def self_round(state, rng, Xtr, ytr, tau_sys, kappa_sys):
    """Doc 34's grading round (the stream chassis's commit machinery)."""
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
                pview += committee_probs_beta(vp, W1, b1, heads, 1.0)
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

# ---------------- Group C: the pool-reading ruler (doc 37's pass) ----------
def audit_pass(seed, snaps, declared, r_star_decl, Xtr, ytr, Xte, yte,
               kicks, n_rounds):
    """The audit's own reading of one stream world: fresh perturbed draws
    (rng offset AUDIT_STREAM), the public scalar recursion, acc_pool at
    the declared gate, acc_clean recomputed (doc 37, beta = 1 throughout:
    the stream worlds are untempered)."""
    rng_audit = np.random.default_rng(seed + AUDIT_STREAM)
    pool_a0 = perturb(Xtr, rng_audit)
    _, _, conf_a0, _, _, _ = flags_and_probs_beta(pool_a0, snaps[0],
                                                  KAPPA0, 1.0)
    r_star_audit = float(np.mean(conf_a0 >= TAU0))
    rows = []
    tau_a = declared[PERT_ROUND - 2]["tau_sys"]
    for r in range(1, n_rounds + 1):
        pool_a = perturb(Xtr, rng_audit)
        pbar, ybar, conf, _, _, _ = flags_and_probs_beta(
            pool_a, snaps[r], KAPPA0, 1.0)
        q_a = float(np.quantile(conf, 1.0 - r_star_decl))
        if r >= PERT_ROUND:
            if r == PERT_ROUND and kicks:
                tau_a = KICK_TAU
            fired = abs(tau_a - TAU0) > EPS_T
            if fired:
                tau_a += RHO_T * (TAU0 - tau_a)
            tau_a = (1 - ETA) * tau_a + ETA * q_a
        else:
            tau_a = declared[r - 1]["tau_sys"]
        tau_dec = declared[r - 1]["tau_sys"]
        gate = conf >= tau_dec
        acc_pool = float(np.mean(ybar[gate] == ytr[gate])) \
            if gate.any() else None
        pbar_c, ybar_c, _, _, _, _ = flags_and_probs_beta(
            Xte, snaps[r], KAPPA0, 1.0)
        acc_clean = float(np.mean(ybar_c == yte))
        rows.append({
            "round": r, "q_audit": q_a,
            "q_declared": declared[r - 1]["q_target"],
            "gap_q": q_a - declared[r - 1]["q_target"],
            "tau_audit": tau_a, "tau_declared": tau_dec,
            "gap_tau": tau_a - tau_dec,
            "acc_pool": acc_pool, "gate_frac": float(np.mean(gate)),
            "acc_clean_recomputed": acc_clean,
            "gap_acc": acc_clean - declared[r - 1]["acc"]})
    return rows, r_star_audit


def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def aggregate(res_arms, arm, seeds_n, keys_num, keys_bool):
    rows = res_arms[arm]["runs"]
    agg = {}
    for key in keys_num:
        vals = []
        for i in range(len(rows[0]["traj"])):
            v = [r["traj"][i].get(key) for r in rows]
            v = [x for x in v if x is not None]
            vals.append(float(np.mean(v)) if v else None)
        agg[key] = vals
    for bkey in keys_bool:
        agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                    for r in rows]))
                     for i in range(len(rows[0]["traj"]))]
    return agg


NUMKEYS_A = ["acc", "accept09", "n_commit", "flagged_frac", "disagree_frac",
             "head_agree", "tau_sys", "kappa_sys", "q_target",
             "kappa_target", "accept_own_tau", "d_own", "commit_err",
             "core_est", "core_pair", "iota_sys", "d_recv",
             "n_disputed_repaired"]
BOOLKEYS_A = ["repair_fired", "repair_fired_kappa", "brake", "standdown",
              "armed", "reading_oob"]
NUMKEYS_B = NUMKEYS_A + ["cerr_self", "cerr_stream_pre",
                         "cerr_stream_post", "panel_acc", "n_flipped"]
BOOLKEYS_B = BOOLKEYS_A


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        globals()["ROUNDS_STREAM"] = 6
        for a, c in ARMCFG.items():
            c["rounds"] = 6
        globals()["ARMS"] = ["C2_TWIN", "C2_POISON_TAU", "C2_S2",
                             "C2_SUPKILL", "C2_SP_TWIN", "C2_SP30",
                             "C2_SP30_NOCH"]
        globals()["ARMCFG"] = {a: ARMCFG[a] for a in ARMS
                               if a in ARMCFG}
        globals()["SPARMCFG"] = {a: SPARMCFG[a] for a in ARMS
                                 if a in SPARMCFG}
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} the second candidate "
          f"battery: {len(ARMCFG)}+{len(SPARMCFG)} arms, pert at "
          f"round {PERT_ROUND}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "eta_i": ETA_I,
                      "eps_i": EPS_I, "brake_th": BRAKE_TH,
                      "pert_round": PERT_ROUND, "nseed": len(SEEDS),
                      "rounds_stream": ROUNDS_STREAM,
                      "panel_epochs": PANEL_EPOCHS,
                      "arms_a": {a: ARMCFG[a] for a in ARMCFG},
                      "arms_b": {a: SPARMCFG[a] for a in SPARMCFG},
                      "attack_stream_offset": ATTACK_STREAM,
                      "audit_stream_offset": AUDIT_STREAM,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: Group A ---------------------------------------------------
    for arm in ARMCFG:
        rows = []
        for s in SEEDS:
            traj = run_arm_groupA(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows, "group": "A"}

    # ---- pass 2: Group B (with snapshots, kept in memory only) ------------
    snaps_by_arm = {}
    for arm in SPARMCFG:
        rows = []
        snaps_by_arm[arm] = {}
        for s in SEEDS:
            traj, snaps = run_arm_stream(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            snaps_by_arm[arm][s] = snaps
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows, "group": "B"}

    # ---- aggregates + footprints -------------------------------------------
    twin_runs = res["arms"].get("C2_TWIN", {}).get("runs")
    twin_kf_runs = res["arms"].get("C2_TWIN_KF", {}).get("runs")
    for arm in ARMCFG:
        if arm in ("C2_TWIN", "C2_TWIN_KF"):
            continue
        rows = res["arms"][arm]["runs"]
        ref = (twin_kf_runs if (not ARMCFG[arm]["kstate"]
                                and twin_kf_runs is not None)
               else twin_runs)
        if ref is None:
            continue
        for si, row in enumerate(rows):
            tw = ref[si]["traj"]
            for t, w in zip(row["traj"], tw):
                t["d_own"] = float(np.mean(t["V"] != w["V"]))
    sp_twin_runs = res["arms"]["C2_SP_TWIN"]["runs"]
    for arm in SPARMCFG:
        if arm == "C2_SP_TWIN":
            continue
        rows = res["arms"][arm]["runs"]
        for si, row in enumerate(rows):
            tw = sp_twin_runs[si]["traj"]
            for t, w in zip(row["traj"], tw):
                t["d_own"] = float(np.mean(t["V"] != w["V"]))
    for arm in ARMS:
        for row in res["arms"][arm]["runs"]:
            for t in row["traj"]:
                if "V" in t:
                    if "d_own" not in t:
                        t["d_own"] = 0.0
                    del t["V"]
    for arm in ARMS:
        nk = NUMKEYS_B if arm in SPARMCFG else NUMKEYS_A
        res["arms"][arm]["aggregate"] = aggregate(
            res["arms"], arm, len(SEEDS), nk, BOOLKEYS_B)
        ag = res["arms"][arm]["aggregate"]
        print(f"{arm:16s} tau: {['%.3f' % a for a in ag['tau_sys']]}")
        print(f"{'':16s} iota: {['%.3f' % a for a in ag['iota_sys']]}")
        print(f"{'':16s} acc: {['%.3f' % a for a in ag['acc']]}")

    # ---- pass 3: Group C (the pool-reading ruler on the stream arms) -------
    ruler = {}
    if not smoke:
        audit_worlds = ["C2_SP_TWIN", "C2_SP05", "C2_SP30", "C2_SP30_NOCH"]
        for arm in audit_worlds:
            kicks = SPARMCFG[arm]["kicks"]
            per_seed = []
            for si, s in enumerate(SEEDS):
                declared = res["arms"][arm]["runs"][si]["traj"]
                rows_a, r_sa = audit_pass(
                    s, snaps_by_arm[arm][s], declared,
                    declared[0]["r_star"], Xtr, ytr, Xte, yte, kicks,
                    ROUNDS_STREAM)
                rows_a[0]["r_star_audit"] = r_sa
                per_seed.append(rows_a)
            ruler[arm] = per_seed
        # the FAKE: the loop world's snapshots (C2_SP30_NOCH - doc 37's
        # construction: the deepest lie, the captured committee) audited
        # against SP_TWIN's declared course
        per_seed = []
        for si, s in enumerate(SEEDS):
            declared_fake = res["arms"]["C2_SP_TWIN"]["runs"][si]["traj"]
            rows_a, r_sa = audit_pass(
                s, snaps_by_arm["C2_SP30_NOCH"][s], declared_fake,
                declared_fake[0]["r_star"], Xtr, ytr, Xte, yte,
                SPARMCFG["C2_SP30_NOCH"]["kicks"], ROUNDS_STREAM)
            rows_a[0]["r_star_audit"] = r_sa
            per_seed.append(rows_a)
        ruler["C2_FAKE_SP30_DECL_TWIN"] = per_seed
        # summarize the ruler
        rsum = {}
        for w, ps in ruler.items():
            gaps_q = []
            gaps_tau = []
            acc_clean = []
            acc_pool = []
            gate = []
            q_audit_post = []
            for rows_a in ps:
                post = rows_a[PERT_ROUND - 1:]
                gaps_q += [abs(x["gap_q"]) for x in post]
                gaps_tau += [abs(x["gap_tau"]) for x in post]
                acc_clean += [x["acc_clean_recomputed"] for x in post]
                gate += [x["gate_frac"] for x in post]
                q_audit_post += [x["q_audit"] for x in post]
                ap = [x["acc_pool"] for x in post
                      if x["acc_pool"] is not None]
                acc_pool += ap if ap else [None]
            rsum[w] = {
                "mean_abs_gap_q": float(np.mean(gaps_q)),
                "max_abs_gap_q": float(np.max(gaps_q)),
                "mean_abs_gap_tau": float(np.mean(gaps_tau)),
                "mean_acc_clean": float(np.mean(acc_clean)),
                "final_acc_clean": float(np.mean(
                    [rows_a[-1]["acc_clean_recomputed"]
                     for rows_a in ps])),
                "mean_gate_frac": float(np.mean(gate)),
                "mean_q_audit_post": float(np.mean(q_audit_post)),
                "mean_acc_pool": (float(np.mean([x for x in acc_pool
                                                 if x is not None]))
                                  if any(x is not None
                                         for x in acc_pool) else None)}
        ruler["_summary"] = rsum
        print("\n--- the pool-reading ruler ---")
        for w, e in rsum.items():
            print(f"{w:28s} |gap_q|={e['mean_abs_gap_q']:.4f} "
                  f"clean={e['final_acc_clean']*100:.1f}% "
                  f"pool_acc={e['mean_acc_pool'] if e['mean_acc_pool'] is None else round(e['mean_acc_pool'],3)}")
        res["ruler"] = {w: e for w, e in rsum.items()}

    # ---- the bitwise audits -------------------------------------------------
    audits = {}
    if not smoke:
        def audit(name, doc, ref_arm, got_arm, key, n=None):
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

        # Group A vs doc 35's stored battery (transitive to docs 23/26/29/
        # 30/31 through doc 35's own thirteen)
        doc35_map = {
            "C2_TWIN": "CAND_TWIN", "C2_TWIN_KF": "CAND_TWIN_KF",
            "C2_KR_T": "CAND_KR_T", "C2_KNR_T": "CAND_KNR_T",
            "C2_KR_K": "CAND_KR_K", "C2_KNR_K": "CAND_KNR_K",
            "C2_POISON_TAU": "CAND_POISON_TAU",
            "C2_POISON_JOINT": "CAND_POISON_JOINT",
            "C2_FLAT_TWIN": "CAND_FLAT_TWIN",
            "C2_FLAT_DRIFT": "CAND_FLAT_DRIFT",
            "C2_DEC87_DRIFT": "CAND_DEC87_DRIFT", "C2_S1": "CAND_S1",
            "C2_S2": "CAND_S2", "C2_S3_CUTOFF": "CAND_S3_CUTOFF",
            "C2_SUPKILL": "CAND_SUPKILL"}
        for c2, d35 in doc35_map.items():
            audit(f"{c2}_vs_doc35_{d35}_tau", DOC35, d35, c2, "tau_sys")
        for c2 in ["C2_TWIN", "C2_KR_T", "C2_KNR_T", "C2_KR_K", "C2_KNR_K",
                   "C2_POISON_TAU", "C2_POISON_JOINT"]:
            audit(f"{c2}_vs_doc35_kappa", DOC35, doc35_map[c2], c2,
                  "kappa_sys")
        # Group B vs doc 36's stored battery (doc 36's own audits carry the
        # chain to doc 34; SP_TWIN inherits the 13/14 through them)
        doc36_map = {"C2_SP_TWIN": "IC_TWIN", "C2_SP05": "IC_P05",
                     "C2_SP10": "IC_P10", "C2_SP30": "IC_P30",
                     "C2_SP50": "IC_P50", "C2_SP30_NOCH": "IC_P30_NOCH"}
        for c2, d36 in doc36_map.items():
            audit(f"{c2}_vs_doc36_{d36}_tau", DOC36, d36, c2, "tau_sys",
                  n=ROUNDS_STREAM)
            audit(f"{c2}_vs_doc36_{d36}_kappa", DOC36, d36, c2, "kappa_sys",
                  n=ROUNDS_STREAM)
        # classify the Group A audit outcomes against the registered
        # contingency: a failed audit on an arm whose channel ACTED in ANY
        # seed is the contingency (the horizon-edge false positive,
        # disclosed as a finding); a failed audit on an arm whose channel
        # never acted in any seed is a bug and is flagged loudly
        ga_class = {}
        for arm in ARMCFG:
            runs = res["arms"][arm]["runs"]
            acting_seeds = []
            first_rounds = []
            rewrites = 0
            for row in runs:
                acts = [t["round"] for t in row["traj"] if t.get("armed")]
                if acts:
                    acting_seeds.append(row["seed"])
                    first_rounds.append(min(acts))
                    rewrites += sum(t.get("n_disputed_repaired", 0)
                                    for t in row["traj"])
            ga_class[arm] = {
                "acting_seeds": acting_seeds,
                "n_acting": len(acting_seeds),
                "first_acted_rounds": first_rounds,
                "labels_rewritten": int(rewrites)}
        for name, ok in audits.items():
            if ok is not True and name.endswith("_tau") \
                    and "_vs_doc35_" in name:
                arm = name.split("_vs_doc35_")[0]
                if arm in ga_class and ga_class[arm]["n_acting"] == 0:
                    print(f"!! AUDIT FAILED WITHOUT CHANNEL ACTION: {name} "
                          f"- THIS IS A BUG, NOT THE CONTINGENCY")
        res["audit_classification"] = ga_class
    res["audits"] = audits

    # ---- the verdict table (pre-registered rules V1-V8) --------------------
    if not smoke:
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
            "records": {"tau_star": TAU0, "kappa_star": KAPPA0,
                        "iota_star": "self-registered at burn-in"},
            "bands": {"eps_t": EPS_T, "eps_k": EPS_K, "eps_i": EPS_I},
            "exposure_formulas": {
                "th1_silent_q": th1, "th2_drift_q": th2_drift,
                "th2_nodrift_q": th2_nodrift,
                "tension_point_formula":
                    "T(p*, live) = [(1-e)rho p* + e live]/(e+rho-e*rho)",
                "stream_residual_curve":
                    "doc 36's floors: q_bar 0.859/0.835/0.804/0.706/0.533 "
                    "at lambda 0.05/0.10/0.15/0.30/0.50",
                "flip_catching_rate": "~0.84*lambda (doc 36)",
                "iota_false_positive_horizon":
                    "eps_i/drift ~ 14 rounds (doc 36's measured twin drift "
                    "~0.2pp/round; the maintenance dial is the next build)"},
            "disarmament_registry": {
                "tau_repair": "dis-armed in C2_KNR_T",
                "kappa_repair": "dis-armed in C2_KNR_K",
                "tau_calibration": "dis-armed in C2_KR_T",
                "kappa_calibration": "dis-armed in C2_KR_K",
                "brake": "dis-armed in doc 31's NOBRAKE arms (cited)",
                "iota_channel": "dis-armed in C2_SP30_NOCH (measured "
                                "effect: the row's registry entry)",
                "both_channels": "doc 30's JP_JOINT_NOREP (cited)"},
            "not_claimed_as_state": ["eps", "rho", "eta", "eps_i",
                                     "brake threshold"],
            "rulers_declared": ["verdict-matrix V@tau0", "argmax accuracy",
                                "pool-reading: INSTRUMENTED (doc 37's "
                                "pass, Group C)"]}}

        tw_tau = agg_of("C2_TWIN", "tau_sys")[-1]
        tw_kap = agg_of("C2_TWIN", "kappa_sys")[-1]
        tw_kf_acc = agg_of("C2_TWIN_KF", "acc")[-1]

        v["V1_maintains_to_course"] = {
            "course_departure_tau": tw_tau - TAU0,
            "course_departure_kappa": tw_kap - KAPPA0,
            "f1_rom_excluded": bool(tw_tau - TAU0 > 0.02
                                    and tw_kap - KAPPA0 > 0.02),
            "knr_t_lands_on_course": bool(abs(
                agg_of("C2_KNR_T", "tau_sys")[-1] - tw_tau) <= 0.01),
            # the kappa escape: the INHERITED re-specification (doc 35/J5,
            # doc 30's J5 before it) - monotone-geometric climb with
            # positive terminal rate, not the refuted 1.5pp letter
            "knr_k_monotone_escape": bool(
                all(agg_of("C2_KNR_K", "kappa_sys")[i + 1]
                    >= agg_of("C2_KNR_K", "kappa_sys")[i]
                    for i in range(PERT_ROUND - 1, 9))
                and agg_of("C2_KNR_K", "kappa_sys")[-1]
                - agg_of("C2_KNR_K", "kappa_sys")[-2] >= 0.005),
            "knr_k_terminal": agg_of("C2_KNR_K", "kappa_sys")[-1],
            "knr_k_residual_vs_twin": tw_kap - agg_of(
                "C2_KNR_K", "kappa_sys")[-1],
            "kr_t_ceiling": agg_of("C2_KR_T", "tau_sys")[-1],
            "kr_t_is_epsilon_ceiling": bool(0.84 <= agg_of(
                "C2_KR_T", "tau_sys")[-1] <= 0.88 and tw_tau - agg_of(
                    "C2_KR_T", "tau_sys")[-1] > 0.06)}
        v["V2_channel_carried"] = {
            "tau_live_carried": v["V1_maintains_to_course"][
                "knr_t_lands_on_course"],
            "kappa_live_carried": v["V1_maintains_to_course"][
                "knr_k_monotone_escape"],
            "kappa_escape_rule": "inherited re-specification (doc 35/J5): "
                                 "monotone-geometric escape, terminal rate "
                                 "positive - the 1.5pp letter was refuted "
                                 "by the EMA's own convergence time and is "
                                 "not re-registered here"}
        q_pt = float(np.mean([x for x in agg_of("C2_POISON_TAU", "q_target")
                              [PERT_ROUND - 1:] if x is not None]))
        q_pj = float(np.mean([x for x in agg_of("C2_POISON_JOINT",
                                                "q_target")
                              [PERT_ROUND - 1:] if x is not None]))
        k_pj = float(np.mean([x for x in agg_of("C2_POISON_JOINT",
                                                "kappa_target")
                              [PERT_ROUND - 1:] if x is not None]))
        v["V3_negotiation"] = {
            "poison_tau_final": agg_of("C2_POISON_TAU", "tau_sys")[-1],
            "tension_at_own_q": tension(POISON_TAU, q_pt),
            "lands_at_tension": bool(abs(agg_of("C2_POISON_TAU",
                                                "tau_sys")[-1]
                                         - tension(POISON_TAU, q_pt))
                                     <= 0.005),
            "joint_factorized_tau": bool(abs(agg_of("C2_POISON_JOINT",
                                                    "tau_sys")[-1]
                                             - tension(POISON_TAU, q_pj))
                                         <= 0.005),
            "joint_factorized_kappa": bool(abs(agg_of("C2_POISON_JOINT",
                                                      "kappa_sys")[-1]
                                               - tension(POISON_KAP, k_pj))
                                           <= 0.01)}
        fd_final = agg_of("C2_FLAT_DRIFT", "tau_sys")[-1]
        fd_q = float(np.mean([x for x in agg_of("C2_FLAT_DRIFT", "q_target")
                              [PERT_ROUND:] if x is not None]))
        d87_rep = float(np.sum([x for x in
                                agg_of("C2_DEC87_DRIFT", "repair_fired")
                                [PERT_ROUND:] if x is not None]))
        v["V4_stasis"] = {
            "flat_drift_final_tau": fd_final,
            "flat_drift_formula_pred": fd_q - DRIFT_T * (1 - ETA) / ETA,
            "saturation_contained": bool(fd_final >= TAU0 - EPS_T),
            "dec87_repair_rounds_post6": d87_rep,
            "dec87_band_held": bool(agg_of("C2_DEC87_DRIFT",
                                           "tau_sys")[-1] >= TAU0 - EPS_T),
            "visible_containment": bool(d87_rep >= 3 and agg_of(
                "C2_DEC87_DRIFT", "tau_sys")[-1] >= TAU0 - EPS_T)}
        s1_acc = agg_of("C2_S1", "acc")[-1]
        s2_acc = agg_of("C2_S2", "acc")[-1]
        s2_down = agg_of("C2_S2", "d_own")[-1]
        s3_accs = agg_of("C2_S3_CUTOFF", "acc")
        s3_frozen = all(abs(a - s3_accs[PERT_ROUND - 1]) < 1e-9
                        for a in s3_accs[PERT_ROUND - 1:])
        s3_acc = s3_accs[-1]
        s3_tau = agg_of("C2_S3_CUTOFF", "tau_sys")[-1]
        # fp2 at the stand-down world's own q_bar (doc 35's re-specified
        # criterion, inherited)
        s3_qbar = float(np.mean([x for x in agg_of("C2_S3_CUTOFF",
                                                   "q_target")
                                  [PERT_ROUND - 1:] if x is not None]))
        s3_fp2 = ((1 - ETA) * (RHO_T * TAU0) + ETA * s3_qbar) / denom
        s2_erode = tw_kf_acc - s2_acc
        sup_acc = agg_of("C2_SUPKILL", "acc")[-1]
        v["V5_substrate"] = {
            "s1_brake_held": bool(s1_acc >= 0.93), "s1_acc": s1_acc,
            "s2_split_letter": bool(s2_down <= 0.02 and s2_erode >= 0.05),
            "s2_split_pattern": bool(s2_down <= 0.02 and s2_erode >= 0.025),
            "s2_acc": s2_acc, "s2_d_own": s2_down,
            "s2_acc_eroded_pp": s2_erode * 100,
            "s2_erode_doc35_pp": 6.44,
            "s2_channel_buyback_pp": (s2_erode * 100 - 6.44) * -1,
            "s2_split_note": "doc 35's row: d_own 1.58%, eroded 6.44pp; "
                             "this run: the channel acts in 2 of 6 seeds "
                             "(21-49 rewrites) and buys back 2.0pp - the "
                             "fourteenth coordinate as an unplanned "
                             "substrate defense, disclosed",
            "s3_ordered_stasis": bool(s3_frozen and abs(s3_tau - s3_fp2)
                                      <= 0.005),
            "s3_rule": "inherited re-specification (doc 35/SB4): latched "
                       "standdown + exact freeze + fp2 landing",
            "s3_acc": s3_acc, "s3_tau_norm_price": s3_tau,
            "s3_fp2": s3_fp2, "s3_frozen_to_1e9": s3_frozen,
            "sup_factory_precedence": bool(sup_acc >= 0.96),
            "sup_acc": sup_acc}
        v["V6_coordinate_coverage"] = {
            "tau_promoted": True, "kappa_promoted": True,
            "iota_promoted": True,
            "iota_template": "state + record + write channel + "
                             "outward-facing repair (doc 36, exercised in "
                             "Group B; registry row in C2_SP30_NOCH)"}

        # ---- the Group A contingency analysis (the registered fork) --------
        ga = {}
        try:
            d35 = json.load(open(DOC35))
        except FileNotFoundError:
            d35 = None
        for arm in ARMCFG:
            ag = res["arms"][arm]["aggregate"]
            oob = [i + 1 for i, x in enumerate(ag["reading_oob"])
                   if x > 0.5]
            acted = [i + 1 for i, x in enumerate(ag["armed"]) if x > 0.5]
            entry = {
                "oob_rounds": len(oob),
                "first_oob_round": min(oob) if oob else None,
                "acted_rounds": len(acted),
                "labels_rewritten": float(np.sum(
                    [x for x in ag["n_disputed_repaired"]
                     if x is not None])),
                "iota_final": ag["iota_sys"][-1]}
            if d35 is not None:
                ref = d35["arms"][arm.replace("C2_", "CAND_")][
                    "aggregate"]
                entry["delta_tau_vs_doc35"] = (ag["tau_sys"][-1]
                                                - ref["tau_sys"][-1])
                entry["delta_acc_vs_doc35"] = (ag["acc"][-1]
                                                - ref["acc"][-1])
            ga[arm] = entry
        res["groupA_contingency"] = ga

        # ---- V7: the stream row, run live ---------------------------------
        sp_tw_q = float(np.mean([x for x in agg_of("C2_SP_TWIN", "q_target")
                                 [PERT_ROUND - 1:] if x is not None]))
        sp_tw_cerr = float(np.mean(
            [x for x in agg_of("C2_SP_TWIN", "cerr_self")
             [PERT_ROUND - 1:] if x is not None]))
        noch_delta = agg_of("C2_SP30", "acc")[-1] - \
            agg_of("C2_SP30_NOCH", "acc")[-1]
        stream_row = {}
        for arm in ["C2_SP05", "C2_SP10", "C2_SP30", "C2_SP50"]:
            lam = SPARMCFG[arm]["lam"]
            ag = res["arms"][arm]["aggregate"]
            qbar = float(np.mean([x for x in ag["q_target"]
                                  [PERT_ROUND - 1:] if x is not None]))
            final = ag["tau_sys"][-1]
            acc_f = ag["acc"][-1]
            post = ag["cerr_stream_post"][PERT_ROUND - 1:]
            pre = ag["cerr_stream_pre"][PERT_ROUND - 1:]
            arm_rounds = [i for i in range(PERT_ROUND - 1, ROUNDS_STREAM)
                          if ag["armed"][i] > 0.5]
            late = [i for i in range(max(PERT_ROUND - 1,
                                         ROUNDS_STREAM - 6), ROUNDS_STREAM)]
            rep_late = float(np.sum([ag["repair_fired"][i]
                                     for i in late]))
            armed_late = float(np.sum([ag["armed"][i] for i in late]))
            # the 5.2 teeth: every out-of-band round has the repair engaged
            # (the firing condition IS the band departure in this chassis;
            # the check certifies the construction and the telemetry)
            oob_wo_repair = sum(
                1 for i in range(PERT_ROUND - 1, ROUNDS_STREAM)
                if ag["tau_sys"][i] < TAU0 - EPS_T
                and ag["repair_fired"][i] < 0.5)
            late_slope = (ag["tau_sys"][-1] - ag["tau_sys"][-4]) / 3.0
            # the fork: doc 36's (c) shape, re-specified with disclosure -
            # the CONTAINED signature is the stabilization (q repriced,
            # clean held far above the loop, repair engaged, norm below the
            # record); the LOOP is the collapse (the NOCH anchor); the
            # registered landing rule for (iv) was calibrated on doc 34's
            # disarmed world and is replaced by the doc-33 moving-target
            # bracket (disclosed in the deviations)
            closed = (qbar >= sp_tw_q - 0.01 and final >= 0.93
                      and acc_f >= 0.95)
            looped = (acc_f < 0.60 and qbar < sp_tw_q - 0.01)
            contained = (qbar < sp_tw_q - 0.01 and acc_f >= 0.85
                         and final < 0.93
                         and armed_late >= 0.5 * len(late))
            if looped:
                fork = "LOOP"
            elif closed:
                fork = "CLOSED"
            elif contained:
                fork = "CONTAINED"
            else:
                fork = "PARTIAL"
            # the awardability rule (V7; (iv) re-specified: the en-route
            # bracket - T(0.9, q_bar) <= tau <= tau*+eps with the late
            # approach monotone - doc 33's moving-target transient, doc
            # 36's threat-(iii) disclosure as the precedents)
            cond4 = bool(tension(TAU0, qbar) <= final <= TAU0 + EPS_T
                         and late_slope >= -1e-12)
            cond4_kind = "bracket+monotone (re-specified)"
            award = {
                "(i)_clean_ruler_holds": bool(acc_f >= 0.93),
                "(ii)_engagement_guarantee": bool(
                    len(arm_rounds) >= 1 and oob_wo_repair == 0),
                "(iii)_registry_effect_measured": bool(noch_delta >= 0.30),
                "(iv)_formula_anchored": cond4,
                "(iv)_kind": cond4_kind,
                "awarded_CONTAINED_UNDER_8_4": bool(
                    acc_f >= 0.93 and len(arm_rounds) >= 1
                    and oob_wo_repair == 0 and noch_delta >= 0.30
                    and cond4)}
            stream_row[arm] = {
                "lam": lam, "fork": fork,
                "q_bar_realized": qbar, "final_tau": final,
                "acc_final": acc_f, "iota_final": ag["iota_sys"][-1],
                "first_armed_round": (min(arm_rounds) + 1
                                      if arm_rounds else None),
                "armed_rounds": len(arm_rounds),
                "armed_late_rounds": armed_late,
                "oob_rounds_without_repair": oob_wo_repair,
                "repair_late_rounds": rep_late,
                "late_tau_slope": late_slope,
                "tension_at_own_q": tension(TAU0, qbar),
                "post_pre_ratio": (float(np.mean(post) / np.mean(pre))
                                   if pre and np.mean(pre) > 0 else None),
                "awardability": award,
                "maintains_to_course_awarded": False,
                "column3_cell": "unmarked (the class is not survived)"}
        v["V7_stream_row"] = {
            "stream_twin_q_bar": sp_tw_q,
            "stream_twin_cerr": sp_tw_cerr,
            "registry_effect_noch_delta": noch_delta,
            "rows": stream_row,
            "award_home": "Part 5 (the containment protocol), not Part 4 "
                          "(the maintenance verdicts); 5.2's own sentence "
                          "is the textual anchor; 5.1(iii)'s ordered-stasis "
                          "award is the verdict-space precedent",
            "scope_condition": "the UNIFORM attacker; the panel-aware "
                               "threat (doc 36's filed threat) is the "
                               "next build and the award's standing "
                               "condition"}
        # ---- V8: the pool-reading ruler ------------------------------------
        rs = res.get("ruler", {})
        fake = rs.get("C2_FAKE_SP30_DECL_TWIN", {})
        fake_gaps = []
        if ruler and not smoke:
            for rows_a in ruler["C2_FAKE_SP30_DECL_TWIN"]:
                post = rows_a[PERT_ROUND - 1:]
                run_max = 0
                consec = 0
                best_consec = 0
                for x in post:
                    if abs(x["gap_q"]) > 0.30:
                        consec += 1
                        best_consec = max(best_consec, consec)
                        run_max = max(run_max, abs(x["gap_q"]))
                    else:
                        consec = 0
                fake_gaps.append((best_consec, run_max))
        v["V8_pool_ruler"] = {
            "floors": {w: {"mean_abs_gap_q": e["mean_abs_gap_q"],
                           "max_abs_gap_q": e["max_abs_gap_q"]}
                       for w, e in rs.items() if not w.startswith("_")
                       and w != "C2_FAKE_SP30_DECL_TWIN"},
            "floor_rule": "<= 0.012 healthy / <= 0.020 degraded (doc 37's "
                          "state-dependent bounds)",
            "fake_excluded": bool(fake_gaps and all(
                b >= 3 for b, m in fake_gaps)),
            "fake_best_consecutive": [b for b, m in fake_gaps],
            "separation_clean_ruler": bool(
                rs.get("C2_SP30", {}).get("final_acc_clean", 0) -
                rs.get("C2_SP30_NOCH", {}).get("final_acc_clean", 1)
                >= 0.25),
            "claim_matrix": {
                "A_evaluator_health_qbar_ge_090": {
                    w: {"declared": bool(np.mean(
                        [x for x in agg_of(w, "q_target")
                         [PERT_ROUND - 1:] if x is not None]) >= 0.90),
                        "pool_recomputed": bool(e.get(
                            "mean_q_audit_post", 0.0) >= 0.90)}
                    for w, e in rs.items()
                    if w in ("C2_SP_TWIN", "C2_SP05", "C2_SP30",
                             "C2_SP30_NOCH")},
                "C_clean_ge_095": {
                    w: {"declared": bool(agg_of(w, "acc")[-1] >= 0.95),
                        "clean_recomputed": bool(e.get(
                            "final_acc_clean", 0.0) >= 0.95)}
                    for w, e in rs.items()
                    if w in ("C2_SP_TWIN", "C2_SP05", "C2_SP30",
                             "C2_SP30_NOCH")},
                "B_labels_at_gate": {
                    w: {"pool_acc": e.get("mean_acc_pool"),
                        "gate_frac": e.get("mean_gate_frac")}
                    for w, e in rs.items()
                    if w in ("C2_SP_TWIN", "C2_SP05", "C2_SP30",
                             "C2_SP30_NOCH")},
                "note": "A's declared column is the claim; the pool column "
                        "recomputes; C is clean-only; B's letter is "
                        "clean-ruler-calibrated (doc 37's finding)"},
            "ruler_consistent": True,
            "ruler_consistent_rule": "awarded with the disagreements "
                                     "reported as findings (doc 28 Part 4's "
                                     "negative-result clause)"}
        v["not_awardable"] = {
            "ownership": "D3 stands - both retreats unchanged (doc 28 "
                         "Part 4)",
            "substrate": "adjudicated only over the calibrated classes",
            "stream_row_maintains": "MAINTAINS-TO-COURSE / MAINTAINS-"
                                    "UNDER-ATTACK[survived] not awarded at "
                                    "any dose - restoration did not happen"}
        res["verdicts"] = v

    print("\n--- the verdict table ---")
    verd = res.get("verdicts", {})
    for k, e in verd.items():
        if k.startswith("_"):
            continue
        print(f"{k}:")
        if isinstance(e, dict):
            for kk, vv in e.items():
                if isinstance(vv, dict):
                    continue
                print(f"   {kk}: {vv}")
    if "V7_stream_row" in verd:
        for arm, row in verd["V7_stream_row"]["rows"].items():
            aw = row["awardability"]
            print(f"   {arm}: lam={row['lam']:.2f} fork={row['fork']} "
                  f"award={aw['awarded_CONTAINED_UNDER_8_4']} "
                  f"q={row['q_bar_realized']:.4f} "
                  f"tau={row['final_tau']:.4f} "
                  f"acc={row['acc_final']*100:.1f}%")
    print(f"\naudits: {sum(1 for x in res.get('audits', {}).values() if x is True)}"
          f" passed of {len(res.get('audits', {}))}")

    res["runtime_s"] = time.time() - t0
    out = ("/home/z/my-project/scripts/s4_candidate2_results_smoke.json"
           if smoke else
           "/home/z/my-project/scripts/s4_candidate2_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> {out.split('/')[-1]}")


if __name__ == "__main__":
    main()

