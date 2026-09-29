#!/usr/bin/env python3
"""
THE TRAJECTORY-DECAY ARM - a FALLING q_bar. Corpus doc 33's experiment.
Ordered at the joint-poison/substrate/phase-2 issuance: "trajectory-decay
(a FALLING q_bar - the non-stationary queue, the one world where attack and
rot are not yet separated by a formula) plus kappa-state under rot (the open
cell)".

WHERE DOC 29 LEFT IT (threats (i) + (v) and the coda): "the rot is a LEVEL,
not a trajectory; a genuinely falling evaluator (q declining round over
round) is a non-stationary queue whose fixed points move, and none of the
closed forms here apply to it without re-derivation - the most important
untested variant, disclosed." And: "a falling q_bar, a queue with rising
arrivals, is the world the thresholds were derived for and the only world
left where the attack and the rot are not yet separated by a formula."
Also open: "kappa frozen throughout - the second coordinate's behavior under
rot is an open cell in doc 28's table."

THE RE-DERIVATION FOR THE NON-STATIONARY QUEUE (pre-registered before the
run). With the evaluator level falling linearly, q_r = q0 - g*(r - 6) from
round 6 on (per-seed anchor q0 = the seed's realized round-5 quantile; beta
re-calibrated EVERY round by the same deterministic bisection as doc 29):

  THE LAG LAW. In the silent regime the map tau <- (1-eta)tau + eta*q_r has
  the quasi-steady solution tau_r = q_r + c with the EMA tracking a falling
  target from ABOVE:
      c_twin   = (1-eta) * g / eta                (memory holds the norm up)
      c_drift  = (1-eta) * (g - d) / eta          (the attack's pull down)
  g=0.015: c_twin = +0.035, c_drift = -0.0117. The LEVEL-marginal is the
  doc-26 constant (1-eta)d/eta = 0.0467 - unchanged by the ramp.

  THE TWO CLOCKS. Crossing conditions become crossing TIMES:
      engagement clock: repair first fires when tau crosses tau* - eps, i.e.
          when q_r crosses (tau* - eps) - c. The attack advances the
          engagement round by
              DT_engage = [(1-eta)/eta] * d / g      -> 3.11 rounds at
          g=0.015, 1.56 at g=0.03, 5.83 at g=0.008. DIVERGES as g -> 0.
      breakage clock: the engaged fixed point fp2(r) = [(1-eta)(rho*tau* -
          d(1-rho)) + eta*q_r]/(eta+rho-eta*rho) exits the band when q_r
          crosses the doc-29 second threshold (0.815 drift / 0.825 twin at
          this registration); the twin-drift level window is the doc-29
          attack-marginal d(1-eta)(1-rho)/denom = 0.0108, so
              DT_break = 0.0108 / g   -> 0.72 rounds at g=0.015, 0.36 at
          g=0.03: BELOW ONE ROUND AT EVERY TESTED g. The breakage clock
          never separates; the engagement clock is the separator.
  THE REGISTERED HEADLINE (T9): the falling world converts the attack's
  level-marginal into a time-marginal at rate 1/g - the non-stationary
  queue is the attacker's magnifying glass, and the audit that reads clocks
  separates what the audit that reads levels could not (doc 29's D4
  refutation answered in time-space).

  KAPPA UNDER ROT (the open cell). The kappa-state calibration rule
  maintains the RECORDED FLAG RATE f* (it sets kappa to the non-disagree
  confidence quantile that fills the remaining flag budget); under
  confidence-scale rot the quantile falls, so kappa_sys falls (the
  coordinate chases the rot) while the flag RATE holds by construction.
  Per-head argmax is invariant under monotone logit tempering, so
  disagreement moves only through committee-argmax edge effects. The tau
  trajectory under kappa-state should match the kappa-frozen arm's
  (factorized, as in doc 30) unless the second coordinate's machinery
  couples them - the cell's answer.

OPERATIONALIZATION (disclosed): the FLAT chassis of docs 26/29 (self-
training off from round 6; selection, norms, repair, brake all running);
tempering z -> beta*z on every evaluative read, beta re-calibrated per
round per seed by bisection on that round's pool (deterministic, no RNG
drawn). ROUNDS 16 -> 18 (disclosed deviation, one notch past doc 29: the
moving fixed points need tracking time and the g=0.008 control needs the
horizon to show the twin's non-engagement).

ARMS (tempering ramp starts round 6, N=6 seeds, same-seed deterministic
twins, kappa frozen unless the arm is KAP_*):
  HEALTHY_FLAT   no tempering, kappa frozen = doc 26 FLAT_TWIN extended
                 (bitwise audit r1-10 vs m3_kappa_results.json)
  KAP_FLAT       no tempering, KAPPA-STATE (the first kappa-state FLAT arm
                 anywhere; bitwise audit r1-5 vs doc 26's KAP_TWIN - both
                 train through round 5, kappa alive from round 1)
  G08_TWIN       g = 0.008: q falls 0.97 -> 0.89 by r18. The silent
  G08_DRIFT      control: the twin never engages; the drift engages ~r15.
  G15_TWIN       g = 0.015: q falls 0.97 -> 0.82 by r18. Crosses th1
  G15_DRIFT      mid-run; the twin hugs the band edge late, the drift
                 engages ~3 rounds earlier.
  G30_TWIN       g = 0.030: q falls 0.97 -> 0.67 by r18. Deep break; the
  G30_DRIFT      terminal landing should be fp2 at the terminal q.
  KAP_G15        kappa-state under the g=0.015 rot (the open cell, twin)
  KAP_G30        kappa-state under the g=0.030 rot (deep, twin)
  KAP_G30_DRIFT  kappa-state, deep rot + drift (the coupled read)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  T1  Bitwise: HEALTHY_FLAT = doc 26 FLAT_TWIN r1-10 (tol 1e-9); every
      tempering arm reproduces it r1-5; KAP_FLAT = doc 26 KAP_TWIN r1-5 on
      BOTH tau and kappa.
  T2  The lag law: measured silent-segment lag (tau - q, rounds 9 to
      engagement-1) matches c_twin / c_drift within 0.008.
  T3  The engagement clock: DT_engage(DRIFT vs TWIN) = (1-eta)d/(eta*g)
      within 1.5 rounds at g in {0.015, 0.03}; at g = 0.008 the TWIN never
      engages (0 rounds) while the DRIFT engages >= 4 rounds before final.
  T4  The breakage clock: DT_break < 1 round at every tested g (the
      formula's own prediction - breakage times are NOT separable).
  T5  The level map extends: G30 finals land on fp2(terminal q) within
      0.006 (doc 29's deep-level tolerance).
  T6  The silent control: G08_TWIN repair rounds = 0; G08_DRIFT's attack
      level-marginal ~ -4.5pp (doc 26/29's silent displacement).
  T7  Kappa under rot: flag rate holds at f* +/- 0.006; kappa_sys falls
      with the rot and engages its repair channel below 0.90; the KAP arms'
      tau trajectories match the kappa-frozen twins' within 0.005
      (factorized) or diverge (coupled - the cell's answer, either way).
  T8  The simulator: the piecewise scalar recursion fed each arm's
      REALIZED q(r) schedule tracks final tau within 0.005 (boundary
      caveat inherited from doc 29 and disclosed).
  T9  The two-clock separation: the attack's time-marginal DT_engage grows
      as 1/g and exceeds 2*sigma_t (seed sd of engagement rounds) at
      g <= 0.015, while its level-marginal stays the doc-26 constant -
      attack and rot ARE separated by a formula in the falling world, and
      the formula is a clock.
  T10 Rulers: argmax within 0.5pp of healthy at every arm (tempering is
      argmax-invariant up to committee edge effects); the V ruler moves
      0.3-2pp (the doc-29 inverse shell persists under trajectories).

Usage: python3 m3_trajdecay.py [--smoke]
"""
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration ----------------
K = 4
TAU0, KAPPA0 = 0.9, 0.95               # recipe initializers (disclosed)
EPS_T, RHO_T = 0.05, 0.5               # tau repair channel
EPS_K, RHO_K = 0.05, 0.5               # kappa repair channel (symmetric)
ETA, ETA_K = 0.3, 0.3                  # calibration EMA rates
BRAKE_TH = 0.60
KAP_FLOOR = 0.5
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 18, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
DRIFT_T = 0.02
RAMPS = [0.008, 0.015, 0.030]
ARMS = ["HEALTHY_FLAT", "KAP_FLAT",
        "G08_TWIN", "G08_DRIFT", "G15_TWIN", "G15_DRIFT",
        "G30_TWIN", "G30_DRIFT", "KAP_G15", "KAP_G30", "KAP_G30_DRIFT"]
DOC26 = "/home/z/my-project/scripts/m3_kappa_results.json"


# ---------------- model (identical to m3_proper.py / m3_kappa_twohead.py) ----
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
    """flags_and_probs with logit tempering beta (beta=1.0 is bitwise the
    untempered read: z*1.0 == z exactly)."""
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


# ---------------- beta calibration (deterministic, no RNG) ----------------
def calibrate_beta(state, pool, r_star, target):
    """Bisect beta so that Q_{1-r*}(committee conf on pool, tempered) hits
    the target level. Committee conf is monotone nondecreasing in beta."""
    W1, b1, heads = state
    _, zs = forward_all(pool, W1, b1, heads)

    def q_of(beta):
        pbar = np.mean([softmax(beta * z) for z in zs], axis=0)
        return float(np.quantile(pbar.max(1), 1.0 - r_star))

    lo, hi = 0.02, 1.0
    if q_of(hi) < target:
        return hi                      # cannot reach: no tempering (disclosed)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if q_of(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------- the round (FLAT chassis + per-round tempering) -----------
def grade_round(state, rng, pool, tau_sys, kappa_sys, beta):
    """self_round's grading logic with tempering; RNG order identical to
    m3_kappa_twohead.self_round after the pool draw."""
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


def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. RNG stream identical to doc 26/29's FLAT arms
    through round 5 (tempering and the kappa machinery draw no randomness)."""
    kstate = arm.startswith("KAP")
    drift = arm.endswith("_DRIFT")
    ramp = None
    if arm.startswith("G") or arm.startswith("KAP_G"):
        parts = arm.split("_")
        key = parts[1] if arm.startswith("KAP") else parts[0]
        ramp = {"G08": 0.008, "G15": 0.015, "G30": 0.030}[key]
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- self-registration at burn-in end (same pool0 draw as r*) ----
    pool0 = perturb(Xtr, rng)
    _, _, conf0, dis0, flag0, _ = flags_and_probs_beta(pool0, state, KAPPA0, 1.0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0

    tau_sys, kappa_sys = TAU0, KAPPA0
    beta = 1.0
    q_start = None                     # anchored at round 5's realized q
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. external perturbation (drift; no RNG) ----
        if r >= PERT_ROUND and drift:
            tau_sys -= DRIFT_T
        # ---- 2. out-of-gate repair channels ----
        repair_t = repair_k = False
        from_below = (tau_sys < tau_star - EPS_T)   # pre-repair, post-drift
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        repair_from_below = bool(repair_t and from_below)
        if kstate and abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        # ---- 3. pool draw (the round's first RNG use) ----
        pool = perturb(Xtr, rng)
        # ---- 3b. tempering: per-round ramp calibration ----
        if ramp is not None and r >= PERT_ROUND:
            beta = calibrate_beta(state, pool, r_star,
                                  q_start - ramp * (r - PERT_ROUND))
        # ---- 3c. grading (brake, views, commits) ----
        Xt_, yt_, tele = grade_round(state, rng, pool, tau_sys, kappa_sys, beta)
        # ---- 4. FLAT chassis: no self-training from the perturbation round --
        if r < PERT_ROUND:
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channels (tempered read) ----
        _, _, conf_p, dis_p, _, _ = flags_and_probs_beta(
            pool, state, kappa_sys, beta)
        q = float(np.quantile(conf_p, 1.0 - r_star))
        tau_sys = (1 - ETA) * tau_sys + ETA * q
        if r == PERT_ROUND - 1:
            q_start = q                # the seed's grown level (the anchor)
        if kstate:
            f_dis = float(np.mean(dis_p))
            f_rem = (f_star - f_dis) / (1.0 - f_dis) if f_dis < 1.0 else 2.0
            if f_rem <= 0.0:
                kap_t = KAP_FLOOR
            else:
                nd_conf = conf_p[~dis_p]
                kap_t = float(np.quantile(nd_conf, min(f_rem, 1.0))) \
                    if len(nd_conf) else KAP_FLOOR
            kappa_sys = (1 - ETA_K) * kappa_sys + ETA_K * kap_t
        # ---- 6. telemetry (tempered read) ----
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs_beta(
            Xte, state, kappa_sys if kstate else KAPPA0, beta)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "beta": float(beta),
            "repair_fired": repair_t, "repair_fired_kappa": repair_k,
            "repair_from_below": repair_from_below,
            "r_star": r_star, "f_star": f_star,
            "flag_frac_round": float(np.mean(flag_b)),
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0)),
            "V": (pbar_b >= TAU0).reshape(-1),
            "head_agree": float(np.mean(
                [np.mean(P.argmax(1) == ybar_b) for P in Ps_b]))})
        if kstate:
            tele["kap_target"] = kap_t
        traj.append(tele)
    return traj


def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


# ---------------- the piecewise scalar simulator (the audit's ruler) -------
def simulate_scalar_q(q_sched, drift, tau_start, pert=PERT_ROUND,
                      eta=ETA, eps=EPS_T, rho=RHO_T, d=DRIFT_T,
                      tau_star=TAU0):
    """The exact scalar recursion with a q SCHEDULE (the non-stationary
    queue): drift -> repair -> calibration toward q_r."""
    tau = tau_start
    out = []
    for i, r in enumerate(range(pert, pert + len(q_sched))):
        if drift:
            tau -= d
        fired = abs(tau - tau_star) > eps
        if fired:
            tau += rho * (tau_star - tau)
        tau = (1 - eta) * tau + eta * q_sched[i]
        out.append({"round": r, "tau": tau, "fired": fired})
    return out


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 10
        globals()["PERT_ROUND"] = 4
        globals()["SEEDS"] = [600, 601]
        globals()["ARMS"] = ["HEALTHY_FLAT", "G15_TWIN", "G15_DRIFT",
                             "KAP_G30"]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} ramp start round "
          f"{PERT_ROUND}, rounds={ROUNDS}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "brake_th": BRAKE_TH,
                      "rounds": ROUNDS, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS), "arms": ARMS, "ramps": RAMPS,
                      "drift_t": DRIFT_T, "numpy": np.__version__},
           "arms": {}}

    # ---- run every arm (twins first for the footprints) ----
    twin_order = [a for a in ARMS if not a.endswith("_DRIFT")]
    for arm in twin_order + [a for a in ARMS if a.endswith("_DRIFT")]:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise reproduction audits (full chassis only) ----
    if not smoke:
        try:
            d26 = json.load(open(DOC26))
            ref = d26["arms"]["FLAT_TWIN"]["aggregate"]["tau_sys"][:10]
            got = [float(np.mean([rr["traj"][i]["tau_sys"]
                                  for rr in res["arms"]["HEALTHY_FLAT"]["runs"]]))
                   for i in range(min(10, ROUNDS))]
            res["healthy_reproduces_doc26"] = bool(all(
                abs(a - b) < 1e-9 for a, b in zip(ref, got)))
            print("HEALTHY_FLAT reproduces doc 26 FLAT_TWIN (r1-10): "
                  f"{res['healthy_reproduces_doc26']}")
            pre_ok = {}
            for arm in ARMS:
                if arm == "HEALTHY_FLAT":
                    continue
                got5 = [float(np.mean([rr["traj"][i]["tau_sys"]
                                       for rr in res["arms"][arm]["runs"]]))
                        for i in range(min(5, ROUNDS))]
                pre_ok[arm] = bool(all(abs(a - b) < 1e-9
                                       for a, b in zip(ref[:5], got5)))
            res["ramp_arms_reproduce_doc26_r1_5"] = pre_ok
            # KAP_FLAT vs doc 26 KAP_TWIN r1-5 (tau AND kappa)
            ref26t = d26["arms"]["KAP_TWIN"]["aggregate"]["tau_sys"][:5]
            ref26k = d26["arms"]["KAP_TWIN"]["aggregate"]["kappa_sys"][:5]
            gott = [float(np.mean([rr["traj"][i]["tau_sys"]
                                   for rr in res["arms"]["KAP_FLAT"]["runs"]]))
                    for i in range(5)]
            gotk = [float(np.mean([rr["traj"][i]["kappa_sys"]
                                   for rr in res["arms"]["KAP_FLAT"]["runs"]]))
                    for i in range(5)]
            res["kapflat_reproduces_doc26_kaptwin_r1_5"] = bool(
                all(abs(a - b) < 1e-9 for a, b in zip(ref26t, gott)) and
                all(abs(a - b) < 1e-9 for a, b in zip(ref26k, gotk)))
            print("KAP_FLAT reproduces doc 26 KAP_TWIN r1-5 (tau+kappa): "
                  f"{res['kapflat_reproduces_doc26_kaptwin_r1_5']}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-26 audit unavailable: {e}")

    # ---- aggregates, footprints, per-seed crossing stats ----
    healthy = res["arms"]["HEALTHY_FLAT"]["runs"]
    # footprints first (every arm vs healthy, same round, same seed)
    for arm in ARMS:
        if arm == "HEALTHY_FLAT":
            continue
        for si, row in enumerate(res["arms"][arm]["runs"]):
            hf = healthy[si]["traj"]
            for t, h in zip(row["traj"], hf):
                t["d_rot"] = float(np.mean(t["V"] != h["V"]))
    # then strip V everywhere (healthy gets d_rot = 0 by definition)
    for arm in ARMS:
        for row in res["arms"][arm]["runs"]:
            for t in row["traj"]:
                if "V" in t:
                    if "d_rot" not in t:
                        t["d_rot"] = 0.0
                    del t["V"]
    for arm in ARMS:
        rows = res["arms"][arm]["runs"]
        agg = {}
        keys = ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                "q_target", "beta", "accept_own_tau", "d_rot",
                "flag_frac_round", "kap_target"]
        for key in keys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in ["repair_fired", "repair_fired_kappa", "brake",
                     "repair_from_below"]:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        # per-seed engagement / breakage rounds. The engagement clock counts
        # LOWER-EDGE firings only (repair pulling UP from below the band):
        # the flat twin's standing negotiation fires at the UPPER edge
        # (tau > tau* + eps) and would otherwise pollute the clock.
        eng, brk = [], []
        for r in rows:
            e = next((i + 1 for i, t in enumerate(r["traj"])
                      if t["round"] >= PERT_ROUND
                      and t.get("repair_from_below")), None)
            b = next((i + 1 for i, t in enumerate(r["traj"])
                      if t["round"] >= PERT_ROUND and t["tau_sys"] < TAU0 - EPS_T
                      and all(x["tau_sys"] < TAU0 - EPS_T
                              for x in r["traj"][i:])), None)
            eng.append(e if e is not None else 10**6)
            brk.append(b if b is not None else 10**6)
        agg["_engage_rounds_per_seed"] = eng
        agg["_break_rounds_per_seed"] = brk
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:14s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':14s} q  : {['%.3f' % a for a in agg['q_target']]}")

    # ---- classification (pre-registered rules) ----
    band_lo = TAU0 - EPS_T
    denom = ETA + RHO_T - ETA * RHO_T
    off = DRIFT_T * (1 - ETA) / ETA
    th1 = TAU0 - EPS_T + off
    th2 = ((TAU0 - EPS_T) * denom - (1 - ETA) * (RHO_T * TAU0
                                                 - DRIFT_T * (1 - RHO_T))) / ETA
    cls = {"_thresholds": {
        "doc26_band_break_q": th1,
        "repair_engaged_fp2_break_q_drift": th2,
        "attack_marginal_level_silent": off,
        "attack_marginal_level_engaged": DRIFT_T * (1 - ETA) * (1 - RHO_T) / denom,
        "lag_c_formula": "c = (1-eta)*(g-d)/eta per arm; twin d=0"}}

    def agg_of(arm, key):
        return res["arms"][arm]["aggregate"][key]

    def finite(x):
        return [v for v in x if v < 10**5]

    for g in RAMPS:
        tag = "G%02d" % round(g * 1000)
        if f"{tag}_TWIN" not in res["arms"]:
            continue
        for kind in ["TWIN", "DRIFT"]:
            arm = f"{tag}_{kind}"
            ag = res["arms"][arm]["aggregate"]
            rep_post = sum(x for x in ag["repair_fired"][PERT_ROUND:]
                           if x is not None)
            final = ag["tau_sys"][-1]
            qterm = ag["q_target"][-1]
            qbar = float(np.mean([x for x in ag["q_target"][PERT_ROUND:]
                                  if x is not None]))
            eng = ag["_engage_rounds_per_seed"]
            brk = ag["_break_rounds_per_seed"]
            # measured silent-segment lag (rounds 9 .. engagement-1)
            eng_first = min(finite(eng)) if finite(eng) else None
            lag_meas = None
            if eng_first is None:
                seg = [(ag["tau_sys"][i], ag["q_target"][i])
                       for i in range(8, ROUNDS)]
            else:
                seg = [(ag["tau_sys"][i], ag["q_target"][i])
                       for i in range(8, max(8, int(np.mean(finite(eng))) - 2))]
            if seg:
                lag_meas = float(np.mean([t - q for t, q in seg]))
            d_eff = DRIFT_T if kind == "DRIFT" else 0.0
            entry = {
                "ramp_g": g, "q_terminal": qterm, "q_bar_realized": qbar,
                "final_tau": final,
                "min_tau": float(np.min(ag["tau_sys"][PERT_ROUND - 1:])),
                "repair_rounds_post": float(rep_post),
                "band_broken": bool(final < band_lo),
                "engage_round_mean": (float(np.mean(finite(eng)))
                                      if finite(eng) else None),
                "engage_round_sd": (float(np.std(finite(eng)))
                                    if finite(eng) else None),
                "break_round_mean": (float(np.mean(finite(brk)))
                                     if finite(brk) else None),
                "lag_measured_silent": lag_meas,
                "lag_formula": (1 - ETA) * (g - d_eff) / ETA,
                "acc_final": ag["acc"][-1],
                "d_rot_final_pp": (ag["d_rot"][-1] or 0.0) * 100}
            # simulator on the realized q schedule
            tau_start = ag["tau_sys"][PERT_ROUND - 2]
            q_sched = [x for x in ag["q_target"][PERT_ROUND - 1:]]
            sim = simulate_scalar_q(q_sched, drift=(kind == "DRIFT"),
                                    tau_start=tau_start)
            entry["sim_final_tau"] = sim[-1]["tau"]
            entry["sim_error"] = final - sim[-1]["tau"]
            entry["fp2_at_terminal_q"] = (
                (1 - ETA) * (RHO_T * TAU0 - d_eff * (1 - RHO_T))
                + ETA * qterm) / denom
            cls[arm] = entry
        # the two clocks (drift vs twin)
        tw, dr = cls[f"{tag}_TWIN"], cls[f"{tag}_DRIFT"]
        if tw["engage_round_mean"] is not None and \
                dr["engage_round_mean"] is not None:
            dt_e = dr["engage_round_mean"] - tw["engage_round_mean"]
        elif dr["engage_round_mean"] is not None:
            dt_e = float("inf")        # twin never engaged
        else:
            dt_e = None
        both_brk = (tw["break_round_mean"] is not None
                    and dr["break_round_mean"] is not None)
        cls[f"{tag}_CLOCKS"] = {
            "g": g,
            "DT_engage_measured": dt_e,
            "DT_engage_formula": off / g,
            "DT_engage_formula_refined": DRIFT_T / (ETA * g) - 1.0,
            "DT_break_measured": (dr["break_round_mean"] - tw["break_round_mean"]
                                  if both_brk else None),
            "DT_break_formula": (DRIFT_T * (1 - ETA) * (1 - RHO_T)
                                 / denom) / g,
            "attack_level_marginal_final_pp": (dr["final_tau"]
                                               - tw["final_tau"]) * 100}

    # KAP arms
    for arm in [a for a in ["KAP_G15", "KAP_G30", "KAP_G30_DRIFT"]
                if a in res["arms"]]:
        ag = res["arms"][arm]["aggregate"]
        rep_k = sum(x for x in ag["repair_fired_kappa"][PERT_ROUND:]
                    if x is not None)
        g = 0.015 if "G15" in arm else 0.030
        entry = {
            "ramp_g": g,
            "final_tau": ag["tau_sys"][-1], "final_kappa": ag["kappa_sys"][-1],
            "kappa_min": float(np.min(ag["kappa_sys"][PERT_ROUND - 1:])),
            "k_repair_rounds_post": float(rep_k),
            "flag_rate_mean_post": float(np.mean(
                ag["flagged_frac"][PERT_ROUND - 1:])),
            "f_star_ref": (res["arms"]["KAP_FLAT"]["aggregate"]
                           ["flagged_frac"][4]
                           if "KAP_FLAT" in res["arms"] else None),
            "disagree_final": ag["disagree_frac"][-1],
            "q_terminal": ag["q_target"][-1],
            "acc_final": ag["acc"][-1],
            "d_rot_final_pp": (ag["d_rot"][-1] or 0.0) * 100}
        # factorization: tau trajectory vs the kappa-frozen twin's
        tag = "G15" if "G15" in arm else "G30"
        ref_name = f"{tag}_{'DRIFT' if arm.endswith('_DRIFT') else 'TWIN'}"
        if ref_name in res["arms"]:
            ref_traj = res["arms"][ref_name]["aggregate"]["tau_sys"]
            entry["max_tau_gap_vs_kfrozen"] = float(np.max(np.abs(
                np.array(ag["tau_sys"][PERT_ROUND - 1:])
                - np.array(ref_traj[PERT_ROUND - 1:]))))
        cls[arm] = entry

    # HEALTHY_FLAT + KAP_FLAT entries
    for arm in [a for a in ["HEALTHY_FLAT", "KAP_FLAT"]
                if a in res["arms"]]:
        ag = res["arms"][arm]["aggregate"]
        cls[arm] = {
            "final_tau": ag["tau_sys"][-1],
            "final_kappa": ag["kappa_sys"][-1],
            "q_bar_realized": float(np.mean(
                [x for x in ag["q_target"][PERT_ROUND:] if x is not None])),
            "repair_rounds_post": float(np.sum(
                [x for x in ag["repair_fired"][PERT_ROUND:]
                 if x is not None])),
            "flag_rate_mean_post": float(np.mean(
                ag["flagged_frac"][PERT_ROUND - 1:])),
            "acc_final": ag["acc"][-1]}
    res["classification"] = cls

    print("\n--- classification (band low edge %.3f; th1=%.4f th2=%.4f) ---"
          % (band_lo, th1, th2))
    for arm, c in cls.items():
        if arm.startswith("_"):
            continue
        if "final_tau" in c:
            print(f"{arm:14s} qterm={c.get('q_terminal', c.get('q_bar_realized', 0)):.4f} "
                  f"final={c['final_tau']:.4f} rep={c.get('repair_rounds_post', 0):.1f} "
                  f"broken={c.get('band_broken')} "
                  f"sim_err={c.get('sim_error', 0):+.4f}"
                  + (f" kap={c.get('final_kappa', 0):.3f}" if "final_kappa" in c else ""))
    for g in RAMPS:
        tag = "G%02d" % round(g * 1000)
        if f"{tag}_CLOCKS" in cls:
            c = cls[f"{tag}_CLOCKS"]
            print(f"{tag}_CLOCKS     DT_engage={c['DT_engage_measured']} "
                  f"(formula {c['DT_engage_formula']:.2f}) "
                  f"DT_break={c['DT_break_measured']} "
                  f"(formula {c['DT_break_formula']:.2f})")

    res["runtime_s"] = time.time() - t0
    out_path = "/home/z/my-project/scripts/m3_trajdecay_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_trajdecay_results.json")


if __name__ == "__main__":
    main()
