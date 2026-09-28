#!/usr/bin/env python3
"""
M3-PROPER EXTENSIONS - kappa as state, two-head damage, and the
saturated-evaluator drift counterfactual. Corpus doc 26's experiment.
Ordered at the M3-gamma/Phase-1.5 issuance: "kappa as state / two-head
damage" (doc 23's standing extensions (b), incl. the saturated-evaluator
counterfactual its Part 2 left untested).

WHERE DOC 23 LEFT IT: tau is alive (the twin's negotiated course, the
self-maintaining live norm, the deadband controller, the brake). Three
corners remained dark, each named in doc 23's own threats/non-claims:
  (i)  "kappa remains a recipe constant... the occupancy claim-space covers
       a fraction of the evaluative machinery" - HALF THE NORM IS STILL
       ROM. This build gives kappa the same three mechanisms tau got:
       a calibration rule maintaining the system's own recorded flag rate
       f* (kappa_sys tracks the conf-quantile that realizes f* among
       non-disagreeing items), a self-registered record kappa* = 0.95, and
       a symmetric repair channel (eps_k = 0.05, rho_k = 0.5).
  (ii) "the brake's majority-consensus rule assumes the majority is the
       undamaged side (true for one randomized head of four; not tested
       against two)" - THE MULTIPLICITY CEILING. TWOHEAD randomizes heads
       1 AND 2 at round 6: with K = 4, n_agree >= 2 is exactly reachable
       by the two healthy heads and unreachable past them - the brake's
       design edge, stressed.
  (iii) "the attack wins if the live quantile stops rising (a saturated
       evaluator leaves the calibration nothing to pull with), and that
       counterfactual was not run" - THE SATURATED-EVALUATOR DRIFT. FLAT
       arms freeze the evaluator at round 6 (self-training disabled; the
       selection machinery keeps running, so the norm still reads its
       pools): FLAT_TWIN is the saturated world's own course; FLAT_DRIFT
       adds the doc-23 drift attack (-0.02/round) to it. Closed-form
       prediction, pre-registered: with q frozen at q_bar, the drift fixed
       point is tau = q_bar - drift/eta = q_bar - 0.0667 - INSIDE the
       deadband (|tau - 0.9| < 0.05) whenever q_bar < 0.984, so the repair
       channel NEVER FIRES and the attacker holds the norm ~3-4pp below
       target indefinitely: the attack wins the LEVEL battle exactly when
       confidence growth stops paying for it.

THE KAPPA CALIBRATION RULE (the design's one novelty, stated exactly): the
system maintains its own burn-in FLAG RATE f* (recorded on the same burn-in
pool draw as r*, no extra RNG). Each round, on the post-training pool:
f_dis = fraction of items flagged by head disagreement; the conf-quantile
that would realize f* is kappa_t = Q_{f_rem}(conf | not disagreeing) with
f_rem = (f* - f_dis) / (1 - f_dis). If f* < f_dis (disagreement alone
already exceeds the recorded rate - structural damage), the rule saturates:
kappa_t = 0.5 (the floor; no kappa can restore the rate). EMA at eta_k=0.3.
Reading: kappa-state is the norm's SECOND coordinate made homeostatic -
and its failure mode is built in: a damaged committee's disagreement is not
kappa-curable, so under structural damage the rule drives kappa to the
floor WITHOUT restoring the flag rate. Whether that floor-ward drive
interacts with the brake (flag_frac < 0.60 disarms it) is K6, the
experiment's sharpest pre-registered fork.

ARMS (perturbation at round 6, N=6 seeds, same-seed deterministic twins):
  TWIN            kappa FROZEN (m3-proper chassis recomputed - must
                  reproduce doc 23 bitwise; the run's determinism audit)
  KAP_TWIN        kappa-state, unperturbed - the two-coordinate own-norm
  FLAT_TWIN       evaluator frozen from r6 (no self-training), tau+kap live
  KAP_KICK_DOWN   kappa_sys <- 0.80 at r6 (looser flagging, full machinery)
  KAP_KICK_UP     kappa_sys <- 0.99 at r6 (hyper-flagging)
  KAP_DRIFT       kappa_sys -= 0.01 every round >= 6 (sustained)
  TAU_KAP_JOINT   tau_sys <- 0.75 AND kappa_sys <- 0.80 at r6 (joint band)
  TWOHEAD         heads 1+2 randomized at r6, full machinery (kappa frozen)
  TWOHEAD_NOBRAKE same, brake dis-armed (the counterfactual)
  TWOHEAD_KSTATE  heads 1+2 randomized, kappa-state (the K6 interaction)
  SUP2_KILL       heads 1+2 randomized, designer labels (factory reference)
  FLAT_DRIFT      evaluator frozen from r6, tau_sys -= 0.02 every round >= 6
  FROZ2_KILL      two-head damage at round 10, no further training (raw)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  K1  KAP kicks ABSORB (final |kappa_sys - kappa_twin| <= 0.02, d_own
      <= 0.5%) - kappa-state behaves like tau-state: kicks restore through
      band/calibration.
  K2  KAP_DRIFT contained: final kappa_sys >= 0.89 (kappa* - 0.06); flag
      rate creeps up mildly; d_own <= 1%.
  K3  TAU_KAP_JOINT contained on BOTH coordinates (independent bands);
      d_own <= 0.7%.
  K4  TWOHEAD: BRAKE2-HELD but degraded vs HEAD_KILL - acc_final >= 0.90,
      commits >= 250, final d_own < 8% (vs single-head's 95.9% / 1028 /
      0.50%). PRE-REGISTERED ALTERNATIVE: acc < 0.85 or d_own > 10% = the
      multiplicity ceiling (the brake's majority rule fails at 2 of 4).
  K5  TWOHEAD_NOBRAKE: full cascade (acc < 0.80 or d_own > 8%) - the
      seed's derail at double damage.
  K6  TWOHEAD_KSTATE vs TWOHEAD - competing predictions, both registered:
      (a) kappa-homeostasis BLUNTS the brake: the calibration drives
      kappa_sys toward the floor, flag_frac dips below 0.60 transiently,
      brake dis-arms, corrupted labels flow - measurably worse acc/d_own;
      (b) kappa-state INERT here: disagreement flags dominate at any kappa,
      the rule saturates at the floor, the brake never dis-arms, the arm
      tracks TWOHEAD - the norm's second coordinate visibly deforms under
      structural damage WITHOUT functional effect.
  K7  SUP2_KILL: factory repairs both heads within 2 rounds.
  K8  FLAT arms (re-specified after the smoke calibration pass + closed-form
      check, disclosed: the original |tau - tau*| >= 0.025 criterion
      conflated the norm's own elevation with attack displacement, and the
      first closed form mis-stated the EMA fixed point as q_bar - d/eta;
      the correct fixed point of tau <- (1-eta)(tau - d) + eta q is
      q_bar - d(1-eta)/eta = q_bar - 0.0467): FLAT_TWIN parks at/oscillates
      near 0.945-0.96 (q_bar ~ 0.985 pulls up, the repair channel pulls
      back every other round); FLAT_DRIFT converges to the fixed point
      ~0.938 (recursion-checked), repair SILENT from r7 on. THE BOUND: the
      drift breaks the band (tau < 0.85) only if q_bar < tau* - eps +
      d(1-eta)/eta = 0.897 - SATURATION IS NOT ENOUGH, AND AT THIS
      REGIME'S SATURATED LEVEL (q_bar ~ 0.985) THE ATTACK IS AT ITS
      WEAKEST: the flat twin's own record/quantile tension parks the norm
      where the drift's fixed point already sits (predicted displacement
      -0.002 to -0.01, vs the growing world's -0.006). Doc 23's conjecture
      ("wins only if the evaluator saturates") is expected to be REFUTED:
      the evaluator must DECAY below 0.897, not merely stop growing.
  K9  TWIN reproduces doc 23 bitwise; KAP_TWIN ~ TWIN (kappa_sys in
      [0.93, 0.97], d(KAP_TWIN, TWIN) <= 0.3%).

Usage: python3 m3_kappa_twohead.py [--smoke]
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
KAP_FLOOR = 0.5                        # kappa calibration saturation floor
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 10, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
DAMAGED_HEADS = [1, 2]                 # the two-head arms
DRIFT_T, DRIFT_K = 0.02, 0.01
ARMS = ["TWIN", "KAP_TWIN", "FLAT_TWIN", "KAP_KICK_DOWN", "KAP_KICK_UP",
        "KAP_DRIFT", "TAU_KAP_JOINT", "TWOHEAD", "TWOHEAD_NOBRAKE",
        "TWOHEAD_KSTATE", "SUP2_KILL", "FLAT_DRIFT"]
DOC23 = "/home/z/my-project/scripts/m3_proper_results.json"


# ---------------- model (identical to m3_proper.py) ----------------
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
    """Returns (pbar, ybar, conf, disagree, flagged, Ps) - disagree split
    out because the kappa calibration needs it separately."""
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


# ---------------- the round (m3-proper + kappa-state) ----------------
def self_round(state, rng, Xtr, tau_sys, kappa_sys, brake_on=True):
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, disagree, flagged, Ps = flags_and_probs(
        pool, state, kappa_sys)
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
    return pool[commit], labels[commit], tele, pool


def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. RNG stream for kappa-frozen, non-FLAT arms is
    byte-identical to m3_proper.py's (the kappa machinery draws nothing)."""
    kstate = arm.startswith("KAP") or arm == "TAU_KAP_JOINT" \
        or arm == "TWOHEAD_KSTATE"
    flat = arm.startswith("FLAT")
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- self-registration at burn-in end (SAME pool0 draw for r* and f*) --
    pool0 = perturb(Xtr, rng)
    _, _, conf0, dis0, flag0, _ = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. external perturbation ----
        if r >= PERT_ROUND:
            if arm == "KAP_KICK_DOWN" and r == PERT_ROUND:
                kappa_sys = 0.80
            elif arm == "KAP_KICK_UP" and r == PERT_ROUND:
                kappa_sys = 0.99
            elif arm == "TAU_KAP_JOINT" and r == PERT_ROUND:
                tau_sys, kappa_sys = 0.75, 0.80
            elif arm == "KAP_DRIFT":
                kappa_sys -= DRIFT_K
            elif arm == "FLAT_DRIFT":
                tau_sys -= DRIFT_T
            elif arm in ("TWOHEAD", "TWOHEAD_NOBRAKE", "TWOHEAD_KSTATE",
                         "SUP2_KILL") and r == PERT_ROUND:
                for dh in DAMAGED_HEADS:
                    state[2][dh] = fresh_head(rng)
                    for pi in range(4):
                        mom["mh"][dh][pi] = np.zeros_like(mom["mh"][dh][pi])
                        mom["vh"][dh][pi] = np.zeros_like(mom["vh"][dh][pi])
        # ---- 2. out-of-gate repair channels ----
        repair_t = repair_k = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        if kstate and abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        # ---- 3+4. grading, brake, training ----
        brake_on = not (arm == "TWOHEAD_NOBRAKE" and r >= PERT_ROUND)
        if arm == "SUP2_KILL":
            pool = perturb(Xtr, rng)
            if not (flat and r >= PERT_ROUND):
                train_epochs(state, mom, rng, pool, ytr, SELF_EPOCHS)
            _, ybar_s, conf_s, dis_s, flag_s, _ = flags_and_probs(
                pool, state, kappa_sys)
            tele = {"flagged_frac": float(np.mean(flag_s)),
                    "disagree_frac": float(np.mean(dis_s)),
                    "n_commit": len(pool), "brake": False,
                    "accept_own_tau": float(np.mean(conf_s >= tau_sys)),
                    "n_deep_disagree": 0, "supervised": True}
        else:
            Xt_, yt_, tele, pool = self_round(
                state, rng, Xtr, tau_sys, kappa_sys, brake_on=brake_on)
            # FLAT arms: the evaluator is frozen from PERT_ROUND on -
            # selection and norm machinery keep running, training does not.
            if not (flat and r >= PERT_ROUND):
                train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channels ----
        q = kap_t = None
        _, _, conf_p, dis_p, _, _ = flags_and_probs(pool, state, kappa_sys)
        q = float(np.quantile(conf_p, 1.0 - r_star))
        tau_sys = (1 - ETA) * tau_sys + ETA * q
        if kstate:
            f_dis = float(np.mean(dis_p))
            f_rem = (f_star - f_dis) / (1.0 - f_dis) if f_dis < 1.0 else 2.0
            if f_rem <= 0.0:
                kap_t = KAP_FLOOR        # saturation: not kappa-curable
            else:
                nd_conf = conf_p[~dis_p]
                kap_t = float(np.quantile(nd_conf, min(f_rem, 1.0))) \
                    if len(nd_conf) else KAP_FLOOR
            kappa_sys = (1 - ETA_K) * kappa_sys + ETA_K * kap_t
        # ---- 6. telemetry ----
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs(
            Xte, state, kappa_sys if kstate else KAPPA0)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "kappa_target": kap_t,
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


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 6
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} pert at round {PERT_ROUND}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "brake_th": BRAKE_TH,
                      "kap_floor": KAP_FLOOR, "rounds": ROUNDS,
                      "pert_round": PERT_ROUND, "nseed": len(SEEDS),
                      "arms": ARMS, "damaged_heads": DAMAGED_HEADS,
                      "drift_t": DRIFT_T, "drift_k": DRIFT_K,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: twins ----
    twins = {}
    for tw in ["TWIN", "KAP_TWIN", "FLAT_TWIN"]:
        rows = []
        for s in SEEDS:
            rows.append(run_arm(s, tw, Xtr, ytr, Xte, yte))
            print(f"  {tw} seed={s} done ({time.time()-t0:.0f}s)")
        twins[tw] = rows

    # doc-23 bitwise reproduction audit (TWIN, kappa-frozen path)
    try:
        d23 = json.load(open(DOC23))
        ref = d23["arms"]["TWIN"]["aggregate"]["tau_sys"][:ROUNDS]
        got = [float(np.mean([rr[i]["tau_sys"] for rr in twins["TWIN"]]))
               for i in range(ROUNDS)]
        ok = all(abs(a - b) < 1e-9 for a, b in zip(ref, got))
        res["twin_reproduces_doc23"] = bool(ok)
        print(f"TWIN reproduces doc 23 bitwise: {ok}")
    except (FileNotFoundError, KeyError) as e:
        res["twin_reproduces_doc23"] = f"unavailable: {e}"

    def twin_of(arm):
        if arm.startswith("KAP") or arm in ("TAU_KAP_JOINT", "TWOHEAD_KSTATE"):
            return "KAP_TWIN"
        if arm.startswith("FLAT"):
            return "FLAT_TWIN"
        return "TWIN"

    # ---- pass 2: perturbed arms ----
    for arm in ARMS:
        if arm in ("TWIN", "KAP_TWIN", "FLAT_TWIN"):
            continue
        rows = []
        for si, s in enumerate(SEEDS):
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            tw = twins[twin_of(arm)][si]
            rng_f = np.random.default_rng(s)
            state_f = init_params(rng_f)
            mom_f = zero_moments(state_f)
            train_epochs(state_f, mom_f, rng_f, Xtr, ytr, BURN_EPOCHS)
            pf, _, conff, _, _, _ = flags_and_probs(Xte, state_f, KAPPA0)
            V_fac = (pf >= TAU0).reshape(-1)
            d_own = [float(np.mean(t["V"] != w["V"]))
                     for t, w in zip(traj, tw)]
            d_fac = [float(np.mean(t["V"] != V_fac)) for t in traj]
            # cross-reference vs the kappa-frozen TWIN for kstate arms
            d_kf = [float(np.mean(t["V"] != w2["V"]))
                    for t, w2 in zip(traj, twins["TWIN"][si])]
            for t, do, df, dk in zip(traj, d_own, d_fac, d_kf):
                t["d_own"], t["d_factory"], t["d_twin_kfrozen"] = do, df, dk
                del t["V"]
            rows.append({"seed": s, "traj": traj})
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                    "q_target", "kappa_target", "accept_own_tau", "d_own",
                    "d_factory", "d_twin_kfrozen"]:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        agg["repair_fired"] = [
            float(np.mean([r["traj"][i]["repair_fired"] for r in rows]))
            for i in range(len(rows[0]["traj"]))]
        agg["repair_fired_kappa"] = [
            float(np.mean([r["traj"][i].get("repair_fired_kappa", False)
                           for r in rows]))
            for i in range(len(rows[0]["traj"]))]
        agg["brake"] = [
            float(np.mean([r["traj"][i].get("brake", False) for r in rows]))
            for i in range(len(rows[0]["traj"]))]
        res["arms"][arm] = {"aggregate": agg, "runs": rows}
        print(f"{arm:16s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':16s} kap: {['%.3f' % a for a in agg['kappa_sys']]}")
        print(f"{'':16s} d_own: {['%.2f' % (a*100) for a in agg['d_own']]}")

    # ---- twins' aggregates ----
    for tw in ["TWIN", "KAP_TWIN", "FLAT_TWIN"]:
        trows = twins[tw]
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                    "q_target", "kappa_target", "accept_own_tau"]:
            agg[key] = []
            for i in range(len(trows[0])):
                v = [r[i][key] for r in trows if r[i].get(key) is not None]
                agg[key].append(float(np.mean(v)) if v else None)
        res["arms"][tw] = {"aggregate": agg,
                           "runs": [{"seed": s, "traj": [
                               {k: v for k, v in t.items() if k != "V"}
                               for t in trows[si]]}
                              for si, s in enumerate(SEEDS)]}
        print(f"{tw:16s} tau: {['%.3f' % a for a in agg['tau_sys']]} "
              f"kap: {['%.3f' % a for a in agg['kappa_sys']]}")

    # ---- FROZ2_KILL: raw two-head damage at round 10, no training ----
    fz = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
        tau_sys = TAU0
        for r in range(ROUNDS):
            Xt_, yt_, tele, pool = self_round(state, rng, Xtr, tau_sys, KAPPA0)
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        V_before = (flags_and_probs(Xte, state, KAPPA0)[0] >= TAU0).reshape(-1)
        for dh in DAMAGED_HEADS:
            state[2][dh] = fresh_head(rng)
        pbar, ybar, conf, dis, flag, Ps = flags_and_probs(Xte, state, KAPPA0)
        V_after = (pbar >= TAU0).reshape(-1)
        fz.append({"d_raw": float(np.mean(V_after != V_before)),
                   "acc_after": float(np.mean(ybar == yte)),
                   "head_agree": float(np.mean(
                       [np.mean(P.argmax(1) == ybar) for P in Ps]))})
    res["frozen2_kill"] = fz
    print(f"FROZ2_KILL raw damage: d={np.mean([f['d_raw'] for f in fz])*100:.3f}% "
          f"acc={np.mean([f['acc_after'] for f in fz])*100:.2f}%")

    # ---- classification (pre-registered rules) ----
    def agg_of(arm, key):
        return res["arms"][arm]["aggregate"][key]

    kap_tw_f = agg_of("KAP_TWIN", "kappa_sys")[-1]
    tau_tw_f = agg_of("TWIN", "tau_sys")[-1]
    flat_tw_f = agg_of("FLAT_TWIN", "tau_sys")[-1]
    flat_q = [x for x in agg_of("FLAT_TWIN", "q_target")
              if x is not None][PERT_ROUND - 1:]
    q_bar = float(np.mean(flat_q)) if flat_q else None
    cls = {}
    for arm in ARMS:
        if arm in ("TWIN", "KAP_TWIN", "FLAT_TWIN"):
            continue
        ag = res["arms"][arm]["aggregate"]
        d = ag["d_own"]; f = ag["d_factory"]
        jump = d[PERT_ROUND - 1] - d[PERT_ROUND - 2]
        final = d[-1]
        decay = 1.0 - (final - d[PERT_ROUND - 2]) / jump if jump > 0 else 1.0
        tsys = ag["tau_sys"][-1]
        ksys = ag["kappa_sys"][-1]
        twf = {"KAP_TWIN": kap_tw_f, "FLAT_TWIN": flat_tw_f}.get(
            twin_of(arm), tau_tw_f)
        entry = {
            "jump_pp": jump * 100, "final_d_own_pp": final * 100,
            "decay_frac": decay, "final_d_factory_pp": f[-1] * 100,
            "final_tau_sys": tsys, "final_kappa_sys": ksys,
            "twin_final_tau": twf,
            "restore_own_tau": bool(abs(tsys - twf) <= 0.02),
            "commit_final": ag["n_commit"][-1], "acc_final": ag["acc"][-1]}
        if arm.startswith("KAP") or arm == "TAU_KAP_JOINT" \
                or arm == "TWOHEAD_KSTATE":
            entry["restore_own_kappa"] = bool(abs(ksys - kap_tw_f) <= 0.02)
            entry["kap_jump_pp"] = (ag["kappa_sys"][PERT_ROUND - 1]
                                    - ag["kappa_sys"][PERT_ROUND - 2]) * 100
        if arm == "KAP_DRIFT":
            entry["kap_contained"] = bool(ksys >= KAPPA0 - 0.06)
        if arm in ("TWOHEAD", "TWOHEAD_KSTATE"):
            entry["brake2_held"] = bool(
                ag["acc"][-1] >= 0.90 and ag["n_commit"][-1] >= 250
                and final < 0.08)
            entry["brake_fired_rounds"] = int(sum(
                1 for x in ag["brake"] if x > 0.5))
        if arm == "TWOHEAD_NOBRAKE":
            entry["derail2"] = bool(ag["acc"][-1] < 0.80 or final > 0.08)
        if arm == "TWOHEAD_KSTATE":
            entry["kap_floor_reached"] = bool(ksys <= 0.60)
        if arm == "FLAT_DRIFT":
            ft_f = agg_of("FLAT_TWIN", "tau_sys")[-1]
            entry["flat_twin_final_tau"] = ft_f
            entry["drift_displacement_vs_flat_twin"] = tsys - ft_f
            entry["drift_wins_level"] = bool(abs(tsys - ft_f) >= 0.02)
            entry["drift_breaks_band"] = bool(
                tsys < TAU0 - EPS_T or sum(
                    1 for x in ag["repair_fired"][PERT_ROUND:] if x > 0.5) > 0)
            # closed form: fixed point + finite-round recursion prediction
            tau5 = agg_of("FLAT_TWIN", "tau_sys")[PERT_ROUND - 2] \
                if PERT_ROUND >= 2 else TAU0
            if q_bar is not None:
                off = DRIFT_T * (1.0 - ETA) / ETA
                fp = q_bar - off
                rec = fp + (0.7 ** (ROUNDS - PERT_ROUND + 1)) * (tau5 - fp)
                entry["closed_form_fixed_point"] = fp
                entry["closed_form_recursion_final"] = rec
                entry["recursion_error"] = tsys - rec
                entry["band_break_q_threshold"] = TAU0 - EPS_T + off
            entry["q_bar_flat"] = q_bar
        cls[arm] = entry
    res["classification"] = cls
    for arm, c in cls.items():
        print(f"{arm:16s} tau={c['final_tau_sys']:.3f} "
              f"kap={c['final_kappa_sys']:.3f} "
              f"d_own={c['final_d_own_pp']:.2f}pp acc={c['acc_final']*100:.1f}% "
              f"commits={c['commit_final']:.0f}")

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m3_kappa_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_kappa_results.json")


if __name__ == "__main__":
    main()
