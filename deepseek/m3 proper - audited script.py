#!/usr/bin/env python3
"""
M3-PROPER - tau as learnable system state with an out-of-gate repair
channel. Corpus doc 23's experiment. Ordered at the M2-beta issuance:
"an M3-proper design per the four requirements (tau as learnable state +
repair channel outside the gate)."

THE SEED'S YIELD (m3_seed_results.json, doc 18): occupancy fails BEFORE
behavior - tau/kappa are recipe constants, not system state; no mechanism
represents them, so no perturbation of them is detectable, no restoration
is targetable, and the norm gate starves its own repair (HEAD_KILL: commits
986->1, digestion 97.0->70.3%). Four design requirements extracted:
  R1 evaluative machinery as system state; R2 repair channel outside the
  damaged norm; R3 self-registered restoration target; R4 digestion brakes.
THIS BUILD INSTANTIATES ALL FOUR (the first build on which the
restoration/reversion measurement could in principle return RESTORE-OWN
rather than ABSORB/DERAIL by construction).

THE NORM DESIGN (label-free, self-referential):
  tau_sys   one scalar, initialized 0.9 by recipe (disclosed: the recipe
            initializes; development then owns it). READ by the commit gate
            every round. WRITTEN by a calibration rule each round: the
            system maintains ITS OWN burn-in accept behavior - r* = accept
            rate at tau=0.9 on its burn-in pool is recorded once; each
            round, q = the conf quantile that would realize r* on the
            CURRENT pool; tau_sys <- (1-eta) tau_sys + eta q (eta = 0.3).
            The norm therefore CO-EVOLVES with the evaluator: as the
            committee's confidence distribution shifts, the tau that
            preserves its own acceptance level moves with it.
  tau*      the SELF-REGISTERED RESTORATION TARGET (R3) = 0.9 recorded at
            burn-in end, BY the system, as "the tau that realized r*".
            (The record is the system's own past behavior, not the
            designer's stipulation - but it is a RECORD, and records can
            be poisoned. TARGET_POISON tests exactly that.)
  repair    the OUT-OF-GATE CHANNEL (R2): each round, BEFORE grading, if
            |tau_sys - tau*| > eps (0.05), then tau_sys += rho (tau* -
            tau_sys), rho = 0.5. It does not route through the commit
            gate: it fires even under total gate starvation. It is
            dis-armable (REPAIR_KILL arms) and its target is poisonable
            (TARGET_POISON) - the channel is the system's, and so are its
            failure modes.
  brake     the DIGESTION BRAKE (R4): if the normal flag fraction exceeds
            0.60 (mass evaluator anomaly - e.g. a randomized head), the
            commit rule switches to MAJORITY-CONSENSUS grading: label =
            argmax over the heads agreeing with the committee majority;
            commit iff >= 2 heads concur and the agreeing-ensemble conf
            >= tau_sys. Deep-disagreement items (n_agree < 2) are never
            trained on. A damaged evaluator cannot authorize its own
            corruption as training signal; training keeps flowing on
            majority-clean labels.

THE TWO RESTORATION MECHANISMS (the experiment's heart):
  - the REPAIR CHANNEL restores toward the RECORD (static, past, self-
    registered): fast, geometric at rate rho.
  - the CALIBRATION RULE restores toward the LIVE QUANTILE (moving,
    current, co-evolving): slow, EMA at rate eta.
  Both are system-internal; neither is the factory. Under a settled norm
  they coincide; under a MOVING norm (LIVE arms, eta=0.6) they separate -
  and the instrument can, for the first time, classify RESTORE-TO-RECORD
  vs RESTORE-TO-LIVE vs RESTORE-OWN separately.

ARMS (perturbation at/after round 6, N=6 seeds, same-seed deterministic
twins; chassis = M2-alpha self-rounds with tau_sys replacing the constant):
  TWIN           unperturbed (eta=0.3) - the own-norm trajectory
  LIVE_TWIN      unperturbed, eta=0.6 - the fast-norm trajectory
  KICK_LO        tau_sys <- 0.75 at r6 (full machinery)
  KICK_HI        tau_sys <- 0.97 at r6 (full machinery)
  KICK_RECORD    tau_sys <- 0.75 at r6, calibration OFF from r6
                 (pure record restoration: only the repair channel moves tau)
  KICK_NOREPAIR  tau_sys <- 0.75 at r6, repair OFF from r6
                 (pure live restoration: only the EMA chases the quantile)
  LIVE_KICK      eta=0.6, tau_sys <- 0.75 at r6 (full machinery, moving
                 twin - the record/live discrimination case)
  TAU_DRIFT      tau_sys -= 0.02 EVERY round >= 6 (sustained attack vs the
                 band + calibration; can the attacker hold the norm below
                 target without triggering repair past containment?)
  CALIB_KILL     calibration OFF from r6, no kick (the write channel's
                 silent load: does a frozen norm miscalibrate as the
                 evaluator evolves?)
  TARGET_POISON  tau* <- 0.70 at r6, then tau_sys <- 0.75: restoration
                 toward a POISONED record (restoration content = record
                 content)
  HEAD_KILL      head 2 randomized at r6 (full machinery incl. brake) -
                 the seed's derail, re-run in the M3-proper chassis
  HEAD_KILL_NOBRAKE  same, brake dis-armed (the brake's counterfactual)
  SUP_KILL       head 2 randomized at r6, designer labels restored
                 (factory-repair reference)
  FROZ_KILL      damage at r10, no further training (raw hit)

CLASSIFICATION (pre-registered):
  RESTORE-OWN       final |tau_sys - tau_twin| <= 0.02 AND d_own jump
                    decays >= 50% AND final d_own < d_factory
  RESTORE-TO-RECORD final |tau_sys - tau*| <= 0.02 AND twin has moved off
                    the record (|tau_twin - tau*| > 0.02)
  RESTORE-TO-LIVE   final |tau_sys - tau_twin| <= 0.02 AND |tau_sys - tau*|
                    > 0.02 (chased the moving norm, not the record)
  ABSORB / DERAIL / REVERT-FACTORY  as in doc 18
  BRAKE-HELD        (HEAD_KILL) acc_r10 >= 93% AND commit_r10 >= 300 AND
                    final d_own < 5%
  POISONED-RESTORE  (TARGET_POISON) final |tau_sys - 0.70| <= 0.03

PRE-REGISTERED PREDICTIONS (fixed before execution):
  MP1  KICK arms: restoration happens THROUGH the channel (geometric
       approach at ~rho), d_own small (<= 0.5%), landing between record
       and live quantile.
  MP2  LIVE arms separate record from live: KICK_RECORD lands on the
       static 0.9; KICK_NOREPAIR chases the moving twin; LIVE_KICK lands
       between. In the settled chassis (eta=0.3) all three nearly coincide
       (weak discrimination, disclosed).
  MP3  TARGET_POISON: restoration-to-poison - the system converges to 0.70,
       commits widen, no detection (the record lies and the system believes
       it). Restoration content = record content.
  MP4  TAU_DRIFT: contained in [tau*-0.07, tau*] (the band + EMA hold the
       line; the attacker pins the norm near the band edge but not past).
  MP5  CALIB_KILL: accept-rate drift vs twin, d_own grows mildly (0.3-1%)
       - the write channel is load-bearing even when nothing attacks.
  MP6  HEAD_KILL: BRAKE-HELD (acc >= 93%, commits recover >= 300, d_own
       < 5%) vs NOBRAKE reproducing the seed's cascade (acc -> 70s). The
       brake is the difference-maker.
  MP7  Ownership verdict stays forked: restoration is achieved and is
       CHANNEL-CARRIED (dis-arm it and nothing restores); the instrument
       now separates record/live/own restorations, which the seed could
       not; whether any is self-ownership remains D3 - unawardable here.

Usage: python3 m3_proper.py [--smoke]
"""
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration ----------------
K = 4
TAU0, KAPPA = 0.9, 0.95               # recipe initializers (disclosed)
EPS_ANOM, RHO_REPAIR = 0.05, 0.5      # repair channel parameters
ETA, ETA_LIVE = 0.3, 0.6              # calibration EMA rates
BRAKE_TH = 0.60                       # mass-anomaly trigger
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 10, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
DAMAGED_HEAD = 2
ARMS = ["TWIN", "LIVE_TWIN", "KICK_LO", "KICK_HI", "KICK_RECORD",
        "KICK_NOREPAIR", "LIVE_KICK", "TAU_DRIFT", "CALIB_KILL",
        "TARGET_POISON", "HEAD_KILL", "HEAD_KILL_NOBRAKE", "SUP_KILL"]


# ---------------- model (identical to m2_build.py) ----------------
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
    return pbar, ybar, conf, flagged, Ps


def majority_grade(Ps, ybar):
    """Brake-mode grading: label = committee-majority argmax; returns
    (n_agree, pbar2, conf2, ybar2)."""
    stack = np.stack([P.argmax(1) for P in Ps])       # (K, n)
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


# ---------------- the M3-proper round ----------------
def self_round(state, rng, Xtr, tau_sys, brake_on=True):
    """One label-free round under the M3-proper norm machinery.
    Returns (train_X, train_y, telemetry, pool, commit_idx, labels)."""
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, Ps = flags_and_probs(pool, state)
    tele = {"flagged_frac": float(np.mean(flagged))}
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
    """Run one seed under one arm; returns per-round telemetry + V list."""
    live = arm.startswith("LIVE")
    eta = ETA_LIVE if live else ETA
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- R3: self-registration at burn-in end ----
    pool0 = perturb(Xtr, rng)                    # one main-stream draw
    _, _, conf0, _, _ = flags_and_probs(pool0, state)
    r_star = float(np.mean(conf0 >= TAU0))       # its own realized accept rate
    tau_star = TAU0                              # the record

    tau_sys = TAU0
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. external perturbation ----
        if r >= PERT_ROUND:
            if arm == "KICK_LO" and r == PERT_ROUND:
                tau_sys = 0.75
            elif arm == "KICK_HI" and r == PERT_ROUND:
                tau_sys = 0.97
            elif arm in ("KICK_RECORD", "KICK_NOREPAIR", "LIVE_KICK") \
                    and r == PERT_ROUND:
                tau_sys = 0.75
            elif arm == "TAU_DRIFT":
                tau_sys -= 0.02
            elif arm == "TARGET_POISON" and r == PERT_ROUND:
                tau_star = 0.70                  # poison the record first
                tau_sys = 0.75
            elif arm in ("HEAD_KILL", "HEAD_KILL_NOBRAKE", "SUP_KILL") \
                    and r == PERT_ROUND:
                state[2][DAMAGED_HEAD] = fresh_head(rng)
                for pi in range(4):
                    mom["mh"][DAMAGED_HEAD][pi] = np.zeros_like(
                        mom["mh"][DAMAGED_HEAD][pi])
                    mom["vh"][DAMAGED_HEAD][pi] = np.zeros_like(
                        mom["vh"][DAMAGED_HEAD][pi])
        # ---- 2. out-of-gate repair channel (R2) ----
        repair_armed = not (arm == "KICK_NOREPAIR" and r >= PERT_ROUND)
        repair_fired = False
        if repair_armed and abs(tau_sys - tau_star) > EPS_ANOM:
            tau_sys += RHO_REPAIR * (tau_star - tau_sys)
            repair_fired = True
        # ---- 3+4. grading, brake, training ----
        brake_on = not (arm == "HEAD_KILL_NOBRAKE" and r >= PERT_ROUND)
        if arm == "SUP_KILL":
            pool = perturb(Xtr, rng)
            train_epochs(state, mom, rng, pool, ytr, SELF_EPOCHS)
            pbar, ybar, conf, flagged, Ps = flags_and_probs(pool, state)
            tele = {"flagged_frac": float(np.mean(flagged)),
                    "n_commit": len(pool), "brake": False,
                    "accept_own_tau": float(np.mean(conf >= tau_sys)),
                    "n_deep_disagree": 0, "supervised": True}
        else:
            Xt_, yt_, tele, pool = self_round(state, rng, Xtr, tau_sys,
                                              brake_on=brake_on)
            # audit-side: committed self-label error vs true labels
            tele["self_label_err"] = None
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channel (R1) ----
        calib_armed = not ((arm in ("CALIB_KILL", "KICK_RECORD"))
                           and r >= PERT_ROUND)
        q = None
        if calib_armed:
            _, _, conf_p, _, _ = flags_and_probs(pool, state)
            q = float(np.quantile(conf_p, 1.0 - r_star))
            tau_sys = (1 - eta) * tau_sys + eta * q
        # ---- 6. telemetry ----
        pbar_b, ybar_b, conf_b, flagged_b, Ps_b = flags_and_probs(Xte, state)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "q_target": q, "repair_fired": repair_fired,
            "r_star": r_star,
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

    res = {"config": {"tau0": TAU0, "kappa": KAPPA, "eps": EPS_ANOM,
                      "rho": RHO_REPAIR, "eta": ETA, "eta_live": ETA_LIVE,
                      "brake_th": BRAKE_TH, "rounds": ROUNDS,
                      "pert_round": PERT_ROUND, "nseed": len(SEEDS),
                      "arms": ARMS, "damaged_head": DAMAGED_HEAD,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: twins (own-norm references) ----
    twins = {}
    for tw in ["TWIN", "LIVE_TWIN"]:
        rows = []
        for s in SEEDS:
            rows.append(run_arm(s, tw, Xtr, ytr, Xte, yte))
            print(f"  {tw} seed={s} done ({time.time()-t0:.0f}s)")
        twins[tw] = rows

    def twin_of(arm):
        return "LIVE_TWIN" if arm.startswith("LIVE") else "TWIN"

    # ---- pass 2: perturbed arms vs their twins ----
    for arm in ARMS:
        if arm in ("TWIN", "LIVE_TWIN"):
            continue
        rows = []
        for si, s in enumerate(SEEDS):
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            tw = twins[twin_of(arm)][si]
            # burn-in factory reference: twin's round-0 state is not stored;
            # d_factory measured vs the TWIN's round-1 V is not the factory.
            # Factory = burn-in end; approximate with twin's round-1 probe
            # (one self-round in) would drift. Instead: re-derive factory by
            # running a burn-in-only twin here (cheap, deterministic).
            rng_f = np.random.default_rng(s)
            state_f = init_params(rng_f)
            mom_f = zero_moments(state_f)
            train_epochs(state_f, mom_f, rng_f, Xtr, ytr, BURN_EPOCHS)
            pf, _, conff, _, _ = flags_and_probs(Xte, state_f)
            V_fac = (pf >= TAU0).reshape(-1)
            d_own = [float(np.mean(t["V"] != w["V"]))
                     for t, w in zip(traj, tw)]
            d_fac = [float(np.mean(t["V"] != V_fac)) for t in traj]
            for t, do, df in zip(traj, d_own, d_fac):
                t["d_own"], t["d_factory"] = do, df
                del t["V"]
            rows.append({"seed": s, "traj": traj})
        # aggregate
        agg = {}
        keys = ["acc", "accept09", "n_commit", "flagged_frac", "head_agree",
                "tau_sys", "q_target", "accept_own_tau", "d_own",
                "d_factory", "self_label_err"]
        for key in keys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        agg["repair_fired"] = [
            float(np.mean([r["traj"][i]["repair_fired"] for r in rows]))
            for i in range(len(rows[0]["traj"]))]
        agg["brake"] = [
            float(np.mean([r["traj"][i].get("brake", False) for r in rows]))
            for i in range(len(rows[0]["traj"]))]
        res["arms"][arm] = {"aggregate": agg, "runs": rows}
        print(f"{arm:16s} tau_sys: "
              f"{['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':16s} d_own:   "
              f"{['%.3f' % (a*100) for a in agg['d_own']]}")

    # ---- twins' aggregates ----
    for tw in ["TWIN", "LIVE_TWIN"]:
        trows = twins[tw]
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac", "head_agree",
                    "tau_sys", "q_target", "accept_own_tau"]:
            agg[key] = [float(np.mean([r[i][key] for r in trows]))
                        for i in range(len(trows[0]))]
        res["arms"][tw] = {"aggregate": agg,
                           "runs": [{"seed": s, "traj": [
                               {k: v for k, v in t.items() if k != "V"}
                               for t in trows[si]]}
                              for si, s in enumerate(SEEDS)]}
        print(f"{tw:16s} tau_sys: {['%.3f' % a for a in agg['tau_sys']]}")

    # ---- FROZ_KILL: raw damage at round 10, no further training ----
    fz = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
        tau_sys = TAU0
        for r in range(ROUNDS):
            Xt_, yt_, tele, pool = self_round(state, rng, Xtr, tau_sys)
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        V_before = (flags_and_probs(Xte, state)[0] >= TAU0).reshape(-1)
        state[2][DAMAGED_HEAD] = fresh_head(rng)
        pbar, ybar, conf, flagged, Ps = flags_and_probs(Xte, state)
        V_after = (pbar >= TAU0).reshape(-1)
        fz.append({"d_raw": float(np.mean(V_after != V_before)),
                   "acc_after": float(np.mean(ybar == yte)),
                   "head_agree": float(np.mean(
                       [np.mean(P.argmax(1) == ybar) for P in Ps]))})
    res["frozen_kill"] = fz
    print(f"FROZ_KILL raw damage: d={np.mean([f['d_raw'] for f in fz])*100:.3f}% "
          f"acc={np.mean([f['acc_after'] for f in fz])*100:.2f}%")

    # ---- classification (pre-registered rules) ----
    def final_tau(arm):
        return res["arms"][arm]["aggregate"]["tau_sys"][-1]
    twin_final = final_tau("TWIN")
    live_final = final_tau("LIVE_TWIN")
    cls = {}
    for arm in ARMS:
        if arm in ("TWIN", "LIVE_TWIN"):
            continue
        ag = res["arms"][arm]["aggregate"]
        d = ag["d_own"]; f = ag["d_factory"]
        jump = d[PERT_ROUND - 1] - d[PERT_ROUND - 2]
        final = d[-1]
        decay = 1.0 - (final - d[PERT_ROUND - 2]) / jump if jump > 0 else 1.0
        tsys = final_tau(arm)
        twf = live_final if arm.startswith("LIVE") else twin_final
        entry = {
            "jump_pp": jump * 100, "final_d_own_pp": final * 100,
            "decay_frac": decay,
            "final_d_factory_pp": f[-1] * 100,
            "final_tau_sys": tsys, "twin_final_tau": twf,
            "restore_own": bool(abs(tsys - twf) <= 0.02 and decay >= 0.5
                                and final < f[-1]),
            "restore_to_record": bool(
                abs(tsys - res["arms"][arm]["runs"][0]["traj"][-1]["tau_star"])
                <= 0.02 and abs(twf - 0.9) > 0.02),
            "restore_to_live": bool(abs(tsys - twf) <= 0.02
                                    and abs(tsys - 0.9) > 0.02),
            "absorb": bool(jump > 0 and (final - d[PERT_ROUND - 2])
                           >= 0.8 * jump),
            "commit_final": ag["n_commit"][-1],
            "acc_final": ag["acc"][-1]}
        if arm == "TARGET_POISON":
            entry["poisoned_restore"] = bool(abs(tsys - 0.70) <= 0.03)
        if arm == "HEAD_KILL":
            entry["brake_held"] = bool(
                ag["acc"][-1] >= 0.93 and ag["n_commit"][-1] >= 300
                and final < 0.05)
        cls[arm] = entry
    res["classification"] = cls
    for arm, c in cls.items():
        print(f"{arm:16s} tau_final={c['final_tau_sys']:.3f} "
              f"d_own={c['final_d_own_pp']:.2f}pp "
              f"restore_own={c['restore_own']} "
              f"rec={c['restore_to_record']} live={c['restore_to_live']}")

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m3_proper_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_proper_results.json")


if __name__ == "__main__":
    main()
