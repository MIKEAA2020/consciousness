#!/usr/bin/env python3
"""
THE SUBSTRATE-AWARE BRAKE - multiplicity cutoff, pollution probe, and the
substrate shell. Corpus doc 31's experiment. Ordered at the kappa/two-head
issuance: "the substrate-aware brake with multiplicity cutoff (doc 26's
design lesson)".

WHERE DOC 26 LEFT IT: at 2-of-4 head damage the brake's keep-training bet
INVERTS on the argmax ruler - the brake arm ends at 89.9% and still falling
while the brake-less arm, starved to zero commits by its own flag
conservatism, freezes at 97.3% (permutation p ~ 0.001). "The brake needs a
multiplicity cutoff it does not have." Doc 28 turned the finding into
governance (8.4's 5.1(iii): past the design edge the stand-down is
MANDATED) and named the exclusion it instruments (F5, the substrate shell:
norm maintenance on top of eroding machinery - "the trunk-pollution
transient has no arm").

THE TWO NEW MECHANISMS (both designer-held switches per the disarmament
registry; both draw NO randomness - the estimators read the pool's head
argmaxes already computed at grading):
  1. THE MULTIPLICITY CUTOFF. Each round, on the pool: for each item the
     plurality count plur = max_c #{heads argmaxing c}; core_est = median
     plur. The majority-consensus grading is trustworthy only while the
     healthy core is a supermajority: STAND DOWN (grade, telemetry, but
     refuse self-training) whenever core_est <= 2. At 1-of-4 core_est = 3
     (keep training - the cutoff arm at 1-of-4 IS the brake arm, by
     construction, disclosed); at 2-of-4 core_est = 2 (stand down); at
     3-of-4 core_est <= 2 (stand down).
  2. THE SUBSTRATE PROBE. The brake reads DIFFERENTIAL damage (heads
     disagreeing); trunk pollution is COMMON damage (all heads reading a
     corrupting shared feature layer). The probe watches the tightest
     head pair: core_pair(r) = max_ij mean(argmax_i == argmax_j) on the
     pool, referenced to the burn-in pool0 read. A sustained decline
     (core_pair <= ref - 0.03 for 2 consecutive rounds) declares
     self-pollution - the training the brake authorized is corrupting the
     substrate - and latches a permanent stand-down. The design bet: the
     healthy pair's agreement declines only when the trunk itself degrades.

THE TRUNK ARMS (the F5 instrument): TRUNK_HALF re-initializes W1's first
64 of 128 columns at round 6 (half the substrate's feature units replaced
by fresh random projections; optimizer moments zeroed; heads intact).
Shared corruption: the heads still largely AGREE with each other (the
damage is common, not differential), so the disagreement route of the flag
rule is blind to it; the conf route (conf < 0.95) is not. The fork, all
three registered: RECOVERY (self-training re-learns the trunk - the
substrate's own live restoration, the first of its kind), HANG (conf
collapse starves the gate - frozen damage), DECAY (the system digests its
own poison - accelerating ruin).

ARMS (perturbation at round 6, N=6 seeds, same-seed deterministic twins,
kappa frozen, ROUNDS=10 chassis):
  TWIN              unperturbed (doc 23 bitwise audit)
  S1_BRAKE          1-of-4 (head 2), full machinery (= doc 23 HEAD_KILL,
                    bitwise audit)
  S2_BRAKE          2-of-4 (heads 1,2), full machinery (= doc 26 TWOHEAD,
                    bitwise audit)
  S2_NOBRAKE        2-of-4, brake dis-armed (= doc 26 TWOHEAD_NOBRAKE,
                    bitwise audit)
  S2_CUTOFF         2-of-4, multiplicity cutoff (stand down at core<=2)
  S2_SUBSTRATE      2-of-4, probe brake (train until pollution detected)
  S3_CUTOFF         3-of-4 (heads 1,2,3), cutoff
  S3_NOBRAKE        3-of-4, brake dis-armed
  TRUNK_HALF        substrate damage, full machinery
  TRUNK_HALF_NOBRAKE same, brake dis-armed
  FROZT_KILL        raw trunk damage at round 10, no training (d_raw ref)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  SB1  Bitwise: TWIN = doc 23 TWIN; S1_BRAKE = doc 23 HEAD_KILL; S2_BRAKE
      = doc 26 TWOHEAD; S2_NOBRAKE = doc 26 TWOHEAD_NOBRAKE (aggregates,
      tol 1e-9).
  SB2  S2_CUTOFF: the mandated stand-down PRESERVES the argmax ruler
      (acc_final >= 0.965, ~= NOBRAKE's frozen 97.3) and SACRIFICES the
      norm coordinate (tau_final < 0.80, repair firing; d_own > 5%): the
      stand-down is norm-breaking. The cutoff's content vs NOBRAKE is that
      the stasis is PRINCIPLED (ordered by an estimator) not accidental
      (starved by gate conservatism).
  SB3  S2_SUBSTRATE fork: (a) the probe FIRES at rounds 7-9 (the healthy
      pair's pool agreement declines >= 3pp under polluted training) and
      the arm lands between the brake and no-brake arms (acc in
      [0.93, 0.965]); (b) the probe NEVER fires (the pool-side decline is
      under threshold in the 5-round window) and the arm tracks S2_BRAKE
      (acc < 0.91) - the probe's blind spot, disclosed either way via the
      logged core_pair trajectories.
  SB4  S3 arms: both freeze (acc >= 0.96); S3_CUTOFF's core_est reads
      <= 2 at r6 (the estimator sees 3-of-4); tau collapses < 0.80 in
      both (the starvation world's q ~ 0.5 drags the calibration; repair
      holds ~0.72 = the repair-engaged fixed point).
  SB5  TRUNK_HALF: core_est >= 3 THROUGHOUT (heads stay correlated under
      shared corruption - the cutoff is structurally blind to substrate
      damage; pre-registered as a structural fact, verified by telemetry);
      flag_frac > 0.60 via the conf route (brake fires); the three-way
      fork read against FROZT_KILL's raw level: RECOVERY if
      acc_r10 >= acc_raw + 5pp; HANG if within 5pp; DECAY if
      acc_r10 < acc_raw - 5pp.
  SB6  The policy map: at 1-of-4 keep-training is optimal-plus-healing
      (S1_BRAKE 95.9%, machinery restored); at 2-of-4 stasis dominates
      (S2_CUTOFF >= 96.5% vs S2_BRAKE < 91%); the multiplicity-optimal
      policy is piecewise and the cutoff implements it with one estimator.

Usage: python3 m3_substrate.py [--smoke]
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
ETA = 0.3                              # calibration EMA rate
BRAKE_TH = 0.60
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 10, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
PROBE_DELTA, PROBE_STREAK = 0.03, 2    # substrate probe parameters
TRUNK_KILL_COLS = 64                   # first 64 of 128 hidden units
ARMS = ["TWIN", "S1_BRAKE", "S2_BRAKE", "S2_NOBRAKE", "S2_CUTOFF",
        "S2_SUBSTRATE", "S3_CUTOFF", "S3_NOBRAKE", "TRUNK_HALF",
        "TRUNK_HALF_NOBRAKE"]
DOC23 = "/home/z/my-project/scripts/m3_proper_results.json"
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


# ---------------- the two substrate estimators (no RNG) ----------------
def head_stats(Ps):
    """core_est = median plurality count; core_pair = max pairwise head
    agreement (the tightest pair = the healthy core under head damage)."""
    stack = np.stack([P.argmax(1) for P in Ps])       # K x n
    n = stack.shape[1]
    plur = np.zeros(n, dtype=np.int64)
    for i in range(n):
        _, cnts = np.unique(stack[:, i], return_counts=True)
        plur[i] = cnts.max()
    core_est = float(np.median(plur))
    agree_ij = [float(np.mean(stack[i] == stack[j]))
                for i in range(K) for j in range(i + 1, K)]
    return core_est, max(agree_ij)


# ---------------- the round (doc-26 chassis, Ps returned) ----------------
def self_round(state, rng, Xtr, ytr, tau_sys, kappa_sys, brake_on=True):
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
    tele["commit_err"] = float(np.mean(labels[commit] != ytr[commit])) \
        if commit.any() else 0.0
    return pool[commit], labels[commit], tele, pool, Ps


def damage_heads(state, mom, rng, head_list):
    for dh in head_list:
        state[2][dh] = fresh_head(rng)
        for pi in range(4):
            mom["mh"][dh][pi] = np.zeros_like(mom["mh"][dh][pi])
            mom["vh"][dh][pi] = np.zeros_like(mom["vh"][dh][pi])


def damage_trunk(state, mom, rng):
    state[0][:, :TRUNK_KILL_COLS] = rng.normal(
        0.0, np.sqrt(2.0 / 64), (64, TRUNK_KILL_COLS))
    mom["mW1"][:, :TRUNK_KILL_COLS] = 0.0
    mom["vW1"][:, :TRUNK_KILL_COLS] = 0.0


def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. RNG stream for TWIN/S1_BRAKE/S2_BRAKE/
    S2_NOBRAKE is byte-identical to doc 23/26's (the estimators and the
    stand-down draw no randomness before the training decision; the
    reference arms never stand down)."""
    dmg = None
    if arm.startswith("S1"):
        dmg = ("heads", [2])
    elif arm.startswith("S2"):
        dmg = ("heads", [1, 2])
    elif arm.startswith("S3"):
        dmg = ("heads", [1, 2, 3])
    elif arm.startswith("TRUNK"):
        dmg = ("trunk", None)
    brake_mode = "std"
    if arm.endswith("NOBRAKE"):
        brake_mode = "nobrake"
    elif arm.endswith("CUTOFF"):
        brake_mode = "cutoff"
    elif arm.endswith("SUBSTRATE"):
        brake_mode = "probe"

    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- self-registration at burn-in end (same pool0 draw) ----
    pool0 = perturb(Xtr, rng)
    _, _, conf0, _, _, Ps0 = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    tau_star = TAU0
    ref_core_pair = head_stats(Ps0)[1]              # the probe's reference

    tau_sys, kappa_sys = TAU0, KAPPA0
    probe_streak, stood_down = 0, False
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. external perturbation (damage at the pert round) ----
        if r == PERT_ROUND and dmg is not None:
            if dmg[0] == "heads":
                damage_heads(state, mom, rng, dmg[1])
            else:
                damage_trunk(state, mom, rng)
        # ---- 2. out-of-gate repair channel ----
        repair_t = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        # ---- 3. grading, brake ----
        brake_on = not (brake_mode == "nobrake" and r >= PERT_ROUND)
        Xt_, yt_, tele, pool, Ps = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys, brake_on=brake_on)
        # ---- 3b. the substrate estimators (no RNG) ----
        core_est, core_pair = head_stats(Ps)
        tele["core_est"], tele["core_pair"] = core_est, core_pair
        # ---- 3c. the stand-down decision ----
        if brake_mode == "cutoff" and core_est <= 2:
            stood_down = True
        if brake_mode == "probe":
            if core_pair <= ref_core_pair - PROBE_DELTA:
                probe_streak += 1
            else:
                probe_streak = 0
            if probe_streak >= PROBE_STREAK:
                stood_down = True
        tele["standdown"] = bool(stood_down)
        tele["probe_streak"] = probe_streak
        # ---- 4. training (refused under stand-down) ----
        if not stood_down:
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channel ----
        _, _, conf_p, _, _, _ = flags_and_probs(pool, state, kappa_sys)
        q = float(np.quantile(conf_p, 1.0 - r_star))
        tau_sys = (1 - ETA) * tau_sys + ETA * q
        # ---- 6. telemetry ----
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs(
            Xte, state, KAPPA0)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": kappa_sys, "kappa_star": KAPPA0,
            "q_target": q, "repair_fired": repair_t, "r_star": r_star,
            "ref_core_pair": ref_core_pair,
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
    print(f"data: train={len(Xtr)} test={len(Xte)} damage at round {PERT_ROUND}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eta": ETA, "brake_th": BRAKE_TH,
                      "rounds": ROUNDS, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS), "arms": ARMS,
                      "probe_delta": PROBE_DELTA,
                      "probe_streak": PROBE_STREAK,
                      "trunk_kill_cols": TRUNK_KILL_COLS,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: the twin ----
    twins = {}
    for tw in ["TWIN"]:
        rows = []
        for s in SEEDS:
            rows.append(run_arm(s, tw, Xtr, ytr, Xte, yte))
            print(f"  {tw} seed={s} done ({time.time()-t0:.0f}s)")
        twins[tw] = rows

    # ---- pass 2: the damage arms ----
    for arm in ARMS:
        if arm == "TWIN":
            continue
        rows = []
        for si, s in enumerate(SEEDS):
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            tw = twins["TWIN"][si]
            rng_f = np.random.default_rng(s)
            state_f = init_params(rng_f)
            mom_f = zero_moments(state_f)
            train_epochs(state_f, mom_f, rng_f, Xtr, ytr, BURN_EPOCHS)
            pf, _, conff, _, _, _ = flags_and_probs(Xte, state_f, KAPPA0)
            V_fac = (pf >= TAU0).reshape(-1)
            d_own = [float(np.mean(t["V"] != w["V"]))
                     for t, w in zip(traj, tw)]
            d_fac = [float(np.mean(t["V"] != V_fac)) for t in traj]
            for t, do, df in zip(traj, d_own, d_fac):
                t["d_own"], t["d_factory"] = do, df
                del t["V"]
            rows.append({"seed": s, "traj": traj})
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "q_target",
                    "accept_own_tau", "d_own", "d_factory", "commit_err",
                    "core_est", "core_pair", "probe_streak", "standdown",
                    "repair_fired"]:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        agg["brake"] = [
            float(np.mean([r["traj"][i].get("brake", False) for r in rows]))
            for i in range(len(rows[0]["traj"]))]
        res["arms"][arm] = {"aggregate": agg, "runs": rows}
        print(f"{arm:18s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':18s} acc: {['%.3f' % a for a in agg['acc']]}")
        print(f"{'':18s} core: {['%.1f' % a for a in agg['core_est']]} "
              f"cpair: {['%.3f' % a for a in agg['core_pair']]}")

    # ---- twin aggregate ----
    trows = twins["TWIN"]
    agg = {}
    for key in ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "q_target",
                "accept_own_tau", "commit_err", "core_est", "core_pair"]:
        agg[key] = []
        for i in range(len(trows[0])):
            v = [r[i][key] for r in trows if r[i].get(key) is not None]
            agg[key].append(float(np.mean(v)) if v else None)
    agg["repair_fired"] = [
        float(np.mean([r[i]["repair_fired"] for r in trows]))
        for i in range(len(trows[0]))]
    agg["brake"] = [
        float(np.mean([r[i].get("brake", False) for r in trows]))
        for i in range(len(trows[0]))]
    res["arms"]["TWIN"] = {"aggregate": agg,
                           "runs": [{"seed": s, "traj": [
                               {k: v for k, v in t.items() if k != "V"}
                               for t in trows[si]]}
                              for si, s in enumerate(SEEDS)]}

    # ---- FROZT_KILL: raw trunk damage at round 10, no training ----
    fz = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
        tau_sys = TAU0
        for r in range(ROUNDS):
            Xt_, yt_, tele, pool, Ps = self_round(
                state, rng, Xtr, ytr, tau_sys, KAPPA0)
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        V_before = (flags_and_probs(Xte, state, KAPPA0)[0] >= TAU0).reshape(-1)
        acc_before = float(np.mean(
            flags_and_probs(Xte, state, KAPPA0)[1] == yte))
        damage_trunk(state, mom, rng)
        pbar, ybar, conf, dis, flag, Ps = flags_and_probs(Xte, state, KAPPA0)
        V_after = (pbar >= TAU0).reshape(-1)
        ce, cp = head_stats(flags_and_probs(Xtr, state, KAPPA0)[5])
        fz.append({"d_raw": float(np.mean(V_after != V_before)),
                   "acc_before": acc_before,
                   "acc_after": float(np.mean(ybar == yte)),
                   "core_est_after": ce, "core_pair_after": cp})
    res["frozen_trunk_kill"] = fz
    print(f"FROZT_KILL raw trunk damage: d={np.mean([f['d_raw'] for f in fz])*100:.3f}% "
          f"acc {np.mean([f['acc_before'] for f in fz])*100:.2f}% -> "
          f"{np.mean([f['acc_after'] for f in fz])*100:.2f}% "
          f"core_est={np.mean([f['core_est_after'] for f in fz]):.1f}")

    # ---- bitwise audits (full chassis only) ----
    if not smoke:
        try:
            d23 = json.load(open(DOC23))
            d26 = json.load(open(DOC26))
            audits = {}
            for arm, (doc, ref_arm) in {
                    "TWIN": (d23, "TWIN"),
                    "S1_BRAKE": (d23, "HEAD_KILL"),
                    "S2_BRAKE": (d26, "TWOHEAD"),
                    "S2_NOBRAKE": (d26, "TWOHEAD_NOBRAKE")}.items():
                ref = doc["arms"][ref_arm]["aggregate"]["tau_sys"][:ROUNDS]
                got = [float(np.mean([r["traj"][i]["tau_sys"]
                                      for r in res["arms"][arm]["runs"]
                                      if r["traj"][i].get("tau_sys") is not None]))
                       for i in range(ROUNDS)]
                audits[arm] = bool(all(abs(a - b) < 1e-9
                                       for a, b in zip(ref, got)))
            res["bitwise_audits"] = audits
            print(f"bitwise audits: {audits}")
        except (FileNotFoundError, KeyError) as e:
            print(f"bitwise audits unavailable: {e}")

    # ---- classification (pre-registered rules) ----
    def agg_of(arm, key):
        return res["arms"][arm]["aggregate"][key]

    cls = {}
    for arm in ARMS:
        if arm == "TWIN":
            continue
        ag = res["arms"][arm]["aggregate"]
        d = ag["d_own"]
        final = d[-1]
        entry = {
            "final_d_own_pp": final * 100,
            "final_d_factory_pp": ag["d_factory"][-1] * 100,
            "final_tau_sys": ag["tau_sys"][-1],
            "acc_final": ag["acc"][-1],
            "acc_r_pert": ag["acc"][PERT_ROUND - 1],
            "commit_final": ag["n_commit"][-1],
            "commit_err_final": ag["commit_err"][-1],
            "core_est_r_pert": ag["core_est"][PERT_ROUND - 1],
            "standdown_frac": float(np.mean(
                [x for x in ag["standdown"][PERT_ROUND - 1:]
                 if x is not None])),
            "repair_rounds_post": float(np.sum(
                [x for x in ag["repair_fired"][PERT_ROUND - 1:]
                 if x is not None]))}
        if arm == "S2_CUTOFF":
            entry["cutoff_stasis"] = bool(ag["acc"][-1] >= 0.965)
            entry["cutoff_norm_breaking"] = bool(ag["tau_sys"][-1] < 0.80)
        if arm == "S2_SUBSTRATE":
            first_sd = next((i for i, x in enumerate(ag["standdown"])
                             if (x or 0) > 0.5), None)
            entry["probe_fired_round"] = (first_sd + 1) if first_sd else None
            entry["probe_caught"] = bool(first_sd is not None
                                         and PERT_ROUND < first_sd + 1 <= ROUNDS)
            entry["lands_between"] = bool(
                0.93 <= ag["acc"][-1] <= 0.965) if first_sd else False
        if arm in ("S2_BRAKE", "S2_NOBRAKE", "S1_BRAKE"):
            entry["acc_still_falling"] = bool(
                ag["acc"][-1] < ag["acc"][-2] < ag["acc"][-3])
        if arm.startswith("TRUNK"):
            raw_acc = float(np.mean([f["acc_after"] for f in fz]))
            entry["acc_raw_ref"] = raw_acc
            entry["verdict"] = ("RECOVERY" if ag["acc"][-1] >= raw_acc + 0.05
                                else "DECAY" if ag["acc"][-1] < raw_acc - 0.05
                                else "HANG")
            entry["core_est_min_post"] = float(np.min(
                [x for x in ag["core_est"][PERT_ROUND - 1:]
                 if x is not None]))
            entry["cutoff_would_stood_down"] = bool(
                entry["core_est_min_post"] <= 2)
        cls[arm] = entry
    res["classification"] = cls
    for arm, c in cls.items():
        print(f"{arm:18s} tau={c['final_tau_sys']:.3f} "
              f"acc={c['acc_final']*100:.1f}% "
              f"d_own={c['final_d_own_pp']:.2f}pp "
              f"sd={c['standdown_frac']:.2f} "
              f"core={c['core_est_r_pert']:.1f}")

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m3_substrate_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_substrate_results.json")


if __name__ == "__main__":
    main()
