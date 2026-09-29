#!/usr/bin/env python3
"""
THE DECAY ARM - testing the 0.897 threshold's other side. Corpus doc 29's
experiment. Ordered at the kappa/two-head issuance: "the decay arm (q_bar
falling through 0.897) to test the band-break formula's other side".

WHERE DOC 26 LEFT IT: the saturated-evaluator counterfactual landed a closed
form - with the quantile frozen at q_bar, the drift attack's EMA fixed point
is q_bar - d(1-eta)/eta and the band-break condition is
q_bar < tau* - eps + d(1-eta)/eta = 0.897.
Doc 26's own threat (iv): "a decayed-evaluator arm (q_bar falling through
0.897) would test the bound's other side and was not run."

THE REFINEMENT (derived here, pre-registered before the run): doc 26's
"band-break" condition is exactly the boundary where the SILENT-repair fixed
point exits the band. Below it the repair channel ENGAGES, and the dynamics
become piecewise. The repair-engaged fixed point (drift d, repair rho,
calibration eta, record tau*) is
    fp2 = [(1-eta)(rho*tau* - d(1-rho)) + eta*q_bar] / [eta + rho - eta*rho]
and the TRUE band break is fp2 < tau* - eps, i.e.
    q_bar < tau* - eps - [eta*rho - ... ]  (exact form below) = 0.815
at this registration (eta=0.3, rho=0.5, d=0.02, tau*=0.9, eps=0.05).
Three regimes, two thresholds:
    q_bar > 0.897   silent containment (doc 26's world: fp in band, repair
                    never fires, the attack is invisible)
    0.815 < q_bar < 0.897   VISIBLE containment (repair fires most rounds,
                    the norm hugs the lower band edge, band HOLDS - doc 26's
                    literal reading "break below 0.897" is refuted here)
    q_bar < 0.815   the break (fp2 out of band; the norm sits below tau*-eps
                    with repair firing EVERY round and losing)
And inside the middle regime a one-drift-marginal window where the ATTACK
alone flips the verdict: fp2_nodrift > tau*-eps > fp2_drift iff
    0.792 < q_bar < 0.815
(the marginal is d(1-eta)(1-rho)/(eta+rho-eta*rho) = 0.0108).

OPERATIONALIZATION (disclosed): the evaluator is frozen from round 6 (the
FLAT chassis of doc 26) and ROTTED by logit tempering z -> beta*z applied to
ALL evaluative reads (pool grading, view ensembles, test battery) from round
6 on. beta is calibrated per seed at round 6, by bisection on that round's
pool logits (deterministic, no RNG), so the realized pool quantile
Q_{1-r*}(conf) lands on the target level. Decay is thus a LEVEL (the rotten
world's stationary confidence), not a trajectory - the same modeling move as
doc 26's saturation, with the dial turned past it. The committee's
discriminative structure is frozen; the rot is confidence-scale rot.
ROUNDS is extended 10 -> 16 (disclosed deviation: 11 post-perturbation
applications leave only 0.7^11 = 2.1% of the initial gap - the fixed
points are read at convergence).

ARMS (tempering starts round 6, N=6 seeds, same-seed deterministic twins):
  HEALTHY_FLAT   no tempering, no drift  = doc 26's FLAT_TWIN, extended
                 (bitwise audit r1-10 against m3_kappa_results.json)
  DEC93_TWIN     q_bar -> 0.93   (above 0.897: the silent side)
  DEC93_DRIFT    + tau_sys -= 0.02 every round >= 6
  DEC87_TWIN     q_bar -> 0.87   (just below 0.897: first threshold crossed)
  DEC87_DRIFT    + drift
  DEC80_TWIN     q_bar -> 0.80   (inside the attack-marginal window)
  DEC80_DRIFT    + drift
  DEC72_TWIN     q_bar -> 0.72   (below 0.815: the second threshold crossed)
  DEC72_DRIFT    + drift

PRE-REGISTERED PREDICTIONS (fixed before execution):
  D1  HEALTHY_FLAT reproduces doc 26's FLAT_TWIN bitwise (r1-10 aggregate
      tau, tol 1e-9); DEC twins reproduce it r1-5 (their stream shares the
      pre-tempering path).
  D2  DEC93: DRIFT settles at fp = q_bar - d(1-eta)/eta = 0.883, repair
      SILENT from r7, band holds; TWIN at ~0.93, silent. Doc 26's closed
      form extends to the decayed world unmodified.
  D3  DEC87_DRIFT: repair ENGAGES (fires >= 5 of rounds 7-16), final tau in
      [0.84, 0.88], band HOLDS (final >= 0.85). The literal doc-26 reading
      (break below 0.897) REFUTED in the window; visible containment.
  D4  DEC80: TWIN holds (final in [0.845, 0.87], repair carries it); DRIFT
      BREAKS (final < 0.85, predicted 0.843, repair fires every round).
      Same world, one drift marginal apart.
  D5  DEC72_TWIN BREAKS with no attack at all (final ~0.818, repair firing);
      DEC72_DRIFT deep break (~0.806). Attribution inverts: below the
      second threshold the breaker is the evaluator, not the attacker.
  D6  All arms track the piecewise scalar recursion (simulator run on the
      realized q_bar of the twin) to <= 0.005 final; firing patterns match.
  D7  Rulers: the argmax ruler is nearly temper-invariant (|acc - healthy|
      <= 0.5pp at all levels); the V ruler is not (d_rot grows with depth).
      Rot is argmax-invisible and V-visible.
  D8  Deep levels: flag_frac -> ~1, brake fires, commits -> ~0 (the
      machinery reads the rot; stasis is already the state).
  D9  The exposure formula is two-sided: a registration can compute BOTH
      thresholds from (tau*, eps, d, eta, rho) without running the attack.

Usage: python3 m3_decay.py [--smoke]
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
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 16, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
DRIFT_T = 0.02
LEVELS = [0.93, 0.87, 0.80, 0.72]
DOC26 = "/home/z/my-project/scripts/m3_kappa_results.json"


def arm_names():
    out = ["HEALTHY_FLAT"]
    for lv in LEVELS:
        tag = "DEC%d" % round(lv * 100)
        out += [tag + "_TWIN", tag + "_DRIFT"]
    return out


ARMS = arm_names()


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


# ---------------- the round (FLAT chassis + tempering + optional drift) ----
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
    """One seed under one arm. RNG stream identical to doc 26's FLAT arms
    through round 5; tempering starts at round 6 and draws no RNG."""
    if arm == "HEALTHY_FLAT":
        target, drift = None, False
    else:
        tag, kind = arm.split("_")
        target = int(tag[3:]) / 100.0
        drift = (kind == "DRIFT")
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- self-registration at burn-in end (same pool0 draw as r*) ----
    pool0 = perturb(Xtr, rng)
    _, _, conf0, _, _, _ = flags_and_probs_beta(pool0, state, KAPPA0, 1.0)
    r_star = float(np.mean(conf0 >= TAU0))
    tau_star, kappa_star = TAU0, KAPPA0

    tau_sys, kappa_sys = TAU0, KAPPA0
    beta = 1.0
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. external perturbation (drift; no RNG) ----
        if r >= PERT_ROUND and drift:
            tau_sys -= DRIFT_T
        # ---- 2. out-of-gate repair channel ----
        repair_t = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        # ---- 3. pool draw (the round's first RNG use) ----
        pool = perturb(Xtr, rng)
        # ---- 3b. tempering calibration at the perturbation round ----
        if r == PERT_ROUND and target is not None:
            beta = calibrate_beta(state, pool, r_star, target)
        # ---- 3c. grading (brake, views, commits) ----
        Xt_, yt_, tele = grade_round(state, rng, pool, tau_sys, kappa_sys, beta)
        # ---- 4. FLAT chassis: no self-training from the perturbation round --
        if r < PERT_ROUND:
            train_epochs(state, mom, rng, Xt_, yt_, SELF_EPOCHS)
        # ---- 5. calibration write channel (tempered read) ----
        _, _, conf_p, _, _, _ = flags_and_probs_beta(pool, state, kappa_sys, beta)
        q = float(np.quantile(conf_p, 1.0 - r_star))
        tau_sys = (1 - ETA) * tau_sys + ETA * q
        # ---- 6. telemetry (tempered read: the ruler reads the rotten
        #      evaluator as it is) ----
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs_beta(
            Xte, state, kappa_sys, beta)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": kappa_sys, "kappa_star": kappa_star,
            "q_target": q, "beta": float(beta),
            "repair_fired": repair_t, "r_star": r_star,
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


# ---------------- the piecewise scalar simulator (the audit's ruler) -------
def simulate_scalar(q_bar, drift, tau_start, rounds=ROUNDS, pert=PERT_ROUND,
                    eta=ETA, eps=EPS_T, rho=RHO_T, d=DRIFT_T, tau_star=TAU0):
    """The exact scalar recursion: drift -> repair -> calibration, with the
    repair channel firing only outside the deadband (the piecewise map)."""
    tau = tau_start
    out = []
    for r in range(pert, rounds + 1):
        if drift:
            tau -= d
        fired = abs(tau - tau_star) > eps
        if fired:
            tau += rho * (tau_star - tau)
        tau = (1 - eta) * tau + eta * q_bar
        out.append({"round": r, "tau": tau, "fired": fired})
    return out


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 8
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        globals()["LEVELS"] = [0.93, 0.72]
        globals()["ARMS"] = arm_names()
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} tempering at round "
          f"{PERT_ROUND}, rounds={ROUNDS}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eta": ETA, "brake_th": BRAKE_TH,
                      "rounds": ROUNDS, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS), "arms": ARMS, "levels": LEVELS,
                      "drift_t": DRIFT_T, "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: the twins (HEALTHY_FLAT + the four DEC twins) ----
    twins = {}
    twin_arms = ["HEALTHY_FLAT"] + ["DEC%d_TWIN" % round(l * 100)
                                    for l in LEVELS]
    for tw in twin_arms:
        rows = []
        for s in SEEDS:
            rows.append(run_arm(s, tw, Xtr, ytr, Xte, yte))
            print(f"  {tw} seed={s} done ({time.time()-t0:.0f}s)")
        twins[tw] = rows

    # doc-26 bitwise reproduction audit (HEALTHY_FLAT vs FLAT_TWIN r1-10;
    # DEC twins vs FLAT_TWIN r1-5) - full chassis only (smoke shifts the
    # perturbation round, so the audit is skipped there)
    if not smoke:
        try:
            d26 = json.load(open(DOC26))
            ref = d26["arms"]["FLAT_TWIN"]["aggregate"]["tau_sys"][:10]
            got = [float(np.mean([rr[i]["tau_sys"] for rr in twins["HEALTHY_FLAT"]]))
                   for i in range(min(10, ROUNDS))]
            ok_full = all(abs(a - b) < 1e-9 for a, b in zip(ref, got))
            res["healthy_reproduces_doc26"] = bool(ok_full)
            print(f"HEALTHY_FLAT reproduces doc 26 FLAT_TWIN (r1-10): {ok_full}")
            pre_ok = {}
            for lv in LEVELS:
                tw = "DEC%d_TWIN" % round(lv * 100)
                got5 = [float(np.mean([rr[i]["tau_sys"] for rr in twins[tw]]))
                        for i in range(min(5, ROUNDS))]
                pre_ok[tw] = bool(all(abs(a - b) < 1e-9
                                      for a, b in zip(ref[:5], got5)))
            res["dec_twins_reproduce_doc26_r1_5"] = pre_ok
            print(f"DEC twins reproduce doc 26 r1-5: {pre_ok}")
        except (FileNotFoundError, KeyError) as e:
            res["healthy_reproduces_doc26"] = f"unavailable: {e}"
            print(f"doc-26 audit unavailable: {e}")

    def twin_of(arm):
        if arm == "HEALTHY_FLAT":
            return "HEALTHY_FLAT"
        tag, _ = arm.split("_")
        return tag + "_TWIN"

    # ---- pass 2: the drift arms ----
    for arm in ARMS:
        if arm.endswith("_TWIN") or arm == "HEALTHY_FLAT":
            continue
        rows = []
        for si, s in enumerate(SEEDS):
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            tw = twins[twin_of(arm)][si]
            # factory reference: the burn-in-end state's untempered V
            rng_f = np.random.default_rng(s)
            state_f = init_params(rng_f)
            mom_f = zero_moments(state_f)
            train_epochs(state_f, mom_f, rng_f, Xtr, ytr, BURN_EPOCHS)
            pf, _, conff, _, _, _ = flags_and_probs_beta(
                Xte, state_f, KAPPA0, 1.0)
            V_fac = (pf >= TAU0).reshape(-1)
            # rot footprint: vs the healthy flat twin's V (same round)
            hf = twins["HEALTHY_FLAT"][si]
            d_rot = [float(np.mean(t["V"] != h["V"]))
                     for t, h in zip(traj, hf)]
            d_own = [float(np.mean(t["V"] != w["V"]))
                     for t, w in zip(traj, tw)]
            d_fac = [float(np.mean(t["V"] != V_fac)) for t in traj]
            for t, do, df, dr in zip(traj, d_own, d_fac, d_rot):
                t["d_own"], t["d_factory"], t["d_rot"] = do, df, dr
                del t["V"]
            rows.append({"seed": s, "traj": traj})
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "q_target",
                    "accept_own_tau", "d_own", "d_factory", "d_rot", "beta"]:
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
        print(f"{arm:16s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':16s} q  : {['%.3f' % a for a in agg['q_target']]}")
        print(f"{'':16s} rep: {['%.1f' % a for a in agg['repair_fired']]}")

    # ---- twins' aggregates + rot footprint vs HEALTHY_FLAT ----
    for tw in twin_arms:
        trows = twins[tw]
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "q_target",
                    "accept_own_tau", "beta"]:
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
        # rot footprint for the DEC twins themselves (vs healthy flat)
        if tw != "HEALTHY_FLAT":
            d_rot = []
            for i in range(len(trows[0])):
                v = [float(np.mean(a[i]["V"] != b[i]["V"]))
                     for a, b in zip(trows, twins["HEALTHY_FLAT"])]
                d_rot.append(float(np.mean(v)))
            agg["d_rot"] = d_rot
        res["arms"][tw] = {"aggregate": agg,
                           "runs": [{"seed": s, "traj": [
                               {k: v for k, v in t.items() if k != "V"}
                               for t in trows[si]]}
                              for si, s in enumerate(SEEDS)]}
        print(f"{tw:16s} tau: {['%.3f' % a for a in agg['tau_sys']]} "
              f"q: {['%.3f' % a for a in agg['q_target']]}")

    # ---- classification + simulator audit (pre-registered rules) ----
    cls = {}

    def agg_of(arm, key):
        return res["arms"][arm]["aggregate"][key]

    band_lo = TAU0 - EPS_T
    off = DRIFT_T * (1 - ETA) / ETA                 # d(1-eta)/eta
    denom = ETA + RHO_T - ETA * RHO_T               # eta + rho - eta*rho
    th1 = TAU0 - EPS_T + off                        # 0.897 (doc 26)
    # exact second threshold: fp2(q) = [(1-e)(rho*tau* - d(1-rho)) + e q]/denom
    # fp2 = tau* - eps  <=>  q = (tau*-eps)*denom - (1-e)(rho*tau* - d(1-rho))
    th2 = ((TAU0 - EPS_T) * denom - (1 - ETA) * (RHO_T * TAU0
                                                 - DRIFT_T * (1 - RHO_T))) / ETA
    th2_nodrift = ((TAU0 - EPS_T) * denom - (1 - ETA) * RHO_T * TAU0) / ETA
    cls["_thresholds"] = {
        "doc26_band_break_q": th1,
        "repair_engaged_fp2_break_q_drift": th2,
        "repair_engaged_fp2_break_q_nodrift": th2_nodrift,
        "attack_marginal": DRIFT_T * (1 - ETA) * (1 - RHO_T) / denom}

    for lv in LEVELS:
        tag = "DEC%d" % round(lv * 100)
        for kind in ["TWIN", "DRIFT"]:
            arm = f"{tag}_{kind}"
            ag = res["arms"][arm]["aggregate"]
            post = slice(PERT_ROUND - 1, ROUNDS)
            rep_post = [x for x in ag["repair_fired"][PERT_ROUND:]
                        if x is not None]
            final = ag["tau_sys"][-1]
            qbar = float(np.mean([x for x in ag["q_target"][PERT_ROUND:]
                                  if x is not None]))
            entry = {
                "target_level": lv, "q_bar_realized": qbar,
                "final_tau": final, "min_tau": float(np.min(
                    ag["tau_sys"][PERT_ROUND - 1:])),
                "repair_rounds_post": float(np.sum(rep_post)),
                "band_broken": bool(final < band_lo),
                "acc_final": ag["acc"][-1],
                "d_rot_final_pp": (ag.get("d_rot", [None])[-1] or 0.0) * 100
                if ag.get("d_rot") else None}
            if kind == "DRIFT":
                twf = res["arms"][tag + "_TWIN"]["aggregate"]["tau_sys"][-1]
                entry["twin_final_tau"] = twf
                entry["attack_marginal_pp"] = (final - twf) * 100
            # simulator audit: piecewise recursion on the realized q_bar
            tau_start = ag["tau_sys"][PERT_ROUND - 2]
            sim = simulate_scalar(qbar, drift=(kind == "DRIFT"),
                                  tau_start=tau_start, rounds=ROUNDS,
                                  pert=PERT_ROUND)
            entry["sim_final_tau"] = sim[-1]["tau"]
            entry["sim_error"] = final - sim[-1]["tau"]
            entry["sim_repair_rounds"] = int(sum(1 for s in sim if s["fired"]))
            cls[arm] = entry
    # HEALTHY_FLAT entry
    ag = res["arms"]["HEALTHY_FLAT"]["aggregate"]
    cls["HEALTHY_FLAT"] = {
        "final_tau": ag["tau_sys"][-1],
        "q_bar_realized": float(np.mean(
            [x for x in ag["q_target"][PERT_ROUND:] if x is not None])),
        "repair_rounds_post": float(np.sum(
            [x for x in ag["repair_fired"][PERT_ROUND:]
             if x is not None])),
        "band_broken": bool(ag["tau_sys"][-1] < band_lo),
        "acc_final": ag["acc"][-1]}
    res["classification"] = cls

    print("\n--- classification (band low edge %.3f; th1=%.4f th2=%.4f "
          "th2_nodrift=%.4f) ---" % (band_lo, th1, th2, th2_nodrift))
    for arm, c in cls.items():
        if arm.startswith("_"):
            continue
        if "final_tau" in c:
            print(f"{arm:16s} qbar={c.get('q_bar_realized', 0):.4f} "
                  f"final={c['final_tau']:.4f} "
                  f"rep={c.get('repair_rounds_post', 0):.1f} "
                  f"broken={c.get('band_broken')} "
                  f"sim_err={c.get('sim_error', 0):+.4f}")

    res["runtime_s"] = time.time() - t0
    out_path = "/home/z/my-project/scripts/m3_decay_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_decay_results.json")


if __name__ == "__main__":
    main()
