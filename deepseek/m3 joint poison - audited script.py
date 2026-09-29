#!/usr/bin/env python3
"""
THE JOINT-POISON ARM - both records corrupted at once. Corpus doc 30's
experiment. Ordered at the kappa/two-head issuance: "the joint-poison arm
(tau* AND kappa* corrupted - doc 28's F3 refinement)".

WHERE DOC 28 LEFT IT (Part 3, F3 - the negotiation fake): "poison each
coordinate separately, then jointly; the joint case is where a fake
negotiation most likely unravels, and it is untested." Doc 23 contained the
single-coordinate poison (TARGET_POISON: tau* <- 0.70, contained at the
tension point 0.836, d_own 0.37%); doc 26 contained the joint KICK with
honest records (TAU_KAP_JOINT: restored on both coordinates). The joint
POISON - both records lying at once - is the untested cell.

THE CLOSED FORM (derived here, validated on doc 23's existing data before
this run): under a poisoned record the repair channel fires every round
(the landing sits far from the record) and the tension point is the
repair-engaged fixed point of the piecewise map
    tau <- (1-eta)(tau + rho(tau*_p - tau)) + eta q:
    T(tau*_p) = [(1-eta) rho tau*_p + eta q_bar] / (eta + rho - eta rho)
With tau*_p = 0.70, q_bar = 0.9925 (doc 23's realized): T = 0.8350 vs the
observed 0.8357 - error 7e-4. Doc 23's r6 value 0.8043 is also exact once
one notes the repair fires AT the deadband edge (0.75 - 0.70 =
0.050000000000000044 > 0.05 in IEEE-754): tau(r6) = 0.7*(0.75-0.025) +
0.3*0.9894 = 0.8043. The kappa analogue, with kappa*_p = 0.75 and the live
flag-quantile k_bar ~ 0.994 (doc 26's KAP_TWIN): T_kappa = 0.863.

THE CONSPIRACY MECHANISM (why the joint case might not factorize): the tau
poison parks tau_sys at ~0.83 (vs the twin's 0.95) - the commit gate WIDENS
(accept_own_tau jumps); the kappa poison parks kappa_sys at ~0.86 (vs the
twin's 0.99) - the flag rule LOOSENS (fewer items flagged, fewer view
re-grades, brake harder to arm). Each poison is contained alone because the
live channel checks its own record; jointly, the widened gate admits dirtier
self-labels while the loosened flag suppresses the machinery that would
catch them - the live evaluator q sags, and BOTH tension points sag with it
(the tension formula's q_bar is endogenous to the joint world). That is the
pre-registered unraveling path; the fork is the experiment.

ARMS (perturbation at round 6, N=6 seeds, same-seed deterministic twins,
full training chassis = doc 26's):
  TWIN             kappa frozen, unperturbed (doc 23 bitwise audit)
  KAP_TWIN         kappa-state, unperturbed (doc 26 bitwise audit)
  JP_TAU           kappa frozen: tau* <- 0.70, tau_sys <- 0.75 at r6
                   (= doc 23's TARGET_POISON - bitwise audit)
  JP_TAU_KSTATE    kappa-state, same tau poison (kappa* honest): does the
                   live second coordinate rescue the first?
  JP_KAP           kappa-state: kappa* <- 0.75, kappa_sys <- 0.80 at r6
                   (tau honest, no tau kick)
  JP_JOINT         kappa-state: tau* <- 0.70, kappa* <- 0.75,
                   tau_sys <- 0.75, kappa_sys <- 0.80 (both records lie,
                   both coordinates kicked - differs from doc 26's
                   TAU_KAP_JOINT only in record content)
  JP_JOINT_NOREP   joint poison, BOTH repair channels dis-armed from r6
                   (the F3 probe: is the joint landing repair-carried?)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  J1  TWIN reproduces doc 23 bitwise; KAP_TWIN reproduces doc 26 bitwise
      (tau and kappa); JP_TAU reproduces doc 23's TARGET_POISON bitwise.
  J2  JP_TAU lands at the tension point 0.835 +/- 0.002 (bitwise makes it
      exact); NOT at the poison (poisoned_restore = False, doc 23's
      criterion |final - 0.70| <= 0.03 refuted again).
  J3  JP_KAP lands at kappa = 0.863 +/- 0.010 (the kappa tension point;
      wider band - the flag-quantile k_bar fluctuates more than q_bar);
      flag_frac <= KAP_TWIN's (looser threshold flags less); brake silent.
  J4  THE FORK, registered as competing predictions:
      (a) FACTORIZED: JP_JOINT lands at (T_tau, T_kappa) evaluated at its
          own realized (q_bar, k_bar): |final_tau - T_tau| <= 0.005,
          |final_kappa - T_kappa| <= 0.01, d_own <= 0.8%, acc >= 96%,
          q_bar sag vs KAP_TWIN <= 0.01. The two negotiations do not talk.
      (b) CONSPIRACY: q_bar sags > 0.01 under the widened commits (commit
          label error rises visibly), the tension points sag with it
          (Delta T = eta*Delta_q/(eta+rho-eta*rho) ~ 0.46 Delta_q), and/or
          d_own > 2%, acc < 94%. The joint poison is more than the sum of
          its parts - F3's fake negotiation unravels exactly when both
          records lie at once.
      (c) a third outcome, classified post hoc with disclosed criteria.
  J5  JP_JOINT_NOREP ESCAPES to the live courses: final tau within 0.02 of
      KAP_TWIN's, final kappa within 0.02 - the records' only channel into
      the norm is the repair pull; disarm both and the norm follows its
      live evaluation past the poisons. The joint landing is
      REPAIR-CARRIED (the record side of the negotiation), the live side
      needs no channel at all.
  J6  JP_TAU_KSTATE: tau landing within 0.01 of JP_TAU's (the second
      coordinate does not rescue the first - containment is the poisoned
      coordinate's own live channel at work). ALTERNATIVE: rescue >= +0.02
      (final tau >= 0.855) via kappa-state's flag response to the widened
      commits.
  J7  commit label error (audit-side): TWIN/KAP_TWIN <= 3%; JP_TAU ~ doc
      23's level; JP_JOINT's is the conspiracy's leading indicator
      (registered: if > 2x KAP_TWIN's, the (b) branch is live regardless
      of where the norm lands).

Usage: python3 m3_jointpoison.py [--smoke]
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
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 10, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
POISON_TAU, POISON_KAP = 0.70, 0.75    # record poisons (-0.20 each)
KICK_TAU, KICK_KAP = 0.75, 0.80        # the joint kicks (doc 26's values)
ARMS = ["TWIN", "KAP_TWIN", "JP_TAU", "JP_TAU_KSTATE", "JP_KAP",
        "JP_JOINT", "JP_JOINT_NOREP"]
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


# ---------------- the round (doc-26 chassis + commit_err telemetry) --------
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
    # audit-side only (no RNG, not system-visible): committed label error
    tele["commit_err"] = float(np.mean(labels[commit] != ytr[commit])) \
        if commit.any() else 0.0
    return pool[commit], labels[commit], tele, pool


def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. RNG stream for kappa-frozen arms is
    byte-identical to m3_proper.py's; for kappa-state arms, to
    m3_kappa_twohead.py's (the kappa machinery draws no randomness)."""
    kstate = arm in ("KAP_TWIN", "JP_TAU_KSTATE", "JP_KAP", "JP_JOINT",
                     "JP_JOINT_NOREP")
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # ---- self-registration at burn-in end (same pool0 draw for r* and f*) --
    pool0 = perturb(Xtr, rng)
    _, _, conf0, dis0, flag0, _ = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. external perturbation (record poisons; no RNG) ----
        if r == PERT_ROUND:
            if arm in ("JP_TAU", "JP_TAU_KSTATE"):
                tau_star = POISON_TAU
                tau_sys = KICK_TAU
            elif arm == "JP_KAP":
                kappa_star = POISON_KAP
                kappa_sys = KICK_KAP
            elif arm in ("JP_JOINT", "JP_JOINT_NOREP"):
                tau_star, kappa_star = POISON_TAU, POISON_KAP
                tau_sys, kappa_sys = KICK_TAU, KICK_KAP
        # ---- 2. out-of-gate repair channels (dis-armable) ----
        no_rep = (arm == "JP_JOINT_NOREP" and r >= PERT_ROUND)
        repair_t = repair_k = False
        if not no_rep and abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        if kstate and not no_rep and abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        # ---- 3+4. grading, brake, training ----
        Xt_, yt_, tele, pool = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
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
                kap_t = KAP_FLOOR
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


def tension_point(poison, live, eta=ETA, rho=RHO_T):
    """Repair-engaged fixed point: [(1-e) rho*poison + e*live]/(e+rho-e*rho)."""
    return ((1 - eta) * rho * poison + eta * live) / (eta + rho - eta * rho)


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 6
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} poison at round {PERT_ROUND}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "brake_th": BRAKE_TH,
                      "rounds": ROUNDS, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS), "arms": ARMS,
                      "poison_tau": POISON_TAU, "poison_kap": POISON_KAP,
                      "kick_tau": KICK_TAU, "kick_kap": KICK_KAP,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: twins ----
    twins = {}
    for tw in ["TWIN", "KAP_TWIN"]:
        rows = []
        for s in SEEDS:
            rows.append(run_arm(s, tw, Xtr, ytr, Xte, yte))
            print(f"  {tw} seed={s} done ({time.time()-t0:.0f}s)")
        twins[tw] = rows

    # bitwise reproduction audits (full chassis only)
    if not smoke:
        try:
            d23 = json.load(open(DOC23))
            ref = d23["arms"]["TWIN"]["aggregate"]["tau_sys"][:ROUNDS]
            got = [float(np.mean([rr[i]["tau_sys"] for rr in twins["TWIN"]]))
                   for i in range(ROUNDS)]
            res["twin_reproduces_doc23"] = bool(all(
                abs(a - b) < 1e-9 for a, b in zip(ref, got)))
            print(f"TWIN reproduces doc 23 bitwise: "
                  f"{res['twin_reproduces_doc23']}")
            d26 = json.load(open(DOC26))
            ref26t = d26["arms"]["KAP_TWIN"]["aggregate"]["tau_sys"][:ROUNDS]
            ref26k = d26["arms"]["KAP_TWIN"]["aggregate"]["kappa_sys"][:ROUNDS]
            gott = [float(np.mean([rr[i]["tau_sys"] for rr in twins["KAP_TWIN"]]))
                    for i in range(ROUNDS)]
            gotk = [float(np.mean([rr[i]["kappa_sys"] for rr in twins["KAP_TWIN"]]))
                    for i in range(ROUNDS)]
            res["kaptwin_reproduces_doc26"] = bool(
                all(abs(a - b) < 1e-9 for a, b in zip(ref26t, gott)) and
                all(abs(a - b) < 1e-9 for a, b in zip(ref26k, gotk)))
            print(f"KAP_TWIN reproduces doc 26 bitwise: "
                  f"{res['kaptwin_reproduces_doc26']}")
        except (FileNotFoundError, KeyError) as e:
            print(f"twin audits unavailable: {e}")

    def twin_of(arm):
        return "TWIN" if arm in ("TWIN", "JP_TAU") else "KAP_TWIN"

    # ---- pass 2: poison arms ----
    bitwise_jptau = None
    for arm in ARMS:
        if arm in ("TWIN", "KAP_TWIN"):
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
                    "d_factory", "d_twin_kfrozen", "commit_err",
                    "repair_fired"]:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
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
        print(f"{'':16s} cerr: {['%.3f' % a for a in agg['commit_err']]}")

    # doc-23 TARGET_POISON bitwise audit for JP_TAU
    if not smoke:
        try:
            d23 = json.load(open(DOC23))
            ref = d23["arms"]["TARGET_POISON"]["aggregate"]["tau_sys"][:ROUNDS]
            got = [float(np.mean([r["traj"][i]["tau_sys"]
                                  for r in res["arms"]["JP_TAU"]["runs"]]))
                   for i in range(ROUNDS)]
            bitwise_jptau = bool(all(abs(a - b) < 1e-9
                                     for a, b in zip(ref, got)))
            res["jptau_reproduces_doc23_targetpoison"] = bitwise_jptau
            print(f"JP_TAU reproduces doc 23 TARGET_POISON bitwise: "
                  f"{bitwise_jptau}")
        except (FileNotFoundError, KeyError) as e:
            print(f"JP_TAU audit unavailable: {e}")

    # ---- twins' aggregates ----
    for tw in ["TWIN", "KAP_TWIN"]:
        trows = twins[tw]
        agg = {}
        for key in ["acc", "accept09", "n_commit", "flagged_frac",
                    "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                    "q_target", "kappa_target", "accept_own_tau",
                    "commit_err"]:
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
        res["arms"][tw] = {"aggregate": agg,
                           "runs": [{"seed": s, "traj": [
                               {k: v for k, v in t.items() if k != "V"}
                               for t in trows[si]]}
                              for si, s in enumerate(SEEDS)]}
        print(f"{tw:16s} tau: {['%.3f' % a for a in agg['tau_sys']]} "
              f"kap: {['%.3f' % a for a in agg['kappa_sys']]}")

    # ---- classification (pre-registered rules) ----
    def agg_of(arm, key):
        return res["arms"][arm]["aggregate"][key]

    kap_tw_f = agg_of("KAP_TWIN", "kappa_sys")[-1]
    tau_tw_f = agg_of("TWIN", "tau_sys")[-1]
    kap_tw_q = float(np.mean([x for x in agg_of("KAP_TWIN", "q_target")
                              if x is not None][PERT_ROUND - 1:]))
    kap_tw_k = float(np.mean([x for x in agg_of("KAP_TWIN", "kappa_target")
                              if x is not None][PERT_ROUND - 1:]))
    cls = {"_live_references": {"kap_twin_final_tau": agg_of(
        "KAP_TWIN", "tau_sys")[-1], "kap_twin_final_kappa": kap_tw_f,
        "twin_final_tau": tau_tw_f, "q_bar_kaptwin": kap_tw_q,
        "k_bar_kaptwin": kap_tw_k}}
    for arm in ARMS:
        if arm in ("TWIN", "KAP_TWIN"):
            continue
        ag = res["arms"][arm]["aggregate"]
        d = ag["d_own"]
        jump = d[PERT_ROUND - 1] - d[PERT_ROUND - 2]
        final = d[-1]
        tsys, ksys = ag["tau_sys"][-1], ag["kappa_sys"][-1]
        qbar = float(np.mean([x for x in ag["q_target"][PERT_ROUND - 1:]
                              if x is not None]))
        entry = {
            "jump_pp": jump * 100, "final_d_own_pp": final * 100,
            "final_d_factory_pp": ag["d_factory"][-1] * 100,
            "final_tau_sys": tsys, "final_kappa_sys": ksys,
            "commit_final": ag["n_commit"][-1], "acc_final": ag["acc"][-1],
            "q_bar_realized": qbar,
            "commit_err_final": ag["commit_err"][-1],
            "repair_rounds": float(np.sum(
                [x for x in ag["repair_fired"][PERT_ROUND - 1:]
                 if x is not None]))}
        if arm.startswith("JP") and "KAP" not in arm.split("_")[-1]:
            pass
        if arm in ("JP_TAU", "JP_TAU_KSTATE", "JP_JOINT", "JP_JOINT_NOREP"):
            entry["tension_point_tau"] = tension_point(POISON_TAU, qbar)
            entry["tension_err_tau"] = tsys - entry["tension_point_tau"]
        if arm in ("JP_KAP", "JP_JOINT", "JP_JOINT_NOREP"):
            kbar = float(np.mean([x for x in ag["kappa_target"]
                                  [PERT_ROUND - 1:] if x is not None]))
            entry["k_bar_realized"] = kbar
            entry["tension_point_kappa"] = tension_point(POISON_KAP, kbar)
            entry["tension_err_kappa"] = ksys - entry["tension_point_kappa"]
            entry["repair_rounds_kappa"] = float(np.sum(
                [x for x in ag["repair_fired_kappa"][PERT_ROUND - 1:]
                 if x is not None]))
        if arm == "JP_TAU":
            entry["poisoned_restore"] = bool(abs(tsys - POISON_TAU) <= 0.03)
            entry["lands_at_tension"] = bool(abs(tsys - 0.835) <= 0.002)
            entry["reproduces_doc23"] = bitwise_jptau
        if arm == "JP_TAU_KSTATE":
            jp_tau_f = res["arms"]["JP_TAU"]["aggregate"]["tau_sys"][-1]
            entry["tau_gap_vs_jptau"] = tsys - jp_tau_f
            entry["rescue"] = bool(tsys - jp_tau_f >= 0.02)
        if arm == "JP_KAP":
            entry["lands_at_tension_kappa"] = bool(
                abs(ksys - 0.863) <= 0.010)
            entry["flag_frac_final"] = ag["flagged_frac"][-1]
        if arm in ("JP_JOINT", "JP_JOINT_NOREP"):
            entry["q_sag_vs_kaptwin"] = qbar - kap_tw_q
            entry["factorized"] = bool(
                abs(tsys - tension_point(POISON_TAU, qbar)) <= 0.005 and
                abs(ksys - tension_point(
                    POISON_KAP, float(np.mean(
                        [x for x in ag["kappa_target"][PERT_ROUND - 1:]
                         if x is not None])))) <= 0.01)
            entry["conspiracy"] = bool(
                final > 0.02 or ag["acc"][-1] < 0.94
                or (qbar - kap_tw_q) < -0.01)
        if arm == "JP_JOINT_NOREP":
            entry["escapes_to_live"] = bool(
                abs(tsys - agg_of("KAP_TWIN", "tau_sys")[-1]) <= 0.02 and
                abs(ksys - kap_tw_f) <= 0.02)
        cls[arm] = entry
    res["classification"] = cls
    for arm, c in cls.items():
        if arm.startswith("_"):
            continue
        print(f"{arm:16s} tau={c['final_tau_sys']:.3f} "
              f"kap={c['final_kappa_sys']:.3f} "
              f"d_own={c['final_d_own_pp']:.2f}pp acc={c['acc_final']*100:.1f}% "
              f"cerr={c['commit_err_final']*100:.1f}%")

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m3_jointpoison_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_jointpoison_results.json")


if __name__ == "__main__":
    main()
