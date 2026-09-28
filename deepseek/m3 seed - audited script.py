#!/usr/bin/env python3
"""
THE M3-DESIGN SEED - perturb tau / kappa / a head; restoration vs. reversion:
the occupancy question, first probe. Corpus doc 18's experiment.

Ladder M3: "S4 under adversarial perturbation. Occupancy (4b): the system
maintains its normativity against targeted attack on its evaluative
machinery." S4's definition: "the system acts to restore its own norms under
perturbation - including bounded perturbation of its evaluative machinery
itself... The discriminating case is perturbation the system ANSWERS rather
than reverts: self-revision with its own reasons, not restoration to factory
settings. The distinction between recovery and reversion is the distinction
between a norm the system has and a norm the system is executing."

THIS IS THE SEED, NOT M3 ITSELF: one build-generation probe that (a) stands
up the restoration/reversion measurement, (b) locates where M2-alpha stands
on it, and (c) extracts the design requirements a real M3 system must meet.
No occupancy claim is possible from this experiment, and none is made.

THE MEASUREMENT DESIGN (pre-registered):
  Base recipe: M2-alpha unchanged (burn-in 30 ep supervised; 10 label-free
  self-rounds; committee tau=0.9, kappa=0.95; 4 epochs/round). Perturbation
  hits at the START OF ROUND 6 (half-way through the self-phase; the system
  has 5 label-free rounds to answer). N=6 seeds per condition, each with its
  same-seed unperturbed TWIN (the counterfactual own-norm trajectory - the
  instrument's gift: full determinism makes the counterfactual exact).

  CONDITIONS (evaluative machinery only - data untouched):
    TAU_LO   tau 0.9 -> 0.75 (norm loosened; commit region widens)
    TAU_HI   tau 0.9 -> 0.97 (norm tightened; commit region narrows)
    KAP_LO   kappa 0.95 -> 0.50 (flag gate all but disabled)
    KAP_HI   kappa 0.95 -> 0.995 (flag gate hypersensitive)
    HEAD_SOFT  head 2's output layer halved (Wo *= 0.5, bo *= 0.5) at round 6
    HEAD_KILL  head 2 re-initialized from fresh random at round 6
  CONTROLS:
    SUP_KILL   HEAD_KILL damage on a purely-supervised continued system
               (true labels, same epochs) - the factory-repair reference:
               what ordinary designer-supervised error-correction does.
    FROZEN     damage-free reference: unperturbed twin itself (= TWIN arm).
    FROZ_KILL  HEAD_KILL applied at round 10 with NO further training -
               the raw-damage reference (quantifies the initial hit).

  COORDINATES measured each round r on the battery (plain committee at the
  ORIGINAL tau=0.9 - the audit-side probe, independent of any perturbed
  internal threshold):
    d_own(r)    verdict disagreement vs the same-seed unperturbed twin
    d_factory(r) verdict disagreement vs the burn-in state (factory)
    acc(r), accept(r) under tau=0.9, commit counts, flagged fraction
    head agree(r) fraction of heads agreeing with committee argmax
    (HEAD arms) damaged head's agreement + its accuracy vs committee labels

  CLASSIFICATION (pre-registered):
    RESTORE-OWN   d_own jumps at r=6, then decays >=50% by r=10, final
                  d_own < d_factory (returns toward ITS OWN trajectory)
    REVERT-FACTORY d_factory shrinks across rounds (drifts toward burn-in)
    ABSORB        d_own jumps and >80% of the jump persists at r=10
    DERAIL        d_own grows without bound relative to jump
  Prediction (horn-one, honest): TAU/KAP arms -> ABSORB or DERAIL (tau and
  kappa are recipe CONSTANTS, not system state: no mechanism represents
  them, so nothing can detect the perturbation, let alone answer it);
  HEAD arms -> the damaged head re-learns toward committee consensus because
  the (designer-inherited) loss form pulls it there - designer-machinery
  repair, the S4 cheap fake at component scale; SUP_KILL repairs faster and
  closer to factory, exposing the difference. The seed's expected yield: the
  located representational absence (occupancy fails before behavior starts)
  and the M3-proper design requirement (evaluative machinery as learnable
  system state).

Usage: python3 m3_seed.py [--smoke]
"""
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
VIEWS, VIEW_SEED = 6, 4242
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6                       # perturbation hits at start of round 6
NSEED = 6
COND_SEEDS = list(range(500, 500 + NSEED))
CONDITIONS = ["TWIN", "TAU_LO", "TAU_HI", "KAP_LO", "KAP_HI",
              "HEAD_SOFT", "HEAD_KILL", "SUP_KILL", "FROZ_KILL"]
DAMAGED_HEAD = 2


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


# ---------------- perturbation pool (identical to m2_build.py) ----------------
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
    return pbar, ybar, conf, flagged, Ps


def self_round(state, mom, rng, Xtr, tau, kappa):
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged, _ = flags_and_probs(pool, state, kappa)
    commit = np.zeros(len(pool), dtype=bool)
    labels = ybar.copy()
    if flagged.any():
        W1, b1, heads = state
        pview = np.zeros((flagged.sum(), 10))
        for _ in range(VIEWS):
            vp = perturb(pool[flagged], rng)
            pview += committee_probs(vp, W1, b1, heads)
        pview /= VIEWS
        confv = pview.max(1)
        okv = confv >= tau
        idx = np.where(flagged)[0]
        commit[idx[okv]] = True
        labels[idx[okv]] = pview[okv].argmax(1)
    commit[~flagged] = conf[~flagged] >= tau
    labels[~flagged] = ybar[~flagged]
    return pool, commit, labels, flagged


# ---------------- battery probes ----------------
def probe(state, Xte, tau=TAU, kappa=KAPPA):
    """Audit-side probe: plain committee at the ORIGINAL tau (0.9)."""
    pbar, ybar, conf, flagged, Ps = flags_and_probs(Xte, state, kappa)
    return pbar, ybar, conf, flagged, Ps


def verdicts(pbar, tau=TAU):
    return (pbar >= tau).reshape(-1)


def head_agreement(Ps, ybar):
    return float(np.mean([np.mean(P.argmax(1) == ybar) for P in Ps]))


# ---------------- the experiment ----------------
def run_cond(seed, cond, Xtr, ytr, Xte, yte):
    """Run one seed under one condition; returns per-round telemetry."""
    rng = np.random.default_rng(seed)
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)

    # factory reference: burn-in state's battery probs/verdicts
    p_fac, _, _, _, _ = probe(state, Xte)
    V_fac = verdicts(p_fac)

    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- perturbation hits at start of round PERT_ROUND ----
        if r == PERT_ROUND:
            if cond == "TAU_LO":  tau_p, kap_p = 0.75, KAPPA
            elif cond == "TAU_HI": tau_p, kap_p = 0.97, KAPPA
            elif cond == "KAP_LO": tau_p, kap_p = TAU, 0.50
            elif cond == "KAP_HI": tau_p, kap_p = TAU, 0.995
            elif cond == "HEAD_SOFT":
                tau_p, kap_p = TAU, KAPPA
                h = state[2][DAMAGED_HEAD]
                h[2] = h[2] * 0.5; h[3] = h[3] * 0.5
                # reset that head's Adam moments (fair re-learning)
                for pi in range(4):
                    mom["mh"][DAMAGED_HEAD][pi] = np.zeros_like(mom["mh"][DAMAGED_HEAD][pi])
                    mom["vh"][DAMAGED_HEAD][pi] = np.zeros_like(mom["vh"][DAMAGED_HEAD][pi])
            elif cond in ("HEAD_KILL", "SUP_KILL"):
                tau_p, kap_p = TAU, KAPPA
                state[2][DAMAGED_HEAD] = fresh_head(rng)
                for pi in range(4):
                    mom["mh"][DAMAGED_HEAD][pi] = np.zeros_like(mom["mh"][DAMAGED_HEAD][pi])
                    mom["vh"][DAMAGED_HEAD][pi] = np.zeros_like(mom["vh"][DAMAGED_HEAD][pi])
            else:
                tau_p, kap_p = TAU, KAPPA
        else:
            tau_p, kap_p = TAU, KAPPA

        # ---- one round of development ----
        if cond == "SUP_KILL":
            # factory-repair reference: continued TRUE-LABEL training,
            # same per-round epochs; pool still perturbed (matched experience)
            pool = perturb(Xtr, rng)
            train_epochs(state, mom, rng, pool, ytr, SELF_EPOCHS)
            commit_n = len(pool)
            flagged_frac = 0.0
        else:
            pool, commit, labels, flagged = self_round(
                state, mom, rng, Xtr, tau_p, kap_p)
            train_epochs(state, mom, rng, pool[commit], labels[commit],
                         SELF_EPOCHS)
            commit_n = int(commit.sum())
            flagged_frac = float(np.mean(flagged))

        # ---- audit-side probe (original tau/kappa) ----
        pbar, ybar, conf, flagged_p, Ps = probe(state, Xte)
        V = verdicts(pbar)
        agree = head_agreement(Ps, ybar)
        d_head = float(np.mean(Ps[DAMAGED_HEAD].argmax(1) != ybar))
        traj.append({
            "round": r, "acc": float(np.mean(ybar == yte)),
            "accept": float(np.mean(conf >= TAU)),
            "commit_n": commit_n, "flagged_frac": flagged_frac,
            "head_agree": agree,
            "damaged_head_disagree": d_head,
            "V": V})
    return traj, V_fac


# ---------------- data ----------------
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
        globals()["COND_SEEDS"] = [500, 501]
        globals()["PERT_ROUND"] = 3
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} pert at round {PERT_ROUND}")

    res = {"config": {"tau": TAU, "kappa": KAPPA, "rounds": ROUNDS,
                      "pert_round": PERT_ROUND, "nseed": len(COND_SEEDS),
                      "conditions": CONDITIONS, "damaged_head": DAMAGED_HEAD,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- pass 1: unperturbed twins (own-norm reference trajectories) ----
    twins = {}
    for s in COND_SEEDS:
        traj, V_fac = run_cond(s, "TWIN", Xtr, ytr, Xte, yte)
        twins[s] = {"traj": traj, "V_fac": V_fac}
        print(f"  TWIN seed={s} done ({time.time()-t0:.0f}s)")

    # ---- pass 2: perturbed arms vs their twins ----
    for cond in CONDITIONS:
        if cond in ("TWIN", "FROZ_KILL"):
            continue
        rows = []
        for s in COND_SEEDS:
            traj, V_fac = run_cond(s, cond, Xtr, ytr, Xte, yte)
            tw = twins[s]["traj"]
            # per-round distances to own twin and to factory
            d_own = [float(np.mean(t["V"] != w["V"]))
                     for t, w in zip(traj, tw)]
            d_fac = [float(np.mean(t["V"] != V_fac)) for t in traj]
            for t, do, df in zip(traj, d_own, d_fac):
                t["d_own"], t["d_factory"] = do, df
                del t["V"]
            rows.append({"seed": s, "traj": traj,
                         "V_fac_diff": float(np.mean(V_fac != twins[s]["V_fac"]))})
        # aggregate
        agg = {}
        for key in ["acc", "accept", "commit_n", "flagged_frac", "head_agree",
                    "damaged_head_disagree", "d_own", "d_factory"]:
            agg[key] = [float(np.mean([r["traj"][i][key] for r in rows]))
                        for i in range(len(rows[0]["traj"]))]
        res["arms"][cond] = {"aggregate": agg, "runs": rows}
        pr = PERT_ROUND - 1  # index of the perturbation round in traj
        print(f"{cond:10s} d_own by round: "
              f"{['%.3f' % (a*100) for a in agg['d_own']]}")

    # ---- FROZ_KILL: damage at the end, no further training ----
    fz = []
    for s in COND_SEEDS:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
        for r in range(ROUNDS):
            pool, commit, labels, flagged = self_round(
                state, mom, rng, Xtr, TAU, KAPPA)
            train_epochs(state, mom, rng, pool[commit], labels[commit],
                         SELF_EPOCHS)
        V_before = verdicts(probe(state, Xte)[0])
        state[2][DAMAGED_HEAD] = fresh_head(rng)
        V_after = verdicts(probe(state, Xte)[0])
        pbar, ybar, conf, flagged_p, Ps = probe(state, Xte)
        fz.append({"d_raw": float(np.mean(V_after != V_before)),
                   "acc_after": float(np.mean(ybar == yte)),
                   "head_agree": head_agreement(Ps, ybar)})
    res["frozen_kill"] = fz
    print(f"FROZ_KILL raw damage: d={np.mean([f['d_raw'] for f in fz])*100:.3f}% "
          f"acc={np.mean([f['acc_after'] for f in fz])*100:.2f}%")

    # ---- classification (pre-registered rules) ----
    twins_agg_commit = float(np.mean(
        [t["commit_n"] for tw in twins.values() for t in tw["traj"]
         if t["round"] == ROUNDS]))
    cls = {}
    for cond, arm in res["arms"].items():
        d = arm["aggregate"]["d_own"]
        f = arm["aggregate"]["d_factory"]
        jump = d[PERT_ROUND - 1] - d[PERT_ROUND - 2]
        final = d[-1]
        decay = 1.0 - (final - d[PERT_ROUND - 2]) / jump if jump > 0 else 1.0
        cls[cond] = {
            "jump_pp": jump * 100, "final_d_own_pp": final * 100,
            "decay_frac": decay,
            "final_d_factory_pp": f[-1] * 100,
            "restore_own": bool(decay >= 0.5 and final < f[-1]),
            "revert_factory": bool(f[-1] < f[PERT_ROUND - 2] - 0.0005),
            "absorb": bool(jump > 0 and (final - d[PERT_ROUND - 2]) >= 0.8 * jump),
            "commit_starved": bool(
                arm["aggregate"]["commit_n"][-1] < 0.10 *
                twins_agg_commit),
            "commit_final": float(arm["aggregate"]["commit_n"][-1])}
    res["classification"] = cls

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m3_seed_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_seed_results.json")


if __name__ == "__main__":
    main()
