#!/usr/bin/env python3
"""
THE STREAM POISON - attack the labels, leave the records alone. Corpus doc
34's experiment. Ordered at the joint-poison issuance via doc 30's coda:
"What is left unperturbed in the claim space is exactly what doc 29 and this
document, read together, expose: the stream between the coordinates - the
labels the widened gate admits under a loosened flag - where the dirt
accumulates that no formula watches. The next attack should not touch the
records at all. It should poison the thing the records are made of."

THE THREAT MODEL (narrower than every previous arm): the attacker has write
access to the TRAINING STREAM between grading and training - each committed
item's label is corrupted with probability lambda (wrong label = (y+1)%10)
- and to NOTHING else. The records tau* = 0.9 and kappa* = 0.95 stay honest
in every arm (never written, never kicked in record space); the kicks, where
present, are STATE kicks only (tau_sys <- 0.75 and/or kappa_sys <- 0.80 at
round 6, doc 30's joint-kick values), so the widened gate and the loosened
flag are recreated WITHOUT touching the records - and the honest record's
repair channel becomes, for the first time, the DEFENSE: the thing pulling
the gate back while the dirt tries to hold it open.

THE MECHANISM UNDER TEST (doc 30's registered-but-inverted conspiracy, now
endogenous): in doc 30 the J4(b) unraveling path needed the widened gate's
dirtier commits to SAG the live evaluator q_bar, and instead q_bar ROSE
(+0.0124: the extra training sharpened the evaluator past its dirtier
labels). That dirt was EMERGENT (24.3% commit error from the wider, looser
machinery) and the assimilation won. This battery makes the dirt DIRECTED
and SCALABLE (lambda in {0.15, 0.30, 0.50}) and asks the binary question:
is the assimilation a property of the chassis's plasticity (it absorbs any
stream at any rate) or of the dirt's magnitude (past some lambda the loop
lives: dirt -> q_bar falls -> calibration drags tau_sys down -> the gate
widens further -> more dirt), with the no-repair arm as the loop
unleashed?

THE CLOSED FORMS THE FORK LANDS ON (pre-registered):
  - If the loop lives and q_bar falls below the band-break threshold
    th2_nodrift = (tau*-eps)(eta+rho-eta*rho) - (1-eta)rho*tau* all over eta
    = 0.792 at this registration, the honest record's repair channel fires
    and loses, and the norm lands at the tension point against its OWN
    honest record with its OWN rotted evaluator:
        T(tau*_honest, q_bar) = [(1-eta)rho*0.9 + eta*q_bar]/(eta+rho-eta*rho)
    - the stream poison, if it wins, converts the honest record into the
    attacker's lever (doc 23's tension arithmetic, now with the record on
    the defense's side and the rot self-inflicted).
  - The bridge to doc 33: the piecewise scalar recursion fed the arm's
    REALIZED q(r) schedule (the non-stationary queue's response function,
    validated to 0.0000 on every FLAT arm in doc 33) should track tau(r)
    here too - the endogenous falling queue and the exogenous ramp are the
    same queue. Tolerance 0.010 (wider than doc 33's 0.005: the
    full-training chassis unfreezes the feedback the FLAT chassis froze;
    disclosed).
  - The dose-response: commit-stream error ~ lambda + (1-lambda)*cerr_self
    by construction; the REGISTERED question is q_bar's response shape -
    linear-in-lambda assimilation vs threshold-in-lambda breakage.

ARMS (perturbation at round 6, N=6 seeds, same-seed deterministic twins,
full doc-26/30 training chassis, kappa-state):
  SP_TWIN        lambda=0, no kicks = doc 26's KAP_TWIN (bitwise anchor)
  SP_KICK        tau_sys <- 0.75 at r6, lambda=0 (the widened gate alone:
                 does the transient dirt close with the gate?)
  SP_NARROW_P30  lambda=0.30, no kicks (dirt alone at the natural gate)
  SP_WIDE_P30    tau_sys <- 0.75 at r6, lambda=0.30 (the race: repair
                 closes the gate vs dirt holds it open)
  SP_WL_P15      tau_sys <- 0.75, kappa_sys <- 0.80 at r6, lambda=0.15
  SP_WL_P30      the same kicks, lambda=0.30 (the main arm - doc 30's
                 joint world rebuilt with honest records and directed dirt)
  SP_WL_P50      the same kicks, lambda=0.50 (the dose)
  SP_WL_P30_NOREP the same kicks + poison, both repair channels dis-armed
                 from r6 (the loop unleashed: nothing pulls the gate back
                 but the calibration, which follows the falling q_bar)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  S1  SP_TWIN reproduces doc 26's KAP_TWIN bitwise (tau and kappa, all 10
      rounds). The poison draws from a SEPARATE generator (seed + 977),
      never the chassis stream, so lambda=0 arms are bitwise by construction.
  S2  THE FORK: (a) ASSIMILATED at lambda <= 0.30: q_bar within 0.01 of
      SP_TWIN's, final tau >= 0.93, acc >= 95%, commit-stream error
      elevated (~lambda + (1-lambda)*15.6%) but the norm coordinates on the
      twin's course. (b) THE LOOP: q_bar sags > 0.01 below twin, tau falls
      below the band with the repair firing and losing, the landing at
      T(0.9, q_bar_final) within 0.008, acc < 93%; and the NOREP arm runs
      away (final tau < 0.80, monotone fall). (c) A third outcome,
      classified post hoc with disclosed criteria.
  S3  Dose-response: q_bar's lambda-response is monotone non-increasing;
      the REGISTERED shape question (linear assimilation vs threshold
      breakage) is answered by the three-point arc.
  S4  SP_KICK: the gate closes by r8-9 (tau_sys >= 0.93), commit error
      modestly elevated vs twin (the wider gate admits softer items), no
      norm movement.
  S5  If the brake fires (flagged_frac > 0.60) at any lambda, the
      majority-route commits are measured separately from the view-route
      (the machinery's last line under directed dirt: which route is
      dirtier?). The kappa coordinate's response: the flag RATE vs f*
      (a rising disagreement component shrinks the rule's remaining
      budget - the rate-controller under dirt, doc 33's lesson in its
      converse form).
  S6  The bridge: the piecewise simulator on the realized q(r) tracks
      tau(r) within 0.010 on every arm (doc 33's response function,
      endogenous version).
  S7  kappa: kappa_sys trajectory vs the twin's; kappa repair engagement;
      the coordinates stay factorized in the stream (the kappa machinery
      reads confidences, not labels) unless the dirt's training feedback
      couples them.
  S8  Rulers: accuracy's response to lambda is the cushion measurement
      (doc 30's 8.7pp of label error vs 0.3pp of accuracy loss); d_own vs
      twin by lambda.

Usage: python3 m3_streampoison.py [--smoke]
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
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 14, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
KICK_TAU, KICK_KAP = 0.75, 0.80        # doc 30's joint-kick values
ATTACK_STREAM = 977                     # the poison's separate rng offset
ARMS = ["SP_TWIN", "SP_KICK", "SP_NARROW_P30", "SP_WIDE_P30",
        "SP_WL_P05", "SP_WL_P10", "SP_WL_P15", "SP_WL_P30", "SP_WL_P50",
        "SP_WL_P30_NOREP"]
LAMPS = {"P05": 0.05, "P10": 0.10, "P15": 0.15, "P30": 0.30,
         "P50": 0.50}
DOC26 = "/home/z/my-project/scripts/m3_kappa_results.json"
DOC30 = "/home/z/my-project/scripts/m3_jointpoison_results.json"


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


# ---------------- the round (doc-30 chassis + route-split cerr) -----------
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
        tele["route"] = "brake"
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
        tele["route"] = "normal"
    tele["n_commit"] = int(commit.sum())
    # audit-side only (no RNG, not system-visible): the system's OWN labeling
    # error among committed items, before any poison
    tele["cerr_self"] = float(np.mean(labels[commit] != ytr[commit])) \
        if commit.any() else 0.0
    return pool[commit], labels[commit], tele, pool, ytr[commit]


def poison_labels(yt_, rng_attack, lam):
    """The stream poison: flip each committed label with probability lam to
    the deterministic wrong class (y+1)%10. Draws from the ATTACK stream
    only - the chassis rng is never touched (lambda=0 arms stay bitwise)."""
    if lam <= 0.0 or len(yt_) == 0:
        return yt_.copy()
    mask = rng_attack.random(len(yt_)) < lam
    out = yt_.copy()
    out[mask] = (out[mask] + 1) % 10
    return out


def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. RNG stream identical to m3_kappa_twohead /
    m3_jointpoison's KAP arms (the poison draws from a separate generator)."""
    lam = 0.0
    for tag, v in LAMPS.items():
        if tag in arm:
            lam = v
            break
    kick_t = arm in ("SP_KICK", "SP_WIDE_P30") or arm.startswith("SP_WL")
    kick_k = arm.startswith("SP_WL")
    no_rep = (arm == "SP_WL_P30_NOREP")
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
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
        # ---- 1. the attack: state kicks (records stay honest everywhere) --
        if r == PERT_ROUND:
            if kick_t:
                tau_sys = KICK_TAU
            if kick_k:
                kappa_sys = KICK_KAP
        # ---- 2. out-of-gate repair channels (dis-armable; honest records) -
        repair_t = repair_k = False
        if not (no_rep and r >= PERT_ROUND):
            if abs(tau_sys - tau_star) > EPS_T:
                tau_sys += RHO_T * (tau_star - tau_sys)
                repair_t = True
            if abs(kappa_sys - kappa_star) > EPS_K:
                kappa_sys += RHO_K * (kappa_star - kappa_sys)
                repair_k = True
        # ---- 3+4. grading, brake, THE POISON, training ----
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        # the attack's write: corrupt the committed labels (records untouched)
        yt_p = poison_labels(yt_, rng_attack, lam if r >= PERT_ROUND else 0.0)
        # audit-side: the stream's true error (what the evaluator trains on)
        tele["cerr_stream"] = (float(np.mean(yt_p != yt_true))
                               if len(yt_p) else 0.0)
        tele["n_flipped"] = int(np.sum(yt_p != yt_))
        train_epochs(state, mom, rng, Xt_, yt_p, SELF_EPOCHS)
        # ---- 5. calibration write channels ----
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
        # ---- 6. telemetry ----
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs(
            Xte, state, kappa_sys)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "kappa_target": kap_t, "lam": lam,
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


# ---------------- the piecewise scalar simulator (doc 33's bridge) --------
def simulate_scalar_q(q_sched, tau_start, pert=PERT_ROUND,
                      eta=ETA, eps=EPS_T, rho=RHO_T, tau_star=TAU0,
                      repair_armed=True):
    """The exact scalar recursion on a REALIZED q schedule (the doc-33
    response function): repair-if-out-of-band, then calibration toward q_r.
    No drift, honest record - the stream-poison world's own arithmetic. The
    NOREP arms run with repair_armed=False (their own configuration)."""
    tau = tau_start
    out = []
    for i, r in enumerate(range(pert, pert + len(q_sched))):
        fired = repair_armed and abs(tau - tau_star) > eps
        if fired:
            tau += rho * (tau_star - tau)
        tau = (1 - eta) * tau + eta * q_sched[i]
        out.append({"round": r, "tau": tau, "fired": fired})
    return out


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["ROUNDS"] = 6
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        globals()["ARMS"] = ["SP_TWIN", "SP_WL_P30", "SP_WL_P30_NOREP"]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} poison at round "
          f"{PERT_ROUND}, rounds={ROUNDS}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "brake_th": BRAKE_TH,
                      "rounds": ROUNDS, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS), "arms": ARMS,
                      "kick_tau": KICK_TAU, "kick_kap": KICK_KAP,
                      "attack_stream_offset": ATTACK_STREAM,
                      "numpy": np.__version__},
           "arms": {}}

    # ---- run every arm ----
    for arm in ARMS:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise audit: SP_TWIN vs doc 26 KAP_TWIN (full chassis only) ----
    if not smoke:
        try:
            d26 = json.load(open(DOC26))
            ref_t = d26["arms"]["KAP_TWIN"]["aggregate"]["tau_sys"][:ROUNDS]
            ref_k = d26["arms"]["KAP_TWIN"]["aggregate"]["kappa_sys"][:ROUNDS]
            got_t = [float(np.mean([rr["traj"][i]["tau_sys"]
                                    for rr in res["arms"]["SP_TWIN"]["runs"]]))
                     for i in range(ROUNDS)]
            got_k = [float(np.mean([rr["traj"][i]["kappa_sys"]
                                    for rr in res["arms"]["SP_TWIN"]["runs"]]))
                     for i in range(ROUNDS)]
            res["sptwin_reproduces_doc26_kaptwin"] = bool(
                all(abs(a - b) < 1e-9 for a, b in zip(ref_t, got_t)) and
                all(abs(a - b) < 1e-9 for a, b in zip(ref_k, got_k)))
            print("SP_TWIN reproduces doc 26 KAP_TWIN bitwise (tau+kappa): "
                  f"{res['sptwin_reproduces_doc26_kaptwin']}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-26 audit unavailable: {e}")

    # ---- aggregates + footprints vs SP_TWIN ----
    twin_runs = res["arms"]["SP_TWIN"]["runs"]
    # footprints first (every arm vs SP_TWIN, same round, same seed)
    for arm in ARMS:
        if arm == "SP_TWIN":
            continue
        for si, row in enumerate(res["arms"][arm]["runs"]):
            tw = twin_runs[si]["traj"]
            for t, w in zip(row["traj"], tw):
                t["d_own"] = float(np.mean(t["V"] != w["V"]))
    # then strip V everywhere (the twin gets d_own = 0 by definition)
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
        keys = ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                "q_target", "kappa_target", "accept_own_tau", "d_own",
                "cerr_self", "cerr_stream", "n_flipped"]
        for key in keys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in ["repair_fired", "repair_fired_kappa", "brake"]:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:16s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':16s} q  : {['%.3f' % a for a in agg['q_target']]}")
        print(f"{'':16s} acc: {['%.3f' % a for a in agg['acc']]}")

    # ---- classification (pre-registered rules) ----
    band_lo = TAU0 - EPS_T
    denom = ETA + RHO_T - ETA * RHO_T
    th2_nodrift = ((TAU0 - EPS_T) * denom - (1 - ETA) * RHO_T * TAU0) / ETA
    twin_q = float(np.mean([x for x in
                            res["arms"]["SP_TWIN"]["aggregate"]["q_target"]
                            [PERT_ROUND - 1:] if x is not None]))
    twin_tau_f = res["arms"]["SP_TWIN"]["aggregate"]["tau_sys"][-1]
    twin_kap_f = res["arms"]["SP_TWIN"]["aggregate"]["kappa_sys"][-1]
    twin_cerr = float(np.mean([x for x in
                               res["arms"]["SP_TWIN"]["aggregate"]["cerr_self"]
                               [PERT_ROUND - 1:] if x is not None]))

    def tension_point(rec, live):
        return ((1 - ETA) * RHO_T * rec + ETA * live) / denom

    cls = {"_references": {
        "twin_q_bar": twin_q, "twin_final_tau": twin_tau_f,
        "twin_final_kappa": twin_kap_f, "twin_cerr_self": twin_cerr,
        "th2_nodrift": th2_nodrift,
        "tension_honest_record_at_twin_q": tension_point(TAU0, twin_q)}}
    for arm in ARMS:
        ag = res["arms"][arm]["aggregate"]
        lam = (res["arms"][arm]["runs"][0]["traj"][-1]["lam"])
        qbar = float(np.mean([x for x in ag["q_target"][PERT_ROUND - 1:]
                              if x is not None]))
        final = ag["tau_sys"][-1]
        # the bridge: simulator on the realized q schedule (repair armed
        # iff the arm's is)
        tau_start = ag["tau_sys"][PERT_ROUND - 2]
        q_sched = [x for x in ag["q_target"][PERT_ROUND - 1:]]
        sim = simulate_scalar_q(q_sched, tau_start,
                                repair_armed=(arm != "SP_WL_P30_NOREP"))
        entry = {
            "lam": lam,
            "final_tau": final, "final_kappa": ag["kappa_sys"][-1],
            "min_tau": float(np.min(ag["tau_sys"][PERT_ROUND - 1:])),
            "q_bar_realized": qbar, "q_sag_vs_twin": qbar - twin_q,
            "acc_final": ag["acc"][-1],
            "cerr_self_final": ag["cerr_self"][-1],
            "commit_final": ag["n_commit"][-1],
            "repair_rounds": float(np.sum(
                [x for x in ag["repair_fired"][PERT_ROUND - 1:]
                 if x is not None])),
            "brake_any": bool(max(ag["brake"][PERT_ROUND - 1:]) > 0),
            "flag_rate_final": ag["flagged_frac"][-1],
            "disagree_final": ag["disagree_frac"][-1],
            "d_own_final_pp": ag["d_own"][-1] * 100,
            "sim_final_tau": sim[-1]["tau"],
            "sim_error": final - sim[-1]["tau"],
            "tension_at_own_q": tension_point(TAU0, qbar)}
        # the fork (S2)
        if arm == "SP_TWIN":
            entry["fork"] = "reference"
        else:
            assimilated = (qbar >= twin_q - 0.01 and final >= 0.93
                           and ag["acc"][-1] >= 0.95)
            looped = (qbar < twin_q - 0.01 and
                      (final < band_lo or ag["acc"][-1] < 0.93))
            if looped:
                entry["fork"] = "LOOP"
                entry["lands_at_tension_honest"] = bool(
                    abs(final - tension_point(TAU0, qbar)) <= 0.008)
            elif assimilated:
                entry["fork"] = "ASSIMILATED"
            else:
                entry["fork"] = "THIRD"
        if arm == "SP_WL_P30_NOREP":
            taus = ag["tau_sys"][PERT_ROUND - 1:]
            entry["runaway"] = bool(final < 0.80 and
                                    all(taus[i + 1] <= taus[i] + 1e-9
                                        for i in range(len(taus) - 1)))
        cls[arm] = entry
    res["classification"] = cls

    # doc-30 references for the comparison table (cited, not re-run)
    if not smoke:
        try:
            d30 = json.load(open(DOC30))
            jp = d30["arms"]["JP_JOINT"]["aggregate"]
            cls["_references"]["doc30_jp_joint"] = {
                "final_tau": jp["tau_sys"][-1],
                "final_kappa": jp["kappa_sys"][-1],
                "q_bar": float(np.mean(jp["q_target"][PERT_ROUND - 1:])),
                "commit_err": jp["commit_err"][-1],
                "acc": jp["acc"][-1]}
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-30 reference unavailable: {e}")

    print("\n--- classification (band low edge %.3f; th2_nodrift=%.4f; "
          "twin q_bar=%.4f) ---" % (band_lo, th2_nodrift, twin_q))
    for arm, c in cls.items():
        if arm.startswith("_"):
            continue
        print(f"{arm:16s} lam={c['lam']:.2f} fork={c.get('fork','')[:12]:12s} "
              f"q={c['q_bar_realized']:.4f} tau={c['final_tau']:.4f} "
              f"kap={c['final_kappa']:.4f} acc={c['acc_final']*100:.1f}% "
              f"cerr={c['cerr_self_final']*100:.1f}% "
              f"sim_err={c['sim_error']:+.4f}")
    if "doc30_jp_joint" in cls.get("_references", {}):
        j = cls["_references"]["doc30_jp_joint"]
        print(f"doc30 JP_JOINT ref: tau={j['final_tau']:.4f} "
              f"kap={j['final_kappa']:.4f} q={j['q_bar']:.4f} "
              f"cerr={j['commit_err']*100:.1f}% acc={j['acc']*100:.1f}%")

    res["runtime_s"] = time.time() - t0
    out_path = "/home/z/my-project/scripts/m3_streampoison_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_streampoison_results.json")


if __name__ == "__main__":
    main()
