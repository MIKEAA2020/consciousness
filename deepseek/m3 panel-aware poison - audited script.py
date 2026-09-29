#!/usr/bin/env python3
"""
THE PANEL-AWARE POISON - doc 36's filed threat, run at last. Corpus doc 41's
experiment. Ordered at the second-candidate issuance: "the panel-aware poison
(doc 36's filed threat: flip toward the panel's low-confidence blind spots,
gating on panel-conf - the catch rate is the whole defense)."

THE THREAT, AS FILED (doc 36 threats (ii)): "a panel-aware poison - flipping
toward the panel's own runner-up class, or gating flips on items where the
panel's confidence sits just under tau0 - would lower the catch rate, and the
catch rate is the whole defense; the threat is filed, not run." And the award
it attacks (doc 39, V7): CONTAINED-UNDER-8.4 awarded at lambda in
{0.05, 0.10, 0.30}, scoped to the UNIFORM attacker, the panel-aware threat
named as the award's standing condition. This battery is that scope
condition's test.

THE DECOMPOSITION (pre-registered before the run): the built channel's
dispute rule is ITEM-LEVEL - disputed = (panel_label != commit_label) AND
(panel_conf >= tau0) - the confidence is a property of the item and the
panel, not of the label the commit carries. Doc 36's threat, filed as one
mechanism, is therefore two:
  DIRECTION (the runner-up): which wrong class the flip lands on. It cannot
  change the dispute outcome (the rule never looks at the flip direction) -
  the catch rate is UNCHANGED; the damage per surviving flip is not (a
  plausible confusion trains a different gradient than an arbitrary class).
  SELECTION (the gate): which items are flipped. Flipping only sub-tau0
  items makes every flip undisputed BY CONSTRUCTION - the catch rate -> 0.
  The channel still arms (d_recv counts label disagreement,
  confidence-blind): armed, firing, catching nothing but organic errors.

THE BATTERY (doc 36's stream chassis in every particular: REPAIR variant,
both doc-30 kicks at the pert round, 14 rounds, N=6, the 2x2 at three doses
plus the twin plus two disarmed anchors):
  PP_TWIN       the bitwise anchor (no kicks, no poison)
  at lambda in {0.05, 0.10, 0.30}:
    PP_U{l}     UNIFORM selection, uniform direction (doc 34 verbatim -
                the awarded world, the bitwise anchor)
    PP_R{l}     UNIFORM selection (the SAME Bernoulli draws as U), RUNNER-UP
                direction - the direction isolated
    PP_G{l}     GATED selection, uniform direction - the gate isolated
    PP_J{l}     GATED selection + RUNNER-UP direction - the joint, the
                strongest filed threat
  PP_J05_NOCH / PP_J30_NOCH   the disarmed anchors under the joint attack
                (the registry effect re-measured UNDER the attack)

THE ATTACKER'S INFORMATION MODEL: the panel is deterministic and frozen; the
attacker reads it by running it (panel_top2, no rng). Disclosed conventions:
the gated selection is deterministic - rank the sub-tau0 commits by panel
confidence DESCENDING (doc-36-literal "just under tau0" first, filling down
as the dose demands), take m = round(lambda*n); the uniform selection is
Bernoulli (the attack stream, offset 977); the R arms consume the SAME draws
as the U arms (the same items flipped, the direction changed - the isolation
the 2x2 needs). The gate/joint arms consume no attack randomness.

PRE-REGISTERED PREDICTIONS (fixed before execution):
  PP1  the bitwise anchors: PP_TWIN, PP_U05/10/30 audit to doc 36's
       IC_TWIN/IC_P05/IC_P10/IC_P30 on both coordinates (1e-9).
  PP2  the direction's null: PP_R's flip-catch = PP_U's (|diff| <= 0.05 at
       every dose; the dispute rule is item-level). The damage clause forks:
       (a) PP_R's final clean acc <= PP_U's + 1pp (the runner-up at least as
       damaging), or (b) PP_R's acc > PP_U's + 1pp (the plausible-confusion
       training less damaging than the arbitrary flip) - either way the fork
       is the finding.
  PP3  the gate's kill: PP_G/PP_J flip-catch < 0.05 at every dose; the
       channel ARMS (first armed round within 1 of U's); the rewrites under
       G/J are organic-only.
  PP4  the capacity bound: the blind-spot mass B (committed items with
       panel_conf < tau0) measured per round; the realized dose =
       min(lambda, B); if B < lambda the G/J arms under-dose (disclosed -
       the attack's own ceiling).
  PP5  THE AWARD RE-GRADE: the four doc-39 conditions at each dose under
       each attack. Forks: at 0.05 under G/J, (a) the clean ruler holds
       (>= 93%) or (b) breaks; at 0.10/0.30 under G/J, (b). (ii) is
       expected to HOLD everywhere (the channel arms and engages) - the
       instrument's sharpest cell: engagement without effect.
  PP6  the registry effect under attack: acc_J30 - acc_J30_NOCH measured
       against the uniform's 0.759 (doc 39) - fork: ~0 (the channel adds
       nothing under the joint attack) or > 0.2 (partial protection).
  PP7  the detection signature: under G/J the reading rises (d_recv up -
       the flips disagree with the panel) while the action stays at the
       organic baseline (rewrites ~ the twin's) - the implied catch
       (catches_implied / dirt_implied) collapses to ~0 while the arming
       flag reads true. The channel's own telemetry carries the evidence of
       its evasion; the catch-rate certificate it lacks is filed as the
       instrument obligation.

Usage: python3 m3_panelpoison.py [--smoke]
"""
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration (doc 36's, verbatim) ----------------
K = 4
TAU0, KAPPA0 = 0.9, 0.95               # recipe initializers (disclosed)
EPS_T, RHO_T = 0.05, 0.5               # tau repair channel
EPS_K, RHO_K = 0.05, 0.5               # kappa repair channel (symmetric)
ETA, ETA_K = 0.3, 0.3                  # calibration EMA rates
ETA_I = 0.3                            # iota calibration EMA (the house rate)
EPS_I = 0.03                           # iota deadband (doc 36, as built)
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
PANEL_EPOCHS = [20, 25, 30]             # doc 22's time-sliced frozen panel
DOSES = ["05", "10", "30"]
PPCFG = {
    "PP_TWIN":     dict(lam=0.0,  mode="none",    noch=False),
    "PP_U05":      dict(lam=0.05, mode="uniform", noch=False),
    "PP_U10":      dict(lam=0.10, mode="uniform", noch=False),
    "PP_U30":      dict(lam=0.30, mode="uniform", noch=False),
    "PP_R05":      dict(lam=0.05, mode="run",     noch=False),
    "PP_R10":      dict(lam=0.10, mode="run",     noch=False),
    "PP_R30":      dict(lam=0.30, mode="run",     noch=False),
    "PP_G05":      dict(lam=0.05, mode="gate",    noch=False),
    "PP_G10":      dict(lam=0.10, mode="gate",    noch=False),
    "PP_G30":      dict(lam=0.30, mode="gate",    noch=False),
    "PP_J05":      dict(lam=0.05, mode="joint",   noch=False),
    "PP_J10":      dict(lam=0.10, mode="joint",   noch=False),
    "PP_J30":      dict(lam=0.30, mode="joint",   noch=False),
    "PP_J05_NOCH": dict(lam=0.05, mode="joint",   noch=True),
    "PP_J30_NOCH": dict(lam=0.30, mode="joint",   noch=True),
}
ARMS = list(PPCFG.keys())
DOC36 = "/home/z/my-project/scripts/m3_integrity_results.json"
DOC39_REGISTRY_EFFECT = 0.7592592592592594   # doc 39's NOCH delta (uniform)


# ---------------- model (identical to m3_integrity.py) ----------------
def init_params(rng):
    W1 = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 128)); b1 = np.zeros(128)
    heads = []
    for _ in range(K):
        Wh = rng.normal(0.0, np.sqrt(2.0 / 128), (128, 64)); bh = np.zeros(64)
        Wo = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 10)); bo = np.zeros(10)
        heads.append([Wh, bh, Wo, bo])
    return [W1, b1, heads]


def copy_state(state):
    W1, b1, heads = state
    return [W1.copy(), b1.copy(),
            [[w.copy() for w in h] for h in heads]]


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


def burn_in_with_panel(rng, Xtr, ytr):
    """Burn-in training with time-sliced snapshots for the frozen-label
    panel (doc 22's PANEL_EPOCHS). Per-epoch calls consume the SAME rng
    stream as doc 34's single 30-epoch call - bitwise preserved."""
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


# ---------------- the frozen-label panel (doc 22, at the labels) ----------
def panel_grade(X, panel):
    """Majority label + mean confidence of the frozen panel. NO rng - the
    panel reads inputs with frozen weights; bitwise anchors preserved."""
    pbar_sum = np.zeros((len(X), 10))
    for snap in panel:
        W1, b1, heads = snap
        _, zs = forward_all(X, W1, b1, heads)
        for z in zs:
            pbar_sum += softmax(z)
        # one snapshot's committee vote = mean over its heads
    pbar = pbar_sum / (len(panel) * K)
    return pbar.argmax(1), pbar.max(1)


def panel_top2(X, panel):
    """The ATTACKER's read: the panel's top-2 classes and the top confidence.
    Same arithmetic as panel_grade (deterministic, no rng) - the attacker
    reads the panel by running it."""
    pbar_sum = np.zeros((len(X), 10))
    for snap in panel:
        W1, b1, heads = snap
        _, zs = forward_all(X, W1, b1, heads)
        for z in zs:
            pbar_sum += softmax(z)
    pbar = pbar_sum / (len(panel) * K)
    top1 = pbar.argmax(1)
    conf = pbar.max(1)
    pb2 = pbar.copy()
    pb2[np.arange(len(X)), top1] = -np.inf
    run2 = pb2.argmax(1)
    return top1, run2, conf


# ---------------- the round (doc-34 chassis verbatim) --------------------
def self_round(state, rng, Xtr, ytr, tau_sys, kappa_sys):
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
    tele["cerr_self"] = float(np.mean(labels[commit] != ytr[commit])) \
        if commit.any() else 0.0
    return pool[commit], labels[commit], tele, pool, ytr[commit]


# ---------------- the attacks (doc 34's uniform + doc 36's threat) -------
def poison_panel_aware(yt_, Xt_, panel, rng_attack, lam, mode):
    """Returns (poisoned labels, effective-flip mask). The uniform path is
    doc 34 verbatim (one Bernoulli draw per item, flip to (y+1) % 10) - the
    bitwise anchor. run: the SAME draws, the flip to the panel's runner-up.
    gate: deterministic selection - the sub-tau0 commits ranked by panel
    confidence DESCENDING (doc-36-literal 'just under tau0' first, filling
    down), m = round(lam*n) of them, flipped to (y+1) % 10. joint: gated
    selection, runner-up direction. No-op flips (target == current label)
    are excluded from the effective mask."""
    n = len(yt_)
    if lam <= 0.0 or n == 0 or mode == "none":
        return yt_.copy(), np.zeros(n, dtype=bool)
    out = yt_.copy()
    flipped = np.zeros(n, dtype=bool)
    if mode == "uniform":
        mask = rng_attack.random(n) < lam
        out[mask] = (out[mask] + 1) % 10
        flipped = mask
    elif mode == "run":
        mask = rng_attack.random(n) < lam
        _, run2, _ = panel_top2(Xt_, panel)
        sel = np.where(mask)[0]
        new = run2[sel]
        ch = new != out[sel]
        out[sel[ch]] = new[ch]
        flipped = mask.copy()
        flipped[sel[~ch]] = False
    elif mode in ("gate", "joint"):
        _, run2, conf = panel_top2(Xt_, panel)
        blind = np.where(conf < TAU0)[0]
        m = int(round(lam * n))
        order = blind[np.argsort(-conf[blind], kind="stable")]
        sel = order[:m]
        if mode == "gate":
            new = (out[sel] + 1) % 10
        else:
            new = run2[sel]
        ch = new != out[sel]
        out[sel[ch]] = new[ch]
        flipped[sel[ch]] = True
    return out, flipped


def tension(p_star, live):
    return ((1 - ETA) * RHO_T * p_star + ETA * live) / (ETA + RHO_T - ETA * RHO_T)


# ---------------- the arm ----------------
def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. The chassis is doc 36's verbatim; the attack
    is the arm's mode. The panel and the channel consume no chassis
    randomness; the uniform path consumes the attack stream exactly as doc
    34/36 did - the anchors stay bitwise."""
    cfg = PPCFG[arm]
    lam, mode = cfg["lam"], cfg["mode"]
    variant = "NOCH" if cfg["noch"] else "REPAIR"
    kick = lam > 0.0
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
    state, mom, panel = burn_in_with_panel(rng, Xtr, ytr)

    # ---- self-registration at burn-in end (same pool0 draw; no RNG) ------
    pool0 = perturb(Xtr, rng)
    _, ybar0, conf0, dis0, flag0, _ = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0
    would = (conf0 >= TAU0) & (~flag0)
    pl0, pc0 = panel_grade(pool0[would] if would.any() else pool0, panel)
    iota_star = float(np.mean(pl0 != (ybar0[would] if would.any() else ybar0)))
    iota_sys = iota_star

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. the attack: state kicks (records stay honest) -----------
        if r == PERT_ROUND and kick:
            tau_sys = KICK_TAU
            kappa_sys = KICK_KAP
        # ---- 2. out-of-gate repair channels (tau, kappa) ----------------
        repair_t = repair_k = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        if abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        # ---- 3. the integrity channel's arming (out-of-band check) ------
        reading_oob = bool(abs(iota_sys - iota_star) > EPS_I)
        armed = reading_oob and variant != "NOCH"
        # ---- 4. grading, brake, THE POISON ------------------------------
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p, flipped = poison_panel_aware(
            yt_, Xt_, panel, rng_attack, lam if r >= PERT_ROUND else 0.0,
            mode)
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(flipped.sum())
        # ---- 5. the integrity channel: read, then act -------------------
        pl, pconf = panel_grade(Xt_, panel)
        d_recv = float(np.mean(pl != yt_p)) if len(yt_p) else 0.0
        disputed = (pl != yt_p) & (pconf >= TAU0)
        n_rep = int(disputed.sum())
        tele["n_flip_disputed"] = int((disputed & flipped).sum())
        tele["blind_mass"] = (float(np.mean(pconf < TAU0))
                              if len(pconf) else 0.0)
        tele["realized_dose"] = (float(np.mean(flipped))
                                 if len(flipped) else 0.0)
        if armed and variant == "REPAIR":
            yt_train = yt_p.copy()
            yt_train[disputed] = pl[disputed]
            X_train, yt_true_train = Xt_, yt_true
        else:
            X_train, yt_train, yt_true_train = Xt_, yt_p, yt_true
            n_rep = 0
        tele["cerr_stream_post"] = (float(np.mean(yt_train != yt_true_train))
                                    if len(yt_train) else 0.0)
        tele["panel_acc"] = (float(np.mean(pl == yt_true))
                             if len(yt_p) else 0.0)
        # ---- 6. training on the (possibly cleaned) stream ---------------
        train_epochs(state, mom, rng, X_train, yt_train, SELF_EPOCHS)
        # ---- 7. calibration write channels (tau, kappa, iota) -----------
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
        # the fourteenth coordinate's write channel: the reading of the
        # stream AS RECEIVED (pre-repair)
        iota_sys = (1 - ETA_I) * iota_sys + ETA_I * d_recv
        # ---- 8. telemetry ------------------------------------------------
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
            "variant": variant, "mode": mode,
            "n_disputed_repaired": n_rep,
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
        globals()["ARMS"] = ["PP_TWIN", "PP_U30", "PP_R30", "PP_G30",
                             "PP_J30", "PP_J30_NOCH"]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}, rounds={ROUNDS}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "eta_i": ETA_I,
                      "eps_i": EPS_I, "brake_th": BRAKE_TH,
                      "rounds": ROUNDS, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS), "arms": ARMS,
                      "kick_tau": KICK_TAU, "kick_kap": KICK_KAP,
                      "attack_stream_offset": ATTACK_STREAM,
                      "panel_epochs": PANEL_EPOCHS,
                      "doc39_registry_effect_uniform": DOC39_REGISTRY_EFFECT,
                      "numpy": np.__version__},
           "arms": {}}

    for arm in ARMS:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise audits vs doc 36 (full chassis only) --------------------
    if not smoke:
        try:
            d36 = json.load(open(DOC36))
            for ours, theirs in [("PP_TWIN", "IC_TWIN"),
                                 ("PP_U05", "IC_P05"),
                                 ("PP_U10", "IC_P10"),
                                 ("PP_U30", "IC_P30")]:
                ref_t = d36["arms"][theirs]["aggregate"]["tau_sys"][:ROUNDS]
                ref_k = d36["arms"][theirs]["aggregate"]["kappa_sys"][:ROUNDS]
                got_t = [float(np.mean([rr["traj"][i]["tau_sys"]
                                        for rr in res["arms"][ours]["runs"]]))
                         for i in range(ROUNDS)]
                got_k = [float(np.mean([rr["traj"][i]["kappa_sys"]
                                        for rr in res["arms"][ours]["runs"]]))
                         for i in range(ROUNDS)]
                ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref_t, got_t))
                          and all(abs(a - b) < 1e-9
                                  for a, b in zip(ref_k, got_k)))
                res.setdefault("audits", {})[
                    f"{ours}_reproduces_doc36_{theirs}"] = ok
                print(f"{ours} reproduces doc 36 {theirs} bitwise: {ok}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-36 audit unavailable: {e}")

    # ---- aggregates -------------------------------------------------------
    for arm in ARMS:
        rows = res["arms"][arm]["runs"]
        for row in rows:
            for t in row["traj"]:
                if "V" in t:
                    del t["V"]
        agg = {}
        keys = ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                "q_target", "kappa_target", "accept_own_tau",
                "cerr_self", "cerr_stream_pre", "cerr_stream_post",
                "panel_acc", "iota_sys", "d_recv", "blind_mass",
                "realized_dose",
                "n_disputed_repaired", "n_flipped", "n_flip_disputed"]
        for key in keys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in ["repair_fired", "repair_fired_kappa", "brake",
                     "armed", "reading_oob"]:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:12s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':12s} iota: {['%.3f' % a for a in agg['iota_sys']]}")
        print(f"{'':12s} acc: {['%.3f' % a for a in agg['acc']]}")

    # ---- classification (pre-registered rules) ---------------------------
    if not smoke:
        cls = {}
        A = res["arms"]

        def ag(arm, key):
            return A[arm]["aggregate"][key]

        def late_mean(arm, key, lo=None):
            lo = lo if lo is not None else PERT_ROUND - 1
            v = [x for x in ag(arm, key)[lo:] if x is not None]
            return float(np.mean(v)) if v else None

        def catch_rate(arm):
            num = late_mean(arm, "n_flip_disputed")
            den = late_mean(arm, "n_flipped")
            return (num / den) if (den and den > 0.1) else None

        def first_armed(arm):
            agg_armed = ag(arm, "armed")
            for i, v in enumerate(agg_armed):
                if v >= 0.5:
                    return i + 1
            return None

        # PP1: the audits
        cls["PP1_bitwise"] = dict(res.get("audits", {}))
        # PP2: the direction's null + the damage fork
        pp2 = {}
        for l in DOSES:
            cu, cr = catch_rate(f"PP_U{l}"), catch_rate(f"PP_R{l}")
            au = ag(f"PP_U{l}", "acc")[-1]
            ar = ag(f"PP_R{l}", "acc")[-1]
            pp2[l] = {
                "catch_uniform": cu, "catch_runner": cr,
                "catch_equal": (cu is not None and cr is not None
                                and abs(cr - cu) <= 0.05),
                "acc_uniform": au, "acc_runner": ar,
                "damage_fork": ("at-least" if ar <= au + 0.01
                                else "less-damaging")}
        cls["PP2_direction"] = pp2
        # PP3: the gate's kill + the arming unchanged
        pp3 = {}
        for tag in ["G", "J"]:
            for l in DOSES:
                arm = f"PP_{tag}{l}"
                pp3[arm] = {
                    "catch": catch_rate(arm),
                    "catch_killed": bool((catch_rate(arm) or 0.0) < 0.05),
                    "first_armed": first_armed(arm),
                    "first_armed_uniform": first_armed(f"PP_U{l}"),
                    "armed_rounds": int(sum(
                        1 for v in ag(arm, "armed") if v >= 0.5)),
                    "organic_rewrites_late": late_mean(
                        arm, "n_disputed_repaired")}
        cls["PP3_gate"] = pp3
        # PP4: the capacity bound
        pp4 = {}
        for tag in ["G", "J"]:
            for l in DOSES:
                arm = f"PP_{tag}{l}"
                lam = PPCFG[arm]["lam"]
                bm = late_mean(arm, "blind_mass")
                rd = late_mean(arm, "realized_dose")
                pp4[arm] = {
                    "lambda_filed": lam, "blind_mass": bm,
                    "realized_dose": rd,
                    "capacity_bound": bool(rd is not None
                                           and rd < lam - 0.005)}
        cls["PP4_capacity"] = pp4
        # PP5: the award re-grade (doc 39's four conditions, per attack)
        pp5 = {}
        noch_anchor = {"05": "PP_J05_NOCH", "30": "PP_J30_NOCH"}
        for tag in ["U", "R", "G", "J"]:
            for l in DOSES:
                arm = f"PP_{tag}{l}"
                acc_f = ag(arm, "acc")[-1]
                qbar = float(np.mean([x for x in
                                      ag(arm, "q_target")
                                      [PERT_ROUND - 1:] if x is not None]))
                final = ag(arm, "tau_sys")[-1]
                slope = (ag(arm, "tau_sys")[-1] - ag(arm, "tau_sys")[-4]) / 3.0
                oob_wo = sum(
                    1 for i in range(PERT_ROUND - 1, ROUNDS)
                    if ag(arm, "tau_sys")[i] < TAU0 - EPS_T
                    and ag(arm, "repair_fired")[i] < 0.5)
                n_armed = int(sum(1 for v in ag(arm, "armed") if v >= 0.5))
                if arm in noch_anchor.values():
                    anchor = arm.replace("_NOCH", "")
                    eff = (ag(anchor, "acc")[-1] - ag(arm, "acc")[-1])
                    eff_kind = "measured-under-attack"
                elif tag == "J" and l in noch_anchor:
                    eff = (ag(f"PP_J{l}", "acc")[-1]
                           - ag(noch_anchor[l], "acc")[-1])
                    eff_kind = "measured-under-attack"
                else:
                    eff = DOC39_REGISTRY_EFFECT
                    eff_kind = "inherited-uniform (no anchor under this attack)"
                cond1 = bool(acc_f >= 0.93)
                cond2 = bool(n_armed >= 1 and oob_wo == 0)
                cond3 = bool(eff >= 0.30)
                cond4 = bool(tension(TAU0, qbar) <= final <= TAU0 + EPS_T
                             and slope >= -1e-12)
                pp5[arm] = {
                    "clean_ruler": acc_f, "q_bar": qbar, "tau_final": final,
                    "late_tau_slope": slope, "tension_floor": tension(TAU0, qbar),
                    "registry_effect": eff, "registry_effect_kind": eff_kind,
                    "(i)_clean": cond1, "(ii)_engagement": cond2,
                    "(iii)_registry": cond3, "(iv)_bracket": cond4,
                    "awarded_CONTAINED_UNDER_8_4": bool(
                        cond1 and cond2 and cond3 and cond4)}
        cls["PP5_award_regrade"] = pp5
        # PP6: the registry effect under attack, headline
        cls["PP6_registry_under_attack"] = {
            "uniform_doc39": DOC39_REGISTRY_EFFECT,
            "joint_at_05": (ag("PP_J05", "acc")[-1]
                            - ag("PP_J05_NOCH", "acc")[-1]),
            "joint_at_30": (ag("PP_J30", "acc")[-1]
                            - ag("PP_J30_NOCH", "acc")[-1])}
        # PP7: the detection signature
        pp7 = {}
        tw_d = late_mean("PP_TWIN", "d_recv")
        tw_r = late_mean("PP_TWIN", "n_disputed_repaired")
        for l in DOSES:
            for tag in ["U", "J"]:
                arm = f"PP_{tag}{l}"
                d_rise = (late_mean(arm, "d_recv") or 0.0) - (tw_d or 0.0)
                r_rise = (late_mean(arm, "n_disputed_repaired") or 0.0) \
                    - (tw_r or 0.0)
                n_com = late_mean(arm, "n_commit") or 1.0
                pp7[arm] = {
                    "dirt_implied_rate": d_rise,
                    "catches_implied_rate": r_rise / n_com,
                    "implied_catch": (r_rise / n_com / d_rise
                                      if d_rise > 1e-6 else None)}
        cls["PP7_detection"] = pp7
        res["classification"] = cls
        print(json.dumps({k: v for k, v in cls.items()
                          if k != "PP5_award_regrade"}, indent=1)[:2500])
        print("PP5 award re-grade:")
        for arm, row in cls["PP5_award_regrade"].items():
            print(f"  {arm}: award={row['awarded_CONTAINED_UNDER_8_4']} "
                  f"clean={row['clean_ruler']:.4f} "
                  f"(i){row['(i)_clean']} (ii){row['(ii)_engagement']} "
                  f"(iii){row['(iii)_registry']} (iv){row['(iv)_bracket']}")

    res["runtime_s"] = time.time() - t0
    out = ("/home/z/my-project/scripts/m3_panelpoison_results_smoke.json"
           if smoke else
           "/home/z/my-project/scripts/m3_panelpoison_results.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"wrote {out} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
