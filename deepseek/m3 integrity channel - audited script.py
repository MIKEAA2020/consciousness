#!/usr/bin/env python3
"""
THE STREAM-SIDE INTEGRITY CHANNEL - the frozen-label panel, and the
fourteenth coordinate. Corpus doc 36's experiment. Ordered at the S4-candidate
issuance via doc 35's Part 3 and doc 34's non-claims: "a stream-side integrity
channel (commit-label auditing against a frozen reference, the doc 22 panel
idea pointed at labels instead of quarantine) is the obvious next build and is
exactly the kind of promotion doc 28's table is for" - the sixth exclusion's
instrument (F6, the poisoned-well fake), the first coordinate aimed at the
attack class that currently wins.

THE PROMOTION (doc 28's coordinate template, third instantiation): a new
coordinate iota, the integrity reading of the received stream -
  - STATE: iota_sys, an EMA (rate eta_i = 0.3, the house rate) of d_r, the
    per-round disagreement between the FROZEN-LABEL PANEL's labels and the
    labels the stream actually delivered (post-attack, pre-repair: the
    channel reads the stream as it arrived).
  - RECORD: iota* self-registered at burn-in on the same pool0 draw that
    registers r* and f* (the healthy panel-vs-committee disagreement on the
    would-be commits; no RNG, deterministic).
  - WRITE CHANNEL: the calibration above (iota_sys <- (1-eta_i) iota_sys +
    eta_i d_r).
  - REPAIR CHANNEL (out-of-gate, like tau's and kappa's): when |iota_sys -
    iota*| > eps_i = 0.03, the channel ARMS and acts on the stream itself -
    REPAIR variant: committed items whose label the panel disputes
    (panel_label != label AND panel_conf >= tau0, doc 22's convention) have
    their label REWRITTEN to the panel's label before training; QUAR variant:
    those items are dropped from training. THE STRUCTURAL NOVELTY: tau's and
    kappa's repair channels repair the STATE toward the record; iota's
    repairs the WORLD toward the record - the first outward-facing repair
    channel in the corpus. The reading itself is never pulled back: while
    the attack runs, iota_sys parks OUTSIDE the band (the standing-
    negotiation pattern's converse: not a negotiation, an occupation).

THE PANEL (doc 22's instrument, pointed at labels): burn-in snapshots at
epochs {20, 25, 30}, never trained after snapshot; panel label = argmax of
the mean softmax across snapshots; panel_conf = its max. The panel reads
INPUTS (which the attacker does not touch) with frozen weights (which the
poison never trains) - the one grader in the system whose substrate is not
the attacker's. F6's exclusion, system-side.

THE QUESTION THE BATTERY CARRIES (doc 34's row): the defense stack - records,
channels, calibration, flags, brake, majority - does not stop the loop at any
rate >= 0.05. Does the fourteenth coordinate? THE FORK:
  (a) CLOSED: at every lambda in {0.05..0.50}: final tau >= 0.93, q_bar
      within 0.01 of the twin's, clean acc >= 95%, stream error that trains
      <= 2x the twin's own labeling error.
  (b) PARTIAL: the floor curve moves right (some rates close, higher ones
      not) - the onset dial re-priced by the channel; report which.
  (c) A third outcome, classified post hoc with disclosed criteria.

ARMS (perturbation at round 6, N=6 seeds, same-seed deterministic twins,
doc 34's full chassis in every particular - the channel's reading consumes
NO chassis rng, and the action fires only out-of-band, so in-band worlds are
bitwise to doc 34):
  IC_TWIN        lambda=0, no kicks, channel armed - the bitwise anchor
                 (expected: never arms -> bitwise to SP_TWIN)
  IC_KICK        tau_sys <- 0.75 at r6, lambda=0 - the FALSE-POSITIVE
                 control (the widened gate's softer commits: does the
                 channel misread sharpening as dirt?)
  IC_NARROW_P30  lambda=0.30, no kicks, channel armed (the defense at the
                 natural gate - doc 34 showed the attack needs no kicks)
  IC_P05..P50    both doc-30 kicks, lambda in {0.05, 0.10, 0.15, 0.30,
                 0.50}, channel armed (REPAIR) - the dose grid, the row
  IC_P30_NOCH    both kicks, lambda=0.30, action DISARMED (reading only) -
                 bitwise to SP_WL_P30: doc 34's world replicated AND the
                 pure-detection arm (the coordinate as reading, not defense)
  IC_P30_QUAR    both kicks, lambda=0.30, QUAR variant (the starvation
                 question: drop the disputed items instead of relabeling)
  IC_P05_QUAR    both kicks, lambda=0.05, QUAR variant (the low-dose
                 starvation: dropping 1-in-20 commits)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  I1  IC_TWIN is bitwise to doc 34's SP_TWIN (both coordinates, all 14
      rounds) AND never arms (iota stays in-band; the healthy panel-
      committee disagreement is a stationary reading). If it arms, that is
      the twin's false-positive, disclosed as such and bitwise breaks.
  I2  THE FORK above, per arm.
  I3  The arming clock: armed by r7 at lambda >= 0.10 (the +8pp footprint
      clears the 3pp deadband in one EMA step); at lambda = 0.05 the +4pp
      footprint needs the compounding (the dirt's own feedback) or the
      EMA's convergence - armed by r9 (doc 33's clock, now the channel's
      own onset dial; the dose-response of the CLOCK is the finding).
  I4  The occupation: once armed under a running attack, the channel fires
      every round thereafter (the received stream keeps its dirt); iota_sys
      parks outside the band at its steady state d_r(lambda) ~ iota* +
      0.84 lambda (the flip-catching rate), NOT at the band edge - the
      first coordinate whose repair channel cannot bring its own reading
      home (it repairs the stream, not itself).
  I5  IC_KICK: transient arming at most 2 rounds (the widened gate's
      softer commits), final tau within 0.005 of doc 34's SP_KICK course,
      acc within 0.3pp - the false alarm is cheap; OR no arming at all
      (doc 34's S4 sharpening keeps the labels panel-clean). Either way
      disclosed.
  I6  The cleaning arithmetic: at every armed round, cerr_stream_post <
      0.5 * cerr_stream_pre (the panel's ~93% accuracy on rewritten labels
      vs the poison's directed dirt); the two-term model post ~
      (1-catch)*pre + catch*panel_err measured per round.
  I7  The QUAR variant at lambda=0.30 drops ~25-30% of commits - doc 31's
      starvation-and-digestion question: does cleanliness buy back volume?
      Fork: QUAR strictly weaker than REPAIR on the rulers, or competitive.
  I8  The two-ruler certificate: at every arm the battery reads BOTH rulers
      first-class (internal telemetry incl. iota AND the clean test); the
      closed arms show both healthy (no shell); the NOCH arm shows doc 34's
      shell bitwise (internals parked on formulas, clean falling) - the
      channel is the difference between the two rows of that table.

Usage: python3 m3_integrity.py [--smoke]
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
ETA_I = 0.3                            # iota calibration EMA (the house rate)
EPS_I = 0.03                           # iota deadband (3pp: tighter than
                                        # tau's 5pp - iota's noise is a rate
                                        # on ~600 items, not a quantile)
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
ARMS = ["IC_TWIN", "IC_KICK", "IC_NARROW_P30",
        "IC_P05", "IC_P10", "IC_P15", "IC_P30", "IC_P50",
        "IC_P30_NOCH", "IC_P30_QUAR", "IC_P05_QUAR"]
LAMPS = {"P05": 0.05, "P10": 0.10, "P15": 0.15, "P30": 0.30, "P50": 0.50}
DOC34 = "/home/z/my-project/scripts/m3_streampoison_results.json"


# ---------------- model (identical to m3_streampoison.py) ----------------
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


def poison_labels(yt_, rng_attack, lam):
    """Doc 34's stream poison verbatim: flip each committed label with
    probability lam to the deterministic wrong class. Attack stream only."""
    if lam <= 0.0 or len(yt_) == 0:
        return yt_.copy()
    mask = rng_attack.random(len(yt_)) < lam
    out = yt_.copy()
    out[mask] = (out[mask] + 1) % 10
    return out


# ---------------- the arm ----------------
def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """One seed under one arm. RNG stream identical to m3_streampoison's
    SP arms (the panel and the channel consume no chassis randomness;
    the action fires only out-of-band, so in-band worlds stay bitwise)."""
    lam = 0.0
    for tag, v in LAMPS.items():
        if tag in arm:
            lam = v
            break
    kick_t = arm in ("IC_KICK",) or arm.startswith("IC_P0") or \
        arm.startswith("IC_P1") or arm.startswith("IC_P3") or \
        arm.startswith("IC_P5")
    kick_k = arm.startswith("IC_P0") or arm.startswith("IC_P1") or \
        arm.startswith("IC_P3") or arm.startswith("IC_P5")
    kick_t = kick_t and not arm.startswith("IC_NARROW")
    if arm == "IC_P30_NOCH":
        variant = "NOCH"
    elif arm.endswith("_QUAR"):
        variant = "QUAR"
    else:
        variant = "REPAIR"
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
    state, mom, panel = burn_in_with_panel(rng, Xtr, ytr)

    # ---- self-registration at burn-in end (same pool0 draw; no RNG) ------
    pool0 = perturb(Xtr, rng)
    _, ybar0, conf0, dis0, flag0, _ = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0
    # the fourteenth coordinate's record: the healthy panel-vs-committee
    # disagreement on the would-be commits of pool0 (normal route)
    would = (conf0 >= TAU0) & (~flag0)
    pl0, pc0 = panel_grade(pool0[would] if would.any() else pool0, panel)
    iota_star = float(np.mean(pl0 != (ybar0[would] if would.any() else ybar0)))
    iota_sys = iota_star

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, ROUNDS + 1):
        # ---- 1. the attack: state kicks (records stay honest) -----------
        if r == PERT_ROUND:
            if kick_t:
                tau_sys = KICK_TAU
            if kick_k:
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
        yt_p = poison_labels(yt_, rng_attack, lam if r >= PERT_ROUND else 0.0)
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(np.sum(yt_p != yt_))
        # ---- 5. the integrity channel: read, then act -------------------
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
        # ---- 6. training on the (possibly cleaned) stream ---------------
        # (unconditional: doc 34 calls train_epochs on the commit stream
        # regardless of emptiness, and the rng stream must match bitwise)
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
        # stream AS RECEIVED (pre-repair) - the channel sees the dirt while
        # it lasts, and stands down when it stops
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
            "variant": variant,
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
        globals()["ARMS"] = ["IC_TWIN", "IC_P30", "IC_P30_NOCH",
                             "IC_P05", "IC_KICK"]
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
                      "numpy": np.__version__},
           "arms": {}}

    for arm in ARMS:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise audits vs doc 34 (full chassis only) --------------------
    if not smoke:
        try:
            d34 = json.load(open(DOC34))
            for ours, theirs in [("IC_TWIN", "SP_TWIN"),
                                 ("IC_P30_NOCH", "SP_WL_P30")]:
                ref_t = d34["arms"][theirs]["aggregate"]["tau_sys"][:ROUNDS]
                ref_k = d34["arms"][theirs]["aggregate"]["kappa_sys"][:ROUNDS]
                got_t = [float(np.mean([rr["traj"][i]["tau_sys"]
                                        for rr in res["arms"][ours]["runs"]]))
                         for i in range(ROUNDS)]
                got_k = [float(np.mean([rr["traj"][i]["kappa_sys"]
                                        for rr in res["arms"][ours]["runs"]]))
                         for i in range(ROUNDS)]
                ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref_t, got_t))
                          and all(abs(a - b) < 1e-9
                                  for a, b in zip(ref_k, got_k)))
                key = f"{ours}_reproduces_doc34_{theirs}"
                res[key] = ok
                print(f"{ours} reproduces doc 34 {theirs} bitwise: {ok}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-34 audit unavailable: {e}")

    # ---- aggregates + footprints vs IC_TWIN ------------------------------
    twin_runs = res["arms"]["IC_TWIN"]["runs"]
    for arm in ARMS:
        if arm == "IC_TWIN":
            continue
        for si, row in enumerate(res["arms"][arm]["runs"]):
            tw = twin_runs[si]["traj"]
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
        rows = res["arms"][arm]["runs"]
        agg = {}
        keys = ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                "q_target", "kappa_target", "accept_own_tau", "d_own",
                "cerr_self", "cerr_stream_pre", "cerr_stream_post",
                "panel_acc", "iota_sys", "d_recv",
                "n_disputed_repaired", "n_flipped"]
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
        print(f"{arm:14s} tau: {['%.3f' % a for a in agg['tau_sys']]}")
        print(f"{'':14s} iota: {['%.3f' % a for a in agg['iota_sys']]}")
        print(f"{'':14s} acc: {['%.3f' % a for a in agg['acc']]}")

    # ---- classification (pre-registered rules) ---------------------------
    band_lo = TAU0 - EPS_T
    twin_q = float(np.mean([x for x in
                            res["arms"]["IC_TWIN"]["aggregate"]["q_target"]
                            [PERT_ROUND - 1:] if x is not None]))
    twin_tau_f = res["arms"]["IC_TWIN"]["aggregate"]["tau_sys"][-1]
    twin_cerr = float(np.mean([x for x in
                               res["arms"]["IC_TWIN"]["aggregate"]["cerr_self"]
                               [PERT_ROUND - 1:] if x is not None]))

    def tension_point(rec, live):
        denom = ETA + RHO_T - ETA * RHO_T
        return ((1 - ETA) * RHO_T * rec + ETA * live) / denom

    cls = {"_references": {
        "twin_q_bar": twin_q, "twin_final_tau": twin_tau_f,
        "twin_cerr_self": twin_cerr, "eps_i": EPS_I, "eta_i": ETA_I,
        "iota_star_twin": res["arms"]["IC_TWIN"]["aggregate"]["iota_sys"][0],
        "tension_honest_at_twin_q": tension_point(TAU0, twin_q)}}
    for arm in ARMS:
        ag = res["arms"][arm]["aggregate"]
        lam = (res["arms"][arm]["runs"][0]["traj"][-1]["lam"])
        qbar = float(np.mean([x for x in ag["q_target"][PERT_ROUND - 1:]
                              if x is not None]))
        final = ag["tau_sys"][-1]
        post = [x for x in ag["cerr_stream_post"][PERT_ROUND - 1:]
                if x is not None]
        pre = [x for x in ag["cerr_stream_pre"][PERT_ROUND - 1:]
               if x is not None]
        arm_rounds = [i + 1 for i in range(PERT_ROUND - 1, ROUNDS)
                      if ag["armed"][i] > 0.5]
        oob_rounds = [i + 1 for i in range(PERT_ROUND - 1, ROUNDS)
                      if ag["reading_oob"][i] > 0.5]
        entry = {
            "lam": lam, "variant": res["arms"][arm]["runs"][0]["traj"][-1]
            ["variant"],
            "final_tau": final, "final_kappa": ag["kappa_sys"][-1],
            "q_bar_realized": qbar, "q_sag_vs_twin": qbar - twin_q,
            "acc_final": ag["acc"][-1],
            "cerr_self_final": ag["cerr_self"][-1],
            "cerr_stream_pre_mean": float(np.mean(pre)) if pre else None,
            "cerr_stream_post_mean": float(np.mean(post)) if post else None,
            "iota_final": ag["iota_sys"][-1],
            "iota_park_vs_star": ag["iota_sys"][-1] -
            res["arms"]["IC_TWIN"]["aggregate"]["iota_sys"][0],
            "d_recv_final": ag["d_recv"][-1],
            "panel_acc_final": ag["panel_acc"][-1],
            "first_armed_round": min(arm_rounds) if arm_rounds else None,
            "armed_rounds": len(arm_rounds),
            "first_oob_round": min(oob_rounds) if oob_rounds else None,
            "oob_rounds": len(oob_rounds),
            "repair_rounds_tau": float(np.sum(
                [x for x in ag["repair_fired"][PERT_ROUND - 1:]
                 if x is not None])),
            "brake_any": bool(max(ag["brake"][PERT_ROUND - 1:]) > 0),
            "d_own_final_pp": ag["d_own"][-1] * 100,
            "tension_at_own_q": tension_point(TAU0, qbar)}
        # the fork (I2)
        if arm == "IC_TWIN":
            entry["fork"] = "reference"
        else:
            closed = (qbar >= twin_q - 0.01 and final >= 0.93
                      and ag["acc"][-1] >= 0.95)
            looped = (qbar < twin_q - 0.01 and
                      (final < band_lo or ag["acc"][-1] < 0.93))
            if looped:
                entry["fork"] = "LOOP"
                entry["lands_at_tension_honest"] = bool(
                    abs(final - tension_point(TAU0, qbar)) <= 0.008)
            elif closed:
                entry["fork"] = "CLOSED"
            else:
                entry["fork"] = "PARTIAL"
        # the cleaning ratio (I6)
        if post and pre and np.mean(pre) > 0:
            entry["cleaning_ratio"] = float(np.mean(post) / np.mean(pre))
        cls[arm] = entry
    res["classification"] = cls

    print("\n--- classification (band low %.3f; twin q_bar=%.4f; "
          "iota*=%.4f; eps_i=%.3f) ---"
          % (band_lo, twin_q,
             res["arms"]["IC_TWIN"]["aggregate"]["iota_sys"][0], EPS_I))
    for arm, c in cls.items():
        if arm.startswith("_"):
            continue
        print(f"{arm:14s} lam={c['lam']:.2f} {c.get('variant',''):6s} "
              f"fork={c.get('fork','')[:8]:8s} q={c['q_bar_realized']:.4f} "
              f"tau={c['final_tau']:.4f} acc={c['acc_final']*100:.1f}% "
              f"iota={c['iota_final']:.3f} "
              f"arm@r{c['first_armed_round']} x{c['armed_rounds']} "
              f"post/pre="
              f"{c.get('cleaning_ratio', float('nan')):.2f}")

    res["runtime_s"] = time.time() - t0
    out_path = "/home/z/my-project/scripts/m3_integrity_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> m3_integrity_results.json")


if __name__ == "__main__":
    main()
