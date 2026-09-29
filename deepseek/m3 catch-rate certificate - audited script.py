#!/usr/bin/env python3
"""
THE CATCH-RATE CERTIFICATE - doc 41's filed instrument obligation, built.
Corpus doc 43's experiment. Ordered at the panel-poison issuance: "the
catch-rate certificate (doc 41's filed instrument obligation: compute the
implied-catch ratio in-channel, register its band, and measure its
false-positive structure against doc 40's dial - the twin's organic drift
is the certificate's own horizon question)."

THE OBLIGATION, AS FILED (doc 41, Part 2.4 and the coda): the channel's
own telemetry carries the evidence of its evasion - under the uniform
attack the reading rises with the action (implied catch 0.675-0.816);
under the joint attack the reading rises and the action does not move
(0.0005-0.0013) - and "the certificate is one division away: catches
over dirt, both measured against the twin's organic baseline. The
channel does not compute it; the filing obligation is that it should."
Doc 41's non-claim, verbatim: "No claim that the implied-catch
certificate is sufficient detection - it separates evasion from organic
drift in this battery's worlds, and its own false-positive structure
(the twin's organic baseline drift, doc 40's dial) is unmeasured against
it." This battery builds the division and measures the non-claim.

THE IN-CHANNEL PROBLEM, AND THE DESIGN (pre-registered before the run):
doc 41's ratio used the TWIN's baseline - a world the deployed channel
cannot run. The certificate must replace the twin with registrations of
the channel's own:
  D_base(r)  the reading's registered course: iota_star (the burn-in
             record) + delta_hat * max(0, r - 5), the COMP cone - the
             drift slope registered on the known-clean prefix (rounds
             1-5, pre-attack; doc 40's registration, applied to every
             arm: the certificate always cones, whatever the channel's
             own dial).
  R_org      the organic dispute-rate baseline: an EMA (rate 0.3, the
             house rate) of the WOULD-FIRE dispute rate (the panel's
             confident disagreements, counted every round, armed or
             not), updated ONLY on in-band rounds (the REREG guard:
             never re-baseline while the channel reads dirt) AND only
             when the round's rate is within 0.01 of the current
             baseline (the jump guard: the organic drift is slow, and a
             jump is dirt - the one-round lag of doc 33's silent
             breakage, refused instead of inherited).
  The division, per round, cumulative over ARMED rounds (doc 41's
  late-means, made in-channel):
    X_raw(r) = d_recv(r) - D_base(r)              (the reading excess)
    N_raw(r) = n_dis_raw(r)/n_com(r) - R_org      (the action excess)
    C(r)     = mean(N_raw over armed) / mean(X_raw over armed)
  THE REGISTERED BAND: the certificate speaks only when armed AND
  X_cum > 0.01 (nothing to explain below one point of elevation);
  C >= 0.25 reads CATCHING; C < 0.25 reads EVASION. The floor is
  registered at an order of magnitude below the uniform worlds' readings
  and two above the joint worlds' - conservative by construction, and
  its margins are themselves a registered measurement (CC6).

THE BATTERY (doc 36/41's chassis in every particular; the certificate
consumes NO randomness and takes NO action - every world must replay
its predecessor bitwise, the certificate's innocence):
  the 14-round attack arms (audit bitwise to doc 41's PP arms):
    CC_TWIN, CC_U05/10/30, CC_G05/10/30, CC_J05/10/30
  the 30-round dial twins (audit bitwise to doc 40's ID arms - the
  certificate's false-positive worlds: the twin's organic drift, under
  every filed fix):
    CC_T30_BASE / CC_T30_WIDE / CC_T30_COMP / CC_T30_REREG

PRE-REGISTERED PREDICTIONS (fixed before execution):
  CC1  THE INNOCENCE AUDITS: all ten 14-round arms reproduce doc 41's
       PP counterparts bitwise (tau/kappa trajectories, 1e-9); all four
       30-round twins reproduce doc 40's ID counterparts bitwise. The
       certificate is a reading, not an act.
  CC2  THE HONEST CATCH: the U arms' final C in [0.35, 0.90] - the
       in-channel cone baseline sits BELOW the twin's realized drift
       (the certificate does not know the drift it has not seen), so C
       reads below doc 41's post-hoc 0.675-0.816, tightest at 0.05 -
       the design's tightest honest cell, its margin reported. Verdict
       CATCHING at every dose.
  CC3  THE EVASION: the G/J arms' C < 0.05 at every dose; verdict
       EVASION at every dose - the gated attack's dirt raises the
       reading and not the action, and the division sees it.
  CC4  THE FALSE-POSITIVE FORK (the battery's point): on the 30-round
       twins' armed rounds - the organic drift crossings that arm the
       channel (doc 40: first arming 14-26) - fork: (a) C >= 0.25 (the
       drift's disagreement is confident-item-heavy; the organic arming
       reads CATCHING - a false attribution, benign for the alarm,
       disclosed as the certificate's blindness), or (b) C < 0.25 with
       X_cum > 0.01 (the drift's disagreement is blind-item-heavy -
       FALSE EVASION ALARMS on healthy worlds; the certificate inherits
       the channel's own false positive). Either branch is the finding;
       the MUM fraction (X_cum <= 0.01) is reported with it.
  CC5  THE DIAL: the certificate's readings on the twins are
       dial-invariant in substance - the same organic rounds read the
       same C across BASE/WIDE/COMP/REREG within noise; the dial moves
       which rounds are armed, not what the ratio says about them;
       REREG's extended in-band rounds move R_org (disclosed).
  CC6  THE BAND'S WIDTH: min over U arms of C >= 3x the floor; max
       over G/J arms of C <= floor/5 - the separation margins that make
       0.25 honest.

Usage: python3 m3_catchcert.py [--smoke] [--group {atk,twin,all}]
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
BURN_EPOCHS, SELF_EPOCHS = 30, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
KICK_TAU, KICK_KAP = 0.75, 0.80        # doc 30's joint-kick values
ATTACK_STREAM = 977                     # the poison's separate rng offset
PANEL_EPOCHS = [20, 25, 30]             # doc 22's time-sliced frozen panel
REREG_K = 5                             # the re-registration period
COMP_REG_END = 5                        # the clean prefix (the slope window)

# ---------------- the certificate's frozen registration -------------------
CERT_FLOOR = 0.25     # the verdict band's floor (registered)
CERT_XMIN = 0.01      # the elevation gate: speak only above one point
CERT_RJUMP = 0.01     # the R_org jump guard (organic drift is slow)
ETA_RORG = 0.3        # the R_org EMA rate (the house rate)

# ---------------- the battery ----------------------------------------------
CCCFG = {
    # the 14-round attack arms - doc 41's PP configs verbatim
    "CC_TWIN": dict(lam=0.0,  mode="none",    eps_i=0.03, fix="none",
                    rounds=14),
    "CC_U05":  dict(lam=0.05, mode="uniform", eps_i=0.03, fix="none",
                    rounds=14),
    "CC_U10":  dict(lam=0.10, mode="uniform", eps_i=0.03, fix="none",
                    rounds=14),
    "CC_U30":  dict(lam=0.30, mode="uniform", eps_i=0.03, fix="none",
                    rounds=14),
    "CC_G05":  dict(lam=0.05, mode="gate",    eps_i=0.03, fix="none",
                    rounds=14),
    "CC_G10":  dict(lam=0.10, mode="gate",    eps_i=0.03, fix="none",
                    rounds=14),
    "CC_G30":  dict(lam=0.30, mode="gate",    eps_i=0.03, fix="none",
                    rounds=14),
    "CC_J05":  dict(lam=0.05, mode="joint",   eps_i=0.03, fix="none",
                    rounds=14),
    "CC_J10":  dict(lam=0.10, mode="joint",   eps_i=0.03, fix="none",
                    rounds=14),
    "CC_J30":  dict(lam=0.30, mode="joint",   eps_i=0.03, fix="none",
                    rounds=14),
    # the 30-round dial twins - doc 40's ID configs verbatim
    "CC_T30_BASE":  dict(lam=0.0, mode="none", eps_i=0.03, fix="none",
                         rounds=30),
    "CC_T30_WIDE":  dict(lam=0.0, mode="none", eps_i=0.05, fix="none",
                         rounds=30),
    "CC_T30_COMP":  dict(lam=0.0, mode="none", eps_i=0.03, fix="comp",
                         rounds=30),
    "CC_T30_REREG": dict(lam=0.0, mode="none", eps_i=0.03, fix="rereg",
                         rounds=30),
}
GROUPS = {
    "atk": ["CC_TWIN", "CC_U05", "CC_U10", "CC_U30",
            "CC_G05", "CC_G10", "CC_G30", "CC_J05", "CC_J10", "CC_J30"],
    "twin": ["CC_T30_BASE", "CC_T30_WIDE", "CC_T30_COMP", "CC_T30_REREG"],
}
DOC41 = "/home/z/my-project/scripts/m3_panelpoison_results.json"
DOC40 = "/home/z/my-project/scripts/m3_iota_dial_results.json"
AUDIT41 = [("CC_TWIN", "PP_TWIN"), ("CC_U05", "PP_U05"),
           ("CC_U10", "PP_U10"), ("CC_U30", "PP_U30"),
           ("CC_G05", "PP_G05"), ("CC_G10", "PP_G10"),
           ("CC_G30", "PP_G30"), ("CC_J05", "PP_J05"),
           ("CC_J10", "PP_J10"), ("CC_J30", "PP_J30")]
AUDIT40 = [("CC_T30_BASE", "ID_TWIN_BASE"), ("CC_T30_WIDE", "ID_TWIN_WIDE"),
           ("CC_T30_COMP", "ID_TWIN_COMP"), ("CC_T30_REREG", "ID_TWIN_REREG")]


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
    pbar = pbar_sum / (len(panel) * K)
    return pbar.argmax(1), pbar.max(1)


def panel_top2(X, panel):
    """The ATTACKER's read: the panel's top-2 classes and the top
    confidence. Same arithmetic as panel_grade (deterministic, no rng)."""
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


# ---------------- the attacks (doc 41's, verbatim) ------------------------
def poison_panel_aware(yt_, Xt_, panel, rng_attack, lam, mode):
    """Doc 41's attack modes verbatim: uniform (doc 34's Bernoulli flip,
    the bitwise anchor), run (same draws, runner-up direction), gate
    (deterministic sub-tau0 selection, uniform direction), joint (gated
    selection, runner-up direction). Returns (labels, effective mask)."""
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


# ---------------- the arm (doc 36/40/41's chassis + the certificate) -----
def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """The merged chassis: doc 41's attack arms and doc 40's dial twins
    under one runner, with the catch-rate certificate riding as passive
    telemetry. The certificate consumes no randomness and takes no
    action; every trajectory must replay its predecessor bitwise."""
    cfg = CCCFG[arm]
    lam, mode = cfg["lam"], cfg["mode"]
    eps_i, fix, rounds = cfg["eps_i"], cfg["fix"], cfg["rounds"]
    kicks = lam > 0.0
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
    yb0 = ybar0[would] if would.any() else ybar0
    iota_star = float(np.mean(pl0 != yb0))
    iota_sys = iota_star

    # ---- the certificate's own registration (dial-independent) ----------
    iota_star_cert = iota_star          # frozen; REREG never moves it
    if would.any():
        r_org0 = float(np.mean((pl0 != yb0) & (pc0 >= TAU0)))
    else:
        r_org0 = 0.0
    r_org = r_org0
    delta_hat = 0.0
    iota_hist = []
    cert_X, cert_N = [], []             # the armed-round accumulators

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, rounds + 1):
        # ---- 1. the attack: state kicks (records stay honest) -----------
        if r == PERT_ROUND and kicks:
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
        # ---- 2b. the channel's arming check (the dial axis) -------------
        if fix == "comp":
            baseline = iota_star + delta_hat * max(0, r - COMP_REG_END)
            reading_oob = bool(abs(iota_sys - baseline) > eps_i)
        else:
            reading_oob = bool(abs(iota_sys - iota_star) > eps_i)
        armed = reading_oob
        # ---- 3. grading, brake, THE POISON ------------------------------
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p, flipped = poison_panel_aware(
            yt_, Xt_, panel, rng_attack, lam if r >= PERT_ROUND else 0.0,
            mode)
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(flipped.sum())
        # ---- 4. the integrity channel: read, then act -------------------
        pl, pconf = panel_grade(Xt_, panel)
        d_recv = float(np.mean(pl != yt_p)) if len(yt_p) else 0.0
        disputed = (pl != yt_p) & (pconf >= TAU0)
        n_rep = int(disputed.sum())
        n_dis_raw = int(disputed.sum())     # would-fire, armed or not
        tele["n_flip_disputed"] = int((disputed & flipped).sum())
        tele["blind_mass"] = (float(np.mean(pconf < TAU0))
                              if len(pconf) else 0.0)
        tele["realized_dose"] = (float(np.mean(flipped))
                                 if len(flipped) else 0.0)
        if armed:
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

        # ---- 4b. THE CERTIFICATE (passive telemetry: no rng, no action) -
        n_com_r = len(yt_p)
        cert_base_r = iota_star_cert + delta_hat * max(0, r - COMP_REG_END)
        x_raw = d_recv - cert_base_r
        drate = (n_dis_raw / n_com_r) if n_com_r else 0.0
        n_raw = drate - r_org
        if armed and n_com_r:
            cert_X.append(x_raw)
            cert_N.append(n_raw)
        x_cum = float(np.mean(cert_X)) if cert_X else None
        n_cum = float(np.mean(cert_N)) if cert_N else None
        if armed and x_cum is not None and x_cum > CERT_XMIN:
            c_ratio = n_cum / x_cum
            cert_verdict = "CATCHING" if c_ratio >= CERT_FLOOR \
                else "EVASION"
        else:
            c_ratio = None
            cert_verdict = "MUM"
        # the organic baseline's guarded update (in-band AND slow-moving)
        rorg_refused = False
        if (not reading_oob) and n_com_r:
            if abs(drate - r_org) <= CERT_RJUMP:
                r_org = (1 - ETA_RORG) * r_org + ETA_RORG * drate
            else:
                rorg_refused = True

        # ---- 5. training on the (possibly cleaned) stream ---------------
        train_epochs(state, mom, rng, X_train, yt_train, SELF_EPOCHS)
        # ---- 6. calibration write channels (tau, kappa, iota) -----------
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
        iota_sys = (1 - ETA_I) * iota_sys + ETA_I * d_recv
        # ---- 6b. the dial's write-backs (comp/rereg; doc 40 verbatim) ---
        iota_hist.append(iota_sys)
        if fix == "comp" and r == COMP_REG_END:
            ys = np.array(iota_hist[:COMP_REG_END])
            xs = np.arange(1, COMP_REG_END + 1)
            delta_hat = float(np.polyfit(xs, ys, 1)[0])
        if fix == "rereg" and r % REREG_K == 0:
            if abs(iota_sys - iota_star) <= eps_i:
                iota_star = float(iota_sys)
                tele["rereg_fired"] = True
        # ---- 7. telemetry -------------------------------------------------
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
            "eps_i": eps_i, "fix": fix, "delta_hat": delta_hat,
            "n_disputed_repaired": n_rep,
            "n_dis_raw": n_dis_raw,
            "cert_base": cert_base_r, "x_raw": x_raw, "n_raw": n_raw,
            "x_cum": x_cum, "n_cum": n_cum, "c_ratio": c_ratio,
            "cert_verdict": cert_verdict, "r_org": r_org,
            "rorg_refused": rorg_refused,
            "acc": float(np.mean(ybar_b == yte)),
            "accept09": float(np.mean(conf_b >= TAU0)),
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


def audit(res, ref_path, pairs, rounds_ref):
    """Bitwise audits vs a predecessor's stored aggregate trajectories."""
    try:
        ref = json.load(open(ref_path))
    except FileNotFoundError as e:
        print(f"audit unavailable: {e}")
        return {}
    out = {}
    for ours, theirs in pairs:
        if ours not in res["arms"]:
            continue
        ref_t = ref["arms"][theirs]["aggregate"]["tau_sys"][:rounds_ref]
        ref_k = ref["arms"][theirs]["aggregate"]["kappa_sys"][:rounds_ref]
        got_t = [float(np.mean([rr["traj"][i]["tau_sys"]
                                for rr in res["arms"][ours]["runs"]]))
                 for i in range(rounds_ref)]
        got_k = [float(np.mean([rr["traj"][i]["kappa_sys"]
                                for rr in res["arms"][ours]["runs"]]))
                 for i in range(rounds_ref)]
        ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref_t, got_t))
                  and all(abs(a - b) < 1e-9
                          for a, b in zip(ref_k, got_k)))
        out[f"{ours}_reproduces_{theirs}"] = ok
        print(f"{ours} reproduces {theirs} bitwise: {ok}")
    return out


def classify(res):
    """The pre-registered CC1-CC6 classification."""
    cls = {}
    A = res["arms"]

    def ag(arm, key):
        return A[arm]["aggregate"][key]

    def runs_traj(arm):
        return A[arm]["runs"]

    # CC1: the innocence audits
    cls["CC1_innocence"] = dict(res.get("audits41", {}))
    cls["CC1_innocence"].update(res.get("audits40", {}))

    # CC2/CC3: the honest catch and the evasion (final round, per arm)
    cc23 = {}
    for arm in GROUPS["atk"]:
        if arm == "CC_TWIN":
            continue
        rows = runs_traj(arm)
        fin_c = [rr["traj"][-1]["c_ratio"] for rr in rows]
        fin_c = [c for c in fin_c if c is not None]
        verdicts = [rr["traj"][-1]["cert_verdict"] for rr in rows]
        n_armed = int(sum(1 for rr in rows if rr["traj"][-1]["armed"]))
        cc23[arm] = {
            "final_C_per_seed": fin_c,
            "final_C_mean": float(np.mean(fin_c)) if fin_c else None,
            "final_verdicts": verdicts,
            "n_armed_seeds_final": n_armed,
            "CATCHING": verdicts.count("CATCHING"),
            "EVASION": verdicts.count("EVASION"),
            "MUM": verdicts.count("MUM")}
    cls["CC2_CC3_readings"] = cc23
    u_cs = [v["final_C_mean"] for k, v in cc23.items()
            if k.startswith("CC_U") and v["final_C_mean"] is not None]
    gj_cs = [v["final_C_mean"] for k, v in cc23.items()
             if (k.startswith("CC_G") or k.startswith("CC_J"))
             and v["final_C_mean"] is not None]
    cls["CC2_honest_catch"] = {
        "min_U_C": min(u_cs) if u_cs else None,
        "band": "[0.35, 0.90]",
        "confirmed": bool(u_cs and 0.35 <= min(u_cs)
                          and max(u_cs) <= 0.90)}
    cls["CC3_evasion"] = {
        "max_GJ_C": max(gj_cs) if gj_cs else None,
        "confirmed": bool(gj_cs and max(gj_cs) < 0.05)}

    # CC4: the false-positive fork on the 30-round twins
    cc4 = {}
    for arm in GROUPS["twin"]:
        rows = runs_traj(arm)
        per_seed = []
        for rr in rows:
            armed_rounds = [t for t in rr["traj"] if t["armed"]]
            n_evade = sum(1 for t in armed_rounds
                          if t["cert_verdict"] == "EVASION")
            n_catch = sum(1 for t in armed_rounds
                          if t["cert_verdict"] == "CATCHING")
            n_mum = sum(1 for t in armed_rounds
                        if t["cert_verdict"] == "MUM")
            fin = rr["traj"][-1]
            per_seed.append({
                "first_armed": armed_rounds[0]["round"]
                if armed_rounds else None,
                "armed_rounds": len(armed_rounds),
                "verdicts_CATCHING": n_catch, "verdicts_EVASION": n_evade,
                "verdicts_MUM": n_mum,
                "final_C": fin["c_ratio"],
                "final_verdict": fin["cert_verdict"],
                "final_x_cum": fin["x_cum"],
                "final_r_org": fin["r_org"]})
        tot_evade = sum(s["verdicts_EVASION"] for s in per_seed)
        tot_armed = sum(s["armed_rounds"] for s in per_seed)
        cc4[arm] = {
            "per_seed": per_seed,
            "total_armed_rounds": tot_armed,
            "total_EVASION_rounds": tot_evade,
            "fork": ("(b) false-evasion alarms" if tot_evade > 0
                     else "(a) no false evasion (organic reads "
                          "CATCHING or MUM)")}
    cls["CC4_false_positive_fork"] = cc4

    # CC5: the dial's substance (C on common armed rounds across dials)
    base_rows = runs_traj("CC_T30_BASE")
    cc5 = {}
    for arm in ["CC_T30_WIDE", "CC_T30_COMP", "CC_T30_REREG"]:
        rows = runs_traj(arm)
        diffs = []
        for rr_b, rr_d in zip(base_rows, rows):
            cb = {t["round"]: t["c_ratio"] for t in rr_b["traj"]
                  if t["c_ratio"] is not None}
            cd = {t["round"]: t["c_ratio"] for t in rr_d["traj"]
                  if t["c_ratio"] is not None}
            common = sorted(set(cb) & set(cd))
            diffs += [abs(cb[r] - cd[r]) for r in common]
        cc5[arm] = {
            "n_common_C_rounds": len(diffs),
            "max_abs_diff": max(diffs) if diffs else None,
            "mean_abs_diff": float(np.mean(diffs)) if diffs else None}
    cls["CC5_dial_substance"] = cc5

    # CC6: the band's width
    cls["CC6_band_width"] = {
        "floor": CERT_FLOOR,
        "min_U_over_floor": (min(u_cs) / CERT_FLOOR
                             if u_cs else None),
        "max_GJ_over_floor": (max(gj_cs) / CERT_FLOOR
                              if gj_cs else None),
        "margin_honest_ok": bool(u_cs and min(u_cs) >= 3 * CERT_FLOOR),
        "margin_evasion_ok": bool(gj_cs and max(gj_cs)
                                  <= CERT_FLOOR / 5.0)}
    return cls


def main():
    smoke = "--smoke" in sys.argv
    group = "all"
    for a in sys.argv:
        if a.startswith("--group="):
            group = a.split("=")[1]
        elif a == "--group":
            pass
    if smoke:
        globals()["ROUNDS"] = 6
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        globals()["CCCFG"]["CC_TWIN"]["rounds"] = 6
        globals()["CCCFG"]["CC_U30"]["rounds"] = 6
        globals()["CCCFG"]["CC_J30"]["rounds"] = 6
        globals()["CCCFG"]["CC_T30_BASE"]["rounds"] = 8
        globals()["CCCFG"]["CC_T30_REREG"]["rounds"] = 8
        sel = ["CC_TWIN", "CC_U30", "CC_J30", "CC_T30_BASE",
               "CC_T30_REREG"]
        globals()["GROUPS"]["atk"] = ["CC_TWIN", "CC_U30", "CC_J30"]
        globals()["GROUPS"]["twin"] = ["CC_T30_BASE", "CC_T30_REREG"]
        group = "all"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/m3_catchcert_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/m3_catchcert_results.json")
    # resume semantics: load existing full results, update this group
    res = None
    if not smoke:
        try:
            res = json.load(open(out_path))
            print(f"resuming: {len(res['arms'])} arms already present")
        except FileNotFoundError:
            res = None
    if res is None:
        res = {"config": {
            "tau0": TAU0, "kappa0": KAPPA0, "eps_i_base": EPS_I,
            "eta_i": ETA_I, "rounds_14": 14, "rounds_30": 30,
            "pert_round": PERT_ROUND, "nseed": len(SEEDS),
            "cert_floor": CERT_FLOOR, "cert_xmin": CERT_XMIN,
            "cert_rjump": CERT_RJUMP, "eta_rorg": ETA_RORG,
            "comp_reg_end": COMP_REG_END, "rereg_k": REREG_K,
            "attack_stream_offset": ATTACK_STREAM,
            "panel_epochs": PANEL_EPOCHS,
            "numpy": np.__version__},
            "arms": {}}

    if group == "all":
        arms = GROUPS["atk"] + GROUPS["twin"]
    elif group in GROUPS:
        arms = GROUPS[group]
    else:
        arms = GROUPS["atk"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise audits ---------------------------------------------------
    if not smoke:
        have41 = [a for a, _ in AUDIT41 if a in res["arms"]]
        if len(have41) == len(AUDIT41):
            res["audits41"] = audit(res, DOC41, AUDIT41, 14)
        have40 = [a for a, _ in AUDIT40 if a in res["arms"]]
        if len(have40) == len(AUDIT40):
            res["audits40"] = audit(res, DOC40, AUDIT40, 30)

    # ---- aggregates -------------------------------------------------------
    for arm in res["arms"]:
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
                "realized_dose", "x_raw", "n_raw", "x_cum", "n_cum",
                "c_ratio", "r_org", "cert_base", "n_dis_raw",
                "n_disputed_repaired", "n_flipped", "n_flip_disputed"]
        for key in keys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in ["repair_fired", "repair_fired_kappa", "brake",
                     "armed", "reading_oob", "rorg_refused"]:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        # the verdict trajectory: majority vote per round
        vagg = []
        for i in range(len(rows[0]["traj"])):
            vs = [r["traj"][i].get("cert_verdict", "MUM") for r in rows]
            vagg.append(max(set(vs), key=vs.count))
        agg["cert_verdict"] = vagg
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:14s} tau: {['%.3f' % a for a in agg['tau_sys'][:6]]}"
              f"...")
        print(f"{'':14s} C:   {['%.3f' % c if c is not None else 'nan'
                              for c in agg['c_ratio']]}")
        print(f"{'':14s} verdict: {agg['cert_verdict']}")

    # ---- classification ---------------------------------------------------
    if not smoke:
        need = set(GROUPS["atk"] + GROUPS["twin"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("CC2:", c["CC2_honest_catch"])
            print("CC3:", c["CC3_evasion"])
            for arm, row in c["CC4_false_positive_fork"].items():
                print(f"CC4 {arm}: {row['fork']} "
                      f"(armed {row['total_armed_rounds']}, "
                      f"evasion {row['total_EVASION_rounds']})")
            print("CC6:", c["CC6_band_width"])

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
