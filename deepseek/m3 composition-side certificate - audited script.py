#!/usr/bin/env python3
"""
THE COMPOSITION-SIDE CERTIFICATE - doc 43's filed dial, built.
Corpus doc 47's experiment. Ordered at the catch-certificate issuance:
"the composition-side certificate (doc 43's filed dial: the blind-mass
trajectory as a registered reading with its own band, calibrated against
the mixed attacker - the quantity the gated attack must consume and the
organic drift must fill; the REREG twin's silent 0.10 is its
false-positive question)."

THE FILING, AS INHERITED (doc 43, Part 3 and the coda): the catch
certificate detects COMPOSITION (the blind-weighted fraction of
disagreement), not attackers - one territory, three residents - and what
would close the laundering cell is "composition-side telemetry the channel
already holds and does not yet read as a certificate - the blind-mass
trajectory itself, the very quantity the gated attacker must consume."
Doc 44 answered with the calibration adversary: the mix launders its
blind dirt under its own honest catches (C = 0.834 CATCHING while 16% of
the dose evades wholly). This battery builds the second coordinate.

THE READING (pre-registered before the run; calibrated on the
predecessors' stored telemetry - docs 40/41/44 - the corpus's standing
calibration convention):
  b(r)     the commit stream's blind mass: mean(panel conf < tau0 over
           committed items) - the telemetry's existing reading, no new
           computation on the world.
  b*       the registered level: b at round 5, the clean prefix's last
           reading (PERT_ROUND = 6 convention; the kicked round 6 is the
           kick's own composition event - the widened gate's blind-heavy
           cohort, 0.187 in every attack arm - and is EXCLUDED from the
           reading window, disclosed: the onset convention is round 7).
  F(r)     the fill: b(r) - b*, computed EVERY ROUND regardless of the
           channel's arming - ALWAYS-ON, the REREG lesson made structural
           (doc 43: an instrument whose speech is gated on another
           instrument's arming inherits that instrument's blind
           schedules).
  F_late   the verdict window: mean of F over the last 4 rounds of the
           horizon (r11-14 for the attack arms, r27-30 for the twins).
  F_bar    the course reading: mean of F over rounds 7..end.

THE REGISTERED BAND on F_late (calibrated: the healthy fill runs
+0.029..+0.062, the attacks at or below +0.012; the REREG runaway
+0.109 against the BASE plateau +0.062; the G arms -0.039..-0.043):
  F_late >  0.08   OVERFILLED  (the drift running un-repaired)
  F_late >= 0.02   ON-COURSE   (the healthy fill)
  F_late >=-0.02   CONSUMED    (the territory held flat or mildly down)
  F_late < -0.02   BURNED      (the territory crashing)
The verdict names are composition-semantic - no attacker language (doc
43's lesson: the reading names the territory's state, not its residents'
intent).

THE BATTERY (doc 36/41/44's chassis in every particular; both
certificates ride as passive telemetry - no randomness consumed, no
action taken; every world must replay its predecessor bitwise):
  the 10 doc-41 attack arms (audit bitwise to PP_*):
    CM_TWIN, CM_U05/10/30, CM_G05/10/30, CM_J05/10/30
  the 6 non-corner doc-44 mixed arms (audit bitwise to MX_*):
    CM_CAP30, CM_HALF30, CM_CAP10, CM_HALF10, CM_CAP30_NOCH,
    CM_CAP10_NOCH   (the corners are the atk group's TWIN/U30/J30/J10)
  the 4 doc-40 dial twins, 30 rounds (audit bitwise to ID_*):
    CM_T30_BASE / CM_T30_WIDE / CM_T30_COMP / CM_T30_REREG

PRE-REGISTERED PREDICTIONS (fixed before execution):
  CM1  THE INNOCENCE AUDITS: all 20 arms reproduce their predecessors
       bitwise (tau/kappa, 1e-9) - the certificates are readings, not
       acts.
  CM2  THE FILL: the healthy worlds' F_late >= 0.02 (ON-COURSE) - the
       twin, the 30-round BASE, and fork: WIDE/COMP (their later arming
       leaves the drift longer unfixed - ON-COURSE or OVERFILLED, both
       branches informative); the low-dose uniform arms fork ON-COURSE
       (the territory intact under a mild loud attack) vs CONSUMED.
  CM3  THE CONSUMPTION: the gated/joint arms read CONSUMED
       (-0.02 <= F_late < 0.02) - the farm holds the territory flat
       while the healthy course fills; THE PAIR breaks the aliasing doc
       43 measured: healthy-armed reads (false-EVASION, ON-COURSE), the
       gated attack reads (EVASION, CONSUMED).
  CM4  THE BURNING: the G arms read BURNED (F_late < -0.02) at every
       dose - the arbitrary direction's footprint, doc 41's G-finding
       now a registered in-channel reading (G burns where J consumes).
  CM5  THE ALWAYS-ON READING: the REREG twin reads OVERFILLED
       (F_late > 0.08) on the rounds the catch certificate was silent
       (5 armed of 30; the certificate speaks when the channel sleeps);
       the BASE twin stays ON-COURSE. THE FALSE-POSITIVE QUESTION
       RESOLVED AS: the runaway fill is flagged - an honest composition
       reading of a real condition (the drift un-repaired, the record
       walking), not a false alarm; the repaired drift's plateau is not.
  CM6  THE PAIR SIGNATURE TABLE: the five world-classes separate on
       (catch verdict, composition verdict): healthy-unarmed (MUM,
       ON-COURSE), healthy-armed (EVASION-fp, ON-COURSE), gated
       (EVASION, CONSUMED), uniform (CATCHING, ON-COURSE or CONSUMED by
       dose), mixed (CATCHING, BURNED).
  CM7  THE LAUNDERING: at equal total dose the mixes' F_late falls
       below the pure uniform's (the stealth half's consumption adds to
       the loud half's burn - the coverage question's two coordinates);
       the boundary cells CAP10/HALF10 read CONSUMED.
  CM8  THE BAND'S MARGINS: the fill floor's two-sided separation, the
       overfill gap (REREG vs BASE), the burn gap (the mixes vs the J
       arms), and the loop world's reading (CAP30_NOCH) - all reported
       as measurements, the floor unmoved by any single margin.

Usage: python3 m3_compcert.py [--smoke] [--group {atk,mix,twin,all}]
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
COMP_REG_END = 5                        # the clean prefix (b* registered)

# ------------- the catch certificate's frozen registration -----------------
CERT_FLOOR = 0.25     # the verdict band's floor (doc 43, registered)
CERT_XMIN = 0.01      # the elevation gate: speak only above one point
CERT_RJUMP = 0.01     # the R_org jump guard (organic drift is slow)
ETA_RORG = 0.3        # the R_org EMA rate (the house rate)

# -------- the composition certificate's frozen registration ----------------
COMP_FILL_FLOOR = 0.02    # ON-COURSE floor (the healthy fill clears)
COMP_OVER_CEIL = 0.08     # OVERFILLED ceiling (the runaway fill)
COMP_BURN_FLOOR = -0.02   # BURNED floor (the territory crashing)
COMP_ONSET = 7            # the reading onset (the kicked round excluded)
COMP_LATE_WIN = 4         # the verdict window (the horizon's last 4)

# ---------------- the battery ----------------------------------------------
CMCFG = {
    # the 10 doc-41 attack arms - PP configs verbatim
    "CM_TWIN": dict(bulk=0.0,  gfrac=0.0, mode="none",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_U05":  dict(bulk=0.05, gfrac=0.0, mode="uniform",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_U10":  dict(bulk=0.10, gfrac=0.0, mode="uniform",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_U30":  dict(bulk=0.30, gfrac=0.0, mode="uniform",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_G05":  dict(bulk=0.05, gfrac=0.0, mode="gate",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_G10":  dict(bulk=0.10, gfrac=0.0, mode="gate",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_G30":  dict(bulk=0.30, gfrac=0.0, mode="gate",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_J05":  dict(bulk=0.05, gfrac=0.0, mode="joint",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_J10":  dict(bulk=0.10, gfrac=0.0, mode="joint",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_J30":  dict(bulk=0.30, gfrac=0.0, mode="joint",
                    eps_i=0.03, fix="none", noch=False, rounds=14),
    # the 6 non-corner doc-44 mixed arms - MX configs verbatim
    "CM_CAP30":      dict(bulk=0.22, gfrac=0.08, mode="mixed",
                          eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_HALF30":     dict(bulk=0.15, gfrac=0.15, mode="mixed",
                          eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_CAP10":      dict(bulk=0.02, gfrac=0.08, mode="mixed",
                          eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_HALF10":     dict(bulk=0.05, gfrac=0.05, mode="mixed",
                          eps_i=0.03, fix="none", noch=False, rounds=14),
    "CM_CAP30_NOCH": dict(bulk=0.22, gfrac=0.08, mode="mixed",
                          eps_i=0.03, fix="none", noch=True, rounds=14),
    "CM_CAP10_NOCH": dict(bulk=0.02, gfrac=0.08, mode="mixed",
                          eps_i=0.03, fix="none", noch=True, rounds=14),
    # the 4 doc-40 dial twins - ID configs verbatim
    "CM_T30_BASE":  dict(bulk=0.0, gfrac=0.0, mode="none", eps_i=0.03,
                         fix="none", noch=False, rounds=30),
    "CM_T30_WIDE":  dict(bulk=0.0, gfrac=0.0, mode="none", eps_i=0.05,
                         fix="none", noch=False, rounds=30),
    "CM_T30_COMP":  dict(bulk=0.0, gfrac=0.0, mode="none", eps_i=0.03,
                         fix="comp", noch=False, rounds=30),
    "CM_T30_REREG": dict(bulk=0.0, gfrac=0.0, mode="none", eps_i=0.03,
                         fix="rereg", noch=False, rounds=30),
}
GROUPS = {
    "atk": ["CM_TWIN", "CM_U05", "CM_U10", "CM_U30",
            "CM_G05", "CM_G10", "CM_G30", "CM_J05", "CM_J10", "CM_J30"],
    "mix": ["CM_CAP30", "CM_HALF30", "CM_CAP10", "CM_HALF10",
            "CM_CAP30_NOCH", "CM_CAP10_NOCH"],
    "twin": ["CM_T30_BASE", "CM_T30_WIDE", "CM_T30_COMP", "CM_T30_REREG"],
}
DOC41 = "/home/z/my-project/scripts/m3_panelpoison_results.json"
DOC44 = "/home/z/my-project/scripts/m3_mixed_results.json"
DOC40 = "/home/z/my-project/scripts/m3_iota_dial_results.json"
AUDIT41 = [("CM_TWIN", "PP_TWIN"), ("CM_U05", "PP_U05"),
           ("CM_U10", "PP_U10"), ("CM_U30", "PP_U30"),
           ("CM_G05", "PP_G05"), ("CM_G10", "PP_G10"),
           ("CM_G30", "PP_G30"), ("CM_J05", "PP_J05"),
           ("CM_J10", "PP_J10"), ("CM_J30", "PP_J30")]
AUDIT44 = [("CM_CAP30", "MX_CAP30"), ("CM_HALF30", "MX_HALF30"),
           ("CM_CAP10", "MX_CAP10"), ("CM_HALF10", "MX_HALF10"),
           ("CM_CAP30_NOCH", "MX_CAP30_NOCH"),
           ("CM_CAP10_NOCH", "MX_CAP10_NOCH")]
AUDIT40 = [("CM_T30_BASE", "ID_TWIN_BASE"), ("CM_T30_WIDE", "ID_TWIN_WIDE"),
           ("CM_T30_COMP", "ID_TWIN_COMP"), ("CM_T30_REREG", "ID_TWIN_REREG")]


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


# ------------- the attacks (docs 41/44's, verbatim semantics) -------------
def poison_all(yt_, Xt_, panel, rng_attack, lam_bulk, gfrac, mode):
    """The unified attack surface. rng consumption is bitwise-critical:
    'uniform'/'mixed' consume one rng_attack.random(n) draw block per
    round (doc 34/41/44's anchor); 'gate'/'joint' consume NOTHING (the
    deterministic sub-tau0 ranking). No-op flips excluded from masks.
    Returns (labels, bulk_mask, gate_mask)."""
    n = len(yt_)
    if n == 0 or (lam_bulk <= 0.0 and gfrac <= 0.0) or mode == "none":
        return yt_.copy(), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    out = yt_.copy()
    bulk = np.zeros(n, dtype=bool)
    gate = np.zeros(n, dtype=bool)
    if mode == "uniform":
        mask = rng_attack.random(n) < lam_bulk
        out[mask] = (out[mask] + 1) % 10
        bulk = mask
    elif mode == "gate":
        _, run2, conf = panel_top2(Xt_, panel)
        blind = np.where(conf < TAU0)[0]
        m = int(round(lam_bulk * n))
        order = blind[np.argsort(-conf[blind], kind="stable")]
        sel = order[:m]
        new = (out[sel] + 1) % 10
        ch = new != out[sel]
        out[sel[ch]] = new[ch]
        gate[sel[ch]] = True
    elif mode == "joint":
        _, run2, conf = panel_top2(Xt_, panel)
        blind = np.where(conf < TAU0)[0]
        m = int(round(lam_bulk * n))
        order = blind[np.argsort(-conf[blind], kind="stable")]
        sel = order[:m]
        new = run2[sel]
        ch = new != out[sel]
        out[sel[ch]] = new[ch]
        gate[sel[ch]] = True
    elif mode == "mixed":
        # doc 44's poison_mixed verbatim: bulk first (the uniform draws),
        # then the deterministic runner-up fill on the remaining blind
        if lam_bulk > 0.0:
            draws = rng_attack.random(n)
            sel = draws < lam_bulk
            out[sel] = (out[sel] + 1) % 10
            bulk = sel
        if gfrac > 0.0:
            _, run2, conf = panel_top2(Xt_, panel)
            blind = np.where((conf < TAU0) & ~bulk)[0]
            m = int(round(gfrac * n))
            order = blind[np.argsort(-conf[blind], kind="stable")]
            sel = order[:m]
            new = run2[sel]
            ch = new != out[sel]
            out[sel[ch]] = new[ch]
            gate[sel[ch]] = True
    else:
        raise ValueError(f"unknown mode {mode}")
    return out, bulk, gate


def tension(p_star, live):
    return ((1 - ETA) * RHO_T * p_star + ETA * live) / (ETA + RHO_T - ETA * RHO_T)


# ------- the arm (docs 41/43/44 chassis + BOTH certificates) --------------
def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """The merged chassis: doc 41's attack arms, doc 44's mixed arms, and
    doc 40's dial twins under one runner, with the catch certificate AND
    the composition certificate riding as passive telemetry. Neither
    certificate consumes randomness or takes action; every trajectory
    must replay its predecessor bitwise."""
    cfg = CMCFG[arm]
    lam_bulk, gfrac, mode = cfg["bulk"], cfg["gfrac"], cfg["mode"]
    eps_i, fix, rounds = cfg["eps_i"], cfg["fix"], cfg["rounds"]
    noch = cfg["noch"]
    lam = lam_bulk + gfrac            # the filed total dose (for kicks)
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

    # ---- the catch certificate's own registration (doc 43, verbatim) ----
    iota_star_cert = iota_star          # frozen; REREG never moves it
    if would.any():
        r_org0 = float(np.mean((pl0 != yb0) & (pc0 >= TAU0)))
    else:
        r_org0 = 0.0
    r_org = r_org0
    delta_hat = 0.0
    iota_hist = []
    cert_X, cert_N = [], []             # the armed-round accumulators

    # ---- the composition certificate's registration ----------------------
    b_star = None                       # registered at round 5 (the prefix)

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
        armed = reading_oob and not noch
        # ---- 3. grading, brake, THE POISON ------------------------------
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p, bulk_m, gate_m = poison_all(
            yt_, Xt_, panel, rng_attack,
            lam_bulk if r >= PERT_ROUND else 0.0,
            gfrac if r >= PERT_ROUND else 0.0, mode)
        flipped = bulk_m | gate_m
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(flipped.sum())
        tele["n_bulk_flipped"] = int(bulk_m.sum())
        tele["n_gate_flipped"] = int(gate_m.sum())
        # ---- 4. the integrity channel: read, then act -------------------
        pl, pconf = panel_grade(Xt_, panel)
        d_recv = float(np.mean(pl != yt_p)) if len(yt_p) else 0.0
        disputed = (pl != yt_p) & (pconf >= TAU0)
        n_rep = int(disputed.sum())
        n_dis_raw = int(disputed.sum())     # would-fire, armed or not
        tele["n_flip_disputed"] = int((disputed & flipped).sum())
        b_r = (float(np.mean(pconf < TAU0))
               if len(pconf) else 0.0)
        tele["blind_mass"] = b_r
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

        # ---- 4b. THE CATCH CERTIFICATE (doc 43, verbatim; passive) -----
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

        # ---- 4c. THE COMPOSITION CERTIFICATE (passive; ALWAYS-ON) ------
        # b* registered on the clean prefix's last round; F computed every
        # round thereafter; the verdict NEVER consults the channel's
        # arming - the REREG lesson made structural.
        if r == COMP_REG_END:
            b_star = b_r
        comp_F = (b_r - b_star) if (b_star is not None) else None

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
            "variant": "NOCH" if noch else "REPAIR",
            "n_disputed_repaired": n_rep,
            "n_dis_raw": n_dis_raw,
            "cert_base": cert_base_r, "x_raw": x_raw, "n_raw": n_raw,
            "x_cum": x_cum, "n_cum": n_cum, "c_ratio": c_ratio,
            "cert_verdict": cert_verdict, "r_org": r_org,
            "rorg_refused": rorg_refused,
            "comp_bstar": b_star, "comp_F": comp_F,
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


# ---------------- the composition verdict (registered) --------------------
def comp_verdict(f_late):
    if f_late is None:
        return "MUM"
    if f_late > COMP_OVER_CEIL:
        return "OVERFILLED"
    if f_late >= COMP_FILL_FLOOR:
        return "ON-COURSE"
    if f_late >= COMP_BURN_FLOOR:
        return "CONSUMED"
    return "BURNED"


def seed_composition(rr):
    """The per-seed composition reading: F_bar (course) and F_late (the
    verdict window) from the stored trajectory."""
    rounds = len(rr["traj"])
    Fs = [t["comp_F"] for t in rr["traj"]
          if t["comp_F"] is not None and t["round"] >= COMP_ONSET]
    late = [t["comp_F"] for t in rr["traj"]
            if t["comp_F"] is not None
            and t["round"] >= rounds - COMP_LATE_WIN + 1]
    f_bar = float(np.mean(Fs)) if Fs else None
    f_late = float(np.mean(late)) if late else None
    return f_bar, f_late, comp_verdict(f_late)


def classify(res):
    """The pre-registered CM1-CM8 classification."""
    cls = {}
    A = res["arms"]

    def runs_traj(arm):
        return A[arm]["runs"]

    # CM1: the innocence audits
    cls["CM1_innocence"] = dict(res.get("audits41", {}))
    cls["CM1_innocence"].update(res.get("audits44", {}))
    cls["CM1_innocence"].update(res.get("audits40", {}))

    # the per-arm composition + catch readings
    comp = {}
    for arm in A:
        rows = runs_traj(arm)
        per_seed = []
        for rr in rows:
            f_bar, f_late, v = seed_composition(rr)
            fin = rr["traj"][-1]
            per_seed.append({
                "F_bar": f_bar, "F_late": f_late,
                "comp_verdict": v,
                "catch_verdict": fin["cert_verdict"],
                "final_C": fin["c_ratio"],
                "armed_rounds": sum(1 for t in rr["traj"] if t["armed"]),
                "final_blind_mass": fin["blind_mass"],
                "final_acc": fin["acc"]})
        f_lates = [s["F_late"] for s in per_seed if s["F_late"] is not None]
        comp[arm] = {
            "per_seed": per_seed,
            "F_late_mean": float(np.mean(f_lates)) if f_lates else None,
            "comp_verdicts": [s["comp_verdict"] for s in per_seed],
            "majority_comp_verdict": (max(set(
                [s["comp_verdict"] for s in per_seed]),
                key=[s["comp_verdict"] for s in per_seed].count)
                if per_seed else None),
            "majority_catch_verdict": (max(set(
                [s["catch_verdict"] for s in per_seed]),
                key=[s["catch_verdict"] for s in per_seed].count)
                if per_seed else None)}
    cls["composition_readings"] = comp

    def fmean(arm):
        v = comp[arm]["F_late_mean"]
        return v if v is not None else 0.0

    # CM2: the fill (healthy worlds)
    healthy = ["CM_TWIN", "CM_T30_BASE", "CM_T30_WIDE", "CM_T30_COMP"]
    cls["CM2_fill"] = {
        arm: {"F_late": fmean(arm),
              "verdict": comp[arm]["majority_comp_verdict"]}
        for arm in healthy if arm in comp}
    cls["CM2_fill"]["fork_U05_U10"] = {
        arm: {"F_late": fmean(arm),
              "verdict": comp[arm]["majority_comp_verdict"]}
        for arm in ["CM_U05", "CM_U10"] if arm in comp}
    hs = [fmean(a) for a in healthy if a in comp]
    cls["CM2_fill"]["min_healthy_F_late"] = min(hs) if hs else None
    cls["CM2_fill"]["floor"] = COMP_FILL_FLOOR
    cls["CM2_fill"]["confirmed"] = bool(
        hs and min(hs) >= COMP_FILL_FLOOR)

    # CM3: the consumption (gated/joint)
    gj = ["CM_J05", "CM_J10", "CM_J30"]
    cls["CM3_consumption"] = {
        arm: {"F_late": fmean(arm),
              "verdict": comp[arm]["majority_comp_verdict"]}
        for arm in gj if arm in comp}
    # the aliasing break: healthy-armed (EVASION, ON-COURSE) vs gated
    # (EVASION, CONSUMED)
    pairs = {}
    for arm in ["CM_T30_BASE", "CM_T30_WIDE", "CM_T30_COMP",
                "CM_T30_REREG", "CM_TWIN"]:
        if arm in comp:
            # the catch verdict on ARMED healthy rounds (per-seed finals)
            cvs = [s["catch_verdict"] for s in comp[arm]["per_seed"]]
            pairs[arm] = {
                "catch": max(set(cvs), key=cvs.count),
                "composition": comp[arm]["majority_comp_verdict"]}
    for arm in gj:
        if arm in comp:
            cvs = [s["catch_verdict"] for s in comp[arm]["per_seed"]]
            pairs[arm] = {
                "catch": max(set(cvs), key=cvs.count),
                "composition": comp[arm]["majority_comp_verdict"]}
    cls["CM3_aliasing_pairs"] = pairs
    gs = [fmean(a) for a in gj if a in comp]
    cls["CM3_consumption"]["confirmed"] = bool(
        gs and all(COMP_BURN_FLOOR <= g < COMP_FILL_FLOOR for g in gs))

    # CM4: the burning (the arbitrary direction)
    g_arms = ["CM_G05", "CM_G10", "CM_G30"]
    cls["CM4_burning"] = {
        arm: {"F_late": fmean(arm),
              "verdict": comp[arm]["majority_comp_verdict"]}
        for arm in g_arms if arm in comp}
    gs4 = [fmean(a) for a in g_arms if a in comp]
    cls["CM4_burning"]["confirmed"] = bool(
        gs4 and all(g < COMP_BURN_FLOOR for g in gs4))

    # CM5: the always-on reading (REREG vs BASE)
    cm5 = {}
    for arm in ["CM_T30_REREG", "CM_T30_BASE"]:
        if arm in comp:
            rr = runs_traj(arm)
            armed_rounds = sum(sum(1 for t in x["traj"] if t["armed"])
                               for x in rr)
            cm5[arm] = {
                "F_late": fmean(arm),
                "verdict": comp[arm]["majority_comp_verdict"],
                "total_armed_seed_rounds": armed_rounds,
                "silent_under_catch": armed_rounds}
    if "CM_T30_REREG" in cm5 and "CM_T30_BASE" in cm5:
        cm5["overfill_gap"] = (cm5["CM_T30_REREG"]["F_late"]
                               - cm5["CM_T30_BASE"]["F_late"])
        cm5["confirmed"] = bool(
            cm5["CM_T30_REREG"]["F_late"] > COMP_OVER_CEIL
            and cm5["CM_T30_BASE"]["F_late"] <= COMP_OVER_CEIL)
    cls["CM5_always_on"] = cm5

    # CM6: the pair-signature table (the five world-classes)
    sig = {}
    classes = {
        "healthy_unarmed": ["CM_TWIN", "CM_T30_REREG"],
        "healthy_armed": ["CM_T30_BASE", "CM_T30_WIDE", "CM_T30_COMP"],
        "gated_attack": ["CM_J05", "CM_J10", "CM_J30"],
        "uniform_attack": ["CM_U05", "CM_U10", "CM_U30"],
        "mixed_attack": ["CM_CAP30", "CM_HALF30", "CM_CAP10", "CM_HALF10"],
    }
    for cl, arms in classes.items():
        rows = []
        for arm in arms:
            if arm not in comp:
                continue
            cvs = [s["catch_verdict"] for s in comp[arm]["per_seed"]]
            rows.append({
                "arm": arm,
                "catch": max(set(cvs), key=cvs.count),
                "composition": comp[arm]["majority_comp_verdict"],
                "F_late": fmean(arm)})
        sig[cl] = rows
    cls["CM6_pair_signatures"] = sig

    # CM7: the laundering (mixes burn below the pure uniform at dose)
    cm7 = {}
    for mix, uni in [("CM_CAP30", "CM_U30"), ("CM_HALF30", "CM_U30"),
                     ("CM_CAP10", "CM_U10"), ("CM_HALF10", "CM_U10")]:
        if mix in comp and uni in comp:
            cm7[f"{mix}_vs_{uni}"] = {
                "mix_F_late": fmean(mix), "uniform_F_late": fmean(uni),
                "mix_burns_more": bool(fmean(mix) < fmean(uni))}
    for arm in ["CM_CAP10", "CM_HALF10"]:
        if arm in comp:
            cm7[f"{arm}_boundary_cell"] = comp[arm]["majority_comp_verdict"]
    cls["CM7_laundering"] = cm7

    # CM8: the band's margins
    atk_all = [a for a in comp if a.startswith(("CM_U", "CM_G", "CM_J",
                                                "CM_CAP", "CM_HALF"))]
    below = [fmean(a) for a in atk_all
             if not A[a]["runs"][0]["traj"][0]["lam"] == 0
             or True]
    below = [fmean(a) for a in atk_all]
    cls["CM8_margins"] = {
        "fill_floor": COMP_FILL_FLOOR,
        "over_ceil": COMP_OVER_CEIL,
        "burn_floor": COMP_BURN_FLOOR,
        "max_attack_F_late": max(below) if below else None,
        "min_healthy_F_late": cls["CM2_fill"].get("min_healthy_F_late"),
        "loop_world_F_late": (fmean("CM_CAP30_NOCH")
                              if "CM_CAP30_NOCH" in comp else None),
        "disarmed_blind_heavy_F_late": (fmean("CM_CAP10_NOCH")
                                        if "CM_CAP10_NOCH" in comp else None),
        "burn_gap_mix_vs_J": (
            min([fmean(a) for a in ["CM_CAP30", "CM_HALF30"]
                 if a in comp]) - max(
                [fmean(a) for a in ["CM_J05", "CM_J10", "CM_J30"]
                 if a in comp])
            if all(a in comp for a in ["CM_CAP30", "CM_J30"]) else None)}
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
        globals()["SEEDS"] = [600, 601]
        globals()["CMCFG"]["CM_TWIN"]["rounds"] = 6
        globals()["CMCFG"]["CM_U30"]["rounds"] = 6
        globals()["CMCFG"]["CM_J30"]["rounds"] = 6
        globals()["CMCFG"]["CM_CAP30"]["rounds"] = 6
        globals()["CMCFG"]["CM_CAP30_NOCH"]["rounds"] = 6
        globals()["CMCFG"]["CM_T30_BASE"]["rounds"] = 8
        globals()["CMCFG"]["CM_T30_REREG"]["rounds"] = 8
        sel = ["CM_TWIN", "CM_U30", "CM_J30", "CM_CAP30",
               "CM_CAP30_NOCH", "CM_T30_BASE", "CM_T30_REREG"]
        globals()["GROUPS"]["atk"] = ["CM_TWIN", "CM_U30", "CM_J30"]
        globals()["GROUPS"]["mix"] = ["CM_CAP30", "CM_CAP30_NOCH"]
        globals()["GROUPS"]["twin"] = ["CM_T30_BASE", "CM_T30_REREG"]
        group = "all"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/m3_compcert_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/m3_compcert_results.json")
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
            "comp_fill_floor": COMP_FILL_FLOOR,
            "comp_over_ceil": COMP_OVER_CEIL,
            "comp_burn_floor": COMP_BURN_FLOOR,
            "comp_onset": COMP_ONSET, "comp_late_win": COMP_LATE_WIN,
            "comp_reg_end": COMP_REG_END,
            "attack_stream_offset": ATTACK_STREAM,
            "panel_epochs": PANEL_EPOCHS,
            "numpy": np.__version__},
            "arms": {}}

    if group == "all":
        arms = GROUPS["atk"] + GROUPS["mix"] + GROUPS["twin"]
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
        have44 = [a for a, _ in AUDIT44 if a in res["arms"]]
        if len(have44) == len(AUDIT44):
            res["audits44"] = audit(res, DOC44, AUDIT44, 14)
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
                "c_ratio", "r_org", "cert_base", "n_dis_raw", "comp_F",
                "n_disputed_repaired", "n_flipped", "n_flip_disputed",
                "n_bulk_flipped", "n_gate_flipped"]
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
        # the verdict trajectories: majority vote per round
        for vkey in ["cert_verdict"]:
            vagg = []
            for i in range(len(rows[0]["traj"])):
                vs = [r["traj"][i].get(vkey, "MUM") for r in rows]
                vagg.append(max(set(vs), key=vs.count))
            agg[vkey] = vagg
        res["arms"][arm]["aggregate"] = agg
        f_lates = []
        for rr in rows:
            _, f_late, _ = seed_composition(rr)
            if f_late is not None:
                f_lates.append(f_late)
        agg["comp_F_late_mean"] = (float(np.mean(f_lates))
                                   if f_lates else None)
        agg["comp_verdict_final"] = comp_verdict(
            agg["comp_F_late_mean"])
        print(f"{arm:15s} tau: {['%.3f' % a for a in agg['tau_sys'][:6]]}"
              f"...")
        print(f"{'':15s} b:   {['%.3f' % b for b in agg['blind_mass']]}")
        print(f"{'':15s} comp: F_late={agg['comp_F_late_mean']}"
              f" -> {agg['comp_verdict_final']}")

    # ---- classification ---------------------------------------------------
    if not smoke:
        need = set(GROUPS["atk"] + GROUPS["mix"] + GROUPS["twin"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("CM2:", {k: v for k, v in c["CM2_fill"].items()
                           if isinstance(v, dict)})
            print("CM3:", c["CM3_consumption"])
            print("CM4:", c["CM4_burning"])
            print("CM5:", c["CM5_always_on"])
            print("CM7:", c["CM7_laundering"])
            print("CM8:", c["CM8_margins"])

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
