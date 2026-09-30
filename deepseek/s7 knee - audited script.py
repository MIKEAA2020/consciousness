#!/usr/bin/env python3
"""
THE KNEE'S LOCATION - the S-curve's bracket tightened, and the window
read at the grid's other end. Corpus doc 63's experiment. Ordered at
the sixth-candidate issuance: "the knee's location (a D35/D40 arm)" -
doc 59's filed fork: "the knee's location, and whether it moves with
dose continuously or itself flips - the two-point dose grid (0.05
superlinear, 0.30 S-shaped) does not resolve between a smooth family
and a phase change, and the corpus does not pretend it does."

Doc 59's threat (i) named the design: "a D35 or D40 would bracket it;
not run, not inferred." The arms (the battery's one design liberty,
taken before any compute and disclosed: the 0.05 FLANKS - the same
p-window measured at the grid's other end, where doc 56's curve is
superlinear, so that the knee's dose-movement question has its
measurable part answered with stored references rather than a new
dose):
    S9_D100_30 (audit, bitwise vs doc 59's S6_D100_30),
    S9_D40_30, S9_D35_30   - the knee's bracket at lambda = 0.30
    S9_D100_05 (audit, bitwise vs doc 56's S5_D100_05),
    S9_D40_05, S9_D35_05   - the same window at lambda = 0.05
The anchors: doc 51's stored C4_SP30_NOCH (0.1859) and C4_SP05_NOCH
(0.8463); the twin doc 51's C4_TWIN (0.9559); max effects 0.7700 and
0.1096. The fraction convention UNIFIED to fraction-of-max (doc 56's
own), with doc 59's fraction-of-full reported beside for continuity:
of max, doc 59's curve reads 0.222/0.677/0.886/0.986 at
p = 0.25/0.5/0.75/1.0; doc 56's reads 0.571/0.757/1.010 at
p = 0.25/0.5/1.0.

PRE-REGISTERED PREDICTIONS (fixed before execution):
  KN1  THE INNOCENCE AUDITS: S9_D100_30 reproduces doc 59's S6_D100_30
       bitwise and S9_D100_05 reproduces doc 56's S5_D100_05 bitwise,
       per seed (twelve comparisons; the chain 51 -> 59 and 48 -> 56
       extended; the determinism count passes two hundred twenty-two,
       210 + 12).
  KN2  THE KNEE AT 0.30: the fractions of max at p = 0.25/0.35/0.40/
       0.50/0.75/1.0 (the interior two measured here, the rest stored),
       the local slopes, and the knee's bracket by the pre-registered
       ENGAGEMENT SLOPE 1.5 (the midpoint between the bottom's ~0.9 and
       the climb's ~1.8; operational, disclosed): the knee sits in the
       first grid interval whose forward slope reaches 1.5. Forks:
       knee in [0.25, 0.35] / in (0.35, 0.40] / in (0.40, 0.50).
  KN3  THE 0.05 FLANKS: the fractions of max at p = 0.35 and 0.40 on
       the 0.05 world, against (i) the concave arc (the chord of doc
       56's stored neighbors: chord(0.35) = 0.645, chord(0.40) = 0.682)
       and (ii) the proportional line f = p (the S-bottom's signature:
       doc 59's 0.30-side knee showed f(0.25) = 0.222 < 0.25). Forks:
       (i) f >= chord - the window concave, no knee at the bottom of
       the grid; (ii) chord > f >= p - the superlinearity weakening
       but above proportional; (iii) f < p - the S-bottom present at
       0.05, the knee living at both ends.
  KN4  THE SHAPE FAMILY: composing KN2 + KN3 - if the window is
       knee-free at 0.05 and kned at 0.30, the shape FLIPS somewhere in
       the grid's interior (0.05, 0.30), and the interior dose is the
       next cell, filed; if the knee is present at both ends, it
       MIGRATES with dose and both locations are measured.
  KN5  THE GROUNDS: the new arms' composition verdicts - at 0.30
       between D25's OVERFILLED (+0.1257) and D50's ON-COURSE
       (+0.0425); at 0.05 between doc 56's stored grounds. The drain's
       p-curve shares the effect curve's knee (doc 59's reading) - the
       fork: does the ground's migration knee with the effect?
  KN6  THE ARMING: repair-blind on the fresh arms, dose-early at 0.30
       (r7) against 0.05's r8 - the coin cannot see the dose.
  KN7  THE TABLE'S INTEGRITY: no MAINTAINS verdicts; the Column-3 cell
       unmarked; every number a measurement; the fresh arms (D35, D40
       at both doses) measured as they come.

Usage: python3 s7_knee.py [--smoke]
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
REPAIR_STREAM = 978                     # the half-buying coin's separate rng offset
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

# ---------------- the battery (the knee's bracket, both ends) ---------
C6CFG = {
    # the knee's row: the S-curve's interior at 0.30
    "S9_D100_30": dict(bulk=0.30, gfrac=0.0, mode="uniform",
                       eps_i=0.03, fix="none", noch=False, rounds=14,
                       rp=1.0),
    "S9_D40_30": dict(bulk=0.30, gfrac=0.0, mode="uniform",
                      eps_i=0.03, fix="none", noch=False, rounds=14,
                      rp=0.40),
    "S9_D35_30": dict(bulk=0.30, gfrac=0.0, mode="uniform",
                      eps_i=0.03, fix="none", noch=False, rounds=14,
                      rp=0.35),
    # the flanks: the same window at the grid's other end
    "S9_D100_05": dict(bulk=0.05, gfrac=0.0, mode="uniform",
                       eps_i=0.03, fix="none", noch=False, rounds=14,
                       rp=1.0),
    "S9_D40_05": dict(bulk=0.05, gfrac=0.0, mode="uniform",
                      eps_i=0.03, fix="none", noch=False, rounds=14,
                      rp=0.40),
    "S9_D35_05": dict(bulk=0.05, gfrac=0.0, mode="uniform",
                      eps_i=0.03, fix="none", noch=False, rounds=14,
                      rp=0.35),
}
GROUPS = {"knee": ["S9_D100_30", "S9_D40_30", "S9_D35_30",
                   "S9_D100_05", "S9_D40_05", "S9_D35_05"]}
DOC51 = "/home/z/my-project/scripts/s4_candidate4_results.json"
DOC56 = "/home/z/my-project/scripts/s5_boundaries_results.json"
DOC59 = "/home/z/my-project/scripts/s6_topdial_results.json"
REFPATHS = {"51": DOC51, "56": DOC56, "59": DOC59}
# the per-seed bitwise audits: the two p=1.0 controls, each against its
# most recent stored predecessor (the chain 51 -> 59; the chain 48 -> 56)
PERSEED_AUDITS = [("S9_D100_30", "S6_D100_30", "59"),
                  ("S9_D100_05", "S5_D100_05", "56")]
# the row->anchor pairing (the conditional's own-anchor rule)
ANCHOR = {
    "S9_D100_30": "DOC51:C4_SP30_NOCH", "S9_D40_30": "DOC51:C4_SP30_NOCH",
    "S9_D35_30": "DOC51:C4_SP30_NOCH", "S9_D100_05": "DOC51:C4_SP05_NOCH",
    "S9_D40_05": "DOC51:C4_SP05_NOCH", "S9_D35_05": "DOC51:C4_SP05_NOCH",
}


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
    cfg = C6CFG[arm]
    lam_bulk, gfrac, mode = cfg["bulk"], cfg["gfrac"], cfg["mode"]
    eps_i, fix, rounds = cfg["eps_i"], cfg["fix"], cfg["rounds"]
    noch = cfg["noch"]
    rp = cfg.get("rp", 1.0)           # the half-buying dial (1.0 = the channel)
    lam = lam_bulk + gfrac            # the filed total dose (for kicks)
    kicks = lam > 0.0
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
    rng_repair = np.random.default_rng(seed + REPAIR_STREAM)
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
            if rp >= 1.0:
                # the unmodified channel - no coin, bitwise-anchored
                yt_train[disputed] = pl[disputed]
            elif rp > 0.0:
                # the half-buying dial: each disputed label repaired with
                # probability rp; one full-length coin draw per armed round
                # from the separate stream (the chassis rng untouched)
                coin = rng_repair.random(len(yt_p))
                sel = disputed & (coin < rp)
                yt_train[sel] = pl[sel]
                n_rep = int(sel.sum())
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


# ---------------- the composition verdict (doc 47's, riding) ----------------
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


# ---------------- the boundaries' verdict machinery -------------------------
RULER_BAR = 0.93      # (i): the declared clean ruler's letter
EFFECT_BAR = 0.30      # (iii): the inherited bar (doc 39)
BETA = 0.30 / 0.7700   # (iii) re-based: doc 53's filed amendment


def ground_readings(res, arm):
    """The composition column on one arm: the mean F_late over the
    verdict window, the per-seed verdicts, the majority, the straddle."""
    A = res["arms"]
    if arm not in A:
        return None
    runs = A[arm]["runs"]
    f_lates, verdicts = [], []
    for rr in runs:
        _, f_late, verd = seed_composition(rr)
        if f_late is not None:
            f_lates.append(f_late)
            verdicts.append(verd)
    if not f_lates:
        return None
    f_mean = float(np.mean(f_lates))
    maj = max(set(verdicts), key=verdicts.count)
    return {"F_late_mean": f_mean,
            "verdict_mean": comp_verdict(f_mean),
            "per_seed_F_late": f_lates,
            "per_seed_verdicts": verdicts,
            "verdict_majority": maj,
            "straddle": bool(comp_verdict(f_mean) != maj),
            "n_below_burn_floor": int(sum(
                1 for f in f_lates if f < COMP_BURN_FLOOR)),
            "n_above_over_ceil": int(sum(
                1 for f in f_lates if f > COMP_OVER_CEIL))}


def stored_ground(ref, arm):
    """The composition reading of a PREDECESSOR's stored arm (the
    references the batteries cite rather than re-run)."""
    f_lates = []
    for rr in ref["arms"][arm]["runs"]:
        n = len(rr["traj"])
        late = [t["comp_F"] for t in rr["traj"]
                if t["comp_F"] is not None and t["round"] >= n - 3]
        if late:
            f_lates.append(np.mean(late))
    if not f_lates:
        return None
    f_mean = float(np.mean(f_lates))
    return {"F_late_mean": f_mean, "verdict": comp_verdict(f_mean),
            "per_seed": f_lates}


def conditions(res, arm, anchor_acc, twin_q, twin_acc, ref_rounds):
    """The four conditions + (v) + the (vi) candidate on one armed row,
    horizon-relative (the overfill rows run 30 rounds; the straddle and
    halfdef rows 14). Returns the row dictionary."""
    A = res["arms"]
    ag = A[arm]["aggregate"]["acc"]
    tau = A[arm]["aggregate"]["tau_sys"]
    qbar = float(np.mean([x for x in A[arm]["aggregate"]["q_target"]
                          [PERT_ROUND - 1:] if x is not None]))
    acc_f = ag[-1]
    final = tau[-1]
    arm_rounds = [i for i in range(PERT_ROUND - 1, len(ag))
                  if A[arm]["aggregate"]["armed"][i] > 0.5]
    late = [i for i in range(max(PERT_ROUND - 1, len(ag) - 6), len(ag))]
    armed_late = float(np.sum(
        [A[arm]["aggregate"]["armed"][i] for i in late]))
    rep = A[arm]["aggregate"]["repair_fired"]
    oob_strict = sum(1 for i in range(PERT_ROUND - 1, len(ag))
                     if tau[i] < TAU0 - EPS_T and rep[i] < 0.5)
    oob_amended = sum(
        1 for i in range(PERT_ROUND - 1, len(ag))
        if tau[i] < TAU0 - EPS_T
        and not (rep[i] >= 0.5
                 or (i + 1 < len(ag) and rep[i + 1] >= 0.5)))
    late_slope = (tau[-1] - tau[-4]) / 3.0
    closed = (qbar >= twin_q - 0.01 and final >= 0.93 and acc_f >= 0.95)
    looped = (acc_f < 0.60 and qbar < twin_q - 0.01)
    contained = (qbar < twin_q - 0.01 and acc_f >= 0.85
                 and final < 0.93 and armed_late >= 0.5 * len(late))
    fork = ("LOOP" if looped else "CLOSED" if closed
            else "CONTAINED" if contained else "PARTIAL")
    cond1 = bool(acc_f >= RULER_BAR)
    cond2 = bool(len(arm_rounds) >= 1 and oob_amended == 0)
    threat = bool(anchor_acc < RULER_BAR)
    effect = acc_f - anchor_acc
    max_effect = twin_acc - anchor_acc
    cond3 = bool(effect >= EFFECT_BAR) if threat else None
    cond3_re = bool(effect >= BETA * max_effect) if threat else None
    cond4 = bool(tension(TAU0, qbar) <= final <= TAU0 + EPS_T
                 and late_slope >= -1e-12)
    gr = ground_readings(res, arm)
    cond5 = None if gr is None else bool(gr["verdict_mean"] != "BURNED")
    cond6 = None if gr is None else bool(
        gr["verdict_mean"] not in ("BURNED", "OVERFILLED"))
    return {"fork": fork, "q_bar_realized": qbar, "final_tau": final,
            "acc_final": acc_f, "anchor_acc": anchor_acc,
            "threat": threat, "effect": effect, "max_effect": max_effect,
            "first_armed_round": (min(arm_rounds) + 1
                                  if arm_rounds else None),
            "armed_rounds": len(arm_rounds),
            "oob_strict": oob_strict, "oob_amended": oob_amended,
            "late_tau_slope": late_slope,
            "tension_at_own_q": tension(TAU0, qbar),
            "(i)": cond1, "(ii)": cond2,
            "(iii)_inherited": cond3, "(iii)_rebased": cond3_re,
            "(iv)": cond4, "(v)": cond5, "(vi)_candidate": cond6,
            "ground_F_late": (gr["F_late_mean"] if gr else None),
            "ground_verdict": (gr["verdict_mean"] if gr else None),
            "ground_per_seed": (gr["per_seed_F_late"] if gr else None),
            "ground_n_below_burn": (gr["n_below_burn_floor"]
                                    if gr else None),
            "ground_n_above_ceil": (gr["n_above_over_ceil"]
                                    if gr else None)}


def audit_perseed(res, pairs, rounds_ref=14):
    """The per-seed audits: each seed's tau/kappa trajectory must
    reproduce the stored predecessor's run for the same arm and seed,
    to 1e-9 (full length; the worlds are literally the same worlds)."""
    out = {}
    for ours, theirs, key in pairs:
        if ours not in res["arms"]:
            continue
        ref = json.load(open(REFPATHS[key]))
        for rr in res["arms"][ours]["runs"]:
            s = rr["seed"]
            match = [x for x in ref["arms"][theirs]["runs"]
                     if x["seed"] == s]
            if not match:
                continue
            t_ref = match[0]["traj"][:rounds_ref]
            t_got = rr["traj"][:rounds_ref]
            ok = bool(len(t_ref) == len(t_got)
                      and all(abs(a["tau_sys"] - b["tau_sys"]) < 1e-9
                              and abs(a["kappa_sys"] - b["kappa_sys"]) < 1e-9
                              for a, b in zip(t_ref, t_got)))
            out[f"{ours}_seed{s}_reproduces_{theirs}"] = ok
            print(f"{ours} seed={s} reproduces {theirs} bitwise: {ok}")
    return out


def classify(res):
    """The pre-registered KN1-KN7 classification: the knee's bracket
    at 0.30, the same window at 0.05, and the shape family's fork."""
    cls = {}
    A = res["arms"]
    d51 = json.load(open(DOC51))
    d56 = json.load(open(DOC56))
    d59 = json.load(open(DOC59))

    # the references: doc 51's stored anchors and twin (both doses)
    def stored_acc(ref, arm, i=-1):
        return float(np.mean([rr["traj"][i]["acc"]
                              for rr in ref["arms"][arm]["runs"]]))

    anchor30 = stored_acc(d51, "C4_SP30_NOCH")
    anchor05 = stored_acc(d51, "C4_SP05_NOCH")
    twin_acc = stored_acc(d51, "C4_TWIN")
    twin_q = float(np.mean([x for x in
                            d51["arms"]["C4_TWIN"]["aggregate"]["q_target"]
                            [PERT_ROUND - 1:] if x is not None]))
    max30 = twin_acc - anchor30
    max05 = twin_acc - anchor05

    # KN1: the innocence audits
    cls["KN1_audits"] = dict(res.get("perseid_audits", {}))

    # ---- the measured rows, both doses ----------------------------------
    rows = {}
    for arm in GROUPS["knee"]:
        if arm not in A:
            continue
        anch = anchor30 if arm.endswith("_30") else anchor05
        mx = max30 if arm.endswith("_30") else max05
        row = conditions(res, arm, anch, twin_q, twin_acc, 14)
        p = C6CFG[arm].get("rp", 1.0)
        row["repair_p"] = p
        row["frac_max"] = (row["effect"] / mx) if mx else None
        rows[arm] = row
    cls["KN_rows"] = rows

    # ---- KN2: the knee at 0.30 ------------------------------------------
    # the curve of fractions-of-max, interior measured, rest stored
    f30 = {0.25: 0.2222, 0.50: 0.6767, 0.75: 0.8865}
    f30[1.0] = rows["S9_D100_30"]["frac_max"]
    for a in ("S9_D35_30", "S9_D40_30"):
        if a in rows:
            f30[C6CFG[a]["rp"]] = rows[a]["frac_max"]
    full_eff30 = rows["S9_D100_30"]["effect"]
    grid = sorted(f30)
    slopes = {(lo, hi): (f30[hi] - f30[lo]) / (hi - lo)
              for lo, hi in zip(grid, grid[1:])}
    ENGAGE = 1.5      # the pre-registered engagement slope
    knee = None
    for (lo, hi), sl in slopes.items():
        if lo >= 0.25 and sl >= ENGAGE:
            knee = (lo, hi)
            break
    cls["KN2_knee30"] = {
        "fractions_of_max": {str(k): v for k, v in sorted(f30.items())},
        "doc59_fractions_of_full": {"0.25": 0.225, "0.50": 0.686,
                                    "0.75": 0.899, "1.0": 1.000},
        "local_slopes": {f"{lo}->{hi}": sl for (lo, hi), sl in
                         slopes.items()},
        "engagement_slope": ENGAGE,
        "knee_interval": knee,
        "full_effect_30": full_eff30}

    # ---- KN3: the 0.05 flanks -------------------------------------------
    f05 = {0.25: 0.571, 0.50: 0.757}
    f05[1.0] = rows["S9_D100_05"]["frac_max"] if "S9_D100_05" in rows \
        else 1.010
    for a in ("S9_D35_05", "S9_D40_05"):
        if a in rows:
            f05[C6CFG[a]["rp"]] = rows[a]["frac_max"]
    chord = {p: 0.571 + (0.757 - 0.571) * (p - 0.25) / 0.25
             for p in (0.35, 0.40)}
    flank = {}
    for p in (0.35, 0.40):
        f = f05.get(p)
        if f is None:
            continue
        limb = ("concave_no_knee" if f >= chord[p]
                else "above_proportional" if f >= p
                else "s_bottom_present")
        flank[p] = {"frac_max": f, "chord": chord[p], "proportional": p,
                    "fork_limb": limb}
    cls["KN3_flanks05"] = {
        "fractions_of_max": {str(k): v for k, v in sorted(f05.items())},
        "readings": flank,
        "any_s_bottom_at_05": bool(
            any(v["fork_limb"] == "s_bottom_present"
                for v in flank.values()))}

    # ---- KN4: the shape family -----------------------------------------
    knee_present_05 = cls["KN3_flanks05"]["any_s_bottom_at_05"]
    cls["KN4_family"] = {
        "knee_interval_at_30": knee,
        "knee_present_at_05": knee_present_05,
        "verdict": ("MIGRATES - the knee present at both ends, both "
                    "locations measured" if knee_present_05 else
                    "FLIPS - the window knee-free at 0.05, kned at 0.30; "
                    "the shape's emergence confined to the grid's "
                    "interior (0.05, 0.30); the interior dose filed as "
                    "the next cell")}

    # ---- KN5: the grounds ------------------------------------------------
    def stored_ground_F(ref, arm):
        return stored_ground(ref, arm)

    cls["KN5_grounds"] = {
        "measured": {a: {"F_late": rows[a]["ground_F_late"],
                         "verdict": rows[a]["ground_verdict"]}
                     for a in rows},
        "doc59_stored": {"D25": 0.1257, "D50": 0.0425, "D75": 0.0200,
                         "D100": -0.0088},
        "doc56_stored": {a: stored_ground_F(d56, a)
                         for a in ("S5_D25_05", "S5_D50_05",
                                   "S5_D100_05")}}

    # ---- KN6: the arming -------------------------------------------------
    cls["KN6_arming"] = {a: rows[a]["first_armed_round"] for a in rows}

    # ---- KN7: the table's integrity --------------------------------------
    cls["KN7_integrity"] = {
        "any_MAINTAINS": False,
        "column3": "unmarked everywhere",
        "fresh_arms": ["S9_D35_30", "S9_D40_30", "S9_D35_05",
                       "S9_D40_05"],
        "audit_arms": ["S9_D100_30 (bitwise to doc 59's S6_D100_30)",
                       "S9_D100_05 (bitwise to doc 56's S5_D100_05)"]}
    return cls


def audit_straddle_perseed(res, ref_path, pairs):
    """The straddle arms' per-seed audits: each ORIGINAL seed's tau/kappa
    trajectory must reproduce doc 51's stored run for the same arm and
    seed, to 1e-9 - the determinism count's first per-seed extension."""
    ref = json.load(open(ref_path))
    out = {}
    for ours, theirs in pairs:
        if ours not in res["arms"]:
            continue
        for rr in res["arms"][ours]["runs"]:
            s = rr["seed"]
            match = [x for x in ref["arms"][theirs]["runs"]
                     if x["seed"] == s]
            if not match:
                continue
            t_ref = match[0]["traj"]
            t_got = rr["traj"]
            ok = bool(len(t_ref) == len(t_got)
                      and all(abs(a["tau_sys"] - b["tau_sys"]) < 1e-9
                              and abs(a["kappa_sys"] - b["kappa_sys"]) < 1e-9
                              for a, b in zip(t_ref, t_got)))
            out[f"{ours}_seed{s}_reproduces_{theirs}"] = ok
            print(f"{ours} seed={s} reproduces {theirs} bitwise: {ok}")
    return out


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    if smoke:
        globals()["SEEDS"] = [600, 601]
        for a in GROUPS["knee"]:
            globals()["C6CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s7_knee_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s7_knee_results.json")
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
            "eta_i": ETA_I, "rounds": 14,
            "pert_round": PERT_ROUND, "nseed": len(SEEDS),
            "cert_floor": CERT_FLOOR, "cert_xmin": CERT_XMIN,
            "cert_rjump": CERT_RJUMP, "eta_rorg": ETA_RORG,
            "comp_fill_floor": COMP_FILL_FLOOR,
            "comp_over_ceil": COMP_OVER_CEIL,
            "comp_burn_floor": COMP_BURN_FLOOR,
            "comp_onset": COMP_ONSET, "comp_late_win": COMP_LATE_WIN,
            "comp_reg_end": COMP_REG_END,
            "ruler_bar": RULER_BAR, "effect_bar": EFFECT_BAR,
            "beta_rebased": BETA, "repair_stream_offset": REPAIR_STREAM,
            "attack_stream_offset": ATTACK_STREAM,
            "panel_epochs": PANEL_EPOCHS,
            "numpy": np.__version__},
            "arms": {}}

    arms = GROUPS["knee"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C6CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- the per-seed bitwise audits: the two p=1.0 controls --------------
    if not smoke:
        need_ps = [a for a, _, _ in PERSEED_AUDITS if a in res["arms"]]
        if len(need_ps) == len(PERSEED_AUDITS):
            res["perseid_audits"] = audit_perseed(res, PERSEED_AUDITS)

    # ---- aggregates ---------------------------------------------------------
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
        vagg = []
        for i in range(len(rows[0]["traj"])):
            vs = [r["traj"][i].get("cert_verdict", "MUM") for r in rows]
            vagg.append(max(set(vs), key=vs.count))
        agg["cert_verdict"] = vagg
        f_lates = []
        for rr in rows:
            _, f_late, _ = seed_composition(rr)
            if f_late is not None:
                f_lates.append(f_late)
        agg["comp_F_late_mean"] = (float(np.mean(f_lates))
                                   if f_lates else None)
        agg["comp_verdict_final"] = comp_verdict(
            agg["comp_F_late_mean"])
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:15s} acc_f={agg['acc'][-1]:.4f} "
              f"tau_f={agg['tau_sys'][-1]:.4f} "
              f"armed_rds={int(sum(agg['armed']))} "
              f"comp={agg['comp_verdict_final']}")

    # ---- classification ----------------------------------------------------
    if not smoke:
        need = set(GROUPS["knee"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the knee's location (both ends of the grid) ===")
            for arm, r in c["KN_rows"].items():
                print(f"  {arm:12s} p={r['repair_p']:.2f} "
                      f"acc={r['acc_final']:.4f} "
                      f"effect={r['effect']:+.4f} "
                      f"({r['frac_max']:.3f} of max) "
                      f"ground={str(r['ground_verdict']):11s} "
                      f"F={r['ground_F_late']:+.4f} "
                      f"armed r{r['first_armed_round']}")
            print("\nKN2 knee at 0.30:", c["KN2_knee30"])
            print("\nKN3 flanks at 0.05:", c["KN3_flanks05"])
            print("\nKN4 family:", c["KN4_family"])
            print("\nKN5 grounds:", c["KN5_grounds"])
            print("KN6 arming:", c["KN6_arming"])
            print("KN1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
