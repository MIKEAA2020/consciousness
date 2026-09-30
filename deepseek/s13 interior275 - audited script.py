#!/usr/bin/env python3
"""
THE CROSSING'S INTERIOR - the dial's last interval, entered. Corpus
doc 77's experiment. Ordered at the ninth issuance: "the crossing's
interior" - doc 71's filed address, stated in its own words: "What
remains in the dial's last interval is not a question the grid can
ask at this resolution; it is the crossing's own address, and the
corpus, having measured both ends, the middle, and now the two steps
between, files the address and stops... if the registrar wants the
interval's interior, the machinery is proven and the cell-design
convention is written" - and doc 71's bracket: lambda* in (0.25,
0.30], the first dose whose window-bottom falls below the
proportional line, the same interval that contains the letter's
collapse.

The cell-design convention (doc 63's, verbatim): ONE dose, the same
six-point p-grid, the anchors fresh, the twin stored. The interior
dose is the MIDPOINT 0.275 - the registered choice, halving the
bracket exactly where the fall's largest step lives:
    S19_ANCHOR_275  - the fresh anchor at 0.275 (NOCH)
    S19_D100_275 .. S19_D25_275 - the six-point p-grid at 0.275
The twin: doc 51's stored C4_TWIN (0.9559), loaded not re-run. The
max effect = twin - anchor_275 on this battery's own seeds. The
fraction convention: fraction-of-max, the dose against its own
battery's max.

THE AUDIT: no stored 0.275 predecessor exists (the dial's storage:
0.0125/0.025/0.05/0.10/0.30/0.50 at doc 51, 0.15 at doc 67, 0.20
and 0.25 at doc 71), so the determinism audit is the CLEAN PREFIX -
every arm's rounds 1-5 must reproduce doc 51's stored C4_SP30
bitwise, per seed (the attack's stream is thresholded, not
re-drawn). Seven arms x six seeds = forty-two prefix audits; the
chain 51 -> 77; the determinism count passes four hundred eighty
(438 + 42).

PRE-REGISTERED PREDICTIONS (fixed before execution):
  CI1  THE INNOCENCE AUDITS: all forty-two clean-prefix comparisons
       bitwise (rounds 1-5, tau and kappa to 1e-9, vs doc 51's
       stored C4_SP30 runs on the same seeds).
  CI2  THE FRESH ANCHOR AT 0.275: the anchor's final accuracy
       against the ladder 0.8463 / 0.6207 / 0.3778 / 0.2404 /
       0.1752 / 0.1859 (0.05 through 0.30, docs 51/67/71). The
       chord at 0.275 through (0.25, 0.30) reads 0.1806. Fork: the
       anchor above / at / below the chord - is the saturation's
       interior flat, or does the coin-flip zone (the last step
       +0.011, three seeds up three down) hide structure?
  CI3  THE CURVE AT 0.275: the fractions of max at p = 0.25/0.35/
       0.40/0.50/0.75/1.0, the local slopes, against (i) the
       proportional line and (ii) the curve's own concave arc. The
       S-bottom fork: (a) f(0.25) > 0.25 - the window's bottom
       still above proportional; (b) f(0.25) < 0.25 - the 0.30
       shape arrived early. And the crossing's PLACEMENT: lambda*
       halved to (0.25, 0.275] or (0.275, 0.30).
  CI4  THE KNEE AT 0.275: the engagement slope 1.5 (doc 63's
       convention) - present (interval named) or absent; the
       migration family (0.35,0.40) at 0.20, (0.25,0.35) at 0.25
       and 0.30 - does the knee sit at (0.25,0.35) again?
  CI5  THE LAST INTERVAL'S INTERIOR (the registered question): the
       six-dose origin-slope family 2.28 / 2.58 / 2.06 / 1.74 /
       s275 / 0.89; the fall's last step (0.25 -> 0.30, carrying
       fifty percent of the whole fall) decomposed at 0.275: d3a =
       s25 - s275 over (0.25, 0.275] and d3b = s275 - s30 over
       (0.275, 0.30]. Fork: FRONT-LOADED (d3a > d3b - the fall's
       weight already spent at the interval's start), BACK-LOADED
       (d3b > d3a - the weight waits for 0.30), or SPLIT-EVEN.
  CI6  THE GROUNDS: the D25 ground's overfill (present at 0.20 and
       0.25 - expected present), the D75/D100 consumption, the
       anchor's ground (the territory-burns signature flat at
       +0.14 through 0.15/0.20/0.25?).
  CI7  THE ARMING AND THE LETTER: the first armed round (r7
       expected - the coin cannot see the dose); the re-based
       letter's demand (beta = 0.3896 of max): the smallest p
       holding it at 0.275. THE COINCIDENCE TEST: doc 71 found the
       letter's collapse interval = the crossing's interval, both
       in (0.25, 0.30]. Fork: the coincidence PRESERVED (both
       halves agree - f(0.25) < 0.25 and < 0.3896 together, or both
       above together) or SPLIT (the crossing and the letter's end
       fall on opposite sides of 0.275). The D100 row's five
       conditions (the award's (ii) expected to fail - the
       0.15-specificity's boundary). Table's integrity: no
       MAINTAINS verdicts; Column-3 unmarked.

Usage: python3 s13_interior275.py [--smoke]
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

# ---------------- the battery (the finishing arm, fresh anchors) --------
C12CFG = {
    # the fresh anchor: the attack at 0.275, the channel never armed
    # (no stored 0.275 predecessor; the dial's storage runs
    # 0.0125/0.025/0.05/0.10/0.30/0.50 at doc 51, 0.15 at doc 67,
    # 0.20 and 0.25 at doc 71)
    "S19_ANCHOR_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="none", noch=True, rounds=14,
                           rp=1.0),
    # the six-point p-grid at 0.275 (D25 through D100)
    "S19_D100_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                         eps_i=0.03, fix="none", noch=False, rounds=14,
                         rp=1.0),
    "S19_D75_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                        eps_i=0.03, fix="none", noch=False, rounds=14,
                        rp=0.75),
    "S19_D50_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                        eps_i=0.03, fix="none", noch=False, rounds=14,
                        rp=0.50),
    "S19_D40_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                        eps_i=0.03, fix="none", noch=False, rounds=14,
                        rp=0.40),
    "S19_D35_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                        eps_i=0.03, fix="none", noch=False, rounds=14,
                        rp=0.35),
    "S19_D25_275": dict(bulk=0.275, gfrac=0.0, mode="uniform",
                        eps_i=0.03, fix="none", noch=False, rounds=14,
                        rp=0.25),
}
GROUPS = {"interior": ["S19_ANCHOR_275", "S19_D100_275", "S19_D75_275",
                       "S19_D50_275", "S19_D40_275", "S19_D35_275",
                       "S19_D25_275"]}
DOC51 = "/home/z/my-project/scripts/s4_candidate4_results.json"
REFPATHS = {"51": DOC51}
# the clean-prefix audits: every arm (the anchor included) against doc
# 51's stored C4_SP30 runs, rounds 1-5 bitwise - there is no stored
# 0.275 predecessor, and the shared clean prefix is the determinism
# anchor (doc 67's adaptation, unchanged)
PERSEED_AUDITS = [(a, "C4_SP30", "51") for a in GROUPS["interior"]]
# the row->anchor pairing (the conditional's own-anchor rule)
ANCHOR = {a: "S19_ANCHOR_275" for a in GROUPS["interior"]}


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
    cfg = C12CFG[arm]
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


def audit_perseed(res, pairs, rounds_ref=5):
    """The per-seed CLEAN-PREFIX audits: each seed's tau/kappa trajectory
    must reproduce the stored predecessor's run for the same arm and seed
    over the shared clean prefix (rounds 1-5), to 1e-9 - the worlds
    diverge at round 6 where the doses differ, and the prefix is the
    determinism anchor (the attack's draws are thresholded, not
    re-drawn)."""
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
    """The pre-registered CI1-CI7 classification: the crossing's interior
    at the dial's last interval, one dose (0.275, the midpoint), the
    cell-design convention."""
    cls = {}
    A = res["arms"]
    d51 = json.load(open(DOC51))

    def stored_acc(ref, arm, i=-1):
        return float(np.mean([rr["traj"][i]["acc"]
                              for rr in ref["arms"][arm]["runs"]]))

    anchor30 = stored_acc(d51, "C4_SP30_NOCH")
    anchor05 = stored_acc(d51, "C4_SP05_NOCH")
    anchor10 = stored_acc(d51, "C4_SP10_NOCH")
    twin_acc = stored_acc(d51, "C4_TWIN")
    twin_q = float(np.mean([x for x in
                            d51["arms"]["C4_TWIN"]["aggregate"]["q_target"]
                            [PERT_ROUND - 1:] if x is not None]))

    # the stored ladder (docs 51/67/71, verified against their results
    # files at pre-registration time)
    anchor15, anchor20, anchor25 = 0.3778, 0.2404, 0.1752

    anchor275 = stored_acc(res, "S19_ANCHOR_275")
    max275 = twin_acc - anchor275

    # ---- CI1: the innocence audits -----------------------------------------
    cls["CI1_audits"] = dict(res.get("perseed_audits", {}))

    # ---- CI2: the fresh anchor and the ladder -------------------------------
    lin275 = anchor25 + (anchor30 - anchor25) * (0.275 - 0.25) / 0.05
    cls["CI2_anchor"] = {
        "anchor275": anchor275, "chord_at_275": lin275,
        "ladder": {"0.05": anchor05, "0.10": anchor10, "0.15": anchor15,
                   "0.20": anchor20, "0.25": anchor25,
                   "0.275": anchor275, "0.30": anchor30},
        "fork": ("above the chord - the last interval recovering"
                 if anchor275 > lin275 + 0.02 else
                 "below the chord - the collapse front-loaded further"
                 if anchor275 < lin275 - 0.02 else
                 "at the chord - the saturation's interior flat"),
        "max_effect_275": max275, "twin_acc": twin_acc,
        "note": "the last stored step 0.25 -> 0.30 was +0.0107, a "
                "per-seed coin flip (three up, three down; sds 0.053 "
                "and 0.074)"}

    # ---- the measured rows ---------------------------------------------------
    rows = {}
    for arm in GROUPS["interior"]:
        if arm not in A or arm == "S19_ANCHOR_275":
            continue
        row = conditions(res, arm, anchor275, twin_q, twin_acc, 14)
        row["repair_p"] = C12CFG[arm].get("rp", 1.0)
        row["dose"] = 0.275
        row["frac_max"] = (row["effect"] / max275) if max275 else None
        rows[arm] = row
    cls["CI_rows"] = rows

    # ---- CI3: the curve at 0.275, the S-bottom, the crossing ----------------
    fd = {C12CFG[a]["rp"]: rows[a]["frac_max"] for a in rows}
    grid = sorted(fd)
    slopes = {(lo, hi): (fd[hi] - fd[lo]) / (hi - lo)
              for lo, hi in zip(grid, grid[1:])}
    chord = {p: fd[0.25] + (fd[0.50] - fd[0.25]) * (p - 0.25) / 0.25
             for p in (0.35, 0.40)}
    bottom = {}
    for p in (0.25, 0.35, 0.40, 0.50):
        f = fd.get(p)
        if f is None:
            continue
        limb = ("above_chord_superlinear" if (p in chord
                and f >= chord[p]) else
                "below_chord_above_proportional" if f >= p
                else "s_bottom_present")
        bottom[p] = {"frac_max": f, "chord": chord.get(p),
                     "proportional": p, "fork_limb": limb}
    s275 = fd[0.25] / 0.25 if 0.25 in fd else None
    cls["CI3_curve"] = {
        "fractions_of_max": {str(k): v for k, v in sorted(fd.items())},
        "local_slopes": {f"{lo}->{hi}": sl for (lo, hi), sl in
                         slopes.items()},
        "bottom_readings": bottom,
        "s_bottom_present": bool(any(v["fork_limb"] == "s_bottom_present"
                                     for v in bottom.values())),
        "origin_slope": s275,
        "f_at_025": fd.get(0.25),
        "crossing_at_275": bool(fd.get(0.25) is not None
                                and fd[0.25] < 0.25),
        "lambda_star_bracket": ("(0.25, 0.275]" if (
            fd.get(0.25) is not None and fd[0.25] < 0.25)
            else "(0.275, 0.30]")}

    # ---- CI4: the knee --------------------------------------------------------
    ENGAGE = 1.5
    knee = None
    for (lo, hi), s in slopes.items():
        if lo >= 0.25 and s >= ENGAGE:
            knee = (lo, hi)
            break
    cls["CI4_knee"] = {
        "engagement_slope": ENGAGE, "knee_interval": knee,
        "knee_present": knee is not None,
        "migration_family": {"0.20": [0.35, 0.40], "0.25": [0.25, 0.35],
                             "0.30": [0.25, 0.35]}}

    # ---- CI5: the last interval's interior -----------------------------------
    f05_stored = {0.25: 0.571, 0.35: 0.601, 0.40: 0.669, 0.50: 0.757,
                  1.0: 1.010}
    f15_stored = {0.25: 0.644, 0.35: 0.776, 0.40: 0.784, 0.50: 0.860,
                  0.75: 0.949, 1.0: 1.006}
    f20_stored = {0.25: 0.516, 0.35: 0.612, 0.40: 0.790, 0.50: 0.840,
                  0.75: 0.960, 1.0: 1.009}
    f25_stored = {0.25: 0.435, 0.35: 0.603, 0.40: 0.742, 0.50: 0.717,
                  0.75: 0.934, 1.0: 0.997}
    f30_stored = {0.25: 0.222, 0.35: 0.415, 0.40: 0.489, 0.50: 0.677,
                  0.75: 0.886, 1.0: 0.986}
    fam = {0.05: f05_stored[0.25] / 0.25, 0.15: f15_stored[0.25] / 0.25,
           0.20: f20_stored[0.25] / 0.25, 0.25: f25_stored[0.25] / 0.25,
           0.275: s275, 0.30: f30_stored[0.25] / 0.25}
    d3a = fam[0.25] - fam[0.275]        # (0.25, 0.275]
    d3b = fam[0.275] - fam[0.30]        # (0.275, 0.30]
    d3 = fam[0.25] - fam[0.30]
    cls["CI5_last_interval"] = {
        "origin_slopes_by_dose": {str(d): fam[d] for d in sorted(fam)},
        "the_falls": {"total_from_peak": fam[0.15] - fam[0.30],
                      "last_interval_total": d3,
                      "d3a_(0.25,0.275]": d3a, "d3b_(0.275,0.30]": d3b,
                      "d3a_share": (d3a / d3 if d3 else None),
                      "d3b_share": (d3b / d3 if d3 else None)},
        "fork": ("FRONT-LOADED - the fall's weight already spent at "
                 "the interval's start" if d3a > d3b + 0.05 else
                 "BACK-LOADED - the weight waits for 0.30"
                 if d3b > d3a + 0.05 else "SPLIT-EVEN"),
        "stored_curves": {"0.05": f05_stored, "0.15": f15_stored,
                          "0.20": f20_stored, "0.25": f25_stored,
                          "0.30": f30_stored}}

    # ---- CI6: the grounds ------------------------------------------------------
    cls["CI6_grounds"] = {
        "measured": {a: {"F_late": rows[a]["ground_F_late"],
                         "verdict": rows[a]["ground_verdict"]}
                     for a in rows},
        "doc67_stored_15": {"D25": 0.0612, "D35": 0.0631, "D40": 0.0609,
                            "D50": 0.0423, "D75": 0.0196, "D100": 0.0230},
        "doc71_stored_20": {"D25": 0.0841, "D35": 0.0795, "D40": 0.0378,
                            "D50": 0.0414, "D75": 0.0161, "D100": 0.0076},
        "doc71_stored_25": {"D25": 0.0939, "D35": 0.0701, "D40": 0.0662,
                            "D50": 0.0412, "D75": 0.0225, "D100": 0.0055},
        "doc59_stored_30": {"D25": 0.1257, "D50": 0.0425, "D75": 0.0200,
                            "D100": -0.0088},
        "D25_overfills": bool(any(
            rows[a]["ground_verdict"] == "OVERFILLED" for a in rows
            if C12CFG[a]["rp"] == 0.25)),
        "anchor_ground": {}}
    gr = ground_readings(res, "S19_ANCHOR_275")
    if gr:
        cls["CI6_grounds"]["anchor_ground"] = {
            "F_late": gr["F_late_mean"], "verdict": gr["verdict_mean"]}

    # ---- CI7: the arming, the letter, the coincidence --------------------------
    BETA_FRAC = EFFECT_BAR / 0.7700
    hold = [p for p in sorted(fd) if fd[p] is not None
            and fd[p] >= BETA_FRAC]
    p_letter = hold[0] if hold else None
    d100 = [a for a in rows if C12CFG[a]["rp"] == 1.0]
    crossing = cls["CI3_curve"]["crossing_at_275"]
    letter_alive = bool(fd.get(0.25) is not None
                        and fd[0.25] >= BETA_FRAC)
    cls["CI7_arming_letter"] = {
        "beta_fraction": BETA_FRAC,
        "first_armed_by_arm": {a: rows[a]["first_armed_round"]
                               for a in rows},
        "smallest_p_holding_the_rebased_letter": p_letter,
        "letter_holds_everywhere_measured": (
            p_letter == min(fd) if fd else None),
        "the_coincidence_test": {
            "crossing_at_275": crossing,
            "letter_alive_at_275": letter_alive,
            "verdict": ("PRESERVED - the crossing and the letter's end "
                        "fall on the same side of the midpoint"
                        if crossing != letter_alive else
                        "SPLIT - the crossing and the letter's end "
                        "fall on opposite sides of 0.275")},
        "conditions_D100": ({k: rows[d100[0]][k]
                             for k in ("(i)", "(ii)", "(iii)_inherited",
                                       "(iii)_rebased", "(iv)", "(v)")}
                            if d100 else None),
        "budget_sheet": {"0.05": "everywhere, max effect 0.110",
                         "0.15": "the full grid, max effect 0.578",
                         "0.20": "smallest p 0.25, max effect 0.716",
                         "0.25": "smallest p 0.25, max effect 0.781",
                         "0.275": (f"smallest p {p_letter}, "
                                   f"max effect {max275:.3f}"),
                         "0.30": "[0.95, 1.0], max effect 0.770"},
        "integrity": {
            "any_MAINTAINS": False,
            "column3": "unmarked everywhere",
            "fresh_arms": list(rows),
            "audit": "the clean prefix, all seven arms x six seeds, "
                     "vs doc 51's stored C4_SP30 (rounds 1-5)"}}
    return cls


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    if smoke:
        globals()["SEEDS"] = [600, 601]
        for a in GROUPS["interior"]:
            globals()["C12CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s13_interior275_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s13_interior275_results.json")
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

    arms = GROUPS["interior"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C12CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- the per-seed clean-prefix audits: all seven arms -------------
    if not smoke:
        need_ps = [a for a, _, _ in PERSEED_AUDITS if a in res["arms"]]
        if len(need_ps) == len(PERSEED_AUDITS):
            res["perseid_audits"] = audit_perseed(res, PERSEED_AUDITS,
                                                  rounds_ref=5)

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
        need = set(GROUPS["interior"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the crossing's interior (the dial at 0.275) ===")
            print("CI2 anchor:", c["CI2_anchor"])
            for arm, r in c["CI_rows"].items():
                print(f"  {arm:12s} p={r['repair_p']:.2f} "
                      f"acc={r['acc_final']:.4f} "
                      f"effect={r['effect']:+.4f} "
                      f"({r['frac_max']:.3f} of max) "
                      f"ground={str(r['ground_verdict']):11s} "
                      f"F={r['ground_F_late']:+.4f} "
                      f"armed r{r['first_armed_round']}")
            print("\nCI3 curve:", c["CI3_curve"])
            print("\nCI4 knee:", c["CI4_knee"])
            print("\nCI5 the last interval's interior:",
                  c["CI5_last_interval"])
            print("\nCI6 grounds:", c["CI6_grounds"])
            print("\nCI7 arming/letter:", c["CI7_arming_letter"])
            print("CI1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
