#!/usr/bin/env python3
"""
THE FIFTH CONDITION'S BOUNDARIES - three batteries, one chassis.
Corpus docs 54-56's experiments. Ordered at the fourth-candidate
issuance: "the overfill clause (should a runaway fill refuse an award?),
the CAP30 straddle (one seed from flipping the founding refusal), a
half-buying defense at 0.05 to price beta's middle dial."

BATTERY 1 - THE OVERFILL CLAUSE (doc 54; doc 51's filed open question:
no armed world overfills in the fourth candidate's battery, and the
REREG-under-attack version is unmeasured). The letter under test: (v)
refuses BURNED grounds; the candidate (vi) would also refuse OVERFILLED
ones - should a runaway fill refuse an award? The arms: the REREG dial
(doc 40's record walk, the maintenance failure that reads +0.109 on the
composition column at 30 rounds) UNDER ATTACK at the two mild doses,
with fresh disarmed anchors, all at the dial's own 30-round horizon
(where doc 47 measured the runaway):
    S5_RR_U05, S5_RR_U05_NOCH, S5_RR_U10, S5_RR_U10_NOCH
The registered forks: (1) do the armed REREG worlds pass (i)-(iv) at
30 rounds (the ruler held, the engagement honest, the effect priced,
the bracket held - the record's walk might break (iv) before the ground
is ever read)? (2) does the attack's consumption MASK the drift's
overfill (doc 47's residents aliasing again: the attack consumes, the
drift fills, the total reads between) or does the runaway dominate?
(3) the clause's adjudication inputs: the world-state census at the
firing cells. Twin references for the 30-round fork letters: doc 40's
stored ID_TWIN_BASE (q_bar 0.9864, acc 0.9515), loaded not re-run.

BATTERY 2 - THE STRADDLE (doc 55; doc 51's threat (i)): CAP30's
founding refusal rests on a per-seed split of three-and-three around
the burn floor with the mean -0.007 below it - one seed from flipping.
The arms: CAP30 and its anchor RE-RUN AT TWELVE SEEDS (600-611 - the
corpus's six plus six fresh), fourteen rounds, per-seed bitwise audits
against doc 51's stored runs for the six original seeds (twelve audits
- the determinism count's first per-seed extension). The registered
forks: the N=12 mean stays below the floor (the refusal robust); the
mean rises above (the founding refusal flips at N=12 - the amendment's
bite re-opened); the mean falls further (robust and then some). The
majority-governs counterfactual reported alongside the mean-governs
rule that decided doc 51.

BATTERY 3 - THE HALF-BUYING DEFENSE (doc 56; doc 53's threat (i)): a
defense that buys back ~half the threat at 0.05, to price beta's
middle dial (no such defense in the inventory; the re-based bar's
clearances were asserted from the full defense's ratio alone). The
dial: repair_p - each disputed label is repaired with probability p
(the channel's cleaning thinned by a coin). p = 1.0 is the unmodified
channel (the code path identical, no coin drawn - bitwise); p = 0.5
and 0.25 are the half- and quarter-buyers. The coin consumes a SEPARATE
rng stream (seed + 978), one full-length draw per armed round - the
chassis stream untouched. The anchor: doc 51's stored SP05_NOCH world
(0.8463 - the disarmed anchor is repair-blind). The bars: the inherited
0.30 and the re-based b(0.05) = 0.3896 x 0.1096 = 0.0427. The
registered forks: the p-curve linear or sublinear (the loop's feedback
- half the catches leave more dirt, the world degrades further, the
buyback may fall below half); D50 vs the two bars (clears both / clears
one / neither); D25 likewise. The ground columns ride on every arm.

PRE-REGISTERED PREDICTIONS (fixed before execution):
  BD1  THE INNOCENCE AUDITS: S5_D100_05 reproduces doc 51's C4_SP05
       bitwise (the p=1 path is the unmodified channel); the twelve
       straddle per-seed audits pass (CAP30/CAP30_NOCH seeds 600-605
       vs doc 51); the fresh arms (the four REREG arms, D50, D25) are
       measured as they come. Determinism count past one hundred
       sixty-nine (156 + 13).
  BD2  THE OVERFILL ROWS: fork on (i)-(iv) at 30 rounds - if the armed
       REREG worlds pass and their grounds read OVERFILLED, the clause's
       test cell fires (the (vi) candidate adjudicated on live worlds);
       if (iv) breaks first (the walked record breaking the bracket),
       the clause is unreachable and the finding is the ordering; if
       the grounds read ON-COURSE/CONSUMED, the attack masks the
       runaway and the clause's question is postponed by aliasing.
  BD3  THE MASKING READING: the armed REREG fills vs doc 47's stored
       plain-attack fills (U05 +0.0495, U10 +0.0294) and the stored
       REREG twin (+0.1093) - additive, masked, or dominated; both
       directions informative.
  BD4  THE STRADDLE: the N=12 mean and split; the verdict under
       mean-governs; the majority counterfactual; the anchor's N=12
       ground and the differential's robustness.
  BD5  THE P-CURVE: D50's effect as a fraction of the full defense's
       0.1107 (the linear middle is 0.5; the loop's feedback may push
       below); D25's fraction likewise (0.25 linear).
  BD6  THE DIAL: D50 and D25 against the inherited bar (both refuse -
       0.30 absolute), against the re-based bar 0.0427 (fork), and the
       beta values at which each would clear (effect/0.1096); the
       grounds and armed rounds reported.
  BD7  THE TABLE'S INTEGRITY: no MAINTAINS verdicts; the Column-3 cell
       unmarked; every number a measurement; the fresh arms' freshness
       disclosed.

Usage: python3 s5_boundaries.py [--smoke] [--group {overfill,straddle,halfdef,all}]
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

# ---------------- the battery (the boundaries filing) ----------------
C5CFG = {
    # battery 1: the overfill clause - the REREG dial under attack, 30 rounds
    "S5_RR_U05":      dict(bulk=0.05, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="rereg", noch=False, rounds=30),
    "S5_RR_U05_NOCH": dict(bulk=0.05, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="rereg", noch=True, rounds=30),
    "S5_RR_U10":      dict(bulk=0.10, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="rereg", noch=False, rounds=30),
    "S5_RR_U10_NOCH": dict(bulk=0.10, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="rereg", noch=True, rounds=30),
    # battery 2: the straddle - CAP30 and its anchor at twelve seeds
    "S5_CAP30":       dict(bulk=0.22, gfrac=0.08, mode="mixed",
                           eps_i=0.03, fix="none", noch=False, rounds=14,
                           seeds=list(range(600, 612))),
    "S5_CAP30_NOCH":  dict(bulk=0.22, gfrac=0.08, mode="mixed",
                           eps_i=0.03, fix="none", noch=True, rounds=14,
                           seeds=list(range(600, 612))),
    # battery 3: the half-buying defense - the repair_p dial at 0.05
    "S5_D100_05":     dict(bulk=0.05, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="none", noch=False, rounds=14,
                           rp=1.0),
    "S5_D50_05":      dict(bulk=0.05, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="none", noch=False, rounds=14,
                           rp=0.5),
    "S5_D25_05":      dict(bulk=0.05, gfrac=0.0, mode="uniform",
                           eps_i=0.03, fix="none", noch=False, rounds=14,
                           rp=0.25),
}
GROUPS = {
    "overfill": ["S5_RR_U05", "S5_RR_U05_NOCH", "S5_RR_U10", "S5_RR_U10_NOCH"],
    "straddle": ["S5_CAP30", "S5_CAP30_NOCH"],
    "halfdef": ["S5_D100_05", "S5_D50_05", "S5_D25_05"],
}
DOC51 = "/home/z/my-project/scripts/s4_candidate4_results.json"
DOC40 = "/home/z/my-project/scripts/m3_iota_dial_results.json"
DOC47 = "/home/z/my-project/scripts/m3_compcert_results.json"
AUDITS = [
    # (ours, theirs, ref doc key, rounds) - the p=1.0 path is the unmodified
    # channel: bitwise against doc 51's C4_SP05
    ("S5_D100_05", "C4_SP05", "51", 14),
    # the straddle arms' seeds 600-605 audit per-seed (the custom pass below)
]
REFPATHS = {"51": DOC51, "40": DOC40, "47": DOC47}
STRADDLE_AUDITS = [("S5_CAP30", "C4_CAP30"), ("S5_CAP30_NOCH", "C4_CAP30_NOCH")]
# the row->anchor pairing (the conditional's own-anchor rule)
ANCHOR = {
    "S5_RR_U05": "S5_RR_U05_NOCH", "S5_RR_U10": "S5_RR_U10_NOCH",
    "S5_CAP30": "S5_CAP30_NOCH",
    "S5_D100_05": "DOC51:C4_SP05_NOCH", "S5_D50_05": "DOC51:C4_SP05_NOCH",
    "S5_D25_05": "DOC51:C4_SP05_NOCH",
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
    cfg = C5CFG[arm]
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


def classify(res):
    """The pre-registered BD1-BD7 classification across the three
    batteries: the overfill rows (30 rounds, doc-40 twin references),
    the straddle (N=12), and the half-buying dial (doc 51's stored
    anchor and full-defense references)."""
    cls = {}
    A = res["arms"]
    d51 = json.load(open(DOC51))
    d40 = json.load(open(DOC40))
    d47 = json.load(open(DOC47))

    # the twin references
    twin_q14 = float(np.mean([x for x in
                              d51["arms"]["C4_TWIN"]["aggregate"]["q_target"]
                              [PERT_ROUND - 1:] if x is not None]))
    twin_acc14 = float(np.mean(
        [rr["traj"][-1]["acc"] for rr in d51["arms"]["C4_TWIN"]["runs"]]))
    twin_q30 = float(np.mean([x for x in
                              d40["arms"]["ID_TWIN_BASE"]["aggregate"]["q_target"]
                              [PERT_ROUND - 1:] if x is not None]))
    twin_acc30 = d40["arms"]["ID_TWIN_BASE"]["aggregate"]["acc"][-1]

    # BD1: the innocence audits
    cls["BD1_innocence"] = dict(res.get("audits", {}))
    cls["BD1_straddle_perseed"] = dict(res.get("straddle_audits", {}))

    # ---- BD2/BD3: the overfill rows ------------------------------------
    of = {}
    for arm, anch in [("S5_RR_U05", "S5_RR_U05_NOCH"),
                      ("S5_RR_U10", "S5_RR_U10_NOCH")]:
        if arm not in A or anch not in A:
            continue
        anchor_acc = float(np.mean(
            [rr["traj"][-1]["acc"] for rr in A[anch]["runs"]]))
        row = conditions(res, arm, anchor_acc, twin_q30, twin_acc30, 30)
        anch_gr = ground_readings(res, anch)
        row["anchor_ground"] = (anch_gr["verdict_mean"]
                                if anch_gr else None)
        row["anchor_ground_F"] = (anch_gr["F_late_mean"]
                                  if anch_gr else None)
        row["anchor_ground_per_seed"] = (anch_gr["per_seed_F_late"]
                                         if anch_gr else None)
        row["ground_delta"] = (row["ground_F_late"]
                               - anch_gr["F_late_mean"]
                               if anch_gr and row["ground_F_late"] is not None
                               else None)
        # the verdict under both letters
        if not row["threat"]:
            v5 = v6 = ("NO-THREAT" if row["(i)"] else "REFUSED-(i)")
        elif not row["(i)"]:
            v5 = v6 = "REFUSED-(i)"
        elif not row["(ii)"]:
            v5 = v6 = "REFUSED-(ii)"
        elif not row["(iii)_inherited"]:
            v5 = v6 = "REFUSED-(iii)"
        elif not row["(iv)"]:
            v5 = v6 = "REFUSED-(iv)"
        elif row["(v)"] is False:
            v5 = "REFUSED-(v)"
            v6 = ("REFUSED-(v)" if row["(vi)_candidate"] is not False
                  else "REFUSED-(v)+(vi)")
        else:
            v5 = "AWARDED-5COND"
            v6 = ("AWARDED-5COND" if row["(vi)_candidate"] is not False
                  else "REFUSED-(vi)")
        row["verdict_5cond"] = v5
        row["verdict_6cond"] = v6
        of[arm] = row
    cls["BD2_overfill_rows"] = of
    cls["BD2_test_cell_fires"] = bool(
        any(r["verdict_5cond"] == "AWARDED-5COND"
            and r["ground_verdict"] == "OVERFILLED"
            for r in of.values()))

    # BD3: the masking reading (the stored references)
    cls["BD3_masking_refs"] = {
        "doc47_REREG_twin_F_late": stored_ground(d47, "CM_T30_REREG"),
        "doc47_U05_F_late": stored_ground(d47, "CM_U05"),
        "doc47_U10_F_late": stored_ground(d47, "CM_U10"),
        "doc47_BASE_twin_F_late": stored_ground(d47, "CM_T30_BASE")}

    # ---- BD4: the straddle ----------------------------------------------
    st = {}
    if "S5_CAP30" in A and "S5_CAP30_NOCH" in A:
        gr = ground_readings(res, "S5_CAP30")
        anch_gr = ground_readings(res, "S5_CAP30_NOCH")
        accs = [rr["traj"][-1]["acc"] for rr in A["S5_CAP30"]["runs"]]
        anch_accs = [rr["traj"][-1]["acc"]
                     for rr in A["S5_CAP30_NOCH"]["runs"]]
        st = {"n_seeds": len(accs),
              "acc_mean": float(np.mean(accs)),
              "anchor_acc_mean": float(np.mean(anch_accs)),
              "F_late_mean": gr["F_late_mean"],
              "F_late_per_seed": gr["per_seed_F_late"],
              "verdict_mean": gr["verdict_mean"],
              "verdict_majority": gr["verdict_majority"],
              "straddle": gr["straddle"],
              "n_below_burn": gr["n_below_burn_floor"],
              "split": f"{gr['n_below_burn_floor']}/{len(gr['per_seed_F_late'])}",
              "anchor_ground": anch_gr["verdict_mean"],
              "anchor_F_late_mean": anch_gr["F_late_mean"],
              "ground_delta": gr["F_late_mean"] - anch_gr["F_late_mean"],
              "doc51_F_late_mean": -0.026863198374291228,
              "mean_governs_verdict": comp_verdict(gr["F_late_mean"]),
              "majority_counterfactual": gr["verdict_majority"],
              "founding_refusal_holds": bool(
                  gr["F_late_mean"] < COMP_BURN_FLOOR)}
    cls["BD4_straddle"] = st

    # ---- BD5/BD6: the half-buying dial -----------------------------------
    hd = {}
    anchor_acc05 = float(np.mean(
        [rr["traj"][-1]["acc"]
         for rr in d51["arms"]["C4_SP05_NOCH"]["runs"]]))
    full_eff = twin_acc14 - anchor_acc05
    for arm in ["S5_D100_05", "S5_D50_05", "S5_D25_05"]:
        if arm not in A:
            continue
        row = conditions(res, arm, anchor_acc05, twin_q14, twin_acc14, 14)
        p = C5CFG[arm].get("rp", 1.0)
        row["repair_p"] = p
        row["effect_fraction_of_full"] = (row["effect"]
                                          / (twin_acc14 - anchor_acc05))
        row["bar_inherited"] = EFFECT_BAR
        row["bar_rebased"] = BETA * full_eff
        row["beta_at_clear"] = (row["effect"] / full_eff
                                if full_eff else None)
        hd[arm] = row
    cls["BD5_p_curve"] = {a: {"p": hd[a]["repair_p"],
                              "effect": hd[a]["effect"],
                              "fraction": hd[a]["effect_fraction_of_full"],
                              "ground": hd[a]["ground_verdict"],
                              "armed_rounds": hd[a]["armed_rounds"]}
                          for a in hd}
    cls["BD6_dial"] = {a: {"effect": hd[a]["effect"],
                           "clears_inherited": hd[a]["(iii)_inherited"],
                           "clears_rebased": hd[a]["(iii)_rebased"],
                           "beta_at_clear": hd[a]["beta_at_clear"],
                           "ground": hd[a]["ground_verdict"]}
                       for a in hd}
    cls["BD5_hd_rows"] = hd

    # BD7: the table's integrity
    cls["BD7_integrity"] = {
        "any_MAINTAINS": False,
        "column3": "unmarked everywhere",
        "fresh_arms": ["S5_RR_U05", "S5_RR_U05_NOCH", "S5_RR_U10",
                       "S5_RR_U10_NOCH", "S5_D50_05", "S5_D25_05",
                       "the straddle's seeds 606-611"],
        "audit_arms": ["S5_D100_05 (bitwise to C4_SP05)",
                       "the twelve straddle per-seed audits"]}
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
    group = "all"
    for a in sys.argv:
        if a.startswith("--group="):
            group = a.split("=")[1]
        elif a == "--group":
            pass
    if smoke:
        globals()["SEEDS"] = [600, 601]
        for a in ["S5_RR_U05", "S5_RR_U05_NOCH", "S5_RR_U10",
                  "S5_RR_U10_NOCH"]:
            globals()["C5CFG"][a]["rounds"] = 8
        globals()["C5CFG"]["S5_CAP30"]["seeds"] = [600, 601]
        globals()["C5CFG"]["S5_CAP30_NOCH"]["seeds"] = [600, 601]
        globals()["GROUPS"]["overfill"] = ["S5_RR_U05", "S5_RR_U05_NOCH",
                                           "S5_RR_U10"]
        globals()["GROUPS"]["straddle"] = ["S5_CAP30", "S5_CAP30_NOCH"]
        globals()["GROUPS"]["halfdef"] = ["S5_D100_05", "S5_D50_05",
                                          "S5_D25_05"]
        group = "all"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s5_boundaries_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s5_boundaries_results.json")
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

    if group == "all":
        arms = GROUPS["overfill"] + GROUPS["straddle"] + GROUPS["halfdef"]
    elif group in GROUPS:
        arms = GROUPS[group]
    else:
        arms = GROUPS["overfill"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C5CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise audits: the p=1.0 control + the straddle per-seed --------
    if not smoke:
        have = [(o, t, k, r) for o, t, k, r in AUDITS if o in res["arms"]]
        if len(have) == len(AUDITS):
            res["audits"] = {}
            for ours, theirs, key, rounds_ref in AUDITS:
                res["audits"].update(
                    audit(res, REFPATHS[key], [(ours, theirs)],
                          rounds_ref))
        need_st = [a for a, _ in STRADDLE_AUDITS if a in res["arms"]]
        if len(need_st) == len(STRADDLE_AUDITS):
            res["straddle_audits"] = audit_straddle_perseed(
                res, DOC51, STRADDLE_AUDITS)

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
        need = set(GROUPS["overfill"] + GROUPS["straddle"]
                   + GROUPS["halfdef"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the overfill rows (30 rounds) ===")
            for arm, r in c["BD2_overfill_rows"].items():
                print(f"  {arm:14s} fork={r['fork']:9s} "
                      f"acc={r['acc_final']:.4f} "
                      f"anchor={r['anchor_acc']:.4f} "
                      f"threat={r['threat']!s:5s} "
                      f"effect={r['effect']:+.4f} "
                      f"ground={str(r['ground_verdict']):11s} "
                      f"(i){r['(i)']!s:5s} (ii){r['(ii)']!s:5s} "
                      f"(iii){r['(iii)_inherited']!s:5s} "
                      f"(iv){r['(iv)']!s:5s} (v){r['(v)']!s:5s} "
                      f"-> 5cond={r['verdict_5cond']} "
                      f"6cond={r['verdict_6cond']}")
            print("BD2 test cell fires:", c["BD2_test_cell_fires"])
            print("\n=== the straddle (N=12) ===")
            print("  ", c["BD4_straddle"])
            print("\n=== the half-buying dial ===")
            for arm, r in c["BD5_hd_rows"].items():
                print(f"  {arm:12s} p={r['repair_p']:.2f} "
                      f"acc={r['acc_final']:.4f} "
                      f"effect={r['effect']:+.4f} "
                      f"({r['effect_fraction_of_full']:.3f} of full) "
                      f"clears_rebased={r['(iii)_rebased']!s:5s} "
                      f"beta@clear={r['beta_at_clear']:.3f} "
                      f"ground={r['ground_verdict']}")
            print("BD3 refs:", c["BD3_masking_refs"])
            print("BD1 audits:", sum(1 for v in
                  res.get("audits", {}).values() if v), "/",
                  len(res.get("audits", {})),
                  "| straddle per-seed:",
                  sum(1 for v in res.get("straddle_audits", {}).values()
                      if v), "/", len(res.get("straddle_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
