#!/usr/bin/env python3
"""
THE b* GAP'S PROVENANCE - the registration's inputs disentangled at the
item level. Corpus doc 68's experiment. Ordered at the seventh-candidate
issuance: "the b* gap's provenance" - doc 60's threat (ii), stated in
its own words: "The upstream cause of b*'s seed-variance is unnamed -
the registration's inputs (r*, the calibration quantile path, the
round-5 perturbation draw) are logged and not disentangled; the round-5
elevation over rounds 1-4 is structural (all twelve seeds jump), but
the jump's magnitude is the draw, and the corpus has not decomposed
it."

Doc 60 established the baseline story: the burn's depth is the round-5
registration's height (corr -0.95), the b* distribution carries the
bimodality (nine seeds 0.045-0.074, a gap, then 603/604/606 at
0.094-0.106), and the registration is a clean-prefix quantity shared
bitwise with the healthy world. What doc 60 did NOT do: name the
upstream. The two registered paths, separable in the stored item logs:
  (i)  the r_star/quantile path - the calibration machinery that
       shapes the commit gate (r_star registered at burn-in end from
       pool0; tau_sys maintained by the quantile EMA; the commit set's
       composition therefore seed-dependent through the GATE, not the
       pool);
  (ii) the round-5 perturbation draw - the pool itself (which items
       drew which perturbation kind and severity at the registration
       round).
This battery re-runs S7_CAP30 (doc 60's own item-instrumented world,
itself bitwise doc 55's) a third time with PROVENANCE instrumentation
- passive, no rng consumed, no computation altered:

  per seed:  the unperturbed panel read (b_clean - the panel's blind
             fraction on the clean training set, the persistent
             vulnerability with NO draw in it at all);
  per round: the full-pool panel read (b_pool - the panel's blind
             fraction on the ENTIRE perturbed pool, committed or not:
             the draw's blinding power with the selection removed);
             the admission rates (the fraction of pool-blind items
             that committed vs the pool-clean's - the gate's blind
             concentration); the perturbation kind shares and the
             blind-rate BY KIND (the draw's fingerprint); the
             VIEWS-path commits (the flagged items rescued by the
             multi-view read); tau_sys/kappa_sys/r_star (the quantile
             path); and b* itself at the registration round.

THE DECOMPOSITION (pre-registered): b* = b_pool + L, where
L = b* - b_pool is the commit gate's blind concentration (the
selection's lift). The gap in the b* distribution (nothing between
0.0744 and 0.0938) must live in b_pool (the draw), in L (the gate),
in b_clean (the persistent panel state - the third upstream doc 60
did not name), or in a mixture - and the battery adjudicates.

PRE-REGISTERED PREDICTIONS (fixed before execution):
  BP1  THE AUDITS: all twelve seeds' tau/kappa trajectories reproduce
       doc 60's stored S7_CAP30 bitwise, full fourteen rounds (the
       instrumentation is passive; the chain 55 -> 60 -> 68; the
       determinism count passes three hundred thirty, 318 + 12).
  BP2  THE DECOMPOSITION AT THE REGISTRATION: per-seed b*, b_pool, and
       L at round 5. Forks: (i) the DRAW carries the gap (the deep
       three's b_pool clearly above the body's, L flat); (ii) the GATE
       carries it (L gaps, b_pool flat); (iii) both gap; (iv) neither
       gaps alone - the gap assembled from flat parts (only possible
       if b_clean separates, see BP5).
  BP3  THE DRAW'S FINGERPRINT: the kind shares at round 5 (noise/
       occlusion/shift) and the blind-rate by kind - does the deep
       three's round-5 draw carry more of the blinding kinds, or
       heavier noise (the sig distribution)?
  BP4  THE GATE'S FINGERPRINT: the admission rates at round 5 (blind
       vs clean), tau_sys/r_star, the commit rate, the VIEWS-path
       commits - does the deep three's gate admit blind items
       differentially?
  BP5  THE PERSISTENT COMPONENT: b_clean (the unperturbed panel read)
       per seed - does the panel's own vulnerability separate the
       groups before any draw is taken? corr(b*, b_clean) across the
       twelve seeds.
  BP6  THE STRUCTURAL ELEVATION: b_pool and b* at rounds 1-5 - the
       round-5 jump over the rounds-1-4 mean, decomposed into the
       draw's part (the b_pool jump) and the gate's part (the L jump);
       the windowed connection (doc 64: the gap survives w35/w14
       windowing - a persistent component's signature) re-read
       through the decomposition.
  BP7  THE VERDICT: the gap's provenance named, with the fractions of
       the deep three's elevation attributable to each path; the
       amendment's reading (what the windowed baseline does and does
       not average away).

Usage: python3 s8_bstarprov.py [--smoke]
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

# ---------------- the battery (the provenance replay) -------------------
C7CFG = {
    # doc 60's own world, re-run a third time with the provenance
    # instrumentation - bitwise doc 60's S7_CAP30, itself bitwise doc 55
    "S13_CAP30": dict(bulk=0.22, gfrac=0.08, mode="mixed",
                      eps_i=0.03, fix="none", noch=False, rounds=14,
                      seeds=list(range(600, 612))),
}
GROUPS = {"bstarprov": ["S13_CAP30"]}
DOC60 = "/home/z/my-project/scripts/s6_bimodal_results.json"
AUDITS = []
REFPATHS = {"60": DOC60}
# the per-seed bitwise audits: every seed vs doc 60's stored S7_CAP30,
# FULL fourteen rounds (the chain 55 -> 60 -> 68)
PERSEED_AUDITS = [("S13_CAP30", "S7_CAP30")]
# the row->anchor pairing (doc 55's stored anchor, loaded not re-run)
ANCHOR = {"S13_CAP30": "DOC55:S5_CAP30_NOCH"}


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


def perturb(X, rng, log=None):
    n = len(X)
    out = X.copy()
    kind = rng.integers(0, 3, n)
    sel = kind == 0
    sig = rng.uniform(0.05, 0.20, n)
    # the provenance replay's passive record: the draw's kinds and
    # severities, copied after the draws - no rng consumed, no value
    # altered (the audit is the proof)
    if log is not None:
        log["kind"] = kind.copy()
        log["sig"] = sig.copy()
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
# the provenance replay's pool log: set by run_arm before each round's
# self_round call; the pool draw records its kinds/severities here
_POOL_LOG = None


def self_round(state, rng, Xtr, ytr, tau_sys, kappa_sys):
    pool = perturb(Xtr, rng, log=_POOL_LOG)
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
    # the bimodality battery's passive extras: the commit mask and the
    # committee's pool read (pre-training) - no rng, no state change
    # (the provenance replay adds the flagged mask - same discipline)
    return (pool[commit], labels[commit], tele, pool, ytr[commit],
            commit, ybar, conf, flagged)


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
    cfg = C7CFG[arm]
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

    # ---- the provenance replay's persistent read (no rng, once) ----------
    # the panel's blind fraction on the UNPERTURBED training set: the
    # persistent vulnerability with no draw in it at all
    global _POOL_LOG
    _, pconf_clean = panel_grade(Xtr, panel)
    b_clean = float(np.mean(pconf_clean < TAU0))

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
    ilogs = []
    plogs = []
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
        _POOL_LOG = {}
        (Xt_, yt_, tele, pool, yt_true, commit_m, ybar_pool,
         conf_pool, flagged_m) = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        poollog = _POOL_LOG
        _POOL_LOG = None
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
                rep_c_mask = disputed.copy()
            elif rp > 0.0:
                # the half-buying dial: each disputed label repaired with
                # probability rp; one full-length coin draw per armed round
                # from the separate stream (the chassis rng untouched)
                coin = rng_repair.random(len(yt_p))
                sel = disputed & (coin < rp)
                yt_train[sel] = pl[sel]
                n_rep = int(sel.sum())
                rep_c_mask = sel.copy()
            else:
                rep_c_mask = np.zeros(len(yt_p), dtype=bool)
            X_train, yt_true_train = Xt_, yt_true
        else:
            X_train, yt_train, yt_true_train = Xt_, yt_p, yt_true
            n_rep = 0
            rep_c_mask = np.zeros(len(yt_p), dtype=bool)
        tele["cerr_stream_post"] = (float(np.mean(yt_train != yt_true_train))
                                    if len(yt_train) else 0.0)
        tele["panel_acc"] = (float(np.mean(pl == yt_true))
                             if len(yt_p) else 0.0)

        # ---- 3b. the item-level log (passive; BM2-BM6's data) ----------
        # identities are Xtr indices; the panel's blind read is of the
        # PERTURBED committed pool (an item may read blind one round and
        # confident the next - the perturbation is redrawn per round)
        idx_commit = np.where(commit_m)[0]
        blind_c = pconf < TAU0
        wrong_c = ybar_pool[commit_m] != ytr[commit_m]
        ilogs.append({
            "r": r,
            "n_commit": int(commit_m.sum()),
            "n_blind": int(blind_c.sum()),
            "blind_idx": idx_commit[blind_c].tolist(),
            "gate_idx": idx_commit[gate_m].tolist(),
            "bulk_idx": idx_commit[bulk_m].tolist(),
            "rep_idx": idx_commit[rep_c_mask].tolist(),
            "blind_conf": conf_pool[commit_m][blind_c].tolist(),
            "blind_wrong": wrong_c[blind_c].tolist(),
            "wrong_rate": (float(np.mean(wrong_c))
                           if len(wrong_c) else 0.0),
            "conf_mean": (float(np.mean(conf_pool[commit_m]))
                          if commit_m.any() else 0.0)})

        # ---- 3c. THE PROVENANCE LOG (passive; BP2-BP6's data) --------
        # the full-pool panel read (the draw's blinding power with the
        # selection removed), the admission rates (the gate's blind
        # concentration), the draw's kind fingerprint, the VIEWS-path
        # commits, and the quantile path - no rng, no state change
        _, pconf_pool = panel_grade(pool, panel)
        blind_pool = pconf_pool < TAU0
        b_pool = float(np.mean(blind_pool))
        adm_blind = (float(np.mean(commit_m[blind_pool]))
                     if blind_pool.any() else None)
        adm_clean = float(np.mean(commit_m[~blind_pool]))
        kind = poollog["kind"]
        sig = poollog["sig"]
        idx_c = np.where(commit_m)[0]
        views_c = flagged_m[idx_c] if len(idx_c) else np.zeros(0, bool)
        blind_committed = pconf < TAU0
        plogs.append({
            "r": r, "b_pool": b_pool, "b_star_reg": (b_r
                          if r == COMP_REG_END else None),
            "lift_L": ((b_r - b_pool) if r == COMP_REG_END else None),
            "adm_blind": adm_blind, "adm_clean": adm_clean,
            "kind_share": {str(kk): float(np.mean(kind == kk))
                           for kk in (0, 1, 2)},
            "blind_by_kind": {
                str(kk): (float(np.mean(blind_pool[kind == kk]))
                          if (kind == kk).any() else None)
                for kk in (0, 1, 2)},
            "sig_mean_kind0": (float(np.mean(sig[kind == 0]))
                               if (kind == 0).any() else None),
            "tau_sys": float(tau_sys), "kappa_sys": float(kappa_sys),
            "r_star": r_star, "n_commit": int(commit_m.sum()),
            "n_blind_committed": int(blind_committed.sum()),
            "n_views_committed": int(views_c.sum()),
            "n_blind_views_committed": int(
                (views_c & blind_committed).sum()),
            "b_clean": b_clean})

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
    return traj, ilogs, plogs, b_clean


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
    """The pre-registered BP1-BP7 classification: the b* gap's
    provenance, read from the provenance logs of the twelve bitwise
    doc-60 worlds."""
    cls = {}
    A = res["arms"]
    runs = A["S13_CAP30"]["runs"]

    # BP1: the per-seed audits
    cls["BP1_audits"] = dict(res.get("perseid_audits", {}))

    # ---- the per-seed provenance table ------------------------------------
    deep = [603, 604, 606]          # doc 60's deep cluster
    marg = [600, 607]               # doc 60's marginals
    seeds = {}
    for rr in runs:
        s = rr["seed"]
        pl = rr["plogs"]
        tr = rr["traj"]
        n = len(tr)
        p5 = pl[COMP_REG_END - 1]           # round 5, the registration
        late = [t["comp_F"] for t in tr
                if t["comp_F"] is not None and t["round"] >= n - 3]
        f_late = float(np.mean(late))
        b_star = p5["b_star_reg"]
        b_pool5 = p5["b_pool"]
        # BP6: the elevation - rounds 1-5 of b_pool and b_committed
        b_pools = {p["r"]: p["b_pool"] for p in pl if p["r"] <= 5}
        b_comms = {p["r"]: (p["n_blind_committed"] / p["n_commit"]
                            if p["n_commit"] else None)
                   for p in pl if p["r"] <= 5}
        b14_pool = float(np.mean([b_pools[r] for r in range(1, 5)]))
        b14_comm = float(np.mean([b_comms[r] for r in range(1, 5)]))
        # the w35 window's pool-mean (the windowed baseline's draw part)
        w35_pool = float(np.mean([b_pools[r] for r in (3, 4, 5)]))
        seeds[s] = {
            "F_late": f_late, "verdict": comp_verdict(f_late),
            "b_star": b_star, "b_pool5": b_pool5,
            "L5": p5["lift_L"], "b_clean": p5["b_clean"],
            "adm_blind5": p5["adm_blind"], "adm_clean5": p5["adm_clean"],
            "kind5": p5["kind_share"], "blind_by_kind5": p5["blind_by_kind"],
            "sig_mean_kind0_5": p5["sig_mean_kind0"],
            "tau5": p5["tau_sys"], "r_star": p5["r_star"],
            "commit_rate5": (p5["n_commit"] / 1347.0),
            "n_views5": p5["n_views_committed"],
            "n_blind_views5": p5["n_blind_views_committed"],
            "b_pool_r14mean": b14_pool, "b_comm_r14mean": b14_comm,
            "b_pool_jump": b_pool5 - b14_pool,
            "b_comm_jump": b_comms[5] - b14_comm,
            "w35_pool": w35_pool,
            "b_pools_1to5": {str(r): b_pools[r] for r in range(1, 6)},
            "b_comms_1to5": {str(r): b_comms[r] for r in range(1, 6)}}
    cls["BP_seeds"] = seeds

    body = [s for s in seeds if s not in deep + marg]

    def grp(key, ss):
        vals = [seeds[s][key] for s in ss
                if seeds[s][key] is not None]
        return float(np.mean(vals)) if vals else None

    def sd(key, ss):
        vals = [seeds[s][key] for s in ss
                if seeds[s][key] is not None]
        return float(np.std(vals)) if vals else None

    # ---- BP2: the decomposition at the registration ----------------------
    def gap(key):
        return (grp(key, deep) - grp(key, body)) if grp(key, body) \
            is not None else None

    cls["BP2_decomposition"] = {
        "groups": {"deep": deep, "marginal": marg, "body": body},
        "b_star": {"deep": grp("b_star", deep), "body": grp("b_star", body),
                   "marg": grp("b_star", marg)},
        "b_pool5": {"deep": grp("b_pool5", deep),
                    "body": grp("b_pool5", body),
                    "marg": grp("b_pool5", marg)},
        "L5": {"deep": grp("L5", deep), "body": grp("L5", body),
               "marg": grp("L5", marg)},
        "b_clean": {"deep": grp("b_clean", deep),
                    "body": grp("b_clean", body),
                    "marg": grp("b_clean", marg)},
        "gaps": {"b_star": gap("b_star"), "b_pool5": gap("b_pool5"),
                 "L5": gap("L5"), "b_clean": gap("b_clean")},
        "sds": {"b_star": sd("b_star", seeds), "b_pool5": sd("b_pool5", seeds),
                "L5": sd("L5", seeds), "b_clean": sd("b_clean", seeds)},
        "fork": None}   # filled below

    g = cls["BP2_decomposition"]["gaps"]
    sds = cls["BP2_decomposition"]["sds"]
    draw_carries = (g["b_pool5"] is not None
                    and abs(g["b_pool5"]) > 2.0 * sds["b_pool5"])
    gate_carries = (g["L5"] is not None
                    and abs(g["L5"]) > 2.0 * sds["L5"])
    clean_carries = (g["b_clean"] is not None
                     and abs(g["b_clean"]) > 2.0 * sds["b_clean"])
    if draw_carries and not gate_carries:
        cls["BP2_decomposition"]["fork"] = \
            "(i) THE DRAW carries the gap (b_pool gaps, L flat)"
    elif gate_carries and not draw_carries:
        cls["BP2_decomposition"]["fork"] = \
            "(ii) THE GATE carries the gap (L gaps, b_pool flat)"
    elif draw_carries and gate_carries:
        cls["BP2_decomposition"]["fork"] = \
            "(iii) BOTH gap (draw and gate)"
    else:
        cls["BP2_decomposition"]["fork"] = \
            ("(iv) neither alone - the gap assembled from flat parts"
             + ("; b_clean separates" if clean_carries
                else "; no single component separates"))

    # ---- BP3: the draw's fingerprint --------------------------------------
    cls["BP3_draw_fingerprint"] = {
        "note": "computed per-seed in BP_seeds; summarized here",
        "kind_occlusion_share": {"deep": float(np.mean(
            [seeds[s]["kind5"]["1"] for s in deep])),
            "body": float(np.mean(
                [seeds[s]["kind5"]["1"] for s in body]))},
        "kind_noise_share": {"deep": float(np.mean(
            [seeds[s]["kind5"]["0"] for s in deep])),
            "body": float(np.mean(
                [seeds[s]["kind5"]["0"] for s in body]))},
        "kind_shift_share": {"deep": float(np.mean(
            [seeds[s]["kind5"]["2"] for s in deep])),
            "body": float(np.mean(
                [seeds[s]["kind5"]["2"] for s in body]))},
        "blind_by_kind_deep": {"occlusion": float(np.mean(
            [seeds[s]["blind_by_kind5"]["1"] for s in deep])),
            "noise": float(np.mean(
                [seeds[s]["blind_by_kind5"]["0"] for s in deep])),
            "shift": float(np.mean(
                [seeds[s]["blind_by_kind5"]["2"] for s in deep]))},
        "blind_by_kind_body": {"occlusion": float(np.mean(
            [seeds[s]["blind_by_kind5"]["1"] for s in body])),
            "noise": float(np.mean(
                [seeds[s]["blind_by_kind5"]["0"] for s in body])),
            "shift": float(np.mean(
                [seeds[s]["blind_by_kind5"]["2"] for s in body]))},
        "sig_mean_kind0": {"deep": grp("sig_mean_kind0_5", deep),
                           "body": grp("sig_mean_kind0_5", body)}}

    # ---- BP4: the gate's fingerprint --------------------------------------
    cls["BP4_gate_fingerprint"] = {
        "adm_blind5": {"deep": grp("adm_blind5", deep),
                       "body": grp("adm_blind5", body)},
        "adm_clean5": {"deep": grp("adm_clean5", deep),
                       "body": grp("adm_clean5", body)},
        "admission_ratio_deep": ((grp("adm_blind5", deep)
                                  / grp("adm_clean5", deep))
                                 if grp("adm_clean5", deep) else None),
        "admission_ratio_body": ((grp("adm_blind5", body)
                                  / grp("adm_clean5", body))
                                 if grp("adm_clean5", body) else None),
        "tau5": {"deep": grp("tau5", deep), "body": grp("tau5", body)},
        "r_star": {"deep": grp("r_star", deep),
                   "body": grp("r_star", body)},
        "commit_rate5": {"deep": grp("commit_rate5", deep),
                         "body": grp("commit_rate5", body)},
        "n_views5": {"deep": grp("n_views5", deep),
                     "body": grp("n_views5", body)},
        "n_blind_views5": {"deep": grp("n_blind_views5", deep),
                           "body": grp("n_blind_views5", body)}}

    # ---- BP5: the persistent component ------------------------------------
    bstars = [seeds[s]["b_star"] for s in seeds]
    bcleans = [seeds[s]["b_clean"] for s in seeds]
    bpools = [seeds[s]["b_pool5"] for s in seeds]
    Ls = [seeds[s]["L5"] for s in seeds]
    cls["BP5_persistent"] = {
        "corr_bstar_bclean": float(np.corrcoef(bstars, bcleans)[0, 1]),
        "corr_bstar_bpool": float(np.corrcoef(bstars, bpools)[0, 1]),
        "corr_bstar_L": float(np.corrcoef(bstars, Ls)[0, 1]),
        "b_clean_sorted": {str(s): seeds[s]["b_clean"] for s in seeds},
        "b_clean_deep_above_body_median": bool(all(
            seeds[s]["b_clean"] > float(np.median(
                [seeds[t]["b_clean"] for t in body])) for s in deep))}

    # ---- BP6: the structural elevation ------------------------------------
    cls["BP6_elevation"] = {
        "b_pool_jump": {"deep": grp("b_pool_jump", deep),
                        "body": grp("b_pool_jump", body)},
        "b_comm_jump": {"deep": grp("b_comm_jump", deep),
                        "body": grp("b_comm_jump", body)},
        "w35_pool": {"deep": grp("w35_pool", deep),
                     "body": grp("w35_pool", body)},
        "w35_gap": gap("w35_pool"),
        "per_seed": {str(s): {"b_pools": seeds[s]["b_pools_1to5"],
                              "b_comms": seeds[s]["b_comms_1to5"]}
                     for s in seeds},
        "read": "the round-5 jump over rounds 1-4, split into the"
                " draw's part (b_pool) and the committed read's jump"}

    # ---- BP7: the verdict --------------------------------------------------
    cls["BP7_verdict"] = {
        "fork": cls["BP2_decomposition"]["fork"],
        "clean_carries": clean_carries,
        "corr_bstar_bclean": cls["BP5_persistent"]["corr_bstar_bclean"],
        "note": "the fractions of the deep three's b* elevation "
                "attributable to the draw (b_pool), the gate (L), and "
                "the persistent panel state (b_clean), read against "
                "the corpus's windowed-baseline finding (doc 64: the "
                "gap survives windowing)"}
    return cls


def audit_perseed(res, ref_path, pairs):
    """The per-seed audits: each seed's tau/kappa trajectory must
    reproduce the reference's stored run for the same arm and seed,
    to 1e-9 - the worlds are literally the same."""
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
        for a in GROUPS["bstarprov"]:
            globals()["C7CFG"][a]["seeds"] = [600, 601]
            globals()["C7CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s8_bstarprov_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s8_bstarprov_results.json")
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
            "pert_round": PERT_ROUND, "nseed": 12,
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

    arms = GROUPS["bstarprov"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C7CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj, ilogs, plogs, b_clean = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj, "items": ilogs,
                         "plogs": plogs, "b_clean": b_clean})
            print(f"  {arm} seed={s} done (b_clean={b_clean:.4f}) "
                  f"({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- the per-seed bitwise audits vs doc 60's stored runs -------------
    if not smoke:
        need_ps = [a for a, _ in PERSEED_AUDITS if a in res["arms"]]
        if len(need_ps) == len(PERSEED_AUDITS):
            res["perseid_audits"] = audit_perseed(
                res, DOC60, PERSEED_AUDITS)

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
        need = set(GROUPS["bstarprov"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the per-seed provenance table ===")
            hdr = ["seed", "b*", "b_pool5", "L5", "b_clean",
                   "adm_bl", "adm_cl", "tau5", "r_star", "views",
                   "bl_views", "F_late"]
            print(("{:>5s} {:>7s} {:>8s} {:>7s} {:>8s} {:>7s} "
                   "{:>7s} {:>7s} {:>7s} {:>6s} {:>9s} {:>8s}").format(
                *hdr))
            for s, r in c["BP_seeds"].items():
                def f4(v):
                    return f"{v:+.4f}" if isinstance(v, float) else "-"
                def f4p(v):
                    return f"{v:.4f}" if isinstance(v, float) else "-"
                def fd(v):
                    return f"{v:d}" if isinstance(v, int) else "-"
                print(("{:>5d} {:>7s} {:>8s} {:>7s} {:>8s} {:>7s} "
                       "{:>7s} {:>7s} {:>7s} {:>6s} {:>9s} {:>8s}").format(
                    s, f4p(r["b_star"]), f4p(r["b_pool5"]),
                    f4(r["L5"]), f4p(r["b_clean"]),
                    f4p(r["adm_blind5"]), f4p(r["adm_clean5"]),
                    f4p(r["tau5"]), f4p(r["r_star"]),
                    fd(r["n_views5"]), fd(r["n_blind_views5"]),
                    f4(r["F_late"])))
            print("\nBP2 decomposition:", c["BP2_decomposition"])
            print("\nBP3 draw fingerprint:", c["BP3_draw_fingerprint"])
            print("\nBP4 gate fingerprint:", c["BP4_gate_fingerprint"])
            print("\nBP5 persistent:", c["BP5_persistent"])
            print("\nBP6 elevation:", c["BP6_elevation"])
            print("\nBP7 verdict:", c["BP7_verdict"])
            print("BP1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
