#!/usr/bin/env python3
"""
THE TEMPERATURE'S OWN REGISTRATION - the committee-panel relationship
made a first-class column. Corpus doc 72's experiment. Ordered at the
eighth issuance: "the temperature's own registration" - doc 68's filed
candidate, stated in its own words: "registering the temperature
itself (the admission-rate differential against the panel, read on
the clean prefix) is the natural candidate, and it is filed as
exactly that: a candidate, priced by this battery's numbers, ordered
or not by the registrar the corpus answers to."

Doc 68 named the b* gap's upstream - the committee-panel
relationship, a persistent seed-level temperature expressed through
the gate's blind-admission rate - but measured it only AT the
registration round (round 5, adm_blind 0.180-0.219 deep against
0.094-0.147 body, no overlap) and only POST HOC, from the provenance
logs. This battery makes the temperature a REGISTERED quantity, in
the chassis's registration layer itself:

  T  := the rounds-1-4 mean of the per-round blind-admission rate -
        the committee's clearance rate on the panel's blind region
        during the clean prefix, BEFORE the registration round and
        before any poison (rounds 1-5 are pre-attack: the poison
        arms at round 6). T is computable in the healthy world.
  D  := the rounds-1-4 mean of (admission rate on panel-clean items)
        minus T - the differential reading, the gate's blind-vs-clean
        admission gap on the same prefix.
Both are registered at round 5 (COMP_REG_END), alongside b*, into
the run's registration record; no rng consumed, no computation
altered - the world is doc 60's own (S16_TEMP30 = S7_CAP30 =
S5_CAP30, the fourth generation), and all twelve seeds must audit
bitwise against doc 60's stored runs across the full fourteen
rounds (the chain 55 -> 60 -> 68 -> 72; the determinism count
passes four hundred twenty-six, 414 + 12).

PRE-REGISTERED PREDICTIONS (fixed before execution):
  TR1  THE AUDITS: all twelve seeds' tau/kappa trajectories reproduce
       doc 60's stored S7_CAP30 bitwise, full fourteen rounds; AND
       the registration reproduces itself - this run's per-seed T
       agrees with the T implied by doc 68's stored provenance logs
       to 1e-9 (the same world, the passive instrumentation).
  TR2  THE REGISTRATION ITSELF: T per seed; the separation (the deep
       three against the body - no overlap at the prefix, as at
       round 5?); the climb (the round-5 admission rate over T); the
       differential D's behaviour (flat like adm_clean, or
       separating like T?).
  TR3  THE FORECASTING: corr(T, b*), corr(T, F_late), corr(T,
       b_clean) across the twelve seeds; the burn verdicts against T
       (do the five burning seeds all sit above the body's median
       T?); T's advantage over b_clean (doc 68's 1.3-sd
       non-separation).
  TR4  THE VERDICT STRATIFICATION: the b*-on-T regression (slope,
       intercept, residual sd against the raw sd - the share of the
       registration's seed-variance the temperature absorbs); the
       deep three's b* elevation after the T-correction (does the
       bimodality survive the covariate?); the T-banded reading.
  TR5  WHAT THE REGISTRATION BUYS AND COSTS: the before-the-poison
       availability (T from the clean prefix - the registrar can
       read the thermometer before the attack arrives); the
       differential's robustness re-read (the anchor subtracts T
       bitwise - doc 68's mechanism, now standing on a registered
       column); the costs (T's own sampling noise, the rounds-1-4
       mean's spread; the draw's interaction margin that T does not
       carry).
  TR6  THE NON-CLAIMS: no amendment executed, no registration
       changed, no letter moved - T is registered ALONGSIDE b*, not
       in place of it; the ~30% draw-interaction margin of the gap
       remains attributed to no mechanism; the causal reading stays
       an interpretation (no battery has manipulated T).

Usage: python3 s10_tempreg.py [--smoke]
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

# ---------------- the battery (the temperature registration) -------------
C9CFG = {
    # doc 60's own world, re-run a FOURTH time with the temperature
    # registered - bitwise doc 60's S7_CAP30 (itself bitwise doc 55's
    # S5_CAP30 and doc 68's S13_CAP30)
    "S16_TEMP30": dict(bulk=0.22, gfrac=0.08, mode="mixed",
                       eps_i=0.03, fix="none", noch=False, rounds=14,
                       seeds=list(range(600, 612))),
}
GROUPS = {"tempreg": ["S16_TEMP30"]}
DOC60 = "/home/z/my-project/scripts/s6_bimodal_results.json"
DOC68 = "/home/z/my-project/scripts/s8_bstarprov_results.json"
AUDITS = []
REFPATHS = {"60": DOC60, "68": DOC68}
# the per-seed bitwise audits: every seed vs doc 60's stored S7_CAP30,
# FULL fourteen rounds (the chain 55 -> 60 -> 68 -> 72)
PERSEED_AUDITS = [("S16_TEMP30", "S7_CAP30")]
# the row->anchor pairing (doc 55's stored anchor, loaded not re-run)
ANCHOR = {"S16_TEMP30": "DOC55:S5_CAP30_NOCH"}


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
    cfg = C9CFG[arm]
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
        # ---- 3d. THE TEMPERATURE'S REGISTRATION (passive; doc 72's layer) --
        # T = the rounds-1-4 mean of the per-round blind-admission rate,
        # registered at round 5 alongside b*; D = the rounds-1-4 clean-
        # admission mean minus T (the differential reading). No rng, no
        # state change - the registration layer only reads what the
        # provenance log already holds (the audit is the proof).
        T_reg = D_reg = None
        if r == COMP_REG_END:
            adm14 = [p["adm_blind"] for p in plogs
                     if p["r"] <= r - 1 and p["adm_blind"] is not None]
            cl14 = [p["adm_clean"] for p in plogs
                    if p["r"] <= r - 1 and p["adm_clean"] is not None]
            T_reg = (float(np.mean(adm14)) if adm14 else None)
            D_reg = (float(np.mean(cl14)) - T_reg
                     if (cl14 and T_reg is not None) else None)
        plogs.append({
            "r": r, "b_pool": b_pool, "b_star_reg": (b_r
                          if r == COMP_REG_END else None),
            "lift_L": ((b_r - b_pool) if r == COMP_REG_END else None),
            "T_reg": T_reg, "D_reg": D_reg,
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
    """The pre-registered TR1-TR6 classification: the temperature's own
    registration, read from the fourth generation of doc 60's world."""
    cls = {}
    A = res["arms"]
    runs = A["S16_TEMP30"]["runs"]

    # ---- TR1: the audits, and the registration's self-agreement --------
    cls["TR1_audits"] = dict(res.get("perseid_audits", {}))
    d68 = json.load(open(DOC68))
    d68_map = {rr["seed"]: rr for rr in d68["arms"]["S13_CAP30"]["runs"]}
    agree = {}
    for rr in runs:
        s = rr["seed"]
        if s not in d68_map:
            continue
        pl68 = d68_map[s]["plogs"]
        adm14 = [p["adm_blind"] for p in pl68
                 if p["r"] <= 4 and p["adm_blind"] is not None]
        T68 = float(np.mean(adm14)) if adm14 else None
        T_new = rr["plogs"][COMP_REG_END - 1]["T_reg"]
        agree[str(s)] = bool(T68 is not None and T_new is not None
                             and abs(T68 - T_new) < 1e-9)
    cls["TR1_T_agreement_with_doc68"] = agree
    cls["TR1_summary"] = {
        "bitwise_audits_vs_doc60": bool(
            all(res.get("perseid_audits", {}).values())
            and len(res.get("perseid_audits", {})) == 12),
        "T_agreement_all": bool(agree) and all(agree.values()),
        "n_agree": int(sum(1 for v in agree.values() if v)),
        "n_disagree": int(sum(1 for v in agree.values() if not v))}

    # ---- the per-seed registration table ---------------------------------
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
        adm_path = {p["r"]: p["adm_blind"] for p in pl if p["r"] <= 5}
        T = p5["T_reg"]
        seeds[s] = {
            "T_reg": T, "D_reg": p5["D_reg"],
            "b_star": p5["b_star_reg"], "b_pool5": p5["b_pool"],
            "L5": p5["lift_L"], "b_clean": p5["b_clean"],
            "adm_blind5": p5["adm_blind"], "adm_clean5": p5["adm_clean"],
            "climb": ((p5["adm_blind"] - T)
                      if (T is not None
                          and p5["adm_blind"] is not None) else None),
            "adm_path_1to5": {str(r): adm_path.get(r)
                              for r in range(1, 6)},
            "F_late": f_late, "verdict": comp_verdict(f_late),
            "burned": bool(f_late < COMP_BURN_FLOOR)}
    cls["TR_seeds"] = seeds
    body = [s for s in seeds if s not in deep + marg]

    def grp(key, ss):
        vals = [seeds[s][key] for s in ss if seeds[s][key] is not None]
        return float(np.mean(vals)) if vals else None

    def rng_(key, ss):
        vals = [seeds[s][key] for s in ss if seeds[s][key] is not None]
        return (float(min(vals)), float(max(vals))) if vals else None

    # ---- TR2: the registration itself ------------------------------------
    Td = [seeds[s]["T_reg"] for s in deep if seeds[s]["T_reg"] is not None]
    Tb = [seeds[s]["T_reg"] for s in body if seeds[s]["T_reg"] is not None]
    Tm = [seeds[s]["T_reg"] for s in marg if seeds[s]["T_reg"] is not None]
    no_overlap = bool(Td and Tb and max(Tb) < min(Td))
    cls["TR2_registration"] = {
        "T_deep": {"mean": grp("T_reg", deep), "range": rng_("T_reg", deep)},
        "T_body": {"mean": grp("T_reg", body), "range": rng_("T_reg", body)},
        "T_marg": {"mean": grp("T_reg", marg), "values": Tm},
        "no_overlap_deep_body": no_overlap,
        "margin": (min(Td) - max(Tb)) if (Td and Tb) else None,
        "D_deep": grp("D_reg", deep), "D_body": grp("D_reg", body),
        "D_reading": ("the differential flat (like the clean-admission)"
                      if (grp("D_reg", deep) is not None
                          and grp("D_reg", body) is not None
                          and abs(grp("D_reg", deep)
                                  - grp("D_reg", body)) < 0.05)
                      else "the differential separating"),
        "climb_deep": grp("climb", deep), "climb_body": grp("climb", body),
        "note": "T registered at round 5 from the rounds-1-4 path; the "
                "climb is the round-5 admission rate over T"}

    # ---- TR3: the forecasting --------------------------------------------
    Ts = [seeds[s]["T_reg"] for s in seeds]
    bs = [seeds[s]["b_star"] for s in seeds]
    fl = [seeds[s]["F_late"] for s in seeds]
    bc = [seeds[s]["b_clean"] for s in seeds]
    ad5 = [seeds[s]["adm_blind5"] for s in seeds]
    burning = [s for s in seeds if seeds[s]["burned"]]
    T_body_median = float(np.median(Tb)) if Tb else None
    burning_above = [bool(seeds[s]["T_reg"] > T_body_median)
                     for s in burning]
    # b_clean's separation at the same criterion (doc 68's 1.3-sd reading)
    bcd = [seeds[s]["b_clean"] for s in deep]
    bcb = [seeds[s]["b_clean"] for s in body]
    sd_all = float(np.std(bc))
    cls["TR3_forecasting"] = {
        "corr_T_bstar": float(np.corrcoef(Ts, bs)[0, 1]),
        "corr_T_Flate": float(np.corrcoef(Ts, fl)[0, 1]),
        "corr_T_bclean": float(np.corrcoef(Ts, bc)[0, 1]),
        "corr_T_adm5": float(np.corrcoef(Ts, ad5)[0, 1]),
        "corr_adm5_bstar": float(np.corrcoef(ad5, bs)[0, 1]),
        "corr_bclean_bstar": float(np.corrcoef(bc, bs)[0, 1]),
        "burning_seeds": burning,
        "burning_all_above_body_median_T": bool(burning_above)
        and all(burning_above),
        "bclean_gap_in_sds": ((float(np.mean(bcd)) - float(np.mean(bcb)))
                              / sd_all if sd_all else None),
        "T_gap_in_T_sds": ((float(np.mean(Td)) - float(np.mean(Tb)))
                           / float(np.std(Ts)) if Ts else None)}

    # ---- TR4: the verdict stratification ---------------------------------
    slope, icept = np.polyfit(Ts, bs, 1)
    fitted = [slope * t + icept for t in Ts]
    resid = [b - f for b, f in zip(bs, fitted)]
    rd = [resid[list(seeds).index(s)] for s in deep]
    rb = [resid[list(seeds).index(s)] for s in body]
    raw_d = [seeds[s]["b_star"] for s in deep]
    raw_b = [seeds[s]["b_star"] for s in body]
    var_abs = 1.0 - float(np.var(resid)) / float(np.var(bs))
    # the bimodality's survival: the largest within-group gap in the
    # sorted residuals against the sorted raw b*'s doc-60 gap
    rs_sorted = sorted(resid)
    res_gaps = [(rs_sorted[i + 1] - rs_sorted[i],
                 (rs_sorted[i], rs_sorted[i + 1]))
                for i in range(len(rs_sorted) - 1)]
    res_maxgap = max(res_gaps) if res_gaps else None
    cls["TR4_stratification"] = {
        "regression": {"slope": float(slope), "intercept": float(icept),
                       "residual_sd": float(np.std(resid)),
                       "raw_bstar_sd": float(np.std(bs)),
                       "variance_absorbed_by_T": var_abs},
        "deep_elevation_raw": float(np.mean(raw_d) - np.mean(raw_b)),
        "deep_elevation_Tcorrected": float(np.mean(rd) - np.mean(rb)),
        "bimodality_after_T": {
            "max_residual_gap": (float(res_maxgap[0])
                                 if res_maxgap else None),
            "at": ([float(x) for x in res_maxgap[1]]
                   if res_maxgap else None),
            "doc60_raw_gap": 0.0938 - 0.0744,
            "reading": ("the bimodality DISSOLVED under the covariate"
                        if (res_maxgap and res_maxgap[0] < 0.012)
                        else "the bimodality SURVIVES the covariate"
                        if res_maxgap else None)},
        "per_seed_residuals": {str(s): float(r)
                               for s, r in zip(seeds, resid)}}

    # ---- TR5: what the registration buys and costs -----------------------
    cls["TR5_buy_cost"] = {
        "availability": "T is a rounds-1-4 quantity, registered at round "
                        "5 - before the poison's round 6, computable in "
                        "the healthy world (the registrar reads the "
                        "thermometer before the attack arrives)",
        "the_differential": "the anchor shares the committee and the "
                            "panel bitwise through the clean prefix - "
                            "the subtraction cancels T (doc 68's "
                            "mechanism, now standing on a registered "
                            "column); the level stands on it",
        "costs": {
            "T_sampling_spread_within_body": (
                float(np.std(Tb)) if Tb else None),
            "unexplained_share": 1.0 - var_abs,
            "note": "the draw's interaction margin (~30% of the gap, "
                    "doc 68) is not T's to explain; T's own noise is "
                    "the rounds-1-4 mean's spread"}}

    # ---- TR6: the non-claims (static) -------------------------------------
    cls["TR6_nonclaims"] = {
        "no_amendment": "no registration changed, no letter moved - T "
                        "registered ALONGSIDE b*, not in place of it",
        "no_causal_claim": "the temperature's causal role stays an "
                           "interpretation - no battery has manipulated "
                           "T (a different burn-in, a re-frozen panel)",
        "integrity": {"any_MAINTAINS": False,
                      "column3": "unmarked",
                      "audit": "twelve per-seed bitwise audits vs doc "
                               "60's stored S7_CAP30, full fourteen "
                               "rounds; the T-agreement vs doc 68's "
                               "stored logs to 1e-9"}}
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
        for a in GROUPS["tempreg"]:
            globals()["C9CFG"][a]["seeds"] = [600, 601]
            globals()["C9CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s10_tempreg_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s10_tempreg_results.json")
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

    arms = GROUPS["tempreg"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C9CFG[arm].get("seeds", SEEDS)
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
        need = set(GROUPS["tempreg"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the temperature's own registration ===")
            hdr = ["seed", "T_reg", "D_reg", "climb", "adm_bl5",
                   "b*", "b_clean", "F_late", "verd"]
            print(("{:>5s} {:>7s} {:>8s} {:>7s} {:>8s} {:>7s} "
                   "{:>8s} {:>8s} {:>9s}").format(*hdr))
            for s, r in c["TR_seeds"].items():
                def f4(v):
                    return f"{v:+.4f}" if isinstance(v, float) else "-"
                def f4p(v):
                    return f"{v:.4f}" if isinstance(v, float) else "-"
                print(("{:>5d} {:>7s} {:>8s} {:>7s} {:>8s} {:>7s} "
                       "{:>8s} {:>8s} {:>9s}").format(
                    s, f4p(r["T_reg"]), f4(r["D_reg"]), f4(r["climb"]),
                    f4p(r["adm_blind5"]), f4p(r["b_star"]),
                    f4p(r["b_clean"]), f4(r["F_late"]), r["verdict"]))
            print("\nTR1:", c["TR1_summary"])
            print("\nTR2 registration:", c["TR2_registration"])
            print("\nTR3 forecasting:", c["TR3_forecasting"])
            print("\nTR4 stratification:", c["TR4_stratification"])
            print("\nTR5 buy/cost:", c["TR5_buy_cost"])
            print("TR1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
