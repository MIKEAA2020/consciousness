#!/usr/bin/env python3
"""
THE COLD-BURN ROAD'S PROVENANCE - doc 73's filed fact, paid: a
quarter of the burn runs through committees the thermometer reads
cold, and the mechanism is unmeasured. Corpus doc 75's experiment.
Ordered at the ninth issuance: "the cold-burn road's provenance" -
doc 73's Part 3, stated in its own words: "the burn's mechanism is
not one mechanism: the committee-panel temperature carries the hot
road (doc 68's verdict, doc 72's column), and something else carries
the cold one - a candidate for whatever the panel and the committee
share that the subtraction cancels, which is to say: not the
temperature, and unmeasured."

The build is TWO worlds on the same 24 seeds:
  S18_CAP30       - the armed census world, SIXTH generation (the
                    original twelve audit bitwise vs doc 60's
                    S7_CAP30; the fresh twelve vs doc 73's stored
                    S17_CAP30 - the fresh seeds now HAVE a
                    predecessor), with the cold-road
                    instrumentation riding passively;
  S18_CAP30_NOCH  - the ANCHOR COMPLETION: doc 55's undefended
                    world extended to the fresh twelve (the
                    original twelve audit bitwise vs doc 55's
                    stored S5_CAP30_NOCH; the fresh twelve carry no
                    stored anchor, disclosed - they become it).
The cold-road instrumentation (all passive - no rng, no state
change; the audits are the proof):
  B_clean_idx  - the panel's PERSISTENT blind region, the item
                 identities behind b_clean (the unperturbed read,
                 once per seed; previously only the rate was
                 stored);
  the item-level split of every round's committed blind mass into
  PERSISTENT (in B_clean_idx - the shared region, what the anchor
  subtraction cancels) and DRAWN (fresh perturbation-blindness);
  the repair CONTENT: on the repaired items, whether the panel's
  label is wrong (the deference cost realized), split by
  repaired-on-flipped vs repaired-on-clean;
  the undisputed flips' LANDING: the poison mass that passes the
  channel unrepaired, split by panel-blind vs panel-confident.

PRE-REGISTERED PREDICTIONS (fixed before execution):
  CB1  THE AUDITS: the armed world's 24 per-seed bitwise audits
       (12 vs doc 60, 12 vs doc 73 - full fourteen rounds; the
       chain 55 -> 60 -> 68 -> 72 -> 73 -> 75); the anchor's 12 vs
       doc 55; the anchor's fresh 12 disclosed predecessor-free;
       the T-agreement vs doc 73's stored logs to 1e-9.
  CB2  THE STRATA (doc 73's, carried): the burn floor -0.02, the
       late window rounds 11-14, the cold road = burners with T at
       or below the 24-seed body median (0.0619). Expected (the
       world is deterministic): the census's 12/12 burn and the
       cold road {607, 615, 619} reproduce bitwise.
  CB3  THE HEIGHT-PATH FORK (the registered question): the cold
       burn's carrier - (i) HEIGHT: the cold burners' registrations
       sit above the temperature line (the b*-on-T residual) while
       their late blind-mass paths are cold-ok-like (the burn is
       the starting line, not the fall); (ii) PATH: their late
       paths fall below the cold-ok paths (the burn is the fall);
       (iii) BOTH; (iv) CONTINUUM: no separating column anywhere -
       the cold stratum is a continuum at the floor and the "road"
       is its tail pressing the verdict boundary. Criterion: the
       strata's overlap on (a) the b*-on-T residual, (b) the late
       b_r mean, read at the two-sample no-overlap test with
       margins disclosed.
  CB4  THE RESIDUAL'S OWN PROVENANCE (the new instrumentation):
       the round-5 committed-blind items' composition - (i) the
       PERSISTENT share dominates (the residual rides the panel's
       persistent blind region: the shared items, what the
       subtraction cancels - the filed candidate); (ii) the DRAWN
       share dominates (fresh-draw blindness: doc 68's interaction
       margin). Criterion: the persistent share of the
       registration's blind mass, cold-burn vs cold-ok.
  CB5  THE ANCHOR DECOMPOSITION AT 24: each burner's armed F_late
       split into the anchor's course (the shared fate) and the
       differential (the defense's work). Fork: the cold burn is
       (i) DIFFERENTIAL-carried (the original-12 prior: 607's
       anchor is fine at +0.0102), (ii) ANCHOR-carried, with the
       fresh twelve's anchors the first storage.
  CB6  THE BLIND-REGION FEEDBACK (the mechanism localization): the
       per-round repairs' content and the undisputed flips'
       landing - does the blind region's fall track (i) the
       DEFERENCE mass (repairs writing wrong panel labels), or
       (ii) the UNREPAIRED GATE POISON (panel-blind flips passing
       untouched), or (iii) neither (the kick/brake structure is
       shared and the fall is arithmetic)? Criteria: the
       cumulative exposures by stratum; the fall's timing against
       the repairs' timing.
  CB7  THE NON-CLAIMS: no amendment executed, no registration
       changed, no letter moved; the causal reading stays an
       interpretation (no battery has manipulated the columns);
       the cold road's census-grade status is inherited at 3/12
       with its post-hoc identification disclosed (doc 73's threat
       (iv)); no MAINTAINS verdicts, no Column-3 marks, no
       occupancy claim; D3 stands.

Usage: python3 s12_coldburn.py [--smoke]
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

# ---------------- the battery (the cold-burn road's provenance) ------------
C11CFG = {
    # the armed census world, SIXTH generation, the cold-road
    # instrumentation riding (bitwise doc 60's S7_CAP30 = doc 73's
    # S17_CAP30 on the shared seeds)
    "S18_CAP30": dict(bulk=0.22, gfrac=0.08, mode="mixed",
                      eps_i=0.03, fix="none", noch=False, rounds=14,
                      seeds=list(range(600, 624))),
    # the ANCHOR COMPLETION: doc 55's undefended world on all 24 seeds
    # (the original twelve bitwise doc 55's S5_CAP30_NOCH; the fresh
    # twelve carry no stored anchor, disclosed - they become it)
    "S18_CAP30_NOCH": dict(bulk=0.22, gfrac=0.08, mode="mixed",
                           eps_i=0.03, fix="none", noch=True, rounds=14,
                           seeds=list(range(600, 624))),
}
GROUPS = {"coldburn": ["S18_CAP30", "S18_CAP30_NOCH"]}
DOC60 = "/home/z/my-project/scripts/s6_bimodal_results.json"
DOC68 = "/home/z/my-project/scripts/s8_bstarprov_results.json"
DOC73 = "/home/z/my-project/scripts/s11_seeds24_results.json"
DOC55 = "/home/z/my-project/scripts/s5_boundaries_results.json"
AUDITS = []
REFPATHS = {"60": DOC60, "68": DOC68, "73": DOC73, "55": DOC55}
# the per-seed bitwise audits, MULTI-REFERENCE: the armed world's
# original twelve vs doc 60's S7_CAP30 and fresh twelve vs doc 73's
# stored S17_CAP30 (full fourteen rounds; the chain
# 55 -> 60 -> 68 -> 72 -> 73 -> 75, and 73 -> 75 on the fresh side);
# the anchor's original twelve vs doc 55's S5_CAP30_NOCH. The anchor's
# fresh twelve carry no stored predecessor - disclosed.
ORIG12 = list(range(600, 612))
FRESH12 = list(range(612, 624))
PERSEED_AUDITS = [
    ("S18_CAP30", "S7_CAP30", "60", ORIG12),
    ("S18_CAP30", "S17_CAP30", "73", FRESH12),
    ("S18_CAP30_NOCH", "S5_CAP30_NOCH", "55", ORIG12),
]
# the row->anchor pairing (this battery's OWN completed anchor)
ANCHOR = {"S18_CAP30": "SELF:S18_CAP30_NOCH"}


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
    cfg = C11CFG[arm]
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
    # persistent vulnerability with no draw in it at all; the cold-road
    # battery additionally stores the region's IDENTITIES (the shared
    # blind region - what the anchor subtraction cancels), still passive
    global _POOL_LOG
    _, pconf_clean = panel_grade(Xtr, panel)
    b_clean = float(np.mean(pconf_clean < TAU0))
    B_clean_idx = np.where(pconf_clean < TAU0)[0]
    B_clean_set = set(B_clean_idx.tolist())
    b_clean_n = int(len(B_clean_idx))

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
        # ---- 3b-2. THE COLD-ROAD LAYER (passive; doc 75's data) -----
        # the committed blind mass split into PERSISTENT (inside the
        # panel's unperturbed blind region - the shared region, what
        # the anchor subtraction cancels) and DRAWN (fresh
        # perturbation-blindness); the repair CONTENT (the panel's
        # label wrong on the repaired items - the deference cost
        # realized, split by repaired-on-flipped vs -clean); and the
        # undisputed flips' LANDING (the poison mass passing the
        # channel, split by panel-blind vs panel-confident). No rng,
        # no state change - identities and labels the round already
        # holds.
        blind_g = idx_commit[blind_c]
        blind_persist = [int(i in B_clean_set) for i in blind_g.tolist()]
        rep_m = rep_c_mask if len(rep_c_mask) else np.zeros(len(yt_p), bool)
        rep_wrong_mask = np.zeros(len(yt_p), dtype=bool)
        if rep_m.any():
            rep_wrong_mask[rep_m] = pl[rep_m] != yt_true[rep_m]
        flip_m = (bulk_m | gate_m) if len(yt_p) else np.zeros(0, bool)
        pass_m = flip_m & ~disputed
        pass_blind = pass_m & (pconf < TAU0)
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
                          if commit_m.any() else 0.0),
            # the cold-road layer
            "n_blind_persist": int(sum(blind_persist)),
            "n_blind_drawn": int(len(blind_persist)
                                 - sum(blind_persist)),
            "persist_share": (float(sum(blind_persist)) / len(blind_persist)
                              if blind_persist else None),
            "n_rep": int(rep_m.sum()),
            "n_rep_wrong": int(rep_wrong_mask.sum()),
            "n_rep_on_flip": int((rep_m & flip_m).sum()),
            "n_rep_on_flip_wrong": int((rep_wrong_mask & flip_m).sum()),
            "n_rep_on_clean": int((rep_m & ~flip_m).sum()),
            "n_rep_on_clean_wrong": int((rep_wrong_mask & ~flip_m).sum()),
            "n_flip_pass": int(pass_m.sum()),
            "n_flip_pass_blind": int(pass_blind.sum()),
            "n_gate_flipped": int(gate_m.sum()),
            "b_clean_n": b_clean_n})

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
    """The pre-registered CB1-CB7 classification: the cold-burn road's
    provenance, read from the sixth generation of the census world with
    the anchor completed and the cold-road instrumentation riding."""
    cls = {}
    A = res["arms"]
    runs = A["S18_CAP30"]["runs"]
    anchor_runs = A["S18_CAP30_NOCH"]["runs"]
    amap = {rr["seed"]: rr for rr in anchor_runs}

    # ---- CB1 (part 2): the T-agreement vs doc 73's stored logs ----------
    d73 = json.load(open(DOC73))
    d73_map = {rr["seed"]: rr for rr in d73["arms"]["S17_CAP30"]["runs"]}
    tagree = {}
    for rr in runs:
        s = rr["seed"]
        if s not in d73_map:
            continue
        T73 = d73_map[s]["plogs"][COMP_REG_END - 1].get("T_reg")
        T_new = rr["plogs"][COMP_REG_END - 1].get("T_reg")
        tagree[str(s)] = bool(T73 is not None and T_new is not None
                              and abs(T73 - T_new) < 1e-9)
    cls["CB1_T_agreement_with_doc73"] = {
        "per_seed": tagree,
        "all": bool(tagree) and all(tagree.values()),
        "n": len(tagree)}

    orig = list(range(600, 612))
    fresh = list(range(612, 624))
    DEEP = [603, 604, 606]
    DEEP_EDGE = 0.0928          # doc 73's corrected deep-cluster edge

    def flate_of(rr):
        n = len(rr["traj"])
        late = [t["comp_F"] for t in rr["traj"]
                if t["comp_F"] is not None and t["round"] >= n - 3]
        return float(np.mean(late)) if late else None

    def late_br_of(rr):
        n = len(rr["traj"])
        late = [t["blind_mass"] for t in rr["traj"] if t["round"] >= n - 3]
        return float(np.mean(late)) if late else None

    # ---- the per-seed table (the registered + cold-road columns) --------
    seeds = {}
    for rr in runs:
        s = rr["seed"]
        p5 = rr["plogs"][COMP_REG_END - 1]
        i5 = rr["items"][COMP_REG_END - 1]
        T = p5["T_reg"]
        bstar = p5["b_star_reg"]
        seeds[s] = {
            "T_reg": T, "b_star": bstar, "b_clean": p5["b_clean"],
            "b_clean_n": i5.get("b_clean_n"),
            "F_late": flate_of(rr), "late_b_r": late_br_of(rr),
            "persist_share5": i5.get("persist_share"),
            "n_blind_persist5": i5.get("n_blind_persist"),
            "n_blind_drawn5": i5.get("n_blind_drawn"),
            "cohort": ("orig" if s in orig else "fresh"),
            "anchor_F_late": (flate_of(amap[s]) if s in amap else None),
            "anchor_late_b_r": (late_br_of(amap[s])
                                if s in amap else None)}
        sd = seeds[s]
        sd["verdict"] = ("BURNED" if sd["F_late"] < COMP_BURN_FLOOR
                         else "CONSUMED" if sd["F_late"] < COMP_FILL_FLOOR
                         else "ON-COURSE" if sd["F_late"] <= COMP_OVER_CEIL
                         else "OVERFILLED")
        sd["burned"] = sd["verdict"] == "BURNED"
        sd["deep_by_rule"] = bool(bstar >= DEEP_EDGE)
        # the cumulative cold-road exposures (rounds 6-14)
        cum = {"rep": 0, "rep_wrong": 0, "rep_on_clean": 0,
               "rep_on_clean_wrong": 0, "flip_pass": 0,
               "flip_pass_blind": 0, "gate": 0, "persist_late": 0,
               "blind_late": 0}
        for it in rr["items"]:
            if it["r"] < PERT_ROUND:
                continue
            cum["rep"] += it.get("n_rep", 0)
            cum["rep_wrong"] += it.get("n_rep_wrong", 0)
            cum["rep_on_clean"] += it.get("n_rep_on_clean", 0)
            cum["rep_on_clean_wrong"] += it.get("n_rep_on_clean_wrong", 0)
            cum["flip_pass"] += it.get("n_flip_pass", 0)
            cum["flip_pass_blind"] += it.get("n_flip_pass_blind", 0)
            cum["gate"] += it.get("n_gate_flipped", 0)
            if it["r"] >= 11:
                cum["persist_late"] += it.get("n_blind_persist", 0)
                cum["blind_late"] += it.get("n_blind", 0)
        sd["cum"] = cum
        sd["persist_share_late"] = (
            cum["persist_late"] / cum["blind_late"]
            if cum["blind_late"] else None)
        if s in amap:
            acum = {"flip_pass": 0, "flip_pass_blind": 0, "gate": 0,
                    "blind_late": 0, "persist_late": 0}
            for it in amap[s]["items"]:
                if it["r"] < PERT_ROUND:
                    continue
                acum["flip_pass"] += it.get("n_flip_pass", 0)
                acum["flip_pass_blind"] += it.get("n_flip_pass_blind", 0)
                acum["gate"] += it.get("n_gate_flipped", 0)
                if it["r"] >= 11:
                    acum["blind_late"] += it.get("n_blind", 0)
                    acum["persist_late"] += it.get("n_blind_persist", 0)
            sd["anchor_cum"] = acum
            sd["anchor_persist_share_late"] = (
                acum["persist_late"] / acum["blind_late"]
                if acum["blind_late"] else None)
    cls["CB_seeds"] = seeds

    # ---- CB2: the strata ---------------------------------------------------
    Ts = {s: seeds[s]["T_reg"] for s in seeds}
    body = [s for s in seeds if s not in DEEP]
    Tmed = float(np.median([Ts[s] for s in body]))
    burners = sorted(s for s in seeds if seeds[s]["burned"])
    cold_burn = sorted(s for s in burners if Ts[s] <= Tmed)
    hot_burn = sorted(s for s in burners if Ts[s] > Tmed)
    cold_ok = sorted(s for s in body if Ts[s] <= Tmed and s not in burners)
    cls["CB2_strata"] = {
        "burn_floor": COMP_BURN_FLOOR, "late_window": "rounds 11-14",
        "body_median_T": Tmed,
        "burners": burners, "n_burned": len(burners),
        "cold_burn": cold_burn, "hot_burn": hot_burn,
        "cold_ok": cold_ok,
        "census_reproduced": sorted(burners) == sorted(
            [600, 603, 604, 606, 607, 615, 616, 617, 619, 620, 622, 623]),
        "cold_road_reproduced": cold_burn == [607, 615, 619]}

    # ---- the b*-on-T regression at 24 (doc 73's world, recomputed) -------
    Tl = [Ts[s] for s in seeds]
    bl = [seeds[s]["b_star"] for s in seeds]
    slope, icept = np.polyfit(Tl, bl, 1)
    resid = {s: seeds[s]["b_star"] - (slope * Ts[s] + icept) for s in seeds}

    # ---- CB3: the height-path fork -----------------------------------------
    def stats(key, gs):
        v = [seeds[s][key] for s in gs if seeds[s][key] is not None]
        return {"mean": float(np.mean(v)) if v else None,
                "range": [float(min(v)), float(max(v))] if v else None,
                "n": len(v)}

    def resid_stats(gs):
        v = [resid[s] for s in gs]
        return {"mean": float(np.mean(v)) if v else None,
                "range": [float(min(v)), float(max(v))] if v else None}

    h_res = resid_stats(cold_burn)
    h_res_ok = resid_stats(cold_ok)
    p_cb = stats("late_b_r", cold_burn)
    p_ok = stats("late_b_r", cold_ok)
    b_cb = stats("b_star", cold_burn)
    b_ok = stats("b_star", cold_ok)
    resid_sep = bool(h_res["range"] and h_res_ok["range"]
                     and h_res_ok["range"][1] < h_res["range"][0])
    path_sep = bool(p_cb["range"] and p_ok["range"]
                    and p_cb["range"][1] < p_ok["range"][0])
    bsep = bool(b_cb["range"] and b_ok["range"]
                and b_ok["range"][1] < b_cb["range"][0])
    if resid_sep and not path_sep:
        fork3 = ("(i) HEIGHT - the residual separates, the late path "
                 "does not: the burn is the starting line")
    elif path_sep and not resid_sep:
        fork3 = ("(ii) PATH - the late path separates, the residual "
                 "does not: the burn is the fall")
    elif resid_sep and path_sep:
        fork3 = "(iii) BOTH - the residual and the path separate"
    elif bsep and not path_sep:
        fork3 = ("(i') HEIGHT-RAW - b* itself separates (though the "
                 "T-residual does not); the burn is the starting line "
                 "carried by the temperature's own slope")
    else:
        fork3 = ("(iv) CONTINUUM - no separating column: the cold "
                 "stratum is a continuum at the floor")
    cls["CB3_height_path"] = {
        "regression": {"slope": float(slope), "intercept": float(icept)},
        "cold_burn": {"b*_residual": h_res, "late_b_r": p_cb, "b*": b_cb},
        "cold_ok": {"b*_residual": h_res_ok, "late_b_r": p_ok, "b*": b_ok},
        "residual_separates": resid_sep, "path_separates": path_sep,
        "b*_separates": bsep,
        "residual_margin": (h_res["range"][0] - h_res_ok["range"][1]
                            if (h_res["range"] and h_res_ok["range"])
                            else None),
        "path_margin": (p_ok["range"][0] - p_cb["range"][1]
                        if (p_cb["range"] and p_ok["range"]) else None),
        "fork": fork3,
        "per_seed_residual": {str(s): float(resid[s]) for s in seeds}}

    # ---- CB4: the residual's own provenance --------------------------------
    ps_cb = stats("persist_share5", cold_burn)
    ps_ok = stats("persist_share5", cold_ok)
    ps_hb = stats("persist_share5", hot_burn)
    nps_cb = stats("n_blind_persist5", cold_burn)
    nps_ok = stats("n_blind_persist5", cold_ok)
    bcl_cb = stats("b_clean", cold_burn)
    bcl_ok = stats("b_clean", cold_ok)
    psr_cb = stats("persist_share_late", cold_burn)
    psr_ok = stats("persist_share_late", cold_ok)
    cls["CB4_residual_provenance"] = {
        "persist_share_at_registration": {"cold_burn": ps_cb,
                                          "cold_ok": ps_ok,
                                          "hot_burn": ps_hb},
        "n_blind_persistent_at_registration": {"cold_burn": nps_cb,
                                               "cold_ok": nps_ok},
        "b_clean": {"cold_burn": bcl_cb, "cold_ok": bcl_ok},
        "persist_share_late": {"cold_burn": psr_cb, "cold_ok": psr_ok},
        "fork": ("(i) PERSISTENT - the registration's blind mass rides "
                 "the panel's persistent blind region (the shared "
                 "structure the subtraction cancels)"
                 if (ps_cb["mean"] is not None and ps_ok["mean"] is not None
                     and ps_cb["mean"] > 0.5 and
                     ps_cb["mean"] > ps_ok["mean"]) else
                 "(ii) DRAWN - fresh perturbation-blindness carries the "
                 "registration's height" if (ps_cb["mean"] is not None
                                             and ps_cb["mean"] <= 0.5)
                 else "mixed - disclosed with the numbers")}

    # ---- CB5: the anchor decomposition at 24 --------------------------------
    dec = {}
    for s in sorted(seeds):
        if seeds[s]["anchor_F_late"] is None:
            continue
        af = seeds[s]["anchor_F_late"]
        arf = seeds[s]["F_late"]
        strat = ("COLD-BURN" if s in cold_burn else
                 "hot-burn" if s in hot_burn else
                 "cold-ok" if s in cold_ok else
                 "deep" if s in DEEP else "body")
        dec[str(s)] = {"stratum": strat, "anchorF": af, "armedF": arf,
                       "diff": arf - af, "T": Ts[s]}
    d_cb = [v["diff"] for v in dec.values() if v["stratum"] == "COLD-BURN"]
    d_ok = [v["diff"] for v in dec.values() if v["stratum"] == "cold-ok"]
    a_cb = [v["anchorF"] for v in dec.values()
            if v["stratum"] == "COLD-BURN"]
    a_ok = [v["anchorF"] for v in dec.values() if v["stratum"] == "cold-ok"]
    cls["CB5_anchor_decomposition"] = {
        "per_seed": dec,
        "cold_burn": {"anchor_mean": float(np.mean(a_cb)) if a_cb else None,
                      "diff_mean": float(np.mean(d_cb)) if d_cb else None},
        "cold_ok": {"anchor_mean": float(np.mean(a_ok)) if a_ok else None,
                    "diff_mean": float(np.mean(d_ok)) if d_ok else None},
        "fork": ("(i) DIFFERENTIAL-carried - the anchors are fine, the "
                 "defense's differential carries the fall"
                 if (a_cb and min(a_cb) > COMP_BURN_FLOOR) else
                 "(ii) ANCHOR-carried - the undefended world shares the "
                 "burn (at least one cold burner's anchor already "
                 "burned)")}

    # ---- CB6: the blind-region feedback ------------------------------------
    def cumstats(key, gs):
        v = [seeds[s]["cum"][key] for s in gs]
        return float(np.mean(v)) if v else None

    hb = sorted(hot_burn)
    cls["CB6_feedback"] = {
        "cumulative_r6_14": {
            "rep_total": {"cold_burn": cumstats("rep", cold_burn),
                          "cold_ok": cumstats("rep", cold_ok),
                          "hot_burn": cumstats("rep", hb)},
            "rep_wrong": {"cold_burn": cumstats("rep_wrong", cold_burn),
                          "cold_ok": cumstats("rep_wrong", cold_ok),
                          "hot_burn": cumstats("rep_wrong", hb)},
            "rep_on_clean": {"cold_burn": cumstats("rep_on_clean",
                                                   cold_burn),
                             "cold_ok": cumstats("rep_on_clean", cold_ok),
                             "hot_burn": cumstats("rep_on_clean", hb)},
            "rep_on_clean_wrong": {
                "cold_burn": cumstats("rep_on_clean_wrong", cold_burn),
                "cold_ok": cumstats("rep_on_clean_wrong", cold_ok),
                "hot_burn": cumstats("rep_on_clean_wrong", hb)},
            "flip_pass": {"cold_burn": cumstats("flip_pass", cold_burn),
                          "cold_ok": cumstats("flip_pass", cold_ok),
                          "hot_burn": cumstats("flip_pass", hb)},
            "flip_pass_blind": {"cold_burn": cumstats("flip_pass_blind",
                                                      cold_burn),
                                "cold_ok": cumstats("flip_pass_blind",
                                                    cold_ok),
                                "hot_burn": cumstats("flip_pass_blind",
                                                     hb)},
            "gate_flips": {"cold_burn": cumstats("gate", cold_burn),
                           "cold_ok": cumstats("gate", cold_ok),
                           "hot_burn": cumstats("gate", hb)}},
        "reading": "the repairs' wrong-label content (the deference "
                   "cost) and the unrepaired blind-landing flips (the "
                   "gate poison), by stratum - the fall's companions, "
                   "disclosed as measured"}

    # ---- CB7: the non-claims (static) ---------------------------------------
    cls["CB7_nonclaims"] = {
        "no_amendment": "no registration changed, no letter moved - the "
                        "cold-road columns are registered ALONGSIDE the "
                        "standing ones",
        "no_causal_claim": "the columns' causal role stays an "
                           "interpretation - no battery has manipulated "
                           "them",
        "cold_road_status": "inherited at 3/12 with doc 73's threat (iv) "
                            "disclosed: the below-median-T identification "
                            "is post-hoc, the count conventional",
        "integrity": {"any_MAINTAINS": False,
                      "column3": "unmarked",
                      "audit": "24 armed per-seed bitwise audits (12 vs "
                               "doc 60, 12 vs doc 73) + 12 anchor audits "
                               "vs doc 55; the anchor's fresh twelve "
                               "predecessor-free, disclosed"}}
    return cls


def audit_perseed(res, pairs):
    """The per-seed audits, MULTI-REFERENCE: each listed seed's
    tau/kappa trajectory must reproduce the named reference's stored
    run for the same arm and seed, to 1e-9 - the worlds are literally
    the same."""
    out = {}
    for ours, theirs, refkey, seedlist in pairs:
        if ours not in res["arms"]:
            continue
        ref = json.load(open(REFPATHS[refkey]))
        for rr in res["arms"][ours]["runs"]:
            s = rr["seed"]
            if s not in seedlist:
                continue
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
        for a in GROUPS["coldburn"]:
            globals()["C11CFG"][a]["seeds"] = [600, 601]
            globals()["C11CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s12_coldburn_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s12_coldburn_results.json")
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
            "pert_round": PERT_ROUND, "nseed": 24,
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

    arms = GROUPS["coldburn"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C11CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj, ilogs, plogs, b_clean = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj, "items": ilogs,
                         "plogs": plogs, "b_clean": b_clean})
            print(f"  {arm} seed={s} done (b_clean={b_clean:.4f}) "
                  f"({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- the per-seed bitwise audits (multi-reference) -------------------
    if not smoke:
        need_ps = [a for a, _, _, _ in PERSEED_AUDITS if a in res["arms"]]
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
        need = set(GROUPS["coldburn"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the cold-burn road's provenance ===")
            hdr = ["seed", "coh", "T_reg", "b*", "resid",
                   "persist5", "F_late", "lateb_r", "anchF", "verd"]
            print(("{:>5s} {:>4s} {:>7s} {:>7s} {:>7s} {:>8s} "
                   "{:>8s} {:>8s} {:>8s} {:>9s}").format(*hdr))
            for s, r in c["CB_seeds"].items():
                def f4(v):
                    return f"{v:+.4f}" if isinstance(v, float) else "-"
                def f4p(v):
                    return f"{v:.4f}" if isinstance(v, float) else "-"
                print(("{:>5d} {:>4s} {:>7s} {:>7s} {:>7s} {:>8s} "
                       "{:>8s} {:>8s} {:>8s} {:>9s}").format(
                    s, r["cohort"][:4], f4p(r["T_reg"]),
                    f4p(r["b_star"]),
                    f4(c["CB3_height_path"]["per_seed_residual"][str(s)]),
                    f4p(r["persist_share5"]), f4(r["F_late"]),
                    f4p(r["late_b_r"]), f4(r["anchor_F_late"]),
                    r["verdict"]))
            print("\nCB2 strata:", c["CB2_strata"])
            print("\nCB3 height-path:",
                  {k: v for k, v in c["CB3_height_path"].items()
                   if k != "per_seed_residual"})
            print("\nCB4 residual provenance:", c["CB4_residual_provenance"])
            print("\nCB5 anchor decomposition (means):",
                  {k: v for k, v in c["CB5_anchor_decomposition"].items()
                   if k != "per_seed"})
            print("\nCB6 feedback:", c["CB6_feedback"])
            print("\nCB1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
