#!/usr/bin/env python3
"""
THE BULK-MATCHED TWIN - doc 78's threat (i), paid at face value: "the
capacity's own 0.25 realized dose has no uniform-matched twin in this
battery (the dial's 0.25 world exists at six seeds with different
instrumentation), so the capacity's 12-burn census is compared across
dose as well as composition; the attribution 'the bulk churn burns
the mid registrations' rests on the mirror's and control's
eliminations of the cold road, not on a bulk-only world at 0.25."
Ordered at the eleventh issuance under the standing rule - only if
highly merited - and adjudicated so on three grounds: it is the
factorial's missing fourth cell (composition x realized dose:
capacity mixed@0.2505, mirror mixed@0.1238, control uniform@0.1291,
THIS battery uniform@0.25), a filed attribution currently rests on
eliminations rather than a direct measurement, and every limb of its
fork is decisive either way. The three candidates declined at the
same adjudication - the saturation band's interior (no filed claim
rides on the fill curve's shape between two measured endpoints and a
measured ceiling), the mode axis (doc 78's threat (iii) is a
disclosed scope boundary, not an evidence gap inside the claimed
attribution), and the eighth-interval cell at 0.29375 (doc 81's own
filing: the next convention's cell, not an expectation - nothing
filed fails by it) - are declined in the corpus's ledger, not run.

The build is ONE world on the same 24 seeds plus its anchor:
  S22_UNIF25       - THE TWIN: the uniform world at bulk = 0.25
                     (mode "uniform", the dial's own convention),
                     armed - the capacity's realized dose (0.2505)
                     with none of the gated mass;
  S22_UNIF25_NOCH  - the twin's ANCHOR (the undefended uniform world
                     at 0.25, 24 seeds).

THE PAIRED DESIGN (the battery's certificate, carried from doc 78):
the uniform attack draws the same one rng_attack.random(n) block per
round as the mixed worlds' bulk arm, the gated fill it lacks was
deterministic, and the chassis streams are the same seeds - so the
twin shares rounds 1-5 BITWISE with the capacity, mirror, and control
worlds, and the temperature T (the rounds-1-4 mean blind-admission
rate) and the registration b* (round 5) are PREFIX quantities: the
twin's census runs on THE SAME registrations, the same strata
definitions, the same body-median T, as doc 73/75/78's. What the mode
can move is everything after round 6: the trajectories, the verdicts,
the F_late census, the anchor's course, the exposures. The census
question - at the capacity's own realized dose, does a world with
none of the gated mass burn? - is thereby asked PAIRED: same seeds,
same starting lines, the dose held, the composition stripped.

THE CROSS-INSTRUMENT TIE (the design's third audit family, new):
the dial's stored 0.25 cells - doc 71's S15_D100_25 (uniform, bulk
0.25, rp 1.0, armed) and S15_ANCHOR_25 (its anchor) - are the SAME
WORLDS on the dial's six seeds (600-605), and this battery's first
six seeds must reproduce them FULL-LENGTH bitwise. Doc 78's threat
(i) set the dial's 0.25 world aside as "different instrumentation";
the audit tests the worlds' identity where the seed sets overlap, and
the census extends them to 24 seeds with the cold-road layer riding.
Registered expectation: PASS (the chassis consumes identical
randomness for a uniform rp-1.0 world; the mixed-mode full-length
chain 55 -> 78 is the family's evidence). A failure would be a
chassis-divergence finding, disclosed, not a silent fallback.

PRE-REGISTERED FORKS (fixed before execution):
  BT1  THE AUDITS AND THE PAIRED CERTIFICATE: 60 per-seed audits in
       three families - 48 clean-prefix (S22_UNIF25 vs doc 73's
       stored S17_CAP30; S22_UNIF25_NOCH vs doc 75's stored
       S18_CAP30_NOCH; rounds 1-5, all 24 seeds each) and 12
       FULL-LENGTH cross-instrument (the twin's and the anchor's
       seeds 600-605 vs doc 71's stored S15_D100_25 and
       S15_ANCHOR_25); the determinism count passes seven hundred
       two (642 + 60). THE CERTIFICATE: the twin's T and b*
       identical to doc 73's stored values on all 24 seeds (prefix
       quantities on a shared prefix).
  BT2  THE MATCHED-DOSE CERTIFICATE: the twin's realized dose (the
       rounds-7-14 mean) within ~0.01 of the capacity's stored
       0.2505 (loaded from doc 78's results file, not transcribed);
       the four-world realized-dose table carried alongside.
  BT3  THE CENSUS AT THE CAPACITY'S DOSE (the registered question):
       the burn count against the capacity's 12, the mirror's 6, the
       control's 0. Fork, checked in order: (0) SUPER-ADDITIVE
       (n >= 16: the uniform twin burns MORE than the capacity - the
       gated mass was PROTECTIVE at 0.25); (i) MATCHED (9 <= n <= 15:
       the dose alone carries the capacity's census - composition
       second-order at this dose); (ii) SPLIT (4 <= n <= 8: both
       arms carry burns); (iii) THIN (n <= 3: the capacity's burns
       are the MIX's act - the gated presence load-bearing even at
       its ~0.039 realized level). The identity readings carried
       alongside: the cold road {607, 615, 619} in or out; the deep
       cluster {603, 604, 606}; the burners' b* against the
       survivors' (the mirror's top-six structure, disclosed post-hoc
       there, directional here); the F_late census's largest internal
       gap against the 0.010 bar; the verdict census.
  BT4  THE PAIRED VERDICT TABLE: per seed, doc 75's stored capacity
       verdict against the twin's verdict; the flip count, the
       directions (burn->consumed against consumed->burn), the
       flips' anatomy (T, b*, the capacity F_late), and the
       four-world verdict map (capacity/mirror/control/twin) per
       seed - the factorial's completed table.
  BT5  THE ANCHOR DECOMPOSITION: the twin anchor's course (the
       control's +0.1405 fill at 0.125 was that battery's own scale
       - at 0.25 the fill expected larger), the differential's sign
       and scale, the arithmetic F_armed = F_anchor + differential
       at the floor -0.02. Fork: (i) ANCHOR-CARRIED (at least one
       twin burner's anchor already burns); (ii) FILLS-AND-SURVIVES
       (no anchor burns, the armed census safe - the fill out-runs
       the churn); (iii) CHURN-DOMINATES (no anchor burns, the
       differential carries whatever burns there are).
  BT6  THE LANDING AND THE EXPOSURES: the passing flips' blind-landing
       fraction (the control's 100% replicated at the higher dose?),
       the repairs' wrong-label content, the repair traffic against
       the control's and the mirror's stored rates, and the
       cold-road columns (the persistent share, the exposures) at
       the twin.

Usage: python3 s16_unif25.py [--smoke]
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

# ---------------- the battery (the bulk-matched twin) ----------------
C13CFG = {
    # THE TWIN - the uniform world at the capacity's realized dose
    # (0.2505), none of the gated mass; the dial's own convention
    "S22_UNIF25": dict(bulk=0.25, gfrac=0.0, mode="uniform",
                       eps_i=0.03, fix="none", noch=False, rounds=14,
                       seeds=list(range(600, 624))),
    # the twin's ANCHOR - the undefended uniform world at 0.25
    "S22_UNIF25_NOCH": dict(bulk=0.25, gfrac=0.0, mode="uniform",
                            eps_i=0.03, fix="none", noch=True, rounds=14,
                            seeds=list(range(600, 624))),
}
GROUPS = {"bulktwin": ["S22_UNIF25", "S22_UNIF25_NOCH"]}
DOC73 = "/home/z/my-project/scripts/s11_seeds24_results.json"
DOC75 = "/home/z/my-project/scripts/s12_coldburn_results.json"
DOC71 = "/home/z/my-project/scripts/s9_finishing_results.json"
DOC78 = "/home/z/my-project/scripts/s14_anothermix_results.json"
AUDITS = []
REFPATHS = {"73": DOC73, "75": DOC75, "71": DOC71}
# the per-seed audits, THREE FAMILIES: the census convention's
# clean-prefix (rounds 1-5, all 24 seeds, vs doc 73's armed world and
# doc 75's anchor) and the CROSS-INSTRUMENT TIE (full-length, the
# dial's six seeds 600-605, vs doc 71's stored 0.25 cells - the same
# worlds by construction). 24 + 24 + 6 + 6 = sixty; the determinism
# count passes seven hundred two (642 + 60).
ALL24 = list(range(600, 624))
DIAL6 = list(range(600, 606))
PERSEED_AUDITS = [
    ("S22_UNIF25", "S17_CAP30", "73", ALL24, 5),
    ("S22_UNIF25_NOCH", "S18_CAP30_NOCH", "75", ALL24, 5),
    ("S22_UNIF25", "S15_D100_25", "71", DIAL6, None),
    ("S22_UNIF25_NOCH", "S15_ANCHOR_25", "71", DIAL6, None),
]
# the row->anchor pairing (the twin's own anchor)
ANCHOR = {"S22_UNIF25": "SELF:S22_UNIF25_NOCH"}
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
    cfg = C13CFG[arm]
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



def audit_perseed(res, pairs):
    """The per-seed audits, MIXED REFERENCE DEPTHS: each listed seed's
    tau/kappa trajectory must reproduce the named reference's stored run
    for the same arm and seed, to 1e-9 - FULL-LENGTH where a stored
    predecessor world exists (the REF arm: the worlds are literally the
    same), and on the SHARED CLEAN PREFIX (rounds 1-5) where the world
    is new (the mirror arms: the attack's draws are thresholded, not
    re-drawn, and the two mixes share the prefix bitwise - doc 67's
    adaptation, the dial batteries' convention)."""
    out = {}
    for ours, theirs, refkey, seedlist, rounds_ref in pairs:
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
            if rounds_ref is not None:
                t_ref = t_ref[:rounds_ref]
                t_got = t_got[:rounds_ref]
            ok = bool(len(t_ref) == len(t_got)
                      and all(abs(a["tau_sys"] - b["tau_sys"]) < 1e-9
                              and abs(a["kappa_sys"] - b["kappa_sys"]) < 1e-9
                              for a, b in zip(t_ref, t_got)))
            out[f"{ours}_seed{s}_reproduces_{theirs}"] = ok
            print(f"{ours} seed={s} reproduces {theirs} "
                  f"({'prefix' if rounds_ref else 'full'}): {ok}")
    return out


def classify(res):
    """The pre-registered BT1-BT6 classification: the bulk-matched twin -
    the uniform world at the capacity's realized dose (0.2505) on the
    same 24 seeds and the same registrations, the census at the
    capacity's dose, the paired verdict table against doc 75's stored
    capacity census, and the completed four-world factorial."""
    cls = {}
    A = res["arms"]
    runs = A["S22_UNIF25"]["runs"]
    anchor_runs = A["S22_UNIF25_NOCH"]["runs"]
    amap = {rr["seed"]: rr for rr in anchor_runs}

    d73 = json.load(open(DOC73))
    d73_map = {rr["seed"]: rr for rr in d73["arms"]["S17_CAP30"]["runs"]}
    d75 = json.load(open(DOC75))
    d75_map = {rr["seed"]: rr for rr in d75["arms"]["S18_CAP30"]["runs"]}
    d78 = json.load(open(DOC78))
    d78_mirror = {rr["seed"]: rr for rr in d78["arms"]["S20_MIXM"]["runs"]}
    d78_control = {rr["seed"]: rr
                   for rr in d78["arms"]["S20_UNIF13"]["runs"]}

    # ---- BT1 (part 2): the paired certificate --------------------------
    # the twin's T and b* identical to doc 73's stored values (prefix
    # quantities on a shared prefix)
    tagree, bagree = {}, {}
    for rr in runs:
        s = rr["seed"]
        if s not in d73_map:
            continue
        T73 = d73_map[s]["plogs"][COMP_REG_END - 1].get("T_reg")
        T_new = rr["plogs"][COMP_REG_END - 1].get("T_reg")
        b73 = d73_map[s]["plogs"][COMP_REG_END - 1].get("b_star_reg")
        b_new = rr["plogs"][COMP_REG_END - 1].get("b_star_reg")
        tagree[str(s)] = bool(T73 is not None and T_new is not None
                              and abs(T73 - T_new) < 1e-12)
        bagree[str(s)] = bool(b73 is not None and b_new is not None
                              and abs(b73 - b_new) < 1e-12)

    orig = list(range(600, 612))
    DEEP = [603, 604, 606]
    COLDROAD = [607, 615, 619]

    def flate_of(rr):
        n = len(rr["traj"])
        late = [t["comp_F"] for t in rr["traj"]
                if t["comp_F"] is not None and t["round"] >= n - 3]
        return float(np.mean(late)) if late else None

    def late_br_of(rr):
        n = len(rr["traj"])
        late = [t["blind_mass"] for t in rr["traj"] if t["round"] >= n - 3]
        return float(np.mean(late)) if late else None

    def verdict_of(f):
        if f is None:
            return None
        return ("BURNED" if f < COMP_BURN_FLOOR
                else "CONSUMED" if f < COMP_FILL_FLOOR
                else "ON-COURSE" if f <= COMP_OVER_CEIL
                else "OVERFILLED")

    # the stored censuses this battery is matched against
    cap_burners = sorted(s for s, rr in d75_map.items()
                         if verdict_of(flate_of(rr)) == "BURNED")
    mirror_burners = sorted(s for s, rr in d78_mirror.items()
                            if verdict_of(flate_of(rr)) == "BURNED")
    control_burners = sorted(s for s, rr in d78_control.items()
                             if verdict_of(flate_of(rr)) == "BURNED")

    cls["BT1_paired_certificate"] = {
        "T_agreement_with_doc73": {
            "per_seed": tagree,
            "all": bool(tagree) and all(tagree.values()),
            "n": len(tagree)},
        "bstar_agreement_with_doc73": {
            "per_seed": bagree,
            "all": bool(bagree) and all(bagree.values()),
            "n": len(bagree)},
        "stored_censuses": {
            "capacity_burners_doc75": cap_burners,
            "mirror_burners_doc78": mirror_burners,
            "control_burners_doc78": control_burners},
        "cross_instrument_tie": "the twelve full-length audits vs doc "
                                "71's stored 0.25 cells are in the "
                                "audit table (perseid_audits)"}

    # ---- the twin's per-seed table --------------------------------------
    seeds = {}
    for rr in runs:
        s = rr["seed"]
        p5 = rr["plogs"][COMP_REG_END - 1]
        i5 = rr["items"][COMP_REG_END - 1]
        seeds[s] = {
            "T_reg": p5["T_reg"], "b_star": p5["b_star_reg"],
            "b_clean": p5["b_clean"],
            "b_clean_n": i5.get("b_clean_n"),
            "F_late": flate_of(rr), "late_b_r": late_br_of(rr),
            "persist_share5": i5.get("persist_share"),
            "n_blind_persist5": i5.get("n_blind_persist"),
            "n_blind_drawn5": i5.get("n_blind_drawn"),
            "cohort": ("orig" if s in orig else "fresh"),
            "in_dial6": bool(s < 606),
            "anchor_F_late": (flate_of(amap[s]) if s in amap else None),
            "anchor_late_b_r": (late_br_of(amap[s])
                                if s in amap else None)}
        sd = seeds[s]
        sd["verdict"] = verdict_of(sd["F_late"])
        sd["burned"] = sd["verdict"] == "BURNED"
        if s in d75_map:
            cf = flate_of(d75_map[s])
            sd["cap_F_late"] = cf
            sd["cap_verdict"] = verdict_of(cf)
            sd["cap_burned"] = sd["cap_verdict"] == "BURNED"
        else:
            sd["cap_F_late"] = sd["cap_verdict"] = sd["cap_burned"] = None
        if s in d78_mirror:
            sd["mirror_verdict"] = verdict_of(flate_of(d78_mirror[s]))
        if s in d78_control:
            sd["control_verdict"] = verdict_of(flate_of(d78_control[s]))
        # the cumulative exposures (rounds 6-14)
        cum = {"rep": 0, "rep_wrong": 0, "rep_on_clean": 0,
               "rep_on_clean_wrong": 0, "flip_pass": 0,
               "flip_pass_blind": 0, "gate": 0, "bulk": 0,
               "persist_late": 0, "blind_late": 0}
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
            cum["bulk"] += len(it.get("bulk_idx", []))
            if it["r"] >= 11:
                cum["persist_late"] += it.get("n_blind_persist", 0)
                cum["blind_late"] += it.get("n_blind", 0)
        sd["cum"] = cum
        sd["persist_share_late"] = (
            cum["persist_late"] / cum["blind_late"]
            if cum["blind_late"] else None)
        if s in amap:
            acum = {"flip_pass": 0, "flip_pass_blind": 0, "gate": 0,
                    "bulk": 0, "blind_late": 0, "persist_late": 0}
            for it in amap[s]["items"]:
                if it["r"] < PERT_ROUND:
                    continue
                acum["flip_pass"] += it.get("n_flip_pass", 0)
                acum["flip_pass_blind"] += it.get("n_flip_pass_blind", 0)
                acum["gate"] += it.get("n_gate_flipped", 0)
                acum["bulk"] += it.get("n_bulk_flipped", 0)
                if it["r"] >= 11:
                    acum["blind_late"] += it.get("n_blind", 0)
                    acum["persist_late"] += it.get("n_blind_persist", 0)
            sd["anchor_cum"] = acum
    cls["BT_seeds"] = seeds

    # ---- BT2: the matched-dose certificate --------------------------------
    def dose714(arms_dict, arm):
        ag = arms_dict[arm]["aggregate"]["realized_dose"]
        return float(np.mean(ag[6:14]))

    twin_dose = dose714(A, "S22_UNIF25")
    cap_dose = dose714(d78["arms"], "S20_CAP30_REF")
    cls["BT2_matched_dose"] = {
        "twin_realized_dose_7_14": twin_dose,
        "capacity_stored": cap_dose,
        "mirror_stored": dose714(d78["arms"], "S20_MIXM"),
        "control_stored": dose714(d78["arms"], "S20_UNIF13"),
        "twin_minus_capacity": twin_dose - cap_dose,
        "certificate_within_0p01": bool(abs(twin_dose - cap_dose)
                                        <= 0.01)}

    # ---- BT3: the census at the capacity's dose ---------------------------
    burners = sorted(s for s in seeds if seeds[s]["burned"])
    n_burn = len(burners)
    verd = [seeds[s]["verdict"] for s in seeds]
    if n_burn >= 16:
        fork3 = ("(0) SUPER-ADDITIVE - the uniform twin burns MORE than "
                 "the capacity: the gated mass was PROTECTIVE at 0.25")
    elif n_burn >= 9:
        fork3 = ("(i) MATCHED - the dose alone carries the capacity's "
                 "census; composition second-order at this dose")
    elif n_burn >= 4:
        fork3 = ("(ii) SPLIT - both arms carry burns at 0.25")
    else:
        fork3 = ("(iii) THIN - the capacity's burns are the MIX's act: "
                 "the gated presence load-bearing at ~0.039 realized")
    fl = sorted(v for v in (seeds[s]["F_late"] for s in seeds)
                if v is not None)
    gaps = [(fl[i + 1] - fl[i], fl[i], fl[i + 1])
            for i in range(len(fl) - 1)]
    if gaps:
        gmax, glo, ghi = max(gaps)
        around_floor = bool(glo < COMP_BURN_FLOOR < ghi)
    else:
        gmax, glo, ghi, around_floor = None, None, None, None
    b_burn = sorted(seeds[s]["b_star"] for s in burners) if burners else []
    b_ok = sorted(seeds[s]["b_star"] for s in seeds if s not in burners)
    cls["BT3_census"] = {
        "n_burned": n_burn, "burners": burners,
        "n_on_course": verd.count("ON-COURSE"),
        "n_consumed": verd.count("CONSUMED"),
        "n_overfilled": verd.count("OVERFILLED"),
        "against": {"capacity": len(cap_burners),
                    "mirror": len(mirror_burners),
                    "control": len(control_burners)},
        "fork": fork3,
        "cold_road_in": [s for s in COLDROAD if s in burners],
        "cold_road_out": [s for s in COLDROAD if s not in burners],
        "deep_cluster_in": [s for s in DEEP if s in burners],
        "burners_b_star": b_burn,
        "survivors_b_star_max": (max(b_ok) if b_ok else None),
        "bstar_separates": bool(b_burn and b_ok
                                and max(b_ok) < min(b_burn)),
        "sorted_F_late": fl,
        "largest_internal_gap": gmax,
        "gap_location": [glo, ghi],
        "gap_straddles_burn_floor": around_floor,
        "moat_bar": 0.010,
        "ground_majority": max(set(verd), key=verd.count),
        "ground_mean_verdict": verdict_of(float(np.mean(fl))
                                          if fl else None)}

    # ---- BT4: the paired verdict table ------------------------------------
    pair = {}
    for s in sorted(seeds):
        if seeds[s]["cap_verdict"] is None:
            continue
        pair[str(s)] = {
            "cap": seeds[s]["cap_verdict"], "twin": seeds[s]["verdict"],
            "mirror": seeds[s].get("mirror_verdict"),
            "control": seeds[s].get("control_verdict"),
            "cap_F": seeds[s]["cap_F_late"],
            "twin_F": seeds[s]["F_late"],
            "T": seeds[s]["T_reg"], "b_star": seeds[s]["b_star"],
            "flipped": seeds[s]["cap_burned"] != seeds[s]["burned"]}
    flipped = [int(k) for k, v in pair.items() if v["flipped"]]
    b2c = [int(k) for k, v in pair.items()
           if v["flipped"] and v["cap"] == "BURNED"]
    c2b = [int(k) for k, v in pair.items()
           if v["flipped"] and v["twin"] == "BURNED"]
    same_burn = [int(k) for k, v in pair.items()
                 if (not v["flipped"]) and v["twin"] == "BURNED"]

    def grp(key, gs):
        v = [pair[str(s)][key] for s in gs if str(s) in pair]
        return {"mean": float(np.mean(v)) if v else None,
                "range": [float(min(v)), float(max(v))] if v else None,
                "n": len(v)}

    cls["BT4_paired_verdicts"] = {
        "per_seed": pair,
        "n_flipped": len(flipped),
        "flipped_seeds": flipped,
        "burn_to_consumed": b2c,
        "consumed_to_burn": c2b,
        "burned_in_both": len(same_burn),
        "the_flips_anatomy": {
            "T": grp("T", flipped),
            "b_star": grp("b_star", flipped),
            "cap_F_late": grp("cap_F", flipped),
            "unflipped_cap_F": grp(
                "cap_F", [int(s) for s in pair if not
                          pair[s]["flipped"]])},
        "note": "the per_seed table carries the four-world verdict map "
                "(capacity / mirror / control / twin) - the factorial's "
                "completed table"}

    # ---- BT5: the anchor decomposition ------------------------------------
    dec = {}
    for s in sorted(seeds):
        if seeds[s]["anchor_F_late"] is None:
            continue
        strat = ("burner" if seeds[s]["burned"] else
                 "cold-road" if s in COLDROAD else
                 "deep" if s in DEEP else "survivor")
        dec[str(s)] = {"stratum": strat,
                       "anchorF": seeds[s]["anchor_F_late"],
                       "armedF": seeds[s]["F_late"],
                       "diff": seeds[s]["F_late"] - seeds[s]["anchor_F_late"],
                       "T": seeds[s]["T_reg"]}
    a_burners = [v["anchorF"] for v in dec.values()
                 if v["stratum"] == "burner"]
    a_all = [v["anchorF"] for v in dec.values()]
    d_all = [v["diff"] for v in dec.values()]
    anch_mean = float(np.mean(a_all)) if a_all else None
    diff_mean = float(np.mean(d_all)) if d_all else None
    if a_burners and min(a_burners) < COMP_BURN_FLOOR:
        fork5 = ("(i) ANCHOR-CARRIED - at least one twin burner's "
                 "anchor already burns: the undefended uniform world "
                 "crashes on its own")
    elif n_burn == 0:
        fork5 = ("(ii) FILLS-AND-SURVIVES - no burns anywhere: the "
                 "anchor fills, the differential churns, the census "
                 "safe")
    elif diff_mean is not None and diff_mean < 0:
        fork5 = ("(iii) CHURN-DOMINATES - no anchor burns; the "
                 "differential (mean %+.4f) carries the fall"
                 % diff_mean)
    else:
        fork5 = ("(ii) FILLS-AND-SURVIVES - the anchor's fill "
                 "out-runs the churn even where seeds burn")

    def stored_anchor_mean(arm):
        v = [flate_of(x) for x in d78["arms"][arm]["runs"]]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    cls["BT5_anchor_decomposition"] = {
        "per_seed": dec,
        "twin_anchor_mean": anch_mean,
        "twin_differential_mean": diff_mean,
        "control_anchor_stored": stored_anchor_mean("S20_UNIF13_NOCH"),
        "mirror_anchor_stored": stored_anchor_mean("S20_MIXM_NOCH"),
        "fork": fork5}

    # ---- BT6: the landing and the exposures -------------------------------
    def stored_repair_rate(arm):
        rr = d78["arms"][arm]["runs"]
        tot = 0
        for x in rr:
            for it in x["items"]:
                if it["r"] >= PERT_ROUND:
                    tot += it.get("n_rep", 0)
        return tot / len(rr)

    tot_rep = sum(seeds[s]["cum"]["rep"] for s in seeds)
    tot_wrong = sum(seeds[s]["cum"]["rep_wrong"] for s in seeds)
    tot_pass = sum(seeds[s]["cum"]["flip_pass"] for s in seeds)
    tot_pass_blind = sum(seeds[s]["cum"]["flip_pass_blind"]
                         for s in seeds)
    cls["BT6_landing_exposures"] = {
        "repairs_total": tot_rep,
        "repairs_per_seed": tot_rep / len(seeds),
        "repairs_wrong_rate": (tot_wrong / tot_rep if tot_rep else None),
        "against_stored": {
            "control_repairs_per_seed": stored_repair_rate("S20_UNIF13"),
            "mirror_repairs_per_seed": stored_repair_rate("S20_MIXM")},
        "flip_pass_total": tot_pass,
        "flip_pass_blind_total": tot_pass_blind,
        "blind_landing_fraction": (tot_pass_blind / tot_pass
                                   if tot_pass else None),
        "control_stored_fraction": 1.0,
        "persist_share_late_mean": float(np.mean(
            [seeds[s]["persist_share_late"] for s in seeds
             if seeds[s]["persist_share_late"] is not None])),
        "reading": "the channel's signature at the higher dose: does "
                   "the poison that passes still land panel-blind in "
                   "every instance, and is the exposure structure "
                   "still stratum-flat?"}

    # ---- the non-claims (static) ------------------------------------------
    cls["BT_nonclaims"] = {
        "no_amendment": "no registration changed, no letter moved - "
                        "the twin's columns are registered ALONGSIDE "
                        "the standing ones",
        "one_dose": "the twin is ONE world at ONE realized dose "
                    "(0.25) in ONE mode family (uniform) - no claim "
                    "about other doses or the gate/joint modes (doc "
                    "78's threat (iii), carried)",
        "bstar_reading": "the burners' b* reading is directional "
                         "(disclosed post-hoc, as at doc 78's "
                         "mirror); the registered fork is about "
                         "counts",
        "no_causal_claim": "the mode's causal role stays an "
                           "interpretation - the battery varies the "
                           "poison's composition, not the defense",
        "integrity": {"any_MAINTAINS": False,
                      "column3": "unmarked",
                      "audit": "60 per-seed audits (48 clean-prefix: "
                               "the twin and its anchor vs doc 73 "
                               "armed and doc 75 anchor, all 24 seeds; "
                               "12 full-length cross-instrument: seeds "
                               "600-605 vs doc 71's stored 0.25 cells)"}}
    return cls


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    if smoke:
        globals()["SEEDS"] = [600, 601]
        for a in GROUPS["bulktwin"]:
            globals()["C13CFG"][a]["seeds"] = [600, 601]
            globals()["C13CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s16_unif25_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s16_unif25_results.json")
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

    arms = GROUPS["bulktwin"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C13CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj, ilogs, plogs, b_clean = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj, "items": ilogs,
                         "plogs": plogs, "b_clean": b_clean})
            print(f"  {arm} seed={s} done (b_clean={b_clean:.4f}) "
                  f"({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- the per-seed audits (three families) ---------------------------
    if not smoke:
        need_ps = [a for a, _, _, _, _ in PERSEED_AUDITS if a in res["arms"]]
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
        need = set(GROUPS["bulktwin"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\n=== the bulk-matched twin "
                  "(uniform 0.25 at the capacity's dose) ===")
            print("BT1 certificate:", {
                "T_agree": c["BT1_paired_certificate"][
                    "T_agreement_with_doc73"]["all"],
                "b*_agree": c["BT1_paired_certificate"][
                    "bstar_agreement_with_doc73"]["all"]})
            hdr = ["seed", "coh", "dial", "T_reg", "b*", "F_late",
                   "lateb_r", "anchF", "capF", "verd", "capverd"]
            print(("{:>5s} {:>4s} {:>4s} {:>7s} {:>7s} {:>8s} "
                   "{:>8s} {:>8s} {:>8s} {:>9s} {:>9s}").format(*hdr))
            for s, r in c["BT_seeds"].items():
                def f4(v):
                    return f"{v:+.4f}" if isinstance(v, float) else "-"
                def f4p(v):
                    return f"{v:.4f}" if isinstance(v, float) else "-"
                print(("{:>5d} {:>4s} {:>4s} {:>7s} {:>7s} {:>8s} "
                       "{:>8s} {:>8s} {:>8s} {:>9s} {:>9s}").format(
                    s, r["cohort"][:4], "yes" if r["in_dial6"] else "-",
                    f4p(r["T_reg"]), f4p(r["b_star"]), f4(r["F_late"]),
                    f4p(r["late_b_r"]), f4(r["anchor_F_late"]),
                    f4(r["cap_F_late"]), r["verdict"],
                    str(r["cap_verdict"])))
            print("\nBT2 matched dose:", c["BT2_matched_dose"])
            print("\nBT3 census:", c["BT3_census"])
            print("\nBT4 paired:", {k: v for k, v in
                  c["BT4_paired_verdicts"].items()
                  if k != "per_seed"})
            print("\nBT5 anchor decomposition:",
                  {k: v for k, v in
                   c["BT5_anchor_decomposition"].items()
                   if k != "per_seed"})
            print("\nBT6 landing and exposures:",
                  c["BT6_landing_exposures"])
            print("\nBT1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
