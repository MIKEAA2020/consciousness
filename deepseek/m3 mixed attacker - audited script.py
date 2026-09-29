#!/usr/bin/env python3
"""
THE MIXED ATTACKER - doc 41's filed synthesis, run at last. Corpus doc 44's
experiment. Ordered at the panel-poison issuance: "the mixed attacker
(uniform on the confident bulk + gated on the blind spot, the dose split at
the capacity bound - doc 41's filed synthesis, the obvious combination of
the two surviving attacks)."

THE SYNTHESIS, AS FILED (doc 41, threats (vi) and the coda): "the mixed
attacker - uniform on the confident bulk plus gated on the blind spot,
splitting the dose - is unmeasured and is the obvious synthesis of this
battery's two surviving attacks." The two survivors it combines: the
UNIFORM attack (doc 34's, the awarded world - caught at 0.88-0.92, its
surviving dirt the channel's residual) and the GATED attack (doc 41's -
wholly undisputed, capacity-bound by the blind spot, harmless in this
chassis). The mix's theory of victory: the bulk part keeps the channel
honestly busy (catches it can point to), the gated part rides the blind
spot whole - and the catch-rate certificate (doc 43, now built) launders
the composition, because the ratio it divides cannot see WHERE the dirt
lives.

THE ATTACKER'S ARITHMETIC (pre-registered before the run): the mix
consumes the uniform draws FIRST (the same stream, the same items as a
pure-U world at lam_bulk - the bitwise anchor), flips them to (y+1)%10
(the arbitrary direction, doc 34 verbatim), then fills the REMAINING
blind spot (sub-tau0 commits not already bulk-flipped) with gated
runner-up flips (doc-36-literal ranking, panel confidence descending,
m = round(frac_blind*n), the farming direction). Realized dose
~ lam_bulk + min(frac_blind, blind mass net of bulk overlap); surviving
dirt ~ (1-catch)*lam_bulk + the gated fill whole.

THE BATTERY (doc 36/41's chassis; the certificate of doc 43 riding as
passive telemetry; the BASE dial; 14 rounds; N=6):
  MX_TWIN       the bitwise anchor (audits to doc 41's PP_TWIN)
  MX_U30        pure bulk, lam_bulk = 0.30 (audits to PP_U30)
  MX_J30        pure gated, frac = 0.30 (audits to PP_J30)
  MX_J10        pure gated, frac = 0.10 (audits to PP_J10)
  MX_CAP30      THE FILED SYNTHESIS at 0.30: bulk 0.22 + gated 0.08
                (the capacity split - the blind spot holds ~8-9%)
  MX_HALF30     bulk 0.15 + gated 0.15 (half/half; the fill caps)
  MX_CAP10      the capacity split at 0.10: bulk 0.02 + gated 0.08
                (blind-heavy - the ratio's far side)
  MX_HALF10     bulk 0.05 + gated 0.05
  MX_CAP30_NOCH / MX_CAP10_NOCH   the disarmed anchors under the mix
                (the registry effect measured UNDER the attack faced)

PRE-REGISTERED PREDICTIONS (fixed before execution):
  MX1  the bitwise audits: MX_TWIN, MX_U30, MX_J30, MX_J10 reproduce
       doc 41's counterparts (tau/kappa, 1e-9) - the mix reduces to its
       components exactly at the corners.
  MX2  THE ARITHMETIC: the mix's surviving dirt exceeds both singles at
       the same total dose (the gated part survives whole, the bulk part
       at the (1-catch) residual); the realized split measured per round
       (bulk realized, gated fill, the no-op fraction on the fill ~10%
       of it, doc 41's filter).
  MX3  THE WORLD: fork - (a) the clean ruler holds under every mix (the
       bulk dirt caught at 0.9, the blind dirt harmless; the world
       stands, as it stood under J), or (b) the mix's surviving bulk
       dirt (the loop's fuel, on the confident bulk) drags the ruler
       below both singles (the synthesis beats both attacks at damage).
  MX4  THE LAUNDERING (the battery's point): the certificate's C under
       the 0.30 mixes lands in [0.40, 0.75] - CATCHING - while a third
       of the dose evades the channel wholly: the ratio certifies the
       catch, not the coverage. At MX_CAP10 (bulk 0.02) the blind
       dominates and C < 0.25 - EVASION. The laundering boundary is the
       measured map: the mix evades the certificate iff the gated fill
       outweighs the bulk's honest contribution to the ratio.
  MX5  THE AWARD: the four doc-39 conditions under the mix, with (iii)
       measured against the NOCH anchors. Fork: the 0.30 mixes PASS all
       four (the world is defended where the world is damageable - the
       award's conditions are world-anchored, and a partial coverage
       that holds the ruler is a true premise, not a false one), or the
       registry effect under the mix falls below 0.30 (the channel's
       contribution, bulk-only, priced below the registered bar). Either
       way the award's semantics under mixed attack is the finding.
  MX6  THE WORLD-SHAPE: q_bar, the commit gate, and the gate fraction
       under the mix vs the singles - the mix reads as a tighter gate
       over a cleaner pool (the selectivity signature, containment-
       shaped), with the disarmed mix collapsing (the loop's floor).

Usage: python3 m3_mixed.py [--smoke]
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
COMP_REG_END = 5                        # the clean prefix (the cone window)

# ---------------- the certificate's frozen registration (doc 43's) --------
CERT_FLOOR = 0.25
CERT_XMIN = 0.01
CERT_RJUMP = 0.01
ETA_RORG = 0.3

# ---------------- the battery ----------------------------------------------
MXCFG = {
    "MX_TWIN":      dict(bulk=0.0,  gfrac=0.0,  noch=False),
    "MX_U30":       dict(bulk=0.30, gfrac=0.0,  noch=False),
    "MX_J30":       dict(bulk=0.0,  gfrac=0.30, noch=False),
    "MX_J10":       dict(bulk=0.0,  gfrac=0.10, noch=False),
    "MX_CAP30":     dict(bulk=0.22, gfrac=0.08, noch=False),
    "MX_HALF30":    dict(bulk=0.15, gfrac=0.15, noch=False),
    "MX_CAP10":     dict(bulk=0.02, gfrac=0.08, noch=False),
    "MX_HALF10":    dict(bulk=0.05, gfrac=0.05, noch=False),
    "MX_CAP30_NOCH": dict(bulk=0.22, gfrac=0.08, noch=True),
    "MX_CAP10_NOCH": dict(bulk=0.02, gfrac=0.08, noch=True),
}
ARMS = list(MXCFG.keys())
DOC41 = "/home/z/my-project/scripts/m3_panelpoison_results.json"
AUDIT41 = [("MX_TWIN", "PP_TWIN"), ("MX_U30", "PP_U30"),
           ("MX_J30", "PP_J30"), ("MX_J10", "PP_J10")]
DOC39_REGISTRY_EFFECT = 0.7592592592592594   # doc 39's NOCH delta (uniform)


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
    pbar_sum = np.zeros((len(X), 10))
    for snap in panel:
        W1, b1, heads = snap
        _, zs = forward_all(X, W1, b1, heads)
        for z in zs:
            pbar_sum += softmax(z)
    pbar = pbar_sum / (len(panel) * K)
    return pbar.argmax(1), pbar.max(1)


def panel_top2(X, panel):
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


# ---------------- THE MIXED ATTACK (the synthesis, as filed) -------------
def poison_mixed(yt_, Xt_, panel, rng_attack, lam_bulk, gfrac):
    """Bulk first, gated second. The bulk path consumes the attack stream
    exactly as doc 34/41's uniform (the bitwise anchor at gfrac = 0); the
    gated fill is deterministic (no rng - the pure-gated corner consumes
    nothing, bitwise to doc 41's J). Returns (labels, bulk_mask,
    gate_mask). No-op flips excluded from both masks."""
    n = len(yt_)
    if n == 0 or (lam_bulk <= 0.0 and gfrac <= 0.0):
        return yt_.copy(), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    out = yt_.copy()
    bulk = np.zeros(n, dtype=bool)
    gate = np.zeros(n, dtype=bool)
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
    return out, bulk, gate


def tension(p_star, live):
    return ((1 - ETA) * RHO_T * p_star + ETA * live) / (ETA + RHO_T - ETA * RHO_T)


# ---------------- the arm (the chassis + the mix + the certificate) ------
def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    cfg = MXCFG[arm]
    lam_bulk, gfrac = cfg["bulk"], cfg["gfrac"]
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

    # ---- the certificate's registration (doc 43's, verbatim) ------------
    iota_star_cert = iota_star
    if would.any():
        r_org0 = float(np.mean((pl0 != yb0) & (pc0 >= TAU0)))
    else:
        r_org0 = 0.0
    r_org = r_org0
    delta_hat = 0.0
    iota_hist = []
    cert_X, cert_N = [], []

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, 15):
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
        # ---- 3. the integrity channel's arming (out-of-band check) ------
        reading_oob = bool(abs(iota_sys - iota_star) > EPS_I)
        armed = reading_oob and not noch
        # ---- 4. grading, brake, THE POISON ------------------------------
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p, bulk_m, gate_m = poison_mixed(
            yt_, Xt_, panel, rng_attack,
            lam_bulk if r >= PERT_ROUND else 0.0,
            gfrac if r >= PERT_ROUND else 0.0)
        flipped = bulk_m | gate_m
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(flipped.sum())
        tele["n_bulk_flipped"] = int(bulk_m.sum())
        tele["n_gate_flipped"] = int(gate_m.sum())
        # ---- 5. the integrity channel: read, then act -------------------
        pl, pconf = panel_grade(Xt_, panel)
        d_recv = float(np.mean(pl != yt_p)) if len(yt_p) else 0.0
        disputed = (pl != yt_p) & (pconf >= TAU0)
        n_rep = int(disputed.sum())
        n_dis_raw = int(disputed.sum())
        tele["n_flip_disputed"] = int((disputed & flipped).sum())
        tele["n_bulk_disputed"] = int((disputed & bulk_m).sum())
        tele["n_gate_disputed"] = int((disputed & gate_m).sum())
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

        # ---- 5b. THE CERTIFICATE (doc 43's, verbatim - passive) ----------
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
        if (not reading_oob) and n_com_r:
            if abs(drate - r_org) <= CERT_RJUMP:
                r_org = (1 - ETA_RORG) * r_org + ETA_RORG * drate

        # ---- 6. training on the (possibly cleaned) stream ---------------
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
        iota_sys = (1 - ETA_I) * iota_sys + ETA_I * d_recv
        iota_hist.append(iota_sys)
        if r == COMP_REG_END:
            ys = np.array(iota_hist[:COMP_REG_END])
            xs = np.arange(1, COMP_REG_END + 1)
            delta_hat = float(np.polyfit(xs, ys, 1)[0])
        # ---- 8. telemetry -------------------------------------------------
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
            "variant": "NOCH" if noch else "REPAIR",
            "n_disputed_repaired": n_rep, "n_dis_raw": n_dis_raw,
            "cert_base": cert_base_r, "x_raw": x_raw, "n_raw": n_raw,
            "x_cum": x_cum, "n_cum": n_cum, "c_ratio": c_ratio,
            "cert_verdict": cert_verdict, "r_org": r_org,
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


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        globals()["PERT_ROUND"] = 3
        globals()["SEEDS"] = [600, 601]
        globals()["ARMS"] = ["MX_TWIN", "MX_U30", "MX_J30", "MX_CAP30",
                             "MX_CAP30_NOCH"]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    res = {"config": {
        "tau0": TAU0, "kappa0": KAPPA0, "eps_i": EPS_I, "eta_i": ETA_I,
        "rounds": 14, "pert_round": PERT_ROUND, "nseed": len(SEEDS),
        "cert_floor": CERT_FLOOR, "cert_xmin": CERT_XMIN,
        "cert_rjump": CERT_RJUMP, "eta_rorg": ETA_RORG,
        "comp_reg_end": COMP_REG_END,
        "attack_stream_offset": ATTACK_STREAM,
        "panel_epochs": PANEL_EPOCHS,
        "doc39_registry_effect_uniform": DOC39_REGISTRY_EFFECT,
        "numpy": np.__version__},
           "arms": {}}

    for arm in ARMS:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- bitwise audits vs doc 41 (the corners) ---------------------------
    if not smoke:
        try:
            d41 = json.load(open(DOC41))
            for ours, theirs in AUDIT41:
                ref_t = d41["arms"][theirs]["aggregate"]["tau_sys"][:14]
                ref_k = d41["arms"][theirs]["aggregate"]["kappa_sys"][:14]
                got_t = [float(np.mean([rr["traj"][i]["tau_sys"]
                                        for rr in res["arms"][ours]["runs"]]))
                         for i in range(14)]
                got_k = [float(np.mean([rr["traj"][i]["kappa_sys"]
                                        for rr in res["arms"][ours]["runs"]]))
                         for i in range(14)]
                ok = bool(all(abs(a - b) < 1e-9 for a, b in zip(ref_t, got_t))
                          and all(abs(a - b) < 1e-9
                                  for a, b in zip(ref_k, got_k)))
                res.setdefault("audits", {})[
                    f"{ours}_reproduces_doc41_{theirs}"] = ok
                print(f"{ours} reproduces doc 41 {theirs} bitwise: {ok}")
        except (FileNotFoundError, KeyError) as e:
            print(f"doc-41 audit unavailable: {e}")

    # ---- aggregates -------------------------------------------------------
    for arm in ARMS:
        rows = res["arms"][arm]["runs"]
        agg = {}
        keys = ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                "q_target", "kappa_target", "accept_own_tau",
                "cerr_self", "cerr_stream_pre", "cerr_stream_post",
                "panel_acc", "iota_sys", "d_recv", "blind_mass",
                "realized_dose", "x_raw", "n_raw", "x_cum", "n_cum",
                "c_ratio", "r_org", "cert_base", "n_dis_raw",
                "n_disputed_repaired", "n_flipped", "n_flip_disputed",
                "n_bulk_flipped", "n_gate_flipped", "n_bulk_disputed",
                "n_gate_disputed"]
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
        vagg = []
        for i in range(len(rows[0]["traj"])):
            vs = [r["traj"][i].get("cert_verdict", "MUM") for r in rows]
            vagg.append(max(set(vs), key=vs.count))
        agg["cert_verdict"] = vagg
        res["arms"][arm]["aggregate"] = agg
        cstr = ['%.3f' % c if c is not None else 'nan'
                for c in agg['c_ratio']]
        print(f"{arm:13s} tau: {['%.3f' % a for a in agg['tau_sys'][:7]]}"
              f"...")
        print(f"{'':13s} C:   {cstr}")
        print(f"{'':13s} verdict: {agg['cert_verdict']}")

    # ---- classification (pre-registered) ----------------------------------
    if not smoke:
        cls = {}
        A = res["arms"]

        def ag(arm, key):
            return A[arm]["aggregate"][key]

        def late_mean(arm, key, lo=None):
            lo = lo if lo is not None else PERT_ROUND - 1
            v = [x for x in ag(arm, key)[lo:] if x is not None]
            return float(np.mean(v)) if v else None

        def catch_rate(arm, numk, denk):
            num = late_mean(arm, numk)
            den = late_mean(arm, denk)
            return (num / den) if (den and den > 0.1) else None

        # MX1: the audits
        cls["MX1_bitwise"] = dict(res.get("audits", {}))
        # MX2: the arithmetic of surviving dirt
        mx2 = {}
        for arm in ARMS:
            if arm == "MX_TWIN":
                continue
            surv_bulk = catch_rate(arm, "n_bulk_flipped", "n_commit") \
                - catch_rate(arm, "n_bulk_disputed", "n_commit")
            surv_gate = catch_rate(arm, "n_gate_flipped", "n_commit")
            mx2[arm] = {
                "bulk_realized": catch_rate(arm, "n_bulk_flipped",
                                            "n_commit"),
                "bulk_caught": catch_rate(arm, "n_bulk_disputed",
                                          "n_commit"),
                "gate_fill": catch_rate(arm, "n_gate_flipped", "n_commit"),
                "gate_caught": catch_rate(arm, "n_gate_disputed",
                                          "n_commit"),
                "surviving_bulk": surv_bulk,
                "surviving_gate": surv_gate,
                "surviving_total": (surv_bulk or 0.0) + (surv_gate or 0.0),
                "total_dose": late_mean(arm, "realized_dose"),
                "blind_mass": late_mean(arm, "blind_mass")}
        # the singles for comparison
        mx2["ref_U30_surviving"] = (
            catch_rate("MX_U30", "n_bulk_flipped", "n_commit")
            - catch_rate("MX_U30", "n_bulk_disputed", "n_commit"))
        mx2["ref_J30_surviving"] = catch_rate("MX_J30", "n_gate_flipped",
                                              "n_commit")
        cls["MX2_arithmetic"] = mx2
        # MX3: the world (the clean ruler)
        mx3 = {}
        for arm in ARMS:
            mx3[arm] = {
                "final_clean": ag(arm, "acc")[-1],
                "q_bar": float(np.mean([x for x in
                                        ag(arm, "q_target")[PERT_ROUND - 1:]
                                        if x is not None])),
                "held": bool(ag(arm, "acc")[-1] >= 0.93)}
        cls["MX3_world"] = mx3
        # MX4: the laundering (the certificate under the mix)
        mx4 = {}
        for arm in ARMS:
            rows = A[arm]["runs"]
            fin_c = [rr["traj"][-1]["c_ratio"] for rr in rows]
            fin_c = [c for c in fin_c if c is not None]
            verdicts = [rr["traj"][-1]["cert_verdict"] for rr in rows]
            mx4[arm] = {
                "final_C_mean": float(np.mean(fin_c)) if fin_c else None,
                "final_verdicts": verdicts,
                "CATCHING": verdicts.count("CATCHING"),
                "EVASION": verdicts.count("EVASION"),
                "MUM": verdicts.count("MUM")}
        cls["MX4_laundering"] = mx4
        # MX5: the award re-grade (doc 39's four conditions)
        mx5 = {}
        noch_anchor = {"MX_CAP30": "MX_CAP30_NOCH",
                       "MX_CAP10": "MX_CAP10_NOCH"}
        for arm in ["MX_U30", "MX_J30", "MX_J10", "MX_CAP30",
                    "MX_HALF30", "MX_CAP10", "MX_HALF10"]:
            acc_f = ag(arm, "acc")[-1]
            qbar = float(np.mean([x for x in
                                  ag(arm, "q_target")[PERT_ROUND - 1:]
                                  if x is not None]))
            final = ag(arm, "tau_sys")[-1]
            slope = (ag(arm, "tau_sys")[-1] - ag(arm, "tau_sys")[-4]) / 3.0
            oob_wo = sum(1 for i in range(PERT_ROUND - 1, 14)
                         if ag(arm, "tau_sys")[i] < TAU0 - EPS_T
                         and ag(arm, "repair_fired")[i] < 0.5)
            n_armed = int(sum(1 for v in ag(arm, "armed") if v >= 0.5))
            if arm in noch_anchor:
                eff = (ag(arm, "acc")[-1]
                       - ag(noch_anchor[arm], "acc")[-1])
                eff_kind = "measured-under-attack"
            else:
                eff = DOC39_REGISTRY_EFFECT
                eff_kind = "inherited-uniform"
            cond1 = bool(acc_f >= 0.93)
            cond2 = bool(n_armed >= 1 and oob_wo == 0)
            cond3 = bool(eff >= 0.30)
            cond4 = bool(tension(TAU0, qbar) <= final <= TAU0 + EPS_T
                         and slope >= -1e-12)
            mx5[arm] = {
                "clean_ruler": acc_f, "q_bar": qbar, "tau_final": final,
                "late_tau_slope": slope,
                "tension_floor": tension(TAU0, qbar),
                "registry_effect": eff, "registry_effect_kind": eff_kind,
                "(i)_clean": cond1, "(ii)_engagement": cond2,
                "(iii)_registry": cond3, "(iv)_bracket": cond4,
                "awarded_CONTAINED_UNDER_8_4": bool(
                    cond1 and cond2 and cond3 and cond4)}
        cls["MX5_award_regrade"] = mx5
        # MX6: the world-shape (selectivity proxies)
        mx6 = {}
        for arm in ["MX_TWIN", "MX_U30", "MX_J30", "MX_CAP30",
                    "MX_CAP30_NOCH"]:
            mx6[arm] = {
                "gate_frac_final": ag(arm, "accept09")[-1],
                "n_commit_final": ag(arm, "n_commit")[-1],
                "q_final": ag(arm, "q_target")[-1],
                "acc_final": ag(arm, "acc")[-1]}
        cls["MX6_worldshape"] = mx6
        res["classification"] = cls
        print("MX5 award re-grade:")
        for arm, row in cls["MX5_award_regrade"].items():
            print(f"  {arm}: award={row['awarded_CONTAINED_UNDER_8_4']} "
                  f"clean={row['clean_ruler']:.4f} "
                  f"(i){row['(i)_clean']} (ii){row['(ii)_engagement']} "
                  f"(iii){row['(iii)_registry']} (iv){row['(iv)_bracket']}")
        print("MX4 laundering:")
        for arm, row in cls["MX4_laundering"].items():
            print(f"  {arm}: C={row['final_C_mean']} "
                  f"CATCH={row['CATCHING']} EVAS={row['EVASION']}")

    res["runtime_s"] = time.time() - t0
    out = ("/home/z/my-project/scripts/m3_mixed_results_smoke.json"
           if smoke else
           "/home/z/my-project/scripts/m3_mixed_results.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"wrote {out} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
