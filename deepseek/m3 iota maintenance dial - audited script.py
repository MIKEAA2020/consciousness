#!/usr/bin/env python3
"""
THE IOTA MAINTENANCE DIAL - doc 36's three filed fixes, run at last. Corpus
doc 40's experiment. Ordered at the second-candidate issuance: "the iota
maintenance dial (doc 36's three filed fixes: the wider band at the tau
convention, periodic re-registration, drift compensation - the twin's
false-positive horizon)."

THE PROBLEM, AS MEASURED (doc 36 section 2.3 and doc 39's Group A): the
frozen panel diverges from the live committee at ~0.2pp per round of
healthy self-training, so the eps_i = 0.03 deadband buys roughly fourteen
rounds before the first false alarm - and the second candidate's battery
found the horizon is ARM-DEPENDENT: every arm that holds the commit gate
soft (record poison, joint kicks) or degrades the committee (substrate
damage) shortens it to 8-10 rounds. The dial's question: what does each
fix buy, and what does it cost?

THE THREE FIXES, AS FILED (doc 36 section 2.3: "the tau convention's 0.05
would buy roughly twenty-five [rounds]; re-registration would buy them all
back. None of these is run here - the coordinate's maintenance schedule is
filed as the next build's dial"):
  WIDE   eps_i = 0.05 (the tau convention). Buys horizon by the law
         rounds ~ eps_i/drift. Costs: the low-dose catch - at lambda =
         0.05 the received stream's footprint is ~0.84*lambda = 0.042,
         and 0.042 + drift must clear 0.05 before the EMA does: the
         bottom dose's arming is delayed or lost. THE TRADE.
  REREG  periodic re-registration: every k = 5 rounds, IF the reading is
         in-band, re-register iota* <- iota_sys (the in-band guard: never
         re-baseline while the channel reads dirt). Buys the horizon back
         by re-anchoring. Costs: the record becomes a course, not a
         constant - doc 36's I1 ("the healthy panel-committee
         disagreement is a stationary reading") is given up, and the
         guard's safety at the measured doses is an empirical question
         (the sub-onset laundering cell - a dirt rate small enough to
         stay in-band while re-registration launders it - is BELOW the
         0.05 onset and stays filed, unmeasured, doc 34's dial).
  COMP   drift compensation: the registered course. delta_hat measured at
         registration on the known-clean prefix (rounds 1-5, pre-attack:
         the least-squares slope of iota_sys), the band becomes the cone
         iota*(r) = iota* + delta_hat * max(0, r - 5) +- eps_i. Buys the
         horizon by tracking the drift. Costs: the record moves on a
         schedule (doc 23's course concept applied to the RECORD - the
         corpus's first scheduled record), and the compensation is only
         as good as the drift's linearity - the residual is measured.

THE ARMS (doc 36's stream chassis in every particular; the twin arms run
30 rounds to see the horizon move; the dose arms 14):
  ID_TWIN_BASE   eps_i = 0.03, 30 rounds - the baseline horizon (audits
                 bitwise to doc 36's IC_TWIN on the first 14 rounds,
                 false positive and all)
  ID_TWIN_WIDE   eps_i = 0.05, 30 rounds - the law's prediction ~23-27
  ID_TWIN_COMP   COMP, 30 rounds - the cone vs the drift
  ID_TWIN_REREG  REREG, 30 rounds - the re-anchoring
  ID_P05_BASE    lambda = 0.05, eps = 0.03 (bitwise to doc 36's IC_P05)
  ID_P05_WIDE    lambda = 0.05, eps = 0.05 - THE TRADE (the fork: armed
                 late, or never within 14)
  ID_P05_COMP    lambda = 0.05, COMP - the catch preserved?
  ID_P05_REREG   lambda = 0.05, REREG - the guard holds?
  ID_P30_BASE    lambda = 0.30, eps = 0.03 (bitwise to IC_P30)
  ID_P30_COMP    lambda = 0.30, COMP - the deep-dose catch

PRE-REGISTERED PREDICTIONS (fixed before execution):
  ID1  BASE's false positive: every seed arms within the 30-round horizon,
       first arming r13-17, 3-15 labels rewritten; the drift rate measured
       at 0.19-0.23pp/round (the horizon law's denominator).
  ID2  WIDE extends the horizon to r23-27 per the law (or censors at 30,
       disclosed); the 0.30-dose catch is unchanged (armed r7 - the 0.25
       footprint clears any band).
  ID3  THE TRADE at 0.05: WIDE's arming is delayed past BASE's r8 - fork:
       (a) armed by r12 (the drift term pushes the EMA over), or (b)
       never within 14 (the catch lost at the bottom dose).
  ID4  COMP kills the false positive for all 30 rounds at every seed (the
       residual |iota - cone| stays under eps_i; the margin reported);
       the catch at both doses preserved (the level shift 0.042+ dwarfs
       the cone's slope error).
  ID5  REREG kills the false positive (the level re-anchors every 5
       rounds); the record's total drift over 30 rounds ~ 0.006-0.010
       (the scheduled-record price, disclosed); the catch at both doses
       preserved (the guard blocks re-registration while out-of-band).
  ID6  The cleaning and the clean ruler at dose are unchanged under COMP
       and REREG (the defense's substance is dose-driven, not
       band-driven); under WIDE at 0.05, if the catch is lost the loop
       returns (acc falls toward doc 34's floor) - the trade's price on
       the ruler.

Usage: python3 m3_iota_dial.py [--smoke]
"""
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration (doc 36's, the dial axis) -----------
K = 4
TAU0, KAPPA0 = 0.9, 0.95
EPS_T, RHO_T = 0.05, 0.5
EPS_K, RHO_K = 0.05, 0.5
ETA, ETA_K = 0.3, 0.3
ETA_I = 0.3
BRAKE_TH = 0.60
KAP_FLOOR = 0.5
BURN_EPOCHS, SELF_EPOCHS = 30, 4
VIEWS = 6
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
PERT_ROUND = 6
NSEED = 6
SEEDS = list(range(600, 600 + NSEED))
KICK_TAU, KICK_KAP = 0.75, 0.80
ATTACK_STREAM = 977
PANEL_EPOCHS = [20, 25, 30]
REREG_K = 5                            # the re-registration period
COMP_REG_END = 5                       # the clean prefix (the slope window)

DIALCFG = {
    "ID_TWIN_BASE":  dict(lam=0.0,  kicks=False, eps_i=0.03, fix="none",
                          rounds=30),
    "ID_TWIN_WIDE":  dict(lam=0.0,  kicks=False, eps_i=0.05, fix="none",
                          rounds=30),
    "ID_TWIN_COMP":  dict(lam=0.0,  kicks=False, eps_i=0.03, fix="comp",
                          rounds=30),
    "ID_TWIN_REREG": dict(lam=0.0,  kicks=False, eps_i=0.03, fix="rereg",
                          rounds=30),
    "ID_P05_BASE":   dict(lam=0.05, kicks=True,  eps_i=0.03, fix="none",
                          rounds=14),
    "ID_P05_WIDE":   dict(lam=0.05, kicks=True,  eps_i=0.05, fix="none",
                          rounds=14),
    "ID_P05_COMP":   dict(lam=0.05, kicks=True,  eps_i=0.03, fix="comp",
                          rounds=14),
    "ID_P05_REREG":  dict(lam=0.05, kicks=True,  eps_i=0.03, fix="rereg",
                          rounds=14),
    "ID_P30_BASE":   dict(lam=0.30, kicks=True,  eps_i=0.03, fix="none",
                          rounds=14),
    "ID_P30_COMP":   dict(lam=0.30, kicks=True,  eps_i=0.03, fix="comp",
                          rounds=14),
}
ARMS = list(DIALCFG.keys())

DOC36 = "/home/z/my-project/scripts/m3_integrity_results.json"


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
    return [W1.copy(), b1.copy(), [[w.copy() for w in h] for h in heads]]


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


def panel_grade(X, panel):
    pbar_sum = np.zeros((len(X), 10))
    for snap in panel:
        W1, b1, heads = snap
        _, zs = forward_all(X, W1, b1, heads)
        for z in zs:
            pbar_sum += softmax(z)
    pbar = pbar_sum / (len(panel) * K)
    return pbar.argmax(1), pbar.max(1)


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


def poison_labels(yt_, rng_attack, lam):
    if lam <= 0.0 or len(yt_) == 0:
        return yt_.copy()
    mask = rng_attack.random(len(yt_)) < lam
    out = yt_.copy()
    out[mask] = (out[mask] + 1) % 10
    return out


# ---------------- the dial's arm ----------------
def run_arm(seed, arm, Xtr, ytr, Xte, yte):
    """Doc 36's run_arm with the dial: configurable eps_i, the COMP cone,
    the REREG schedule. The BASE arms are bitwise to doc 36 (identical
    code paths); the fixes change only the arming logic."""
    cfg = DIALCFG[arm]
    lam, kicks, eps_i, fix, rounds = (cfg["lam"], cfg["kicks"],
                                      cfg["eps_i"], cfg["fix"],
                                      cfg["rounds"])
    rng = np.random.default_rng(seed)
    rng_attack = np.random.default_rng(seed + ATTACK_STREAM)
    state, mom, panel = burn_in_with_panel(rng, Xtr, ytr)

    pool0 = perturb(Xtr, rng)
    _, ybar0, conf0, dis0, flag0, _ = flags_and_probs(pool0, state, KAPPA0)
    r_star = float(np.mean(conf0 >= TAU0))
    f_star = float(np.mean(flag0))
    tau_star, kappa_star = TAU0, KAPPA0
    would = (conf0 >= TAU0) & (~flag0)
    pl0, pc0 = panel_grade(pool0[would] if would.any() else pool0, panel)
    iota_star = float(np.mean(pl0 != (ybar0[would] if would.any()
                                      else ybar0)))
    iota_sys = iota_star
    delta_hat = 0.0
    iota_hist = []

    tau_sys, kappa_sys = TAU0, KAPPA0
    traj = []
    for r in range(1, rounds + 1):
        # ---- 1. the attack: state kicks ------------------------------------
        if r == PERT_ROUND and kicks:
            tau_sys = KICK_TAU
            kappa_sys = KICK_KAP
        # ---- 2. out-of-gate repair channels (tau, kappa) -------------------
        repair_t = repair_k = False
        if abs(tau_sys - tau_star) > EPS_T:
            tau_sys += RHO_T * (tau_star - tau_sys)
            repair_t = True
        if abs(kappa_sys - kappa_star) > EPS_K:
            kappa_sys += RHO_K * (kappa_star - kappa_sys)
            repair_k = True
        # ---- 2b. the DIAL: the arming check, by fix ------------------------
        if fix == "comp":
            baseline = iota_star + delta_hat * max(0, r - COMP_REG_END)
            reading_oob = bool(abs(iota_sys - baseline) > eps_i)
        else:
            reading_oob = bool(abs(iota_sys - iota_star) > eps_i)
        armed = reading_oob
        # ---- 3. grading, brake, THE POISON ----------------------------------
        Xt_, yt_, tele, pool, yt_true = self_round(
            state, rng, Xtr, ytr, tau_sys, kappa_sys)
        yt_p = poison_labels(yt_, rng_attack, lam if r >= PERT_ROUND else 0.0)
        tele["cerr_stream_pre"] = (float(np.mean(yt_p != yt_true))
                                   if len(yt_p) else 0.0)
        tele["n_flipped"] = int(np.sum(yt_p != yt_))
        # ---- 4. the integrity channel: read, then act -----------------------
        pl, pconf = panel_grade(Xt_, panel)
        d_recv = float(np.mean(pl != yt_p)) if len(yt_p) else 0.0
        disputed = (pl != yt_p) & (pconf >= TAU0)
        n_rep = int(disputed.sum())
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
        # ---- 5. training ----------------------------------------------------
        train_epochs(state, mom, rng, X_train, yt_train, SELF_EPOCHS)
        # ---- 6. calibration write channels (tau, kappa, iota) ---------------
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
        # ---- 6b. the DIAL's write-backs (comp/rereg) ------------------------
        iota_hist.append(iota_sys)
        if fix == "comp" and r == COMP_REG_END:
            # register the drift slope on the clean prefix (pre-attack)
            ys = np.array(iota_hist[:COMP_REG_END])
            xs = np.arange(1, COMP_REG_END + 1)
            delta_hat = float(np.polyfit(xs, ys, 1)[0])
        if fix == "rereg" and r % REREG_K == 0:
            if abs(iota_sys - iota_star) <= eps_i:
                iota_star = float(iota_sys)
                tele["rereg_fired"] = True
        # ---- 7. telemetry -----------------------------------------------------
        pbar_b, ybar_b, conf_b, dis_b, flag_b, Ps_b = flags_and_probs(
            Xte, state, kappa_sys)
        tele.update({
            "round": r, "tau_sys": float(tau_sys), "tau_star": tau_star,
            "kappa_sys": float(kappa_sys), "kappa_star": kappa_star,
            "q_target": q, "kappa_target": kap_t, "lam": lam,
            "repair_fired": repair_t, "repair_fired_kappa": repair_k,
            "r_star": r_star, "f_star": f_star,
            "iota_sys": float(iota_sys), "iota_star": float(iota_star),
            "d_recv": d_recv, "armed": armed, "reading_oob": reading_oob,
            "eps_i": eps_i, "fix": fix,
            "delta_hat": delta_hat,
            "n_disputed_repaired": n_rep,
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
        globals()["COMP_REG_END"] = 2
        for a, c in DIALCFG.items():
            c["rounds"] = 6
        globals()["ARMS"] = ["ID_TWIN_BASE", "ID_TWIN_COMP",
                             "ID_TWIN_REREG", "ID_P05_BASE", "ID_P05_WIDE"]
        globals()["DIALCFG"] = {a: DIALCFG[a] for a in ARMS}
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} the iota dial: "
          f"{len(ARMS)} arms, pert at round {PERT_ROUND}")

    res = {"config": {"tau0": TAU0, "kappa0": KAPPA0, "eps_t": EPS_T,
                      "rho_t": RHO_T, "eps_k": EPS_K, "rho_k": RHO_K,
                      "eta": ETA, "eta_k": ETA_K, "eta_i": ETA_I,
                      "eps_i_base": 0.03, "eps_i_wide": 0.05,
                      "rereg_k": REREG_K, "comp_reg_end": COMP_REG_END,
                      "brake_th": BRAKE_TH, "pert_round": PERT_ROUND,
                      "nseed": len(SEEDS),
                      "arms": {a: DIALCFG[a] for a in ARMS},
                      "numpy": np.__version__},
           "arms": {}}

    for arm in ARMS:
        rows = []
        for s in SEEDS:
            traj = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj})
            print(f"  {arm} seed={s} done ({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- aggregates ----------------------------------------------------------
    numkeys = ["acc", "accept09", "n_commit", "flagged_frac",
               "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
               "q_target", "kappa_target", "accept_own_tau", "cerr_self",
               "cerr_stream_pre", "cerr_stream_post", "panel_acc",
               "iota_sys", "iota_star", "d_recv", "n_disputed_repaired",
               "n_flipped", "delta_hat"]
    boolkeys = ["repair_fired", "repair_fired_kappa", "brake", "armed",
                "reading_oob", "rereg_fired"]
    for arm in ARMS:
        rows = res["arms"][arm]["runs"]
        agg = {}
        for key in numkeys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in boolkeys:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        res["arms"][arm]["aggregate"] = agg
        ag = res["arms"][arm]["aggregate"]
        print(f"{arm:14s} tau: {['%.3f' % a for a in ag['tau_sys']]}")
        print(f"{'':14s} iota: {['%.3f' % a for a in ag['iota_sys']]}")
        print(f"{'':14s} acc: {['%.3f' % a for a in ag['acc']]}")

    # ---- the per-seed false-positive horizons --------------------------------
    horizons = {}
    for arm in ARMS:
        per_seed = []
        for row in res["arms"][arm]["runs"]:
            acts = [t["round"] for t in row["traj"] if t["armed"]]
            per_seed.append({
                "seed": row["seed"],
                "first_armed": min(acts) if acts else None,
                "armed_rounds": len(acts),
                "rewrites": int(sum(t["n_disputed_repaired"]
                                    for t in row["traj"]))})
        horizons[arm] = per_seed
    res["horizons"] = horizons

    # ---- the bitwise audits vs doc 36 (the BASE arms only) -------------------
    audits = {}
    if not smoke:
        def audit(name, ref_arm, got_arm, key, n=14):
            try:
                d = json.load(open(DOC36))
                ref = d["arms"][ref_arm]["aggregate"][key]
                got = res["arms"][got_arm]["aggregate"][key]
                m = min(n, len(ref), len(got))
                ok = all(abs(a - b) < 1e-9
                         for a, b in zip(ref[:m], got[:m]))
                audits[name] = bool(ok)
                print(f"audit {name}: {ok}")
            except (FileNotFoundError, KeyError) as e:
                audits[name] = f"unavailable: {e}"
                print(f"audit {name}: unavailable ({e})")
        audit("ID_TWIN_BASE_vs_doc36_IC_TWIN_tau", "IC_TWIN",
              "ID_TWIN_BASE", "tau_sys")
        audit("ID_TWIN_BASE_vs_doc36_IC_TWIN_kappa", "IC_TWIN",
              "ID_TWIN_BASE", "kappa_sys")
        audit("ID_P05_BASE_vs_doc36_IC_P05_tau", "IC_P05", "ID_P05_BASE",
              "tau_sys")
        audit("ID_P05_BASE_vs_doc36_IC_P05_kappa", "IC_P05", "ID_P05_BASE",
              "kappa_sys")
        audit("ID_P30_BASE_vs_doc36_IC_P30_tau", "IC_P30", "ID_P30_BASE",
              "tau_sys")
        audit("ID_P30_BASE_vs_doc36_IC_P30_kappa", "IC_P30", "ID_P30_BASE",
              "kappa_sys")
    res["audits"] = audits

    # ---- the dial's classification (pre-registered ID rules) -----------------
    if not smoke:
        cls = {"_references": {}}
        tw = res["arms"]["ID_TWIN_BASE"]["aggregate"]
        # ID1: the baseline horizon + the drift rate
        drift = float(np.polyfit(np.arange(6, 31), tw["iota_sys"][5:30],
                                 1)[0])
        firsts = [h["first_armed"] for h in horizons["ID_TWIN_BASE"]]
        cls["_references"]["drift_rate_per_round"] = drift
        cls["_references"]["horizon_law_eps03"] = 0.03 / max(drift, 1e-9)
        cls["_references"]["horizon_law_eps05"] = 0.05 / max(drift, 1e-9)
        cls["ID1_base_false_positive"] = {
            "all_seeds_armed_within_30": all(f is not None for f in firsts),
            "first_armed_per_seed": firsts,
            "law_prediction_r13_17": all(
                f is not None and 13 <= f <= 17 for f in firsts)}
        # ID2: WIDE's horizon
        firsts_w = [h["first_armed"] for h in horizons["ID_TWIN_WIDE"]]
        cls["ID2_wide_horizon"] = {
            "first_armed_per_seed": firsts_w,
            "law_prediction_r23_27": all(
                f is None or 23 <= f <= 27 for f in firsts_w),
            "censored_seeds": sum(1 for f in firsts_w if f is None)}
        # ID3: THE TRADE at 0.05
        cls["ID3_the_trade"] = {
            "base_first_armed": [h["first_armed"]
                                 for h in horizons["ID_P05_BASE"]],
            "wide_first_armed": [h["first_armed"]
                                 for h in horizons["ID_P05_WIDE"]],
            "wide_acc_final": res["arms"]["ID_P05_WIDE"]["aggregate"]
            ["acc"][-1],
            "base_acc_final": res["arms"]["ID_P05_BASE"]["aggregate"]
            ["acc"][-1],
            "the_loop_returned": bool(res["arms"]["ID_P05_WIDE"]
                                      ["aggregate"]["acc"][-1] < 0.85)}
        # ID4: COMP
        cls["ID4_comp"] = {
            "twin_false_positives": sum(
                1 for h in horizons["ID_TWIN_COMP"]
                if h["first_armed"] is not None),
            "p05_first_armed": [h["first_armed"]
                                for h in horizons["ID_P05_COMP"]],
            "p30_first_armed": [h["first_armed"]
                                for h in horizons["ID_P30_COMP"]],
            "p30_acc_final": res["arms"]["ID_P30_COMP"]["aggregate"]
            ["acc"][-1],
            "twin_margin_final": (0.03 - abs(
                res["arms"]["ID_TWIN_COMP"]["aggregate"]["iota_sys"][-1]
                - (res["arms"]["ID_TWIN_COMP"]["aggregate"]["iota_star"][-1]
                   + res["arms"]["ID_TWIN_COMP"]["aggregate"]
                   ["delta_hat"][-1] * 25)))}
        # ID5: REREG
        rereg_moves = []
        for row in res["arms"]["ID_TWIN_REREG"]["runs"]:
            stars = [t["iota_star"] for t in row["traj"]]
            rereg_moves.append(stars[-1] - stars[0])
        cls["ID5_rereg"] = {
            "twin_false_positives": sum(
                1 for h in horizons["ID_TWIN_REREG"]
                if h["first_armed"] is not None),
            "record_drift_total_per_seed": rereg_moves,
            "p05_first_armed": [h["first_armed"]
                                for h in horizons["ID_P05_REREG"]],
            "p05_acc_final": res["arms"]["ID_P05_REREG"]["aggregate"]
            ["acc"][-1]}
        # ID6: the dose arms' cleaning and rulers
        for arm in ["ID_P05_BASE", "ID_P05_WIDE", "ID_P05_COMP",
                    "ID_P05_REREG", "ID_P30_BASE", "ID_P30_COMP"]:
            ag = res["arms"][arm]["aggregate"]
            post = [x for x in ag["cerr_stream_post"]
                    [PERT_ROUND - 1:] if x is not None]
            pre = [x for x in ag["cerr_stream_pre"]
                   [PERT_ROUND - 1:] if x is not None]
            cls.setdefault("ID6_dose_arms", {})[arm] = {
                "acc_final": ag["acc"][-1],
                "q_bar": float(np.mean([x for x in ag["q_target"]
                                        [PERT_ROUND - 1:]
                                        if x is not None])),
                "cleaning_ratio": (float(np.mean(post) / np.mean(pre))
                                   if pre and np.mean(pre) > 0 else None)}
        res["classification"] = cls
        print("\n--- the dial ---")
        for k, e in cls.items():
            if k.startswith("_"):
                print(f"{k}: {e}")
                continue
            print(f"{k}:")
            for kk, vv in e.items():
                if not isinstance(vv, dict):
                    print(f"   {kk}: {vv}")

    res["runtime_s"] = time.time() - t0
    out = ("/home/z/my-project/scripts/m3_iota_dial_results_smoke.json"
           if smoke else
           "/home/z/my-project/scripts/m3_iota_dial_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\ndone in {res['runtime_s']:.0f}s -> {out.split('/')[-1]}")


if __name__ == "__main__":
    main()
