#!/usr/bin/env python3
"""
THE EXCLUSION-FILTER FARMING CAMPAIGN - executing doc 22's disclosed threat
against M2-gamma. Corpus doc 25's experiment. Ordered at the M2-gamma
issuance: "exclusion-filter farming (the disclosed threat to Gamma)".

THE THREAT, ON THE RECORD (doc 22, Part 4 + Threats): "the mechanism of the
win is a conservative panel excluding a quarter of the training throughput...
a purely supervised build with a comparable exclusion filter would plausibly
land near Gamma's point without any self-phase at all" - and "the frontier
report does not include an exclusion-filter farming family." This campaign
supplies that family set. Target: Gamma (v=0.436%, delta=0.715%), with
beta (0.619/0.855), alpha (0.523/0.904), r3 (0.292/0.596) as secondary
references.

THE DECOMPOSITION (each family removes one more ingredient of Gamma):
  X1-indpanel    Gamma's chassis EXACTLY, but the frozen panel is an
                 INDEPENDENTLY-trained supervised reference (one stream,
                 seed 5000, snapshots at ep 20/25/30, shared by all family
                 members) rather than the member's own burn-in past.
                 Tests: does "the system's own past" contribute anything
                 measurable beyond "a frozen conservative grader"?
  X3-norelease   Gamma's chassis with NO release channel at all: quarantined
                 items simply age out (drop after 2 rounds). Tests: is the
                 panel's 1.1% release decision load-bearing? X3 shares
                 Gamma's seed set and RNG stream, so X3-vs-Gamma differs
                 only through the ~35 items Gamma released - the release
                 channel's path amplification, measured directly.
  X4-poolfilter  The closest PURELY SUPERVISED replica of Gamma's data
                 process: 30ep burn-in on full Xtr (true labels, as
                 Gamma's burn-in), then 10 rounds x 4ep on fresh perturbed
                 pools minus reference-flagged items (flag = reference
                 disagreement | conf < kappa), TRUE labels on what
                 remains. No self-phase, no committee evolution, no
                 quarantine loop - the deflationary reading executed.
  X2-conf{15,25,35}  Plain 70ep supervised pipeline (r3 chassis) on Xtr
                 minus the q% lowest reference-confidence items - the
                 static exclusion dose sweep (denominator: the train set,
                 disclosed; Gamma's 25% is pool-throughput-denominated).
  X2m-margin25   Exclusion by smallest top1-top2 margin at q=25 (the
                 ambiguity axis, not the confidence axis).
  X2o-oracle     Exclude the items the reference committee misclassifies
                 (endogenous rate) - the confident-learning / label-noise
                 cleaning attack, the farmer's sharpest data tool.

All families: matched K=4 committee architecture, purely supervised
designer-side machinery only, homogeneous registered norms (amendment 4:
every member registers tau=0.9, kappa=0.95 where applicable), disposition =
plain committee at tau=0.9, battery = the 4,500-claim held-out set.
Reference committee for all filters: seed 5000, 30ep, never a family member.

PRE-REGISTERED EXPECTATIONS (fixed before execution):
  X1 (replica): |v_X1 - v_Gamma| <= 0.08pp AND |delta_X1 - delta_Gamma| <=
      0.15pp - the own-past property is inert; Gamma's point replicates
      with any frozen conservative grader. Alternative: X1 drifts toward
      beta - the shared burn-in stream is load-bearing.
  X2 (static dose): v_X2(q) < v_Gamma for every q (predicted band
      0.25-0.40) - static supervised exclusion NARROWS (the ambiguous
      items whose seed-dependent memorization produces width are exactly
      what exclusion removes); the static arc does NOT pass through
      Gamma's point. Alternative: some q lands inside Gamma's tolerance
      box - the disclosed threat confirmed at that dose.
  X3 (no release): |v_X3 - v_Gamma| <= 0.05pp - the 1.1% release channel
      is inert; Gamma ~= pure exclusion inside its chassis. Plus the path
      amplification measurement: mean per-seed V-level disagreement
      X3-vs-Gamma ridden by ~35 released items.
  X4 (process replica): X4 lands BETWEEN r3 and Gamma on width (0.30-0.45)
      at delta <= 0.75 - the perturbed-pool exclusion process reproduces
      part of Gamma's position without any self-phase. Alternative: X4 ~=
      r3 (the pool structure is inert without the evolving committee).
  X5 (pooled frontier): no exclusion family dominates Gamma (no
      v >= 0.436 at delta <= 0.715 among them - predicted, because
      exclusion narrows); Gamma's residual content = the COMPOSITE
      position (self-phase width + exclusion compliance), which neither
      the static-dose arc nor (per doc 21) the F9 heterogeneity arc
      reaches at matched delta. F9s-0.3's outright dominance of Gamma
      (doc 22's standing concession) is expected to stand.

Usage: python3 exclusion_farming.py [--smoke] [--reset]
  Checkpointed after each family (foreground chunks, exact resume).
"""
import itertools
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration ----------------
K = 4
TAU, KAPPA = 0.9, 0.95
BURN_EPOCHS, ROUNDS, SELF_EPOCHS = 30, 10, 4
PANEL_EPOCHS = [20, 25, 30]
QUAR_MAX_AGE = 2
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
G_SEEDS = [0] + list(range(100, 112))     # gamma-chassis families (13 runs)
F_BASE = 4200                              # farming-family seed bases
REF_SEED = 5000                            # the independent reference stream
DOSE_GRID = [15, 25, 35]
X4_RATE = 0.25                             # X4 rate-matched exclusion dose
X4C_ACCEPT = 0.70                          # X4c rate-matched gate (Gamma's
                                            # mean commit rate 824-1078/1347)

CKPT_JSON = "/home/z/my-project/scripts/exclusion_results.json"
CKPT_NPZ = "/home/z/my-project/scripts/tmp/exclusion_V.npz"
FARM_JSON = "/home/z/my-project/scripts/farming_results.json"
F9_JSON = "/home/z/my-project/scripts/f9_results.json"
M2G_JSON = "/home/z/my-project/scripts/m2gamma_results.json"
M2G_NPZ = "/home/z/my-project/scripts/tmp/m2gamma_V.npz"


# ---------------- model (identical to m2_build.py / m2gamma_build.py) -------
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


def flags_and_probs(X, state):
    W1, b1, heads = state
    _, zs = forward_all(X, W1, b1, heads)
    Ps = [softmax(z) for z in zs]
    pbar = np.mean(Ps, axis=0)
    ybar = pbar.argmax(1)
    conf = pbar.max(1)
    disagree = np.zeros(len(X), dtype=bool)
    for P in Ps:
        disagree |= (P.argmax(1) != ybar)
    flagged = disagree | (conf < KAPPA)
    return pbar, ybar, conf, flagged


# ---------------- audit-side ----------------
def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def fam_stats(V_list, truth):
    n = len(V_list)
    pw = [float(np.mean(V_list[a] != V_list[b]))
          for a, b in itertools.combinations(range(n), 2)]
    stack = np.stack(V_list)
    unan = np.all(stack, 0) | ~np.any(stack, 0)
    deltas = [float(np.mean(V != truth)) for V in V_list]
    return {"v_mean": float(np.mean(pw)), "v_sd": float(np.std(pw)),
            "v_min": float(np.min(pw)), "v_max": float(np.max(pw)),
            "contested": float(np.mean(~unan)),
            "delta_mean": float(np.mean(deltas)),
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas)),
            "n_pairs": len(pw)}


# ---------------- the gamma chassis (verbatim from m2gamma_build.py) --------
def self_round_gamma(state, mom, rng, Xtr, quar, ytr, panel, release_on=True):
    """Gamma's round. release_on=False gives X3: quarantine never releases,
    items age out. The RNG stream is identical either way (release decisions
    consume no randomness)."""
    released_X, released_y, released_true = [], [], []
    kept, dropped = [], []
    audit = {"n_quar_regraded": len(quar),
             "panel_label_agree_committee": None,
             "panel_release_rate": None, "committee_release_rate_would": None}
    if quar:
        Xq = np.stack([q[0] for q in quar])
        unflag_n = np.zeros(len(quar), dtype=int)
        conf_sum = np.zeros(len(quar))
        pbar_sum = np.zeros((len(quar), 10))
        for snap in panel:
            pj, yj, cj, fj = flags_and_probs(Xq, snap)
            unflag_n += (~fj).astype(int)
            conf_sum += cj
            pbar_sum += pj
        panel_conf = conf_sum / len(panel)
        panel_pbar = pbar_sum / len(panel)
        panel_label = panel_pbar.argmax(1)
        release = (unflag_n >= 2) & (panel_conf >= TAU)
        pc, yc, cc, fc = flags_and_probs(Xq, state)
        would = (~fc) & (cc >= TAU)
        audit["panel_label_agree_committee"] = float(np.mean(panel_label == yc))
        audit["panel_release_rate"] = float(np.mean(release))
        audit["committee_release_rate_would"] = float(np.mean(would))
        for k, (Xqi, ti, age) in enumerate(quar):
            if release_on and release[k]:
                released_X.append(Xqi)
                released_y.append(int(panel_label[k]))
                released_true.append(int(ytr[ti]))
            elif age + 1 >= QUAR_MAX_AGE:
                dropped.append((Xqi, ti))
            else:
                kept.append((Xqi, ti, age + 1))
    pool = perturb(Xtr, rng)
    pbar, ybar, conf, flagged = flags_and_probs(pool, state)
    commit = (~flagged) & (conf >= TAU)
    labels = ybar.copy()
    for i in np.where(flagged)[0]:
        kept.append((pool[i], i, 1))
    if released_X:
        Xtr_ = np.concatenate([pool[commit], np.stack(released_X)])
        ytr_ = np.concatenate([labels[commit], np.array(released_y)])
    else:
        Xtr_, ytr_ = pool[commit], labels[commit]
    tele = {"n_commit": int(commit.sum()), "n_released": len(released_X),
            "n_dropped": len(dropped), "n_quar": len(kept), **audit}
    return Xtr_, ytr_, kept, dropped, tele


def burn_in(rng, Xtr, ytr):
    state = init_params(rng)
    mom = zero_moments(state)
    train_epochs(state, mom, rng, Xtr, ytr, BURN_EPOCHS)
    return state, mom


def run_gamma_chassis(seed, Xtr, ytr, Xte, yte, panel, release_on=True):
    """X1/X3 runner: gamma chassis, panel supplied externally.
    RNG stream identical to M2G for the same seed (burn-in + pools)."""
    rng = np.random.default_rng(seed)
    state, mom = burn_in(rng, Xtr, ytr)
    quar, dropped_all, teles = [], [], []
    for r in range(ROUNDS):
        Xtr_, ytr_, quar, dropped, tele = self_round_gamma(
            state, mom, rng, Xtr, quar, ytr, panel, release_on=release_on)
        dropped_all.extend(dropped)
        teles.append(tele)
        train_epochs(state, mom, rng, Xtr_, ytr_, SELF_EPOCHS)
    P = committee_probs(Xte, *state)
    V = (P >= TAU).reshape(-1)
    acc = float(np.mean(P.argmax(1) == yte))
    dose = len(dropped_all) / (ROUNDS * len(Xtr))
    return V, acc, dose, teles, len(dropped_all)


# ---------------- the supervised families -----------------------------------
def run_static_filter(seeds, Xtr, ytr, Xte, yte, truth, keep_mask, tag):
    """X2/X2m/X2o: plain 70ep supervised pipeline on the filtered train set.
    keep_mask is designer-fixed (shared across all members - the filter is
    the designer's, not the run's)."""
    V_list, accs = [], []
    t0 = time.time()
    Xf, yf = Xtr[keep_mask], ytr[keep_mask]
    for s in seeds:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        train_epochs(state, mom, rng, Xf, yf, 70)
        P = committee_probs(Xte, *state)
        V_list.append((P >= TAU).reshape(-1))
        accs.append(float(np.mean(P.argmax(1) == yte)))
    stats = fam_stats(V_list, truth)
    stats["acc_mean"] = float(np.mean(accs))
    stats["n_kept"] = int(keep_mask.sum())
    stats["dose_static"] = float(1.0 - keep_mask.mean())
    stats["cfg"] = tag
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats, np.stack(V_list)


def run_poolfilter(seeds, Xtr, ytr, Xte, yte, truth, ref_state):
    """X4: supervised replica of gamma's data process. 30ep burn-in on full
    Xtr (true labels); then 10 rounds x 4ep: pool = perturb(Xtr), drop the
    RATE-MATCHED lowest-reference-confidence 25% of the pool (gamma's dose;
    the naive flag rule fired on 0% in the smoke pass - the memorized
    reference is conf>=0.95 on perturbed copies and its heads never
    disagree - disclosed), train on the rest with TRUE labels."""
    V_list, accs, doses = [], [], []
    t0 = time.time()
    for s in seeds:
        rng = np.random.default_rng(s)
        state, mom = burn_in(rng, Xtr, ytr)
        n_flag_tot = 0
        for r in range(ROUNDS):
            pool = perturb(Xtr, rng)
            _, _, conf_ref, _ = flags_and_probs(pool, ref_state)
            cut = np.quantile(conf_ref, X4_RATE)
            flagged_ref = conf_ref < cut
            n_flag_tot += int(flagged_ref.sum())
            train_epochs(state, mom, rng, pool[~flagged_ref],
                         ytr[~flagged_ref], SELF_EPOCHS)
        P = committee_probs(Xte, *state)
        V_list.append((P >= TAU).reshape(-1))
        accs.append(float(np.mean(P.argmax(1) == yte)))
        doses.append(n_flag_tot / (ROUNDS * len(Xtr)))
    stats = fam_stats(V_list, truth)
    stats["acc_mean"] = float(np.mean(accs))
    stats["dose_pool_mean"] = float(np.mean(doses))
    stats["cfg"] = "X4-poolfilter (rate-matched 25% by reference conf)"
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats, np.stack(V_list)


def run_gate_replica(seeds, Xtr, ytr, Xte, yte, truth, ref_state):
    """X4c: the MAXIMAL supervised replica of gamma's process structure -
    rate-matched gate + true labels, everything frozen. Each round: pool =
    perturb(Xtr); keep the top-70% by reference confidence (rate-matched
    to gamma's mean commit rate); train 4ep with TRUE labels. Removes the
    last ingredient: the evolving committee itself. If X4c ~= Gamma, every
    ingredient of the third generation is replicable by frozen machinery;
    if not, the evolution's selection content is the residue."""
    V_list, accs, doses = [], [], []
    t0 = time.time()
    for s in seeds:
        rng = np.random.default_rng(s)
        state, mom = burn_in(rng, Xtr, ytr)
        n_keep_tot = 0
        for r in range(ROUNDS):
            pool = perturb(Xtr, rng)
            _, _, conf_ref, _ = flags_and_probs(pool, ref_state)
            cut = np.quantile(conf_ref, 1.0 - X4C_ACCEPT)
            keep = conf_ref >= cut
            n_keep_tot += int(keep.sum())
            train_epochs(state, mom, rng, pool[keep], ytr[keep], SELF_EPOCHS)
        P = committee_probs(Xte, *state)
        V_list.append((P >= TAU).reshape(-1))
        accs.append(float(np.mean(P.argmax(1) == yte)))
        doses.append(n_keep_tot / (ROUNDS * len(Xtr)))
    stats = fam_stats(V_list, truth)
    stats["acc_mean"] = float(np.mean(accs))
    stats["dose_pool_mean"] = float(1.0 - np.mean(doses))
    stats["cfg"] = "X4c: frozen rate-matched gate + true labels (max replica)"
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats, np.stack(V_list)


# ---------------- main -------------------------------------------------------
def main():
    smoke = "--smoke" in sys.argv
    reset = "--reset" in sys.argv
    if smoke:
        globals()["G_SEEDS"] = [0, 100]
        globals()["DOSE_GRID"] = [25]
        globals()["F_BASE"] = 4300
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    print(f"data: train={len(Xtr)} test={len(Xte)} claims={len(truth)}")

    # ---- the independent reference stream (the designer's panel/filter) ----
    rng_ref = np.random.default_rng(REF_SEED)
    ref_state = init_params(rng_ref)
    ref_mom = zero_moments(ref_state)
    ref_panel = []
    done = 0
    for ep in PANEL_EPOCHS:
        train_epochs(ref_state, ref_mom, rng_ref, Xtr, ytr, ep - done)
        ref_panel.append(copy_state(ref_state))
        done = ep
    # reference conf / margin / error on the train set (the filter signals)
    Pref = committee_probs(Xtr, *ref_state)
    ref_conf = Pref.max(1)
    srt = np.sort(Pref, axis=1)
    ref_margin = srt[:, -1] - srt[:, -2]
    ref_err = Pref.argmax(1) != ytr
    print(f"reference stream: seed={REF_SEED} train-acc="
          f"{np.mean(Pref.argmax(1) == ytr)*100:.2f}% "
          f"median-conf={np.median(ref_conf):.3f}")

    # 5-fold CV reference (out-of-fold): the honest oracle filter. The
    # in-sample reference has memorized Xtr (train-acc 100% at 30ep), so
    # in-sample error is EMPTY - the naive oracle is vacuous, disclosed;
    # the farmer must go out-of-fold (confident learning, exactly as the
    # data-cleaning literature does).
    oof_err = np.zeros(len(Xtr), dtype=bool)
    oof_conf = np.zeros(len(Xtr))
    fold_edges = np.linspace(0, len(Xtr), 6).astype(int)
    for fi in range(5):
        lo, hi = fold_edges[fi], fold_edges[fi + 1]
        val_idx = np.arange(lo, hi)
        tr_idx = np.concatenate([np.arange(0, lo), np.arange(hi, len(Xtr))])
        rng_cv = np.random.default_rng(REF_SEED + 100 + fi)
        cv_state = init_params(rng_cv)
        cv_mom = zero_moments(cv_state)
        train_epochs(cv_state, cv_mom, rng_cv, Xtr[tr_idx], ytr[tr_idx], 30)
        Pv = committee_probs(Xtr[val_idx], *cv_state)
        oof_err[val_idx] = Pv.argmax(1) != ytr[val_idx]
        oof_conf[val_idx] = Pv.max(1)
    print(f"CV reference: out-of-fold err={oof_err.mean()*100:.2f}% "
          f"(the X2o exclusion rate, endogenous)")

    # ---- resume ----
    results = None
    Vstore = {}
    if not reset:
        try:
            results = json.load(open(CKPT_JSON))
            Vstore = {k: v for k, v in np.load(CKPT_NPZ).items()}
            print(f"resuming: {len(results.get('families', {}))} families done")
        except (FileNotFoundError, Exception):
            results = None; Vstore = {}
    if results is None:
        results = {}
    results["config"] = {
        "K": K, "tau": TAU, "kappa": KAPPA, "rounds": ROUNDS,
        "burn": BURN_EPOCHS, "self_epochs": SELF_EPOCHS,
        "quar_max_age": QUAR_MAX_AGE, "panel_epochs": PANEL_EPOCHS,
        "ref_seed": REF_SEED, "g_seeds": len(G_SEEDS),
        "dose_grid": DOSE_GRID, "numpy": np.__version__}
    results["targets"] = {
        "m2g": {"v": 0.00436, "delta": 0.00715},
        "m2b": {"v": 0.00619, "delta": 0.00855},
        "m2a": {"v": 0.00523, "delta": 0.00904},
        "r3": {"v": 0.00292, "delta": 0.00596}}
    results.setdefault("families", {})

    def family_done(name):
        return name in results["families"] and name in Vstore

    def record(name, stats, Vmat):
        results["families"][name] = stats
        Vstore[name] = Vmat
        with open(CKPT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        np.savez_compressed(CKPT_NPZ, **Vstore)
        print(f"{name:16s} v={stats['v_mean']*100:6.3f}% "
              f"delta={stats['delta_mean']*100:6.3f}% acc={stats['acc_mean']*100:5.2f}% "
              f"dose={stats.get('dose_pool', stats.get('dose_pool_mean', stats.get('dose_static', 0)))*100:4.1f}% "
              f"[{stats.get('elapsed_s', 0):.0f}s]", flush=True)

    # ---- X1: independent panel in the gamma chassis ----
    if not family_done("X1-indpanel"):
        Vs, accs, doses, teles, drops = [], [], [], [], []
        for s in G_SEEDS:
            V, acc, dose, tele, nd = run_gamma_chassis(
                s, Xtr, ytr, Xte, yte, ref_panel, release_on=True)
            Vs.append(V); accs.append(acc); doses.append(dose); drops.append(nd)
            teles.append(tele)
        stats = fam_stats(Vs, truth)
        stats["acc_mean"] = float(np.mean(accs))
        stats["dose_pool"] = float(np.mean(doses))
        stats["dropped_total"] = int(np.mean(drops))
        agg = {k: float(np.mean([t.get(k) for t in rt if t.get(k) is not None]))
               for rt in teles for k in ["panel_release_rate",
               "committee_release_rate_would", "panel_label_agree_committee"]}
        stats["panel_audit_means"] = agg
        stats["cfg"] = "X1: gamma chassis, independent reference panel"
        stats["elapsed_s"] = round(time.time() - t0, 1)
        record("X1-indpanel", stats, np.stack(Vs))

    # ---- X3: no release channel ----
    if not family_done("X3-norelease"):
        Vs, accs, doses = [], [], []
        for s in G_SEEDS:
            V, acc, dose, tele, nd = run_gamma_chassis(
                s, Xtr, ytr, Xte, yte, ref_panel, release_on=False)
            Vs.append(V); accs.append(acc); doses.append(dose)
        stats = fam_stats(Vs, truth)
        stats["acc_mean"] = float(np.mean(accs))
        stats["dose_pool"] = float(np.mean(doses))
        stats["cfg"] = "X3: gamma chassis, quarantine never releases"
        stats["elapsed_s"] = round(time.time() - t0, 1)
        record("X3-norelease", stats, np.stack(Vs))

    # ---- X4: supervised process replica (exclusion only) ----
    if not family_done("X4-poolfilter"):
        stats, Vmat = run_poolfilter(
            [F_BASE + i for i in range(12)], Xtr, ytr, Xte, yte, truth,
            ref_state)
        record("X4-poolfilter", stats, Vmat)

    # ---- X4c: maximal supervised replica (rate-matched gate) ----
    if not family_done("X4c-gatereplica"):
        stats, Vmat = run_gate_replica(
            [F_BASE + 50 + i for i in range(12)], Xtr, ytr, Xte, yte,
            truth, ref_state)
        record("X4c-gatereplica", stats, Vmat)

    # ---- X2 static dose sweep ----
    for q in DOSE_GRID:
        name = f"X2-conf{q}"
        if family_done(name):
            continue
        ncut = int(round(len(Xtr) * q / 100.0))
        order = np.argsort(ref_conf, kind="mergesort")   # ascending conf
        keep = np.ones(len(Xtr), dtype=bool)
        keep[order[:ncut]] = False                        # drop lowest-conf q%
        stats, Vmat = run_static_filter(
            [F_BASE + 100 + q + i for i in range(12)], Xtr, ytr, Xte, yte,
            truth, keep, f"X2: static conf filter, drop {q}% lowest conf")
        record(name, stats, Vmat)

    # ---- X2m: margin axis ----
    if not family_done("X2m-margin25"):
        ncut = int(round(len(Xtr) * 0.25))
        order = np.argsort(ref_margin, kind="mergesort")
        keep = np.ones(len(Xtr), dtype=bool)
        keep[order[:ncut]] = False
        stats, Vmat = run_static_filter(
            [F_BASE + 300 + i for i in range(12)], Xtr, ytr, Xte, yte,
            truth, keep, "X2m: static margin filter, drop 25% smallest margin")
        record("X2m-margin25", stats, Vmat)

    # ---- X2o: CV-oracle / confident-learning ----
    if not family_done("X2o-oracle"):
        keep = ~oof_err
        stats, Vmat = run_static_filter(
            [F_BASE + 400 + i for i in range(12)], Xtr, ytr, Xte, yte,
            truth, keep, "X2o: drop out-of-fold-misclassified (endogenous)")
        record("X2o-oracle", stats, Vmat)

    if len(results["families"]) < 9 and not smoke:
        print(f"checkpoint pass complete: {len(results['families'])}/9 - re-run")
        return

    # ---- release-channel path amplification: X3 vs Gamma, per seed ----
    try:
        m2gV = np.load(M2G_NPZ)["m2g"]
        x3V = Vstore["X3-norelease"]
        n = min(len(m2gV), len(x3V))
        amp = [float(np.mean(x3V[i] != m2gV[i])) for i in range(n)]
        results["release_channel_amplification"] = {
            "per_seed_V_disagreement_x3_vs_gamma": amp,
            "mean": float(np.mean(amp)),
            "note": ("X3 shares Gamma's seeds and RNG stream; disagreement "
                     "rides only the ~35 items Gamma's panel released")}
        print(f"X3-vs-Gamma V disagreement (release-channel path effect): "
              f"{np.mean(amp)*100:.3f}%")
    except (FileNotFoundError, KeyError) as e:
        results["release_channel_amplification"] = {"error": str(e)}

    # ---- pooled frontier vs every prior point ----
    pts, keys = [], []
    for src, path in [("farm", FARM_JSON), ("sweep", F9_JSON)]:
        try:
            d = json.load(open(path))
            for nm, st in d["families"].items():
                pts.append((st["v_mean"], st["delta_mean"]))
                keys.append(f"{src}:{nm}")
        except FileNotFoundError:
            pass
    for nm, st in results["families"].items():
        pts.append((st["v_mean"], st["delta_mean"])); keys.append(f"excl:{nm}")
    for nm, (v_, d_) in [("m2g", (0.00436, 0.00715)),
                         ("m2b", (0.00619, 0.00855)),
                         ("m2a", (0.00523, 0.00904)),
                         ("r3", (0.00292, 0.00596))]:
        pts.append((v_, d_)); keys.append(nm)

    def dominates(p, t):
        return p[0] >= t[0] and p[1] <= t[1] and (p[0] > t[0] or p[1] < t[1])

    fr = {}
    for nm, tgt in [("m2g", (0.00436, 0.00715)),
                    ("m2b", (0.00619, 0.00855)),
                    ("m2a", (0.00523, 0.00904))]:
        doms = [k for k, p in zip(keys, pts) if dominates(p, tgt)]
        excl_doms = [k for k in doms if k.startswith("excl:")]
        fr[nm] = {"dominated_by": doms, "dominated_by_exclusion_only": excl_doms}
        print(f"frontier vs {nm}: {len(doms)} dominators, "
              f"exclusion-only: {excl_doms}")
    # tolerance-box match vs Gamma: |dv| <= 0.08pp, |ddelta| <= 0.15pp
    tol = [k for k, p in zip(keys, pts)
           if abs(p[0] - 0.00436) <= 0.0008 and abs(p[1] - 0.00715) <= 0.0015]
    fr["gamma_tolerance_box"] = tol
    print(f"inside Gamma's tolerance box: {tol}")
    results["frontier_report"] = fr

    results["runtime_s"] = time.time() - t0
    with open(CKPT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(CKPT_NPZ, **Vstore)
    print(f"\ndone in {results['runtime_s']:.0f}s -> exclusion_results.json")


if __name__ == "__main__":
    main()
