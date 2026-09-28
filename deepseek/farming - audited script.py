#!/usr/bin/env python3
"""
THE SMARTER FARMING ATTACK - can width-at-compliance be farmed?
Corpus doc 17's experiment. Ordered at the M2-α issuance: "smarter farming
attacks on the 2D frontier (can width-at-compliance be farmed?)"

M2-α's Part 4 finding: C4 (loose supervised, 15 ep) matches the self-taught
family's WIDTH (v 0.489% vs 0.523%) but not its COMPLIANCE (delta 1.296% vs
0.904%) - width alone is designable; width-at-compliance looked harder. This
campaign attacks that residual with designer-side tricks ONLY - no self-phase
anywhere - under amendment 1 (frontier reporting): every family is a farming
control of the auditor's construction.

THE ATTACK FAMILIES (all: M2-α committee architecture, purely supervised,
disposition = plain committee at tau=0.9, N=12 runs, battery = the 4,500-claim
held-out set):
  F1  epoch curve: total epochs in {15,20,25,30,40,55,70} - the naive
      undertraining axis, 7 sub-families (F1-70 doubles as an r3 replication
      with fresh seeds).
  F2  bagging: 70 ep on a per-run bootstrap resample (drawn by run RNG).
  F3  augmix (incoherent): 70 ep, per-run augmentation regime drawn from
      {none, gauss .05/.10/.15/.20, mask3x3, shift1px}, fresh draws each epoch.
  F3b augmix (coherent): all 12 runs train under the SAME regime (gauss .15)
      - the coherent-shift injection: attacks the 3rd coordinate too.
  F4a label smoothing eps=0.10 - kept from the scouting run: it COLLAPSES
      the accept region (max-prob capped near 1-eps+eps/10 < tau-interaction)
      and is reported as a finding, not just a failure.
  F4b label smoothing eps=0.02 (calibrated).
  F5  random-epoch stopping: per-run e ~ U[20,70], a disclosed randomized
      designer stopping rule.
  F6  output-bias jitter (THE DIRECT ATTACK): 70 ep, then a per-run Gaussian
      jitter vector added coherently to every head's output bias;
      sigma_b in {0.05, 0.10, 0.20} (calibrated; scouting showed <=0.02
      inert) - width is now a DIAL.
  F7  per-head subbagging: each head sees a random 80% of each batch
      (per-batch per-head masks, run RNG) - designer-installed head diversity.
  F8  THE ADAPTIVE COMPOSITE (added after the scouting run, disclosed as
      the farmer's second move): coherent-g15 augmentation + output-bias
      jitter sigma_b in {0.10, 0.20}. The scouting pass IS the farmer's
      iterative optimization - single tricks did not dominate, so the
      adversary combines the compliance/shift injector with the width dial.
  F9  PER-RUN THRESHOLD JITTER (the farmer's third move, added after the
      audited run showed F6 inert - second scouting pass, disclosed): each
      family member's disposition uses its OWN tau_i ~ N(0.9, sigma_tau),
      sigma_tau in {0.02, 0.05, 0.10}. The most direct width dial of all:
      install norm HETEROGENEITY across family members. Prediction (H6,
      pre-registered before this extension ran): sigma_tau=0.05 approaches
      or dominates the M2-plain point; the farm is VISIBLE in the
      registration (heterogeneous tau_i) but invisible to the Tier-0
      statistic - the frontier must therefore be read at registration
      level, not statistic level.

TARGETS (from m2_results.json, third-party numbers): M2-alpha plain
(v 0.505%, delta 0.809%) and registered (v 0.523%, delta 0.904%); r3
(0.292%, 0.596%); C4 (0.489%, 1.296%).

CALIBRATION DISCLOSURE (smoke pass, N=3, before the config was frozen):
  - smoothing eps=0.10 collapses the accept region (delta 4.5%) -> kept as
    F4a finding; the attack version is F4b eps=0.02.
  - jitter sigma_b<=0.02 is inert (v 0.281% ~ r3) -> frozen sweep {0.05,.1,.2}.
  - no single trick dominated either M2 target at N=3 -> F8 composite added
    as the adaptive farmer's second move. All reported numbers below come
    from the single audited execution at N=12.

PRE-REGISTERED EXPECTATIONS (fixed before the audited execution):
  H1: the F1 epoch curve is monotone (v down, delta down together); no epoch
      point reaches (v>=0.50%, delta<=0.91%) - the naive axis stays dominated.
  H2: F6 alone does NOT cleanly dominate (the width/compliance exchange rate
      of pure jitter is ~2:1 with the M2 points); F8 (composite) DOES
      dominate both M2 targets - width-at-compliance is farmable by an
      adaptive, statistic-aware designer.
  H3: F3b reproduces the coherent-shift coordinate (shift ~0.3% vs r3 at
      BETTER compliance than r3) - the 3rd signature coordinate is farmable.
  H4: F2/F4b/F5/F7 land mid-frontier; F5 tracks the epoch-curve interpolation.
  H5 (instrument-level, if H2/H3 hold): every Tier-0 scalar is targetable by
      a statistic-aware designer; amendment 1 must strengthen from C4-class
      controls to adversarial farming families; Tier-0 weight for
      self-generation approaches zero - the farming attack is the
      audit-completeness conjecture's empirical shadow at the instrument.

Usage: python3 farming_attack.py [--smoke] [--reset]
  Checkpointed: after each family, results + verdict matrices are saved;
  re-running skips completed families (needed because the sandbox kills
  long background jobs - the audit runs as sequential foreground chunks,
  each resuming from the last checkpoint; determinism makes this exact).
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
TAU = 0.9
BATCH, LR = 32, 1e-3
SPLIT_SEED = 12345
NFAM = 12
EPOCH_GRID = [15, 20, 25, 30, 40, 55, 70]
SIGMA_B = [0.05, 0.10, 0.20]
SMOOTH_EPS = 0.10
SUBBAG_FRAC = 0.80
COHERENT_REGIME = 2          # gauss sigma=0.15 in the regime grid below
REGIMES = ["none", "g05", "g10", "g15", "g20", "mask", "shift"]
R3_SEEDS = list(range(200, 212))   # r3 replication (fresh, in-script)

FAMILIES = []  # (name, seed_base, config dict) - built in main


# ---------------- model (identical to m2_build.py) ----------------
def init_params(rng):
    W1 = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 128)); b1 = np.zeros(128)
    heads = []
    for _ in range(K):
        Wh = rng.normal(0.0, np.sqrt(2.0 / 128), (128, 64)); bh = np.zeros(64)
        Wo = rng.normal(0.0, np.sqrt(2.0 / 64), (64, 10)); bo = np.zeros(10)
        heads.append([Wh, bh, Wo, bo])
    return [W1, b1, heads]


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


def train_epochs(state, mom, rng, X, y, epochs, smooth=0.0, subbag=False,
                 perturb_fn=None):
    """Joint supervised CE over K heads. Farmer's knobs: label smoothing,
    per-head subbagging, per-epoch augmentation. All drawn by the run's RNG."""
    W1, b1, heads = state
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    n = len(X)
    for _ in range(epochs):
        Xe = perturb_fn(X, rng) if perturb_fn is not None else X
        order = rng.permutation(n)
        for i0 in range(0, n, BATCH):
            idx = order[i0:i0 + BATCH]
            Xb, yb = Xe[idx], y[idx]
            a1, zs = forward_all(Xb, W1, b1, heads)
            Ps = [softmax(z) for z in zs]
            dZs = []
            for P in Ps:
                if smooth > 0.0:
                    T = np.full_like(P, smooth / 10.0)
                    T[np.arange(len(yb)), yb] += 1.0 - smooth
                else:
                    T = np.zeros_like(P)
                    T[np.arange(len(yb)), yb] = 1.0
                dZs.append((P - T) / len(yb))
            if subbag:
                keep = [rng.random(len(yb)) < SUBBAG_FRAC for _ in range(K)]
            else:
                keep = [None] * K
            dA1 = np.zeros_like(a1)
            gheads = []
            for hi, ((Wh, bh, Wo, bo), dZ) in enumerate(zip(heads, dZs)):
                if keep[hi] is not None:
                    dZ = dZ * keep[hi][:, None] / SUBBAG_FRAC
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


# ---------------- single-regime perturbation (farmer's augmentation) --------
def perturb_regime(X, rng, regime):
    out = X.copy()
    if regime == 0:
        return out
    if regime in (1, 2, 3, 4):
        sig = [0.05, 0.10, 0.15, 0.20][regime - 1]
        out = np.clip(X + rng.normal(0, 1, X.shape) * sig, 0, 1)
    elif regime == 5:  # mask3x3, fixed for the epoch's draw
        n = len(X)
        ii = rng.integers(0, 6, n); jj = rng.integers(0, 6, n)
        for k in range(n):
            v = out[k].reshape(8, 8); v[ii[k]:ii[k] + 3, jj[k]:jj[k] + 3] = 0.0
            out[k] = v.reshape(-1)
    elif regime == 6:  # shift1px
        n = len(X)
        di = rng.integers(-1, 2, n); dj = rng.integers(-1, 2, n)
        for k in range(n):
            v = out[k].reshape(8, 8)
            out[k] = np.roll(v, (di[k], dj[k]), axis=(0, 1)).reshape(-1)
    return out


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


def run_family(name, seeds, Xtr, ytr, Xte, yte, truth, cfg):
    """Train one farming family; return (stats, verdict matrix, acc list)."""
    V_list, accs, taus = [], [], []
    t0 = time.time()
    for s in seeds:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        # --- farmer's per-run design draws (all by run RNG, all disclosed) ---
        epochs = cfg.get("epochs")
        if epochs == "random":
            epochs = int(rng.integers(20, 71))
        Xr, yr = Xtr, ytr
        if cfg.get("bag"):                      # F2 bootstrap
            idx = rng.integers(0, len(Xtr), len(Xtr))
            Xr, yr = Xtr[idx], ytr[idx]
        regime = cfg.get("regime")
        if regime == "draw":                    # F3 incoherent: per-run regime
            regime = int(rng.integers(0, len(REGIMES)))
        pfn = None
        if regime is not None and regime != 0:
            pfn = lambda X, r, rg=regime: perturb_regime(X, r, rg)
        train_epochs(state, mom, rng, Xr, yr, epochs,
                     smooth=cfg.get("smooth", 0.0),
                     subbag=cfg.get("subbag", False),
                     perturb_fn=pfn)
        if cfg.get("sigma_b"):                  # F6 coherent output-bias jitter
            jit = rng.normal(0.0, cfg["sigma_b"], 10)
            for h in state[2]:
                h[3] = h[3] + jit
        tau_run = TAU
        if cfg.get("sigma_tau"):               # F9 per-run threshold jitter
            tau_run = float(max(0.5, min(0.99, rng.normal(TAU, cfg["sigma_tau"]))))
        P = committee_probs(Xte, *state)
        V_list.append((P >= tau_run).reshape(-1))
        taus.append(tau_run)
        accs.append(float(np.mean(P.argmax(1) == yte)))
    stats = fam_stats(V_list, truth)
    stats["acc_mean"] = float(np.mean(accs))
    stats["seeds"] = list(seeds)
    stats["cfg"] = {k: v for k, v in cfg.items()}
    stats["tau_jitter_actual"] = [round(t, 4) for t in taus]
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats, np.stack(V_list), accs


def pareto_front(points):
    """Indices of Pareto-minimal points (minimize v and delta)."""
    keep = []
    for i, (vi, di) in enumerate(points):
        dominated = any(
            (vj <= vi and dj <= di) and (vj < vi or dj < di)
            for j, (vj, dj) in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def main():
    smoke = "--smoke" in sys.argv
    reset = "--reset" in sys.argv
    if smoke:
        globals()["NFAM"] = 3
        globals()["EPOCH_GRID"] = [15, 40, 70]
        globals()["SIGMA_B"] = [0.01]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    print(f"data: train={len(Xtr)} test={len(Xte)} claims={len(truth)}")

    # ---- resume from checkpoint if present ----
    ckpt_json = "/home/z/my-project/scripts/farming_results.json"
    ckpt_npz = "/home/z/my-project/scripts/tmp/farming_V.npz"
    results = None
    Vstore = {}
    if not reset:
        try:
            results = json.load(open(ckpt_json))
            Vstore = {k: v for k, v in np.load(ckpt_npz).items()}
            done = set(results.get("families", {}).keys())
            if "r3" in Vstore:
                done.add("__r3__")
            print(f"resuming: {len(done)} families already done")
        except (FileNotFoundError, KeyError, Exception):
            results = None; Vstore = {}
    if results is None:
        results = {}

    if "__r3__" not in (set(results.get("families", {}).keys()) | {"x"}) \
            and "r3" not in Vstore:
        # ---- r3 replication (fresh in-script reference) ----
        r3_V = []
        for s in R3_SEEDS:
            rng = np.random.default_rng(s)
            state = init_params(rng)
            mom = zero_moments(state)
            train_epochs(state, mom, rng, Xtr, ytr, 70)
            P = committee_probs(Xte, *state)
            r3_V.append((P >= TAU).reshape(-1))
        r3_stats = fam_stats(r3_V, truth)
        E_r3 = np.stack(r3_V).sum(0) >= (len(r3_V) // 2 + 1)
        print(f"r3 replication: v={r3_stats['v_mean']*100:.3f}% "
              f"delta={r3_stats['delta_mean']*100:.3f}%")
        results["r3_replication"] = r3_stats
        Vstore["r3"] = np.stack(r3_V)
    else:
        r3_V = list(Vstore["r3"])
        r3_stats = results["r3_replication"]
        E_r3 = Vstore["r3"].sum(0) >= (Vstore["r3"].shape[0] // 2 + 1)

    results["config"] = {"K": K, "tau": TAU, "nfam": NFAM,
                         "epoch_grid": EPOCH_GRID, "sigma_b": SIGMA_B,
                         "smooth": SMOOTH_EPS, "subbag_frac": SUBBAG_FRAC,
                         "r3_seeds": len(R3_SEEDS), "numpy": np.__version__}
    results["targets"] = {
        "m2_plain": {"v": 0.00505, "delta": 0.00809},
        "m2_registered": {"v": 0.00523, "delta": 0.00904},
        "c4": {"v": 0.00489, "delta": 0.01296},
        "m2_consensus_shift_vs_r3": 0.00311}
    results.setdefault("families", {})
    # ---- build the attack families ----
    fams = []
    for ep in EPOCH_GRID:                      # F1 epoch curve
        fams.append((f"F1-ep{ep}", 2000 + ep * 20, {"epochs": ep}))
    fams.append(("F2-bag", 2200, {"epochs": 70, "bag": True}))
    fams.append(("F3-augmix", 2300, {"epochs": 70, "regime": "draw"}))
    fams.append(("F3b-coherent-g15", 2350, {"epochs": 70, "regime": 3}))
    fams.append(("F4a-smooth10", 2400, {"epochs": 70, "smooth": 0.10}))
    fams.append(("F4b-smooth02", 2450, {"epochs": 70, "smooth": 0.02}))
    fams.append(("F5-randepoch", 2500, {"epochs": "random"}))
    for sb in SIGMA_B:                         # F6 the direct attack
        fams.append((f"F6-jit{sb}", 2600 + int(sb * 1000) * 20,
                     {"epochs": 70, "sigma_b": sb}))
    fams.append(("F7-subbag80", 2700, {"epochs": 70, "subbag": True}))
    for sb in [0.10, 0.20]:                    # F8 the adaptive composite
        fams.append((f"F8-comp-g15-jit{sb}", 2800 + int(sb * 1000) * 20,
                     {"epochs": 70, "regime": 3, "sigma_b": sb}))
    for st_ in [0.02, 0.05, 0.10]:             # F9 threshold jitter
        fams.append((f"F9-taujit{st_}", 2900 + int(st_ * 1000) * 20,
                     {"epochs": 70, "sigma_tau": st_}))

    for name, base, cfg in fams:
        if name in results["families"] and name in Vstore:
            continue  # already checkpointed
        seeds = [base + i for i in range(NFAM)]
        stats, Vmat, accs = run_family(name, seeds, Xtr, ytr, Xte, yte,
                                       truth, cfg)
        # coherent shift of this family's consensus vs r3 consensus
        E_f = Vmat.sum(0) >= (Vmat.shape[0] // 2 + 1)
        stats["shift_vs_r3"] = float(np.mean(E_f != E_r3))
        results["families"][name] = stats
        Vstore[name] = Vmat
        # ---- checkpoint ----
        with open(ckpt_json, "w") as f:
            json.dump(results, f, indent=2)
        np.savez_compressed(ckpt_npz, **Vstore)
        print(f"{name:20s} v={stats['v_mean']*100:6.3f}% "
              f"(sd {stats['v_sd']*100:.3f}) delta={stats['delta_mean']*100:6.3f}% "
              f"acc={stats['acc_mean']*100:5.2f}% shift={stats['shift_vs_r3']*100:5.3f}% "
              f"[{stats['elapsed_s']}s]", flush=True)

    if len(results["families"]) < len(fams):
        print(f"checkpoint pass complete: {len(results['families'])}/{len(fams)} "
              "families done - re-run to continue")
        return

    # ---- pooled Pareto analysis vs the M2-alpha targets ----
    pts, keys = [], []
    for name, st in results["families"].items():
        pts.append((st["v_mean"], st["delta_mean"])); keys.append(name)
    pf = pareto_front(pts)
    results["pareto_front"] = [keys[i] for i in pf]

    def dominates(pt, tgt):
        return pt[0] >= tgt[0] and pt[1] <= tgt[1] and (pt[0] > tgt[0] or pt[1] < tgt[1])

    for tgt_name, tgt in [("m2_plain", (0.00505, 0.00809)),
                          ("m2_registered", (0.00523, 0.00904))]:
        doms = [keys[i] for i, p in enumerate(pts) if dominates(p, tgt)]
        # best compliance among families with width >= target width
        wide = [(st["delta_mean"], name) for name, st in results["families"].items()
                if st["v_mean"] >= tgt[0]]
        results[f"attack_{tgt_name}"] = {
            "dominated_by": doms,
            "best_delta_at_matched_width": min(wide) if wide else None}
        print(f"attack {tgt_name}: dominated_by={doms} "
              f"best_delta_at_width={min(wide) if wide else None}")

    # tolerance version: v >= 0.95*v_tgt AND delta <= delta_tgt + 0.05pp
    for tgt_name, tgt in [("m2_plain_tol", (0.00505 * 0.95, 0.00809 + 0.0005)),
                          ("m2_reg_tol", (0.00523 * 0.95, 0.00904 + 0.0005))]:
        doms = [keys[i] for i, p in enumerate(pts)
                if p[0] >= tgt[0] and p[1] <= tgt[1]]
        results[f"attack_{tgt_name}"] = {"dominated_by": doms}
        print(f"attack {tgt_name} (tolerance): {doms}")

    # ---- margin structure: where does the farmed width live? ----
    # flip-vs-consensus rate by |P - tau| bin; margin proxy = an audit-side
    # r3 run's battery probs (seed 200, family-independent reference)
    margin_bins = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01)]
    prng = np.random.default_rng(200)
    st_ = init_params(prng)
    mom_ = zero_moments(st_)
    train_epochs(st_, mom_, prng, Xtr, ytr, 70)
    Pm = committee_probs(Xte, *st_)
    margin = np.abs(Pm.reshape(-1) - TAU)
    ms = {}
    for name, Vmat in Vstore.items():
        E = Vmat.sum(0) >= (Vmat.shape[0] // 2 + 1)
        rows = []
        for lo, hi in margin_bins:
            sel = (margin >= lo) & (margin < hi)
            if sel.sum() == 0:
                continue
            flip = float(np.mean(Vmat[:, sel] != E[sel]))
            rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                         "flip_vs_consensus": flip})
        ms[name] = rows
    results["margin_structure"] = ms

    results["runtime_s"] = time.time() - t0
    np.savez_compressed(ckpt_npz, **Vstore)
    with open(ckpt_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndone in {results['runtime_s']:.0f}s -> farming_results.json")


if __name__ == "__main__":
    main()
