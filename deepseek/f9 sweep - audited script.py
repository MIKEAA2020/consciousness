#!/usr/bin/env python3
"""
THE F9 SATURATION SWEEP - mapping the farm's full frontier.
Corpus doc 21's experiment. Ordered at the M2-beta issuance: "an F9 extension
sweep (sigma_tau to delta-saturation) to map the farm's full frontier."

WHERE THE CAMPAIGN LEFT IT (farming_results.json): F9 per-run threshold
jitter sigma_tau=0.10 DOMINATES the M2-alpha plain point (v=0.518%, delta=
0.787% vs 0.505/0.809) - norm heterogeneity is a width dial invisible to the
Tier-0 statistic. One point is not a frontier. This sweep traces the whole
arc: sigma_tau from 0.01 to 0.50, N=12 runs per level, plus the farmer's
composite (coherent-g15 augmentation + threshold jitter) at three levels to
test whether the compliance injector extends the arc.

PRE-REGISTERED EXPECTATIONS (fixed before execution):
  S1 (saturation): v(sigma) is monotone increasing and SATURATES at the
      in-band verdict mass M = mean fraction of battery claims with conf in
      [0.5, 0.99] - the width ceiling is a property of the pipeline's margin
      distribution, not of the farmer. The farmer rents width up to M.
  S2 (price): delta(sigma) grows monotonically toward the in-band ERROR mass
      (the wrongness the rented width lets in); the (v, delta) arc traced by
      the sweep is the farm's full frontier and every previously-measured
      family point (22 farming + M2-alpha/beta + C4/r3) lies ON or BELOW it.
  S3 (decomposition): the scatter-model predicted v - each run's per-claim
      class probability modeled as family-mean + iid residual from the
      pooled empirical scatter, pair disagreement = straddle probability
      over the two realized thresholds - satisfies v_measured(sigma)
      ~= v_scatter_predicted(sigma) + (v_r3 - v_scatter_predicted(0))
      i.e. the frontier is a functional of the run-level probability
      distribution alone: conf scatter (the seed lottery, not farmable per
      doc 17) + margin mass (farmable at will up to M).
  S4 (composite): F10 (g15 + tau-jitter) shifts the arc OUTWARD (better
      delta at matched v) - the injector buys compliance at the boundary
      and the frontier extends beyond pure norm heterogeneity.

Usage: python3 f9_sweep.py [--smoke] [--reset]
  Checkpointed after each family (foreground chunks; determinism = exact
  resume), mirroring farming_attack.py.
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
SIGMA_GRID = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
COMP_SIGMA = [0.05, 0.10, 0.20]
TAU_CLIP = (0.5, 0.99)

CKPT_JSON = "/home/z/my-project/scripts/f9_results.json"
CKPT_NPZ = "/home/z/my-project/scripts/tmp/f9_V.npz"
FARM_JSON = "/home/z/my-project/scripts/farming_results.json"
FARM_NPZ = "/home/z/my-project/scripts/tmp/farming_V.npz"


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


def train_epochs(state, mom, rng, X, y, epochs, perturb_fn=None):
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


def perturb_g15(X, rng):
    return np.clip(X + rng.normal(0, 1, X.shape) * 0.15, 0, 1)


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


def run_family(name, seeds, Xtr, ytr, Xte, yte, truth, sigma_tau, g15=False):
    """One sweep family: purely supervised 70ep, per-run tau_i ~ N(0.9, s),
    clipped to [0.5, 0.99] (identical draw rule to the campaign's F9)."""
    V_list, accs, taus, probs, accepts = [], [], [], [], []
    t0 = time.time()
    for s in seeds:
        rng = np.random.default_rng(s)
        state = init_params(rng)
        mom = zero_moments(state)
        pfn = perturb_g15 if g15 else None
        train_epochs(state, mom, rng, Xtr, ytr, 70, perturb_fn=pfn)
        tau_run = TAU
        if sigma_tau:
            tau_run = float(max(TAU_CLIP[0], min(TAU_CLIP[1],
                          rng.normal(TAU, sigma_tau))))
        P = committee_probs(Xte, *state)
        V_list.append((P >= tau_run).reshape(-1))
        taus.append(tau_run)
        probs.append(P)
        accs.append(float(np.mean(P.argmax(1) == yte)))
        accepts.append(float(np.mean(P.max(1) >= tau_run)))
    stats = fam_stats(V_list, truth)
    stats["acc_mean"] = float(np.mean(accs))
    stats["accept_mean"] = float(np.mean(accepts))
    stats["tau_mean"] = float(np.mean(taus))
    stats["tau_sd"] = float(np.std(taus))
    stats["tau_jitter_actual"] = [round(t, 4) for t in taus]
    stats["seeds"] = list(seeds)
    stats["cfg"] = {"epochs": 70, "sigma_tau": sigma_tau, "g15": g15}
    stats["elapsed_s"] = round(time.time() - t0, 1)
    # ---- verdict-relevant margin statistics: claims are (image, class)
    # pairs, verdict = P[n,c] >= tau - the FLATTENED class probability,
    # not the argmax confidence ----
    Pm = np.stack(probs)                      # (R, 450, 10)
    Pf = Pm.reshape(len(probs), -1)           # (R, 4500) per-claim probs
    cbar = Pf.mean(0)                         # family-mean per claim
    # in-band verdict mass: claims whose class-prob falls in the tau clip
    # range [0.5, 0.99] - the maximum mass threshold heterogeneity can flip
    stats["inband_mass"] = float(np.mean(
        (cbar >= TAU_CLIP[0]) & (cbar <= TAU_CLIP[1])))
    stats["inband_argmax_conf"] = float(np.mean(
        (Pm.mean(0).max(1) >= TAU_CLIP[0]) & (Pm.mean(0).max(1) <= TAU_CLIP[1])))
    # conf scatter across runs (per claim, family-relative)
    resid = (Pf - cbar).reshape(-1)
    stats["conf_scatter_mad"] = float(np.mean(np.abs(resid)))
    # ---- scatter-model predicted v (S3): treat each run's per-claim prob
    # as c + e, e drawn iid from the pooled empirical residual distribution;
    # pair disagreement = P(straddle the two thresholds) ----
    e_sorted = np.sort(resid)
    n_e = len(e_sorted)
    def ecdf(x):
        return np.searchsorted(e_sorted, x, side="right") / n_e
    pr = []
    for (i, j) in itertools.combinations(range(len(taus)), 2):
        t_lo, t_hi = sorted([taus[i], taus[j]])
        F_lo = ecdf(t_lo - cbar)              # P(e < t_lo - c)
        F_hi = ecdf(t_hi - cbar)
        p_dis = (1 - F_lo) * F_hi + F_lo * (1 - F_hi)
        pr.append(float(np.mean(p_dis)))
    stats["v_scatter_predicted"] = float(np.mean(pr))
    return stats, np.stack(V_list)


def main():
    smoke = "--smoke" in sys.argv
    reset = "--reset" in sys.argv
    if smoke:
        globals()["NFAM"] = 3
        globals()["SIGMA_GRID"] = [0.02, 0.10, 0.30]
        globals()["COMP_SIGMA"] = [0.10]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    print(f"data: train={len(Xtr)} test={len(Xte)} claims={len(truth)}")

    # ---- r3 reference: reuse the campaign's replication (exact V matrix) ----
    farm = json.load(open(FARM_JSON))
    Vfarm = np.load(FARM_NPZ)
    r3_V = Vfarm["r3"]
    r3_stats = farm["r3_replication"]
    E_r3 = r3_V.sum(0) >= (r3_V.shape[0] // 2 + 1)
    v0 = r3_stats["v_mean"]
    print(f"r3 reference (reused): v={v0*100:.3f}% delta={r3_stats['delta_mean']*100:.3f}%")

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

    results["config"] = {"K": K, "tau": TAU, "nfam": NFAM,
                         "sigma_grid": SIGMA_GRID, "comp_sigma": COMP_SIGMA,
                         "tau_clip": list(TAU_CLIP), "numpy": np.__version__}
    results["targets"] = {
        "m2a_plain": {"v": 0.00505, "delta": 0.00809},
        "m2a_registered": {"v": 0.00523, "delta": 0.00904},
        "m2b": {"v": 0.00619, "delta": 0.00855},
        "r3": {"v": v0, "delta": r3_stats["delta_mean"]},
        "f9_010_campaign": {"v": 0.00518, "delta": 0.00787}}
    results.setdefault("families", {})

    fams = []
    for s in SIGMA_GRID:
        fams.append((f"F9s-{s}", 3200 + int(s * 1000) * 3, {"sigma_tau": s}))
    for s in COMP_SIGMA:
        fams.append((f"F10-comp-{s}", 3600 + int(s * 1000) * 3,
                     {"sigma_tau": s, "g15": True}))

    for name, base, cfg in fams:
        if name in results["families"] and name in Vstore:
            continue
        seeds = [base + i for i in range(NFAM)]
        stats, Vmat = run_family(name, seeds, Xtr, ytr, Xte, yte, truth,
                                 cfg["sigma_tau"], g15=cfg.get("g15", False))
        E_f = Vmat.sum(0) >= (Vmat.shape[0] // 2 + 1)
        stats["shift_vs_r3"] = float(np.mean(E_f != E_r3))
        results["families"][name] = stats
        Vstore[name] = Vmat
        with open(CKPT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        np.savez_compressed(CKPT_NPZ, **Vstore)
        print(f"{name:14s} v={stats['v_mean']*100:6.3f}% delta={stats['delta_mean']*100:6.3f}% "
              f"pred_scat={stats['v_scatter_predicted']*100:6.3f}% inband={stats['inband_mass']*100:5.2f}% "
              f"acc={stats['acc_mean']*100:5.2f}% shift={stats['shift_vs_r3']*100:5.3f}% "
              f"[{stats['elapsed_s']}s]", flush=True)

    if len(results["families"]) < len(fams):
        print(f"checkpoint pass complete: {len(results['families'])}/{len(fams)} - re-run to continue")
        return

    # ---- saturation & decomposition analysis ----
    sig, vv, dd, pt, ib = [], [], [], [], []
    for s in SIGMA_GRID:
        st = results["families"][f"F9s-{s}"]
        sig.append(s); vv.append(st["v_mean"]); dd.append(st["delta_mean"])
        pt.append(st["v_scatter_predicted"]); ib.append(st["inband_mass"])
    results["saturation"] = {
        "sigma": sig, "v": vv, "delta": dd,
        "v_scatter_predicted": pt, "inband_mass": ib,
        "v_final_over_inband": vv[-1] / ib[-1] if ib[-1] else None,
        "v_increment_over_r3": [x - v0 for x in vv],
        "pred_increment": [p for p in pt],
        "offset_meas_minus_pred": [x - v0 - p for x, p in zip(vv, pt)]}
    sat = results["saturation"]
    print("\nsaturation curve v(sigma):",
          [f"{x*100:.2f}" for x in vv])
    print("in-band mass M:          ",
          [f"{x*100:.2f}" for x in ib])
    print("v - pred_tau - v_r3:     ",
          [f"{x*100:+.3f}" for x in sat["offset_meas_minus_pred"]])

    # ---- the pooled frontier: sweep arc vs every measured family point ----
    pts, keys = [], []
    for name, st in farm["families"].items():
        pts.append((st["v_mean"], st["delta_mean"])); keys.append(f"farm:{name}")
    for name, st in results["families"].items():
        pts.append((st["v_mean"], st["delta_mean"])); keys.append(f"sweep:{name}")
    for nm, (v_, d_) in [("m2a_plain", (0.00505, 0.00809)),
                         ("m2a_reg", (0.00523, 0.00904)),
                         ("m2b", (0.00619, 0.00855))]:
        pts.append((v_, d_)); keys.append(nm)

    def dominates(p, t):
        return p[0] >= t[0] and p[1] <= t[1] and (p[0] > t[0] or p[1] < t[1])

    frontier_report = {}
    for nm, tgt in [("m2a_plain", (0.00505, 0.00809)),
                    ("m2a_registered", (0.00523, 0.00904)),
                    ("m2b", (0.00619, 0.00855))]:
        doms = [k for k, p in zip(keys, pts) if dominates(p, tgt)]
        frontier_report[nm] = {"dominated_by": doms}
        print(f"frontier vs {nm}: {len(doms)} dominating points")
    # best delta at each width level (the arc itself, summarized)
    levels = [0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]
    arc = []
    for lv in levels:
        cand = [(st_d, k) for k, (st_v, st_d) in zip(keys, pts) if st_v >= lv]
        arc.append({"width_level": lv,
                    "best_delta": min(cand) if cand else None})
    frontier_report["arc_best_delta_at_width"] = arc
    results["frontier_report"] = frontier_report

    # S3 check: is the offset constant?
    off = sat["offset_meas_minus_pred"]
    results["s3_offset"] = {
        "mean": float(np.mean(off)), "sd": float(np.std(off)),
        "v_r3_baseline": v0,
        "constant_within": bool(np.std(off) < 0.30 * max(abs(np.mean(off)), 1e-4))}

    results["runtime_s"] = time.time() - t0
    with open(CKPT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(CKPT_NPZ, **Vstore)
    print(f"\ndone in {results['runtime_s']:.0f}s -> f9_results.json")


if __name__ == "__main__":
    main()
