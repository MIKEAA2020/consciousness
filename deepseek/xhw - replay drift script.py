#!/usr/bin/env python3
"""
XHW REPLAY DRIFT - the cross-hardware replay drift measurement (A2-chaos line).
Corpus doc 15's experiment. Measures, on the A4 pilot design (MLP 64-128-64-10,
digits, 60 epochs, seed-0 token), what happens to the M1 instrument's
deference control and its quantities when the EXECUTION ENVIRONMENT varies:
BLAS kernel class (OPENBLAS_CORETYPE: SkylakeX/Haswell/Nehalem/Sandybridge/
Prescott - genuinely different SIMD/FMA rounding regimes on the same CPU),
thread count (negative control), and float precision (float32 axis).

Environment-manifest convention (M1 spec 2.2): the manifest is DESIGN for
same-environment audits and HISTORY otherwise - so every drift measured here
is a measurement about the HISTORY term: same seed, same code, same data,
different kernel -> is it still "the same run"?

Arms:
  A. token drift: seed-0 token under each variant; bitwise weight equality,
     battery-verdict disagreement vs the default-env token, per-epoch
     trajectory divergence, margin structure of the drift, delta vs IGM.
  B. quantity invariance: full 25-run family under default vs under Nehalem;
     compare v, contested, delta; cross-kernel same-seed vs cross-kernel
     different-seed disagreement (is kernel drift inside the lottery band?).
  C. chaos amplification: 1-ulp flip of single weights at epochs 0/12/24/36/48
     (same env, same RNG stream); per-epoch delta-norm growth curve, fitted
     growth rate, final battery drift of a ONE-BIT perturbation.

PRE-REGISTERED EXPECTATIONS (fixed before execution):
  X1 CORETYPE variants produce bitwise-different final weights; threads=4
     reproduces bitwise (negative control); float32 differs.
  X2 verdict drift d(G_Ei, G_E0) > 0 but within/near the seed-lottery band.
  X3 divergence grows exponentially then saturates; cross-kernel and 1-ulp
     drift share the growth law (chaos is the mechanism).
  X4 the deference zero is manifest-relative (same seed != same run across
     kernels) - the history term splinters into (seed, manifest).
  X5 instrument quantities (v, contested, delta) are manifest-invariant
     within family noise - the etiology-meter survives; the deference base
     does not.
  X6 cross-kernel disagreement is boundary-concentrated (the same penumbra).
  X7 float32 drift exceeds kernel drift; still boundary-concentrated.

Usage: python3 xhw_replay_drift.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

BASE = "/home/z/my-project/scripts/tmp/xhw"
WORKER = "/home/z/my-project/scripts/xhw_worker.py"
TAU = 0.9
MARGIN_BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01)]

VARIANTS = [
    ("skx_default", {}, 1),          # reference (CPU-detected kernels)
    ("haswell", {"OPENBLAS_CORETYPE": "Haswell"}, 1),
    ("nehalem", {"OPENBLAS_CORETYPE": "Nehalem"}, 1),
    ("sandybridge", {"OPENBLAS_CORETYPE": "Sandybridge"}, 1),
    ("prescott", {"OPENBLAS_CORETYPE": "Prescott"}, 1),
    ("threads4", {}, 4),             # negative control
]
ULP_CONFIGS = [
    # (at_epoch, "wi:i:j", lr, epochs) - base design = lr 1e-3, 60 epochs
    (0, "0:0:0", 1e-3, 60), (0, "1:3:5", 1e-3, 60), (0, "2:7:2", 1e-3, 60),
    (12, "0:0:0", 1e-3, 60), (24, "0:0:0", 1e-3, 60),
    (36, "0:0:0", 1e-3, 60), (48, "0:0:0", 1e-3, 60),
    # chaos-boundary probe: hotter designs (same architecture/data)
    (0, "0:0:0", 3e-3, 60), (0, "0:0:0", 1e-2, 60),
    (0, "0:0:0", 1e-2, 200),
]


def run_worker(env_extra, threads, args, tag):
    env = dict(os.environ)
    env.update(env_extra)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    t0 = time.time()
    r = subprocess.run([sys.executable, WORKER] + args, env=env,
                       capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"[{tag}] worker failed:\n{r.stderr[-1500:]}")
    print(f"  [{tag}] {dt:.1f}s  {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
    return dt


def load_igm():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=12345, stratify=y)
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    return truth


def growth_fit(delta, wnorm):
    rel = delta[1:] / wnorm[1:]          # skip epoch 0 (identical init)
    ep = np.arange(1, len(delta))
    sel = (rel > 1e-13) & (rel < 1e-2)
    if sel.sum() >= 5:
        slope, intercept = np.polyfit(ep[sel], np.log10(rel[sel]), 1)
        pred = slope * ep[sel] + intercept
        r2 = 1 - np.sum((np.log10(rel[sel]) - pred) ** 2) / \
            np.sum((np.log10(rel[sel]) - np.log10(rel[sel]).mean()) ** 2)
    else:
        slope = r2 = None
    return {"slope_log10_per_epoch": None if slope is None else float(slope),
            "r2": None if r2 is None else float(r2),
            "final_rel_delta": float(rel[-1]),
            "max_rel_delta": float(rel.max()),
            "first_epoch_rel_gt_1e-8": int(ep[rel > 1e-8][0]) if (rel > 1e-8).any() else None}


def margin_struct(p_ref, disagree_mask):
    margin = np.abs(p_ref.reshape(-1) - TAU)
    out = []
    for lo, hi in MARGIN_BINS:
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        out.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                    "drift_rate": float(np.mean(disagree_mask[sel]))})
    return out


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arm in ("all", "A"):
        shutil.rmtree(f"{BASE}", ignore_errors=True)
    os.makedirs(BASE, exist_ok=True)
    truth = load_igm()
    res = {"config": {"tau": TAU, "ulp_configs": [f"ep{a}:{w}:lr{l}:ep{n}"
                                                for a, w, l, n in ULP_CONFIGS],
                      "numpy": np.__version__}, "variants": {}}

    # ---------- Arm A: token drift across environments ----------
    if arm in ("all", "A"):
        tok = {}
        times = {}
        for name, env_extra, threads in VARIANTS:
            d = f"{BASE}/{name}"
            os.makedirs(d, exist_ok=True)
            times[name] = run_worker(env_extra, threads, ["token", "0", d], name)
            tok[name] = np.load(f"{d}/token.npz")
        # float32 axis
        d = f"{BASE}/f32"
        os.makedirs(d, exist_ok=True)
        times["f32"] = run_worker({}, 1, ["token", "0", d, "--f32"], "f32")
        tok["f32"] = np.load(f"{d}/token.npz")
        import pickle
        with open(f"{BASE}/_times.pkl", "wb") as f:
            pickle.dump(times, f)

    if arm in ("all", "A", "analyze"):
        import pickle
        with open(f"{BASE}/_times.pkl", "rb") as f:
            times = pickle.load(f)
        tok = {name: np.load(f"{BASE}/{name}/token.npz")
               for name in [v[0] for v in VARIANTS] + ["f32"]}

        ref = tok["skx_default"]
        ref_snaps, ref_probs = ref["snaps"], ref["probs"]
        ref_V = (ref_probs >= TAU).reshape(-1)
        ref_delta_igm = float(np.mean(ref_V != truth))
        res["reference"] = {"delta_igm": ref_delta_igm,
                            "final_w_sha": str(hash(ref_snaps[-1].tobytes()))}

        for name in list(tok):
            if name == "skx_default":
                continue
            t = tok[name]
            snaps, probs = t["snaps"], t["probs"]
            bitwise = bool(np.array_equal(snaps[-1], ref_snaps[-1]))
            V = (probs >= TAU).reshape(-1)
            d_verd = float(np.mean(V != ref_V))
            d_arg = float(np.mean(probs.argmax(1) != ref_probs.argmax(1)))
            delta_igm = float(np.mean(V != truth))
            # trajectory divergence vs reference (per epoch)
            dd = np.linalg.norm(snaps - ref_snaps, axis=1)
            wn = np.linalg.norm(ref_snaps, axis=1)
            g = growth_fit(np.linalg.norm(snaps - ref_snaps, axis=1), wn) \
                if not bitwise else None
            res["variants"][name] = {
                "bitwise_identical_final_weights": bitwise,
                "d_verdicts_vs_ref": d_verd,
                "d_argmax_vs_ref": d_arg,
                "delta_igm": delta_igm,
                "final_rel_w_delta": float(dd[-1] / wn[-1]),
                "init_rel_w_delta": float(dd[1] / wn[1]),
                "growth": g,
                "margin_drift": margin_struct(ref_probs, (V != ref_V)) if d_verd > 0 else [],
                "runtime_s": times[name],
            }
            print(f"  {name:12s} bitwise={bitwise} d_verdicts={d_verd*100:.4f}% "
                  f"d_argmax={d_arg*100:.4f}% final_rel_d={dd[-1]/wn[-1]:.2e}")

    # replication check: reference token must match the M0/P1 run's deference
    # (same env, same seed) - cross-checked against m0_results.json later.

    # ---------- Arm B: quantity invariance across kernels ----------
    if arm in ("all", "B"):
        ddef = f"{BASE}/fam_default"; os.makedirs(ddef, exist_ok=True)
        run_worker({}, 1, ["family", ddef, ",".join(str(s) for s in range(25))], "fam_default")
        dneh = f"{BASE}/fam_nehalem"; os.makedirs(dneh, exist_ok=True)
        run_worker({"OPENBLAS_CORETYPE": "Nehalem"}, 1,
                   ["family", dneh, ",".join(str(s) for s in range(25))], "fam_nehalem")

    if arm in ("all", "B", "analyze"):
        import itertools
        Pd = np.load(f"{BASE}/fam_default/family.npz")["probs"]   # (25, 450, 10)
        Pn = np.load(f"{BASE}/fam_nehalem/family.npz")["probs"]    # nehalem
        Vd = (Pd >= TAU).reshape(25, -1)
        Vn = (Pn >= TAU).reshape(25, -1)

        def fam_stats(V):
            seeds = V.shape[0]
            pw = [float(np.mean(V[a] != V[b]))
                  for a, b in itertools.combinations(range(seeds), 2)]
            unan = np.all(V, 0) | ~np.any(V, 0)
            deltas = [float(np.mean(V[s] != truth)) for s in range(seeds)]
            return {"v_mean": float(np.mean(pw)), "v_sd": float(np.std(pw)),
                    "v_min": float(np.min(pw)), "v_max": float(np.max(pw)),
                    "contested": float(np.mean(~unan)),
                    "delta_mean": float(np.mean(deltas)),
                    "delta_min": float(np.min(deltas)),
                    "delta_max": float(np.max(deltas))}

        sd, sn = fam_stats(Vd), fam_stats(Vn)
        same_seed = [float(np.mean(Vd[s] != Vn[s])) for s in range(25)]
        diff_seed = [float(np.mean(Vd[a] != Vn[b]))
                     for a in range(25) for b in range(25) if a != b]
        res["quantity_invariance"] = {
            "default": sd, "nehalem": sn,
            "cross_kernel_same_seed_mean": float(np.mean(same_seed)),
            "cross_kernel_same_seed_max": float(np.max(same_seed)),
            "cross_kernel_diff_seed_mean": float(np.mean(diff_seed)),
            "same_seed_drift_per_seed": [float(x) for x in same_seed],
        }
        print(f"  family default: v={sd['v_mean']*100:.3f}% contested={sd['contested']*100:.2f}%")
        print(f"  family nehalem: v={sn['v_mean']*100:.3f}% contested={sn['contested']*100:.2f}%")
        print(f"  cross-kernel same-seed drift: mean={np.mean(same_seed)*100:.4f}% "
              f"max={np.max(same_seed)*100:.4f}%  (lottery band: {np.mean(diff_seed)*100:.3f}%)")

    # ---------- Arm C: chaos amplification of a single ulp ----------
    if arm in ("all", "C"):
        for at, which, lr, epn in ULP_CONFIGS:
            run_worker({}, 1, ["ulp", "0", f"{BASE}", str(at), which,
                               str(lr), str(epn)], f"ulp{at}:{which}:lr{lr}")

    if arm in ("all", "C", "analyze"):
        res["ulp"] = {}
        for at, which, lr, epn in ULP_CONFIGS:
            fn = f"{BASE}/ulp_{at}_{which.replace(':', '-')}_lr{lr}_ep{epn}.npz"
            z = np.load(fn)
            entry = {"at_epoch": at, "which": which, "lr": lr, "epochs": epn,
                     "d_verdicts": float(z["d_verdicts"]),
                     "d_argmax": float(z["d_argmax"]),
                     **growth_fit(z["delta"], z["wnorm"])}
            res["ulp"][f"ep{at}:{which}:lr{lr}:ep{epn}"] = entry
            print(f"  ulp ep{at} {which} lr={lr} ep={epn}: "
                  f"d_verdicts={entry['d_verdicts']*100:.4f}% "
                  f"final_rel={entry['final_rel_delta']:.2e} "
                  f"slope={entry['slope_log10_per_epoch']}")

    res["runtime_note"] = "runtimes under concurrent load; indicative only"
    with open("/home/z/my-project/scripts/xhw_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("-> scripts/xhw_results.json")


if __name__ == "__main__":
    main()
