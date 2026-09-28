#!/usr/bin/env python3
"""
M0 SCALING - perturbation batteries, distribution shift, harder corpora.
Corpus doc 14's experiment. Scales the M1 instrument (v1.0 Parts 5-6) beyond
the A4 pilot's clean battery: battery-side covariate shift (10 perturbation
slices at graded severities), out-of-distribution / under-determined-IGM
slices, training-side label-noise shift, and a second, harder corpus (MNIST).

DESIGN P1 (pilot's, re-run): MLP 64-128-64-10, xent, Adam, 60 epochs, f64,
  25 re-instantiations (seeds 0..24), token = seed 0.
DESIGN P1-noise: identical, but the TRAINING labels are corrupted at 20%
  (flip to uniformly random other class; corruption seed 777 = part of the
  DESIGN, fixed across the family, disclosed). Battery = clean test, IGM =
  true labels. 25 runs.
DESIGN P2 (harder corpus): MNIST subsample 12,000 train / 3,000 test
  (stratified, split seed 12345), inputs /255, MLP 784-128-64-10, 12 epochs,
  16 re-instantiations (seeds 0..15), token = seed 0.
IGM (all families): accept (x,y) iff y is the true label of x; bridge
  P(y|x) >= tau, tau = 0.9.
BATTERY SLICES (all deterministic under fixed slice seeds, independent of
  run seeds):
  clean            - 450 held-out digits images (4,500 claims)
  gauss05/10/20    - additive Gaussian noise sigma in {0.05,0.10,0.20}, clip
  bright10/20      - additive brightness +{0.10,0.20}, clip
  contrast07/14    - contrast x0.7 / x1.4 about 0.5, clip
  mask3x3          - zero a random 3x3 block per image
  maskrand20       - zero a random 20% of pixels
  shift1px         - roll each image by a random {-1,0,1}^2 offset
  noiseU (OOD)     - 450 uniform-noise images; IGM = accept NOTHING
  blank  (OOD)     - 40 all-zero images; IGM = accept NOTHING
  blend50 (stip.)  - 450 blends 0.5a+0.5b of random test pairs; IGM
                     (STIPULATED): accept both constituent labels
  P2 clean + P2 gauss15 (sigma 0.15) on the 3,000-image MNIST battery.

PRE-REGISTERED EXPECTATIONS (fixed before execution; script frozen after):
  E1 clean slice reproduces the pilot (v~0.30%, contested~1.0%).
  E2 v grows monotonically with severity; delta grows faster than v.
  E3 spec-typical fraction of delta rises with severity (severity converts
     lottery-residue into design-residue: etiology shifts HISTORY->DESIGN).
  E4 OOD noise/blank: false accepts present but modest; family spread low.
  E5 blend50 (under-determined IGM): the widest dispersion and highest delta.
  E6 P1-noise: delta elevated on clean battery; v elevated modestly.
  E7 P2: contested fraction below P1 (more data -> fewer boundary claims).
  E8 per-run accept rates correlate positively across slices.

Usage: python3 m0_scaling.py
"""
import itertools
import json
import time

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split

TAU = 0.9
EPOCHS, BATCH, LR, H1, H2 = 60, 32, 1e-3, 128, 64
SEEDS = list(range(25))
TOKEN = 0
NOISE_RATE, NOISE_SEED = 0.20, 777
SPLIT_SEED = 12345
MARGIN_BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01)]

# ---- model (identical to the A4 pilot) ----


def init_params(rng, d_in):
    Ws = [rng.normal(0.0, np.sqrt(2.0 / d_in), (d_in, H1)),
          rng.normal(0.0, np.sqrt(2.0 / H1), (H1, H2)),
          rng.normal(0.0, np.sqrt(2.0 / H2), (H2, 10))]
    bs = [np.zeros(H1), np.zeros(H2), np.zeros(10)]
    return Ws, bs


def forward(X, Ws, bs):
    z1 = X @ Ws[0] + bs[0]; a1 = np.maximum(z1, 0.0)
    z2 = a1 @ Ws[1] + bs[1]; a2 = np.maximum(z2, 0.0)
    return a2 @ Ws[2] + bs[2]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(seed, Xtr, ytr, epochs=EPOCHS, d_in=64):
    rng = np.random.default_rng(seed)
    Ws, bs = init_params(rng, d_in)
    m = [np.zeros_like(w) for w in Ws]; v = [np.zeros_like(w) for w in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
    b1a, b2a, eps, t = 0.9, 0.999, 1e-8, 0
    n = len(Xtr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i0 in range(0, n, BATCH):
            idx = order[i0:i0 + BATCH]
            Xb, yb = Xtr[idx], ytr[idx]
            z1 = Xb @ Ws[0] + bs[0]; a1 = np.maximum(z1, 0.0)
            z2 = a1 @ Ws[1] + bs[1]; a2 = np.maximum(z2, 0.0)
            z3 = a2 @ Ws[2] + bs[2]
            P = softmax(z3)
            P[np.arange(len(yb)), yb] -= 1.0
            dZ3 = P / len(yb)
            gW3 = a2.T @ dZ3; gb3 = dZ3.sum(0)
            dA2 = dZ3 @ Ws[2].T; dZ2 = dA2 * (z2 > 0)
            gW2 = a1.T @ dZ2; gb2 = dZ2.sum(0)
            dA1 = dZ2 @ Ws[1].T; dZ1 = dA1 * (z1 > 0)
            gW1 = Xb.T @ dZ1; gb1 = dZ1.sum(0)
            gW, gb = [gW1, gW2, gW3], [gb1, gb2, gb3]
            t += 1
            for k in range(3):
                m[k] = b1a * m[k] + (1 - b1a) * gW[k]
                v[k] = b2a * v[k] + (1 - b2a) * gW[k] ** 2
                Ws[k] -= LR * (m[k] / (1 - b1a ** t)) / (np.sqrt(v[k] / (1 - b2a ** t)) + eps)
                mb[k] = b1a * mb[k] + (1 - b1a) * gb[k]
                vb[k] = b2a * vb[k] + (1 - b2a) * gb[k] ** 2
                bs[k] -= LR * (mb[k] / (1 - b1a ** t)) / (np.sqrt(vb[k] / (1 - b2a ** t)) + eps)
    return Ws, bs


def probs_of(Ws, bs, X):
    return softmax(forward(X, Ws, bs))


# ---- perturbation families (deterministic under slice seeds) ----
def p_gauss(X, sigma, rng):
    return np.clip(X + rng.normal(0.0, sigma, X.shape), 0.0, 1.0)


def p_bright(X, off, rng):
    return np.clip(X + off, 0.0, 1.0)


def p_contrast(X, c, rng):
    return np.clip((X - 0.5) * c + 0.5, 0.0, 1.0)


def p_mask_block(X, rng):
    n = len(X)
    out = X.copy().reshape(n, 8, 8)
    i = rng.integers(0, 6, n); j = rng.integers(0, 6, n)
    for k in range(n):
        out[k, i[k]:i[k] + 3, j[k]:j[k] + 3] = 0.0
    return out.reshape(n, 64)


def p_mask_rand(X, p, rng):
    return X * (rng.random(X.shape) >= p)


def p_shift(X, rng):
    n = len(X)
    out = X.copy().reshape(n, 8, 8)
    di = rng.integers(-1, 2, n); dj = rng.integers(-1, 2, n)
    for k in range(n):
        out[k] = np.roll(out[k], (di[k], dj[k]), axis=(0, 1))
    return out.reshape(n, 64)


def build_slices(Xte, yte):
    """Return dict: slice_name -> (X_slice, igm_truth_flat (n*10,))."""
    S = {}
    S["clean"] = (Xte, None)  # truth from yte, filled by caller
    specs = [
        ("gauss05", 501, lambda r: p_gauss(Xte, 0.05, r)),
        ("gauss10", 502, lambda r: p_gauss(Xte, 0.10, r)),
        ("gauss20", 503, lambda r: p_gauss(Xte, 0.20, r)),
        ("bright10", 504, lambda r: p_bright(Xte, 0.10, r)),
        ("bright20", 505, lambda r: p_bright(Xte, 0.20, r)),
        ("contrast07", 506, lambda r: p_contrast(Xte, 0.7, r)),
        ("contrast14", 507, lambda r: p_contrast(Xte, 1.4, r)),
        ("mask3x3", 508, lambda r: p_mask_block(Xte, r)),
        ("maskrand20", 509, lambda r: p_mask_rand(Xte, 0.20, r)),
        ("shift1px", 510, lambda r: p_shift(Xte, r)),
    ]
    for name, sd, fn in specs:
        S[name] = (fn(np.random.default_rng(sd)), None)

    # OOD: uniform noise, IGM = accept nothing
    r = np.random.default_rng(511)
    S["noiseU"] = (r.uniform(0.0, 1.0, (450, 64)), "reject_all")
    # OOD: blank (all-zero) images, IGM = accept nothing
    S["blank"] = (np.zeros((40, 64)), "reject_all")
    # stipulated-IGM blends: 450 pairs, alpha 0.5, accept both true labels
    r = np.random.default_rng(512)
    ia = r.integers(0, len(Xte), 450); ib = r.integers(0, len(Xte), 450)
    Xb = 0.5 * Xte[ia] + 0.5 * Xte[ib]
    truth = np.zeros((450, 10), dtype=bool)
    for k in range(450):
        truth[k, yte[ia[k]]] = True
        truth[k, yte[ib[k]]] = True
    S["blend50"] = (Xb, truth)
    return S


def slice_truth(X, yte, tag):
    if tag is None:  # same-image slices: truth from the source labels
        return (yte[:, None] == np.arange(10)).reshape(-1)
    if isinstance(tag, str):  # "reject_all"
        return np.zeros(len(X) * 10, dtype=bool)
    return tag.reshape(-1)  # precomputed truth table


# ---- per-slice instrument quantities ----


def slice_stats(probs_by_seed, truth, tau=TAU):
    """probs_by_seed: dict seed -> (n,10) prob table on this slice."""
    V = {s: (P >= tau).reshape(-1) for s, P in probs_by_seed.items()}
    seeds = sorted(V)
    pairs = list(itertools.combinations(seeds, 2))
    pw = [float(np.mean(V[a] != V[b])) for a, b in pairs]
    stack = np.stack([V[s] for s in seeds])
    unanimous = np.all(stack, 0) | ~np.any(stack, 0)
    contested = float(np.mean(~unanimous))
    deltas = [float(np.mean(V[s] != truth)) for s in seeds]
    # token-centered decomposition vs family ensemble (excl. token)
    others = [s for s in seeds if s != TOKEN] if TOKEN in V else seeds[1:]
    if TOKEN in V and len(others) >= 2:
        st = np.stack([V[s] for s in others])
        E = st.sum(0) >= (len(others) // 2 + 1)
        Vt = V[TOKEN]
        spec = float(np.mean((Vt != truth) & (E != truth)))
        idio_e = float(np.mean((Vt != truth) & (E == truth)))
        idio_c = float(np.mean((Vt == truth) & (E != truth)))
        dE = float(np.mean(E != truth))
        r_E = float(np.mean(Vt != E))
    else:
        spec = idio_e = idio_c = dE = r_E = None
    # OOD slices: report accept (false-accept) rates
    acc_rates = [float(np.mean(V[s])) for s in seeds]
    fa = [float(np.mean(V[s] & ~truth)) for s in seeds] if not truth.all() else None
    return {
        "n_claims": int(len(truth)),
        "v_mean": float(np.mean(pw)), "v_sd": float(np.std(pw)),
        "v_min": float(np.min(pw)), "v_max": float(np.max(pw)),
        "contested": contested,
        "delta_mean": float(np.mean(deltas)), "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "accept_mean": float(np.mean(acc_rates)),
        "accept_min": float(np.min(acc_rates)), "accept_max": float(np.max(acc_rates)),
        "spec_typical": spec, "idio_error": idio_e, "idio_correct": idio_c,
        "delta_ensemble": dE, "r_E_token": r_E,
    }


def margin_structure(token_probs, V, seeds, tau=TAU):
    margin = np.abs(token_probs.reshape(-1) - tau)
    others = [s for s in seeds if s != TOKEN]
    out = []
    for lo, hi in MARGIN_BINS:
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        per = [float(np.mean(V[TOKEN][sel] != V[s][sel])) for s in others]
        out.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                    "mean_d": float(np.mean(per))})
    return out


# ---------------- data ----------------


def load_digits_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def load_mnist_data():
    m = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = m.data.astype(np.float64) / 255.0
    y = m.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=12000, test_size=3000, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def corrupt_labels(ytr, rate, seed):
    rng = np.random.default_rng(seed)
    yc = ytr.copy()
    flip = rng.random(len(ytr)) < rate
    newc = rng.integers(0, 9, len(ytr))  # uniform over 0..8, shifted to skip true
    newc = (newc + yc + 1) % 10
    yc[flip] = newc[flip]
    return yc


# ---------------- main ----------------


def main():
    t0 = time.time()
    res = {"config": {"tau": TAU, "epochs": EPOCHS, "p1_seeds": len(SEEDS),
                      "p2_seeds": 16, "p2_epochs": 12,
                      "noise_rate": NOISE_RATE, "noise_seed": NOISE_SEED,
                      "numpy": np.__version__},
           "slices": {}, "p2_slices": {}, "p1noise": {}}

    # ============ P1: pilot design, scaled battery ============
    Xtr, ytr, Xte, yte = load_digits_data()
    slices = build_slices(Xte, yte)
    print(f"[P1] train={len(Xtr)} test={len(Xte)} slices={list(slices)}")

    probs = {name: {} for name in slices}
    states = {}
    for s in SEEDS:
        Ws, bs = train(s, Xtr, ytr)
        states[s] = (Ws, bs)
        for name, (X, _) in slices.items():
            probs[name][s] = probs_of(Ws, bs, X)
        print(f"  P1 run {s:<3d} ({time.time()-t0:.0f}s)")

    # deference control on the P1 token (same env): retrain seed 0
    Ws2, bs2 = train(TOKEN, Xtr, ytr)
    defer_ok = all(
        np.array_equal(probs_of(Ws2, bs2, X), probs["clean"][TOKEN])
        for X, _ in [slices["clean"]])
    res["deference_control"] = {"bitwise_identical": bool(defer_ok)}
    print("  deference control (same-env retrain):", defer_ok)

    for name, (X, tag) in slices.items():
        truth = slice_truth(X, yte, tag)
        res["slices"][name] = slice_stats(probs[name], truth)
        V = {s: (probs[name][s] >= TAU).reshape(-1) for s in SEEDS}
        res["slices"][name]["margin"] = margin_structure(probs[name][TOKEN], V, SEEDS)
        st = res["slices"][name]
        print(f"  slice {name:11s} v={st['v_mean']*100:.3f}% contested={st['contested']*100:.2f}% "
              f"delta={st['delta_mean']*100:.3f}% acc={st['accept_mean']*100:.1f}%")

    # cross-slice run-level correlation of accept rates (seed quality stability)
    names = list(slices)
    ar = np.array([[float(np.mean((probs[n][s] >= TAU))) for n in names] for s in SEEDS])
    C = np.corrcoef(ar.T)
    res["cross_slice_accept_rate_corr"] = {
        "matrix": [[float(x) for x in row] for row in C], "slice_order": names}
    res["cross_slice_corr_mean_offdiag"] = float(
        (C.sum() - np.trace(C)) / (len(names) ** 2 - len(names)))

    # ============ P1-noise: training-side label-noise shift ============
    ynoisy = corrupt_labels(ytr, NOISE_RATE, NOISE_SEED)
    pn = {}
    for s in SEEDS:
        Ws, bs = train(s, Xtr, ynoisy)
        pn[s] = probs_of(Ws, bs, Xte)
        print(f"  P1-noise run {s:<3d} ({time.time()-t0:.0f}s)")
    truth = (yte[:, None] == np.arange(10)).reshape(-1)
    st = slice_stats(pn, truth)
    res["p1noise"] = st
    V = {s: (pn[s] >= TAU).reshape(-1) for s in SEEDS}
    # accuracy of noisy family vs true labels (argmax)
    res["p1noise"]["argmax_acc_mean"] = float(np.mean(
        [np.mean(pn[s].argmax(1) == yte) for s in SEEDS]))
    print(f"  P1-noise: v={st['v_mean']*100:.3f}% delta={st['delta_mean']*100:.3f}% "
          f"argmax-acc={res['p1noise']['argmax_acc_mean']*100:.2f}%")

    # ============ P2: MNIST ============
    Xtr2, ytr2, Xte2, yte2 = load_mnist_data()
    p2slices = {"clean": (Xte2, None),
                "gauss15": (p_gauss(Xte2, 0.15, np.random.default_rng(601)), None)}
    p2probs = {n: {} for n in p2slices}
    P2_SEEDS = list(range(16))
    for s in P2_SEEDS:
        Ws, bs = train(s, Xtr2, ytr2, epochs=12, d_in=784)
        for n, (X, _) in p2slices.items():
            p2probs[n][s] = probs_of(Ws, bs, X)
        print(f"  P2 run {s:<3d} ({time.time()-t0:.0f}s)")
    for n, (X, _) in p2slices.items():
        truth = slice_truth(X, yte2, None)
        st = slice_stats(p2probs[n], truth)
        res["p2_slices"][n] = st
        print(f"  P2 slice {n:8s} v={st['v_mean']*100:.3f}% contested={st['contested']*100:.2f}% "
              f"delta={st['delta_mean']*100:.3f}%")
    res["p2_acc_mean"] = float(np.mean(
        [np.mean(p2probs["clean"][s].argmax(1) == yte2) for s in P2_SEEDS]))

    res["runtime_s"] = time.time() - t0
    with open("/home/z/my-project/scripts/m0_results.json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(
        "/home/z/my-project/scripts/tmp/m0_probs.npz",
        **{f"p1_{n}_{s}": probs[n][s] for n in slices for s in [TOKEN]},
        **{f"p1n_{s}": pn[s] for s in [TOKEN]},
        **{f"p2_{n}_{s}": p2probs[n][s] for n in p2slices for s in [TOKEN]})
    print(f"\ndone in {res['runtime_s']:.0f}s -> scripts/m0_results.json")


if __name__ == "__main__":
    main()
