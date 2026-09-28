#!/usr/bin/env python3
"""
A4 - the ensemble argument, run on a real pipeline.
The corpus's first quantitative artifact. Implements M1 instrument
specification v1.0, Parts 5-6: deference control, variance bound,
divergence profile, dose-response price schedule, structural decompositions.

DESIGN  : MLP 64-128-64-10 (ReLU, softmax), cross-entropy, Adam(lr 1e-3,
          beta1 .9, beta2 .999, eps 1e-8), batch 32, 60 epochs, float64,
          inputs /16, stratified split seed 12345 (1347 train / 450 test).
OBJECTIVE (a) optimizing signal: multinomial cross-entropy minimization.
          (b) registered norm (IGM): accept (x,y) iff y is the true label
              of x; bridge: accept iff P(y|x) >= tau, tau = 0.9 primary
              (sensitivity tau in {.7,.8,.95}).
HISTORY : one uint32 seed per run -> single RNG stream (init draws, then
          per-epoch permutations). Nothing else varies across runs.
Runs    : N=25 (seeds 0..24). Token system = seed 0. Deference control =
          seed 0 re-run (must reproduce battery verdicts bitwise).
Continuations: J=8 (seeds 101..108) from token checkpoints at epoch
          floor(60*f), f in {0,.2,.4,.6,.8,1.0}.
Battery : 450 held-out images x 10 label-claims = 4500 claims.
Data    : UCI Optical Recognition of Handwritten Digits (sklearn bundle).
Env     : single-threaded BLAS, CPU, numpy float64 throughout.

Usage: python3 a4_ensemble_variance.py [--calibrate]
"""
import hashlib
import itertools
import json
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- frozen configuration ----------------
SPLIT_SEED = 12345
SEEDS = list(range(25))            # re-instantiation family (design + objective, redrawn history)
TOKEN_SEED = 0
CONT_SEEDS = list(range(101, 117))  # 16 fresh continuation seeds
F_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EPOCHS = 60
BATCH = 32
LR = 1e-3
H1, H2 = 128, 64
TAU = 0.9
TAUS = [0.7, 0.8, 0.9, 0.95]
MARGIN_BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01)]
CKPT_EPOCHS = {0.2: 12, 0.4: 24, 0.6: 36, 0.8: 48, 1.0: 60}  # f -> completed epochs


# ---------------- model ----------------
def init_params(rng):
    Ws = [rng.normal(0.0, np.sqrt(2.0 / 64), (64, H1)),
          rng.normal(0.0, np.sqrt(2.0 / H1), (H1, H2)),
          rng.normal(0.0, np.sqrt(2.0 / H2), (H2, 10))]
    bs = [np.zeros(H1), np.zeros(H2), np.zeros(10)]
    return Ws, bs


def forward(X, Ws, bs):
    z1 = X @ Ws[0] + bs[0]; a1 = np.maximum(z1, 0.0)
    z2 = a1 @ Ws[1] + bs[1]; a2 = np.maximum(z2, 0.0)
    z3 = a2 @ Ws[2] + bs[2]
    return z1, a1, z2, a2, z3


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(seed, Xtr, ytr, epochs=EPOCHS, ckpt=None, start_epoch=0,
          capture_epochs=()):
    """Train from scratch, or continue from `ckpt` at `start_epoch`.
    The RNG stream is the run's realized history: init draws (if from
    scratch) then per-epoch permutations, in one stream."""
    rng = np.random.default_rng(seed)
    if ckpt is None:
        Ws, bs = init_params(rng)
        m = [np.zeros_like(w) for w in Ws]; v = [np.zeros_like(w) for w in Ws]
        mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
        t = 0
    else:
        Ws = [w.copy() for w in ckpt["Ws"]]; bs = [b.copy() for b in ckpt["bs"]]
        m = [x.copy() for x in ckpt["m"]]; v = [x.copy() for x in ckpt["v"]]
        mb = [x.copy() for x in ckpt["mb"]]; vb = [x.copy() for x in ckpt["vb"]]
        t = ckpt["t"]
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    n = len(Xtr)
    snaps = {}
    if 0 in capture_epochs:  # the initialization state itself
        snaps[0] = snap(Ws, bs, m, v, mb, vb, t)
    for ep in range(start_epoch, epochs):
        order = rng.permutation(n)
        for i0 in range(0, n, BATCH):
            idx = order[i0:i0 + BATCH]
            Xb, yb = Xtr[idx], ytr[idx]
            z1, a1, z2, a2, z3 = forward(Xb, Ws, bs)
            P = softmax(z3)
            P[np.arange(len(yb)), yb] -= 1.0
            dZ3 = P / len(yb)
            gW3 = a2.T @ dZ3; gb3 = dZ3.sum(0)
            dA2 = dZ3 @ Ws[2].T; dZ2 = dA2 * (z2 > 0)
            gW2 = a1.T @ dZ2; gb2 = dZ2.sum(0)
            dA1 = dZ2 @ Ws[1].T; dZ1 = dA1 * (z1 > 0)
            gW1 = Xb.T @ dZ1; gb1 = dZ1.sum(0)
            gW = [gW1, gW2, gW3]; gb = [gb1, gb2, gb3]
            t += 1
            for k in range(3):
                m[k] = b1a * m[k] + (1 - b1a) * gW[k]
                v[k] = b2a * v[k] + (1 - b2a) * gW[k] ** 2
                Ws[k] -= LR * (m[k] / (1 - b1a ** t)) / (np.sqrt(v[k] / (1 - b2a ** t)) + eps)
                mb[k] = b1a * mb[k] + (1 - b1a) * gb[k]
                vb[k] = b2a * vb[k] + (1 - b2a) * gb[k] ** 2
                bs[k] -= LR * (mb[k] / (1 - b1a ** t)) / (np.sqrt(vb[k] / (1 - b2a ** t)) + eps)
        if (ep + 1) in capture_epochs:
            snaps[ep + 1] = snap(Ws, bs, m, v, mb, vb, t)
    return {"Ws": Ws, "bs": bs, "m": m, "v": v, "mb": mb, "vb": vb, "t": t}, snaps


def snap(Ws, bs, m, v, mb, vb, t):
    return {"Ws": [w.copy() for w in Ws], "bs": [b.copy() for b in bs],
            "m": [x.copy() for x in m], "v": [x.copy() for x in v],
            "mb": [x.copy() for x in mb], "vb": [x.copy() for x in vb], "t": t}


def probs_of(state, Xte):
    _, _, _, _, z3 = forward(Xte, state["Ws"], state["bs"])
    return softmax(z3)


def verdicts(probs, tau):
    return (probs >= tau).reshape(-1)


def disagree(a, b):
    return float(np.mean(a != b))


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


# ---------------- data ----------------
def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


# ---------------- main ----------------
def main():
    calibrate = "--calibrate" in sys.argv
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    truth = (yte[:, None] == np.arange(10)).reshape(-1)  # IGM on the battery
    n_params = (64 * H1 + H1) + (H1 * H2 + H2) + (H2 * 10 + 10)
    print(f"data: train={len(Xtr)} test={len(Xte)} battery={len(truth)} params={n_params}")

    seeds = SEEDS[:6] if calibrate else SEEDS
    runs = {}
    for s in seeds:
        st, _ = train(s, Xtr, ytr)
        runs[s] = probs_of(st, Xte)
        print(f"  run seed={s:<3d} acc={float(np.mean(runs[s].argmax(1) == yte)):.4f} "
              f"accept@{TAU}={float(np.mean(verdicts(runs[s], TAU))):.4f} "
              f"({time.time()-t0:.0f}s)")

    if calibrate:
        vv = [disagree(verdicts(runs[a], TAU), verdicts(runs[b], TAU))
              for a, b in itertools.combinations(seeds, 2)]
        aa = [float(np.mean(runs[a].argmax(1) != runs[b].argmax(1)))
              for a, b in itertools.combinations(seeds, 2)]
        print(f"CALIBRATION: battery-disagreement mean={np.mean(vv):.5f} "
              f"range=[{min(vv):.5f},{max(vv):.5f}]  argmax-disagreement mean={np.mean(aa):.5f}")
        return

    # ---- token system + deference control ----
    token_state, snaps = train(TOKEN_SEED, Xtr, ytr,
                               capture_epochs=(0,) + tuple(CKPT_EPOCHS.values()))
    token_probs = probs_of(token_state, Xte)
    assert sha(token_probs) == sha(runs[TOKEN_SEED]), "token rerun mismatch"
    replay_state, _ = train(TOKEN_SEED, Xtr, ytr)
    replay_probs = probs_of(replay_state, Xte)
    deference_hash_equal = sha(replay_probs) == sha(token_probs)
    r_deference = disagree(verdicts(replay_probs, TAU), verdicts(token_probs, TAU))
    print(f"deference control: bitwise-identical={deference_hash_equal} d={r_deference}")

    # ---- variance bound (A4 core) ----
    V = {s: verdicts(runs[s], TAU) for s in SEEDS}
    pairs = list(itertools.combinations(SEEDS, 2))
    pairwise = [disagree(V[a], V[b]) for a, b in pairs]
    argmax_dis = [float(np.mean(runs[a].argmax(1) != runs[b].argmax(1))) for a, b in pairs]
    others = [s for s in SEEDS if s != TOKEN_SEED]
    r_R = float(np.mean([disagree(V[TOKEN_SEED], V[s]) for s in others]))

    # ---- contested fraction: where the family is not unanimous ----
    unanimous = np.all(np.stack([V[s] for s in SEEDS]), axis=0) | \
                ~np.any(np.stack([V[s] for s in SEEDS]), axis=0)
    contested = float(np.mean(~unanimous))

    # ---- auditor's ensemble (Tier-0 best predictor; excludes the token) ----
    stack = np.stack([V[s] for s in others])           # (24, 4500)
    E = stack.sum(0) >= (len(others) // 2 + 1)         # majority, tie -> reject
    r_E = disagree(V[TOKEN_SEED], E)
    E_probs = np.mean(np.stack([runs[s] for s in others]), axis=0)
    E_acc = float(np.mean(E_probs.argmax(1) == yte))
    tok_acc = float(np.mean(token_probs.argmax(1) == yte))
    accs = [float(np.mean(runs[s].argmax(1) == yte)) for s in SEEDS]

    # ---- divergence from IGM + decomposition (token-centered, vs E) ----
    Vt, dE = V[TOKEN_SEED], E
    delta_token = disagree(Vt, truth)
    delta_runs = [disagree(V[s], truth) for s in SEEDS]
    delta_E = disagree(E.astype(bool), truth)
    spec_typ = float(np.mean((Vt != truth) & (dE != truth)))
    idio_err = float(np.mean((Vt != truth) & (dE == truth)))
    idio_cor = float(np.mean((Vt == truth) & (dE != truth)))
    false_accepts = float(np.mean(Vt & ~truth))    # accepted a false claim
    true_rejects = float(np.mean(~Vt & truth))     # rejected a true claim
    r_E_runs = [disagree(V[s], E) for s in others]

    # ---- leave-one-out symmetric idiosyncrasy ----
    full = np.stack([V[s] for s in SEEDS])
    loo = []
    for k, s in enumerate(SEEDS):
        sub = np.delete(full, k, axis=0)
        Ek = sub.sum(0) >= (sub.shape[0] // 2 + 1)
        loo.append(disagree(V[s], Ek))

    # ---- dose-response (per-trace retreat price schedule) ----
    dose = []
    for f in F_FRACS:
        ep = CKPT_EPOCHS[f] if f > 0 else 0
        if f == 1.0:
            # no training remains: the "continuation" is the checkpoint itself.
            # Verify directly against the token's final state (deference point).
            dp = disagree(verdicts(probs_of(snaps[60], Xte), TAU), Vt)
            assert dp == 0.0, "f=1.0 checkpoint must equal token final state"
            ds = [0.0] * len(CONT_SEEDS)
        else:
            ck = snaps[ep]
            ds = [disagree(verdicts(probs_of(train(cs, Xtr, ytr, ckpt=ck,
                                                  start_epoch=ep)[0], Xte), TAU), Vt)
                  for cs in CONT_SEEDS]
        dose.append({"f": f, "epoch": ep, "mean": float(np.mean(ds)),
                     "min": float(np.min(ds)), "max": float(np.max(ds))})
        print(f"  dose f={f:.1f} ep={ep}: d mean={dose[-1]['mean']:.5f} "
              f"range=[{dose[-1]['min']:.5f},{dose[-1]['max']:.5f}]")

    # ---- margin stratification (variance lives where?) ----
    margin = np.abs(token_probs.reshape(-1) - TAU)
    mstruct = []
    for lo, hi in MARGIN_BINS:
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        per = [disagree(Vt[sel], V[s][sel]) for s in others]
        mstruct.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(sel.sum()),
                        "mean_d": float(np.mean(per))})

    # ---- threshold sensitivity ----
    tau_sens = []
    for tau in TAUS:
        Vs = {s: verdicts(runs[s], tau) for s in SEEDS}
        pw = [disagree(Vs[a], Vs[b]) for a, b in pairs]
        dl = [disagree(Vs[s], truth) for s in SEEDS]
        tau_sens.append({"tau": tau, "v_mean": float(np.mean(pw)),
                         "v_max": float(np.max(pw)),
                         "delta_mean": float(np.mean(dl)),
                         "accept_mean": float(np.mean([Vs[s].mean() for s in SEEDS]))})

    # ---- description-length accounting ----
    w_bytes = n_params * 8
    ck_bytes = w_bytes * 3                      # weights + Adam m,v
    traj_bytes = 61 * ck_bytes + EPOCHS * len(Xtr) * 8 + 4   # states + orders + seed
    spec_str = (f"MLP 64-{H1}-{H2}-10 relu softmax xent adam lr={LR} b={BATCH} "
                f"ep={EPOCHS} split={SPLIT_SEED} tau={TAU} digits/16")
    results = {
        "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "hidden": [H1, H2],
                   "n_runs": len(SEEDS), "n_continuations": len(CONT_SEEDS),
                   "tau": TAU, "params": n_params, "train_n": len(Xtr),
                   "test_n": len(Xte), "battery_n": int(len(truth)),
                   "numpy": np.__version__},
        "deference": {"bitwise_identical": bool(deference_hash_equal),
                      "d": r_deference},
        "variance": {"pairwise_mean": float(np.mean(pairwise)),
                     "pairwise_sd": float(np.std(pairwise)),
                     "pairwise_min": float(np.min(pairwise)),
                     "pairwise_max": float(np.max(pairwise)),
                     "argmax_disagreement_mean": float(np.mean(argmax_dis)),
                     "r_R_token_centered": r_R,
                     "battery_contested_fraction": contested},
        "ensemble": {"r_E_token_centered": r_E,
                     "r_E_runs_mean": float(np.mean(r_E_runs)),
                     "r_E_runs_max": float(np.max(r_E_runs)),
                     "E_acc": E_acc, "token_acc": tok_acc,
                     "acc_mean": float(np.mean(accs)),
                     "acc_min": float(np.min(accs)), "acc_max": float(np.max(accs))},
        "etiology": {"delta_token": delta_token,
                     "delta_mean": float(np.mean(delta_runs)),
                     "delta_min": float(np.min(delta_runs)),
                     "delta_max": float(np.max(delta_runs)),
                     "delta_ensemble": delta_E,
                     "spec_typical_error": spec_typ,
                     "idiosyncratic_error": idio_err,
                     "idiosyncratic_correctness": idio_cor,
                     "false_accepts": false_accepts,
                     "true_rejects": true_rejects,
                     "loo_idiosyncrasy_mean": float(np.mean(loo)),
                     "loo_idiosyncrasy_max": float(np.max(loo))},
        "dose_response": dose,
        "margin_structure": mstruct,
        "tau_sensitivity": tau_sens,
        "description_lengths": {
            "spec_bytes": len(spec_str.encode()), "seed_bytes": 4,
            "weights_bytes": w_bytes, "trajectory_bytes": traj_bytes,
            "battery_bits": int(len(truth)),
            "battery_bytes": len(truth) / 8.0,
            "spec_string": spec_str},
        "runtime_s": time.time() - t0,
    }
    with open("/home/z/my-project/scripts/a4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== A4 RESULTS =====")
    print(f"variance bound  v: mean={results['variance']['pairwise_mean']:.5f} "
          f"sd={results['variance']['pairwise_sd']:.5f} "
          f"range=[{results['variance']['pairwise_min']:.5f},{results['variance']['pairwise_max']:.5f}]")
    print(f"  contested battery fraction (family not unanimous): {contested:.5f}")
    print(f"  (argmax-level disagreement mean={results['variance']['argmax_disagreement_mean']:.5f})")
    print(f"r_R (token vs re-instantiations): {r_R:.5f}")
    print(f"r_E (token vs auditor ensemble):  {r_E:.5f}   ensemble acc={E_acc:.4f} token acc={tok_acc:.4f}")
    print(f"delta (vs IGM): token={delta_token:.5f} mean={results['etiology']['delta_mean']:.5f} "
          f"ensemble={delta_E:.5f}")
    print(f"decomposition: spec-typical={spec_typ:.5f} idio-error={idio_err:.5f} idio-correct={idio_cor:.5f}")
    print(f"LOO idiosyncrasy: mean={np.mean(loo):.5f} max={np.max(loo):.5f}")
    print(f"description lengths: spec={len(spec_str.encode())}B seed=4B "
          f"weights={w_bytes}B trajectory={traj_bytes}B battery={len(truth)/8.0:.1f}B")
    print("margin structure:", json.dumps(mstruct))
    print("tau sensitivity:", json.dumps(tau_sens))
    print(f"runtime: {results['runtime_s']:.0f}s")
    print("results -> scripts/a4_results.json")


if __name__ == "__main__":
    main()
