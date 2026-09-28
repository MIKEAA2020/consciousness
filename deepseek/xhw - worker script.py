#!/usr/bin/env python3
"""
XHW worker - trains the P1 pilot design under whatever environment it inherits
(CORETYPE / threads / precision are set by the orchestrating parent BEFORE
this process imports numpy). Writes results to an .npz + .json in the out dir.

Modes:
  token <seed> <outdir> [--f32]       one run, per-epoch weight snapshots
  family <outdir> <seed...>           many runs, battery probs per seed
  ulp <seed> <outdir> <at_epoch> <which>   token + 1-ulp-perturbed twin,
                                      per-epoch delta norms (same process,
                                      same env; the ONLY difference is 1 ulp
                                      in one weight at <at_epoch>)
"""
import json
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

EPOCHS, BATCH, LR, H1, H2, TAU = 60, 32, 1e-3, 128, 64, 0.9
SPLIT_SEED = 12345


def init_params(rng):
    Ws = [rng.normal(0.0, np.sqrt(2.0 / 64), (64, H1)),
          rng.normal(0.0, np.sqrt(2.0 / H1), (H1, H2)),
          rng.normal(0.0, np.sqrt(2.0 / H2), (H2, 10))]
    bs = [np.zeros(H1), np.zeros(H2), np.zeros(10)]
    return Ws, bs


def forward(X, Ws, bs):
    z1 = X @ Ws[0] + bs[0]; a1 = np.maximum(z1, 0.0)
    z2 = a1 @ Ws[1] + bs[1]; a2 = np.maximum(z2, 0.0)
    return z1, a1, z2, a2, a2 @ Ws[2] + bs[2]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(seed, Xtr, ytr, snap_epochs=None, ulp_at=None, dtype=np.float64,
          lr=LR, epochs=EPOCHS):
    """ulp_at = (epoch, wi, i, j): flip W[wi][i,j] by +1 ulp at epoch start."""
    rng = np.random.default_rng(seed)
    Xtr = Xtr.astype(dtype); ytr = ytr.astype(np.int64)
    Ws, bs = init_params(rng)
    Ws = [w.astype(dtype) for w in Ws]
    m = [np.zeros_like(w) for w in Ws]; v = [np.zeros_like(w) for w in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
    b1a, b2a, eps, t = 0.9, 0.999, 1e-8, 0
    n = len(Xtr)
    snaps = {}
    if snap_epochs and 0 in snap_epochs:
        snaps[0] = np.concatenate([w.ravel() for w in Ws] + [b.ravel() for b in bs])
    for ep in range(epochs):
        if ulp_at is not None and ep == ulp_at[0]:
            _, wi, i, j = ulp_at
            Ws[wi][i, j] = np.nextafter(Ws[wi][i, j].astype(np.float64),
                                        np.inf).astype(dtype)
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
            gW, gb = [gW1, gW2, gW3], [gb1, gb2, gb3]
            t += 1
            for k in range(3):
                m[k] = b1a * m[k] + (1 - b1a) * gW[k]
                v[k] = b2a * v[k] + (1 - b2a) * gW[k] ** 2
                Ws[k] -= LR * (m[k] / (1 - b1a ** t)) / (np.sqrt(v[k] / (1 - b2a ** t)) + eps)
                mb[k] = b1a * mb[k] + (1 - b1a) * gb[k]
                vb[k] = b2a * vb[k] + (1 - b2a) * gb[k] ** 2
                bs[k] -= LR * (mb[k] / (1 - b1a ** t)) / (np.sqrt(vb[k] / (1 - b2a ** t)) + eps)
        if snap_epochs and (ep + 1) in snap_epochs:
            snaps[ep + 1] = np.concatenate(
                [w.ravel() for w in Ws] + [b.ravel() for b in bs])
    return Ws, bs, snaps, epochs


def load_data():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return Xtr, ytr, Xte, yte


def probs_of(Ws, bs, X, dtype=np.float64):
    _, _, _, _, z3 = forward(X.astype(dtype), Ws, bs)
    return softmax(z3)


def main():
    mode = sys.argv[1]
    if mode == "token":
        seed, outdir = int(sys.argv[2]), sys.argv[3]
        f32 = "--f32" in sys.argv
        dtype = np.float32 if f32 else np.float64
        Xtr, ytr, Xte, yte = load_data()
        Ws, bs, snaps, _ = train(seed, Xtr, ytr,
                                 snap_epochs=list(range(0, EPOCHS + 1)), dtype=dtype)
        P = probs_of(Ws, bs, Xte, dtype)
        np.savez(outdir + "/token.npz",
                 snaps=np.stack([snaps[e] for e in range(EPOCHS + 1)]),
                 probs=P.astype(np.float64), seed=seed, f32=f32)
        print("token done", seed, "f32" if f32 else "f64")

    elif mode == "family":
        outdir = sys.argv[2]
        seeds = [int(x) for x in sys.argv[3].split(",")]
        Xtr, ytr, Xte, yte = load_data()
        P = {}
        for s in seeds:
            Ws, bs, _, _ = train(s, Xtr, ytr)
            P[s] = probs_of(Ws, bs, Xte)
        np.savez(outdir + "/family.npz",
                 probs=np.stack([P[s] for s in seeds]),
                 seeds=np.array(seeds))
        print("family done", len(seeds))

    elif mode == "ulp":
        seed, outdir, at_epoch, which = (
            int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
        wi, i, j = (int(x) for x in which.split(":"))
        lr = float(sys.argv[6]) if len(sys.argv) > 6 else LR
        epochs = int(sys.argv[7]) if len(sys.argv) > 7 else EPOCHS
        Xtr, ytr, Xte, yte = load_data()
        Ws0, bs0, _, _ = train(seed, Xtr, ytr, lr=lr, epochs=epochs)
        P0 = probs_of(Ws0, bs0, Xte)
        Ws1, bs1, snaps1, _ = train(seed, Xtr, ytr,
                                    ulp_at=(at_epoch, wi, i, j),
                                    snap_epochs=list(range(0, epochs + 1)),
                                    lr=lr, epochs=epochs)
        P1 = probs_of(Ws1, bs1, Xte)
        # per-epoch delta vs the unperturbed trajectory: retrain with snaps
        _, _, snaps0, _ = train(seed, Xtr, ytr,
                                snap_epochs=list(range(0, epochs + 1)),
                                lr=lr, epochs=epochs)
        S0 = np.stack([snaps0[e] for e in range(epochs + 1)])
        S1 = np.stack([snaps1[e] for e in range(epochs + 1)])
        delta = np.linalg.norm(S1 - S0, axis=1)
        wnorm = np.linalg.norm(S0, axis=1)
        d_verdicts = float(np.mean((P0 >= TAU) != (P1 >= TAU)))
        d_argmax = float(np.mean(P0.argmax(1) != P1.argmax(1)))
        np.savez(outdir + f"/ulp_{at_epoch}_{which.replace(':', '-')}_lr{lr}_ep{epochs}.npz",
                 delta=delta, wnorm=wnorm,
                 d_verdicts=d_verdicts, d_argmax=d_argmax)
        print(f"ulp done at={at_epoch} {which} lr={lr} ep={epochs} "
              f"d_verdicts={d_verdicts:.6f} final_rel={delta[-1]/wnorm[-1]:.2e}")


if __name__ == "__main__":
    main()
