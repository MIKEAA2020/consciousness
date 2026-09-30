#!/usr/bin/env python3
"""v13 recon probe (pre-registration baselines, read-only):
(i)  doc 73's stored census (s11_seeds24_results.json): the cold-burn
     stratum's anatomy - the burn's round-by-round path (b_r, adm_blind,
     the repairs, the flips, the brake, tau) for the burners against the
     cold non-burners, hot against cold;
(ii) the anchor decomposition (doc 55's stored S5_CAP30_NOCH, seeds
     600-611): armed F_late against anchor F_late - which burns the
     undefended world shares (the subtraction's cancelled component);
(iii) the shared columns (b_clean, iota_star, r_star, D_reg, r_org0's
     proxy) against the cold stratum - what the thermometer does not
     carry that the anchor subtraction cancels.
"""
import json

import numpy as np

CEN = "/home/z/my-project/scripts/s11_seeds24_results.json"
D55 = "/home/z/my-project/scripts/s5_boundaries_results.json"

res = json.load(open(CEN))
runs = res["arms"]["S17_CAP30"]["runs"]
c = res["classification"]
seeds = {rr["seed"]: rr for rr in runs}

deep = [603, 604, 606]
burners = [s for s in seeds
           if float(np.mean([t["comp_F"] for t in seeds[s]["traj"]
                             if t["comp_F"] is not None
                             and t["round"] >= 11])) < -0.02]
Ts = {s: seeds[s]["plogs"][4]["T_reg"] for s in seeds}
body = [s for s in seeds if s not in deep]
Tmed = float(np.median([Ts[s] for s in body]))
cold_burn = [s for s in burners if Ts[s] <= Tmed]
hot_burn = [s for s in burners if Ts[s] > Tmed]
cold_ok = [s for s in body if Ts[s] <= Tmed and s not in burners]

print("=" * 76)
print("(i) the census's strata (24 seeds; burn floor -0.02, late win 11-14)")
print(f"body-median T = {Tmed:.4f}")
print(f"burners ({len(burners)}): {sorted(burners)}")
print(f"  hot  ({len(hot_burn)}): {sorted(hot_burn)}")
print(f"  cold ({len(cold_burn)}): {sorted(cold_burn)}")
print(f"cold non-burners ({len(cold_ok)}): {sorted(cold_ok)}")


def flate(s):
    tr = seeds[s]["traj"]
    return float(np.mean([t["comp_F"] for t in tr
                          if t["comp_F"] is not None
                          and t["round"] >= 11]))


def path(s, key, rounds=range(1, 15)):
    tr = seeds[s]["traj"]
    out = []
    for t in tr:
        if t["round"] in rounds and t.get(key) is not None:
            out.append((t["round"], t[key]))
    return out


def ppath(s, key):
    return [(p["r"], p[key]) for p in seeds[s]["plogs"]
            if p.get(key) is not None]


def row(s, label):
    p5 = seeds[s]["plogs"][4]
    t0 = seeds[s]["traj"][0]
    print(f"{s:>4d} {label:>10s} T={Ts[s]:.4f} b*={p5['b_star_reg']:.4f} "
          f"F_late={flate(s):+.4f} b_cl={p5['b_clean']:.4f} "
          f"iot*={t0['iota_star']:.4f} r*={t0['r_star']:.4f}")


print("\n-- the strata's registered columns --")
for s in sorted(cold_burn):
    row(s, "COLD BURN")
for s in sorted(cold_ok):
    row(s, "cold ok")
for s in sorted(hot_burn):
    row(s, "hot burn")

print("\n-- the burn's round-by-round path: b_r (committed blind mass) --")
groups = {"coldburn": sorted(cold_burn), "coldok": sorted(cold_ok)[:4],
          "hotburn": sorted(hot_burn)[:4]}
for gname, gs in groups.items():
    print(f"  [{gname}]")
    for s in gs:
        br = [round(t["blind_mass"], 4) for t in seeds[s]["traj"]]
        print(f"   {s:>4d} b_r r1-14: {br}")
print("  (the registration b* per seed: "
      f"{ {s: round(seeds[s]['plogs'][4]['b_star_reg'], 4) for s in sorted(burners)} })")

print("\n-- the blind-ADMISSION path (adm_blind, the hot road's carrier) --")
for gname, gs in groups.items():
    print(f"  [{gname}]")
    for s in gs:
        ab = [round(p["adm_blind"], 3) if p["adm_blind"] is not None else None
              for p in seeds[s]["plogs"]]
        print(f"   {s:>4d} adm_bl r1-14: {ab}")

print("\n-- the repairs and the flips (per round, r6+) --")
for gname, gs in groups.items():
    print(f"  [{gname}]")
    for s in gs:
        rep = [(t["round"], t["n_disputed_repaired"], t["n_flipped"],
                t["n_gate_flipped"], t["n_bulk_flipped"],
                t["n_flip_disputed"])
               for t in seeds[s]["traj"] if t["round"] >= 6]
        print(f"   {s:>4d} (r, rep, flip, gate, bulk, flipDIS): {rep}")

print("\n-- the stream's error content: cerr pre/post repair (r6+) --")
for gname, gs in groups.items():
    print(f"  [{gname}]")
    for s in gs:
        ce = [(t["round"], round(t["cerr_stream_pre"], 4),
               round(t["cerr_stream_post"], 4))
              for t in seeds[s]["traj"] if t["round"] >= 6]
        print(f"   {s:>4d} (r, pre, post): {ce}")

print("\n-- the gate's machinery: brake, flagged_frac, tau (r6+) --")
for gname, gs in groups.items():
    print(f"  [{gname}]")
    for s in gs:
        gm = [(t["round"], int(t["brake"]), round(t["flagged_frac"], 3),
               round(t["tau_sys"], 3), t["n_commit"])
              for t in seeds[s]["traj"] if t["round"] >= 6]
        print(f"   {s:>4d} (r, brk, flag, tau, ncom): {gm}")

print("=" * 76)
print("(ii) the anchor decomposition (doc 55's stored anchor, seeds 600-611)")


def seedcomp(rr, latewin=4):
    n = len(rr["traj"])
    late = [t["comp_F"] for t in rr["traj"]
            if t["comp_F"] is not None and t["round"] >= n - latewin + 1]
    return float(np.mean(late)) if late else None


d55 = json.load(open(D55))
amap = {rr["seed"]: rr for rr in d55["arms"]["S5_CAP30_NOCH"]["runs"]}
print("seed  stratum     anchorF  armedF   diff     T")
for s in sorted(seeds):
    if s not in amap:
        continue
    af = seedcomp(amap[s])
    arf = flate(s)
    strat = ("COLD-BURN" if s in cold_burn else
             "hot-burn" if s in hot_burn else
             "cold-ok" if s in cold_ok else
             "deep" if s in deep else "body")
    print(f"{s:>4d}  {strat:>10s}  {af:+.4f}  {arf:+.4f}  "
          f"{arf - af:+.4f}  {Ts[s]:.4f}")

print("=" * 76)
print("(iii) the shared columns against the cold stratum (all 24 seeds)")
cols = {}
for s in seeds:
    p5 = seeds[s]["plogs"][4]
    t0 = seeds[s]["traj"][0]
    cols[s] = {
        "b_clean": p5["b_clean"], "iota_star": t0["iota_star"],
        "r_star": t0["r_star"], "D_reg": p5["D_reg"],
        "bpool5": p5["b_pool"], "L5": p5["lift_L"],
        "rorg0": seeds[s]["traj"][5].get("r_org"),
        "acc0": seeds[s]["traj"][0]["acc"],
    }


def groupstats(gs, key):
    v = [cols[s][key] for s in gs if cols[s][key] is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


for key in cols[600].keys():
    mcb, scb = groupstats(cold_burn, key)
    mco, sco = groupstats(cold_ok, key)
    mhb, shb = groupstats(hot_burn, key)
    if mcb is None:
        continue
    print(f"  {key:>10s}: cold-burn {mcb:.4f}+-{scb:.4f}  "
          f"cold-ok {mco:.4f}+-{sco:.4f}  hot-burn {mhb:.4f}+-{shb:.4f}")
print("\nstrata sizes:", len(cold_burn), len(cold_ok), len(hot_burn),
      len([s for s in seeds if s not in
           cold_burn + cold_ok + hot_burn + deep]), "deep")
