#!/usr/bin/env python3
"""probe_v14_recon.py - doc 82's extraction pass: the twin burner's
anatomy, the landing exceptions, the stratum exposures, and the
four-world nesting read from s16_unif25_results.json."""
import json
import numpy as np

res = json.load(open("/home/z/my-project/scripts/s16_unif25_results.json"))
c = res["classification"]
seeds = {int(k): v for k, v in c["BT_seeds"].items()}

print("=== the burner 606 vs the survivors ===")
for s in [606]:
    r = seeds[s]
    print(f"seed {s}: T={r['T_reg']:.4f} b*={r['b_star']:.4f} "
          f"F_late={r['F_late']:+.4f} anchorF={r['anchor_F_late']:+.4f} "
          f"diff={r['F_late']-r['anchor_F_late']:+.4f}")
    print(f"  cum: rep={r['cum']['rep']} rep_wrong={r['cum']['rep_wrong']} "
          f"flip_pass={r['cum']['flip_pass']} "
          f"flip_pass_blind={r['cum']['flip_pass_blind']} "
          f"persist_late_share={r['persist_share_late']}")
surv = [s for s in seeds if s != 606]
for key in ["rep", "rep_wrong", "flip_pass", "flip_pass_blind"]:
    v = [seeds[s]["cum"][key] for s in surv]
    print(f"survivors' {key}: mean {np.mean(v):.1f} "
          f"range [{min(v)}, {max(v)}]")
v = [seeds[s]["persist_share_late"] for s in surv
     if seeds[s]["persist_share_late"] is not None]
print(f"survivors' persist_late_share: mean {np.mean(v):.4f}")

print("\n=== the landing exceptions (flip_pass != flip_pass_blind) ===")
for s in sorted(seeds):
    cu = seeds[s]["cum"]
    if cu["flip_pass"] != cu["flip_pass_blind"]:
        print(f"  seed {s}: pass={cu['flip_pass']} "
              f"blind={cu['flip_pass_blind']} "
              f"(confident-passing flips: "
              f"{cu['flip_pass'] - cu['flip_pass_blind']})")

print("\n=== the four-world verdict map (the nesting) ===")
pair = c["BT4_paired_verdicts"]["per_seed"]
cap12 = sorted(int(k) for k, v in pair.items() if v["cap"] == "BURNED")
mir6 = sorted(int(k) for k, v in pair.items() if v["mirror"] == "BURNED")
twn1 = sorted(int(k) for k, v in pair.items() if v["twin"] == "BURNED")
ctl0 = sorted(int(k) for k, v in pair.items()
              if v["control"] == "BURNED")
print(f"capacity 12: {cap12}")
print(f"mirror 6:    {mir6}  (subset of capacity: "
      f"{set(mir6) <= set(cap12)})")
print(f"twin 1:      {twn1}  (subset of mirror: "
      f"{set(twn1) <= set(mir6)})")
print(f"control 0:   {ctl0}")
extra6 = sorted(set(cap12) - set(mir6))
print(f"the capacity's extra six (the interaction's burns): {extra6}")
print("their b*:", sorted(seeds[s]["b_star"] for s in extra6))
print("their twin verdicts:", [pair[str(s)]["twin"] for s in extra6])

print("\n=== the b* ordering (burn depth by world) ===")
order = sorted(seeds, key=lambda s: -seeds[s]["b_star"])
for s in order[:8]:
    print(f"  seed {s}: b*={seeds[s]['b_star']:.4f} "
          f"cap={pair[str(s)]['cap'][:4]} "
          f"mir={str(pair[str(s)]['mirror'])[:4]} "
          f"twn={pair[str(s)]['twin'][:4]}")

print("\n=== the aggregates ===")
for arm in ["S22_UNIF25", "S22_UNIF25_NOCH"]:
    ag = res["arms"][arm]["aggregate"]
    print(f"{arm}: acc path r1/r6/r14 = "
          f"{ag['acc'][0]:.4f}/{ag['acc'][5]:.4f}/{ag['acc'][-1]:.4f}, "
          f"comp={ag['comp_verdict_final']}, "
          f"F_late_mean={ag['comp_F_late_mean']:+.4f}, "
          f"armed_rds={int(sum(ag['armed']))}")
    rd = ag["realized_dose"]
    print(f"  realized_dose r7..r14: "
          f"{[round(x, 4) for x in rd[6:14]]}")

print("\n=== the arithmetic check (F = anchor + differential) ===")
af = [seeds[s]["anchor_F_late"] for s in seeds]
ff = [seeds[s]["F_late"] for s in seeds]
df = [f - a for f, a in zip(ff, af)]
print(f"anchor mean {np.mean(af):+.4f}, diff mean {np.mean(df):+.4f}, "
      f"armed mean {np.mean(ff):+.4f}")
print(f"606: anchor {seeds[606]['anchor_F_late']:+.4f}, "
      f"diff {seeds[606]['F_late']-seeds[606]['anchor_F_late']:+.4f}, "
      f"armed {seeds[606]['F_late']:+.4f}")
b = {int(k): v for k, v in c["BT5_anchor_decomposition"]["per_seed"].items()}
worst = min(b, key=lambda k: b[k]["anchorF"])
print(f"lowest anchor: seed {worst} at {b[worst]['anchorF']:+.4f} "
      f"(stratum {b[worst]['stratum']})")

print("\n=== BT3 gap detail ===")
bt3 = c["BT3_census"]
print(f"largest gap {bt3['largest_internal_gap']:.4f} at "
      f"{bt3['gap_location']} (bar {bt3['moat_bar']})")
