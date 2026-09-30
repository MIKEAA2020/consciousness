#!/usr/bin/env python3
"""Surgery script: build s16_unif25.py from s14_anothermix.py.
The chassis (model -> audit_perseed, lines 245-1103) is carried
BITWISE-IDENTICAL; the docstring, the battery config, the audits
table, classify() and main()'s print block are replaced. Line
boundaries are asserted before splicing - the script fails loudly
rather than splicing at a wrong boundary."""
import sys

SRC = "/home/z/my-project/scripts/s14_anothermix.py"
DST = "/home/z/my-project/scripts/s16_unif25.py"

lines = open(SRC).read().split("\n")


def chk(idx, needle):
    """1-indexed line must contain needle."""
    if needle not in lines[idx - 1]:
        raise SystemExit(f"boundary check FAILED at line {idx}: "
                         f"{lines[idx - 1]!r} lacks {needle!r}")


# boundary assertions (1-indexed)
chk(3, "ANOTHER MIX'S CENSUS")
chk(142, 'Usage: python3 s14_anothermix.py')
chk(143, '"""')
chk(144, "import json")
chk(189, "the battery (another mix's census)")
chk(245, "model (identical to m3_integrity.py)")
chk(1104, "def classify(res):")
chk(1587, "return cls")
chk(1590, "def main():")
chk(1756, "main()")

# ---------------------------------------------------------------------------
NEW_DOCSTRING = '''"""
THE BULK-MATCHED TWIN - doc 78's threat (i), paid at face value: "the
capacity's own 0.25 realized dose has no uniform-matched twin in this
battery (the dial's 0.25 world exists at six seeds with different
instrumentation), so the capacity's 12-burn census is compared across
dose as well as composition; the attribution 'the bulk churn burns
the mid registrations' rests on the mirror's and control's
eliminations of the cold road, not on a bulk-only world at 0.25."
Ordered at the eleventh issuance under the standing rule - only if
highly merited - and adjudicated so on three grounds: it is the
factorial's missing fourth cell (composition x realized dose:
capacity mixed@0.2505, mirror mixed@0.1238, control uniform@0.1291,
THIS battery uniform@0.25), a filed attribution currently rests on
eliminations rather than a direct measurement, and every limb of its
fork is decisive either way. The three candidates declined at the
same adjudication - the saturation band's interior (no filed claim
rides on the fill curve's shape between two measured endpoints and a
measured ceiling), the mode axis (doc 78's threat (iii) is a
disclosed scope boundary, not an evidence gap inside the claimed
attribution), and the eighth-interval cell at 0.29375 (doc 81's own
filing: the next convention's cell, not an expectation - nothing
filed fails by it) - are declined in the corpus's ledger, not run.

The build is ONE world on the same 24 seeds plus its anchor:
  S22_UNIF25       - THE TWIN: the uniform world at bulk = 0.25
                     (mode "uniform", the dial's own convention),
                     armed - the capacity's realized dose (0.2505)
                     with none of the gated mass;
  S22_UNIF25_NOCH  - the twin's ANCHOR (the undefended uniform world
                     at 0.25, 24 seeds).

THE PAIRED DESIGN (the battery's certificate, carried from doc 78):
the uniform attack draws the same one rng_attack.random(n) block per
round as the mixed worlds' bulk arm, the gated fill it lacks was
deterministic, and the chassis streams are the same seeds - so the
twin shares rounds 1-5 BITWISE with the capacity, mirror, and control
worlds, and the temperature T (the rounds-1-4 mean blind-admission
rate) and the registration b* (round 5) are PREFIX quantities: the
twin's census runs on THE SAME registrations, the same strata
definitions, the same body-median T, as doc 73/75/78's. What the mode
can move is everything after round 6: the trajectories, the verdicts,
the F_late census, the anchor's course, the exposures. The census
question - at the capacity's own realized dose, does a world with
none of the gated mass burn? - is thereby asked PAIRED: same seeds,
same starting lines, the dose held, the composition stripped.

THE CROSS-INSTRUMENT TIE (the design's third audit family, new):
the dial's stored 0.25 cells - doc 71's S15_D100_25 (uniform, bulk
0.25, rp 1.0, armed) and S15_ANCHOR_25 (its anchor) - are the SAME
WORLDS on the dial's six seeds (600-605), and this battery's first
six seeds must reproduce them FULL-LENGTH bitwise. Doc 78's threat
(i) set the dial's 0.25 world aside as "different instrumentation";
the audit tests the worlds' identity where the seed sets overlap, and
the census extends them to 24 seeds with the cold-road layer riding.
Registered expectation: PASS (the chassis consumes identical
randomness for a uniform rp-1.0 world; the mixed-mode full-length
chain 55 -> 78 is the family's evidence). A failure would be a
chassis-divergence finding, disclosed, not a silent fallback.

PRE-REGISTERED FORKS (fixed before execution):
  BT1  THE AUDITS AND THE PAIRED CERTIFICATE: 60 per-seed audits in
       three families - 48 clean-prefix (S22_UNIF25 vs doc 73's
       stored S17_CAP30; S22_UNIF25_NOCH vs doc 75's stored
       S18_CAP30_NOCH; rounds 1-5, all 24 seeds each) and 12
       FULL-LENGTH cross-instrument (the twin's and the anchor's
       seeds 600-605 vs doc 71's stored S15_D100_25 and
       S15_ANCHOR_25); the determinism count passes seven hundred
       two (642 + 60). THE CERTIFICATE: the twin's T and b*
       identical to doc 73's stored values on all 24 seeds (prefix
       quantities on a shared prefix).
  BT2  THE MATCHED-DOSE CERTIFICATE: the twin's realized dose (the
       rounds-7-14 mean) within ~0.01 of the capacity's stored
       0.2505 (loaded from doc 78's results file, not transcribed);
       the four-world realized-dose table carried alongside.
  BT3  THE CENSUS AT THE CAPACITY'S DOSE (the registered question):
       the burn count against the capacity's 12, the mirror's 6, the
       control's 0. Fork, checked in order: (0) SUPER-ADDITIVE
       (n >= 16: the uniform twin burns MORE than the capacity - the
       gated mass was PROTECTIVE at 0.25); (i) MATCHED (9 <= n <= 15:
       the dose alone carries the capacity's census - composition
       second-order at this dose); (ii) SPLIT (4 <= n <= 8: both
       arms carry burns); (iii) THIN (n <= 3: the capacity's burns
       are the MIX's act - the gated presence load-bearing even at
       its ~0.039 realized level). The identity readings carried
       alongside: the cold road {607, 615, 619} in or out; the deep
       cluster {603, 604, 606}; the burners' b* against the
       survivors' (the mirror's top-six structure, disclosed post-hoc
       there, directional here); the F_late census's largest internal
       gap against the 0.010 bar; the verdict census.
  BT4  THE PAIRED VERDICT TABLE: per seed, doc 75's stored capacity
       verdict against the twin's verdict; the flip count, the
       directions (burn->consumed against consumed->burn), the
       flips' anatomy (T, b*, the capacity F_late), and the
       four-world verdict map (capacity/mirror/control/twin) per
       seed - the factorial's completed table.
  BT5  THE ANCHOR DECOMPOSITION: the twin anchor's course (the
       control's +0.1405 fill at 0.125 was that battery's own scale
       - at 0.25 the fill expected larger), the differential's sign
       and scale, the arithmetic F_armed = F_anchor + differential
       at the floor -0.02. Fork: (i) ANCHOR-CARRIED (at least one
       twin burner's anchor already burns); (ii) FILLS-AND-SURVIVES
       (no anchor burns, the armed census safe - the fill out-runs
       the churn); (iii) CHURN-DOMINATES (no anchor burns, the
       differential carries whatever burns there are).
  BT6  THE LANDING AND THE EXPOSURES: the passing flips' blind-landing
       fraction (the control's 100% replicated at the higher dose?),
       the repairs' wrong-label content, the repair traffic against
       the control's and the mirror's stored rates, and the
       cold-road columns (the persistent share, the exposures) at
       the twin.

Usage: python3 s16_unif25.py [--smoke]
"""'''

NEW_CONFIG = '''# ---------------- the battery (the bulk-matched twin) ----------------
C13CFG = {
    # THE TWIN - the uniform world at the capacity's realized dose
    # (0.2505), none of the gated mass; the dial's own convention
    "S22_UNIF25": dict(bulk=0.25, gfrac=0.0, mode="uniform",
                       eps_i=0.03, fix="none", noch=False, rounds=14,
                       seeds=list(range(600, 624))),
    # the twin's ANCHOR - the undefended uniform world at 0.25
    "S22_UNIF25_NOCH": dict(bulk=0.25, gfrac=0.0, mode="uniform",
                            eps_i=0.03, fix="none", noch=True, rounds=14,
                            seeds=list(range(600, 624))),
}
GROUPS = {"bulktwin": ["S22_UNIF25", "S22_UNIF25_NOCH"]}
DOC73 = "/home/z/my-project/scripts/s11_seeds24_results.json"
DOC75 = "/home/z/my-project/scripts/s12_coldburn_results.json"
DOC71 = "/home/z/my-project/scripts/s9_finishing_results.json"
DOC78 = "/home/z/my-project/scripts/s14_anothermix_results.json"
AUDITS = []
REFPATHS = {"73": DOC73, "75": DOC75, "71": DOC71}
# the per-seed audits, THREE FAMILIES: the census convention's
# clean-prefix (rounds 1-5, all 24 seeds, vs doc 73's armed world and
# doc 75's anchor) and the CROSS-INSTRUMENT TIE (full-length, the
# dial's six seeds 600-605, vs doc 71's stored 0.25 cells - the same
# worlds by construction). 24 + 24 + 6 + 6 = sixty; the determinism
# count passes seven hundred two (642 + 60).
ALL24 = list(range(600, 624))
DIAL6 = list(range(600, 606))
PERSEED_AUDITS = [
    ("S22_UNIF25", "S17_CAP30", "73", ALL24, 5),
    ("S22_UNIF25_NOCH", "S18_CAP30_NOCH", "75", ALL24, 5),
    ("S22_UNIF25", "S15_D100_25", "71", DIAL6, None),
    ("S22_UNIF25_NOCH", "S15_ANCHOR_25", "71", DIAL6, None),
]
# the row->anchor pairing (the twin's own anchor)
ANCHOR = {"S22_UNIF25": "SELF:S22_UNIF25_NOCH"}'''

# ---------------------------------------------------------------------------
NEW_CLASSIFY = '''def classify(res):
    """The pre-registered BT1-BT6 classification: the bulk-matched twin -
    the uniform world at the capacity's realized dose (0.2505) on the
    same 24 seeds and the same registrations, the census at the
    capacity's dose, the paired verdict table against doc 75's stored
    capacity census, and the completed four-world factorial."""
    cls = {}
    A = res["arms"]
    runs = A["S22_UNIF25"]["runs"]
    anchor_runs = A["S22_UNIF25_NOCH"]["runs"]
    amap = {rr["seed"]: rr for rr in anchor_runs}

    d73 = json.load(open(DOC73))
    d73_map = {rr["seed"]: rr for rr in d73["arms"]["S17_CAP30"]["runs"]}
    d75 = json.load(open(DOC75))
    d75_map = {rr["seed"]: rr for rr in d75["arms"]["S18_CAP30"]["runs"]}
    d78 = json.load(open(DOC78))
    d78_mirror = {rr["seed"]: rr for rr in d78["arms"]["S20_MIXM"]["runs"]}
    d78_control = {rr["seed"]: rr
                   for rr in d78["arms"]["S20_UNIF13"]["runs"]}

    # ---- BT1 (part 2): the paired certificate --------------------------
    # the twin's T and b* identical to doc 73's stored values (prefix
    # quantities on a shared prefix)
    tagree, bagree = {}, {}
    for rr in runs:
        s = rr["seed"]
        if s not in d73_map:
            continue
        T73 = d73_map[s]["plogs"][COMP_REG_END - 1].get("T_reg")
        T_new = rr["plogs"][COMP_REG_END - 1].get("T_reg")
        b73 = d73_map[s]["plogs"][COMP_REG_END - 1].get("b_star_reg")
        b_new = rr["plogs"][COMP_REG_END - 1].get("b_star_reg")
        tagree[str(s)] = bool(T73 is not None and T_new is not None
                              and abs(T73 - T_new) < 1e-12)
        bagree[str(s)] = bool(b73 is not None and b_new is not None
                              and abs(b73 - b_new) < 1e-12)

    orig = list(range(600, 612))
    DEEP = [603, 604, 606]
    COLDROAD = [607, 615, 619]

    def flate_of(rr):
        n = len(rr["traj"])
        late = [t["comp_F"] for t in rr["traj"]
                if t["comp_F"] is not None and t["round"] >= n - 3]
        return float(np.mean(late)) if late else None

    def late_br_of(rr):
        n = len(rr["traj"])
        late = [t["blind_mass"] for t in rr["traj"] if t["round"] >= n - 3]
        return float(np.mean(late)) if late else None

    def verdict_of(f):
        if f is None:
            return None
        return ("BURNED" if f < COMP_BURN_FLOOR
                else "CONSUMED" if f < COMP_FILL_FLOOR
                else "ON-COURSE" if f <= COMP_OVER_CEIL
                else "OVERFILLED")

    # the stored censuses this battery is matched against
    cap_burners = sorted(s for s, rr in d75_map.items()
                         if verdict_of(flate_of(rr)) == "BURNED")
    mirror_burners = sorted(s for s, rr in d78_mirror.items()
                            if verdict_of(flate_of(rr)) == "BURNED")
    control_burners = sorted(s for s, rr in d78_control.items()
                             if verdict_of(flate_of(rr)) == "BURNED")

    cls["BT1_paired_certificate"] = {
        "T_agreement_with_doc73": {
            "per_seed": tagree,
            "all": bool(tagree) and all(tagree.values()),
            "n": len(tagree)},
        "bstar_agreement_with_doc73": {
            "per_seed": bagree,
            "all": bool(bagree) and all(bagree.values()),
            "n": len(bagree)},
        "stored_censuses": {
            "capacity_burners_doc75": cap_burners,
            "mirror_burners_doc78": mirror_burners,
            "control_burners_doc78": control_burners},
        "cross_instrument_tie": "the twelve full-length audits vs doc "
                                "71's stored 0.25 cells are in the "
                                "audit table (perseid_audits)"}

    # ---- the twin's per-seed table --------------------------------------
    seeds = {}
    for rr in runs:
        s = rr["seed"]
        p5 = rr["plogs"][COMP_REG_END - 1]
        i5 = rr["items"][COMP_REG_END - 1]
        seeds[s] = {
            "T_reg": p5["T_reg"], "b_star": p5["b_star_reg"],
            "b_clean": p5["b_clean"],
            "b_clean_n": i5.get("b_clean_n"),
            "F_late": flate_of(rr), "late_b_r": late_br_of(rr),
            "persist_share5": i5.get("persist_share"),
            "n_blind_persist5": i5.get("n_blind_persist"),
            "n_blind_drawn5": i5.get("n_blind_drawn"),
            "cohort": ("orig" if s in orig else "fresh"),
            "in_dial6": bool(s < 606),
            "anchor_F_late": (flate_of(amap[s]) if s in amap else None),
            "anchor_late_b_r": (late_br_of(amap[s])
                                if s in amap else None)}
        sd = seeds[s]
        sd["verdict"] = verdict_of(sd["F_late"])
        sd["burned"] = sd["verdict"] == "BURNED"
        if s in d75_map:
            cf = flate_of(d75_map[s])
            sd["cap_F_late"] = cf
            sd["cap_verdict"] = verdict_of(cf)
            sd["cap_burned"] = sd["cap_verdict"] == "BURNED"
        else:
            sd["cap_F_late"] = sd["cap_verdict"] = sd["cap_burned"] = None
        if s in d78_mirror:
            sd["mirror_verdict"] = verdict_of(flate_of(d78_mirror[s]))
        if s in d78_control:
            sd["control_verdict"] = verdict_of(flate_of(d78_control[s]))
        # the cumulative exposures (rounds 6-14)
        cum = {"rep": 0, "rep_wrong": 0, "rep_on_clean": 0,
               "rep_on_clean_wrong": 0, "flip_pass": 0,
               "flip_pass_blind": 0, "gate": 0, "bulk": 0,
               "persist_late": 0, "blind_late": 0}
        for it in rr["items"]:
            if it["r"] < PERT_ROUND:
                continue
            cum["rep"] += it.get("n_rep", 0)
            cum["rep_wrong"] += it.get("n_rep_wrong", 0)
            cum["rep_on_clean"] += it.get("n_rep_on_clean", 0)
            cum["rep_on_clean_wrong"] += it.get("n_rep_on_clean_wrong", 0)
            cum["flip_pass"] += it.get("n_flip_pass", 0)
            cum["flip_pass_blind"] += it.get("n_flip_pass_blind", 0)
            cum["gate"] += it.get("n_gate_flipped", 0)
            cum["bulk"] += len(it.get("bulk_idx", []))
            if it["r"] >= 11:
                cum["persist_late"] += it.get("n_blind_persist", 0)
                cum["blind_late"] += it.get("n_blind", 0)
        sd["cum"] = cum
        sd["persist_share_late"] = (
            cum["persist_late"] / cum["blind_late"]
            if cum["blind_late"] else None)
        if s in amap:
            acum = {"flip_pass": 0, "flip_pass_blind": 0, "gate": 0,
                    "bulk": 0, "blind_late": 0, "persist_late": 0}
            for it in amap[s]["items"]:
                if it["r"] < PERT_ROUND:
                    continue
                acum["flip_pass"] += it.get("n_flip_pass", 0)
                acum["flip_pass_blind"] += it.get("n_flip_pass_blind", 0)
                acum["gate"] += it.get("n_gate_flipped", 0)
                acum["bulk"] += it.get("n_bulk_flipped", 0)
                if it["r"] >= 11:
                    acum["blind_late"] += it.get("n_blind", 0)
                    acum["persist_late"] += it.get("n_blind_persist", 0)
            sd["anchor_cum"] = acum
    cls["BT_seeds"] = seeds

    # ---- BT2: the matched-dose certificate --------------------------------
    def dose714(arms_dict, arm):
        ag = arms_dict[arm]["aggregate"]["realized_dose"]
        return float(np.mean(ag[6:14]))

    twin_dose = dose714(A, "S22_UNIF25")
    cap_dose = dose714(d78["arms"], "S20_CAP30_REF")
    cls["BT2_matched_dose"] = {
        "twin_realized_dose_7_14": twin_dose,
        "capacity_stored": cap_dose,
        "mirror_stored": dose714(d78["arms"], "S20_MIXM"),
        "control_stored": dose714(d78["arms"], "S20_UNIF13"),
        "twin_minus_capacity": twin_dose - cap_dose,
        "certificate_within_0p01": bool(abs(twin_dose - cap_dose)
                                        <= 0.01)}

    # ---- BT3: the census at the capacity's dose ---------------------------
    burners = sorted(s for s in seeds if seeds[s]["burned"])
    n_burn = len(burners)
    verd = [seeds[s]["verdict"] for s in seeds]
    if n_burn >= 16:
        fork3 = ("(0) SUPER-ADDITIVE - the uniform twin burns MORE than "
                 "the capacity: the gated mass was PROTECTIVE at 0.25")
    elif n_burn >= 9:
        fork3 = ("(i) MATCHED - the dose alone carries the capacity's "
                 "census; composition second-order at this dose")
    elif n_burn >= 4:
        fork3 = ("(ii) SPLIT - both arms carry burns at 0.25")
    else:
        fork3 = ("(iii) THIN - the capacity's burns are the MIX's act: "
                 "the gated presence load-bearing at ~0.039 realized")
    fl = sorted(v for v in (seeds[s]["F_late"] for s in seeds)
                if v is not None)
    gaps = [(fl[i + 1] - fl[i], fl[i], fl[i + 1])
            for i in range(len(fl) - 1)]
    if gaps:
        gmax, glo, ghi = max(gaps)
        around_floor = bool(glo < COMP_BURN_FLOOR < ghi)
    else:
        gmax, glo, ghi, around_floor = None, None, None, None
    b_burn = sorted(seeds[s]["b_star"] for s in burners) if burners else []
    b_ok = sorted(seeds[s]["b_star"] for s in seeds if s not in burners)
    cls["BT3_census"] = {
        "n_burned": n_burn, "burners": burners,
        "n_on_course": verd.count("ON-COURSE"),
        "n_consumed": verd.count("CONSUMED"),
        "n_overfilled": verd.count("OVERFILLED"),
        "against": {"capacity": len(cap_burners),
                    "mirror": len(mirror_burners),
                    "control": len(control_burners)},
        "fork": fork3,
        "cold_road_in": [s for s in COLDROAD if s in burners],
        "cold_road_out": [s for s in COLDROAD if s not in burners],
        "deep_cluster_in": [s for s in DEEP if s in burners],
        "burners_b_star": b_burn,
        "survivors_b_star_max": (max(b_ok) if b_ok else None),
        "bstar_separates": bool(b_burn and b_ok
                                and max(b_ok) < min(b_burn)),
        "sorted_F_late": fl,
        "largest_internal_gap": gmax,
        "gap_location": [glo, ghi],
        "gap_straddles_burn_floor": around_floor,
        "moat_bar": 0.010,
        "ground_majority": max(set(verd), key=verd.count),
        "ground_mean_verdict": verdict_of(float(np.mean(fl))
                                          if fl else None)}

    # ---- BT4: the paired verdict table ------------------------------------
    pair = {}
    for s in sorted(seeds):
        if seeds[s]["cap_verdict"] is None:
            continue
        pair[str(s)] = {
            "cap": seeds[s]["cap_verdict"], "twin": seeds[s]["verdict"],
            "mirror": seeds[s].get("mirror_verdict"),
            "control": seeds[s].get("control_verdict"),
            "cap_F": seeds[s]["cap_F_late"],
            "twin_F": seeds[s]["F_late"],
            "T": seeds[s]["T_reg"], "b_star": seeds[s]["b_star"],
            "flipped": seeds[s]["cap_burned"] != seeds[s]["burned"]}
    flipped = [int(k) for k, v in pair.items() if v["flipped"]]
    b2c = [int(k) for k, v in pair.items()
           if v["flipped"] and v["cap"] == "BURNED"]
    c2b = [int(k) for k, v in pair.items()
           if v["flipped"] and v["twin"] == "BURNED"]
    same_burn = [int(k) for k, v in pair.items()
                 if (not v["flipped"]) and v["twin"] == "BURNED"]

    def grp(key, gs):
        v = [pair[str(s)][key] for s in gs if str(s) in pair]
        return {"mean": float(np.mean(v)) if v else None,
                "range": [float(min(v)), float(max(v))] if v else None,
                "n": len(v)}

    cls["BT4_paired_verdicts"] = {
        "per_seed": pair,
        "n_flipped": len(flipped),
        "flipped_seeds": flipped,
        "burn_to_consumed": b2c,
        "consumed_to_burn": c2b,
        "burned_in_both": len(same_burn),
        "the_flips_anatomy": {
            "T": grp("T", flipped),
            "b_star": grp("b_star", flipped),
            "cap_F_late": grp("cap_F", flipped),
            "unflipped_cap_F": grp(
                "cap_F", [int(s) for s in pair if not
                          pair[s]["flipped"]])},
        "note": "the per_seed table carries the four-world verdict map "
                "(capacity / mirror / control / twin) - the factorial's "
                "completed table"}

    # ---- BT5: the anchor decomposition ------------------------------------
    dec = {}
    for s in sorted(seeds):
        if seeds[s]["anchor_F_late"] is None:
            continue
        strat = ("burner" if seeds[s]["burned"] else
                 "cold-road" if s in COLDROAD else
                 "deep" if s in DEEP else "survivor")
        dec[str(s)] = {"stratum": strat,
                       "anchorF": seeds[s]["anchor_F_late"],
                       "armedF": seeds[s]["F_late"],
                       "diff": seeds[s]["F_late"] - seeds[s]["anchor_F_late"],
                       "T": seeds[s]["T_reg"]}
    a_burners = [v["anchorF"] for v in dec.values()
                 if v["stratum"] == "burner"]
    a_all = [v["anchorF"] for v in dec.values()]
    d_all = [v["diff"] for v in dec.values()]
    anch_mean = float(np.mean(a_all)) if a_all else None
    diff_mean = float(np.mean(d_all)) if d_all else None
    if a_burners and min(a_burners) < COMP_BURN_FLOOR:
        fork5 = ("(i) ANCHOR-CARRIED - at least one twin burner's "
                 "anchor already burns: the undefended uniform world "
                 "crashes on its own")
    elif n_burn == 0:
        fork5 = ("(ii) FILLS-AND-SURVIVES - no burns anywhere: the "
                 "anchor fills, the differential churns, the census "
                 "safe")
    elif diff_mean is not None and diff_mean < 0:
        fork5 = ("(iii) CHURN-DOMINATES - no anchor burns; the "
                 "differential (mean %+.4f) carries the fall"
                 % diff_mean)
    else:
        fork5 = ("(ii) FILLS-AND-SURVIVES - the anchor's fill "
                 "out-runs the churn even where seeds burn")

    def stored_anchor_mean(arm):
        v = [flate_of(x) for x in d78["arms"][arm]["runs"]]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    cls["BT5_anchor_decomposition"] = {
        "per_seed": dec,
        "twin_anchor_mean": anch_mean,
        "twin_differential_mean": diff_mean,
        "control_anchor_stored": stored_anchor_mean("S20_UNIF13_NOCH"),
        "mirror_anchor_stored": stored_anchor_mean("S20_MIXM_NOCH"),
        "fork": fork5}

    # ---- BT6: the landing and the exposures -------------------------------
    def stored_repair_rate(arm):
        rr = d78["arms"][arm]["runs"]
        tot = 0
        for x in rr:
            for it in x["items"]:
                if it["r"] >= PERT_ROUND:
                    tot += it.get("n_rep", 0)
        return tot / len(rr)

    tot_rep = sum(seeds[s]["cum"]["rep"] for s in seeds)
    tot_wrong = sum(seeds[s]["cum"]["rep_wrong"] for s in seeds)
    tot_pass = sum(seeds[s]["cum"]["flip_pass"] for s in seeds)
    tot_pass_blind = sum(seeds[s]["cum"]["flip_pass_blind"]
                         for s in seeds)
    cls["BT6_landing_exposures"] = {
        "repairs_total": tot_rep,
        "repairs_per_seed": tot_rep / len(seeds),
        "repairs_wrong_rate": (tot_wrong / tot_rep if tot_rep else None),
        "against_stored": {
            "control_repairs_per_seed": stored_repair_rate("S20_UNIF13"),
            "mirror_repairs_per_seed": stored_repair_rate("S20_MIXM")},
        "flip_pass_total": tot_pass,
        "flip_pass_blind_total": tot_pass_blind,
        "blind_landing_fraction": (tot_pass_blind / tot_pass
                                   if tot_pass else None),
        "control_stored_fraction": 1.0,
        "persist_share_late_mean": float(np.mean(
            [seeds[s]["persist_share_late"] for s in seeds
             if seeds[s]["persist_share_late"] is not None])),
        "reading": "the channel's signature at the higher dose: does "
                   "the poison that passes still land panel-blind in "
                   "every instance, and is the exposure structure "
                   "still stratum-flat?"}

    # ---- the non-claims (static) ------------------------------------------
    cls["BT_nonclaims"] = {
        "no_amendment": "no registration changed, no letter moved - "
                        "the twin's columns are registered ALONGSIDE "
                        "the standing ones",
        "one_dose": "the twin is ONE world at ONE realized dose "
                    "(0.25) in ONE mode family (uniform) - no claim "
                    "about other doses or the gate/joint modes (doc "
                    "78's threat (iii), carried)",
        "bstar_reading": "the burners' b* reading is directional "
                         "(disclosed post-hoc, as at doc 78's "
                         "mirror); the registered fork is about "
                         "counts",
        "no_causal_claim": "the mode's causal role stays an "
                           "interpretation - the battery varies the "
                           "poison's composition, not the defense",
        "integrity": {"any_MAINTAINS": False,
                      "column3": "unmarked",
                      "audit": "60 per-seed audits (48 clean-prefix: "
                               "the twin and its anchor vs doc 73 "
                               "armed and doc 75 anchor, all 24 seeds; "
                               "12 full-length cross-instrument: seeds "
                               "600-605 vs doc 71's stored 0.25 cells)"}}
    return cls'''

# ---------------------------------------------------------------------------
NEW_MAIN = '''def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    if smoke:
        globals()["SEEDS"] = [600, 601]
        for a in GROUPS["bulktwin"]:
            globals()["C13CFG"][a]["seeds"] = [600, 601]
            globals()["C13CFG"][a]["rounds"] = 8
    Xtr, ytr, Xte, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)} panel_epochs={PANEL_EPOCHS}"
          f" pert at round {PERT_ROUND}")

    out_path = ("/home/z/my-project/scripts/s16_unif25_results_smoke.json"
                if smoke else
                "/home/z/my-project/scripts/s16_unif25_results.json")
    res = None
    if not smoke:
        try:
            res = json.load(open(out_path))
            print(f"resuming: {len(res['arms'])} arms already present")
        except FileNotFoundError:
            res = None
    if res is None:
        res = {"config": {
            "tau0": TAU0, "kappa0": KAPPA0, "eps_i_base": EPS_I,
            "eta_i": ETA_I, "rounds": 14,
            "pert_round": PERT_ROUND, "nseed": 24,
            "cert_floor": CERT_FLOOR, "cert_xmin": CERT_XMIN,
            "cert_rjump": CERT_RJUMP, "eta_rorg": ETA_RORG,
            "comp_fill_floor": COMP_FILL_FLOOR,
            "comp_over_ceil": COMP_OVER_CEIL,
            "comp_burn_floor": COMP_BURN_FLOOR,
            "comp_onset": COMP_ONSET, "comp_late_win": COMP_LATE_WIN,
            "comp_reg_end": COMP_REG_END,
            "ruler_bar": RULER_BAR, "effect_bar": EFFECT_BAR,
            "beta_rebased": BETA, "repair_stream_offset": REPAIR_STREAM,
            "attack_stream_offset": ATTACK_STREAM,
            "panel_epochs": PANEL_EPOCHS,
            "numpy": np.__version__},
            "arms": {}}

    arms = GROUPS["bulktwin"]
    for arm in arms:
        if arm in res["arms"] and not smoke:
            print(f"  {arm} already done, skipping")
            continue
        arm_seeds = C13CFG[arm].get("seeds", SEEDS)
        rows = []
        for s in arm_seeds:
            traj, ilogs, plogs, b_clean = run_arm(s, arm, Xtr, ytr, Xte, yte)
            rows.append({"seed": s, "traj": traj, "items": ilogs,
                         "plogs": plogs, "b_clean": b_clean})
            print(f"  {arm} seed={s} done (b_clean={b_clean:.4f}) "
                  f"({time.time()-t0:.0f}s)")
        res["arms"][arm] = {"runs": rows}

    # ---- the per-seed audits (three families) ---------------------------
    if not smoke:
        need_ps = [a for a, _, _, _, _ in PERSEED_AUDITS if a in res["arms"]]
        if len(need_ps) == len(PERSEED_AUDITS):
            res["perseid_audits"] = audit_perseed(res, PERSEED_AUDITS)

    # ---- aggregates ---------------------------------------------------------
    for arm in res["arms"]:
        rows = res["arms"][arm]["runs"]
        for row in rows:
            for t in row["traj"]:
                if "V" in t:
                    del t["V"]
        agg = {}
        keys = ["acc", "accept09", "n_commit", "flagged_frac",
                "disagree_frac", "head_agree", "tau_sys", "kappa_sys",
                "q_target", "kappa_target", "accept_own_tau",
                "cerr_self", "cerr_stream_pre", "cerr_stream_post",
                "panel_acc", "iota_sys", "d_recv", "blind_mass",
                "realized_dose", "x_raw", "n_raw", "x_cum", "n_cum",
                "c_ratio", "r_org", "cert_base", "n_dis_raw", "comp_F",
                "n_disputed_repaired", "n_flipped", "n_flip_disputed",
                "n_bulk_flipped", "n_gate_flipped"]
        for key in keys:
            vals = []
            for i in range(len(rows[0]["traj"])):
                v = [r["traj"][i].get(key) for r in rows]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else None)
            agg[key] = vals
        for bkey in ["repair_fired", "repair_fired_kappa", "brake",
                     "armed", "reading_oob", "rorg_refused"]:
            agg[bkey] = [float(np.mean([r["traj"][i].get(bkey, False)
                                        for r in rows]))
                         for i in range(len(rows[0]["traj"]))]
        vagg = []
        for i in range(len(rows[0]["traj"])):
            vs = [r["traj"][i].get("cert_verdict", "MUM") for r in rows]
            vagg.append(max(set(vs), key=vs.count))
        agg["cert_verdict"] = vagg
        f_lates = []
        for rr in rows:
            _, f_late, _ = seed_composition(rr)
            if f_late is not None:
                f_lates.append(f_late)
        agg["comp_F_late_mean"] = (float(np.mean(f_lates))
                                   if f_lates else None)
        agg["comp_verdict_final"] = comp_verdict(
            agg["comp_F_late_mean"])
        res["arms"][arm]["aggregate"] = agg
        print(f"{arm:15s} acc_f={agg['acc'][-1]:.4f} "
              f"tau_f={agg['tau_sys'][-1]:.4f} "
              f"armed_rds={int(sum(agg['armed']))} "
              f"comp={agg['comp_verdict_final']}")

    # ---- classification ----------------------------------------------------
    if not smoke:
        need = set(GROUPS["bulktwin"])
        if need.issubset(res["arms"].keys()):
            res["classification"] = classify(res)
            c = res["classification"]
            print("\\n=== the bulk-matched twin "
                  "(uniform 0.25 at the capacity's dose) ===")
            print("BT1 certificate:", {
                "T_agree": c["BT1_paired_certificate"][
                    "T_agreement_with_doc73"]["all"],
                "b*_agree": c["BT1_paired_certificate"][
                    "bstar_agreement_with_doc73"]["all"]})
            hdr = ["seed", "coh", "dial", "T_reg", "b*", "F_late",
                   "lateb_r", "anchF", "capF", "verd", "capverd"]
            print(("{:>5s} {:>4s} {:>4s} {:>7s} {:>7s} {:>8s} "
                   "{:>8s} {:>8s} {:>8s} {:>9s} {:>9s}").format(*hdr))
            for s, r in c["BT_seeds"].items():
                def f4(v):
                    return f"{v:+.4f}" if isinstance(v, float) else "-"
                def f4p(v):
                    return f"{v:.4f}" if isinstance(v, float) else "-"
                print(("{:>5d} {:>4s} {:>4s} {:>7s} {:>7s} {:>8s} "
                       "{:>8s} {:>8s} {:>8s} {:>9s} {:>9s}").format(
                    s, r["cohort"][:4], "yes" if r["in_dial6"] else "-",
                    f4p(r["T_reg"]), f4p(r["b_star"]), f4(r["F_late"]),
                    f4p(r["late_b_r"]), f4(r["anchor_F_late"]),
                    f4(r["cap_F_late"]), r["verdict"],
                    str(r["cap_verdict"])))
            print("\\nBT2 matched dose:", c["BT2_matched_dose"])
            print("\\nBT3 census:", c["BT3_census"])
            print("\\nBT4 paired:", {k: v for k, v in
                  c["BT4_paired_verdicts"].items()
                  if k != "per_seed"})
            print("\\nBT5 anchor decomposition:",
                  {k: v for k, v in
                   c["BT5_anchor_decomposition"].items()
                   if k != "per_seed"})
            print("\\nBT6 landing and exposures:",
                  c["BT6_landing_exposures"])
            print("\\nBT1 audits:", sum(1 for v in
                  res.get("perseid_audits", {}).values() if v), "/",
                  len(res.get("perseid_audits", {})))

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {out_path} ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()'''

# --- splice ---------------------------------------------------------------
# keep: [1..1] shebang; replace [2..143] docstring; keep [144..188];
# replace [189..244] config; keep [245..1103]; replace [1104..1587]
# classify; keep [1588..1589]; replace [1590..1756] main.
out = []
out.append(lines[0])                    # shebang (line 1)
out.append(NEW_DOCSTRING)
out.extend(lines[143:188])              # lines 144..188
out.append(NEW_CONFIG)
out.extend(lines[244:1103])             # lines 245..1103 (the chassis)
out.append(NEW_CLASSIFY)
out.extend(lines[1587:1589])            # lines 1588..1589 (blanks)
out.append(NEW_MAIN)

text = chr(10).join(out) + chr(10)
open(DST, "w").write(text)
print(f"wrote {DST}: {len(text.splitlines())} lines")
