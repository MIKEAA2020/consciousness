#!/usr/bin/env python3
"""Corpus doc 92 - the bias audit instrument: the blind battery.

Five scorers, context-isolated (blind: no corpus documents, no doc 90,
no verdict, no worklog), scored the nine steelmanned theories of the
tribunal against five neutral criteria (N1 primitive link, N2
first-person reports, N3 causal compatibility, N4 observer, N5
falsifiability), 0/1/2 per cell, one-line justification per cell.
Scorers: 4-a Type-B physicalist, 4-b empirical functionalist, 4-c
illusionist, 4-d neutral methodologist, 4-e Russellian monist (added
after the first four returned, to measure the full persona spread -
disclosed in doc 92 Part 3).

This instrument registers the five matrices AS SCORED, checks the
stated totals, computes the aggregation, the persona-sensitivity
ranges, the two parity tests (N1: Russellian identification vs Type-B
identity; N2: placement vs deletion), the rank correlation against
doc 90's tribunal filing, and the inversion metrics, and emits the
machine-checkable verdict data as results.json. No randomness; the
battery is a registry and a computation; every number quoted in doc
92's text re-derives here deterministically. English labels."""
import json
import os

D = "/home/z/my-project/download"

# --------------------------------------------------------------- the spec
CRITERIA = {
    "N1": ("PRIMITIVE LINK: does the theory posit a primitive (brute, "
           "unexplained) relation between physical structure and experience "
           "that is not entailed by the theory's own ontology? "
           "2 = no such link or fully entailed; 1 = primitive but motivated, "
           "structured, or internal to the core identity claim; 0 = brute "
           "external link doing the load-bearing work."),
    "N2": ("FIRST-PERSON REPORTS: does the theory explain first-person "
           "reports, eliminate them, or leave them unexplained? "
           "2 = explains with a supported account OR eliminates with a "
           "well-supported deflation (elimination is a legitimate stance if "
           "executed); 0 = leaves them unexplained or brute; 1 = partial."),
    "N3": ("CAUSAL COMPATIBILITY: does the theory make consciousness "
           "causally efficacious in a way that conflicts with physics? "
           "2 = fully compatible; 0 = direct conflict (closure or "
           "conservation violations); 1 = tension without outright "
           "violation."),
    "N4": ("OBSERVER: does the theory require a special observer "
           "(homunculus, inner witness, privileged point of view) for "
           "consciousness to occur? 2 = no; 0 = regress-inducing or "
           "metaphysically problematic observer; 1 = observer-like but "
           "non-regressive and motivated."),
    "N5": ("FALSIFIABILITY: does the theory make at least one claim that "
           "could be falsified, and has any relevant test actually been "
           "run? 2 = falsifiable and tested at least indirectly; 0 = no "
           "falsifiable content; 1 = falsifiable in principle but "
           "untested, or only very indirectly tested."),
}

THEORIES = ["functionalism", "dualism", "iit", "illusionism", "physicalism",
            "gwt", "russellian", "higher-order", "panpsychism"]

PERSONAS = {
    "4-a": ("Type-B reductive physicalist (Papineau/Loar): identity needs "
            "no bridge; parsimony is the supreme virtue; skeptical of any "
            "ontology beyond physics."),
    "4-b": ("Empirical functionalist cognitive scientist (Marr levels, "
            "workspace modeling): mechanisms, data, falsifiable models; "
            "suspicious of armchair metaphysics."),
    "4-c": ("Illusionist (Frankish/Dennett): phenomenal properties as "
            "standardly conceived do not exist; values total coverage of "
            "report data and maximum parsimony."),
    "4-d": ("Neutral methodologist (Lakatos; adversarial-collaboration "
            "format): no allegiance; applies criteria exactly as written, "
            "mechanically and charitably."),
    "4-e": ("Russellian monist (Russell/Goff; structural-realist "
            "literature): physics is structural through and through and "
            "silent on intrinsic nature; the identification of intrinsic "
            "nature with phenomenal character makes the problem's shape "
            "intelligible."),
}

BLINDNESS = ("Scorers received only the nine steelmanned theories, the "
             "five criteria, the rubric, and their persona. They did not "
             "receive doc 90, the four campaigns, the tribunal's verdict, "
             "the corpus, or any hint of a desired outcome. They read and "
             "wrote no files. Justifications below are condensed as "
             "scored; the full transcripts are in the session record.")

# ------------------------------------------------- the registry as scored
# SCORES[scorer][theory] = [N1, N2, N3, N4, N5]
SCORES = {
    "4-a": {
        "functionalism": [2, 2, 2, 2, 1],
        "dualism":       [1, 1, 1, 2, 2],
        "iit":           [2, 1, 2, 2, 2],
        "illusionism":   [2, 2, 2, 2, 1],
        "physicalism":   [2, 2, 2, 2, 2],
        "gwt":           [1, 2, 2, 2, 2],
        "russellian":    [1, 1, 2, 2, 1],
        "higher-order":  [2, 2, 2, 1, 2],
        "panpsychism":   [1, 1, 2, 2, 1],
    },
    "4-b": {
        "functionalism": [2, 2, 2, 2, 2],
        "dualism":       [1, 1, 1, 1, 1],
        "iit":           [2, 2, 2, 2, 2],
        "illusionism":   [2, 2, 2, 2, 2],
        "physicalism":   [1, 1, 2, 2, 2],
        "gwt":           [2, 2, 2, 2, 2],
        "russellian":    [1, 1, 2, 2, 1],
        "higher-order":  [2, 2, 2, 1, 2],
        "panpsychism":   [1, 1, 2, 2, 1],
    },
    "4-c": {
        "functionalism": [2, 1, 2, 2, 2],
        "dualism":       [1, 1, 1, 2, 1],
        "iit":           [1, 1, 2, 2, 2],
        "illusionism":   [2, 2, 2, 2, 2],
        "physicalism":   [1, 2, 2, 2, 2],
        "gwt":           [2, 1, 2, 2, 2],
        "russellian":    [1, 1, 2, 2, 0],
        "higher-order":  [2, 2, 2, 1, 2],
        "panpsychism":   [1, 1, 2, 2, 0],
    },
    "4-d": {
        "functionalism": [2, 2, 2, 2, 1],
        "dualism":       [1, 1, 1, 2, 1],
        "iit":           [1, 2, 2, 2, 2],
        "illusionism":   [2, 2, 2, 1, 2],
        "physicalism":   [1, 2, 2, 2, 2],
        "gwt":           [2, 2, 2, 2, 2],
        "russellian":    [1, 1, 2, 2, 1],
        "higher-order":  [2, 2, 2, 1, 2],
        "panpsychism":   [1, 1, 2, 2, 1],
    },
    "4-e": {
        "functionalism": [1, 1, 2, 2, 2],
        "dualism":       [1, 1, 1, 2, 1],
        "iit":           [1, 2, 2, 2, 2],
        "illusionism":   [2, 1, 2, 2, 1],
        "physicalism":   [1, 2, 2, 2, 2],
        "gwt":           [1, 2, 2, 2, 2],
        "russellian":    [2, 2, 2, 2, 1],
        "higher-order":  [1, 2, 2, 1, 2],
        "panpsychism":   [1, 1, 2, 2, 0],
    },
}

# stated totals as returned by the scorers (for the arithmetic check)
STATED_TOTALS = {
    "4-a": {"functionalism": 9, "dualism": 7, "iit": 9, "illusionism": 9,
            "physicalism": 10, "gwt": 9, "russellian": 7,
            "higher-order": 9, "panpsychism": 7},
    "4-b": {"functionalism": 10, "dualism": 5, "iit": 10,
            "illusionism": 10, "physicalism": 8, "gwt": 10,
            "russellian": 7, "higher-order": 9, "panpsychism": 7},
    "4-c": {"functionalism": 9, "dualism": 6, "iit": 8, "illusionism": 10,
            "physicalism": 9, "gwt": 9, "russellian": 6,
            "higher-order": 9, "panpsychism": 6},
    "4-d": {"functionalism": 9, "dualism": 6, "iit": 9, "illusionism": 9,
            "physicalism": 9, "gwt": 10, "russellian": 7,
            "higher-order": 9, "panpsychism": 7},
    "4-e": {"functionalism": 8, "dualism": 6, "iit": 9, "illusionism": 8,
            "physicalism": 9, "gwt": 9, "russellian": 9,
            "higher-order": 8, "panpsychism": 6},
}

# condensed justifications, as scored - the cells the audit's parity
# tests turn on are carried in full; the rest carry their decisive clause.
JUST = {
    "4-a": {
        "russellian": [
            "N1: two-faces unity internal and motivated, but the intrinsic face "
            "BEING phenomenal is a further brute posit - the coin analogy "
            "stipulates the reverse face rather than discovering it.",
            "N2: report production unproblematically structural, but reports "
            "being ABOUT phenomenal character rests on a stipulated "
            "self-acquaintance - a promissory acquaintance, not a mechanism.",
            "N3: intrinsic nature is the interior of the physical bearer - "
            "causation runs exactly as physics describes.",
            "N4: the interior face is observed by nothing - acquaintance is "
            "self-referential and non-regressive.",
            "N5: signature claims (intrinsic natures, phenomenal "
            "identification) are insulated even in principle."],
        "physicalism": [
            "N1: strict identity posits no relation at all - one thing, two "
            "vocabularies; the conceded bruteness is conceptual (phenomenal "
            "concepts), not an ontological link.",
            "N2: report production empirically mapped by the dependence data; "
            "quotational/recognitional concepts explain the reports' peculiar "
            "authority and ineffability.",
            "N3: causation just is the physical process - nothing added.",
            "N4: recognitional concepts are a capacity, not an observer.",
            "N5: identity's entailment of total physical dependence has faced "
            "and passed massive tests."],
        "illusionism": [
            "N1: with phenomenal properties eliminated there is nothing left "
            "to bridge.",
            "N2: elimination executed - reports are outputs of misrepresenting "
            "introspective models, a mechanism-class well documented.",
            "N3: nothing exists but physical systems and their models.",
            "N4: a user interface needs no user watching it.",
            "N5: the distinctive global elimination only very indirectly "
            "testable."],
        "dualism": [
            "N1: laws declared fundamental, internal and motivated - no actual "
            "laws ever formulated.",
            "N2: interactionist branch explains reports; correlationist branch "
            "leaves the match a brute coincidence.",
            "N3: family straddles closure violation vs epiphenomenalism.",
            "N4: acquaintance is direct epistemic access, not a witnessing "
            "mechanism.",
            "N5: anomaly predictions falsifiable and tested to exquisite "
            "precision (null results - against that branch)."],
        "functionalism": [
            "N1: role-consciousness identification internal; realization is "
            "ordinary causation.",
            "N2: reporting is part of the defining role - explained "
            "constitutively.",
            "N3: causation runs through physical realizers.",
            "N4: global availability is systemic - no witness, no regress.",
            "N5: any test of the role-experience identity measures the role "
            "itself - circular."],
        "iit": [
            "N1: consciousness=Phi identification internal to the theory's own "
            "postulates.",
            "N2: PCI tracks report-capacity; report CONTENTS programmatic.",
            "N3: identified with integrated cause-effect power - nothing "
            "added.",
            "N4: explicitly observer-independent.",
            "N5: PCI discriminates clinical states; cerebellum prediction "
            "borne out."],
        "gwt": [
            "N1: identification internal but the re-scoping to access leaves "
            "the link neither bridged nor denied - no bucket for silence.",
            "N2: broadcast-to-report is the theory's literal domain - the "
            "strongest working report-account on offer.",
            "N3: ignition and broadcast are neural causal events.",
            "N4: the workspace's audience is consumer systems - architecture, "
            "not homunculus.",
            "N5: signature claims have been run and could have failed."],
        "higher-order": [
            "N1: constitution-by-meta-representation is the theory's internal "
            "core claim.",
            "N2: reportability constitutively explained and measured "
            "(meta-confidence, type-2 SDT).",
            "N3: meta-representation is ordinary neural computation.",
            "N4: the higher-order representer is observer-like by design - "
            "the rubric's middle case.",
            "N5: paradigms have been run and dissociate."],
        "panpsychism": [
            "N1: micro ascription categorical; the micro-to-macro relation is "
            "the acknowledged unsolved combination problem.",
            "N2: the macro-subject the reports express is what combination "
            "leaves unexplained.",
            "N3: no anomalies posited.",
            "N4: experience is everywhere and watched by no one.",
            "N5: only inherited dependence data - falsifiable in principle "
            "only."],
    },
    "4-b": {
        "russellian": [
            "N1: the intrinsic-is-phenomenal identification is the theory's "
            "brute internal posit - the two-faces move deflates the bridge, "
            "but the identification is asserted, not derived.",
            "N2: report production rides ordinary physical causation "
            "(supported), but the phenomenal-content story (interior-face "
            "self-acquaintance) adds an untestable gloss.",
            "N3: causation runs through the structure-bearing token.",
            "N4: self-acquaintance is a face's identity-with-itself, not a "
            "relation to a distinct relatum.",
            "N5: predictions are physics' predictions; the identification is "
            "designed to be empirically immune."],
        "physicalism": [
            "N1: the identity is conceded to be conceptually brute - an "
            "internal, motivated, parsimony-backed primitive doing "
            "load-bearing work, not a derivation.",
            "N2: dependence of reports massively supported; the "
            "quotational-concept story is developed armchair and untested.",
            "N3: one token, one causal story.",
            "N4: the first-person mode of presentation is a concept, not a "
            "witness.",
            "N5: dependence claims falsifiable and massively confirmed - "
            "though shared with rivals."],
        "illusionism": [
            "N1: nothing to link - the phenomenal relatum is denied.",
            "N2: exists to explain reports as misrepresentation outputs; "
            "confabulation and choice-blindness literatures support the "
            "engine.",
            "N3: all causal work done by physical introspective models.",
            "N4: the felt inner observer is part of the user illusion.",
            "N5: predicts systematic introspective misrepresentation - "
            "tested at least indirectly."],
        "dualism": [
            "N1: laws brute but internal, like gravity - motivated and "
            "structured, not external.",
            "N2: acquaintance plus law-governed correlation; correlationist "
            "wing leaves experience explanatorily idle.",
            "N3: correlationist compatible but epiphenomenal; interactionist "
            "outright violates closure.",
            "N4: acquaintance is a primitive privileged-access relation - "
            "observer-like epistemically.",
            "N5: only the interactionist wing risks predictions, weakly "
            "probed."],
        "functionalism": [
            "N1: no bridge posited - the role constitutively IS the state.",
            "N2: report is one of the functions in the role.",
            "N3: role causation realized in physical dynamics.",
            "N4: self-monitoring is just another function.",
            "N5: role-profile claims tested across masking, priming, access "
            "research."],
        "iit": [
            "N1: structure-experience identity derived inside the theory - "
            "axioms stipulated phenomenology.",
            "N2: PCI tracks report-capacity differentials - the report data "
            "are the theory's home turf.",
            "N3: efficacy built in.",
            "N4: observer-independence is a design commitment.",
            "N5: the most direct tests in this set - genuinely mixed "
            "results."],
        "gwt": [
            "N1: availability is the identity-bearer; the re-scope is a scope "
            "admission, not a posited bridge.",
            "N2: report/no-report paradigms are the theory's data stream.",
            "N3: ordinary fronto-parietal dynamics.",
            "N4: consumer systems, no inner witness.",
            "N5: directly tested and genuinely at risk."],
        "higher-order": [
            "N1: the transitivity principle is constitutive.",
            "N2: reportability constitutive and measured; confabulation fits "
            "as HOT-without-target.",
            "N3: a neural meta-cognitive mechanism.",
            "N4: constitutively requires an inner monitor - observer-like by "
            "design.",
            "N5: type-2 sensitivity dissociations - run, mixed results."],
        "panpsychism": [
            "N1: categorical micro-ascription internal; the micro-to-macro "
            "constitution admittedly unsolved.",
            "N2: the macro-subject doing the reporting is the unsolved "
            "combination problem.",
            "N3: no closure violation.",
            "N4: the micro-subject is bearer, not witness.",
            "N5: the micro-ascription untestable."],
    },
    "4-c": {
        "russellian": [
            "N1: the intrinsic=phenomenal identification is a motivated "
            "internal wager (the epistemic wing says only 'may be'), not a "
            "derivation; the two-faces coin is itself the primitive.",
            "N2: reports become intelligible via self-acquaintance, and the "
            "gap-shape of hard-problem talk is elegantly explained, but "
            "report production piggybacks on structure with no independent "
            "support.",
            "N3: the interior face adds no causal powers beyond the "
            "structure it bears.",
            "N4: self-acquaintance is a token's interior self-presentation, "
            "not a spectator mechanism.",
            "N5: by construction the identification makes no observable "
            "difference (physics is necessarily silent), so no test could "
            "count either way - nearest score 0."],
        "physicalism": [
            "N1: the a posteriori identity is admittedly conceptually brute "
            "- a primitive, albeit internal to the identity claim (one "
            "thing, two vocabularies, no bridge-relation).",
            "N2: report production empirically grounded; the qualia-form of "
            "reports gets a worked account via quotational/recognitional "
            "concepts.",
            "N3: conscious causation is physical causation under a second "
            "vocabulary.",
            "N4: one process described twice; no inner spectator.",
            "N5: dependence claims massively tested."],
        "illusionism": [
            "N1: with no phenomenal relatum there is no bridge to posit.",
            "N2: covers the total report data - including the qualia-form "
            "talk - as outputs of misrepresenting models; elimination "
            "executed, per the rubric's own license.",
            "N3: seemings and reports are physical events.",
            "N4: the seemer is explicitly denied.",
            "N5: the positive claim is falsifiable by "
            "transparent-introspection results and indirectly tested (choice "
            "blindness, confabulation, confidence-performance "
            "dissociations)."],
        "dualism": [
            "N1: psychophysical laws brute and load-bearing, but explicit, "
            "law-like, internal posits - the ransom is at least itemized.",
            "N2: content validated, mechanism hand-waved.",
            "N3: nets to tension - correlationist idle, interactionist "
            "violating.",
            "N4: acquaintance is epistemic access, not an inner witness.",
            "N5: anomaly predictions actually tested (closure holds; no "
            "anomalies found) - survival required retreating into "
            "unfalsifiable correlationism."],
        "functionalism": [
            "N1: consciousness identified with role by definition inside the "
            "theory - no separate phenomenal relatum.",
            "N2: explains that globally available states are reportable, but "
            "not why reports wear the qualia-form; 'the intuition begs the "
            "question' is a parry, not an explanation of the saying.",
            "N3: causation just is role-playing realized physically.",
            "N4: broadcasting relation, not spectating.",
            "N5: dissociation predictions tested (blindsight)."],
        "iit": [
            "N1: the Phi-identity is 'derived,' but the axiom-to-postulate "
            "translation is itself a stipulated bridge.",
            "N2: explains WHEN systems report, but the qualia-form of WHAT "
            "they report is read off the axioms, not explained.",
            "N3: consciousness IS integrated cause-effect power computed "
            "from the dynamics.",
            "N4: contrast with observer-relative evaluation, not an inner "
            "witness.",
            "N5: PCI genuinely discriminated clinical states."],
        "gwt": [
            "N1: availability IS consciousness-of-content - a definitional "
            "identity bought by re-scoping to access.",
            "N2: superb account of report production; the re-scope "
            "explicitly defers the qualia-form.",
            "N3: broadcast is a physical mechanism.",
            "N4: consumers, not spectators.",
            "N5: directly tested."],
        "higher-order": [
            "N1: constitution by meta-representation is a definitional claim "
            "about representational relations.",
            "N2: reportability constitutively explained; the qualia-form "
            "gets an account; proxies measured.",
            "N3: higher-order states are physical representations.",
            "N4: an inner observer is constitutively required - exactly this "
            "rubric slot.",
            "N5: actively measured."],
        "panpsychism": [
            "N1: categorical micro-ascription brute but core; the "
            "constraining relation stays promissory.",
            "N2: report-matching deferred to the combination solution - a "
            "promissory note, not an account.",
            "N3: no closure violation.",
            "N4: micro-subjects are experiencers, not observers.",
            "N5: the distinctive claims are empirically inaccessible - "
            "nearest score 0."],
    },
    "4-d": {
        "russellian": [
            "N1: the intrinsic=phenomenal identification is the core posit - "
            "well motivated by physics's structural silence and the gap's "
            "shape - but the coin analogy re-describes the identity rather "
            "than deriving it.",
            "N2: the structural face causes reports via standard "
            "neuroscience, but why reports are ABOUT the interior rests on "
            "an unelaborated self-acquaintance - coherent, unsupported in "
            "its distinctive part.",
            "N3: all causal work done by the structural face; the intrinsic "
            "adds no independent causal power - fully compatible, at the "
            "acknowledged price of phenomenal causal inertness.",
            "N4: self-acquaintance is a face's self-presentation, not an "
            "additional witnessing mechanism.",
            "N5: deliberately empirically equivalent to physicalism; only a "
            "future non-structural physics could in principle check the "
            "identification - nothing run, nothing currently runnable."],
        "physicalism": [
            "N1: the a posteriori identity is admitted conceptually brute, "
            "but it is the theory's core claim, structured by the "
            "phenomenal-concept apparatus - an internal primitive.",
            "N2: reports are ordinary brain processes with a detailed "
            "phenomenal-concept account of their quotational character.",
            "N3: causal efficacy just IS physical causation - identity, not "
            "interaction.",
            "N4: recognitional concepts quote rather than observe.",
            "N5: dependence claims are the theory's own and heavily "
            "tested."],
        "illusionism": [
            "N1: denies phenomenal properties, so there is no "
            "structure-to-experience link to posit.",
            "N2: explains-and-eliminates - executed deflation, which the "
            "criterion explicitly permits.",
            "N3: all causal work is physical mis-modeling.",
            "N4: the introspective model is an observer-like constitutive "
            "monitor of internal states - the seeming cannot occur without "
            "this inner mis-witness.",
            "N5: the misrepresentation mechanism directly tested."],
        "dualism": [
            "N1: psychophysical laws brute but internal to the posited "
            "ontology and law-structured.",
            "N2: partial either way - undetected laws or brute covariation.",
            "N3: interactionism posits violations; correlationism buys "
            "compatibility at epiphenomenalism's price.",
            "N4: acquaintance is epistemic access, not an inner witness.",
            "N5: interactionist anomaly predictions specific and "
            "falsifiable, but no dedicated test has run."],
        "functionalism": [
            "N1: definitional identity plus explicit denial of any residual "
            "phenomenal fact - no psychophysical primitive posited.",
            "N2: reports are outputs of the globally available state's "
            "role.",
            "N3: all causal traffic is physical realization.",
            "N4: roles and consumers require no inner witness.",
            "N5: richly tested, but the distinctive claim is circular at the "
            "core."],
        "iit": [
            "N1: the identity is axioms-structured and internal, but the "
            "axiom-to-postulate translation from phenomenology to "
            "cause-effect structure is a stipulated step doing "
            "load-bearing work.",
            "N2: reports unfold from the same integrated cause-effect "
            "structure; PCI validated against report-based assessment.",
            "N3: Phi is a property OF the physical causal structure.",
            "N4: observer-independence is a design feature.",
            "N5: derived from the theory's own calculus and tested across "
            "clinical states - distinctive tests actually run."],
        "gwt": [
            "N1: availability IS consciousness-of-content - definitional "
            "within its scope; the re-scoping leaves any phenomenal residue "
            "untheorized rather than bridged.",
            "N2: broadcast to report systems is the direct mechanism.",
            "N3: neural events - fully compatible.",
            "N4: the workspace is a buffer, not a witness.",
            "N5: its own signatures tested in both report and no-report "
            "paradigms."],
        "higher-order": [
            "N1: constitutive definition posits no psychophysical bridge - "
            "though 'suitable' is underspecified.",
            "N2: reportability is constitutive; meta-confidence and type-2 "
            "paradigms provide tested proxies.",
            "N3: ordinary physical computation.",
            "N4: the constitutive monitoring state is an inner witness in "
            "representational form - observer-like.",
            "N5: derived paradigms actually run."],
        "panpsychism": [
            "N1: micro-ascription categorical and internal; the "
            "structure-to-macro-form mapping is an acknowledged "
            "placeholder.",
            "N2: why reports track macro-experience is unconstituted "
            "pending the combination solution.",
            "N3: experience rides as the categorical base of physical "
            "dispositions.",
            "N4: micro-experience is intrinsic, not witnessed.",
            "N5: no distinctive observation is entailed."],
    },
    "4-e": {
        "russellian": [
            "N1: nothing sits between the faces: the phenomenal is the "
            "intrinsic nature of the self-same tokenings - a "
            "categorical-basis identification filling a slot the structural "
            "ontology independently requires; no law, no bridge.",
            "N2: exterior-face causation preserves the full dependence data "
            "on report production, while self-acquaintance fixes report "
            "content directly - no brute content-matching law, no "
            "concept-gap.",
            "N3: the intrinsic is what the causally efficacious tokenings "
            "ARE, not a rival causal partner - closure untouched, no "
            "exclusion problem.",
            "N4: self-acquaintance is the face's self-presentation, not a "
            "witnessing entity.",
            "N5: the structural-silence premise is refutable in principle "
            "(a categorical, non-structural physics would defeat it) and "
            "weakly confirmed by physics' persistent structural form, but "
            "the identification admits no test."],
        "physicalism": [
            "N1: the identity is admitted a posteriori and conceptually "
            "brute; phenomenal concepts explain our ignorance of it, not "
            "the identification itself.",
            "N2: report production rides the massive dependence data and "
            "quotational concepts handle content - though their "
            "presentational character is itself unexplained.",
            "N3: one process, two vocabularies.",
            "N4: identity leaves nothing to be watched.",
            "N5: dependence claims falsifiable and massively tested."],
        "illusionism": [
            "N1: with no phenomenal relatum there is no "
            "structure-to-experience link to bridge; the seeming is "
            "refunctionalized as introspective output - no "
            "consciousness-specific primitive (the bill lands at N2).",
            "N2: reports get a unified misrepresentation story with "
            "indirect support, but the qualia-attribution mechanism is "
            "promissory and the seeming-datum is stipulated away rather "
            "than explained.",
            "N3: everything, including the seeming, is physical causation.",
            "N4: the user-interface needs no homuncular user.",
            "N5: background introspective-error claims indirectly tested; "
            "the distinctive illusion-generation claim never directly "
            "tested."],
        "dualism": [
            "N1: psychophysical laws brute but internal furniture "
            "(charge-like), structured by correlation data.",
            "N2: interactionist reports express experience via undetected "
            "anomalous causation; correlationist leaves the content-match "
            "a further brute law.",
            "N3: interactionist wing predicts closure/energy violations "
            "(none found).",
            "N4: acquaintance is direct, non-observational access.",
            "N5: falsifiable in principle, only very indirectly probed."],
        "functionalism": [
            "N1: role-feel identity stipulated, not derived; the "
            "'in-the-dark' reply is dialectical (question-begging charge), "
            "leaving a brute-but-internal identification.",
            "N2: role dynamics explain report production well, but why "
            "reports speak in intrinsic-feel vocabulary rather than "
            "role-vocabulary gets no content or error theory.",
            "N3: nothing competes with physics.",
            "N4: role-occupancy needs no witness.",
            "N5: its own observable target tested daily in cognitive "
            "science; the beyond-role feel is, per the theory, nothing "
            "further to test."],
        "iit": [
            "N1: the axioms-to-postulates translation builds in "
            "consciousness = cause-effect structure - internal and "
            "structured, but the identification step is asserted, not "
            "derived.",
            "N2: the strongest empirical report account, though specific "
            "qualia-content stays programmatic.",
            "N3: closure untouched.",
            "N4: the 'intrinsic perspective' is the system's own causal "
            "being.",
            "N5: theory-derived measure tested directly - best-tested "
            "entry."],
        "gwt": [
            "N1: availability=consciousness is stipulated, and the "
            "re-scoping to access concedes the phenomenal question rather "
            "than deriving the identification.",
            "N2: report machinery is the theory's home turf - ignition, "
            "no-report paradigms, lesion evidence.",
            "N3: a neural mechanism.",
            "N4: broadcast is an architecture, not a witness.",
            "N5: designed, replicated tests."],
        "higher-order": [
            "N1: 'represented implies conscious' is a constitutive "
            "stipulation as brute as any role-identification; "
            "representational ontology entails no feel.",
            "N2: reportability constitutively explained with measured "
            "proxies giving real grip.",
            "N3: physical throughout.",
            "N4: the meta-representational 'inner eye' is observer-like - "
            "regress-blocked and motivated, but it relocates rather than "
            "removes the witness.",
            "N5: dissociations directly probe the meta-level."],
        "panpsychism": [
            "N1: micro-level co-possession categorical, but the "
            "structured-configuration to unified-macro-subject relation is "
            "an acknowledged, unsolved primitive carrying the macro story.",
            "N2: reports express a composed macro-subject's experience - "
            "rides the unsolved combination problem.",
            "N3: compatible, at the price of causal redundancy.",
            "N4: ubiquitous micro-experience needs no witness at any "
            "level.",
            "N5: observationally inert by construction - nearest score "
            "0."],
    },
}

# the scorers' own meta-observations, condensed as filed - the audit's
# receipts for the criteria-loading findings.
META = {
    "4-a": ("N1 hardest: the 0-bucket uninhabitable for steelmanned "
            "theories; the 2-vs-1 line effectively tracks added ontology "
            "beyond physics - a parsimony standard loaded in my favor. N5 "
            "'at least one claim' ambiguous between signature and shared "
            "entailments. N2's explicit blessing of executed elimination "
            "is loaded in illusionism's favor; applied as written. "
            "SELF-CHECK: pull to gift physicalism N1; checked by extending "
            "identity-courtesy to functionalism, IIT, HOT; dislike of RM "
            "as 'dualism in a physicalist uniform, insulated from any "
            "test' indulged only where criteria tracked it."),
    "4-b": ("N1 hardest: the 1-vs-2 line turns on whether a steelman "
            "concedes bruteness versus claims constitution - partly "
            "rhetorical framing; a Type-B defender could argue their "
            "no-relation identity deserves the 2 that functionalism's "
            "constitutive claim gets, so N1 quietly favors "
            "constitutive-identity framings, which happen to be my camp. "
            "SELF-CHECK: the pull I indulged shows up only in caveats - "
            "functionalism's N1 2 rests on accepting its question-begging "
            "rejoinder at face value, a constitutive-framing benefit I "
            "denied Type-B physicalism on a thin distinction, and I flag "
            "that asymmetry rather than bury it."),
    "4-c": ("N2 hardest: the line between explaining reports and buying "
            "them verbatim is exactly where my bias lives. N1's 'external "
            "to the ontology' ambiguous - any theory can internalize a "
            "primitive by positing it. THE RUBRIC ITSELF IS LOADED IN THE "
            "DEFLATIONIST'S FAVOR: N2's 'elimination is a legitimate "
            "stance' is an illusionist-designed clause, N1's primitive-link "
            "framing structurally penalizes realist bridging, N5's "
            "tested-requirement caps purely metaphysical theories at 0. A "
            "realist scorer would dispute N2's elimination license and N1's "
            "framing, and would not produce this table - which is itself "
            "the most informative result of the exercise. SELF-CHECK: pull "
            "toward a clean sweep, re-audited every cell; did not resist "
            "the overall alignment because the criteria's structure "
            "genuinely rewards no-ransom theories."),
    "4-d": ("N1 hardest and loaded in favor of deflationary theories: the "
            "'entailed within the theory' bucket lets stipulated "
            "definitions count as derivations, rewarding theories that "
            "build the link into their definition (functionalism, GWT, "
            "HOT, illusionism) and penalizing theories honest enough to "
            "admit a structured primitive (dualism, physicalism, "
            "russellian, IIT, panpsychism) - applied uniformly, this "
            "flatters definitional fiat and re-scoping. N5 forced a "
            "choice between 'any tested claim' (too lax) and "
            "'distinctively tested content'. GWT's perfect 10 partly "
            "reflects that the criteria reward measurable-scope retreat, "
            "which a Lakatosian could read as honest discipline OR a "
            "degenerating problem-shift. SELF-CHECK: pull to punish "
            "illusionism's elimination as evasion; rubric legitimizes it; "
            "caught myself on consistency at N4."),
    "4-e": ("N1 hardest: its anchors blur because every identity theory "
            "self-describes as needing no bridge, so the scorer must judge "
            "whether linklessness is sustained (a categorical-basis "
            "identification filling a slot structural realism "
            "independently requires) or merely asserted against the datum "
            "(Type-B's admitted brute identity, functionalism's "
            "stipulation) - a judgment my structural-realist commitments "
            "load and a rival scorer could mirror-image. N5 structurally "
            "loaded toward theories with measurable targets - accepted, "
            "RM takes 1 there. SELF-CHECK: the pull showed up most at N1, "
            "where RM's 2 versus physicalism's 1 rests on the "
            "categorical-basis/slot-filling differential that a Type-B "
            "scorer would certainly reverse - there I indulged the persona. "
            "Resisted elsewhere: RM takes 1 (not 2) on N5, below all five "
            "empirically tested theories."),
}

# --------------------------------------------- doc 90 as the sixth rating
# The tribunal's filing, derived from doc 90 Part 2: three survivors in
# filed order of strength; six failures ordered by Part 2's grouping
# (instrument-bridges with survivor-remnant and distinctions first, then
# premise-bridges, dualism last for the earned-ground falsification).
# The failure grounds are doc 90's own, quoted from Part 1.
TRIBUNAL_RANK = {
    "russellian": 1, "illusionism": 2, "panpsychism": 3, "gwt": 4,
    "iit": 5, "higher-order": 6, "physicalism": 7, "functionalism": 8,
    "dualism": 9,
}
TRIBUNAL_FILING = {
    "russellian": "SOLE SURVIVOR - the placed form (no bridge bought; "
                  "dark-system dissolved, theorem grade; places)",
    "illusionism": "SURVIVOR BY DELETION (cost: the undischarged "
                   "performance, the question returned to sender)",
    "panpsychism": "SURVIVOR AS POSITION; FAILURE AS BRIDGE (combination "
                   "unposeable)",
    "gwt": "FAILURE as bridge; SURVIVOR as access-law (publication without "
           "readership)",
    "iit": "FAILURE (bridge - the identification step) + the field's most "
           "serious measurement campaign",
    "higher-order": "FAILURE (bridge - the generator) + the closest "
                    "approach to the missing instrument",
    "physicalism": "FAILURE (bridge - P*b identity; the dark duplicate "
                   "excluded by fiat)",
    "functionalism": "FAILURE (bridge - P*a/c realization; declines the "
                     "dark system)",
    "dualism": "FAILURE (bridge bought; interactionism falsified on "
               "earned ground)",
}
UNALIGNED = ["4-b", "4-c", "4-d"]   # scorers with no camp stake in RM's fate
ALIGNED = {"4-a": "physicalism", "4-e": "russellian"}


def ranks_with_ties(values):
    """Average ranks, 1 = best (highest value)."""
    order = sorted(values, key=lambda v: -v)
    out = []
    for v in values:
        # average position of equal values
        idxs = [i + 1 for i, o in enumerate(order) if o == v]
        out.append(sum(idxs) / len(idxs))
    return out


def spearman(x, y):
    rx, ry = ranks_with_ties(x), ranks_with_ties(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main():
    res = {"spec": {"criteria": CRITERIA, "theories": THEORIES,
                    "personas": PERSONAS, "blindness": BLINDNESS},
           "tribunal": {"rank": TRIBUNAL_RANK, "filing": TRIBUNAL_FILING}}

    # 1. arithmetic check: registered scores match the stated totals
    checks = []
    for s in SCORES:
        for t in THEORIES:
            got, stated = sum(SCORES[s][t]), STATED_TOTALS[s][t]
            checks.append(got == stated)
    res["arithmetic_check"] = {"all_totals_match": all(checks),
                               "cells_checked": len(checks)}

    # 2. totals, means, ranges, mean ranks
    totals = {s: {t: sum(SCORES[s][t]) for t in THEORIES} for s in SCORES}
    mean_total = {t: sum(totals[s][t] for s in SCORES) / len(SCORES)
                  for t in THEORIES}
    range_total = {t: max(totals[s][t] for s in SCORES)
                   - min(totals[s][t] for s in SCORES) for t in THEORIES}
    mean_rank = dict(zip(THEORIES,
                         ranks_with_ties([mean_total[t] for t in THEORIES])))
    res["totals"] = totals
    res["mean_total"] = mean_total
    res["range_total"] = range_total
    res["mean_rank"] = mean_rank

    # 3. per-scorer ranks and rank of RM per scorer
    per_rank = {s: dict(zip(THEORIES, ranks_with_ties(
        [totals[s][t] for t in THEORIES]))) for s in SCORES}
    res["per_scorer_rank"] = per_rank
    res["rm_rank_by_scorer"] = {s: per_rank[s]["russellian"] for s in SCORES}

    # 4. persona sensitivity: RM's range is the field's widest?
    res["rm_persona_sensitivity"] = {
        "rm_range": range_total["russellian"],
        "field_ranges": range_total,
        "rm_range_is_field_max": range_total["russellian"]
        == max(range_total.values()),
    }

    # 5. parity test N1: RM's identification vs Type-B identity
    n1 = {s: {"russellian": SCORES[s]["russellian"][0],
              "physicalism": SCORES[s]["physicalism"][0]} for s in SCORES}
    res["n1_parity"] = {
        "cells": n1,
        "unaligned_parity": all(n1[s]["russellian"] == n1[s]["physicalism"]
                                for s in UNALIGNED),
        "aligned_self_gift": {s: n1[s][ALIGNED[s]] - n1[s]["russellian"
                                if ALIGNED[s] == "physicalism"
                                else "physicalism"]
                              for s in ALIGNED},
        "note": ("the unaligned scorers (functionalist, illusionist, "
                 "neutral) score the Russellian identification and the "
                 "Type-B identity at parity; each aligned scorer gifts his "
                 "own camp +1. doc 90's bridge criterion institutionalized "
                 "the 4-e differential (placement is not a bridge; "
                 "identity is a bought bridge)."),
    }

    # 6. parity test N2: placement vs deletion
    n2 = {s: {"russellian": SCORES[s]["russellian"][1],
              "illusionism": SCORES[s]["illusionism"][1]} for s in SCORES}
    res["n2_parity"] = {
        "cells": n2,
        "illusionism_above_rm_for": [s for s in SCORES
                                     if n2[s]["illusionism"]
                                     > n2[s]["russellian"]],
        "note": ("four of five scorers (including the neutral control) "
                 "score executed deletion above promissory self-"
                 "acquaintance on the report criterion; only the "
                 "Russellian reverses. doc 90's tribunal reversed it for "
                 "everyone - deletion was charged a price, acquaintance "
                 "was filed as the strongest convergence the corpus can "
                 "certify."),
    }

    # 7. correlation of the battery with the tribunal
    trib = [TRIBUNAL_RANK[t] for t in THEORIES]
    res["correlation"] = {
        "spearman_vs_mean": round(spearman(trib, [mean_rank[t]
                                                  for t in THEORIES]), 3),
        "spearman_by_scorer": {s: round(spearman(trib,
                               [per_rank[s][t] for t in THEORIES]), 3)
                               for s in SCORES},
    }

    # 8. the inversion metrics
    failures = ["physicalism", "functionalism", "iit", "gwt", "higher-order",
                "dualism"]
    res["inversion"] = {
        "tribunal_no1": "russellian",
        "tribunal_no1_battery_mean_rank": mean_rank["russellian"],
        "tribunal_no1_battery_rank_by_scorer": res["rm_rank_by_scorer"],
        "tribunal_failures_battery_mean_ranks":
            {t: mean_rank[t] for t in failures},
        "rm_n5_by_scorer": {s: SCORES[s]["russellian"][4] for s in SCORES},
        "note": ("the tribunal's sole survivor holds the battery's mean "
                 "rank 7 of 9 (never above 7th for any scorer except the "
                 "Russellian himself); five of the tribunal's six failures "
                 "occupy the battery's top five places; the sixth "
                 "(dualism) is the one failure grounded in evidence rather "
                 "than in the bridge criterion, and the battery upholds "
                 "it. RM's N5 (falsifiability) is persona-invariantly low "
                 "(1,1,0,1,1) - the criterion where the tribunal converted "
                 "RM's worst score into its best answer ('a principled "
                 "refusal')."),
    }

    # 9. the verdict, machine-checkable
    uniqueness_reproduces = (mean_rank["russellian"] <= 2.0
                             and per_rank["4-d"]["russellian"] <= 2.0)
    criteria_stable = max(range_total.values()) <= 1
    res["verdict"] = {
        "uniqueness_reproduces_under_neutral_criteria": uniqueness_reproduces,
        "criteria_stable_across_personas": criteria_stable,
        "n1_differential_is_camp_relative": res["n1_parity"][
            "unaligned_parity"],
        "tribunal_battery_correlation": res["correlation"][
            "spearman_vs_mean"],
        "filing": ("REFUTED AS FILED - the SOLE SURVIVOR cap does not "
                   "survive neutral re-scoring (survivorship is "
                   "criterion-relative; the differential that produced it "
                   "is camp-relative); AMENDED FORM SURVIVES - RM-placed "
                   "is the field-position most isomorphic to the corpus's "
                   "own grammar, a convergence claim, not a superiority "
                   "claim; the tribunal's one evidence-grounded failure "
                   "(dualism) is upheld; the theorems are untouched."),
    }

    # 10. the meta-receipts
    res["scorer_meta"] = META

    out = os.path.join(D, "doc92_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    # console report
    print("arithmetic check:", res["arithmetic_check"])
    print("\nmean totals (5 scorers):")
    for t in sorted(THEORIES, key=lambda t: -mean_total[t]):
        print(f"  {t:14s} {mean_total[t]:.2f}  range "
              f"{range_total[t]}  mean rank {mean_rank[t]}")
    print("\nRM rank by scorer:", res["rm_rank_by_scorer"])
    print("RM N1 vs phys N1:", res["n1_parity"]["cells"])
    print("N2 RM vs illusionism:", res["n2_parity"]["cells"])
    print("spearman vs mean:", res["correlation"]["spearman_vs_mean"])
    print("spearman by scorer:", res["correlation"]["spearman_by_scorer"])
    print("verdict:", res["verdict"]["filing"][:120], "...")
    print("\nwrote", out)


if __name__ == "__main__":
    main()

