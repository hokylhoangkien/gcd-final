# Bounded CRiSH action-strength audit

Twelve fresh batch-128 draws from the ordinary Cars sampler, six per R2 checkpoint at 100 and 150. Batch IDs differ from the prior V0/V2 and V3 audits. Training, the two probe folds, and the sealed queries are ID-disjoint within each batch. The sealed reference bank excludes all those IDs.

We freeze the E3 eligibility and evidence ranking computed at MOVE .2. MOVE curves retain only proposals whose source probability is at least .4 in both views. Choose up to two common-support MOVE and two ABSTAIN candidates per batch. Invalid counts below cover all E3 MOVE proposals at each requested mass, before the cap. The same source/destination and action IDs are used across strengths.

KEEP uses original SimGCD. The full joint step updates the weight-normalized classifier and block 11 through classification, InfoNCE and SupCon, with checkpoint SGD state, excluding the R2 loss. MOVE transfers the indicated mass. ABSTAIN scales the original teacher rows by 1−β with the same denominator and no renormalization. At fixed parameters and detached targets the edit gradient is affine in strength, so one cached edit differential per candidate scales exactly in real arithmetic. Each strength still gets its own SGD step and feature forward. The original doses and KEEP are checked against the functional E3 classifier step.

V0 uses the original frozen classifier probe objective. Discovery ranks positive gains that satisfy the non-increasing known-CE guard. The discovery winner must pass both checks on the separate confirmation fold. No action follows rejection. All decisions are frozen before sealed semantic evaluation. These singleton decisions do not model E3 multi-action or fallback behavior.

| Arm | Candidate-batches | Semantic gain All / Old / New (×10⁶) | Harmful | Edit gradient norm classifier / block 11 | V0 gain discovery / confirmation (×10⁶) |
|---|---:|---|---:|---|---|
| KEEP | 12 | 0 / 0 / 0 | 0 | 0 / 0 | 0 / 0 |
| move/0.05 | 21 | +0.098 / -1.031 / +0.701 | 9 | 0.0039 / 0.0113 | -0.01 / +0.02 |
| move/0.1 | 21 | +0.189 / -2.076 / +1.400 | 9 | 0.0078 / 0.0226 | +0.03 / +0.04 |
| move/0.2 | 21 | +0.413 / -4.130 / +2.838 | 9 | 0.0156 / 0.0451 | +0.12 / +0.13 |
| move/0.4 | 21 | +1.311 / -7.196 / +5.777 | 9 | 0.0312 / 0.0903 | +0.23 / +0.30 |
| abstain/0.25 | 24 | -0.141 / -0.777 / +0.258 | 9 | 0.00346 / 0.00859 | +0.02 / -0.03 |
| abstain/0.5 | 24 | -0.273 / -1.563 / +0.535 | 10 | 0.00692 / 0.0172 | +0.07 / -0.07 |
| abstain/1.0 | 24 | -0.520 / -3.172 / +1.137 | 10 | 0.0138 / 0.0344 | +0.18 / -0.16 |

Invalid MOVE proposal counts by mass — 0.05: 0, 0.1: 0, 0.2: 0, 0.4: 1

| V0 policy | Accepted | Harmful / accepted | Gain | Same-ID fixed E3 dose | Same-ID random strength | Same-kind/strength random ID |
|---|---:|---:|---|---|---|---|
| original | 6 | 3/6 | -1.528 / -5.065 / +0.136 | -1.528 / -5.065 / +0.136 | -1.388 / -4.511 / +0.034 | -2.373 / -7.347 / +0.282 |
| adaptive | 6 | 3/6 | -2.424 / -6.862 / -0.381 | -1.528 / -5.065 / +0.136 | -1.388 / -4.511 / +0.034 | -3.008 / -8.425 / -0.159 |

- original — accepted: 6, confirmation_gain: 5, discovery: 1

- adaptive — accepted: 6, confirmation_gain: 5, discovery: 1

Policy gains use the same All / Old / New units and average over all 12 batches with zero for KEEP/no action. Harmful means All gain < −1e−7. The fixed-dose and random-strength controls intervene on exactly the selected candidate and batches. They are forced comparisons and can choose a strength V0 would reject. The random-ID control retains the selected kind and dose and is uniform over the batch's eligible IDs. Unmatched original-versus-adaptive policy means can reflect different intervention counts. These checkpoint-conditioned one-step effects do not estimate train-time accuracy.

[Sealed results and frozen decision provenance](r2_strength_diagnostic.json)
