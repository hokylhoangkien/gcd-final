# Coordinated training upgrade: implementation and open method gate

Status: implemented experimental options; effectiveness and novelty unvalidated.
This extends the existing CRiSH/B0 runtime, not the pinned upstream checkout.
`reports/04_method_and_research_plan.md` retains the broader research discussion
and open related-work leads.

## Non-self row-normalized USE

Let F be the existing nonnegative, min-max-scaled confusion matrix for two
views, ordered [all view 1, all view 2]. The legacy path intentionally keeps
the released `F / F.sum(dim=1)` broadcasting and zero-row rejection.

In row mode first set F_aa=0. For positive support, use

    S_ab = F_ab / sum_(c != a) F_ac.

For a zero-support row, set S_a,p(a)=1 and every other entry to zero, where
p(a) is the other view of the same image. This positive-view fallback does not
infer any inter-image semantic relationship. Remove self and reorder columns
with the original info_nce helper. Original LabelSmoothingLoss then uses

    q_a = (1-smoothing) e_p(a) + smoothing S_a.

Each target sums to one. This changes both target shape and effective per-anchor
loss scale relative to the legacy source. It is a correctness-motivated
ablation, not a mathematically guaranteed accuracy improvement. Original
true-label SupCon and pseudo-SSE code are unchanged. `target_drift` and action
scoring use the same normalization mode as SGD. B0 upgraded receives the same
change to separate normalization benefits from repair benefits.

Implementation: `crish/rebuild.py`, `crish/loss.py`, shared trainer/runtime config.
PyTorch's probability-target cross entropy formula supports this distinction:
https://docs.pytorch.org/docs/2.4/generated/torch.nn.CrossEntropyLoss.html .

## Full-feature shortlist, bounded local cross-view verification

Reserve both ID-disjoint probe folds first. Uniformly sample unlabeled query
anchors from the remaining training rows. With cache budget B and k neighbors,
use at most floor(B/(k+1)) anchors (at least one). For each, compute exact
Euclidean kNN over all eligible normalized frozen-backbone features, excluding
self. Stream query/reference blocks and retain top k; do not allocate N-by-N.
Build a unique cache containing anchors and their neighbors, then random-fill
remaining cache slots. This uses no pseudo/hidden class label for selection.

The existing independent augmentations and reciprocal/cross-view evidence are
computed only within that cache. Thus this is global frozen-feature
shortlisting followed by local augmented-view evidence, not exact global
cross-view kNN and not hyperbolic kNN. Probes are excluded from all reference
and proposal/cache rows. The existing repair support, protected-label,
target-drift and independent-confirmation guards remain in force.

At shipped settings B=128, k=5, at most 21 queries use the full reference pool.
Reference blocks contain at most 1,024 rows; deadlines are checked between
blocks. CPU/GPU tensor operations themselves are not preempted by that check.
The enclosing worker deadline remains the final runtime bound.

Implementation: `crish/reliability.py`, `crish/sampling.py`, `crish/engine.py`.
Logs explicitly distinguish full reference counts from actual cache counts and
state that augmented-view neighbors are local.

## Larger legal probes

Upgraded profiles use 16 legally labeled and 32 unlabeled images per fold.
Labeled selection cycles through randomized classes without replacement;
classes come only from the compressed legal labeled-target registry. Unlabeled
selection remains uniform without known/novel identity. Discovery, confirmation
and the proposal cache are sample-disjoint. Rare classes are never duplicated
to fill a quota. Insufficient support returns the existing Keep fallback.

These sizes and sampling are hypotheses about decision robustness, not measured
optima. One-step projector-only probe gains do not establish effects on the
trainable backbone or final discovery. Warmup/refresh/action budgets, objective
coefficients and the independently confirmed acceptance criterion are unchanged.
Synthetic DINOHead preflight scales row counts/dimensions for a bounded tensor
check; only real-data smoke exercises the shipped row budgets.

## Comparisons and novelty risk

Existing flags independently select legacy/row USE, random/global-neighbor
evidence and random/class-balanced known probes; probe counts are independent
settings. Compare each change under locked source, split, backbone, optimizer,
seed and evaluation rule, then compare the combined profile. Include legacy
B0, corrected B0 and matched corrected CRiSH. A multi-seed sweep is not launched
here. Main accuracy is All, with Old/New, best/last, time and memory retained.
No oracle metric or hidden training annotation enters these new components.
The existing checkpoint selector still uses evaluation labels explicitly.

Neither target normalization, neighbor selection nor stratified sampling is
claimed as a new scientific contribution. HypCD already supplies hyperbolic
USE/SSE: https://arxiv.org/html/2504.06120v1 . HypGCD section on local hierarchy
already uses hyperbolic K-reciprocal neighbors and triplets:
https://duannyu.github.io/assets/papers/hypgcd.pdf . RCA is a confidence and
cross-view-alignment overlap lead:
https://proceedings.mlr.press/v306/yu26l.html . LEAH models intrinsic image
hierarchy with token pruning/construction:
https://ojs.aaai.org/index.php/AAAI/article/view/39236 .

RCA/LEAH full-method access was blocked in this session's browser review; do not
substitute these summaries for the required method comparison. The earlier
method report also lists CiPR, ProtoGCD, SEAL, gradient reweighting, CURE and
ConGCD. Those gates and the need for heuristic/compute controls remain open.

## Evidence status

Unit/integration checks use synthetic CPU data, no pretrained weights. They
test probability mass, finite backward, source arithmetic parity, supervised
term preservation, exact nearest-neighbor support, disjoint legal probes,
hidden-label invariance, disabled-repair parity and resume. Test receipts/logs
identify actual runs. CUDA, full-data accuracy, multi-seed gains and CUB
generalization for the upgraded profiles are **not yet run**.

## Evidence/probe-only trial decision — 2026-11-08

Prioritize an isolated trial of the existing evidence/probe options before
changing the objective or loosening acceptance guards. A random 128-row cache
may miss relevant neighbors; the small reference probes (4 legally labeled /
8 unlabeled rows per fold) may yield noisy repair decisions. Neither limitation
has yet been established as the dominant cause of benchmark error. Global
frozen-feature shortlisting and larger probes are plausible interventions,
not an empirical ranking of the changes most likely to improve accuracy.

Prepared local plan:
`runs/plans/evidence-probe-only-20261108T110242Z-fe28609c/`.
It compares seed-1 reference CRiSH against CRiSH with
`evidence_sampling=global_neighbors`, `probe_sampling=class_balanced`,
`probe_known=16`, `probe_unlabeled=32`. Both retain legacy USE normalization,
the same loss, repair limits and checkpoint protocol. All other loaded config
fields match. Separate smoke and full configs, config hashes, source/split
provenance and sequential reviewable commands are saved there. This is a
joint evidence/probe trial; a positive result would still need separate
ablations to attribute any improvement. It does not replace the matched B0
comparison or establish a novel contribution.

Four launcher dry runs passed. A 60-second-bounded CPU synthetic DINOHead
forward/backward and repair-scoring check passed for the combined options
with legacy USE (one scored proposal, zero accepted). This fixture scales
dimensions, sample counts and some guards; it verifies execution and finite
gradients, not the 16/32 real-data budgets or repair quality. No training epochs
or accuracy measurements were produced. Larger probes and full-pool neighbor
search may cost more and can still reinforce wrong neighborhoods or favor
known classes. Inspect acceptance/fallback reasons and runtime in real-data
bounded smoke before spending on a full run; the 120-second refresh budget
remains unchanged. CPU tests do not establish GPU readiness.

First remote stage, after transferring/recreating the ignored plan on the
matching Linux checkout and checking GPU/data/weights:
`.venv/bin/python -m gcd_runtime.launch preflight --config runs/plans/evidence-probe-only-20261108T110242Z-fe28609c/reference.json`.
The prepared `commands.txt` is not an automatic sweep; no command was executed
on a remote GPU during this decision.

## Final profile handoff — 2026-11-08

At the owner's subsequent request to finalize the source, this same
evidence/probe combination is promoted to portable `*_final*.json` profiles
for Cars/CUB, B0/CRiSH and full/smoke. The default user entrypoint and matched
plan preparer select `final`. Reference and upgraded settings are preserved.
Final uses legacy USE, global frozen-feature shortlisting and legal
class-balanced 16/32 probes. It changes no objective coefficients, repair
acceptance guard, warmup or full-run budget. CUB receives the same method
choices, not dataset-specific tuning. The manuscript's generated table now
compares reference/final; upgraded remains an explicitly separate row-USE
ablation. No new algorithm, empirical benefit or novelty claim is implied by
the profile name. Fresh runs are required after the CLI source fingerprint
change; historical checkpoints keep their original source/config.
