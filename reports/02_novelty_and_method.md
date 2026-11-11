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

## Online base-pair reliability experiment — 2026-11-11

The owner requests an implementation change and explicitly freezes paper files.
The present discrete repair schedule permits at most 38 accepted edits in 200
epochs and resets them at every extraction. This motivates a separate online
control, not an assertion that accepting more edits is beneficial.

Proposed insertion: detach current two-view backbone embeddings, aggregate
duplicate sample IDs separately in each view, normalize, and build kNN graphs
over distinct minibatch IDs. For each observed undirected edge, RN is mean
reciprocity over the two views and CVS is persistence of either directed edge
across views. Select RN, CVS, or RN*CVS. Unobserved edges remain unknown and
keep weight one; they are never interpreted as negative semantic evidence.
Use `w=1-ramp*(1-floor)*(1-score)`, with independent component switches,
10-epoch warmup, 10-epoch ramp and initial floor .25. These are predeclared
experimental settings, not tuned optima.

Apply weights only to base-level pseudo-positive numerators with at least one
unlabeled endpoint. Keep same-ID and legally labeled-labeled weights exactly
one, and retain original positive counts. Thus protected coefficients do not
shift through renormalization. Existing Abstain still removes its eligible
relations; the new floor is not a floor for already abstained pairs. Genuine
supervised, USE and coarse-level objectives stay unchanged. Collapse/insufficient
unique-ID evidence uses Keep weights; nonfinite inputs fail explicitly.
The exact same weighting must enter the cloned lookahead objective and its
target-drift audit. This avoids inconsistency between proposal scoring and SGD.

Overlap is substantial: [HypGCD equations 15–16](https://duannyu.github.io/assets/papers/hypgcd.pdf)
already mines reciprocal neighbors; [Learning to Distinguish Samples, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8177_ECCV_2024_paper.php)
uses a learned neighbor graph and cross-view filtering. Its PDF fetch timed out
in this session, so the full-method gate is still open. [Let's Start Over,
IJCAI 2024](https://ijcai.org/proceedings/2024/532) uses distance-aware
consistency selection and progressive retraining. RCA/LEAH gates described
above remain open. This online weighting is therefore implemented as an
experimental denoising control/hybrid option, not a renamed novel contribution.
Its concrete scope differs from sample filtering/retraining: detached local
edge weights, fixed protected coefficients, base pseudo-SSE only, and coherent
integration with CRiSH's independent-confirmation repair. None proves novelty.

Required controls are SelEx B0, legacy CRiSH (`pair_reliability=off`), RN only,
CVS only, both with discrete repair disabled, and both with CRiSH. Match source,
seed, training budget and SS-last evaluation; inspect Old/New, weight histograms,
eligible/changed pair counts, no-evidence fallbacks, time and memory. Revert the
new option if matched measured outcomes are negative. CPU tests establish
bounded scores/coefficient semantics/backward/parity/resume, not efficacy.
