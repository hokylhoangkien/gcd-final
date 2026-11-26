# CRiSH-R on SimGCD · Stanford Cars

**Decision before training.** GO for the predeclared R0 → R1 → R2 experiment with external budget guards. The first question is whether changing a positive relationship improves over learning from the same unedited graph. The bounded test verifies the optimization mechanism and gives no estimate of final accuracy. Feature-level counterfactual selection (R4) remains deferred. [Measurements and limitations](CRISH_RELATIONAL_SMOKE.md)

## Hypothesis and comparison

The E3 audit found that a virtual classifier update leaves backbone features fixed. Increasing verified distillation exposure from 250 occurrences in E3 to 549 in E3-F coincided with All changing from 56.51 to 56.49 in one reported seed. CRiSH-R therefore tests a different intervention: change the strength of an existing feature-to-feature attraction. Stronger attraction has a positive corrective direction; reducing attraction can protect a questionable association. The proposed eventual verifier should judge the resulting feature relationships on independent samples. This implementation establishes the evidence → intervention part and instruments the realized geometric effect. [Mechanistic audit](CRISH_MECHANISTIC_AUDIT.md) · [Recovered E3-F results](E3F_RECOVERED_RESULTS.md)

| Variant | Objective | Edge policy | Scientific comparison |
|---|---|---|---|
| R0 | Original SimGCD | Graph off | Matched trajectory for the cost/benefit of any graph term |
| R1 | SimGCD + weighted relational loss | KEEP every baseline edge | R1 − R0 measures adding ordinary relational learning |
| R2 | Same objective, graph support, refresh inputs, warmup, coefficient and denominator as R1 | Cross-view evidence edits the weights | R2 − R1 tests repair over the relational baseline; R2 − R0 tests net effect |
| R3, future | Same objective | Randomization matched to intervention mass | Extension contract only |
| R4, future | Same objective | Feature-level factual/trial verification with independent probe | Extension contract only |

R1 and R2 use an identical graph *construction*. Different weight updates can produce different backbone features and graphs in later epochs. Their trajectories are therefore allowed to diverge. Equal normalization does not imply equal positive mass after edits. Both planned and realized mass matter in interpreting any difference. No variant adds a new trainable head.

## Graph construction and intervention

Use only the image-only unlabeled training subset, including both known and novel semantic classes, with no access to that distinction. The official SSB split is built once per stage. During training each unlabeled row returns its image, stable original ID and sentinel −1. Labeled images retain their ordinary SimGCD supervision. The labeled set also keeps its contributions to the original unsupervised terms.

At the start of each active epoch, snapshot normalized ViT CLS features in eval mode. Encode one deterministic center crop and a separate pair of independently sampled official training crops for every unlabeled ID. Freeze the backbone for graph construction and restore model mode and training RNG afterwards. Compute exact Euclidean top-$k$ on unit features within each of the three views. This ordering agrees with cosine distance up to ties. No approximate shortlist is used.

Let $A^v_{ij}=1$ if $i$ is among $j$'s $k$ nearest neighbors **and** $j$ is among $i$'s in view $v$. Self edges are excluded. Undirected support is $E=\{\{i,j\}:A^0_{ij}=1\}$ for center view 0. For every edge in $E$, set $R_{ij}=R_{ji}=0.5$. Edges outside $E$ are absent from the loss. Compute $s_{ij}=A^1_{ij}+A^2_{ij}$ independently of construction of $E$.

| Two-crop votes $s_{ij}$ | R1 weight | R2 action | $\Delta R_{ij}$ | R2 weight |
|---:|---:|---|---:|---:|
| 0 | .50 | WEAKEN | −.25 | .25 |
| 1 | .50 | KEEP | 0 | .50 |
| 2 | .50 | STRENGTHEN | +.25 | .75 |

The sole operation is $R'=R+\Delta R$ on the fixed support. The contract enforces finite weights in $[0,1]$. A zero-weight edge gives exactly zero contribution and zero gradient. Zero implements edge-level abstention, reachable through the contract and covered by a unit test. The locked R2 policy retains a residual attraction for zero-vote edges, since crop instability does not prove semantic disagreement. Canonical ID pairs identify edits; an edge can be encountered from either endpoint. Each refresh replaces the whole graph.

### Loss and gradient

Use the unchanged Cars objective $L_S=.65(D+E+I)+.35(C+S)$ from the parent audit, with its implementation-specific reductions and temperatures. Let $h_{iv}=f_\theta(x_{iv})$, $u_{iv}=h_{iv}/\|h_{iv}\|$, and $p_j$ be the detached, normalized center-view CLS key for peer $j$ at refresh. For one batch with $B=128$ sampled occurrences and $V=2$ views:

$$
L=L_S+\lambda_R L_{\rm rel},\qquad
L_{\rm rel}=\frac{1}{VBk}\sum_{i\in {\rm draws}}\sum_{v=1}^{V}
\sum_{j:\{i,j\}\in E} R'_{ij}(1-u_{iv}^{\top}p_j).
$$

$M=VBk=1280$ includes labeled rows, missing neighbor slots and repeated sample draws. Missing slots contribute zero. Each orientation is charged only when its anchor is actually sampled. The cached peer need not occur in the minibatch. Weights are never normalized by their sum or by active edge count.

For nonzero $h$, the change relative to R1 for a single sampled anchor view is

$$
\delta\nabla_{h_{iv}} L=
-\frac{\lambda_R}{VBk\|h_{iv}\|}
(I-u_{iv}u_{iv}^{\top})\sum_j\Delta R_{ij}p_j,\qquad
\delta\nabla_{\theta_{11}}L=\sum_{iv}J_{\theta_{11}}(h_{iv})^\top\delta\nabla_{h_{iv}}L .
$$

Strengthening rotates the gradient-descent direction toward the peer on the unit sphere. Weakening reduces that attraction. A nonnegative edge still attracts in the absolute objective. Multiple edges and shared block parameters can cancel or couple their effects. The direct edit gradient on detached keys, classifier and projector parameters is zero. The usual SimGCD branches continue to update their own parameters. Later logits can change because the backbone changes.

**Locked settings.** R1 and R2 both use $\lambda_R=1.0$. R2 uses edit size .25. Ten warmup epochs precede the first graph at the start of epoch 11. Refresh the exact reciprocal $k=5$ graph each active epoch with $R=.5$. The coefficient per unit peer tangent for a single edited occurrence is $\lambda_R|\Delta R|/(VBk)=1.953125\times10^{-4}$ per view before the feature-normalization Jacobian. The four-batch full-graph diagnostic selected this dose on paired gradient and feature responses. Later-epoch strength and semantic efficacy remain unmeasured. [Sensitivity results and lock criterion](CRISH_RELATIONAL_SENSITIVITY.md)

### Evidence and exposure ledger

Each epoch records the baseline graph as canonical ID pairs with their cross-view vote and selected signed delta. A common baseline weight plus those records reconstructs all edited weights. Graph health contains degree-zero counts, reciprocal-edge Jaccard between the two crops, vote histogram, mean coordinate variance and mean resultant length of normalized center keys. Under complete directional collapse the variance tends to zero and the resultant length to one. These are descriptive measurements.

An exposure is one sampled anchor/peer relationship contributing on the two views of a successful optimizer step. Multiple draws or the other endpoint can count again. The ledger contains per-edge multiplicities, distinct edited anchor IDs, strengthen/weaken totals, and $m_{\rm abs}=\sum_{\rm encounters}|\Delta R_{ij}|$. The sum of coefficients on both views over the epoch is $\lambda_R m_{\rm abs}/(Bk)$ for fixed batch size. Neither mass measures the actual gradient norm. The first mixed batch of every active epoch separately records block-11 norms, ratios and cosines for the weighted relational term and the weighted edit-only differential against the unedited SimGCD gradient. Zero-vector cosines are null. Runtime, peak allocated VRAM, per-epoch mean losses, actual LR and full precision All/Old/New are exported.

### Cost and failure modes

For $N$ unlabeled rows, feature width $d=768$ and per-image backbone forward cost $F$, each active refresh costs three passes $O(3NF)$ and three exact top-$k$ searches with $O(3N^2d)$ distance work (plus top-$k$ selection). Distances are blockwise, with $O(bN)$ distance workspace, $O(Nd)$ snapshot storage and $O(Nk)$ graph storage. Each training step adds $O(VBkd)$ arithmetic. The first-batch gradient diagnostic incurs additional backward traversals once per active epoch. R0 skips all graph computations. The smoke used a reduced graph and does not benchmark full-epoch GPU time.

* A reciprocal neighborhood can group similar backgrounds or body styles across Cars classes. Repeated crop agreement can strengthen the same mistake. Graph generation and policy are model-dependent.
* The attraction-only objective can contract embeddings. SimGCD contrastive/entropy terms compete with it. Monitor collapse statistics, Old/New and realized dose alongside any All improvement.
* Per-epoch detached keys can become stale as block 11 updates. Low-degree or never-drawn anchors get less exposure. The ordinary sampler remains free to repeat or miss any anchor.
* A one-seed, early-stopped, transductive experiment supports an exploratory budget decision. A paper claim needs independent replication and, eventually, comparison with intervention-mass-matched random edits.

## Protocol and external decision rules

Stanford Cars, seed 1, official SSB split, 50% labeled proportion in old classes, DINOv1 ViT-B/16, only backbone block 11 trainable, original SimGCD head. Two official crops, FP32, batch 128, replacement sampler weighting each known-labeled row 1 and each unlabeled row $n_L/n_U$, epoch length equal to merged dataset size with drop-last. SGD starts at .1 with momentum .9 and weight decay $5\times10^{-5}$ on regularized parameters. Teacher temperature moves .07→.04 over the original 30 epochs. Cosine LR has $T_{\max}=200$ and $\eta_{\min}=.0001$ for **every** stage. The stage endpoint never shortens or restarts the scheduler. Deterministic kernels and fixed seeds support resume on a matching software/device stack.

The official v2 transductive evaluator runs after the update epoch, using a separate annotated evaluation loader. Its metrics are written to disk. Training and graph selection receive no evaluation values. The external controller stops workers at milestones. A separate guard process entry point can reproduce the decision from exported metrics. Sealed semantic geometry runs after the budget decision and is never read by it.

Let $\Delta$ be candidate minus reference **mean percentage points** over exactly twenty matching one-based epochs, computed from full-precision exported fractions. R1 compares with R0. R2 must satisfy the rule separately against R1 and R0. The least favorable classification applies. Any missing or mismatched required trajectory yields BLOCKED, even when a different comparison is available.

| Completed epoch and window | NO-GO if either condition holds | Positive decision | Remaining cases |
|---|---|---|---|
| 100, epochs 81–100 | $\Delta All\leq-.5$ or $\Delta New\leq-1.0$ | GO if $\Delta All\geq1.0$ and $\Delta New\geq.5$ | GRAY ZONE, continue to 150 |
| 150, epochs 131–150 | $\Delta All<.3$ or $\Delta New<0$ | STRONG GO if $\Delta All\geq1.0$ and $\Delta New\geq.5$ | MARGINAL, continue to 200 |

R0 always targets 200. The guard checks R0 completion, milestone availability, run identity, variant, matched protocol, contiguous epoch numbering and finite in-range window accuracies. Each decision records hashes of the input window rows and the exact deltas/reason. A stopped R1 may continue solely to supply R2's reference window by explicit request. The original NO-GO and the override remain in the journal. These thresholds manage compute; they are not tests of statistical significance.

## Linux GPU operation and artifact recovery

The [generic Linux runbook](CRISH_RELATIONAL_LINUX.md) covers installation, the full-graph benchmark, standalone R2, matched R0/R1/R2, manual decisions, recovery and verified SSH downloads. The decision table above applies to matched mode. Standalone R2 uses the same worker, loss, sampler, optimizer, 200-epoch schedule and v2 evaluation. It can run without reference trajectories and defaults to a manual review at completed epochs 100 and 150.

## Changes relative to E3

| New path | Mechanism |
|---|---|
| relational/graph.py, relational/__init__.py | Image-only sparse reciprocal graph, edge-policy interface, detached peer loss, gradient-support and exposure accounting |
| relational/train.py | Locked Cars builder, original sampler and SimGCD terms, trainable block path, graph refresh and stage worker |
| relational/state.py | Atomic ten-epoch/milestone retention, identity checks, RNG restoration and journal recovery |
| relational/guard.py, relational/run.py | Window evaluator and external stage controller, BLOCKED/NO-GO, reference-only continuation |
| relational/offline.py | Sealed, fixed 980-image old/novel semantic neighborhood and centroid evaluation |
| relational/export.py | Artifact allowlist, tar, per-member and archive integrity verification |
| relational/smoke.py, validate_relational.py | Real-Cars block-11 diagnostic, CPU contracts, official upstream parity and CUDA replay |
| scripts/run_relational.sh, scripts/resume_relational.sh | Timed Linux tmux launch and explicit resume |
| reports/CRISH_RELATIONAL_DESIGN.md, reports/CRISH_RELATIONAL_SMOKE.md and smoke receipt | Protocol lock, measurements, assumptions and recommendation |

Original E3 files remain available on the same branch. The relational runner imports the existing image-only adapter, exact blockwise kNN, isolated eval/RNG context, checkpoint primitives, official losses and v2 evaluator. It does not invoke the E3 pseudo-target edits or classifier-only verifier. The policy hook takes canonical edge IDs, detached evidence votes and fixed settings, returning one signed delta per edge. A future R3 implementation can inject a mass-matched random policy; a future R4 policy can carry its own feature updater and independent probes. Neither is registered as a runnable variant. No command here changes the sampler to force exposure.
