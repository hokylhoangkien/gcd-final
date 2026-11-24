# Counterfactual-verified supervision repair in SimGCD

Historical report for commit `6af77dc`. The current primary E3 method, Cars configuration, exposure rules and bounded benchmark are documented in [CRISH_SIMGCD_GLOBAL.md](CRISH_SIMGCD_GLOBAL.md). The one-batch E3 fixture and matched-dose commands below refer to the historical commit.

## Provenance and research question

Can geometry-aware repair selected by a counterfactual update improve SimGCD independently of SelEx? The implementation and checks below establish a way to run that experiment. Accuracy on a GCD benchmark has yet to be measured.

| Input | Revision |
| --- | --- |
| Source, hokylhoangkien/gcd-final, feat/local-adaptive-acceptance | 1c6ada810b451e5a1f3b36b930c550489917cf0a |
| Target, CVMI-Lab/SimGCD, official upstream | f843f27195701b7c8c5713cfd17499ad67c86247 |
| Local adapted branch | feat/crish-simgcd. Run `git rev-parse HEAD` to identify the checkout used for each run. |

Reviewed the source method audit in `reports/24_method_audit_20261124.md`, the Cars local-adaptive config, and the candidate, evidence, refresh, lookahead, probe and supervision paths. The audit identified base-level dominance, a projector/evaluation mismatch, dimension-dependent probe weighting and material effects of acceptance caps and deadlines. These directly affect the interpretation of the port.

## File inventory and integration

| File | Change and entry points |
| --- | --- |
| train.py | Modified. The single-device CLI registers experiment options and constructs the image-only training subset in active modes. In train(), Repairs.due selects the refresh batch. The same backbone forward supplies features and logits. Repairs.loss supplies the distillation scalar. Repairs.finish_step records actual exposure after SGD/GradScaler. |
| crish/__init__.py | Created. Local package. |
| crish/ids.py | Vendored ID-alignment utilities from the pinned source. |
| crish/reliability.py | Vendored original evidence implementation. The active path uses blockwise_euclidean_knn and sparse_evidence. |
| crish/data.py | Created. ImageOnly reads images and stable IDs from the upstream training subset and returns a sentinel for the unlabeled target. |
| crish/repair.py | Created. Action, proposals, corrected_distill, VirtualClassifier.logits/step/probe, and select implement proposals, target edits, paired virtual SGD, confirmation and sequential selection. |
| crish/runtime.py | Created. arguments and Repairs implement CLI, schedule, matching, wall-clock accounting and JSONL logging. |
| validate_crish.py | Created. Bounded gradient, alignment, leakage, isolation and active-mode checks. |
| smoke_train.py | Created. Executes the pinned upstream train() and adapted train() end to end on identical synthetic inputs, including evaluation and checkpoint reload. |
| reports/CRISH_SIMGCD_ADAPTATION.md | Created. Method, validation, scientific scope and experiment commands. |

## Method crosswalk

| Component | Original CRiSH | Adapted CRiSH | Reason | Scientific impact |
| --- | --- | --- | --- | --- |
| Geometry | Normalized backbone Euclidean neighbors around a globally shortlisted pool, two two-view trials in source config | Exact Euclidean neighbors of normalized backbone outputs on the update subset of one real batch, one two-view trial | Use the exact crops that generated live teacher supervision without extra backbone passes | Evidence is local and noisier. Joint reliability is binary with one trial. A small or unrepresentative batch can miss a genuine boundary. |
| Boundary rank | Measured reciprocal cross-partition minus within-partition edge mass, divided by measured degree | Same rank and joint reciprocal cross-view edge construction. Partitions come from the argmax of the mean of two sharpened classifier distributions | SimGCD has classifier slots and an online teacher | Partition membership can change every step. Unobserved pairs contribute no negative evidence. |
| Destination proposal | Existing prototype ordered by distance, sufficient reciprocal support and source/destination size | One eligible destination with greatest total measured support, then mean teacher probability, then class index. Source count at least 3, target count at least 2, at least one joint edge. Source probability must cover the edit in both views | There is no fitted SelEx centroid table. Teacher mass and slot support are directly observable | Changes the proposal distribution. Fragmented or rare classes may have few legal MOVEs. |
| KEEP, ABSTAIN, MOVE | Keep, suppress pseudo-SSE for a sample/level, or change membership and rebuild affected supervision including USE coefficients | Keep teacher distribution, suppress that row's distillation, or transfer fixed teacher mass between classifier slots. Labeled rows are ineligible | SimGCD consumes distributions in DistillLoss | A MOVE is a soft edit and may preserve the argmax. Edits have no direct effect on another row's teacher target. |
| Protected supervision | Legal labeled registry and bound on shifts in rebuilt coefficients | Labeled rows' direct target shift is identically zero. Retain a known-loss guard in both probe sets. MOVE bounded by mass, ABSTAIN intentionally removes one row | The original nonlocal USE coefficient shift has no direct analogue | Model parameters are shared, so indirect effects remain possible. |
| Hierarchy | Level-dependent SSE on nested/aggregated partitions. The source audit found all 463 logged accepted actions at base level | Flat classifier slots only | Official SimGCD supplies a single classifier partition | Experiments address base-level repair. |
| Virtual optimizer | Cloned projector SGD on detached features | Functional FP32 SGD of last_layer.weight_v with the live scale, LR, weight decay and momentum state | Official DINOHead computes logits directly from normalized backbone features, beside the projector MLP | Measures sensitivity of the classifier used for inference, with frozen features. |
| Independent probe | Class-balanced known contrastive objective, opposite-view projector MSE, variance floor, fresh confirmation IDs | Known classification CE and opposite-view distributional CE, same frozen teacher for all branches, fresh confirmation IDs, strict no-increase guard for known CE | Class-space objective aligns with classifier and has no embedding-dimension division or projection variance floor | Independent examples, shared teacher. Known folds are membership-stratified, not class-balanced. |
| Schedule and budgets | Warmup 10, refresh every 5 epochs, 16 boundaries, k=5, 120 s, up to 8 accepted and 8 confirmations in source config. Cached supervision is available to training batches | Same warmup, cadence, boundary limit, k and deadline defaults. Fixed ceiling 1 or configurable adaptive cap (default 4), one fresh confirmation per round. An edit lives for the scheduled optimizer step only | The teacher is recomputed each step. One-step exposure avoids applying stale slot edits and permits exact dose replay | In a 200-epoch run with defaults there are 38 eligible steps and at most 38 edits in E3. This may be too sparse for a measurable effect. |
| Sequential decision | Rebuild and check gain versus keeping the current accepted set | Revalidate MOVE eligibility after prior MOVEs; score every remaining eligible candidate against the current set. Also require the whole set to beat the original unedited factual update | Enforces incremental utility and the original-supervision comparison for groups | E4 can stop before its cap and is stricter than the original marginal rule. |

## Objectives and counterfactual comparison

In the source SelEx path, a supervision state S supplies pseudo-memberships, masks and similarity coefficients to the original loss. In the Euclidean branch the structure is (1−s) USE(S) + s [known SupCon + sum of level-weighted pseudo-SSE(S)] / 2. KEEP uses S, ABSTAIN masks a pseudo-SSE row/level, and MOVE rebuilds S for the changed membership. The source probe measures class-balanced known neighbor log loss, opposite-view normalized feature MSE and an embedding standard-deviation floor. Its gain is the difference of probe losses after factual and corrected virtual steps.

In SimGCD, let z be the two views' logits, q_v = softmax(stopgrad(z_v)/T_teacher(epoch)), p_v = softmax(z_v/T_student), and B the original batch size. For target distributions r, the repaired distillation loss is

    D(r) = − sum over i,v,c of r[v,i,c] log p[1−v,i,c] / (2B)
    KEEP       r = q
    ABSTAIN    r[:,i,:] = 0
    MOVE s→t   r[:,i,:] = q[:,i,:] + m (e_t − e_s),  m = 0.2 by default

An empty action set calls upstream DistillLoss directly. Nonempty edits compute softmax/log-softmax in FP32. Each MOVE preserves mass and nonnegativity in both views. ABSTAIN fixes the original denominator, so other rows retain their coefficients. Gradients from supervised classification, memax, representation InfoNCE and supervised contrastive learning flow through their official paths. Real optimizer groups, LR scheduler, DINO backbone configuration, architecture, augmentation, split, evaluation functions and checkpoint policy follow train.py upstream.

For a proposal set A, the virtual classifier computes cosine logits with the official weight-normalization scale and a copy of the trainable direction V. Its SGD objective on the scoring subset is

    L_A = (1−s) [D(r_A) + λ_memax (sum_c p̄_c log p̄_c + log K)] + s CE_known
    V_A = SGD(V_initial, ∂L_A/∂V, live classifier optimizer settings and copied momentum)

The representation losses have zero partial derivative with respect to this classifier parameter. Backbone and MLP gradients still participate in the subsequent real update. All branches reuse the same normalized features, rows, teacher logits and snapshot. Each step clones its own V and uses out-of-place momentum arithmetic. It never calls the live optimizer, weight-normalization hooks, dropout, data augmentation or model forward. Candidate/fold randomization uses a private Python RNG. An AMP run still scores in FP32, using features and teacher logits observed in the real forward.

Before scoring, partition unique IDs without replacement into a virtual update set, one discovery fold and separate confirmation folds. Discard every occurrence of a repeated sampler ID from the three roles. Reserve the same folds in all modes, using one discovery plus max(fixed, cap) confirmation folds. Each contains up to four known and four unlabeled IDs by default, with at least one of each. Keep sufficient rows for the virtual training subset; insufficient distinct examples yield KEEP.

For each fold H, freeze a reference distribution u from the unedited, pre-update classifier at temperature 0.1. Then evaluate

    J_H(V) = mean_known CE(y, p_V) + mean_unlabeled CE(u_opposite_view, p_V)
    gain(A versus F) = J_H(V_F) − J_H(V_A)

Both terms average over the two views. Targets, masks, fold IDs, normalization and reference probabilities are shared across treatments. The action never masks an evaluation row or changes the probe denominator. An ABSTAIN can trivially reduce its training loss but still lose on J. This is a surrogate for preserving known semantics and transferring predictions across views on other examples. Unlabeled semantic correctness cannot be identified by this objective.

On discovery, rank by incremental gain versus the current plan F, initialized empty. Require finite gain above the configured threshold and non-increasing known CE, both incrementally and against V_empty. Recheck the winner on the next unused confirmation fold under both comparisons. A failed confirmation ends selection. All candidate plans start at V_initial, including later rounds. The actual optimizer executes a single joint update on the full batch, with accepted edits mapped back by row/ID. The probe is held out of the virtual gradient and geometry calculation; it belongs to the real baseline training batch. Independence applies within a refresh, not across epochs or to the teacher identity.

**Virtual-update scope.** Classifier-only needs one cached forward and a small parameter snapshot. It measures a parameter that actually determines predictions. A partial-backbone variant could include the last trainable transformer block and capture local feature movement, but would need identical stochastic forwards, extra activation storage, a full gradient objective and cloned optimizer/buffer state for that block. Updating the complete trainable backbone would more closely approximate the real step at much higher cost per candidate. Neither alternative was used in the measured runs. A positive one-step gain is a prediction about this restricted surrogate.

## Modes, logging and dose

| Mode | Decision and comparison |
| --- | --- |
| E0 | Default upstream training path. Optional run_seed permits a paired initialization. |
| E1 | Uniform randomized eligible choices of the requested type from a randomized boundary shortlist. Requires the matching E3 ledger. This controls choice among feasible edits and is conditional on geometry eligibility. |
| E2 | Evidence ranking only. First supported MOVE at the highest-ranked row, otherwise ABSTAIN. Optional E3 ledger constrains the type and number of edits. |
| E3 | Primary method. Geometry, factual vs corrected virtual updates, independent confirmation and a fixed maximum count. Vetoes may reduce the realized dose to zero. |
| E4 | Same verification with configurable adaptive upper bound. Recompute cumulative targets and recheck remaining MOVE eligibility. Number accepted depends on gain, support, folds, cap and deadline. |

The per-run checkpoints/crish.jsonl records epoch, seed/config signature, discovery and confirmation deltas, cumulative group gains, known-CE guard deltas, action IDs and destinations, candidate and action counts, stop reason, applied edits and synchronized refresh elapsed time. Confirmation fields exist only for discovery winners. An accepted edit counts as applied only if the real optimizer stepped. Match E1/E2 to E3's applied per-epoch sequence of action types, same mass and one exposure per ID. Reject absent epochs, incompatible settings or an unrealizable dose. Matching controls count, timing, types and transferred teacher mass; target entropy, gradient norm, sample identity, evolving weights and actual benefit can still differ.

## Validation and measured cost

Windows, RTX 3050 Ti, Python 3.12.10, PyTorch 2.6.0+cu124, torchvision 0.21.0. Only bounded synthetic work was run. The fixture uses a small linear backbone and the official DINOHead. Fixed synthetic seeds exercise selection. These numbers describe code execution.

| Check | Observed result |
| --- | --- |
| python validate_crish.py | 5/5 passed on CPU, 2.215 s. E3/E1/E2 each applied one ABSTAIN. E4 applied two. |
| python validate_crish.py --cuda | 5/5 passed, 2.535 s. Active-mode validation used CUDA; deterministic reference checks also run on CPU. |
| python smoke_train.py | Actual upstream and adapted train(), evaluation, scheduler and checkpoints on one GPU FP32 batch. E0 parameters and SGD momentum buffers bitwise equal to upstream. E3/E1/E2 each applied one MOVE. E4 applied two. |
| python smoke_train.py --fp16 | Original GradScaler skipped the first synthetic update in both upstream and E0; model states matched. E3 accepted one MOVE, E4 two, applied zero. E1/E2 replayed zero. |
| python smoke_train.py --fp16 --epochs 16 | One synthetic batch per epoch. The initial eight-step attempt ended before any accepted edit affected SGD. With sixteen steps and the stock scaler, E0 and upstream matched bit-for-bit. At zero-based epoch 14, E3/E1/E2 each applied one MOVE and E4 applied two. Other epochs had zero applied dose. |
| Isolation, alignment and gradients | Repeated virtual SGD agrees with a separately stepped classifier including nonzero momentum. Live parameters, optimizer momentum, gradient buffers and RNG checked. Disjoint score/probe IDs, duplicate-ID exclusion, labeled-row protection, teacher mass transfer, zero abstained-row gradient, stable extreme logits and gain-threshold rejection checked. A trap dataset checks image-only access. |

One FP32 train() refresh took 166.41 ms for E3, 16.30 ms for E1, 15.82 ms for E2 and 214.81 ms for E4 on the synthetic GPU fixture. CPU unit-fixture refresh times were 28.18, 5.00, 4.80 and 47.13 ms respectively. These are individual synchronized selection-plus-distillation timings that can include pending GPU work. They do not estimate production DINO wall time, peak memory or an overhead percentage against E0. Record matched whole-run wall time, device, number of scored candidates and stop reasons in benchmark runs.

Unexecuted: full training, real-image DINO dataset runs, per-dataset loader integration, multi-GPU train_mp.py, production peak VRAM, original PyTorch 1.10 runtime, and statistical evaluation of Old/New/All accuracy. The official dataset split builder and evaluator legitimately use annotations. Active CRiSH training receives images, IDs, membership and legal known labels only. No evaluation output is sent to the selector. The source split/evaluation protocol remains a dependency. Use Linux/WSL for the stock CLI experiment directory naming scheme (the upstream timestamp includes a vertical bar). The Windows trainer smoke uses a temporary output path.

## Run subsequent experiments

Use the single-device trainer, configure dataset paths in upstream config.py, provide the official DINO weights/cache and install upstream dependencies compatible with your host. The commands below are Bash commands for Stanford Cars with the settings from scripts/run_cars.sh. Use the matching official script for another dataset. Run each experiment in a fresh upstream output directory. The E0 command is a reference for provenance; reuse an existing compatible reproduction.

~~~bash
COMMON=(--dataset_name scars --batch_size 128 --grad_from_block 11 --epochs 200
  --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5
  --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07
  --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --run_seed 1)
REPAIR=(--crish_warmup 10 --crish_every 5 --crish_fixed 1 --crish_cap 4
  --crish_boundaries 16 --crish_k 5 --crish_probe 4 --crish_seed 1
  --crish_mass 0.2 --crish_gain 0 --crish_seconds 120)
# Reference only: python train.py "${COMMON[@]}" --crish_mode E0 --exp_name cars_e0_s1
python train.py "${COMMON[@]}" "${REPAIR[@]}" --crish_mode E3 --exp_name cars_e3_s1
DOSE=/absolute/path/to/the/completed/E3/run/checkpoints/crish.jsonl
python train.py "${COMMON[@]}" "${REPAIR[@]}" --crish_mode E1 --crish_dose "$DOSE" --exp_name cars_e1_s1
python train.py "${COMMON[@]}" "${REPAIR[@]}" --crish_mode E2 --crish_dose "$DOSE" --exp_name cars_e2_matched_s1
python train.py "${COMMON[@]}" "${REPAIR[@]}" --crish_mode E4 --exp_name cars_e4_s1
# Optional unpaired evidence-only ceiling ablation
python train.py "${COMMON[@]}" "${REPAIR[@]}" --crish_mode E2 --exp_name cars_e2_fixed_s1
~~~

For baseline reuse, verify upstream commit or reviewed equivalent, DINO variant and weight digest, split file and image-ID membership, legal-label fraction and label remapping, transform, sampler, optimizer/LR, batch size, epochs, trainable blocks, loss weights and temperatures, precision, evaluation implementation (v2 here), checkpoint/epoch and metric aggregation. For paired inference also establish initialization, data order, seed and environment. An existing result with unknown seed may be a descriptive baseline only. The E3 signature checks selected arguments, not dataset or checkpoint contents. A compatible historical E0 alone cannot establish that verification beats geometry-only selection. Compare paired E3 against matched E1/E2, report per-seed New/Old/All, actual dose, acceptance by action, guard failures, zero-dose epochs and uncertainty.

Important reviewer concerns include teacher self-confirmation, small known-biased folds, multiple comparisons on a reused discovery fold, and distribution shift from batch-local neighbors. Fresh confirmation reduces selection bias but cannot establish novel-class correctness. A time-limited E4 run can appear to adapt its quantity when it merely exhausts its compute budget. Low realized exposure and classifier-only lookahead can also make a null result difficult to interpret. No benchmark improvement or independence-of-SelEx performance claim follows from the smoke runs.
