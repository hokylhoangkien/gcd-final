# Pairwise D1 exposure and independent geometry verification

## Feasibility runs

The two paired experiments start separately from official DINO initialization. Each has its own CLIP reference trajectory, current assignments, model weights, optimizer, scheduler, RNG checkpoint, journal, run lock and export directory. Each uses a single sampler and crop stream shared by the two arms. CLIP–Random intersects live eligible occurrences by batch row, ID and source, and uses different supported destinations. CLIP–DINO truncates both ranked lists to their smaller live count. Both preserve the original budget, teacher mass, MOVE mass 0.2, warmup and epoch clock. The previous three-arm mode remains the default.

With the [Linux environment](CRISH_D1_ABLATION_DESIGN.md) configured, run the following commands separately on a single GPU. Each command launches one stage and returns to the shell.

```bash
D1_ABLATION_PAIR=clip-random bash scripts/run_d1_ablation.sh 20
# After the first session exits, start the independent second experiment.
D1_ABLATION_PAIR=clip-dino bash scripts/run_d1_ablation.sh 20
# Continue the requested pair once the milestone has been reviewed.
D1_ABLATION_PAIR=clip-random bash scripts/run_d1_ablation.sh 50
```

Use the same pair and execution mode to resume after interruption. Pair directories under `D1_ABLATION_ROOT` are named `clip-random` and `clip-dino`. Each pair stops at 20, 50, 100, 150 and 200. Epoch 20 is an optional first stop for a new three-arm run. The existing immutable TAR publication and 200-epoch LR schedule apply at epoch 20. Keep already-started runs pinned to their original code revision for exact resume.

Each pair journal records per-batch live supply, matched count, KEEP and edit-only gradient norms, their ratio and cosine for classifier and block 11. Epoch records report unique edited IDs, total moved mass, mixed-row exposure, and candidate and live shortages. Milestone receipts aggregate those fields and deduplicate IDs across epochs. Gradient dose sums the raw norms of the weighted MOVE-minus-KEEP loss gradient on edited batches. This convention measures the strength of the supervision perturbation before SGD momentum, weight decay and learning-rate scaling. It does not equal parameter displacement. The two CLIP references can receive different realized exposure; comparisons use the reference belonging to each pair.

The bounded pilot starts both pairs separately from the verified R2-100 recovery state, reuses the audited full-bank refresh proposals by SHA-256, and executes two real shared batches per pair. These batches sample a small part of the active epoch; their exposure can exceed the whole-epoch percentage cap. The pilot writes `reports/D1_PAIR_PILOT.json` at successful completion. It does not forecast performance at epoch 20.

## Frozen CLIP readout

The frozen OpenAI CLIP ViT-B/16 supplied 6,144 verified unlabeled embeddings. The same encoder and center-view transform supplied 2,000 legal anchor embeddings. Spherical K-means uses 196 centers, fixes legal anchor assignments during fitting, and allows the unlabeled training bank to participate transductively. It freezes predictions for all three predeclared clustering seeds before v2 Hungarian scoring. Mean All / Old / New accuracy was **45.63 / 53.46 / 41.85%**. These seeds vary center initialization on a single dataset split. [Per-seed results](D1_CLIP_SEMI_KMEANS.json)

## Independent counterfactual diagnostic

Two mixed batches of 128 per snapshot supplied up to four unique, live-eligible CLIP MOVE candidates. Each batch uses disjoint training IDs, a 3,072-ID reference bank, two probe folds (32 unlabeled and 16 legal Known each), and 64 separate semantic evaluation IDs. CLIP ranks candidate risk and supplies the destination. Source agreement, both-view source mass and the original distillation denominator govern eligibility.

The verifier freezes the ten nearest **DINOv2** reference IDs of each probe, ranked by cosine. For probe i, direction d_i is the mean checkpoint DINOv1 unit feature of neighbors 1–5 minus that of neighbors 6–10. The proxy score after a virtual update is the average probe dot product between its updated unit feature and d_i. This tests whether a local external neighborhood ordering strengthens under a MOVE. Near neighbors ranked 6–10 can share the true class, which limits semantic specificity.

For KEEP and every candidate the diagnostic reloads identical checkpoint parameters and saved SGD state. It computes the full original SimGCD gradient and applies one virtual SGD step to classifier direction weights and block 11. Projector gradients contribute through the backbone. The differential is the weighted change in distillation loss from transferring mass 0.2. The highest discovery gain alone proceeds to the final decision. Both discovery and confirmation must exceed 1e-7 versus KEEP, and legal Known CE must not increase on either fold. There is no fallback. Direct takes the highest CLIP-risk eligible candidate and Random draws one eligible candidate on exactly the batches accepted by CF. All policies KEEP on rejected batches.

Action decisions and all independent evaluation features were frozen before semantic scoring. Evaluation uses a disjoint probe's cosine margin between its true class centroid and the hardest competing class centroid in the reference bank. Category labels enter this evaluation only, apart from legal Known labels used for the training objective, its guard and the anchored clustering readout.

| Snapshot | Candidates across two batches | Accepted | CF / Direct / Random semantic margin change on the accepted batch |
|---|---:|---:|---|
| D1-100 | 8 | 0 | No selected edits |
| D1-150 | 8 | 1 | −1.645e−6 / −1.589e−5 / +7.394e−7 |
| D1-200 | 8 | 0 | No selected edits |
| E0-200 | 7 | 0 | No selected edits |

One acceptance among 31 candidates gives insufficient evidence for policy superiority. The accepted MOVE had a small negative sealed margin gain. Late-checkpoint cosine learning rates, float32 virtual features, fixed reference directions, one random draw, and a single transductive trajectory constrain interpretation. One-step changes do not measure continued training or generalization. The verifier remains offline. [Per-candidate gains, guards and matched controls](D1_GEOMETRY_CF.json)

## Reproduction

Run from the repository root with the Cars assets and verified audit caches from the previous audits. The watchdog records the child PID, timeout and exit code. Give each GPU job a separate execution window.

```bash
mkdir -p dev_outputs/d1-pair-diagnostics
python -u -m scripts.audit_r2_watchdog dev_outputs/d1-pair-diagnostics/clip.status 1200 scripts.audit_clip_semikmeans
python -u -m scripts.audit_r2_watchdog dev_outputs/d1-pair-diagnostics/cf.status 1200 scripts.audit_d1_geometry_cf
python -u -m scripts.audit_r2_watchdog dev_outputs/d1-pair-diagnostics/pair.status 1200 scripts.audit_d1_pair_exposure
```

Focused tests cover pair-specific matching, bitwise joint resume through an edit and torn journal, epoch-20 export idempotence, strict geometry confirmation and the existing D1/three-arm invariants. Full training and an actual Linux epoch-20 trajectory remain unexecuted.
