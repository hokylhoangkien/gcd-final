# CRiSH-D1 bounded smoke and runtime

**Decision.** GO for a guarded exploratory Cars run after preflight and a bounded benchmark in the selected execution mode on the target Linux GPU. The locked starting policy is 10% candidate coverage, mass .2, up to 4% mixed-row exposure per epoch and eight occurrences in a batch. No full training was launched. Accuracy effects and the first post-warmup epoch's eligibility remain unverified.

## Real RTX 3050 Ti pilot

Both pilots used seed-1 Cars, the verified R2-100 model and SGD state, 6,144 unlabeled images for the refresh, audited CLIP vectors, and the first 12 batch-128 draws. The audit used the exact frozen-prefix VJP path in 8-image chunks and a zero-worker loader. The benchmark shape and loss denominator match production. The loader and target GPU timing can differ.

The initial label-blind mass-.8 pilot applied 71 edits on 70 distinct IDs. Its first edited batch had edit-to-KEEP gradient ratios of 3.305 for the classifier and 2.075 for block 11. This motivated keeping E3's .2 mass for the locked setting. The second pilot verified .2 on the same starting snapshot and sampler stream. Neither pilot used ground truth or model accuracy for selection.

| Locked pilot quantity | Observed |
|---|---:|
| Valid positive-risk IDs / proposals | 3,067 / 614 |
| Structurally invalid destinations across bank | 0 |
| Encounters / applied occurrences / unique IDs | 76 / 74 / 71 |
| Stale source / low teacher mass | 0 / 1 |
| Dropped by batch cap / epoch cap in the bounded window | 1 / 0 |
| Moved mass | 14.8 |
| Exposure in the first 1,536 mixed rows | 4.818% |
| Full-epoch budget | 322 of 8,064 mixed rows |

A bounded prefix can exceed the full-epoch exposure fraction. The remaining epoch budget limits subsequent edits. The 4.818% measured on the window is not a full-epoch estimate.

| Batch | Classifier KEEP / edit L2 | Classifier ratio / cosine | Block-11 KEEP / edit L2 | Block-11 ratio / cosine |
|---|---|---|---|---|
| 1 | .05513 / .04555 | .826 / +.005 | .23472 / .12178 | .519 / −.079 |
| 2 | .05958 / .03306 | .555 / +.011 | .23443 / .13130 | .560 / +.089 |
| 3 | .06250 / .02938 | .470 / −.008 | .27950 / .11638 | .416 / −.017 |

The weighted edit gradient is substantial on these batches. It is nearly orthogonal to KEEP for the classifier and changes sign in block 11. This measures gradient strength and direction at a trained snapshot. It does not establish that moving a particular source or training from initialization improves accuracy. E3 and R2 used different intervention scopes, so their weaker historical diagnostic ratios provide context.

## Paired evidence and retrieval stability

On the same 614 CLIP-ranked IDs, using the same source slots, reference folds and maximum-peer support, DINO had 614 valid alternatives. The DINO and CLIP destinations agreed on 105 of 614 IDs (17.1%). Independent rank lists used the same cap of 614 and positive-risk gate. DINO supplied 130 candidates and shared 49 IDs with CLIP. The gate shortfall is retained in the diagnostic.

An independent complementary partition at the same checkpoint retained 403 of 614 top CLIP candidates. The destination agreed on 58.1% of the original candidate IDs. Reference selection is consequently a source of proposal variability. The comparison was label-blind and does not assess which individual disagreements are correct.

## Runtime

| Locked measurement | Time or memory |
|---|---:|
| DINO center snapshot, 6,144 images | 82.0 s |
| CLIP retrieval and planning | 2.45 s |
| Extra paired DINO and alternate-fold diagnostics | 4.75 s |
| 12 training steps including three gradient measurements | 114.3 s |
| Peak CUDA allocation, refresh / training | 768 / 712 MiB |
| Extrapolated active epoch / 200 epochs | 12.4 min / 41.2 h |

The local forecast scales the nine non-diagnostic-step timings to 63 batches, uses a full center snapshot as the evaluation-time proxy, and charges refresh only after the ten-epoch warmup. It excludes archive time and operator pauses. The initial pilot projected about 26.4 hours on the same laptop, so the forecast is sensitive to local runtime throughput. Direct batch-128 training, the batch-256 official evaluator, and complete tmux/SSH operation have not been timed on the intended Linux server. Use the target-mode benchmark in the design before planning wall time.

## Validation

Seven focused tests passed for numerical slot support, disjoint query/bank folds, illegal labels, stale and low-mass handling, probability conservation, exposure budgets, cache ID/split mismatches, zero-edit baseline equality, bitwise optimizer/scheduler/RNG resume with a replayed weighted draw, archive verification, RNG isolation and automatic blocking at manual milestones. Ruff and the launcher's Bash syntax check passed. CPU resume validation used a small fixture through the production checkpoint and export helpers.

On eight real Cars rows from the same checkpoint, the direct zero-edit D1 step matched the existing SimGCD training step bitwise. The 8-image VJP path differed by at most 2.98e-8 in a trainable parameter. The parity pass used saved SGD momentum and weight decay. The actual Linux controller was not launched on this Windows host.

Local benchmark receipts and raw logs are under `dev_outputs/d1-smoke`. Both label-blind benchmarks ran with a 15-minute watchdog. [Design and generic Linux commands](CRISH_DUAL_BACKBONE_DESIGN.md)
