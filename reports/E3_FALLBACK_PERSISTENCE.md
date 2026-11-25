# E3 confirmation fallback and revalidated persistence

Reference method: global E3 at `a63f29c`. The implementation extends the branch after `a70760a` so the existing Cars launcher and checkpoint format remain usable. Both flags default to false. The researcher reported a completed reference run with All/Old/New 56.51/71.51/49.26, 190 refreshes, 151 confirmation vetoes, 19 cap hits, zero timeouts, 455 accepted actions and 250 applied occurrences. Accepted actions count decisions; occurrences count optimizer exposures and may repeat an ID. Unique never-applied actions cannot be calculated by subtracting those two totals.

| Setting | Reference E3 | Extension |
| --- | --- | --- |
| Discovery | Exact virtual classifier SGD for each eligible edit and the current factual plan | Same real SGD snapshot and score objective. For fallback, freeze the complete qualifying ranking in descending marginal gain, with stable ties. Recreate the exact candidate update on demand. |
| Confirmation | One winner per round, stop at first veto | With `--crish_confirmation_fallback`, advance within the frozen ranking after a veto. Exclude the rejected action identity for the rest of the refresh. An acceptance starts a new round with exact cumulative re-scoring. |
| Refresh budgets | Eight accepted actions, eight confirmation folds, 300 seconds | Fallback reserves a fixed family of at most 12 attempts, including winners, retries and a started attempt interrupted by timeout. Maximum eight accepts. Retain the prefix confirmed within the deadline. |
| Inactive actions | End at the epoch boundary | With `--crish_revalidated_persistence`, an accepted action with zero actual applications becomes a nomination for a later refresh. Plans still start empty. |
| Training | Stanford Cars, DINOv1 ViT-B/16, batch 128, WeightedRandomSampler, official optimizer, losses and evaluation | Same training path. Warmup 10, refresh every epoch, evidence 512 × two two-view trials, exact k=5, 64 boundaries, score batch 256, MOVE mass 0.2. |

## Folds, guards and multiple comparisons

Fallback preassigns one discovery fold and twelve confirmation folds before seeing any candidate gains. Every fold has 32 known and 64 unlabeled IDs. Every tested proposal consumes a new fold; a veto, nonfinite score or interrupted attempt cannot release it. Factual, current-plan and proposed updates see the same crops and frozen targets on that fold. Acceptance requires the original strict marginal and cumulative total-gain tests and both known-class no-increase guards. The next candidate is selected solely by its earlier discovery rank. Veto gains cannot reorder the list. Once the list or attempt budget is exhausted, the refresh stops.

On the actual Cars split, 77 legally labeled classes have at least 18 examples, but only 10 have the 26 examples needed to reuse the same 16 classes across all 13 folds. The first nine folds use the original shared 16-class panel. Four additional folds select a second eligible panel from unused known IDs. Each fold retains two distinct examples per known class. The unlabeled pool supplies 832 unique IDs across the 13 folds; carried IDs are excluded from probe selection. Insufficient known, unlabeled or scoring support stops the refresh. The second panel can change which old classes exercise the guard.

The fixed attempt budget, locked ranking, permanent veto per action within the refresh and disjoint folds bound repeated selection. They do **not** establish a family-wise error rate for accepted actions. To audit evidence separately, each finite confirmation reports a one-sided paired sign diagnostic on marginal proxy improvement. Each block combines one legal known class's mean loss with an equal-size block of unlabeled IDs in their preassigned random order. Both views stay with their image. For B blocks and S strictly positive differences, the diagnostic reports the Binomial(B, 0.5) upper tail and min(1, 12p). Ties count as nonpositive. A value below 0.05 for the adjusted diagnostic is interpretable under conditionally independent block signs with null positive probabilities at most one half. Shared features, adaptive training and a finite transductive population limit that interpretation. The diagnostic describes block direction, does not establish improvement in novel-class accuracy, does not control error across epochs, and never relaxes or adds a training acceptance gate.

## Carry lifecycle

An action is identified by sample ID, kind, source and destination. If first accepted at epoch e and never applied by a successful optimizer step, it may compete at e+1, e+2 and e+3. Reacceptance preserves e. At e+4 the nomination expires before sampling. A later ordinary proposal can arise independently. A veto or stale occurrence leaves the nomination inactive and eligible only for the remaining lifetime.

Carried unlabeled IDs are exact global kNN query anchors inside the existing evidence budget. Fresh crops and both reciprocal evidence trials rebuild their geometry. Only a regenerated action with the same identity can reserve a boundary slot. The current two-view teacher must support its source and MOVE mass. It then competes on the exact discovery objective and must pass independent confirmation again before entering the epoch plan. The scoring and training guards are reapplied to the current teacher. A successful optimizer step with at least one edited occurrence retires the nomination. A sampler miss, stale source, insufficient mass or AMP-skipped step does not count. Suite checkpoints retain pending identities and original epochs. Recovery truncates uncommitted ledger output before replay.

Nominating a prior action uses part of the 512-ID evidence and 64-boundary budgets. Fresh candidates can be displaced, and repeated nominations can favor persistent teacher errors. The 300-second deadline can also change which attempts finish. Neither flag modifies the training sampler or guarantees exposure.

## Configuration and records

From the repository root, the following command selects both flags for a future run. Pass either flag alone to enable only that feature. The standalone Cars launcher uses cap 8.

~~~bash
bash scripts/run_cars_crish.sh --crish_confirmation_fallback --crish_revalidated_persistence
~~~

`crish/confirmation.py` handles fresh-fold attempts and the sign diagnostic. `crish/global_repair.py` manages ranking, fold reservations and carry state. `crish/sampling.py` includes carried IDs in exact global neighborhood queries. `crish/runtime.py` registers flags; `crish/experiment.py` omits inactive flags from existing suite identities and `train.py` includes pending nominations in resumable suite checkpoints.

In `crish-global.jsonl`, `confirmation_attempts` records round, discovery rank/gain, action identity, consumed fold, marginal/group confirmation gains, both known deltas, adjusted diagnostic and result. `probe_ids` maps fold numbers to exact IDs. Timeout or nonfinite scores can have unavailable deltas. The refresh-level stop distinguishes discovery, exhausted ranking, attempt budget, cap, support and deadline. `carried`, `carry_geometry_ids`, `carry_teacher_eligible_ids`, `reaccepted` and `expired` describe the nomination funnel. The exposure record includes `applied_ids`, occurrences per type, stale/low-mass/skipped counts where observed, and `carry_pending` with original epochs.

## Bounded validation

`python validate_e3_extensions.py` runs seven CPU tests with the project dependencies installed. The parity test loads the method and its dependencies directly from the pinned Git revision. Across five fixture refreshes all deterministic ledger fields, including candidate decisions and gains, match with both flags off. The two paths also produce bitwise-equal distillation gradients and identical applied-ID/exposure records. Timings are excluded. Other tests exercise a real next-ranked retry on synthetic features, scripted known/gain vetoes and the twelfth-attempt limit, insufficient probe support, carry expiry, AMP skips, retirement after application, the sign diagnostic and deterministic checkpoint recovery with a nonempty carry state.

The [raw Cars smoke ledger](cars_e3_extensions_smoke.jsonl) records two consecutive refreshes with both flags enabled and one real batch of 128 per refresh on an RTX 3050 Ti. It used 8,144 real training images, pretrained DINOv1 and a newly initialized classifier at schedule index 10. The trained model from the researcher's 200-epoch run was not used. This ledger predates the final redundant probe-size summary and explicit null placeholders for interrupted scores; the fold IDs and completed gains are present.

| Schedule index | Time (s) | Attempts | Accepted | Carried / geometry / teacher / reaccepted | Actually applied IDs | Stop |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 10 | 300.27 | 5 | 4 ABSTAIN | 0 / 0 / 0 / 0 | 0 | Timeout during attempt 5; all four confirmed actions retained |
| 11 | 266.58 | 8 | 8 ABSTAIN | 4 / 3 / 3 / 1 | 0 | Acceptance cap |

Both refreshes reserved 1,248 distinct probe IDs. ID 1603 passed verification again at index 11 with its original index 10 retained. Neither training batch encountered an accepted ID. No completed Cars confirmation was vetoed in this short window, so real-data retry frequency is unmeasured; the retry path was observed in the CPU integration test. These observations do not estimate an accuracy or exposure improvement. No full training or ablation was run.
