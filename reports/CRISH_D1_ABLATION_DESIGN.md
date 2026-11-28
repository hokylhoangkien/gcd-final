# CRiSH-D1 matched-exposure ablations

Original D1 remains at commit `9468369`. The coordinated experiment lives in [dual/ablation.py](../dual/ablation.py), on branch `feat/crish-dual-ablation`. Its reference arm shares the original CLIP proposal algorithm, then accepts the same number of eligible edits as its controls in every shared batch. Compare these arms with each other. The completed historical D1 used its natural exposure and is a descriptive reference.

| Arm | Candidate and source | Destination |
|---|---|---|
| D1-matched | CLIP maximum peer support difference, positive risk, top 10% ceiling | Highest-supported alternative slot |
| D1-Random | Same candidate IDs and snapshot source slots as D1-matched | Uniform choice over slots occupied in the Random model's opposite fold, excluding the source and original CLIP destination |
| D1-DINO | Same support difference, positive-risk gate, ID tie-break and budget from this arm's current normalized DINOv1 final CLS | Highest-supported alternative slot |

CLIP and DINO each use the existing `dual.policy.scores` and complementary query/reference folds. Refreshes use their own current classifier assignments. Random alternatives are sampled once per refresh using the run seed, epoch, and ID. Missing supported alternatives remove the candidate from the matched Clip/Random pair. The random arm measures the value of choosing the CLIP destination conditional on a CLIP-selected eligible source and excluding the original destination. DINO versus D1 changes the evidence for both proposal selection and destination retrieval. The DINO IDs need not be the same as the CLIP IDs.

## Matching actual dose

Each batch is drawn once by the original weighted replacement sampler, with its original two-view crop generation. The three arms consume that same batch in the same order. Each arm independently runs the unchanged source-change and both-view teacher-mass eligibility checks. CLIP and Random keep only their common eligible occurrences, with the same batch row, ID and source. DINO uses its own ranked eligible list. Truncate all three lists to the minimum of these support counts, remaining epoch budgets and the original per-batch cap. The moved mass is 0.2 per occurrence in every arm. There is zero editing in all three arms if an arm supplies no eligible row, including when a refresh times out. No padding, fallback or eligibility relaxation.

Both DINO shortages at refresh and shortages of eligible sampled occurrences are recorded. The journal includes candidate quota and eligibility, per-batch desired/available/applied counts, stale source, low mass, Random support failures and the committed edit rows in each arm. Total moved mass matches exactly. Unique exposure and distinct-ID composition can differ between CLIP and DINO. Matching also conditions the CLIP exposure on opportunities for Random and DINO, which affects the population the controlled contrast describes. Count and mass equality do not establish equality of gradient dose or semantic opportunity.

The worker reuses the original D1 optimizer, scheduler, loss, MOVE and guard. It preserves seed 1, Cars SSB, batch 128, 200-epoch cosine schedule, ten warmup epochs, positive alternative support, 300-second refresh timeout per arm, batch cap 8 and mixed-row epoch ceiling 4%. It previews deterministic train-mode logits for eligibility, then verifies bitwise equality on every edited arm before computing the update. The shared RNG stream advances once per batch. Full epoch boundaries atomically save all three models, momentum, scheduler and RNG together. The committed journal offset is restored before continuation. Official transductive evaluation is isolated from proposal selection.

## Validation and future execution

```bash
python -m unittest dual.test_d1 dual.test_ablation -v
python -m dual.ablation outputs/ablation-smoke "$CLIP_CACHE" --until 1 --steps 1 --micro
# Future coordinated training and resume. Each --until is a completed epoch.
python -m dual.ablation outputs/d1-ablation "$CLIP_CACHE" --until 100
python -m dual.ablation outputs/d1-ablation "$CLIP_CACHE" --until 150
python -m dual.ablation outputs/d1-ablation "$CLIP_CACHE" --until 200
```

Use the same Cars/DINO environment variables and verified CLIP cache as in [D1 design](CRISH_DUAL_BACKBONE_DESIGN.md). Run the coordinated worker under your existing tmux session. Keep the output and recovery file on persistent storage. The `--steps` smoke runs inside the original warmup and never saves partial-epoch state. Unit tests cover actual matched optimizer steps with valid edits and bitwise KEEP parity, candidate shortage without padding, the original label and MOVE guards, separate evidence banks, deterministic random assignments and exact joint checkpoint resume on a small CPU model. The production-scale memory and time footprint of the three-arm harness remains unmeasured. No matched full trajectories were launched in this audit.
