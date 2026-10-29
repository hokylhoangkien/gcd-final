# AGENTS.md — gcd-final

> Master instructions for Codex working in this repository. Read this file completely before editing or running anything.
> Project owner communication language: **Vietnamese**. Scientific documents, technical comments, filenames, tables, and code can use English.
> Last updated: 2026-10-29.

## 0. Mission and non-negotiable research scope

Build a **scientifically defensible Generalized Category Discovery (GCD) research project**, starting from **Hyp-SelEx / HypCD (CVPR 2025)** and investigating a potentially novel **reliability-aware hyperbolic hierarchical learning** extension. The goal is a **general GCD method**, NOT a car-specific classification trick.

**Stanford Cars is the shared comparison benchmark within the user's research group**, not the contribution itself. Optimize and report **All accuracy** as the main comparison metric, while also reporting **Old/Known and New/Novel** performance to detect trade-offs. Additional datasets (e.g., CUB or CIFAR-100) are desirable to test generalization, but only after Stanford Cars is working and resources permit.

**Research status:** Proposed method is a *hypothesis*, not a validated contribution. Do NOT promise to exceed any published baseline, announce SOTA, or invent experiments, numbers, novelty, statistical significance, citations, or publication readiness.

## 1. IMPORTANT: existing folder, code, and running process first

This repository might already contain experiments, datasets, checkpoints, partially completed coding, or a long-running process. **DO NOT overwrite, delete, reclone, restart, or kill anything by default.**

Before implementing:

1. Inspect `pwd`, `git status`, directories, existing `.md` files, scripts/configs, README, experiment logs, process IDs, and checkpoint timestamps.
2. Identify any currently running Codex/Python/training processes **without disrupting them**. On macOS/Linux, inspect `ps -axo pid,ppid,etime,%cpu,%mem,command`, narrow the search to relevant commands, and inspect accessible logs. Use `nvidia-smi` only if available; on macOS note that CUDA may not exist. Never assume a process is stuck solely because it has run for a long time.
3. Produce `reports/00_workspace_audit.md`: current implementation, existing changes, process status, what is missing, system and compute capabilities, current blockers, and recommended next safe action.
4. Continue or integrate existing work if relevant. If existing code differs from this spec, preserve it, document differences, and create a separate branch or clearly isolated module for the new approach.
5. If a process is unhealthy, **explain the evidence and safe stop/resume options**. Do not forcibly kill, cancel, or replace it without explicit approval.
6. Avoid destructive shell commands (`rm -rf`, forced git resets, mass file replacements, process killing). Do not commit datasets, weights, large artifacts, API keys, or credentials.

## 2. Authoritative sources — read before coding

Official baseline:
- HypCD (CVPR 2025) code: https://github.com/Visual-AI/HypCD
- Paper: https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Hyperbolic_Category_Discovery_CVPR_2025_paper.html
- Project: https://visual-ai.github.io/hypcd/
- arXiv: https://arxiv.org/abs/2504.06120

Novelty-risk / related-work checks (read the method, not just abstracts):
- Reliable Confidence Alignment for GCD (ICML 2026): https://proceedings.mlr.press/v306/yu26l.html
- Learning Intrinsic Hierarchy for GCD / LEAH (AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/view/39236
- Hyperbolic Hierarchical Representation Learning for GCD (TNNLS 2025): https://pubmed.ncbi.nlm.nih.gov/40839495/
- Search for additional directly related work before claiming novelty, especially confidence-based pseudo-labeling, hyperbolic metric learning, pair mining, uncertainty, and neighborhood-consistency methods.

**Important repository-specific facts to verify locally:** The public HypCD README mentions Python 3.8/conda, `requirements.txt`, `kmeans_pytorch`, pretrained DINO/DINOv2 weights, `config.py`, Semantic Shift Benchmark (SSB) splits, and scripts that invoke **Slurm `sbatch`**. Do **not** blindly execute Slurm commands on a local machine. Inspect which scripts actually implement Hyp-SelEx and their true flags/paths. Do not invent nonexistent filenames, imports, or CLI parameters.

## 3. Research question and candidate approach

**Working research title:** *Reliability-Aware Hierarchical Representation Learning for Generalized Category Discovery*.

**Question:** Can geometric and augmentation-derived reliability signals reduce noisy unlabeled-pair supervision in hyperbolic GCD, improving balanced known/novel discovery without depending on category-specific prompts or labels?

**Starting point:** Reproduce Hyp-SelEx faithfully. Inspect the actual learning objectives, hyperbolic feature mapping, distance/angle operations, sample mining, self-training, classifier vs. non-parametric evaluation, and any existing confidence or consistency logic. Only then choose a precise insertion point.

**Candidate module, provisional name RAH-GCD (subject to novelty review):**

1. Extract the same image features / hyperbolic embeddings used by the baseline.
2. Estimate **pair/neighborhood reliability without ground-truth labels for unlabeled samples**. Candidate cues: reciprocal top-k neighbors under existing hyperbolic similarity; cross-view neighborhood agreement (two augmentations); normalized local similarity margin; and confidence when *meaningful predictive probabilities actually exist*. If Hyp-SelEx is non-parametric at the selected stage, do **not** pretend softmax classifier confidence exists.
3. Build a bounded reliability score `r_ij` in `[0,1]` and apply it to **only the appropriate unlabeled pair/neighbor terms**, preserving the official supervised loss and the baseline hyperbolic geometry. Use safe normalization and a warm-up/ramp-up schedule. A possible form (NOT mandatory until code inspection):

   `L_pair = sum_{(i,j) in P} r_ij * ell_ij / (sum_{(i,j) in P} r_ij + eps)`

   where `P` is the relevant pair set and `ell_ij` the existing pair loss. Preserve stable gradients, guard against zero weights, NaNs, invalid Poincaré-ball radii, and trivial all-zero weighting.
4. Prevent *confirmation bias* and *known-class domination*: audit reliability distributions separately for labeled/unlabeled sets, measure excluded pairs and novel discovery behavior, and test a warm-up, floor, or quota only if justified.
5. Keep the design **dataset-agnostic and switchable** (`method=baseline|proposed`; individually switchable reliability components). Keep comparable optimization settings whenever possible.
6. **Reject or revise this idea if the novelty review shows it duplicates prior work.** In that case, write a specific differentiation proposal with evidence before implementing a cosmetic rename. Never hide this risk.

Avoid brittle hard-coded car labels, car-specific augmentations, injected known/novel identities, prompt engineering using class lists, or use of unknown ground-truth labels for training-time scoring.

## 4. Evaluation protocol: comparability before scores

- Use the **official Stanford Cars SSB/GCD split and official labeled-data fraction** as implemented in the chosen baseline; inspect and record the actual split files and class counts. Do not assume the data are simply split randomly 98/98 without checking the benchmark implementation.
- Explicitly distinguish **known-class labels available for supervised training** from **unlabeled samples** and **evaluation-only ground truth**. No evaluation-label leakage into training, feature selection, clustering, pseudo-label construction, or hyperparameter tuning.
- Record: dataset version, split file/checksum, number of known/novel classes, labeled fraction, train/eval pool definition, pretrained checkpoint (DINO or DINOv2 variant), feature dimensions, number of clusters/classes used, input size, transforms, loss terms, optimizer, learning rate, epochs, batch size, hardware, seed, evaluation script and matching protocol.
- Official clustering evaluation may use Hungarian matching after predictions to compute accuracy; replicate the baseline's metric protocol **without using evaluation labels to train**. Document exactly how All, Old, and New are computed. Check their sample-weighted relationship rather than assuming All is an unweighted mean.
- A comparison is labeled **fair** only when essential factors match (split, label fraction, pretrained backbone, features, protocol, class-count assumption, and evaluation metric). Do not present scores from different backbones as pure algorithmic gains.
- Primary metric: **All (%)**; also Old (%) and New (%). For multiple seeds report mean ± std and exact per-seed values, not only the best run.
- Results should be **observed**, **paper-reported**, or **not yet run** — clearly distinguished. Never fill empty result cells with made-up estimates.

## 5. Implementation plan: execute in safe, verifiable stages

### Stage A — Audit and reproduce

- Inspect local workspace; identify and preserve active work. If HypCD does not exist locally, fetch the official code **only when network access and permissions allow** into a clearly named subfolder or tracked dependency; pin git commit SHA. Never assume a network download succeeded.
- Read all relevant Hyp-SelEx configs/scripts and map end-to-end: data -> transforms -> backbone -> hyperbolic mapping -> losses -> clustering/prediction -> evaluation.
- Create `reports/01_baseline_protocol.md` with exact paths, commands, paper vs code discrepancies, and reproducibility caveats.
- Check licenses, dataset availability, pretrained weights, dependencies, Python/CUDA/MPS/CPU compatibility. The upstream repo may require adapting Slurm `sbatch` scripts into local runnable commands; preserve their hyperparameters.
- Add a **fast data integrity check** (counts, image paths, duplicates as appropriate, disjoint labeled/unlabeled membership definitions, known/novel class assignments, sample IDs). Protect against data leakage; do not fabricate missing split or images.
- **Minimal smoke test first**: imports, CPU one-batch forward/loss/backward (if supported) and small evaluation fixture, explicit timeout, then brief bounded training test. A CPU smoke test is not a valid full benchmark result.
- Reproduce the unchanged Hyp-SelEx baseline on Stanford Cars **only when dataset, weights, compute, and runtime are ready**. Record exact logs and checkpoints. If hardware is insufficient, finish implementation/tests and report the blocker rather than starting an unbounded training run.

### Stage B — Implement candidate method

- Isolate new components in readable, typed modules where practical; add config toggles and keep an untouched baseline path.
- Respect pretrained backbone compatibility and existing hyperbolic manifold implementation. Avoid reimplementing Poincaré operations if the repo already has stable versions.
- Implement reliable neighborhood selection, bounded weights, and warm-up based on findings in `reports/02_novelty_and_method.md`.
- Add unit tests for score bounds, symmetry or documented directionality, no invalid neighbors, normalization, zero-neighbor fallback, label masking, NaN/Inf prevention, gradient flow and baseline toggle parity.
- Instrument relevant diagnostics (confidence/reliability histograms, active-pair fraction, neighbor agreement, per-epoch All/Old/New if the official protocol allows it).

### Stage C — Ablation and evidence

Minimum controlled variants:

| Variant | Description |
|---|---|
| B0 | Official unchanged Hyp-SelEx |
| B1 | Baseline + neighborhood reliability only |
| B2 | Baseline + cross-view stability only |
| B3 | Baseline + both reliability components + justified warm-up |

Additional experiments, if feasible: fixed vs. adaptive weighting, different k, and sensitivity to curvature/similarity threshold (only where relevant). If some component is not implementable due to code structure, document why and propose the nearest valid test.

Use at least 3 matched seeds **when compute permits**; otherwise explicitly describe preliminary single-seed results. Compare All, Old, New, mean±std, training time, memory peak, and key hyperparameters. Preserve negative and null results.

### Stage D — Manuscript-ready outputs

Produce publication-quality **evidence and writing materials**, not a claim of acceptance:

- Problem formulation, motivation, specific novelty vs. closest prior work.
- Algorithm/pseudocode with correct symbols and losses matching implemented code.
- Reproducible experimental setup and fair baselines.
- Main table, ablation table, efficiency table, training curves, Old/New trade-off, and optional embedding visualizations with honest captioning.
- Limitations, failure cases, compute constraints, and threats to validity.
- Reference list with verified titles/authors/venues/links; never invent bibliographic entries.
- Prefer methodology that can later be validated on at least one additional GCD dataset; do not claim broad generalization from only Stanford Cars.

## 6. Proposed repository deliverables (adapt to existing layout)

Create or update these only where useful; do not duplicate existing equivalents:

```text
AGENTS.md                             # This instruction file
README.md                             # Reproducible quickstart and status
configs/                              # Baseline / proposed / ablation configs
scripts/                              # Local smoke, train, evaluate, ablation runners
src/ or existing upstream modules/     # Minimal tested method extension
tests/                                # Unit and protocol tests
reports/00_workspace_audit.md         # Existing processes, environment, code inventory
reports/01_baseline_protocol.md       # Exact benchmark protocol and commands
reports/02_novelty_and_method.md      # Literature comparison, math, insertion point
reports/03_experiments.md             # Actual runs, tables, failures, compute
reports/04_paper_outline.md           # Evidence-grounded paper plan
results/                               # JSON/CSV per-seed metrics (if runs completed)
logs/                                  # Run logs (gitignored as appropriate)
checkpoints/                           # Checkpoints (gitignored)
```

Every reported run should have a unique run ID with config snapshot, git SHA, seed, environment, command, timestamp, log path, and metrics. Add `.gitignore` to avoid tracking dataset images, weight files, cache, checkpoints, and private/local paths.

## 7. Long-running training and process policy — prevent endless jobs

- Before starting anything expensive, print estimated **resource needs and actual** hardware availability, dataset/weight readiness, selected command, log path, and bounded stop condition. Do not claim expected time to completion as a promise.
- Use **foreground bounded smoke tests first**. For full training, use the real script's finite epoch/step options, configurable maximum runtime, periodic progress logs, and checkpoint/resume. Avoid `while true`, repeated download loops, indefinite polling, and hidden detached jobs.
- If a process is already running, inspect health/progress first; don't start a competing training process or kill the existing one. Seek permission before terminating or replacing it.
- Write stdout/stderr to discoverable logs, surface PID and progress periodically, and report the last completed epoch/step and actual metrics. Handle failure states (missing CUDA, OOM, missing data, dependency conflicts) explicitly.
- Do not run a costly multi-seed sweep automatically unless available compute and user's intent clearly permit it. A usable codebase plus verified smoke tests is preferable to hanging indefinitely.

## 8. Completion checklist / definition of done

**Code/research readiness** (may be completed without GPU):

- [ ] Existing work and active processes inventoried, without destructive changes.
- [ ] Relevant Hyp-SelEx path and baseline protocol mapped and written down.
- [ ] Dependency/data readiness known; missing items listed precisely.
- [ ] Baseline and proposed paths switchable; unit and smoke tests pass.
- [ ] Novelty assessment written with direct comparison to RCA, LEAH, HypGCD and the most relevant additional papers.
- [ ] All/Old/New evaluation matches reference protocol and has leakage safeguards.
- [ ] Logs, reproducible commands, configs, and run metadata present.

**Empirical claim readiness** (only if full experiments actually finish):

- [ ] Reproduced baseline documented; discrepancy from paper investigated.
- [ ] Proposed method measured with matched evaluation conditions.
- [ ] Ablation and per-seed results logged; variability and negative results shown.
- [ ] Paper tables/figures generated from real files rather than typed-in guesses.
- [ ] Cross-dataset generalization claim made only after running additional datasets.

## 9. Communication requirements — essential

1. Respond in **Vietnamese**, concise but technically precise.
2. Begin with the current audit status, active process findings, baseline readiness, and an explicit plan. Then perform safe useful work immediately; do not ask questions already answerable by inspecting the project.
3. After meaningful steps, state actual changed files, executed commands, passed/failed tests, resource blockers, and next exact command.
4. Never say “training finished”, “paper ready”, “beats SOTA”, or “novel contribution” without corresponding evidence.
5. When blocked by resources or missing datasets, stop gracefully with fully prepared code/tests/docs and a precise action list; **do not spin endlessly or invent outputs**.

## 10. Immediate instruction for this session

**FIRST ACTION:** inventory the current `gcd-final` directory, git status, active processes, logs, and available hardware. Write `reports/00_workspace_audit.md`. **SECOND:** inspect relevant HypCD/Hyp-SelEx implementation and official paper, document exact Stanford Cars protocol and novelty risks. **THIRD:** make the smallest baseline-preserving, testable improvement supported by that audit. Run only bounded local smoke tests until benchmark prerequisites are verified. Do not delete existing work or initiate a long GPU sweep without evidence of readiness.

## 11. Chiến lược develop locally, train remotely (bổ sung 2026-10-29)

- MacBook Air M5/16GB chỉ dùng phát triển, unit/static tests, synthetic smoke và batch nhỏ có timeout; CPU mặc định, MPS chỉ khi tương thích. Không training dài, sweep hoặc tải pretrained weights lớn trên Mac.
- Huấn luyện chính thức trên Linux + NVIDIA CUDA. Dùng cấu hình device và đường dẫn tương đối/biến môi trường; không đưa đường dẫn máy cá nhân vào code/config portable.
- Giữ upstream HypCD nguyên vẹn; phần runtime/cải tiến tách biệt, config bật/tắt. Không tuyên bố proposed đã dùng được khi novelty gate chưa đạt.
- Chuẩn bị môi trường CUDA, bootstrap, train/evaluate, seed, checkpoint/resume, log, metrics All/Old/New và điều kiện dừng. Chuyển code bằng Git/submodule, không commit data/weights/artifacts.
- Chỉ xác nhận CUDA sau tensor forward/backward thật trên NVIDIA. Trước training phải qua preflight dữ liệu/weights/import/CUDA và bounded smoke; không bịa benchmark.
