# CRiSH research summary

The newly reproduced E0 transductive v2 baseline is **55.91 / 72.96 / 47.67% All / Old / New** on Stanford Cars at epoch 200. The checkpoint and full training log were audited. The historical E3 endpoint is 56.51 / 71.51 / 49.26%, and the manually recovered E3-F endpoint is 56.49 / 73.36 / 48.35%. R2 ended at checkpoint 150 with 55.21 / 70.66 / 47.74%. The current evidence covers a single seed and different adaptation and training conditions.

On the reused G6-confirm panel at 10% coverage, frozen E0 CLIP correction changes aligned All accuracy by +2.64 pp. The confidence-selected policy with the same CLIP retriever changes it by -2.93 pp. The full report gives matched-random contrasts and paired uncertainty. These are sealed offline assignment changes. The controlled continuation diagnostics separately track exposure, feature response and panel transfer.

- [E0 baseline audit and frozen geometry](CRISH_E0_BASELINE_AUDIT.md)
- [Experiment matrix](CRISH_EXPERIMENT_MATRIX.md)
- [R2 mechanistic audit](CRISH_RELATIONAL_MECHANISTIC_AUDIT.md)
- [Locked held-out CLIP confirmation](CRISH_VLM_HELDOUT_CONFIRMATION.md)
- [Gradient conflict audit](CRISH_GRADIENT_CONFLICT.md)
- [Hard and soft CLIP continuation](CRISH_SOFT_TARGET_TRAINING.md)
