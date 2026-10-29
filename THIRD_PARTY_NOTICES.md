# Third-party code and provenance

`upstream/HypCD` is an unmodified Git submodule from https://github.com/Visual-AI/HypCD,
pinned at `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73`.

`gcd_runtime/trainer.py` derives its reference losses, pseudo-label hierarchy,
training loop and evaluation routines from `train/train_HypSelEx.py` at that commit.
Copyright (c) 2025 Visual AI Lab. The full MIT notice is retained at
`LICENSES/HypCD-MIT.txt`. Project modifications add explicit runtime hooks,
reliability weighting, portable invocation, checkpoint/resume and a preregistered
final-epoch checkpoint policy. Reuse is attributed; these reference components
are not claimed as original research contributions.

Backbones, manifold operations, dataset loaders and evaluation helpers are imported
from the pinned upstream submodule and retain their original notices, including
Meta's notices in the transformer modules. Dependency packages have their own licenses.
