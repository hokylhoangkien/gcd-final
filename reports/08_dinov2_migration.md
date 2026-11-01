# DINOv2 migration

2026-10-31: User explicitly requests switching to DINOv2. Separate portable source release; do not update source underneath the v1 process. Stop only the owned v1 launcher using Ctrl+C in its tmux pane, wait for shell prompt, and preserve existing epoch checkpoint/logs. An interrupted epoch is lost; no v1 checkpoint may resume as v2.

Architecture follows pinned upstream train_HypSelEx: models.vision_transformer2.vit_base, patch14, four register tokens, 768 dimensions, pretrained image size518 with positional interpolation to training224. Strict state_dict loading. Official Meta checkpoint URL is used. Runtime config/gate/probe now route to v2 weights; v1 remains supported with lazy v2 import.

xformers0.0.28.post1 declares torch2.4.1 in PyPI metadata. Install wheel only with no dependency replacement, then pip check. XFORMERS_DISABLED=1 selects upstream attention fallback consistently; this is a runtime choice to disclose for comparisons, not evidence of exact paper reproduction. No upstream source changed.

scripts/start_dinov2_linux.sh refuses occupied GPU, uses a separate venv, downloads correct weights, runs real CUDA/image/reliability gate, then one active-reliability full-batch epoch limited1800s before full200 epochs/max86400s. Trial warmup/ramp0 is operational only; final config preserves10/20 schedule and starts from pretrained weights. V1 files remain separate. Do not resume v1 into v2. CUDA v2 performance and VRAM remain unverified locally; any gate/trial failure stops script before full run.

Validation: original26 CPU tests passed via scripts/run_smoke.py; two added tests pass for matching v1/v2 optimization settings and requiring v2 weights even if v1 weights exist. These are not CUDA/backbone-v2 tests. Shell syntax and full suite checked before packaging. No local weights downloaded, no remote process stopped by agent. Paper DINOv2 numbers remain paper-reported and require protocol matching; no gain or novelty claim.
