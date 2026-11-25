# Stanford Cars E3-F recovered results

**Provenance.** The researcher manually recovered these aggregate figures from terminal outputs. The E3-F server was terminated before artifact download. The checkpoint and raw per-refresh JSONL are permanently unavailable. These are transcribed results; their arithmetic can be checked, but the run cannot be independently verified from raw artifacts.

| Run | Value |
|---|---|
| Dataset | Stanford Cars |
| Seed | 1 |
| Epochs | 200/200 |
| Commit | 0b992da6e165bcfec02a4566046e051a29c968d6 |
| Confirmation fallback | ON |
| Revalidated persistence | OFF |
| Final All / Old / New | 56.49 / 73.36 / 48.35 (%) |

| Metric | Recovered value |
|---|---:|
| Refreshes | 190 |
| Accepted | 964 (708 ABSTAIN, 256 MOVE) |
| Applied occurrences | 549 (416 ABSTAIN, 133 MOVE) |
| Unique applied IDs summed per epoch | 422 |
| Confirmation attempts | 1,597 |
| Gain veto | 597 |
| Known veto | 36 |
| Stop on confirmation budget | 68 |
| Stop on accepted cap | 56 |
| Stop on discovery | 47 |
| Stop on ranking exhausted | 19 |
| Training rows | 1,532,160 |
| Optimizer steps | 11,970 |
| Encounters | 616 |
| Stale source | 65 |
| Low teacher mass | 2 |
| Timeout | 0 |
| Mean refresh runtime | 49.582867 s |
| Max refresh runtime | 54.466289 s |
| Total overhead | 2.616874 h |

| Consecutive refresh window | Accepted | Applied occurrences |
|---|---:|---:|
| First 99 | 696 | 366 |
| Last 91 | 268 | 183 |

The 422 is a sum of within-epoch unique counts; the number of globally distinct applied IDs is unknown. The totals reconcile: 964 + 597 + 36 = 1,597 attempts, 549 + 65 + 2 = 616 encounters, and 68 + 56 + 47 + 19 = 190 refreshes. Mean duration times 190 gives 2.616874 hours after rounding. Per-attempt gains, discovery ranks, class-block signs, per-ID exposure, trained geometry and gradient trajectories cannot be reconstructed from these aggregates. See the [mechanistic audit](CRISH_MECHANISTIC_AUDIT.md) for the comparison with independently verified E3 data.
