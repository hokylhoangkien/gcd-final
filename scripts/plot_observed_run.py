"""Build static paper figures from an observed local CRiSH run; never train.

Missing epochs remain missing, protocols stay separate, and checkpoint-selected
scores are explicitly secondary. Matplotlib is imported only when rendering.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SS_PROTOCOL = "balanced_ss_kmeans_train_unlabeled"
METRICS = ("all_acc", "old_acc", "new_acc")
COLORS = {"all_acc": "#1764A0", "old_acc": "#C47A12", "new_acc": "#16816A",
          "move": "#1764A0", "abstain": "#C47A12", "dose": "#7654A3"}
FIGURES = (
    "01_learning_dynamics", "02_intervention_dose", "03_checkpoint_sensitivity",
    "04_probe_and_stopping", "05_recorded_compute",
)


def records(path: Path):
    """Stream the large CRiSH log instead of retaining prototype arrays."""
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid/incomplete JSON") from exc


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite/non-numeric {name}: {value}")
    return float(value)


def validate_ss(row: dict) -> None:
    if (row.get("status") != "observed" or row.get("protocol") != SS_PROTOCOL
            or row.get("unit") != "fraction" or row.get("matching") != "hungarian_v2_global"):
        raise ValueError("Require observed balanced SS, fraction units and global Hungarian-v2")
    for name in METRICS:
        if not 0 <= finite(row[name], name) <= 1:
            raise ValueError(f"Out-of-range {name}")
    n, old, new = (row[k] for k in ("n_unlabeled", "n_old", "n_new"))
    if any(type(v) is not int or v <= 0 for v in (n, old, new)) or old + new != n:
        raise ValueError("Invalid evaluation pool counts")
    weighted = (old * row["old_acc"] + new * row["new_acc"]) / n
    if not math.isclose(weighted, row["all_acc"], abs_tol=1e-10, rel_tol=0):
        raise ValueError("All does not match the sample-weighted Old/New relationship")
    if type(row.get("epoch")) is not int or row["epoch"] < 0:
        raise ValueError("Invalid zero-based epoch")


def ss_history(rows: list[dict]) -> list[dict]:
    """Reject mixed protocols rather than silently replacing missing SS."""
    history = {}
    reference = None
    for row in rows:
        validate_ss(row)
        if row.get("purpose") not in ("monitor_only", "final"):
            raise ValueError("Selected checkpoint reports cannot enter the learning curve")
        signature = tuple(row.get(k) for k in
                          ("seed", "n_unlabeled", "n_old", "n_new", "ss_n_init", "ss_max_iter"))
        if reference is not None and signature != reference:
            raise ValueError("Evaluation conditions changed inside the SS history")
        reference = signature
        epoch = row["epoch"]
        if epoch in history and any(row[k] != history[epoch][k] for k in METRICS):
            raise ValueError(f"Conflicting SS reports at epoch={epoch}")
        history[epoch] = row
    return [history[e] for e in sorted(history)]


def gap_series(rows: list[dict], key: str, epochs: int, scale: float = 1) -> list[float]:
    """NaNs interrupt lines; no interpolation across missing measurements."""
    values = [math.nan] * epochs
    for row in rows:
        if not 0 <= row["epoch"] < epochs:
            raise ValueError("Epoch outside the configured horizon")
        values[row["epoch"]] = finite(row[key], key) * scale
    return values


def identity(action: dict) -> tuple:
    return tuple(action.get(k) for k in ("version", "sample_id", "level", "kind", "source", "target"))


def unique_epochs(rows: list[dict], name: str) -> dict[int, dict]:
    result = {}
    for row in rows:
        epoch = row["epoch"]
        if epoch in result:
            raise ValueError(f"Duplicate {name} at epoch={epoch}")
        result[epoch] = row
    return result


def extract_refreshes(path: Path, doses: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    refreshes, actions = [], []
    for raw in records(path):
        epoch = raw["epoch"]
        accepted = [d for d in raw.get("decisions", []) if d.get("status") == "accepted"]
        if (len(accepted) != raw.get("accepted")
                or (accepted and raw.get("status") != "completed")
                or raw.get("applied_actions", len(accepted)) != len(accepted)):
            raise ValueError(f"Accepted/applied action mismatch at epoch={epoch}")
        if any(d["kind"] not in ("move", "abstain") for d in accepted):
            raise ValueError("Unexpected accepted action type")
        dose = doses.get(epoch)
        # The CRiSH record also stores the net before/after snapshot dose.
        if dose is None:
            dose = raw.get("dose")
        if dose is None:
            raise ValueError(f"No observed dose for refresh epoch={epoch}")
        if raw.get("dose") is not None:
            if any(raw["dose"].get(k) != dose.get(k) for k in ("changed_ids", "levels")):
                raise ValueError(f"Conflicting net dose at epoch={epoch}")
        levels = {d["level"]: d for d in dose["levels"]}
        if 0 not in levels:
            raise ValueError("Missing base level dose")
        level0 = levels[0]
        count = Counter(d["kind"] for d in accepted)
        seconds = finite(raw["seconds"], "refresh seconds")
        if seconds < 0:
            raise ValueError("Negative refresh seconds")
        refreshes.append(dict(epoch=epoch, epoch_number=epoch + 1, accepted=len(accepted),
            move=count["move"], abstain=count["abstain"], cap=raw.get("acceptance_cap"),
            stop_reason=raw.get("stop_reason", "unrecorded"), seconds=seconds,
            changed_ids=dose["changed_ids"], changed_labeled_ids=dose["changed_labeled_ids"],
            level0_changed_directed_pairs=level0["changed_directed_image_positive_pairs"],
            level0_target_mass_before=level0["unweighted_target_mass_before"],
            level0_target_mass_after=level0["unweighted_target_mass_after"],
            level0_target_mass_removed_percent=100 * (level0["unweighted_target_mass_before"]
                                                      - level0["unweighted_target_mass_after"]),
            changed_other_levels=sum(d["changed_directed_image_positive_pairs"]
                                     for level, d in levels.items() if level != 0)))
        for decision in accepted:
            matches = [g for g in raw.get("gains", []) if identity(g) == identity(decision)]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely match accepted action gain at epoch={epoch}")
            gain = matches[0]
            discovery, confirmation = gain.get("exact_discovery"), gain.get("confirmation")
            if discovery is None or confirmation is None:
                raise ValueError("Missing accepted discovery/confirmation gain")
            actions.append(dict(epoch=epoch, epoch_number=epoch + 1, action=decision["kind"],
                sample_id=decision["sample_id"], level=decision["level"],
                source=decision.get("source"), target=decision.get("target"),
                discovery_gain=finite(discovery["gain"], "discovery gain"),
                confirmation_gain=finite(confirmation["gain"], "confirmation gain")))
    unique_epochs(refreshes, "refresh")
    return sorted(refreshes, key=lambda r: r["epoch"]), actions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence(run: Path) -> dict:
    cfg, metadata = read_json(run / "config.json"), read_json(run / "metadata.json")
    if cfg.get("dataset_name") != "scars":
        raise ValueError("This paper figure template is specifically labeled Stanford Cars")
    if metadata.get("status") != "completed":
        raise ValueError("Archive must have metadata.status=completed")
    if (cfg.get("method") != "crish" or not cfg.get("crish", {}).get("enabled")
            or cfg.get("population_policy", {}).get("enabled")
            or cfg.get("pair_reliability", {}).get("enabled")):
        raise ValueError("This figure set describes the isolated local CRiSH variant")
    train = list(records(run / "metrics.jsonl"))
    if any(r.get("status") != "observed" for r in train):
        raise ValueError("Synthetic/non-observed training records are not paper evidence")
    unique_epochs(train, "training record")
    train.sort(key=lambda r: r["epoch"])
    epochs = cfg["epochs"]
    if [r["epoch"] for r in train] != list(range(epochs)):
        raise ValueError("Require the complete observed training history")
    for r in train:
        for name in ("loss", "seconds", "peak_cuda_allocated_bytes"):
            finite(r[name], name)
        if r["seconds"] < 0 or r["peak_cuda_allocated_bytes"] < 0:
            raise ValueError("Negative measured time/memory")
    ss = ss_history(list(records(run / "epoch_metrics.jsonl")))
    final = read_json(run / "evaluation_last.json")
    validate_ss(final)
    if final.get("purpose") != "final" or final["epoch"] != epochs - 1:
        raise ValueError("Primary report must be the fixed final epoch")
    if final.get("selected_using_evaluation_labels") is not False:
        raise ValueError("Primary final report must not be selected by evaluation labels")
    if not ss or any(final[k] != ss[-1][k] for k in ("epoch", *METRICS)):
        raise ValueError("Final report disagrees with the observed SS curve")
    doses = unique_epochs(list(records(run / "intervention_diagnostics.jsonl")), "dose")
    refreshes, actions = extract_refreshes(run / "crish.jsonl", doses)
    if any(d["changed_labeled_ids"] for d in doses.values()):
        raise ValueError("Labeled image assignments/masks changed; investigate before paper rendering")
    if any(r["changed_other_levels"] for r in refreshes):
        raise ValueError("This level-0 dose figure needs extending for repairs at other levels")
    # Selected checkpoints retain their own paired Old/New values and protocol.
    endpoints = []
    for name, label, role in (
        ("evaluation_last.json", "Fixed final", "primary"),
        ("evaluation_best_old_test.json", "Selected by test Old", "secondary_test_selected"),
        ("evaluation_best_ss_all.json", "Post-hoc max SS All", "posthoc_ss_selected"),
    ):
        row = read_json(run / name)
        validate_ss(row)
        if any(row[k] != final[k] for k in ("seed", "n_unlabeled", "n_old", "n_new")):
            raise ValueError("Checkpoint comparison has mismatched seed/pool")
        if role != "primary" and row.get("selected_using_evaluation_labels") is not True:
            raise ValueError("Secondary checkpoint selector provenance is inconsistent")
        endpoints.append(dict(label=label, role=role, epoch_number=row["epoch"] + 1,
                              **{k + "_percent": 100 * row[k] for k in METRICS}))
    for name in METRICS:
        peak = max(ss, key=lambda r: r[name])  # sorted; earliest tie wins
        if name == "all_acc":
            if endpoints[-1]["epoch_number"] != peak["epoch"] + 1:
                raise ValueError("Stored max-SS-All checkpoint disagrees with available history")
            if any(not math.isclose(endpoints[-1][k + "_percent"], 100 * peak[k], abs_tol=1e-10)
                   for k in METRICS):
                raise ValueError("Stored best checkpoint does not preserve its paired Old/New scores")
    train_map = {r["epoch"]: r for r in train}
    for refresh in refreshes:
        if refresh["seconds"] > train_map[refresh["epoch"]]["seconds"]:
            raise ValueError("Refresh timer exceeds its enclosing recorded epoch timer")
    sources = ["config.json", "metadata.json", "readiness.json", "metrics.jsonl",
               "epoch_metrics.jsonl", "intervention_diagnostics.jsonl", "crish.jsonl",
               "evaluation_last.json", "evaluation_best_old_test.json", "evaluation_best_ss_all.json"]
    return dict(config=cfg, metadata=metadata, readiness=read_json(run / "readiness.json"),
                train=train, ss=ss, doses=doses, refreshes=refreshes, actions=actions,
                endpoints=endpoints, sources={name: sha256(run / name) for name in sources})


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No observed rows to export: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7C8792", "axes.labelcolor": "#263544",
        "text.color": "#263544", "xtick.color": "#52616E", "ytick.color": "#52616E",
        "axes.axisbelow": True, "grid.color": "#E1E6EB", "grid.linewidth": .55,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return plt


def save_figure(fig, destination: Path, name: str) -> None:
    for extension in ("pdf", "svg", "png"):
        fig.savefig(destination / f"{name}.{extension}", dpi=320, bbox_inches="tight", pad_inches=.08)


def render(e: dict, out: Path) -> None:
    plt = configure_plotting()
    import numpy as np
    cfg, ss, train, refreshes = (e[k] for k in ("config", "ss", "train", "refreshes"))
    epoch = np.arange(1, cfg["epochs"] + 1)
    subtitle = f"Stanford Cars · local adaptive · seed {cfg['seed']}"
    # 1. Raw learning dynamics; a real blank at absent SS epochs.
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.05), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    for key, label in zip(METRICS, ("All", "Old / Known", "New / Novel")):
        ax.plot(epoch, gap_series(ss, key, cfg["epochs"], 100), color=COLORS[key],
                lw=1.4 if key == "all_acc" else 1.05, label=label)
        ax.scatter([cfg["epochs"]], [100 * ss[-1][key]], s=20, color=COLORS[key], zorder=5)
    if ss[0]["epoch"] > 0:
        ax.axvspan(.5, ss[0]["epoch"] + .5, color="#EDF0F3")
    ax.set(title="(a) Observed SS discovery accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.set_xlim(.5, cfg["epochs"] + 2)
    values = [100 * r[k] for r in ss for k in METRICS]
    ax.set_ylim(max(0, math.floor(min(values) / 5) * 5 - 2), min(100, max(values) + 3))
    ax.legend(loc="lower right", frameon=True, edgecolor="white")
    ax.grid(axis="y")
    axes[1].plot(epoch, gap_series(train, "loss", cfg["epochs"]), color="#465B70", lw=1.2)
    for ep in range(cfg["level_interval"] + 1,
                    min(cfg["epochs"], cfg["level_interval"] * (len(train[-1]["intervention"]["levels"]) - 1)) + 1,
                    cfg["level_interval"]):
        axes[1].axvline(ep, color="#C3CAD1", lw=.65, ls=":")
    axes[1].set(title="(b) Training objective", xlabel="Epoch", ylabel="Mean training loss")
    axes[1].grid(axis="y")
    fig.suptitle(subtitle, fontsize=10.5)
    save_figure(fig, out, FIGURES[0]); plt.close(fig)

    # 2. Net snapshot dose, not per-action sums or minibatch exposure counts.
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.7), sharex=True, constrained_layout=True)
    x = [r["epoch_number"] for r in refreshes]
    move, abstain = ([r[k] for r in refreshes] for k in ("move", "abstain"))
    axes[0].bar(x, move, width=3.4, color=COLORS["move"], label=f"Move (n={sum(move)})")
    axes[0].bar(x, abstain, bottom=move, width=3.4, color=COLORS["abstain"],
                label=f"Abstain (n={sum(abstain)})")
    caps = {r["cap"] for r in refreshes}
    if len(caps) == 1 and None not in caps:
        cap = next(iter(caps))
        axes[0].axhline(cap, ls="--", lw=.8, color="#7C8792", label=f"Acceptance cap ({cap})")
    axes[0].set(title="(a) Accepted actions per refresh", ylabel="Actions")
    axes[0].set_yticks([0, 2, 4, 6, 8])
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.03), frameon=False)
    axes[0].set_ylim(0, max(max(r["accepted"] for r in refreshes), cfg["crish"]["max_accepted"]) + 2.6)
    axes[1].bar(x, [r["level0_changed_directed_pairs"] for r in refreshes], width=3.4, color="#536E85")
    axes[1].set(title="(b) Net positive-relation changes at level 0", ylabel="Directed image pairs")
    y = [r["level0_target_mass_removed_percent"] for r in refreshes]
    axes[2].vlines(x, 0, y, color=COLORS["dose"], lw=1)
    axes[2].scatter(x, y, color=COLORS["dose"], s=16, zorder=3)
    axes[2].set(title="(c) Mean target-coefficient mass removed at level 0",
                xlabel="Refresh epoch", ylabel="Removed mass (%)")
    for ax in axes:
        ax.grid(axis="y")
        ax.set_xlim(5, cfg["epochs"] + 2)
    fig.suptitle(subtitle, fontsize=10.5)
    save_figure(fig, out, FIGURES[1]); plt.close(fig)

    # 3. All endpoints use the same SS evaluation; selectors differ explicitly.
    fig, ax = plt.subplots(figsize=(7.5, 3.1), constrained_layout=True)
    endpoints = e["endpoints"]
    ax.axhspan(-.38, .38, color="#EDF3F8", zorder=0)
    for i, (key, label, marker) in enumerate(zip(METRICS, ("All", "Old / Known", "New / Novel"), ("o", "s", "^"))):
        ys = np.arange(len(endpoints)) + (i - 1) * .19
        vals = [r[key + "_percent"] for r in endpoints]
        ax.scatter(vals, ys, s=34, color=COLORS[key], marker=marker, label=label, zorder=3)
        for value, y0 in zip(vals, ys):
            ax.annotate(f"{value:.2f}", (value, y0), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=COLORS[key])
    ax.set_yticks(range(len(endpoints)))
    ax.set_yticklabels([f"{r['label']} · epoch {r['epoch_number']}\n"
                        + ("Primary" if r["role"] == "primary" else "Secondary; evaluation-label selected")
                        for r in endpoints], fontsize=8)
    ax.set_ylim(len(endpoints) - .5, -.5)
    vals = [r[k + "_percent"] for r in endpoints for k in METRICS]
    ax.set_xlim(math.floor(min(vals)) - 2, min(103, max(vals) + 5))
    ax.set_xlabel("Balanced semi-supervised clustering accuracy (%)")
    ax.set_title("Checkpoint selection sensitivity — one training trajectory", pad=30)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.01), frameon=False, borderaxespad=0)
    ax.grid(axis="x")
    save_figure(fig, out, FIGURES[2]); plt.close(fig)

    # 4. Gains are conditioned on acceptance, never accuracy improvements.
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    actions = e["actions"]
    gains = [a[k] for a in actions for k in ("discovery_gain", "confirmation_gain")]
    positive = min(gains) > 0
    for kind, marker in (("move", "o"), ("abstain", "^")):
        selected = [a for a in actions if a["action"] == kind]
        axes[0].scatter([a["discovery_gain"] for a in selected],
                        [a["confirmation_gain"] for a in selected], s=22, alpha=.65,
                        marker=marker, color=COLORS[kind], linewidths=.3,
                        edgecolors="white", label=f"{kind.title()} (n={len(selected)})")
    lo, hi = min(gains), max(gains)
    axes[0].plot([lo, hi], [lo, hi], color="#8E99A4", lw=.8, ls="--", label="Equal gains")
    for axis in ("x", "y"):
        getattr(axes[0], f"set_{axis}scale")("log" if positive else "symlog",
                                          **({} if positive else {"linthresh": 1e-7}))
    axes[0].set(title="(a) Accepted-action probe gains", xlabel="Discovery loss decrease",
                ylabel="Confirmation loss decrease")
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    axes[0].grid()
    stop = Counter(r["stop_reason"] for r in refreshes)
    keys = sorted(stop, key=lambda k: -stop[k])
    axes[1].barh(range(len(keys)), [stop[k] for k in keys], color="#536E85", height=.6)
    axes[1].set_yticks(range(len(keys)))
    labels = {"time_budget": "Time budget", "confirmation_known_guard": "Known-label guard",
              "confirmation_failed": "Confirmation failed", "acceptance_cap": "Acceptance cap",
              "no_eligible_candidates": "No eligible candidates"}
    axes[1].set_yticklabels([labels.get(k, k) for k in keys])
    axes[1].invert_yaxis()
    for i, key in enumerate(keys):
        axes[1].text(stop[key] + .3, i, str(stop[key]), va="center", fontsize=8)
    axes[1].set(title="(b) Recorded refresh stop reasons", xlabel="Refreshes")
    axes[1].set_xlim(0, max(stop.values()) + 3)
    axes[1].grid(axis="x")
    fig.suptitle(subtitle, fontsize=10.5)
    save_figure(fig, out, FIGURES[3]); plt.close(fig)

    # 5. Enclosing timers include repair; no baseline-overhead claim.
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.05), constrained_layout=True)
    axes[0].plot(epoch, [r["seconds"] for r in train], color="#465B70", lw=1,
                 label="Recorded epoch total")
    axes[0].scatter(x, [r["seconds"] for r in refreshes], s=14, color=COLORS["abstain"],
                    label="Local refresh (included above)", zorder=3)
    axes[0].set(title="(a) Recorded training timers", xlabel="Epoch", ylabel="Time (s)")
    axes[0].set_ylim(0, max(r["seconds"] for r in train) * 1.25)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    axes[1].plot(epoch, [r["peak_cuda_allocated_bytes"] / 2**30 for r in train],
                 color=COLORS["dose"], lw=1.1)
    axes[1].set(title="(b) Trainer peak CUDA allocation", xlabel="Epoch", ylabel="Allocated memory (GiB)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(axis="y")
    fig.suptitle(subtitle + " · recorded RTX 3090", fontsize=10.5)
    save_figure(fig, out, FIGURES[4]); plt.close(fig)


def write_outputs(e: dict, run: Path, out: Path) -> dict:
    data = out / "data"
    data.mkdir()
    cfg, train, ss = e["config"], e["train"], e["ss"]
    write_csv(data / "ss_history.csv", [dict(epoch_number=r["epoch"] + 1,
        **{k + "_percent": 100 * r[k] for k in METRICS}, n_old=r["n_old"], n_new=r["n_new"],
        protocol=r["protocol"], purpose=r["purpose"]) for r in ss])
    write_csv(data / "training.csv", [dict(epoch_number=r["epoch"] + 1, loss=r["loss"],
        steps=r["steps"], recorded_epoch_seconds=r["seconds"],
        trainer_peak_cuda_GiB=r["peak_cuda_allocated_bytes"] / 2**30) for r in train])
    write_csv(data / "refreshes.csv", e["refreshes"])
    write_csv(data / "accepted_actions.csv", e["actions"])
    write_csv(data / "checkpoint_endpoints.csv", e["endpoints"])
    counts = Counter(a["action"] for a in e["actions"])
    peak = max(ss, key=lambda r: r["all_acc"])
    final = ss[-1]
    summary = dict(run_id=run.name, seed=cfg["seed"], dataset=cfg["dataset_name"],
        completed=e["metadata"]["status"] == "completed", variant="local_adaptive",
        training_epochs=len(train), ss_observations=len(ss),
        missing_ss_epochs=[i + 1 for i in range(cfg["epochs"]) if i not in {r["epoch"] for r in ss}],
        dose_observations=len(e["doses"]), refreshes=len(e["refreshes"]), accepted_actions=dict(counts),
        changed_labeled_id_epochs=sum(r["changed_labeled_ids"] for r in e["doses"].values()),
        recorded_epoch_time_hours=sum(r["seconds"] for r in train) / 3600,
        recorded_refresh_time_minutes=sum(r["seconds"] for r in e["refreshes"]) / 60,
        trainer_peak_cuda_GiB=max(r["peak_cuda_allocated_bytes"] for r in train) / 2**30,
        final={k + "_percent": 100 * final[k] for k in METRICS},
        posthoc_max_all_epoch=peak["epoch"] + 1,
        posthoc_max_all_percent=100 * peak["all_acc"],
        posthoc_peak_to_final_drop_pp=100 * (peak["all_acc"] - final["all_acc"]),
        stop_reasons=dict(Counter(r["stop_reason"] for r in e["refreshes"])))
    (data / "observed_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "captions.tex").write_text(r"""% Candidate captions; insert only after reviewing the observed evidence.
% All figures describe one seed of one resumed trajectory, without a baseline.
\newcommand{\CarsLearningCaption}{Observed learning dynamics on Stanford Cars for corrected SelEx with local adaptive CRiSH (seed 1). Left: balanced semi-supervised $k$-means accuracy on the unlabeled training pool, with global Hungarian matching. Old and New refer to known and novel classes. SS measurements are available only for epochs 7--200; missing epochs are not imputed. Right: epoch-mean training loss; dotted lines mark scheduled hierarchy-level additions, which change the objective. Curves are raw observations without smoothing or uncertainty bands.}
\newcommand{\CarsDoseCaption}{Realized local intervention dose over 38 refreshes. (a) Accepted Move and Abstain actions; the dashed line is the acceptance cap. (b) Net changes in directed positive image relations at hierarchy level 0, computed between pre- and post-repair snapshots rather than summed over actions. (c) Reduction in mean unweighted target-coefficient mass at level 0. Snapshot relations do not measure their actual minibatch exposure. No labeled image assignment or abstention mask changed in the available dose records.}
\newcommand{\CarsCheckpointCaption}{Checkpoint-selection sensitivity along one Stanford Cars training trajectory. Every point uses the same balanced SS protocol and retains the checkpoint's paired All/Old/New scores. The fixed final checkpoint is primary. Selection by disjoint-test Old accuracy and post-hoc selection by SS All use evaluation labels and are secondary analyses, not independent methods or matched baseline comparisons.}
\newcommand{\CarsProbeCaption}{Local probe and stopping diagnostics. (a) Discovery and confirmation probe-loss decreases for accepted actions only, with logarithmic axes and an equal-gain reference. Positive gains are conditioned on the acceptance rule and do not imply improved clustering accuracy. Actions within a refresh share a training state and are not independent trials. (b) Recorded terminal stop reasons across refreshes.}
\newcommand{\CarsComputeCaption}{Recorded resource use on an NVIDIA RTX 3090. (a) Epoch timers include feature extraction, repair and training; refresh timers are a subset and must not be added again. Separate evaluation workers, checkpoint I/O and launcher overhead are excluded. (b) Trainer-process peak CUDA allocated memory after each epoch's reset, not total device occupancy or evaluator peak. Imported histories include resumed epochs; summed timers are not uninterrupted wall-clock duration.}
""")
    (out / "README.md").write_text(f"""# Observed Stanford Cars figures

Run `{run.name}`; seed {cfg['seed']}; corrected SelEx + **local adaptive CRiSH**.
Metadata status: completed. These are measurements from one resumed trajectory,
not a matched-baseline comparison or a multi-seed estimate.

## Recommended placement

| Figure | Placement | Purpose |
|---|---|---|
| `01_learning_dynamics` | Main analysis candidate | Raw All/Old/New curves and training loss; reveal late decline. |
| `02_intervention_dose` | Main method analysis candidate | Accepted actions and actual net supervision changes. |
| `03_checkpoint_sensitivity` | Appendix / limitations | Distinguish fixed-final from evaluation-label-selected endpoints. |
| `04_probe_and_stopping` | Appendix | Probe selection and why refreshes stop. |
| `05_recorded_compute` | Appendix / efficiency diagnostics | Observed timers and trainer memory; no baseline-overhead claim. |

Each figure is provided as vector PDF/SVG and 320-dpi PNG. English captions
are in `captions.tex`; portable source tables and summary are in `data/`.

## Observations and limits

- Fixed final: All {100*final['all_acc']:.2f}%, Old {100*final['old_acc']:.2f}%,
  New {100*final['new_acc']:.2f}%. All is sample-weighted using
  {final['n_old']} Old and {final['n_new']} New images, not their unweighted mean.
- Post-hoc maximum observed SS All: {100*peak['all_acc']:.2f}% at epoch
  {peak['epoch']+1}; final is {summary['posthoc_peak_to_final_drop_pp']:.2f} pp lower.
  This does not establish overfitting or a causal effect of repairs.
- {len(ss)} SS observations; epochs {summary['missing_ss_epochs']} are absent.
  Available dose records cover epochs {min(e['doses'])+1}--{max(e['doses'])+1}.
- {len(e['refreshes'])} refreshes; {sum(counts.values())} accepted actions:
  {counts['move']} Move and {counts['abstain']} Abstain. Do not treat these
  as independent experiments or count repeated edits as unique dataset images.
- All refresh dose changes in this import occur at level 0; the script checks
  and reports other-level changes if present in future data.
- Recorded epoch timers sum to {summary['recorded_epoch_time_hours']:.2f} h;
  refresh timers sum to {summary['recorded_refresh_time_minutes']:.2f} min and
  are included in epoch timers. Peak trainer allocation is
  {summary['trainer_peak_cuda_GiB']:.2f} GiB. These exclude separate evaluation
  workers and are not total wall time or measured overhead against baseline.
- Run metadata records a resume from `ss-every1-migration/resume.pt`;
  inherited training history is not one uninterrupted launcher session.
- No model weights, per-image predictions or embeddings were supplied.
  Confusion matrices, t-SNE/UMAP, per-class accuracy, pair correctness and
  uncertainty bands cannot be reconstructed from these aggregate metrics.
- Missing a matched baseline, additional seeds and additional datasets.
  These figures support descriptive analysis, not an algorithmic-gain claim.

## Reproduce

From the repository root (a new output directory is required):

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_observed_run.py \\
  runs/{run.name} --output paper/figures/cars-local-seed1-rebuilt
```

Install `environments/requirements-figures.txt` if needed. Input artifacts
remain unchanged. `manifest.json` records input/output checksums, plotting
versions, source script hash, command and analysis timestamp. Remote training
source SHA is `{e['metadata'].get('git_sha')}`; the analysis source is distinct.
""")
    return summary


def overview(out: Path, summary: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont
    tile_width, tile_height, pad = 1000, 760, 30
    canvas = Image.new("RGB", (tile_width * 2 + pad * 3, tile_height * 3 + pad * 4), "#EEF2F5")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(FIGURES):
        tile = Image.open(out / f"{name}.png").convert("RGB")
        tile.thumbnail((tile_width - 20, tile_height - 20))
        x, y = pad + (i % 2) * (tile_width + pad), pad + (i // 2) * (tile_height + pad)
        draw.rounded_rectangle((x, y, x + tile_width, y + tile_height), radius=12, fill="white")
        canvas.paste(tile, (x + (tile_width - tile.width)//2, y + (tile_height - tile.height)//2))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except OSError:
        font = ImageFont.load_default()
    x, y = pad * 2 + tile_width, pad * 3 + tile_height * 2
    draw.multiline_text((x + 30, y + 120),
        "OBSERVED CARS RUN | SEED 1\n\n"
        f"Final All / Old / New: {summary['final']['all_acc_percent']:.2f} / "
        f"{summary['final']['old_acc_percent']:.2f} / {summary['final']['new_acc_percent']:.2f}%\n\n"
        f"{summary['refreshes']} refreshes: {summary['accepted_actions']['move']} Move, "
        f"{summary['accepted_actions']['abstain']} Abstain\n\n"
        "Primary: fixed final epoch 200\n"
        "Selected checkpoints: secondary analysis\n"
        "No matched baseline or multi-seed uncertainty\n\n"
        "PDF / SVG / PNG + source CSV + captions",
        font=font, fill="#263544", spacing=12)
    canvas.save(out / "overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Complete validation before making output files; never overwrite a figure set.
    if args.output.exists():
        parser.error("Output already exists; select a new directory to preserve existing figures")
    evidence = load_evidence(args.run_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    render(evidence, args.output)
    summary = write_outputs(evidence, args.run_dir, args.output)
    overview(args.output, summary)
    import matplotlib
    import numpy
    import PIL
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10)
    manifest = dict(run_id=args.run_dir.name, generated_utc=datetime.now(timezone.utc).isoformat(),
        analysis_command=sys.argv, analysis_git_sha=revision.stdout.strip(),
        training_git_sha=evidence["metadata"].get("git_sha"),
        script_sha256=sha256(Path(__file__)), matplotlib=matplotlib.__version__,
        numpy=numpy.__version__, pillow=PIL.__version__,
        source_files_sha256=evidence["sources"],
        output_files_sha256={str(p.relative_to(args.output)): sha256(p)
                             for p in sorted(args.output.rglob("*")) if p.is_file()},
        validation="observed SS protocol, sample weighting, epoch gaps, action identities and net dose checked")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="rendered_observed_only", output=str(args.output), summary=summary), indent=2))


if __name__ == "__main__":
    main()
