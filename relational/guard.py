"""External exploratory budget evaluator. Consumes exported metrics, never model inputs."""

import argparse
import hashlib
import json
from decimal import Decimal as D
from pathlib import Path


def grade(epoch, all_pp, new_pp):
    if epoch not in (100, 150):
        raise ValueError("Expected one-based milestone 100 or 150")
    if epoch == 100 and (all_pp <= D("-.5") or new_pp <= -1):
        return "NO-GO"
    if epoch == 150 and (all_pp < D(".3") or new_pp < 0):
        return "NO-GO"
    if all_pp >= 1 and new_pp >= D(".5"):
        return "GO" if epoch == 100 else "STRONG GO"
    return "GRAY ZONE" if epoch == 100 else "MARGINAL"


def window(folder, variant, epoch, protocol):
    meta = json.loads((folder / "config.json").read_text())
    if meta != {"variant": variant, "protocol": protocol}:
        raise ValueError(f"{variant} configuration or seed mismatch")
    if not (folder / f"epoch-{epoch:03}.pt").is_file():
        raise ValueError(f"{variant} epoch {epoch} is not committed")
    if variant == "R0" and not (folder / "epoch-200.pt").is_file():
        raise ValueError("Matched R0 must complete 200 epochs")
    values = []
    digest = hashlib.sha256()
    with (folder / "metrics.jsonl").open("rb") as f:
        for expected in range(1, epoch + 1):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"{variant} is missing committed metrics for epoch {expected}"
                )
            record = json.loads(line, parse_float=D)
            if type(record["epoch"]) != int or record["epoch"] != expected:
                raise ValueError(f"{variant} epoch sequence differs at {expected}")
            if expected > epoch - 20:
                acc = [D(record[key]) for key in ("all", "new")]
                if not all(v.is_finite() and 0 <= v <= 1 for v in acc):
                    raise ValueError(f"{variant} invalid accuracy in window")
                values.append(acc)
                digest.update(line)
    return [sum(column) * 100 / 20 for column in zip(*values)], digest.hexdigest()


def evaluate(root, variant, epoch, *, allow_missing_references=False):
    if variant not in ("R1", "R2") or epoch not in (100, 150):
        raise ValueError("Guards apply to R1/R2 at 100/150")
    result = {
        "variant": variant,
        "epoch": epoch,
        "first_epoch": epoch - 19,
        "comparisons": {},
        "exploratory": True,
    }
    try:
        protocol = json.loads((root / variant / "config.json").read_text())["protocol"]
        own, sha = window(root / variant, variant, epoch, protocol)
        result.update(
            mean_all_pp=float(own[0]), mean_new_pp=float(own[1]), window_sha256=sha
        )
        for ref in ["R0"] if variant == "R1" else ["R0", "R1"]:
            try:
                base, sha = window(root / ref, ref, epoch, protocol)
            except (OSError, ValueError, KeyError, ArithmeticError, TypeError) as exc:
                if not allow_missing_references:
                    raise
                result.setdefault("unavailable_references", {})[ref] = str(exc)
                continue
            da, dn = [x - y for x, y in zip(own, base)]
            result["comparisons"][ref] = {
                "delta_all_pp": float(da),
                "delta_new_pp": float(dn),
                "decision": grade(epoch, da, dn),
                "reference_window_sha256": sha,
            }
    except (OSError, ValueError, KeyError, ArithmeticError, TypeError) as exc:
        return dict(result, decision="BLOCKED", reason=str(exc))
    if result.get("unavailable_references"):
        return dict(
            result,
            decision="UNVERIFIED",
            reason="One or more matched reference windows are unavailable",
        )
    grades = {r["decision"] for r in result["comparisons"].values()}
    result["decision"] = next(
        g for g in ("NO-GO", "MARGINAL", "GRAY ZONE", "STRONG GO", "GO") if g in grades
    )
    result["reason"] = "; ".join(
        f"{ref} {r['decision']} (All {r['delta_all_pp']:+.6f} pp, New {r['delta_new_pp']:+.6f} pp)"
        for ref, r in result["comparisons"].items()
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("variant", choices=("R1", "R2"))
    p.add_argument("epoch", type=int, choices=(100, 150))
    args = p.parse_args()
    print(json.dumps(evaluate(args.root, args.variant, args.epoch), indent=2))
