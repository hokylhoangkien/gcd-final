"""Read-only Cars diagnostics. The training loop never imports this module."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import DINOHead, DistillLoss, SupConLoss, info_nce_logits


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def metrics(x, a, b, labels, slots, weights, pairs):
    x = unit(x)
    distance = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(distance, 0)
    silhouette = silhouette_samples(distance, labels, metric="precomputed")
    np.fill_diagonal(distance, np.inf)
    nn = np.argsort(distance, axis=1)[:, :5]
    purity = (labels[nn] == labels[:, None]).mean(1)
    sums = np.stack([x[labels == k].sum(0) for k in range(196)])
    centroids = unit(sums)
    own = (x * unit(sums[labels] - x)).sum(1)
    sims = x @ centroids.T
    sims[np.arange(len(x)), labels] = -np.inf
    gap = own - sims.max(1)
    # Align occupied classifier partition centroids to semantic centroids.
    occupied = np.unique(slots)
    partitions = unit(np.stack([x[slots == k].mean(0) for k in occupied]))
    affinity = partitions @ centroids.T
    rows, cols = linear_sum_assignment(-affinity)
    cm = centroids @ centroids.T
    np.fill_diagonal(cm, -np.inf)
    stats = dict(occupied=len(occupied), occupied_centroid_alignment=float(affinity[rows, cols].mean()),
                 nearest_inter_centroid_distance=float((1 - cm.max(1)).mean()))
    if weights is not None:
        v = centroids @ unit(weights).T
        rows, cols = linear_sum_assignment(-v)
        stats.update(prototype_hungarian_cos=float(v[rows, cols].mean()),
                     old_slot_identity_cos=float(v.diagonal()[:98].mean()))
    arrays = dict(own_centroid_cos=own, purity5=purity, silhouette=silhouette,
                  semantic_margin=gap, crossview_cos=(unit(a) * unit(b)).sum(1))
    stats["subgroups"] = {}
    for name, mask in [("all", labels >= 0), ("old", labels < 98), ("novel", labels >= 98)]:
        stats["subgroups"][name] = {k: float(v[mask].mean()) for k, v in arrays.items()}
    if pairs:
        left, right = np.array(pairs).T
        stats["matched_applied_minus_control"] = {
            k: float((v[left] - v[right]).mean()) for k, v in arrays.items()}
    return stats


def batch_images(ids, transform, view, dataset, device):
    images = []
    for key in ids:
        with Image.open(dataset.data[key]) as im:
            # A crop is a pure function of ID and view across every checkpoint.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(17041 + 3 * int(key) + view)
                images.append(transform(im.convert("RGB")))
    return torch.stack(images).to(device)


def grads(head, backbone, h, tokens, labels, ids, legal_count, temp, micro):
    """VJPs at the same forward state. Hidden labels only annotate finished records."""
    h = h.detach().requires_grad_()
    f, z = head(h)
    batch = len(ids)
    v = head.last_layer.weight_v
    label = torch.as_tensor(labels[ids[:legal_count]], device=z.device, dtype=torch.long)
    p = (z / .1).softmax(-1)
    q = (z.detach() / temp).softmax(-1).reshape(2, batch, -1)
    lp = (z / .1).log_softmax(-1).reshape_as(q)
    projections = F.normalize(f.reshape(2, batch, -1)[:, :legal_count].transpose(0, 1), dim=-1)
    ce = F.cross_entropy(z.reshape_as(q)[:, :legal_count].flatten(0, 1) / .1, label.repeat(2))
    nce_z, nce_y = info_nce_logits(f, device=h.device)
    terms = dict(CE=.35 * ce, Distill=-.65 * (q.flip(0) * lp).sum(-1).mean(),
                 ME=.65 * (p.mean(0) * p.mean(0).log()).sum(),
                 InfoNCE=.65 * F.cross_entropy(nce_z, nce_y),
                 SupCon=.35 * SupConLoss()(projections, label))
    params = tuple(backbone.blocks[-1].parameters())

    def gradients(loss):
        dh, dv = torch.autograd.grad(loss, (h, v), retain_graph=True, allow_unused=True)
        if dv is None:
            dv = torch.zeros_like(v)
        block = [torch.zeros_like(param) for param in params]
        active = torch.nonzero(dh.abs().amax(1) > 0).flatten()
        for rows in active.split(micro):
            if rows.numel():
                hidden = backbone.norm(backbone.blocks[-1](tokens[rows.cpu()].to(h.device)))[:, 0]
                partial = torch.autograd.grad(hidden, params, dh[rows])
                for acc, g in zip(block, partial):
                    acc.add_(g)
        return dict(features=dh.flatten().detach(), classifier=dv.flatten().detach(),
                    block11=torch.cat([g.flatten() for g in block]))

    base = {k: gradients(term) for k, term in terms.items()}
    base["total"] = {k: sum(g[k] for g in base.values()) for k in base["CE"]}
    norms = {k: {s: g.norm().item() for s, g in vgs.items()} for k, vgs in base.items()}
    result = []
    # First eight unseen rows by the fixed random draw, independent of hidden labels.
    for row in range(legal_count, legal_count + 8):
        source, target = q[:, row].mean(0).topk(2).indices.tolist()
        feasible = bool((q[:, row, source] >= .2).all())
        for kind in ("abstain", "move"):
            # Exact scalar difference from KEEP. MOVE direction is also defined
            # algebraically outside support, and must be marked infeasible there.
            if kind == "move":
                delta = .65 * .2 * (lp[:, row, source] - lp[:, row, target]).sum() / (2 * batch)
            else:
                delta = .65 * (q.flip(0)[:, row] * lp[:, row]).sum() / (2 * batch)
            g = gradients(delta)
            comparisons = {}
            for space, x in g.items():
                n = x.norm().item()
                cos = {k: (float(torch.dot(x, gg[space])) / (n * norms[k][space])
                           if n and norms[k][space] else None) for k, gg in base.items()}
                comparisons[space] = dict(norm=n, ratio=n / norms["total"][space], cosine=cos)
            result.append(dict(id=int(ids[row]), group="old" if labels[ids[row]] < 98 else "novel",
                               kind=kind, mass_supported=feasible if kind == "move" else True,
                               source=source, target=target if kind == "move" else None,
                               gradients=comparisons))
    return dict(weighted_loss_norms=norms, rows=result, teacher_temperature=temp)


def read_ledger(path):
    applied, candidates = set(), Counter()
    if path:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["event"] == "exposure":
                applied.update(r["applied_ids"])
            elif r["event"] == "refresh":
                candidates.update({d["id"] for d in r["decisions"] if d["round"] == 0})
    return applied, candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cars", "dino-repo", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--pretrained-reference", type=Path)
    p.add_argument("--ledger", type=Path)
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--micro", type=int, default=4)
    args = p.parse_args()
    if args.per_class < 2 or args.micro < 1:
        p.error("Need at least two examples per class and a positive microbatch")
    import config
    config.car_root = str(args.cars.resolve())
    from data.augmentations import get_transform
    from data.get_datasets import get_class_splits
    from data.stanford_cars import CarsDataset, get_scars_datasets

    split = get_class_splits(SimpleNamespace(dataset_name="scars", use_ssb_splits=True))
    subsets = get_scars_datasets(None, None, split.train_classes, .5)
    dataset = CarsDataset(data_dir=config.car_root)
    order = list(split.train_classes) + list(split.unlabeled_classes)
    labels = np.array([order.index(c - 1) for c in dataset.target])
    known = subsets["train_labelled"].uq_idxs
    unknown = subsets["train_unlabelled"].uq_idxs
    rng = np.random.default_rng(104729)
    panel = np.concatenate([rng.choice(unknown[labels[unknown] == k], args.per_class, replace=False)
                            for k in range(len(order))])
    grad_ids = np.r_[rng.choice(known, 64, replace=False), rng.choice(unknown, 64, replace=False)]
    trans, center = get_transform("imagenet", 224, SimpleNamespace(interpolation=3, crop_pct=.875))
    torch.manual_seed(1)
    device = torch.device("cuda")
    torch.set_num_threads(2)
    backbone = torch.hub.load(str(args.dino_repo.resolve()), "dino_vitb16", source="local", pretrained=False)
    head = DINOHead(768, 196)
    model = torch.nn.Sequential(backbone, head).eval()
    if args.checkpoint:
        # Suite checkpoints include RNG/optimizer metadata. Use trusted files only.
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        epoch = int(saved["epoch"]) - 1
        del saved
    else:
        backbone.load_state_dict(torch.load(args.pretrained_reference, map_location="cpu", weights_only=True))
        epoch = 0
    model.to(device)
    src = args.checkpoint or args.pretrained_reference
    with src.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    output = dict(source=src.name, sha256=digest, state="checkpoint" if args.checkpoint else "DINO reference",
                  epoch=epoch, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  panel_ids=panel.tolist(), gradient_batch_ids=grad_ids.tolist(), micro=args.micro)
    applied, frequency = read_ledger(args.ledger)
    remaining = {j for j, key in enumerate(panel) if key not in applied}
    pairs = []
    for i, key in enumerate(panel):
        if key in applied:
            options = [j for j in remaining if labels[panel[j]] == labels[key]]
            if options:
                j = min(options, key=lambda t: (abs(frequency[panel[t]] - frequency[key]), int(panel[t])))
                remaining.remove(j)
                pairs.append((i, j))
    output["matched_ids"] = [[int(panel[i]), int(panel[j])] for i, j in pairs]
    all_views = []
    with torch.no_grad():
        for view, transform in enumerate((center, trans, trans)):
            data = []
            for start in range(0, len(panel), args.micro):
                images = batch_images(panel[start:start + args.micro], transform, view, dataset, device)
                h = backbone(images)
                f, z = head(h)
                data.append(tuple(y.cpu().numpy() for y in (h, f, (z / .1).softmax(-1))))
            all_views.append(tuple(np.concatenate(chunks) for chunks in zip(*data)))
            print(f"Encoded panel view {view}", flush=True)
    prob = all_views[0][2]
    slots = prob.argmax(1)
    y = labels[panel]
    output["geometry"] = {}
    for i, name in enumerate(("backbone", "projected", "decision")):
        output["geometry"][name] = metrics(*(v[i] for v in all_views), y, slots,
                                            head.last_layer.weight.detach().cpu().numpy() if i == 0 else None, pairs)
    cm = np.zeros((196, 196), dtype=int)
    np.add.at(cm, (y, slots), 1)
    n_pairs = lambda v: (v * (v - 1) // 2).sum()
    within = n_pairs(cm)
    rows, cols = linear_sum_assignment(-cm)
    mapped = np.empty(196, dtype=int)
    mapped[cols] = rows
    sorted_prob = np.sort(prob, axis=1)
    output["decisions"] = dict(split_rate=1 - within / n_pairs(cm.sum(1)),
                               merge_rate=1 - within / max(1, n_pairs(cm.sum(0))),
                               occupied=int(np.count_nonzero(cm.sum(0))),
                               top2_margin=float(np.mean(sorted_prob[:, -1] - sorted_prob[:, -2])),
                               crossview_agreement=float(np.mean(all_views[1][2].argmax(1) == all_views[2][2].argmax(1))),
                               old_to_new=float(np.mean(mapped[slots[y < 98]] >= 98)),
                               new_to_old=float(np.mean(mapped[slots[y >= 98]] < 98)))
    # Cache inputs to the trainable block; preserve the 128-row denominators.
    tokens, features = [], []
    with torch.no_grad():
        for view in (1, 2):
            for start in range(0, len(grad_ids), args.micro):
                imgs = batch_images(grad_ids[start:start + args.micro], trans, view, dataset, device)
                x = backbone.prepare_tokens(imgs)
                for block in list(backbone.blocks)[:-1]:
                    x = block(x)
                tokens.append(x.cpu())
                features.append(backbone.norm(backbone.blocks[-1](x))[:, 0])
    temp = DistillLoss(30, 200).teacher_temp_schedule[epoch]
    output["gradients"] = grads(head, backbone, torch.cat(features), torch.cat(tokens), labels,
                                grad_ids, 64, temp, args.micro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
