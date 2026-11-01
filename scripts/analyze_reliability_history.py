"""Read-only, CPU-only audit of existing runtime logs; no training imports.

Metrics epoch e usually measures SS-kmeans BEFORE epoch e updates. Diagnostics
epoch e measures the weighting DURING those updates. Do not join them as an
after-update effect. No scalar history establishes a causal method benefit.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

SS = 'balanced_ss_kmeans_train_unlabeled'


def read_jsonl(path):
    data = path.read_bytes()
    lines = data.splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not data.endswith(b'\n'):
                continue  # Snapshot while writer is appending its last record.
            raise ValueError(f'Malformed record: {path}:{index + 1}')
        result.append(row)
    return result


def load_runs(run_dirs):
    metrics, diagnostics = [], []
    seen, reference = set(), None
    for directory in run_dirs:
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        if reference is not None and config != reference:
            raise ValueError('Run configs differ; do not silently merge experiments')
        reference = config
        for row in read_jsonl(directory / 'metrics.jsonl'):
            epoch = row['epoch']
            if epoch in seen:
                raise ValueError(f'Duplicate epoch {epoch}; choose the intended resume chain')
            seen.add(epoch)
            for key in ('All', 'Old', 'New'):
                if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                    raise ValueError(f'Invalid {key} at epoch {epoch}')
            row = dict(row, run=directory.name)
            row['completed_updates_at_measurement'] = epoch if row['protocol'] == SS else epoch + 1
            metrics.append(row)
        path = directory / 'reliability.jsonl'
        if path.exists():
            diagnostics.extend(dict(r, run=directory.name) for r in read_jsonl(path))
    return sorted(metrics, key=lambda r: r['epoch']), diagnostics, reference


def aggregate(records):
    """Pair-weighted summary, NOT an average giving tiny/large batches equal mass."""
    pairs = sum(r['eligible_pairs'] for r in records)
    if pairs:
        weight = sum(r['effective_weight_mass'] for r in records) / pairs
        down = sum(r['eligible_pairs'] * r['downweighted_fraction']
                   for r in records if r['eligible_pairs']) / pairs
    else:
        weight = down = None
    return dict(snapshots=len(records), eligible_ordered_pairs=pairs,
                pair_weight_mean=weight, downweighted_fraction=down,
                # Equal-snapshot anchor mean: records do not store anchor counts.
                fallback_snapshot_mean=mean(r['fallback_anchor_fraction'] for r in records))


def analyze(metrics, diagnostics, block_size=10):
    rows = [r for r in metrics if r['protocol'] == SS]
    if not rows:
        raise ValueError('No SS-kmeans history; do not mix plain-kmeans metrics')
    lines = [
        '# Audit từ log hiện có', '',
        'Chỉ phân tích mô tả; không có baseline đối chứng và không suy ra quan hệ nhân quả.',
        'SS metrics epoch e đo trước train e (đã hoàn tất e epoch); diagnostics e đo trong train e.',
        'Std dưới đây là dao động giữa epoch, không phải multi-seed uncertainty.', '',
        '| SS epoch index | Updates đã xong lúc đo | N | All mean±std % | Old % | New % |',
        '|---|---|---:|---:|---:|---:|'
    ]
    tail = rows[-4 * block_size:]
    # Blocks are anchored at the newest record, matching earlier terminal summaries.
    first = len(tail) % block_size or block_size
    blocks, offset = [], 0
    while offset < len(tail):
        size = first if offset == 0 else block_size
        blocks.append(tail[offset:offset + size])
        offset += size
    for block in blocks:
        all_values = [100 * r['All'] for r in block]
        lo, hi = block[0]['epoch'], block[-1]['epoch']
        lines.append(f'| {lo}–{hi} | {lo}–{hi} | {len(block)} | '
                     f'{mean(all_values):.2f}±{pstdev(all_values):.2f} | '
                     f'{100*mean(r["Old"] for r in block):.2f} | '
                     f'{100*mean(r["New"] for r in block):.2f} |')
    recent_epochs = {r['epoch'] for r in rows[-block_size:]}
    groups = defaultdict(list)
    for row in diagnostics:
        if row['epoch'] in recent_epochs:
            groups[row['level']].append(row)
    summaries = []
    lines += ['', 'Diagnostics trong các epoch train tương ứng nhóm gần nhất; không phải hiệu quả accuracy.',
              'Chỉ các batch được log, không đại diện chắc chắn toàn bộ dữ liệu.', '',
              '| Level | Snapshots | Eligible ordered pairs | Pair-weighted mean | Downweighted % | Fallback snapshot mean % | Hierarchy coefficient |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for level, records in sorted(groups.items()):
        summary = dict(level=level, **aggregate(records))
        summaries.append(summary)
        w, d = summary['pair_weight_mean'], summary['downweighted_fraction']
        ws, ds = ('N/A', 'N/A') if w is None else (f'{w:.3f}', f'{100*d:.2f}')
        lines.append(f'| {level} | {len(records)} | {summary["eligible_ordered_pairs"]} | '
                     f'{ws} | {ds} | {100*summary["fallback_snapshot_mean"]:.2f} | {2**(-level-1):.6f} |')
    lines += ['',
              'Hierarchy coefficient chỉ là hệ số loss, không đo gradient thực tế.',
              'Trọng số chuẩn hóa theo anchor: giảm raw weight không đồng nghĩa giảm gradient cùng tỷ lệ.',
              'Fallback cao có nghĩa cơ chế bảo vệ thường giữ weight1; chưa chứng minh chọn sai cặp.',
              'Logs hiện không có pair identity/ground-truth correctness hay baseline-gradient counterfactual.',
              'Muốn biết score tốt: audit offline precision của pseudo-positive được giữ/giảm trọng số trên checkpoint cố định.',
              'Muốn biết accuracy tăng do module: baseline/ablation matched; trước/sau warm-up không phải đối chứng.',
              'Không chỉnh config, chọn best checkpoint hoặc dùng evaluation labels để tune từ báo cáo này.',
              'Plain-kmeans cuối run được loại khỏi bảng SS; phải evaluate checkpoint cuối bằng protocol thống nhất.',
              'Resume có thay dependency kmeans: giữ provenance bản sửa và dùng giống nhau cho đối chứng.']
    return '\n'.join(lines) + '\n', summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--output', type=Path, help='New directory; refuses to overwrite existing output')
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error('block-size must be positive')
    metrics, diagnostics, config = load_runs(args.run)
    report, summaries = analyze(metrics, diagnostics, args.block_size)
    print(report)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'report.md').write_text(report)
        (args.output / 'diagnostics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
        fields = ('run', 'epoch', 'protocol', 'completed_updates_at_measurement', 'All', 'Old', 'New')
        with (args.output / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(metrics)


if __name__ == '__main__':
    main()
