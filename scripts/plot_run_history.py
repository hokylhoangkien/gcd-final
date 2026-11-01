"""Plot real SS-kmeans history and final evaluation; CPU only, no checkpoint load."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from analyze_reliability_history import SS, load_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs='+', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metrics, _, config = load_runs(args.run)
    rows = [r for r in metrics if r['protocol'] == SS]
    final = json.loads(args.evaluation.read_text())
    if final['protocol'] != SS or final['unit'] != 'fraction':
        raise ValueError('Final evaluation must use the same SS protocol/unit')
    if not rows:
        raise ValueError('Missing SS history')
    args.output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    x = [r['completed_updates_at_measurement'] for r in rows]
    colors = {'All': '#2166ac', 'Old': '#6a3d9a', 'New': '#d95f02'}
    for key, ax in [('All', axes[0]), ('Old', axes[1]), ('New', axes[1])]:
        values = [100 * r[key] for r in rows]
        smooth = [mean(values[max(0, i-9):i+1]) for i in range(len(values))]
        ax.plot(x, values, color=colors[key], alpha=.22, linewidth=.8)
        ax.plot(x, smooth, color=colors[key], linewidth=1.7, label=f'{key}: trailing 10-evaluation mean')
        ax.scatter([config['epochs']], [100 * final[key.lower() + '_acc']],
                   color=colors[key], marker='*', s=110, zorder=5,
                   label=f'{key}: final checkpoint evaluation')
    warmup = config['reliability']['warmup_epochs']
    full = warmup + config['reliability']['ramp_epochs'] - 1
    for ax in axes:
        ax.axvline(warmup, color='gray', linestyle=':', linewidth=.8)
        ax.axvline(full, color='gray', linestyle='--', linewidth=.8)
        ax.set_xlabel('Completed training epochs at evaluation')
        ax.set_ylabel('Clustering accuracy (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='lower right')
    axes[0].set_title('All accuracy')
    axes[1].set_title('Old / New trade-off')
    fig.suptitle('Proposed DINOv2, Stanford Cars SSB, seed 1')
    fig.text(.5, .015, 'Train-unlabeled SS-kmeans with labeled anchors. Thin: individual evaluations. '
             'Vertical markers: reliability starts / reaches full strength. No matched baseline.',
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(args.output / 'history.png', dpi=200)
    fig.savefig(args.output / 'history.pdf')
    plt.close(fig)
    with (args.output / 'history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['completed_epochs', 'All_percent', 'Old_percent', 'New_percent', 'kind'])
        for r in rows:
            writer.writerow([r['completed_updates_at_measurement'], *(100*r[k] for k in ('All', 'Old', 'New')), 'monitor'])
        writer.writerow([config['epochs'], *(100*final[k] for k in ('all_acc', 'old_acc', 'new_acc')), 'final_evaluation'])
    print(args.output / 'history.png')


if __name__ == '__main__':
    main()
