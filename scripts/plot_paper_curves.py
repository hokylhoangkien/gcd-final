"""Static observed SS-v2 curves with refresh markers and fixed-final endpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_registry import Source, inspect_run
from scripts.plot_observed_run import ss_history


def render(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    row, refreshes, _ = inspect_run(run)
    if row['status'] != 'observed_final_single_seed':
        raise ValueError('Require verified final endpoint')
    source = Source(run)
    try:
        history = list(source.rows('epoch_metrics.jsonl'))
        history = [r for r in history if r.get('status') == 'observed'
                   and r.get('purpose') == 'monitor_only'
                   and r.get('protocol') == 'balanced_ss_kmeans_train_unlabeled']
        history.append(source.obj('evaluation_last.json'))
        history = ss_history(history)
        cfg = source.obj('config.json')
    finally:
        source.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index = {r['epoch']:r for r in history}
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout='constrained')
    for key, label, color in [('all_acc','All','#1764A0'), ('old_acc','Old','#C47A12'), ('new_acc','New','#16816A')]:
        values = [100*index[e][key] if e in index else math.nan for e in range(cfg['epochs'])]
        ax.plot(range(1,cfg['epochs']+1), values, label=label, color=color, lw=1.4)
        ax.scatter([cfg['epochs']], [values[-1]], color=color, s=24)
    for event in refreshes:
        ax.axvline(event['epoch_number'], color='#778899', alpha=.17, lw=.6)
    ax.set(xlabel='Epoch (one-based)', ylabel='Accuracy (%)', title=f"{row['dataset']} — seed {row['seed']}, fixed-final SS-v2")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=.15)
    for suffix in ('pdf','svg','png'):
        fig.savefig(output/('learning_curves.'+suffix), dpi=180)
    plt.close(fig)
    caption = ('Observed single-seed All/Old/New on the unlabeled training pool using balanced '
        'SS-k-means and global Hungarian-v2 matching. Gaps remain unmeasured. Gray vertical lines '
        'mark recorded CRiSH refreshes (log epoch + 1); endpoint dots mark the fixed-final checkpoint. '
        'Evaluation labels are used only for reporting. Curves do not measure causal action effects. '
        'No seed-averaged improvement is established.\n')
    (output/'caption.txt').write_text(caption)
    (output/'data.json').write_text(json.dumps(dict(run=row, history=history, refreshes=refreshes), indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(source=str(run), config_sha256=row['config_sha256'],
        checkpoint_sha256=row['checkpoint_sha256'], outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file()}), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.output)
