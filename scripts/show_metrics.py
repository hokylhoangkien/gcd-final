"""Print saved epoch losses and All/Old/New percentages; never run evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcd_runtime.metrics import collect, print_peaks, read_rows, summarize_run


def show(run_dir):
    run_dir = Path(run_dir)
    losses = {row['epoch']: row for row in read_rows(run_dir/'metrics.jsonl')}
    histories, finals = collect(run_dir)
    if not histories and losses:
        histories['training_loss_only'] = {}
    if not histories and not finals:
        print('No saved epoch metrics. No historical accuracy can be reconstructed from loss alone.')
        return
    for protocol, metrics in histories.items():
        print(f'\nProtocol: {protocol}')
        print('Epoch     Loss    All(%)    Old(%)    New(%)  Status / Purpose')
        for epoch in sorted(set(losses) | set(metrics)):
            loss = losses.get(epoch, {}).get('loss')
            row = metrics.get(epoch, {})
            cells = ['--' if row.get(key) is None else f'{100*row[key]:.2f}'
                     for key in ('all_acc', 'old_acc', 'new_acc')]
            loss_text = '--' if loss is None else f'{loss:.4f}'
            status = row.get('status', 'not_evaluated')
            print(f'{epoch+1:5d} {loss_text:>8} {cells[0]:>9} {cells[1]:>9} {cells[2]:>9}  '
                  f'{status} / {row.get("purpose", "--")}')
    print_peaks(summarize_run(run_dir))
    for name, row in finals.items():
        print(f'{name}: epoch={row.get("epoch_number", "unspecified")} '
              f'All={100*row["all_acc"]:.2f}% Old={100*row["old_acc"]:.2f}% '
              f'New={100*row["new_acc"]:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error('run directory does not exist')
    show(args.run_dir)
