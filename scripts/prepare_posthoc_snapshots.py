"""Join saved pre-refresh partitions to offline evaluation labels; analysis only."""
import argparse
import csv
import json
from pathlib import Path
import re


def export(run, labels, output):
    import numpy as np
    run = Path(run)
    truth = {}
    with Path(labels).open() as handle:
        for row in csv.DictReader(handle):
            sample, label = int(row['sample_id']), int(row['true_label'])
            if sample in truth or label < 0:
                raise ValueError('Duplicate ID or missing offline label')
            truth[sample] = label
    paths = sorted(run.glob('partitions_epoch_*.npz'))
    if not paths:
        raise ValueError('No pre-refresh partition tensors available; text imports are insufficient')
    metadata = json.loads((run/'metadata.json').read_text())
    with Path(output).open('x') as handle:
        for path in paths:
            epoch = int(re.fullmatch(r'partitions_epoch_(\d+)\.npz', path.name).group(1))
            with np.load(path, allow_pickle=False) as snapshot:
                ids = snapshot['ids'].tolist()
                if any(sample not in truth for sample in ids):
                    raise ValueError('Offline labels do not cover every snapshot ID')
                for key in sorted(k for k in snapshot.files if k.startswith('level_')):
                    handle.write(json.dumps(dict(run_id=metadata['run_id'], epoch=epoch,
                        level=int(key.split('_')[1]), sample_ids=ids,
                        true_labels=[truth[i] for i in ids], clusters=snapshot[key].tolist(),
                        labeled=snapshot['labeled'].tolist()), allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--labels', type=Path, required=True, help='Offline CSV: sample_id,true_label')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.labels, args.output)
