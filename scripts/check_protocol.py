"""Verify shipped worker arguments against pinned dataset-specific commands."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.protocol import compare_worker
from gcd_runtime.config import ROOT, load_config
from gcd_runtime.readiness import verify_upstream


if __name__ == '__main__':
    verify_upstream()
    reports = {}
    for prefix in ('', 'cub_'):
        for method in ('baseline', 'crish'):
            for upgrade in ('', '_upgraded'):
                for smoke in ('', '_smoke'):
                    name = prefix+method+upgrade+smoke
                    report = compare_worker(load_config(ROOT/f'configs/{name}.json'), smoke=bool(smoke))
                    reports[name] = {key: value for key, value in report.items() if key != 'effective_reference_settings'}
    print(json.dumps(dict(kind='source_cli_comparison_not_benchmark', profiles=reports), indent=2))
    if any(not report['cli_hyperparameters_match'] for report in reports.values()):
        raise SystemExit('Shipped dataset/DINOv2 CLI differs from its pinned upstream command')
