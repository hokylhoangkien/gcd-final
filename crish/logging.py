"""Small append-only audit trace with measured timing; no oracle label fields."""
from contextlib import contextmanager
import json
from pathlib import Path
import time


class AuditTrace:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.costs = {}

    @contextmanager
    def measure(self, component: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.costs[component] = self.costs.get(component, 0.0) + time.perf_counter() - start

    def append(self, decision: dict):
        # Reject unsupported JSON/nonfinite numbers before writing any bytes.
        payload = json.dumps(dict(decision=decision, measured_seconds=self.costs), allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as handle:
            handle.write(payload + '\n')
