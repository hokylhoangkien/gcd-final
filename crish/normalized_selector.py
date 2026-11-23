"""Training-only pooled component scales and one-SE dose selection."""
import math
from statistics import median

COMPONENTS = ('known', 'view', 'variance')


def component_statistics(candidates, history=None, *, floor_ratio=.25):
    """Pool within-candidate residuals, never candidate mean differences.

    candidates: {dose: [{known, view, variance}, ...]}; zero-dose excluded.
    History contains raw (not floored) scales from earlier refreshes only.
    """
    if not candidates:
        return {}, {}
    counts = {len(rows) for rows in candidates.values()}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError('Normalized selection needs >=2 equally sized chunks per dose')
    m = counts.pop()
    means = {q: {c: math.fsum(float(row[c]) for row in rows)/m for c in COMPONENTS}
             for q, rows in candidates.items()}
    if any(not math.isfinite(v) for a in means.values() for v in a.values()):
        raise ValueError('Nonfinite component gain')
    raw = {c: math.sqrt(math.fsum((float(row[c])-means[q][c])**2 for q, rows in candidates.items()
                                  for row in rows)/(len(candidates)*(m-1))) for c in COMPONENTS}
    scales = {}
    for c in COMPONENTS:
        previous = (history or {}).get(c, [])
        effective = max(raw[c], floor_ratio*median(previous)) if previous else raw[c]
        scales[c] = dict(raw=raw[c], effective=effective, se=effective/math.sqrt(m),
                         degenerate_scale=effective == 0)
    out = {}
    for q, mean in means.items():
        t = {c: mean[c]/scales[c]['se'] if scales[c]['se'] else 0. for c in COMPONENTS}
        absolute = math.fsum(abs(v) for v in mean.values())
        normalized = math.fsum(abs(v) for v in t.values())
        out[q] = dict(chunks=candidates[q], mean=mean, t=t,
                      share_raw={c: abs(mean[c])/absolute if absolute else 0. for c in COMPONENTS},
                      share_t={c: abs(t[c])/normalized if normalized else 0. for c in COMPONENTS})
    return out, scales


def failures(statistics, *, view_min=1., guard_se=2., known_guard=True, confirmation=False):
    t = statistics['t']
    reasons = []
    if (t['view'] <= view_min if confirmation else t['view'] < view_min):
        reasons.append('view_gain')
    if known_guard and t['known'] < -guard_se:
        reasons.append('known_guard')
    if t['variance'] < -guard_se:
        reasons.append('variance_guard')
    return reasons


def select_dose(statistics, *, t_min=1., guard_se=2., known_guard=True):
    eligible = []
    for q, row in statistics.items():
        row['failures'] = failures(row, view_min=t_min, guard_se=guard_se, known_guard=known_guard)
        if not row['failures']:
            eligible.append(q)
    if not eligible:
        return 0., 0.
    highest = max(eligible, key=lambda q: (statistics[q]['t']['view'], -q))
    selected = min(q for q in eligible if statistics[q]['t']['view'] >= statistics[highest]['t']['view']-1.)
    return selected, highest
