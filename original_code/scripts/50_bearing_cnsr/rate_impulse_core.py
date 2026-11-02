"""Fixed impulse/rate action and source distances; no payload I/O or neural code."""
from collections import defaultdict
from statistics import mean
import numpy as np

ARMS = ('F', 'C', 'N', 'W', 'C0', 'G0', 'G1', 'N0')
NEW = ('C0', 'G0', 'G1', 'N0')
PURE = 'RECORDED_SPEED_CHANGE_FIXED_RECORDED_LOAD'


def require(value, message):
    if not value:
        raise ValueError(message)


def actions(factual, timing, original, q):
    f, c, u = [np.asarray(v, dtype=np.float64) for v in (factual, timing, original)]
    require(f.shape == c.shape == u.shape and f.ndim == 2, 'same query axes')
    require(q > 0 and np.isfinite(q) and all(np.isfinite(v).all() for v in (f, c, u)), 'finite positive action')
    uq = c - f + u
    if q == 1:
        return {k: f.copy() for k in NEW}
    return dict(C0=f + q * uq - u, G0=q * f,
                G1=f + (q - 1) * u, N0=f + uq / q - u)


def distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] and len(a) and len(b), 'nonempty distance axes')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'finite distance')
    ab = np.linalg.norm(a[:, None] - b[None], axis=-1).mean()
    aa = np.linalg.norm(a[:, None] - a[None], axis=-1).mean()
    bb = np.linalg.norm(b[:, None] - b[None], axis=-1).mean()
    value = float(2 * ab - aa - bb)
    floor = float(128 * np.finfo(float).eps * max(1., 2 * ab + aa + bb))
    require(value >= -floor, 'negative energy beyond roundoff')
    return max(value, 0.), floor


def summarize(features, reference):
    pairs = {k: distance(features[k], reference) for k in ARMS}
    ds = {k: v[0] for k, v in pairs.items()}
    floor = max(v[1] for v in pairs.values())
    require(ds['F'] > floor, 'same informative factual units; never silently omit')
    return dict(distances=ds, floor=floor,
                skill={k: (ds['F'] - ds[k]) / ds['F'] for k in ARMS})


def panel(rows):
    identities = defaultdict(list)
    for row in rows:
        u = row['unit']
        identities[tuple(sorted((u['source'], u['destination']))), u['label'], u['recorded_identity']].append(row)
    im = {key: {k: mean(r['summary']['skill'][k] for r in values) for k in ARMS}
          for key, values in identities.items()}
    classes = defaultdict(list)
    for (pair, label, identity), value in im.items():
        classes[pair, label].append(value)
    pairs = defaultdict(list)
    for (pair, label), values in classes.items():
        pairs[pair].append({k: mean(v[k] for v in values) for k in ARMS})
    pm = {''.join(pair): {k: mean(v[k] for v in values) for k in ARMS}
          for pair, values in pairs.items()}
    return dict(units=len(rows), identities=len(im), pairs=len(pm),
                means={k: mean(v[k] for v in pm.values()) for k in ARMS}, pair_means=pm,
                identity_increment_positive=sum(v['C0'] > v['C'] for v in im.values()),
                identity_details=[dict(pair=''.join(key[0]), label=key[1], identity=key[2], means=v)
                                  for key, v in sorted(im.items())])


def aggregate(rows):
    strata = defaultdict(list)
    for r in rows:
        strata[r['unit']['protocol'], r['unit']['transition_kind']].append(r)
    result = {}
    for (protocol, kind), subset in sorted(strata.items()):
        value = panel(subset)
        value['directions'] = {name: panel([r for r in subset if (r['q'] > 1) == up])
                               for name, up in (('up', True), ('down', False))}
        result[protocol + '/' + kind] = value
    pure = [result[ds + '/' + PURE] for ds in ('pu_clock4', 'phm09_w3')]
    checks = {}
    increments = []
    for ds, v in zip(('PU', 'PHM'), pure):
        delta = v['means']['C0'] - v['means']['C']; increments.append(delta)
        checks[ds + '_nonnegative_increment'] = dict(value=delta, pass_=delta >= 0)
        checks[ds + '_majority_identities'] = dict(value=v['identity_increment_positive'],
            denominator=v['identities'], pass_=v['identity_increment_positive'] > v['identities'] / 2)
        for direction in ('up', 'down'):
            x = v['directions'][direction]['means']['C0']
            checks[ds + '_' + direction + '_positive_skill'] = dict(value=x, pass_=x > 0)
    delta = mean(increments)
    checks['two_protocol_increment_0p02'] = dict(value=delta, pass_=delta >= .02)
    for control in ('G0', 'G1', 'N0'):
        delta = mean(v['means']['C0'] - v['means'][control] for v in pure)
        checks['over_' + control + '_0p01'] = dict(value=delta, pass_=delta >= .01)
    j = next(v for k, v in result.items() if k.startswith('jnu_speed/'))
    delta = j['means']['C0'] - j['means']['C']
    checks['JNU_floor_minus_0p02'] = dict(value=delta, pass_=delta >= -.02)
    passed = all(v['pass_'] for v in checks.values())
    return dict(status='SOURCE_QUALIFIED_CLASSIFICATION_UNTESTED' if passed else 'RATE_IMPULSE_SOURCE_QUALIFICATION_FAILED',
                source_qualification_passed=passed, checks=checks, strata=result)
