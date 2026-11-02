"""Source-rate numerical operations and matched sampler geometry, no I/O/torch."""
from fractions import Fraction
import numpy as np
from natural_mapping_probe_core import Physics
from source_rate_plan import ARMS, require


def shared_edit_mask(uniforms, domains, classes, per_cell):
    values = np.asarray(uniforms)
    require(values.shape == (domains * classes * per_cell,), 'same old full-length uniform draw')
    first = (values.reshape(domains, classes, per_cell)[0] < .5)
    return np.broadcast_to(first[None], (domains, classes, per_cell)).reshape(-1).copy()


def batch_rates(sources, rates, classes, per_cell, mask):
    correct = np.concatenate([np.full(classes * per_cell, float(Fraction(rates[d]))) for d in sources])
    ring = np.concatenate([np.full(classes * per_cell, float(Fraction(rates[sources[(i + 1) % len(sources)]])))
                           for i in range(len(sources))])
    mask = np.asarray(mask, bool)
    require(mask.shape == correct.shape and np.array_equal(np.sort(correct[mask]), np.sort(ring[mask])),
            'actual active-edit rate multiset is identical, not just nominal q list')
    return correct, ring


def make_group(group, inverse, qualified, charge):
    physics = Physics(charge)
    source = group['source']; n = 32768
    s, params = inverse['s'], inverse['params']
    require(s.shape == (2, 48, n) and params.shape == (2, 48, 3) and
        np.isfinite(s).all() and np.isfinite(params).all() and np.all(s >= 0), 'old frozen inverse arrays')
    observed, component = qualified['observed'][source], qualified['u_normalized'][source]
    require(observed.shape == component.shape == (n,) and np.isfinite(observed).all() and
            np.isfinite(component).all(), 'qualified complete observed/old u1 arrays')
    scale = float(qualified['scale'][source])
    require(np.isfinite(scale) and scale > 0, 'positive old physical-unit source scale')
    starts, window = group['starts'], group['window']
    crops = lambda x: np.stack([x[start:start + window] for start in starts])
    factual = crops(observed); physics.charge('encoded_windows', 3)
    clean = physics.spectrum(factual)
    changed = []
    for text in group['ratios']:
        ratio = float(Fraction(text))
        for start in starts:
            physics.functions['support'](n, group['kernel_length'], ratio, start, start + window)
        charge('context_renders', 1)
        rendered = physics.functions['render_phase'](s[source], params[source], ratio, group['kernel_length'])
        waves = crops(observed + scale * (rendered - component))
        require(np.isfinite(waves).all(), 'finite three complete physical-unit edited crops')
        physics.charge('encoded_windows', 3); changed.append(physics.spectrum(waves))
    return dict(clean=clean, event=np.stack(changed), ratios=np.asarray(group['ratios']),
        group_key=np.array(group['key']), inverse_identity=np.array(group['inverse_identity']),
        origin=np.array(group['origin']), starts=np.asarray(starts),
        old_u1_reused_from_complete_source_qualification=np.array(True))


def assemble(case, packets):
    result = {k: [] for k in ('clean', 'fixed', 'coupled', 'ring', 'y', 'g', 'domain', 'offset')}
    for item in case['bank']:
        packet = packets[item['group']]
        require(str(packet['origin']) == item['record'] and str(packet['group_key']) == item['group'],
                'exact bank origin and numerical group')
        by_ratio = {str(q): v for q, v in zip(packet['ratios'], packet['event'])}
        fixed = np.stack([by_ratio[q] for q in ('2/3', '3/2')], axis=1)
        coupled = (np.repeat(by_ratio[item['correct_ratio']][:, None], 2, axis=1)
                   if case['source_speed_diversity'] else fixed)
        ring = (np.repeat(by_ratio[item['ring_ratio']][:, None], 2, axis=1)
                if case['source_speed_diversity'] else fixed)
        for name, value in (('clean', packet['clean']), ('fixed', fixed), ('coupled', coupled), ('ring', ring)):
            result[name].append(value)
        result['y'].extend([item['label']] * 3)
        result['g'].extend([item['record']] * 3); result['domain'].extend([item['domain']] * 3)
        result['offset'].extend((packet['starts'] + 65536).tolist())
    result = {k: np.concatenate(v) if k in ('clean', 'fixed', 'coupled', 'ring') else np.asarray(v)
              for k, v in result.items()}
    require(len(result['y']) == case['bank_windows'], 'all original source bank records, three windows each')
    if not case['source_speed_diversity']:
        require(np.array_equal(result['fixed'], result['coupled']) and np.array_equal(result['fixed'], result['ring']),
                'same-clock input aliases are exact')
    return result
