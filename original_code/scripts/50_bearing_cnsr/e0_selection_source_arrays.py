"""Assemble source arrays supplied by a future authorized loader.

Pure NumPy; no file/model/authorization interface. Caller must prove file SHA,
cache provenance and a new grant BEFORE supplying real arrays to this module.
"""
import numpy as np
from audit_e0_source_geometry import covered, temporal_disjoint


def frozen(value):
    value = np.asarray(value);value.setflags(write=False);return value


class DomainRows:
    def __init__(self, arrays, records, protocol, domain):
        selected = sorted((k, r) for k, r in records.items()
                          if (r['protocol'], r['domain']) == (protocol, domain))
        y = np.asarray([r['label'] for k, r in selected for _ in r['offsets']], np.int64)
        g = np.asarray([k for k, r in selected for _ in r['offsets']])
        offset = np.asarray([v for k, r in selected for v in r['offsets']], np.int64)
        if not len(y):raise ValueError('declared source domain has no metadata')
        x = arrays['x']
        if x.shape != (len(y), 2048) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('finite source FP32 logFFT2048')
        if any(not np.array_equal(arrays[k], v) for k, v in (('y', y), ('g', g), ('offset', offset))):
            raise ValueError('full domain label/group/offset order does not match original metadata')
        self.protocol, self.domain = protocol, domain
        self.x = frozen(x.view());self.y = frozen(y);self.g = frozen(g);self.offset = frozen(offset)
        self.positions = {};cursor = 0
        for key, r in selected:
            if len(set(r['offsets'])) != len(r['offsets']):raise ValueError('duplicate domain offset')
            self.positions[key] = {o: cursor + i for i, o in enumerate(r['offsets'])};cursor += len(r['offsets'])


def allowed_support(case, records, requested):
    if not requested:raise PermissionError('explicit nonempty auxiliary support')
    for key, offsets in requested.items():
        if key not in case['roles'] or key not in records or not offsets or len(set(offsets)) != len(offsets):
            raise PermissionError('unknown/empty/duplicate source support')
        r, old = records[key], case['roles'][key]
        if r['protocol'] != case['task']['protocol'] or r['domain'] not in case['task']['sources']:
            raise PermissionError('target/foreign source record refused')
        if list(offsets) != sorted(offsets) or not set(offsets) <= set(old['tr']):
            raise PermissionError('auxiliary support must remain inside original source training offsets')
        if case['task']['protocol'] == 'jnu_speed':
            if not temporal_disjoint(offsets, old['va'], r['window']):raise PermissionError('source validation raw interval overlaps')
        elif old['va']:
            raise PermissionError('recording-group validation cannot enter auxiliary support')


def source_pack(case, records, features, wanted):
    allowed_support(case, records, wanted)
    if set(features) != set(case['task']['sources']):raise PermissionError('only current declared source domains')
    values = {k: [] for k in ('x', 'y', 'g', 'domain', 'offset')}
    for key, offsets in sorted(wanted.items()):
        r = records[key];f = features[r['domain']]
        if (f.protocol, f.domain) != (case['task']['protocol'], r['domain']):raise PermissionError('source domain cache identity')
        ix = [f.positions[key][o] for o in offsets]
        values['x'].append(f.x[ix]);values['y'].append(f.y[ix]);values['g'].append(f.g[ix])
        values['offset'].append(f.offset[ix]);values['domain'].append(np.repeat(r['domain'], len(ix)))
    result = {k: frozen(np.concatenate(v)) for k, v in values.items()}
    if set(result['y'].tolist()) != set(range(case['task']['classes'])):
        raise ValueError('all auxiliary global classes required; missing domain cells are not fake-filled')
    return result


def query_pack(case, records, bank, *, bank_item_indices, training_support, probe_support, mode):
    """Bind frozen teacher query rows to EXACT historical bank crop positions.

    Query-record independence does not imply an unseen inverse generator or
    unseen donor: donor overlap is returned explicitly, never renamed away.
    """
    allowed_support(case, records, training_support);allowed_support(case, records, probe_support)
    if mode not in ('record', 'temporal'):raise ValueError('explicit source independence type')
    if mode == 'record' and set(training_support) & set(probe_support):raise PermissionError('teacher has seen query recording')
    if mode == 'temporal':
        if case['task']['protocol'] != 'jnu_speed':raise PermissionError('temporal mode is the existing JNU protocol only')
        for key in set(training_support) & set(probe_support):
            if not temporal_disjoint(training_support[key], probe_support[key], records[key]['window']):
                raise PermissionError('teacher/probe raw intervals share samples')
    items = case['bank'];n = 4 * len(items)
    y = np.repeat([records[v['record']]['label'] for v in items], 4).astype(np.int64)
    g = np.repeat([v['record'] for v in items], 4)
    domain = np.repeat([records[v['record']]['domain'] for v in items], 4)
    for name, expected in (('y', y), ('g', g), ('domain', domain)):
        if not np.array_equal(bank[name], expected):raise ValueError('edited bank label/record/domain row drift')
    if bank['clean'].shape != (n, 2048) or bank['event'].shape != (n, 2, 2048):raise ValueError('two-factor historical bank axes')
    for name in ('clean', 'event'):
        if bank[name].dtype != np.float32 or not np.isfinite(bank[name]).all():raise ValueError('finite FP32 source bank')
    indices = list(bank_item_indices)
    if not indices or indices != sorted(indices) or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(items) for i in indices):
        raise ValueError('unique fixed bank item order')
    rows, starts, window_ids, donor_overlap = [], [], [], set()
    for index in indices:
        item = items[index];key = item['record'];r = records[key];window = r['window']
        if key not in probe_support or len(item['starts']) != 4:raise PermissionError('query outside declared held-out training support')
        for j, start in enumerate(item['starts']):
            if not covered(probe_support[key], start, start + window, window):raise PermissionError('query crop crosses held-out support boundary')
            if key in training_support and not temporal_disjoint(training_support[key], [start], window):raise PermissionError('query crop shares teacher raw samples')
            rows.append(4 * index + j);starts.append(start);window_ids.append(f'{key}|{start}')
        partners = case['pairs'][item['pair']] if item['pair'] is not None else [key]
        for donor in partners:
            if donor not in case['roles'] or not case['roles'][donor]['tr']:raise PermissionError('foreign inverse donor')
            if donor in training_support:donor_overlap.add(donor)
            if mode == 'temporal':
                if donor not in probe_support or not covered(probe_support[donor], 65536, 98304, window):
                    raise PermissionError('legacy inverse context outside early JNU source support')
                if any(s < 98304 and s + window > 65536 for s in training_support.get(donor, [])):
                    raise PermissionError('inverse context shares teacher raw samples')
    if len(set(window_ids)) != len(window_ids):raise ValueError('duplicate physical query crop')
    result = dict(clean=frozen(bank['clean'][rows].copy()), event=frozen(bank['event'][rows].copy()),
        y=frozen(y[rows]), g=frozen(g[rows]), domain=frozen(domain[rows]),
        offset=frozen(np.asarray(starts, np.int64)), row_index=frozen(np.asarray(rows, np.int64)),
        window_ids=window_ids, independence_mode=mode,
        donor_file_ids_also_in_teacher_training=sorted(donor_overlap),
        inverse_generator_unseen=False, new_entity_validation=False)
    return result
