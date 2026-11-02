"""Source-only numerical probe compiler. Import and prepare are stdlib-only."""
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sqlite3
import protected_response_run_spec as parent
from inventory_protected_response import covered

ROOT = parent.ROOT
OUT = ROOT / 'results/wfdg/ress_e0_natural_condition_probe'
CODE = ('natural_mapping_probe_plan.py', 'natural_mapping_probe_core.py',
        'run_natural_mapping_probe.py', 'test_natural_mapping_probe.py',
        'clock_real_core.py', 'clock_real_data.py', 'protected_response_payloads.py',
        'protected_response_run_spec.py', 'protected_response_journal.py',
        'inventory_protected_response.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def clock(scope):
    require(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']),
            'original preparation-inclusive total deadline')


def json_write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_case(case, records, key, *, context=False, side=None, offsets=None):
    require(key in case['roles'] and key in records, 'record in current case source roles')
    record, role = records[key], case['roles'][key]
    require(record['domain'] in case['task']['sources'] and
        record['protocol'] == case['task']['protocol'] and
        record.get('bearing') not in ('K006', 'KA08', 'KA09'), 'current-case source, not reserved')
    if context:
        require(covered(role['tr'], record['window']) and not any(
            a < 98304 and a + record['window'] > 65536 for a in role['va']),
            'complete inverse context covered only by source train')
    if side is not None:
        require(side in ('tr', 'va') and all(a in role[side] for a in offsets),
                'exact current-case reference role and offsets')


def prepare():
    scope_path = OUT / 'SCOPE_V1.json'
    scope = read(scope_path); clock(scope)
    require(sha(ROOT / scope['catalog']) == scope['catalog_sha256'], 'unchanged V2 catalog')
    catalog = read(ROOT / scope['catalog'])
    require(len(catalog['units']) == 274, 'complete274 context proposal')
    plan = parent.load_plan()  # Metadata only; does not reenter the closed runtime.
    registry_path = ROOT / 'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json'
    registry = read(registry_path)
    require(canonical({k: v for k, v in registry.items() if k != 'fingerprint'}) ==
            registry['fingerprint'], 'original source registry fingerprint')
    features = {(b['binding']['protocol'], b['binding']['domain']): b['binding']
                for b in registry['caches'] if b['kind'] == 'full_feature'}
    old_lock_path = parent.OUT / 'SOURCE_RUN_LOCK_V1.json'
    old_lock = read(old_lock_path)
    database = Path(old_lock['database'])
    final_path = parent.OUT / 'SOURCE_QUALIFICATION_FINAL_V1.json'
    old_final = read(final_path)
    require(sha(database) == old_final['database']['sha256'], 'unchanged closed parent database bytes')
    db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        require(db.execute("SELECT value FROM meta WHERE key='DONE'").fetchone(), 'parent source qualification closed')
        rows = {k: json.loads(v) for k, v in db.execute("SELECT key,detail FROM units WHERE status='DONE'")}
    finally:
        db.close()
    units, payloads = [], {}

    def binding(value, kind):
        p = str(Path(value['path']).resolve())
        require(Path(p).drive.upper() == 'D:' and Path(p).is_file(), 'existing D-only scientific cache')
        require(kind in ('inverse', 'qualified_context', 'source_feature'), 'only source scientific inputs')
        key = canonical(dict(path=p, sha256=value['sha256']))
        if key in payloads:
            require(payloads[key]['binding']['sha256'] == value['sha256'], 'one immutable byte identity')
        else:
            payloads[key] = dict(kind=kind, binding=dict(value, path=p))
        return key

    for original in catalog['units']:
        unit = dict(original)
        case_key = unit['protocol'] + '/' + unit['proposed_from_case'] + '/s0'
        case = plan['cases'][case_key]
        pair = case['pairs'][unit['inverse_pair_key']]
        require(pair == unit['inverse_pair'], 'actual selected source-pair identity')
        for record in pair:
            source_case(case, plan['records'], record, context=True)
        identity = plan['case_inverses'][case_key][unit['inverse_pair_key']]
        inv = plan['inverses'][identity]['binding']
        require(inv['pair'] == pair, 'actual versioned inverse binding, not guessed filename')
        receipt = rows['I/' + identity]
        require(receipt['scope_fingerprint'] == old_lock['scope_fingerprint'], 'completed measurement packet lineage')
        if unit['protocol'] != 'jnu_speed':
            require(unit['natural_reference']['record'] not in pair, 'reference recording withheld from this inverse')
        reference = unit['natural_reference']
        source_case(case, plan['records'], reference['record'], side=reference['role'], offsets=reference['offsets'])
        r = plan['records'][reference['record']]
        require(r['label'] == unit['label'] and r['domain'] == unit['destination'], 'natural reference label/domain')
        if unit['protocol'] == 'jnu_speed':
            require(reference['role'] == 'va' and all(a >= 98304 or a + r['window'] <= 65536
                for a in reference['offsets']), 'JNU disjoint original validation time block')
        q = Fraction(unit['correct_ratio'])
        actual = Fraction(str(r['shaft_hz'])) / Fraction(str(plan['records'][unit['origin_record']]['shaft_hz']))
        require(abs(float(q - actual)) <= 1e-12, 'correct source-only clock ratio')
        f = features[unit['protocol'], unit['destination']]
        require(f['FFT_code_sha256'] == registry['expected_FFT_code_sha256'], 'same frozen logFFT frontend provenance')
        unit.update(case_key=case_key,
            inverse_binding=binding(inv, 'inverse'),
            context_binding=binding(receipt['packet'], 'qualified_context'),
            reference_feature_binding=binding(f, 'source_feature'),
            actual_inverse_identity=identity,
            original_source_case_sha256=canonical(case))
        units.append(unit)
    require(sha(ROOT / 'scripts/50_bearing_cnsr/clock_real_data.py') ==
            registry['expected_FFT_code_sha256'], 'current exact old frontend code')
    require(len(payloads) <= scope['limits']['payload_decodes'], 'all caches fit finite decode budget')
    per_unit = []
    for unit in units:
        q = Fraction(unit['correct_ratio']); phases = 1 + q.numerator + q.denominator
        per_unit.append(dict(key=unit['unit_key'],
            fft_1d_transforms=96 * phases + 3 + 12,
            fft_points=(96 * phases + 3) * 65536 + 12 * unit['window']))
    inputs = {str(scope_path): sha(scope_path), str(ROOT / scope['catalog']): sha(ROOT / scope['catalog']),
        str(registry_path): sha(registry_path), str(old_lock_path): sha(old_lock_path),
        str(final_path): sha(final_path),
        **plan['input_sha256']}
    lock = dict(kind=scope['kind'], scope=scope, units=units, payloads=payloads,
        input_sha256=inputs,
        code_sha256={str(ROOT / 'scripts/50_bearing_cnsr' / p): sha(ROOT / 'scripts/50_bearing_cnsr' / p) for p in CODE},
        formal_forecast=dict(units=274, context_renders=822, encoded_windows=3288,
            reference_windows=822, unique_payloads=len(payloads),
            fft_1d_transforms=sum(v['fft_1d_transforms'] for v in per_unit),
            fft_points=sum(v['fft_points'] for v in per_unit)),
        unit_fft_bounds=per_unit, parent_database_readonly=str(database),
        parent_scope_fingerprint=old_lock['scope_fingerprint'],
        source_only=True, execution_ready=True,
        proposed_at=datetime.now().astimezone().isoformat())
    lock['fingerprint'] = canonical(lock)
    clock(scope); json_write(OUT / 'RUN_LOCK_V1.json', lock)
    return lock


if __name__ == '__main__':
    lock = prepare()
    print(json.dumps(dict(fingerprint=lock['fingerprint'], forecast=lock['formal_forecast']), indent=2))
