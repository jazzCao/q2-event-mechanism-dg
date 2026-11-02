"""Bind three lexicographic source pairs; stdlib only, zero array opens."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/wfdg/ress_abduction_action_qualification'
PARENT = ROOT/'results/wfdg/ress_source_rate_screen'


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_role(case, records, key):
    record, role = records[key], case['roles'][key]
    require(record['protocol'] == case['task']['protocol'] and
        record['domain'] in case['task']['sources'] and record.get('bearing') not in ('K006', 'KA08', 'KA09'),
        'actual unprotected source role')
    start = end = None
    for offset in sorted(role['tr']):
        if end is not None and offset <= end:
            end = max(end, offset+record['window'])
        else:
            start, end = offset, offset+record['window']
        if start <= 65536 and end >= 98304:
            break
    else:
        raise ValueError('full source inverse context not covered by TRAIN')
    require(not any(a < 98304 and a+record['window'] > 65536 for a in role['va']), 'no validation overlap')


def run():
    require(not (OUT/'SOURCE_INPUT_LOCK_V1.json').exists(), 'already bound; no metadata replay')
    scope = json.loads((OUT/'SCOPE_V1.json').read_text(encoding='utf8'))
    require(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']), 'original total deadline')
    counts = dict(source_metadata_JSON_opens=0, source_metadata_JSON_bytes=0)
    inputs = {}
    def read(name, expected=None):
        path = PARENT/name
        size = path.stat().st_size
        require(counts['source_metadata_JSON_opens']+1 <= scope['limits']['source_metadata_JSON_opens'] and
                counts['source_metadata_JSON_bytes']+size <= scope['limits']['source_metadata_JSON_bytes'], 'metadata quota before read')
        counts['source_metadata_JSON_opens'] += 1; counts['source_metadata_JSON_bytes'] += size
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        require(expected is None or digest == expected, 'closed metadata byte binding')
        inputs[str(path)] = digest
        return json.loads(data)
    receipt = read('CLOSED_RECEIPT_V1.json')
    require(receipt['execution'] == 'COMPLETE' and receipt['producer']['native_absent_verified'], 'parent actually closed')
    neural = read('NEURAL_LOCK_V2.json', receipt['files_sha256']['NEURAL_LOCK_V2.json'])
    prep = read('SOURCE_PREPARATION_LOCK_V1.json', neural['input_sha256'][str(PARENT/'SOURCE_PREPARATION_LOCK_V1.json')])
    require(prep['fingerprint'] == canonical({k: v for k, v in prep.items() if k != 'fingerprint'}), 'source preparation fingerprint')
    groups = {item['key']: item for item in prep['groups']}
    selected = []
    for protocol in ('pu_clock4', 'phm09_w3', 'jnu_speed'):
        cases = sorted((v for v in prep['cases'] if v['seed'] == 0 and v['task']['protocol'] == protocol), key=lambda v: v['key'])
        case = cases[0]
        original = neural['original_cases'][case['key']]
        require(canonical(original) == case['original_case_sha256'] and
                canonical(original['roles']) == case['original_roles_sha256'], 'original case and roles unchanged')
        chosen = sorted((groups[item['group']] for item in case['bank']),
                        key=lambda g: (g['inverse_identity'], g['source'], g['key']))[0]
        asset = prep['assets'][chosen['inverse_identity']]
        pair = chosen['pair']
        require(asset['inverse']['pair'] == pair and pair in original['pairs'].values(), 'actual source pair')
        for key in pair:
            check_role(original, neural['records'], key)
        require(len(set(neural['records'][key]['label'] for key in pair)) == 1, 'same coarse class source inverse')
        bindings = {}
        for kind in ('inverse', 'qualified_context'):
            binding = asset[kind]
            path = Path(binding['path'])
            require(path.drive.upper() == 'D:' and path.is_file() and path.suffix == '.npz', 'existing D source-only NPZ')
            bindings[kind] = dict(binding, size_bytes=path.stat().st_size)
        selected.append(dict(protocol=protocol, case_key=case['key'], original_case_sha256=case['original_case_sha256'],
            original_roles_sha256=case['original_roles_sha256'], group=chosen, bindings=bindings,
            records={key: {k: neural['records'][key].get(k) for k in ('key', 'protocol', 'domain', 'label', 'bearing', 'fs', 'window', 'shaft_hz')}
                     for key in pair}, source_context_role='TRAIN_ONLY_65536_98304'))
    total_bytes = sum(b['size_bytes'] for item in selected for b in item['bindings'].values())
    require(total_bytes <= scope['limits']['source_inverse_NPZ_bytes'], 'three inverse plus three context byte cap')
    for name in ('clock_real_core.py', 'clock_real_data.py'):
        path = ROOT/'scripts/50_bearing_cnsr'/name
        require(sha(path) == prep['code_sha256'][str(path)], 'same original physics and frontend')
    require('torch' not in sys.modules and 'numpy' not in sys.modules, 'metadata-only compiler')
    require(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']), 'deadline before lock')
    value = dict(kind='THREE_LEXICOGRAPHIC_SOURCE_INVERSE_PAIRS_V1', scope=scope,
        amendment_sha256=sha(OUT/'PRE_PAYLOAD_AMENDMENT.md'), metadata_input_sha256=inputs,
        counts=counts, selected=selected, total_payload_bytes=total_bytes,
        payload_limit=dict(inverse=3, qualified_context=3, combined_bytes=scope['limits']['source_inverse_NPZ_bytes']),
        ratios=['1', '2/3', '3/2'], coefficient_l1_addition_fraction=.01,
        construction='first and second ranked carrier, shortest decay, phase0/pi; seed=sum nonnegative old coefficients over channels',
        source_choice_uses_target_or_accuracy=False, arrays_opened=0,
        at=datetime.now().astimezone().isoformat())
    value['fingerprint'] = canonical(value)
    with (OUT/'SOURCE_INPUT_LOCK_V1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(fingerprint=value['fingerprint'], counts=counts, total_payload_bytes=total_bytes,
        selected=[dict(protocol=v['protocol'], case_key=v['case_key'], pair=v['group']['pair'], source=v['group']['source']) for v in selected]), indent=2))


if __name__ == '__main__':
    run()
