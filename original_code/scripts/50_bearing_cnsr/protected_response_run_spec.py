"""Whole source plan/compiler, stdlib-only and no scientific payload reads."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from inventory_protected_response import merge_record
from protected_response_journal import canonical as scope_canonical

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/wfdg/ress_protected_response_preparation'
STORE = Path('D:/CodexTrainingArtifacts/causal-wfdg-01a0d84f/ress_protected_response_qualification/data')
DATABASE = OUT/'control/CONTROL.sqlite'
KIND = 'PROTECTED_RESPONSE_SOURCE_QUALIFICATION_V1'
GRANT_KIND = 'PROTECTED_RESPONSE_SOURCE_GRANT_V1'
ZERO = ('neural_models', 'optimizer_updates', 'target_predictions', 'new_inverse_fits')
MODULES = ('protected_response_run_spec.py', 'protected_response_run_runtime.py', 'protected_response_run_engine.py',
    'run_protected_response_source.py', 'protected_response_core.py', 'protected_response_payloads.py',
    'protected_response_numeric_audit.py', 'protected_response_journal.py', 'protected_response_journal_v2.py',
    'inventory_protected_response.py', 'clock_real_core.py', 'clock_real_data.py', 'stable_channel_core.py')


def read(path):
    path = Path(path)
    if path.suffix != '.json':
        raise PermissionError('metadata JSON only before source authorization')
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def environment():
    return dict(python_executable=str(Path(sys.executable).resolve()), python_version=platform.python_version(),
        numpy=importlib.metadata.version('numpy'), scipy=importlib.metadata.version('scipy'))


def load_plan():
    graph_path, meta_path = OUT/'SOURCE_WORK_GRAPH_V1.json', OUT/'SOURCE_METADATA_V1.json'
    graph, meta = read(graph_path), read(meta_path)
    if sha(graph_path) != '727112137308bf7619eac68381220466831d2c7dc659c3bcf7bcc806e30ed39d' or sha(meta_path) != graph['meta_binding']['sha256']:
        raise PermissionError('original full81 whole-source graph identity')
    documents, cases, records, inputs = {}, {}, {}, {str(graph_path): sha(graph_path), str(meta_path): sha(meta_path)}
    for c in graph['cases']:
        ref = c['original_case_reference']; path = ROOT/ref['path']
        if str(path) not in documents:
            documents[str(path)] = read(path); inputs[str(path)] = sha(path)
            rows = documents[str(path)].get('data', documents[str(path)])['records']
            for k, r in rows.items():
                records[k] = merge_record(records[k], r) if k in records else r
        x = documents[str(path)]
        for name in ref['field']:
            x = x[name]
        original = x[ref['index']]
        if canonical(original) != c['case_sha256'] or canonical(original['roles']) != c['role_sha256']:
            raise PermissionError('unchanged case and roles before numerical imports')
        cases[c['key']] = {k: v for k, v in original.items() if k != 'inventory'}
    inverses = {v['identity']: v for v in graph['inverse_units']}
    measurements = {v['record']: v for v in graph['measurement_units']}
    case_inverses = {}
    for c in meta['cases']:
        table = {}
        for pk, pair in cases[c['key']]['pairs'].items():
            matches = [i for i in c['required_inverse'] if meta['inverse_catalog'][i]['pair'] == pair]
            if len(matches) != 1:
                raise PermissionError('one exact original inverse per source pair/case')
            table[pk] = matches[0]
        case_inverses[c['key']] = table
    return dict(synthetic=False, graph=graph, meta=meta, cases=cases, records=records,
        inverses=inverses, measurements=measurements, case_inverses=case_inverses, input_sha256=inputs)


def compile_units(plan):
    units, order, costs = {}, [], {}
    def add(key, description, **bound):
        units[key] = description; order.append(key)
        costs[key] = dict(owned_packet_writes=1, owned_hash_checks=1)
        costs[key].update(bound)
    for audit in (False, True):
        prefix = 'A' if audit else ''
        for record, value in plan['measurements'].items():
            raw = value['kind'] == 'recover_existing_PU_measurement_only'
            add(prefix+'M/'+record, dict(kind=prefix+'M', item=record, case=value['witness_case']),
                source_payload_reads=1, source_hash_checks=1, raw_decodes=int(raw), context_decodes=int(not raw),
                owned_reads=int(audit), **({'owned_hash_checks': 2} if audit else {}))
        for identity, value in plan['inverses'].items():
            # Two optional measurement packets, and the generated inverse in audit.
            add(prefix+'I/'+identity, dict(kind=prefix+'I', item=identity, case=value['witness_case']),
                source_payload_reads=1, source_hash_checks=1, inverse_decodes=1, owned_reads=2+int(audit),
                fft_calls=6, fft_1d_transforms=194, fft_points=194*65536,
                **{'owned_hash_checks': 3+int(audit)})
        for c in plan['graph']['cases']:
            n = len(plan['cases'][c['key']]['bank']); required = len(set(plan['case_inverses'][c['key']].values()))
            per_calls, per_transforms = (20, 80) if audit else (334, 1336)
            add(prefix+'B/'+c['key'], dict(kind=prefix+'B', item=c['key'], case=c['key']),
                source_payload_reads=1, source_hash_checks=1, bank_decodes=1,
                owned_reads=required+int(audit), source_context_views=2*n,
                fft_calls=per_calls*n, fft_1d_transforms=per_transforms*n,
                fft_points=per_transforms*n*c['task']['window'],
                **{'owned_hash_checks': 1+required+int(audit)})
    limits = {}
    for value in costs.values():
        for k, v in value.items():
            limits[k] = limits.get(k, 0)+v
    limits.update(adoption_hash_checks=8, **{k: 0 for k in ZERO})
    return units, order, costs, limits


def make_lock(plan, *, store=STORE, database=DATABASE, code=None, total_seconds=21600):
    descriptions, order, units, limits = compile_units(plan)
    code = {str(ROOT/'scripts/50_bearing_cnsr'/n): sha(ROOT/'scripts/50_bearing_cnsr'/n) for n in MODULES} if code is None else code
    lock = dict(kind=KIND, synthetic=plan['synthetic'], execution_ready=False,
        scope=dict(input_sha256=plan['input_sha256'], code_sha256=code, environment=environment(), unit_descriptions=descriptions, order=order),
        units=units, limits=limits, store=str(Path(store).resolve()), database=str(Path(database).resolve()), total_seconds=total_seconds)
    lock['scope_fingerprint'] = scope_canonical(dict(scope=lock['scope'], units=units, limits=limits))
    lock['fingerprint'] = canonical(lock)
    return lock
