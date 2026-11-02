"""Metadata/synthetic-only comparison preparation, NOT a neural run command."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pcr_physics_adapter as p

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_pcr_comparison_preparation'
MANIFEST = ROOT / 'results/wfdg/ress_event_mechanism_factorial/DATA_MANIFEST.json'
OLD_LOCK = ROOT / 'results/wfdg/ress_event_mechanism_factorial/RUN_LOCK.json'
FOLDS = ('BCDtoA', 'ACDtoB', 'ABDtoC', 'ABCtoD')
SEEDS = (0, 1, 2)
FRONTENDS = ('NATIVE40', 'SHORT_PHYSICAL_WIDTH')
PROFILES = ('AUTHOR_CE', 'CE_PLUS_REAL_HEALTHY', 'PCR_FAULT_TO_HEALTHY')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def output(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if canonical(read(path)) != canonical(value):
            raise FileExistsError('preserve prior preparation artifact; use a new version: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(data)


def case_metadata(case, records):
    task = case['task']
    if task['fold'] not in FOLDS or case['seed'] not in SEEDS or task['classes'] != 3:
        raise ValueError('fixed existing PU scope')
    groups = {'source_train': [], 'source_validation': [], 'target': []}
    counts = {name: Counter() for name in groups}
    source_domains = set(task['sources'])
    if source_domains | {task['target']} != set('ABCD') or task['target'] in source_domains:
        raise ValueError('fixed 3-source/1-target task')
    for key, role in sorted(case['roles'].items()):
        r = records[key]
        if r['domain'] not in source_domains or set(role) != {'tr', 'va'}:
            raise PermissionError('source roles must stay source-only')
        active = [k for k in ('tr', 'va') if role[k]]
        if len(active) != 1 or tuple(role[active[0]]) != p.SHORT_OFFSETS:
            raise PermissionError('record-disjoint original roles, all original 60 windows')
        name = 'source_train' if active[0] == 'tr' else 'source_validation'
        groups[name].append(key)
    groups['target'] = sorted(k for k, r in records.items() if r['domain'] == task['target'])
    flattened = [key for group in groups.values() for key in group]
    if len(flattened) != 1440 or len(set(flattened)) != 1440:
        raise ValueError('source/validation/target records must partition all 1440')
    for name, keys in groups.items():
        for key in keys:
            r = records[key]
            if (r['bearing'] in {'K006', 'KA08', 'KA09'} or r['fs'] != 64000
                    or r['shaft_hz'] not in (15., 25.) or r['window'] != 5120
                    or tuple(r['offsets']) != p.SHORT_OFFSETS):
                raise PermissionError('known PU physical/window/retained-bearing boundary')
            count = p.native_layout(p.RECORD_CAP, r['fs'], r['shaft_hz'])['windows']
            counts[name]['records'] += 1
            counts[name]['short_windows'] += 60
            counts[name]['native_windows'] += count
            counts[name][f"domain={r['domain']};local_class={r['label']};records"] += 1
            counts[name][f"domain={r['domain']};local_class={r['label']};native_windows"] += count
            counts[name][f"domain={r['domain']};local_class={r['label']};short_windows"] += 60
    if [counts[n]['records'] for n in groups] != [864, 216, 360]:
        raise ValueError('unchanged original record roles')
    # No target label is used to form an auxiliary pool, only count metadata.
    pools = {'factual': {}, 'extra_healthy': {}, 'fault_cf': {}}
    for domain in task['sources']:
        for cls in range(3):
            keys = [k for k in groups['source_train'] if records[k]['domain'] == domain and records[k]['label'] == cls]
            if not keys:
                raise ValueError('every source domain/class must remain represented')
            pools['factual'][f'{domain}/{cls}'] = keys
            if cls == 0:
                pools['extra_healthy'][domain] = keys
            else:
                pools['fault_cf'][f'{domain}/{cls}'] = keys
    return {'task': task, 'seed': case['seed'], 'roles': groups,
            'counts': {n: dict(c) for n, c in counts.items()},
            'source_train_pools': pools,
            'original_source_fingerprint': case['inventory']['source_fingerprint'],
            'role_fingerprint': canonical(groups)}


def bills(cases):
    formal_models = len(cases) * len(FRONTENDS) * len(PROFILES)
    formal_updates = formal_models * 2000
    source = sum(c['counts']['source_validation']['native_windows']
                 + c['counts']['source_validation']['short_windows'] for c in cases) * len(PROFILES) * 3
    target = sum(c['counts']['target']['native_windows']
                 + c['counts']['target']['short_windows'] for c in cases) * len(PROFILES) * 2
    # Two existing cases x two frontends x three profiles x (10 uninterrupted + 5+5 resume).
    qualification_updates = 2 * 2 * 3 * 20
    # Native author forwards factual and auxiliary separately; CE only has no aux forward.
    formal_forward = len(cases) * len(FRONTENDS) * 2000 * 200 * (1 + 2 + 2)
    qualification_forward = 2 * 2 * 20 * 200 * (1 + 2 + 2)
    qualification_source = 12 * 4 * 128
    return dict(formal_models=formal_models, formal_updates=formal_updates,
                qualification_updates=qualification_updates, repair_attempt_reserve=60,
                total_update_attempt_cap=formal_updates + qualification_updates + 60,
                formal_training_forward_windows=formal_forward,
                qualification_training_forward_windows=qualification_forward,
                formal_source_predictions=source, qualification_source_predictions=qualification_source,
                planned_source_predictions=source + qualification_source,
                formal_target_predictions=target,
                note='Proposed counts, not approved or executed. Repair forward/prediction reserves must be budgeted explicitly before a neural approval request.')


def prepare():
    if 'torch' in sys.modules:
        raise PermissionError('this entry point is non-neural and must not import Torch')
    raw = read(MANIFEST)
    lock = read(OLD_LOCK)
    legacy_hash = sha256(json.dumps({k: v for k, v in lock.items() if k != 'fingerprint'},
                                   sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    if legacy_hash != lock['fingerprint']:
        raise ValueError('historical manifest binding lock invalid')
    if lock['inputs'][str(MANIFEST).replace('/', '\\')] != digest(MANIFEST):
        raise ValueError('historical metadata SHA drift')
    records = {k: r for k, r in raw['records'].items() if r['protocol'] == 'pu_clock4'}
    cases = [case_metadata(c, records) for c in raw['cases'] if c['task']['protocol'] == 'pu_clock4']
    if {(c['task']['fold'], c['seed']) for c in cases} != {(f, s) for f in FOLDS for s in SEEDS} or len(cases) != 12:
        raise ValueError('exact existing 4 tasks x 3 seeds')
    budget = bills(cases)
    if budget != dict(formal_models=72, formal_updates=144000, qualification_updates=240,
                      repair_attempt_reserve=60, total_update_attempt_cap=144300,
                      formal_training_forward_windows=48000000, qualification_training_forward_windows=80000,
                      formal_source_predictions=2501928, qualification_source_predictions=6144,
                      planned_source_predictions=2508072, formal_target_predictions=2779920,
                      note=budget['note']):
        raise AssertionError('all finite proposed counts must be independently reconciled: ' + repr(budget))
    result = subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8',
                             str(Path(__file__).with_name('test_pcr_physics_adapter.py'))],
                            capture_output=True, text=True, encoding='utf-8', check=False)
    if result.returncode:
        print(result.stdout + result.stderr, flush=True)
        raise RuntimeError('synthetic fixture exit ' + str(result.returncode))
    model_source, removed = p.author_model_definition_source(strip_logging=True)
    model_path = OUT / 'AUTHOR_MODEL_PREPARED_NOT_EXECUTED.py'
    if model_path.exists() and model_path.read_text(encoding='utf-8') != model_source:
        raise FileExistsError('preserve prepared model version')
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(model_source)
    manifest = dict(version='PCR_METADATA_PROPOSAL_V1_NOT_EXECUTABLE_AUTHORITY',
                    source_metadata=dict(path=str(MANIFEST), sha256=digest(MANIFEST),
                                         historical_lock_fingerprint=lock['fingerprint']),
                    records=records, cases=cases, frontends=FRONTENDS, profiles=PROFILES,
                    observation_support=dict(cap_samples=p.RECORD_CAP, sampling_rate=64000,
                                             duration_seconds=p.RECORD_CAP / 64000,
                                             short_offsets=p.SHORT_OFFSETS, short_samples=5120,
                                             native_revs=40, native_step_revs=1),
                    proposed_bill=budget,
                    source_label_mapping={'local_0_H': 0, 'local_1_OF': 2, 'local_2_IF': 1},
                    nominal_geometry=dict(pitch_mm=28.55, ball_mm=6.75, balls=8, contact_degrees=0,
                                          role='Table 1 KB24/FAG example; not per-bearing measured geometry',
                                          source='https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf',
                                          source_sha256='c1f75c79cfbcbcbfd24b32bf034cebb76d1c3ff7864827575d37387bb0955f84'),
                    author_source_sha256=p.AUTHOR_SHA,
                    prepared_model=dict(path=str(model_path), sha256=digest(model_path),
                                        logging_only_assignments_removed=removed,
                                        neural_construction=False, cuda_equivalence='NOT_RUN_NO_NEW_APPROVAL'),
                    execution_state='PREPARATION_ONLY_NOT_READY_FOR_NEURAL_LAUNCH')
    manifest['fingerprint'] = canonical(manifest)
    output(OUT / 'METADATA_PROPOSAL.json', manifest)
    source_names = ('pcr_physics_adapter.py', 'test_pcr_physics_adapter.py', 'prepare_pcr_comparison.py')
    receipt = dict(status='SYNTHETIC_AND_METADATA_PREPARATION_PASS_NOT_CUDA',
                   at=datetime.now().astimezone().isoformat(),
                   fixture_command=result.args, fixture_exit_code=result.returncode,
                   fixture_output=result.stdout + result.stderr,
                   proposed_manifest_sha256=digest(OUT / 'METADATA_PROPOSAL.json'),
                   code_sha256={str(Path(__file__).with_name(n)): digest(Path(__file__).with_name(n)) for n in source_names},
                   actual_neural_models=0, actual_neural_forwards=0, actual_updates=0,
                   actual_real_signal_feature_or_checkpoint_reads=0,
                   unresolved=['durable approved-campaign runtime', 'actual CUDA output/gradient/RNG/state equivalence',
                               'disk restore and exactly-once attempt admission', 'full independent source/target audit',
                               'repair forward/prediction reserves and actual timing admission', 'new human finite comparison approval'])
    output(OUT / 'PREPARATION_RECEIPT.json', receipt)
    print(json.dumps(dict(status=receipt['status'], proposed=budget, neural_execution=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
