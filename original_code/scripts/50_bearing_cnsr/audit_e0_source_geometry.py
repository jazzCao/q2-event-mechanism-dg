"""Metadata-only auxiliary support for the unchanged27-task main panel."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from audit_e0_train_only_design import balanced_folds, source_roles

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_e0_source_geometry'
INPUTS = ('results/wfdg/ress_event_mechanism_factorial/DATA_MANIFEST.json',
          'results/wfdg/ress_event_phm_main/DATA_PROPOSAL.json',
          'results/wfdg/ress_e0_train_only_design/AUDIT_RECEIPT.json',
          'results/wfdg/ress_e0_train_only_design/CPU_TESTS.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def temporal_disjoint(left, right, window):
    left, right = sorted(left), sorted(right)
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] + window <= right[j]:
            i += 1
        elif right[j] + window <= left[i]:
            j += 1
        else:
            return False
    return True


def covered(starts, lo, hi, window):
    intervals = []
    for start in sorted(starts):
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], start + window)
        else:
            intervals.append([start, start + window])
    return any(a <= lo and hi <= b for a, b in intervals)


def partition_time(train, validation, window):
    assert train and len(set(train)) == len(train) and len(set(validation)) == len(validation)
    assert temporal_disjoint(train, validation, window), 'original validation overlap'
    boundary = (min(train) + max(train) + window) // 2
    early = sorted(s for s in train if s + window <= boundary)
    late = sorted(s for s in train if s >= boundary)
    guard = sorted(set(train) - set(early) - set(late))
    assert early and late and temporal_disjoint(early, late, window)
    assert set(early) | set(late) | set(guard) == set(train)
    assert not set(early) & set(late)
    return dict(boundary=boundary, early=early, late=late, auxiliary_guard=guard)


def original_roles(case, records, whole_record):
    task = case['task']
    assert set(task['sources']) and task['target'] not in task['sources']
    for name, role in case['roles'].items():
        r = records[name]
        assert r['protocol'] == task['protocol'] and r['domain'] in task['sources']
        assert set(role) == {'tr', 'va'}
        assert not set(role['tr']) & set(role['va'])
        assert set(role['tr'] + role['va']) <= set(r['offsets'])
        if whole_record:
            assert bool(role['tr']) != bool(role['va'])
            assert (role['tr'] or role['va']) == r['offsets']
        else:
            assert temporal_disjoint(role['tr'], role['va'], r['window'])
    train = {k for k, v in case['roles'].items() if v['tr']}
    validation = {k for k, v in case['roles'].items() if v['va']}
    assert {b['record'] for b in case['bank']} == train
    for item in case['bank']:
        assert len(item['starts']) == 4
        if item['pair'] is not None:
            assert set(case['pairs'][item['pair']]) <= train
    return train, validation


def pu_geometry(case, records):
    train, validation = original_roles(case, records, True)
    grouped = defaultdict(list)
    for name in train:
        r = records[name]
        assert r['bearing'] not in {'K006', 'KA08', 'KA09'}
        grouped[r['domain'], r['bearing']].append(name)
    held = [set(), set()]
    for names in grouped.values():
        assert len(names) >= 2 and len(names) % 2 == 0
        for i, name in enumerate(sorted(names, key=lambda k: (records[k]['run'], k))):
            held[i % 2].add(name)
    folds = []
    expected = {(d, y) for d in case['task']['sources'] for y in range(case['task']['classes'])}
    for i in (0, 1):
        learning, probe = train - held[i], held[i]
        assert not (learning | probe) & validation and not learning & probe
        assert {(records[k]['domain'], records[k]['label']) for k in learning} == expected
        assert {(records[k]['domain'], records[k]['label']) for k in probe} == expected
        assert {records[k]['bearing'] for k in learning} == {records[k]['bearing'] for k in probe}
        folds.append(dict(fold=i, train_records=sorted(learning), probe_records=sorted(probe),
                          training_windows=sum(len(case['roles'][k]['tr']) for k in learning),
                          probe_windows=sum(len(case['roles'][k]['tr']) for k in probe),
                          global_classes=case['task']['classes'],
                          present_training_domain_class_cells=len(expected),
                          new_entity_validation=False))
    assert not held[0] & held[1] and held[0] | held[1] == train
    return dict(mode='SAME_BEARINGS_DIFFERENT_SOURCE_TRAIN_RECORDINGS', folds=folds,
                original_train_records=len(train), original_validation_records=len(validation),
                preserved_bearing_count=len({records[k]['bearing'] for k in train}),
                original_main_roles_changed=False, new_entity_validation=False)


def jnu_geometry(case, records):
    train, _ = original_roles(case, records, False)
    parts = {k: partition_time(case['roles'][k]['tr'], case['roles'][k]['va'], records[k]['window'])
             for k in sorted(train)}
    window = case['task']['window']
    # Every existing query is early; one late-trained helper suffices. Do not
    # add an unused reverse teacher just to mimic a recording protocol.
    for item in case['bank']:
        part = parts[item['record']]
        for start in item['starts']:
            assert covered(part['early'], start, start + window, window), 'bank query not in early probe'
            assert temporal_disjoint(part['late'], [start], window)
        partners = case['pairs'][item['pair']] if item['pair'] is not None else [item['record']]
        for name in partners:
            assert covered(parts[name]['early'], 65536, 98304, window), 'legacy context not in early support'
            assert all(s + window <= 65536 or s >= 98304 for s in parts[name]['late'])
    expected = {(d, y) for d in case['task']['sources'] for y in range(case['task']['classes'])}
    assert {(records[k]['domain'], records[k]['label']) for k in parts} == expected
    return dict(mode='SAME_FILE_EARLY_PROBE_LATE_TRAIN_FULL_INTERVAL_DISJOINT', temporal_roles=parts,
                training_windows=sum(len(p['late']) for p in parts.values()),
                factual_probe_windows=sum(len(p['early']) for p in parts.values()),
                auxiliary_guard_windows=sum(len(p['auxiliary_guard']) for p in parts.values()),
                original_main_training_windows=sum(len(v['tr']) for v in case['roles'].values()),
                present_training_domain_class_cells=len(expected), global_classes=case['task']['classes'],
                proposed_helper_slots=1, original_main_roles_changed=False,
                main_training_guard_windows_removed=0, new_recording_validation=False,
                new_entity_validation=False, all_bank_queries_and_legacy_contexts_disjoint_from_helper_training=True)


def main():
    old, phm, previous, previous_tests = [read(ROOT / path) for path in INPUTS]
    assert previous_tests['tests'] == 7
    for name, digest in previous['output_sha256'].items():
        assert sha(ROOT / 'results/wfdg/ress_e0_train_only_design' / name) == digest
    cases = old['cases'] + phm['data']['cases']
    assert len(cases) == 81
    profiles = {}; rows = []
    for case in cases:
        protocol = case['task']['protocol']
        records = phm['data']['records'] if protocol == 'phm09_w3' else old['records']
        if protocol == 'pu_clock4':
            geometry = pu_geometry(case, records)
        elif protocol == 'jnu_speed':
            geometry = jnu_geometry(case, records)
        else:
            assert protocol == 'phm09_w3'
            sources, training, validation = source_roles(case, records)
            folds = balanced_folds(sources, training, records)
            assert all(not (set(f['train_records']) | set(f['probe_records'])) & validation for f in folds)
            geometry = dict(mode='ORIGINAL_TRAIN_POOL_CROSS_SOURCE_CLASS_INTERLEAVED_RECORDINGS', folds=folds,
                            original_train_records=26, original_validation_records=6,
                            original_main_roles_changed=False, new_entity_validation=False)
        ident = canonical(dict(protocol=protocol, sources=case['task']['sources'], roles=case['roles']))
        if ident not in profiles:
            profiles[ident] = dict(profile=ident, protocol=protocol, sources=case['task']['sources'],
                geometry=geometry, original_role_fingerprint=canonical(case['roles']), case_references=[])
        else:
            assert profiles[ident]['geometry'] == geometry
        profiles[ident]['case_references'].append(dict(fold=case['task']['fold'], seed=case['seed']))
        rows.append(dict(protocol=protocol, fold=case['task']['fold'], seed=case['seed'], profile=ident,
                         main_role_fingerprint=canonical(case['roles']), auxiliary_mode=geometry['mode']))
    summary = {}
    for protocol in ('pu_clock4', 'phm09_w3', 'jnu_speed'):
        selected = [p for p in profiles.values() if p['protocol'] == protocol]
        summary[protocol] = dict(original_case_references=sum(r['protocol'] == protocol for r in rows),
            unique_role_profiles=len(selected), proposed_helper_slots=sum(len(p['geometry']['folds']) if 'folds' in p['geometry']
                else p['geometry']['proposed_helper_slots'] for p in selected),
            original_main_roles_changed=False, neural_qualification='NOT_RUN')
    OUT.mkdir(parents=True, exist_ok=True)
    values = {'GEOMETRY_PROPOSAL.json': dict(status='METADATA_ONLY_NOT_TRAINING_LOCK', profiles=list(profiles.values()),
                    helper_initializations_not_yet_specified=True, training_authorized=False),
              'SOURCE_GEOMETRY_AUDIT.json': dict(status='THREE_PROTOCOL_SOURCE_SUPPORT_METADATA_FEASIBLE',
                    case_references=rows, summary=summary, new_models=0, new_forward=0, new_predictions=0,
                    target_evaluation=False, original_main_roles_changed=False, method_effect='NOT_RUN')}
    for name, value in values.items():
        with (OUT / name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2);stream.write('\n')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    receipt = dict(status='PASS_METADATA_ONLY_NOT_NEURAL_QUALIFICATION', at_unix=time.time(),
        input_sha256={p: sha(ROOT / p) for p in INPUTS},
        code_sha256={str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('audit_e0_train_only_design.py'))},
        output_sha256={name: sha(OUT / name) for name in values}, new_raw_reads=0,
        new_feature_or_probability_payload_reads=0, new_model_loads=0, new_training=0,
        new_forward=0, new_predictions=0, new_FFT=0, new_inverse_fits=0, training_authorized=False)
    with (OUT / 'AUDIT_RECEIPT.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2);stream.write('\n')
    print(json.dumps(dict(status='THREE_PROTOCOL_METADATA_PASS_NOT_NEURAL', summary=summary)))


if __name__ == '__main__':
    main()
