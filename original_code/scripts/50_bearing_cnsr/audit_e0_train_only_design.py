"""Metadata-only feasibility, not a new training entry or a grant."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_e0_train_only_design'
INPUTS = ('results/wfdg/ress_event_phm_main/DATA_PROPOSAL.json',
          'results/wfdg/ress_event_mechanism_factorial/DATA_MANIFEST.json',
          'results/wfdg/ress_e0_source_validity/TEACHER_MANIFEST.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_roles(case, records):
    sources = sorted(case['task']['sources'])
    assert len(sources) == len(set(sources)) == 2
    expected = {k for k, v in records.items() if v['domain'] in sources}
    assert set(case['roles']) == expected and len(expected) == 32
    train, validation = set(), set()
    for name, role in case['roles'].items():
        tr, va = role['tr'], role['va']
        assert bool(tr) != bool(va)
        chosen = tr if tr else va
        assert chosen == records[name]['offsets'] and len(chosen) == 99
        (train if tr else validation).add(name)
    assert len(train) == 26 and len(validation) == 6 and not train & validation
    assert {r['record'] for r in case['bank']} == train and len(case['bank']) == 26
    for item in case['bank']:
        assert len(item['starts']) == 4
        assert item['record'] in train and set(case['pairs'][item['pair']]) <= train
    return sources, train, validation


def balanced_folds(sources, train, records):
    """Use training labels only. Rare cells cross sources rather than vanish."""
    sources = sorted(sources)
    cells = {(d, y): sorted(k for k in train if records[k]['domain'] == d and records[k]['label'] == y)
             for d in sources for y in range(8)}
    patterns = {y: tuple(len(cells[d, y]) for d in sources) for y in range(8)}
    assert set(patterns.values()) <= {(1, 1), (2, 2)}
    rare = [y for y in range(8) if patterns[y] == (1, 1)]
    assert len(rare) == 3 and len(train) == 26
    held = [set(), set()]
    for y in range(8):
        for domain_index, domain in enumerate(sources):
            names = cells[domain, y]
            if len(names) == 1:
                held[(rare.index(y) + domain_index) % 2].add(names[0])
            else:
                for repeat_index, name in enumerate(names):
                    held[repeat_index].add(name)
    assert not held[0] & held[1] and held[0] | held[1] == set(train)
    folds = []
    for index in range(2):
        learning, probe = set(train) - held[index], held[index]
        assert len(learning) == len(probe) == 13 and not learning & probe
        assert {records[k]['label'] for k in learning} == {records[k]['label'] for k in probe} == set(range(8))
        domain_counts = Counter(records[k]['domain'] for k in learning)
        assert sorted(domain_counts.values()) == [6, 7]
        present = {(records[k]['domain'], records[k]['label']) for k in learning}
        missing = [(d, y) for d in sources for y in range(8) if (d, y) not in present]
        assert len(missing) == 3
        folds.append(dict(fold=index, train_records=sorted(learning), probe_records=sorted(probe),
                          training_windows=1287, probe_windows=1287,
                          training_source_record_counts=dict(domain_counts),
                          missing_training_source_class_cells=[dict(domain=d, label=y) for d, y in missing],
                          global_classes=list(range(8)), is_new_entity_split=False))
    return folds


def old_teacher_coverage(sources, train, validation, teachers):
    matches = [t for t in teachers if sorted(t['sources']) == sorted(sources)]
    legal = [t for t in matches if not set(t['train_records']) & validation]
    covered = set().union(*(set(t['probe_records']) & train for t in legal)) if legal else set()
    return dict(legal_old_teacher_count=len(legal), old_teacher_independent_train_record_coverage=len(covered),
                training_records_total=len(train), legal_teacher_keys=sorted(t['key'] for t in legal),
                illegal_teacher_validation_overlap_counts=sorted({len(set(t['train_records']) & validation) for t in matches}))


def main():
    data, old, teachers = [read(ROOT / p) for p in INPUTS]
    records = data['data']['records']
    cases = [v for v in old['cases'] if v['task']['protocol'] == 'phm09_w3'] + data['data']['cases']
    assert len(cases) == 60 and len(teachers) == 72
    profiles, audits = {}, []
    for case in cases:
        sources, train, validation = source_roles(case, records)
        ident = dict(sources=sources, train_records=sorted(train), validation_records=sorted(validation))
        key = canonical(ident)
        folds = balanced_folds(sources, train, records)
        assert all(not (set(f['train_records']) | set(f['probe_records'])) & validation for f in folds)
        if key not in profiles:
            profiles[key] = dict(profile=key, **ident, proposed_folds=folds, original_case_references=[])
        profiles[key]['original_case_references'].append(dict(fold=case['task']['fold'], seed=case['seed']))
        audits.append(dict(fold=case['task']['fold'], seed=case['seed'], profile=key,
                           original_source_train_recordings=len(train), original_source_validation_recordings=len(validation),
                           singleton_training_source_class_cells=6,
                           **old_teacher_coverage(sources, train, validation, teachers)))
    support = Counter((a['legal_old_teacher_count'], a['old_teacher_independent_train_record_coverage']) for a in audits)
    payload = dict(status='TRAIN_ONLY_CROSSFIT_METADATA_FEASIBLE_WITH_REDUCED_DOMAIN_CLASS_SUPPORT',
        neural_qualification='NOT_RUN', candidate_effect='NOT_RUN', training_authorized=False,
        case_audits=audits, source_profiles=len(profiles), proposed_fold_slots=2 * len(profiles),
        source_profiles_with_same_target_references_are_not_independent=True,
        old_teacher_coverage_counts=[dict(legal_teachers=a, independently_covered_original_train_recordings=b, cases=n)
                                   for (a, b), n in sorted(support.items())],
        unchanged_original_main_train_recordings_per_case=26, unchanged_original_main_validation_recordings_per_case=6,
        proposed_teacher_train_recordings=13, proposed_teacher_probe_recordings=13,
        proposed_teacher_training_global_classes=8, proposed_teacher_training_domain_class_cells=13,
        original_qualification_teacher_training_domain_class_cells=16,
        old_qualification_quality_does_not_transfer=True,
        new_models=0, new_model_loads=0, new_predictions=0, target_evaluation=False)
    proposal = dict(status='METADATA_ONLY_NOT_A_TRAINING_LOCK', profiles=list(profiles.values()),
        teachers_per_fold_not_approved=3, hypothetical_new_teachers_if_three_initializations=6 * len(profiles),
        sampling_requires_new_specification='Original sixteen nonempty domain-class sampler cannot be reused unchanged.',
        original_main_roles_changed=False, validation_labels_must_not_be_consumed_by_future_teachers=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('ROLE_AUDIT.json', payload), ('PROPOSED_CROSSFIT_MANIFEST.json', proposal)]:
        with (OUT / name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2);stream.write('\n')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    receipt = dict(status='METADATA_REVIEW_COMPLETE_NOT_NEURAL_PASS', at_unix=time.time(),
        code_sha256=sha(__file__), input_sha256={p: sha(ROOT / p) for p in INPUTS},
        output_sha256={name: sha(OUT / name) for name in ('ROLE_AUDIT.json', 'PROPOSED_CROSSFIT_MANIFEST.json')},
        new_raw_reads=0, new_cache_payload_reads=0, new_model_loads=0, new_training=0,
        new_forward=0, new_predictions=0, new_FFT=0, new_inverse_fits=0, target_evaluation=False,
        training_authorized=False)
    with (OUT / 'AUDIT_RECEIPT.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2);stream.write('\n')
    print(json.dumps({k: payload[k] for k in ('status', 'source_profiles', 'proposed_fold_slots', 'old_teacher_coverage_counts')}))


if __name__ == '__main__':
    main()
