"""Finite neural schedule compiler; metadata only, never imports Torch."""
from pathlib import Path
import importlib.metadata
import sqlite3
import sys
import source_rate_plan as p
from e0_selection_entry_spec import local_closure

LOCK = p.OUT / 'NEURAL_LOCK_V2.json'
P0_CASES = ('pu_clock4/BCDtoA/s0', 'pu_clock4/ABCtoD/s0',
            'phm09_w3/adj_AC_B/s0', 'jnu_speed/A/s0')
STEPS = (0, 1000, 2000)


def paths(store, name, kind, step):
    return str(Path(store) / 'neural' / name / f'{kind}_{step:04d}' / ('state.pt' if kind == 'checkpoint' else 'probabilities.npz'))


def expected(records, case, side):
    keys = sorted(case['roles']) if side != 'target' else sorted(k for k, r in records.items()
        if (r['protocol'], r['domain']) == (case['task']['protocol'], case['task']['target']))
    vectors = dict(y=[], g=[], offset=[], domain=[])
    for key in keys:
        record = records[key]
        offsets = record['offsets'] if side == 'target' else case['roles'][key][side]
        for offset in offsets:
            vectors['y'].append(record['label']); vectors['g'].append(key)
            vectors['offset'].append(offset); vectors['domain'].append(record['domain'])
    return vectors


def compile_neural():
    source_path = p.OUT / 'SOURCE_PREPARATION_LOCK_V1.json'
    bank_path = p.OUT / 'SOURCE_BANKS_V1.json'
    source, banks = p.read(source_path), p.read(bank_path)
    p.clock(source['campaign'])
    p.require(banks['status'] == 'FULL_SOURCE_RATE_BANKS_COMPLETE_NOT_CLASSIFICATION' and
        banks['fingerprint'] == source['fingerprint'], 'all22 complete current source banks first')
    source_db_path = p.OUT / 'SOURCE_CONTROL.sqlite'
    db = sqlite3.connect(source_db_path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        p.require(db.execute("SELECT v FROM meta WHERE k='DONE'").fetchone() is not None and
            db.execute("SELECT count(*) FROM units WHERE state='DONE'").fetchone()[0] == 2946 and
            db.execute("SELECT count(*) FROM units WHERE state!='DONE'").fetchone()[0] == 0,
            'actual complete source producer, not merely a prewritten result file')
    finally: db.close()
    original = p.parent.load_plan()
    cases = {c['key']: c for c in source['cases']}
    for key, case in cases.items():
        old = original['cases'][key]
        p.require(p.canonical(old) == case['original_case_sha256'], 'unchanged original case')
    registry_path = p.ROOT / 'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json'
    anchor_path = p.ROOT / 'results/wfdg/ress_e0_source_selection/ANCHOR_METADATA_V1.json'
    registry, anchors = p.read(registry_path), p.read(anchor_path)
    features = [x['binding'] for x in registry['caches'] if x['kind'] == 'full_feature']
    initial = {v['case_key']: v['initial_encoder_sha256'] for v in anchors['cases'] if v['case_key'] in cases}
    p.require(len(features) == 17 and set(initial) == set(cases), 'complete actual source-feature/initialization bindings')
    store = source['campaign']['store']; chains = {}; checkpoints = {}; predictions = {}; aliases = {}; vector_table = {}
    def add(name, key, arm, qualification=False, branch=None):
        case = cases[key]; old = original['cases'][key]
        chains[name] = dict(case_key=key, arm=arm, qualification=qualification, branch=branch,
            steps=5 if qualification else 2000)
        saved = (5,) if branch == 'continuous' else (2, 5) if qualification else tuple(range(0, 2001, 250))
        for step in saved:
            path = paths(store, name, 'checkpoint', step)
            checkpoints[path] = dict(chain=name, step=step)
        for step in ((0, 5) if qualification else STEPS):
            vectors = expected(original['records'], old, 'va')
            if qualification: vectors = {k: v[:32] for k, v in vectors.items()}
            vector_key = ('qualification_source/' if qualification else 'source/') + key
            vector_table[vector_key] = vectors
            path = paths(store, name, 'source', step)
            predictions[path] = dict(chain=name, case_key=key, side='source', step=step,
                qualification=qualification, expected_key=vector_key)
        if not qualification:
            vectors = expected(original['records'], old, 'target')
            vector_key = 'target/' + key; vector_table[vector_key] = vectors
            for policy in ('last', 'selected'):
                # Selected step is resolved source-only; the path encodes policy, not a guessed step.
                path = str(Path(store) / 'neural' / name / f'target_{policy}.npz')
                predictions[path] = dict(chain=name, case_key=key, side='target', policy=policy,
                    qualification=False, expected_key=vector_key)
    for key in P0_CASES:
        for arm in p.ARMS:
            for branch in ('continuous', 'resume'):
                add('P0/' + key + '/' + arm + '/' + branch, key, arm, True, branch)
    for key, case in sorted(cases.items()):
        for arm in case['real_arms']: add('MAIN/' + key + '/' + arm, key, arm)
        for arm, origin in case['aliases'].items(): aliases['MAIN/' + key + '/' + arm] = 'MAIN/' + key + '/' + origin
    real = {k: v for k, v in chains.items() if not v['qualification']}
    p.require(len(real) == 76 and len(aliases) == 12 and len(checkpoints) == 732, 'exact real/alias/checkpoint schedule')
    forward = sum(v['steps'] * 2 * (120 // (len(cases[v['case_key']]['task']['sources']) * cases[v['case_key']]['task']['classes'])) *
                  len(cases[v['case_key']]['task']['sources']) * cases[v['case_key']]['task']['classes'] for v in chains.values())
    source_n = sum(len(vector_table[v['expected_key']]['y']) for v in predictions.values() if v['side'] == 'source')
    target_n = sum(len(vector_table[v['expected_key']]['y']) for v in predictions.values() if v['side'] == 'target')
    limits = dict(source['campaign']['limits'], feature_decodes=17, feature_hashes=17,
        bank_decodes=22, bank_hashes=22, checkpoint_writes=732, checkpoint_reads=260,
        checkpoint_hashes=1700, probability_writes=len(predictions), probability_reads=2 * len(predictions),
        probability_hashes=4 * len(predictions))
    p.require(source_n <= limits['source_prediction_windows'] and target_n <= limits['target_prediction_windows'] and
              forward <= limits['training_forward_windows'], 'complete actual prediction/forward forecast fits campaign')
    entries = [Path(__file__), Path(__file__).with_name('source_rate_screen_runtime.py'),
        Path(__file__).with_name('source_rate_screen_data.py'), Path(__file__).with_name('source_rate_neural.py'),
        Path(__file__).with_name('run_source_rate_screen.py'), Path(__file__).with_name('test_source_rate_screen.py')]
    tests_path = p.OUT / 'NEURAL_CPU_TESTS_V3.json'; tests = p.read(tests_path)
    p.require(tests['status'] == 'PASS' and tests['tests'] == 11 and tests['torch_imported'] is False,
        'actual complete zero-neural CPU regression suite')
    for name, digest in tests['code_sha256'].items():
        p.require(p.sha(Path(__file__).with_name(name)) == digest, 'exact tested neural entry bytes')
    inputs = dict(source['input_sha256'], **{str(path): p.sha(path) for path in
        (source_path, bank_path, source_db_path, registry_path, anchor_path, tests_path,
         p.OUT / 'NEURAL_EXECUTION_SPEC_V1.md',
         p.ROOT / 'results/wfdg/ress_protected_response_classification/STANDING_AUTHORIZATION_V1.md')})
    lock = dict(kind='SOURCE_RATE_NEURAL_SCREEN_V1', campaign=source['campaign'],
        source_fingerprint=source['fingerprint'], source_bank_result_sha256=p.sha(bank_path),
        cases=cases, original_cases={k: original['cases'][k] for k in cases}, records=original['records'],
        banks=banks['banks'], bank_aliases=banks['aliases'], feature_bindings=features,
        expected_FFT_code_sha256=registry['expected_FFT_code_sha256'],
        initial_encoder_sha256=initial, chains=chains, aliases=aliases, checkpoints=checkpoints,
        predictions=predictions, vectors=vector_table, limits=limits, input_sha256=inputs, code_sha256=local_closure(entries),
        forecast=dict(real_models=76, logical_models=88, formal_updates=152000, qualification_updates=160,
            training_forward_windows=forward, source_prediction_windows=source_n, target_prediction_windows=target_n,
            checkpoints=len(checkpoints), predictions=len(predictions), planned_model_builds=352),
        gate=dict(primary='last', secondary='selected_no_rescue', effect_acc=.003,
            effect_other_metrics_floor=0., protocol_acc_floor=-.002, both_seed_acc_strict_positive=True,
            specificity_acc=.001, specificity_both_seed_nonnegative=True,
            specificity_varied_PU_or_PHM_positive=True, epsilon=1e-12),
        source_selection='mean per-source-domain validation accuracy; steps0/1000/2000; earliest strict best including0',
        neural_policy=dict(AMP=False, scheduler=None, optimizer='Adam', lr=1e-4,
            betas=[.9,.999], eps=1e-8, weight_decay=0, foreach=False,
            train_batch='2*D*Y*floor(120/(D*Y)); full factual + matched auxiliary Mixup',
            checkpoint_every=250, actual_disk_restore_step=1000),
        environment=dict(executable=str(Path(sys.executable).resolve()), python=sys.version,
            numpy=importlib.metadata.version('numpy'), torch=importlib.metadata.version('torch')),
        execution_ready=True, new_raw_reads=0, new_inverse_fits=0, protected_reads=0)
    lock['fingerprint'] = p.canonical(lock)
    return lock


def verify(lock):
    p.require(lock['fingerprint'] == p.canonical({k: v for k, v in lock.items() if k != 'fingerprint'}), 'neural scientific lock fingerprint')
    for kind in ('code_sha256', 'input_sha256'):
        for path, digest in lock[kind].items(): p.require(p.sha(path) == digest, 'locked binding drift: ' + path)
    p.require(lock['execution_ready'] and lock['forecast']['formal_updates'] == 152000, 'fixed complete scope')
    p.clock(lock['campaign'])


if __name__ == '__main__':
    result = compile_neural(); p.save(LOCK, result)
    print(__import__('json').dumps(dict(fingerprint=result['fingerprint'], forecast=result['forecast'], limits=result['limits']), indent=2))
