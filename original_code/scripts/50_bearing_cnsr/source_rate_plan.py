"""Compile the whole fixed source-rate screen; stdlib metadata only."""
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sqlite3
import protected_response_run_spec as parent
from natural_mapping_probe_plan import source_case
from design_e0_natural_source_probe_v2 import centered_three, valid_support

ROOT = parent.ROOT
OUT = ROOT / 'results/wfdg/ress_source_rate_screen'
ARMS = ('F0_MATCHED_FACTUAL_BANK', 'E0_MATCHED_FIXED_RATE',
        'C0_SOURCE_COUPLED_RATE', 'N0_SOURCE_RING_RATE')
SOURCE_CODE = ('source_rate_plan.py', 'source_rate_core.py', 'source_rate_source_run.py',
    'test_source_rate_core.py', 'natural_mapping_probe_core.py',
    'natural_mapping_probe_plan.py', 'design_e0_natural_source_probe_v2.py',
    'check_e0_natural_source_support.py', 'protected_response_run_spec.py',
    'protected_response_payloads.py', 'inventory_protected_response.py',
    'clock_real_core.py', 'clock_real_data.py', 'protected_response_journal.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def clock(scope):
    require(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']),
            'original8h campaign deadline, not reset')


def choose_cases(cases):
    folds = {'pu_clock4': None, 'jnu_speed': None,
             'phm09_w3': {'adj_AC_B', 'adj_HJ_I', 'ext_AB_F', 'ext_IJ_E'}}
    selected = {key: value for key, value in cases.items()
        if value['seed'] in (0, 1) and value['task']['protocol'] in folds and
        (folds[value['task']['protocol']] is None or value['task']['fold'] in folds[value['task']['protocol']])}
    require(len(selected) == 22, 'all original11 tasks and two seeds')
    return selected


def rate_definitions(case, records):
    sources = case['task']['sources']
    clocks = {d: {records[k]['shaft_hz'] for k, role in case['roles'].items()
                    if records[k]['domain'] == d and role['tr']} for d in sources}
    require(all(len(v) == 1 for v in clocks.values()), 'one physical clock per source domain')
    diverse = len(set(next(iter(v)) for v in clocks.values())) > 1
    require(len(set(next(iter(v)) for v in clocks.values())) <= 2, 'locked at most two source clock groups')
    rates = {}
    if diverse:
        for item in case['bank']:
            origin = item['record']; pair = case['pairs'][item['pair']]
            other = next(k for k in pair if k != origin)
            ratio = records[other]['shaft_hz'] / records[origin]['shaft_hz']
            q = Fraction(ratio).limit_denominator(12)
            require(q != 1 and abs(float(q) - ratio) <= 1e-12, 'qualified nonidentity rational source partner clock')
            domain = records[origin]['domain']; text = str(q)
            require(domain not in rates or rates[domain] == text, 'q_i fixed within source domain for every class and record')
            rates[domain] = text
        require(set(rates) == set(sources), 'all source domain q values exist')
    return diverse, rates, {d: rates[sources[(i + 1) % len(sources)]]
                            for i, d in enumerate(sources)} if diverse else {}


def prepare_source():
    scope_path = OUT / 'CAMPAIGN_SCOPE_V1.json'
    campaign = read(scope_path); clock(campaign)
    data = parent.load_plan(); cases = choose_cases(data['cases']); records = data['records']
    old_lock_path = parent.OUT / 'SOURCE_RUN_LOCK_V1.json'
    old_final_path = parent.OUT / 'SOURCE_QUALIFICATION_FINAL_V1.json'
    old_lock, final = read(old_lock_path), read(old_final_path)
    require(sha(final['database']['path']) == final['database']['sha256'], 'closed complete source qualification DB unchanged')
    db = sqlite3.connect(Path(final['database']['path']).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        require(db.execute("SELECT value FROM meta WHERE key='DONE'").fetchone(), 'source qualification already closed')
        receipts = {k: json.loads(v) for k, v in db.execute("SELECT key,detail FROM units WHERE status='DONE'")}
    finally:
        db.close()
    groups, assets, compiled = {}, {}, []
    for key, case in sorted(cases.items()):
        diverse, rates, ring = rate_definitions(case, records)
        task = case['task']; starts = centered_three(32768, task['window']); bank = []
        for item in case['bank']:
            origin, pair_key = item['record'], item['pair']
            require(pair_key in case['pairs'], 'every full original bank origin has a bound inverse')
            pair = case['pairs'][pair_key]
            for record in pair:
                source_case(case, records, record, context=True)
            identity = data['case_inverses'][key][pair_key]
            inverse = data['inverses'][identity]['binding']
            require(inverse['pair'] == pair, 'actual inverse version and source pair')
            receipt = receipts['I/' + identity]
            require(receipt['scope_fingerprint'] == old_lock['scope_fingerprint'], 'old physical-unit context packet lineage')
            if identity not in assets:
                assets[identity] = dict(inverse=inverse, qualified_context=receipt['packet'])
            source = pair.index(origin); domain = records[origin]['domain']
            group = canonical(dict(inverse_identity=identity, source=source))
            if group not in groups:
                groups[group] = dict(key=group, inverse_identity=identity, source=source,
                    origin=origin, pair=pair, witness_case=key,
                    fs=records[origin]['fs'], kernel_length=__import__('math').ceil(.004 * records[origin]['fs']),
                    window=task['window'], starts=starts, ratios=set())
            g = groups[group]
            require(g['origin'] == origin and g['starts'] == starts, 'same immutable group geometry')
            values = {'2/3', '3/2'} | ({rates[domain], ring[domain]} if diverse else set())
            for text in values:
                require(valid_support(32768, g['kernel_length'], Fraction(text), starts, task['window']), 'all full-source new query support')
            g['ratios'].update(values)
            bank.append(dict(record=origin, domain=domain, label=records[origin]['label'],
                group=group, correct_ratio=rates.get(domain), ring_ratio=ring.get(domain)))
        compiled.append(dict(key=key, task=task, seed=case['seed'], original_case_sha256=canonical(case),
            original_roles_sha256=canonical(case['roles']), source_speed_diversity=diverse,
            rates=rates, ring_rates=ring, source_training_windows=case['training_windows'],
            source_validation_windows=case['validation_windows'], bank=bank,
            bank_windows=3 * len(bank), real_arms=list(ARMS if diverse else ARMS[:2]),
            aliases={} if diverse else {ARMS[2]: ARMS[1], ARMS[3]: ARMS[1]}))
    for g in groups.values():
        g['ratios'] = sorted(g['ratios'], key=Fraction)
    groups = sorted(groups.values(), key=lambda g: (g['inverse_identity'], g['source']))
    models = sum(len(c['real_arms']) for c in compiled)
    require(models == 76 and sum(len(c['aliases']) for c in compiled) == 12, '76 real models,12 exact logical aliases')
    renders = sum(len(g['ratios']) for g in groups)
    logical_fft = sum(sum(96 * Fraction(q).numerator + 1 for q in g['ratios']) +
                      3 * (len(g['ratios']) + 1) for g in groups)
    fft_points = sum(sum((96 * Fraction(q).numerator + 1) * 65536 for q in g['ratios']) +
                     3 * (len(g['ratios']) + 1) * g['window'] for g in groups)
    fft_calls = sum(sum(2 * Fraction(q).numerator + 1 for q in g['ratios']) + len(g['ratios']) + 1 for g in groups)
    encoded = sum(3 * (len(g['ratios']) + 1) for g in groups)
    # Actual bound asset paths, no array/header/hash read yet.
    for identity, value in assets.items():
        for binding in value.values():
            require(Path(binding['path']).drive.upper() == 'D:' and Path(binding['path']).is_file(), 'existing D source payload')
    limits = dict(source_payload_decodes=2 * len(assets), source_payload_hashes=2 * len(assets),
        context_renders=renders, fft_calls=fft_calls + 200,
        fft_1d_transforms=logical_fft + 2000, fft_points=fft_points + 100000000,
        encoded_windows=encoded + 100, group_writes=len(groups), bank_writes=22,
        group_reads=len(groups), bank_audit_reads=22,
        owned_payload_decodes=len(groups) + 22, owned_hash_checks=2 * (len(groups) + 22),
        neural_models=0, optimizer_updates=0, target_predictions=0, new_raw_reads=0, new_inverse_fits=0)
    path = ROOT / 'results/wfdg/ress_event_pair_objective/SCREEN_RESULT_V1.json'
    old = read(path)
    keys = {(c['task']['protocol'], c['task']['fold'], c['seed']) for c in compiled}
    anchors = [v for v in old['rows'] if v['arm'] in ('E0_EVENTS', 'B1_IN_MIX') and
               (v['protocol'], v['fold'], v['seed']) in keys]
    require(len(anchors) == 88, 'complete unchanged old E0/IN two-policy slice')
    anchor_out = dict(parent_path=str(path), parent_sha256=sha(path), rows=anchors)
    save(OUT / 'ARCHIVED_ANCHORS_V1.json', anchor_out)
    inputs = {str(scope_path): sha(scope_path), str(old_lock_path): sha(old_lock_path),
        str(old_final_path): sha(old_final_path), **data['input_sha256'],
        str(OUT / 'PRETRAINING_AMENDMENT_V1.md'): sha(OUT / 'PRETRAINING_AMENDMENT_V1.md'),
        str(OUT / 'ARCHIVED_ANCHORS_V1.json'): sha(OUT / 'ARCHIVED_ANCHORS_V1.json')}
    lock = dict(kind='SOURCE_RATE_FULL_BANK_PREPARATION_V1', campaign=campaign,
        cases=compiled, groups=groups, assets=assets, limits=limits,
        input_sha256=inputs,
        code_sha256={str(ROOT / 'scripts/50_bearing_cnsr' / name): sha(ROOT / 'scripts/50_bearing_cnsr' / name) for name in SOURCE_CODE},
        original_case_metadata_refs={c['key']: next(v['original_case_reference'] for v in data['graph']['cases'] if v['key'] == c['key']) for c in compiled},
        forecast=dict(real_models=models, source_cases=22, source_groups=len(groups), old_inverse_packets=len(assets),
            render_calls=renders, fft_1d_transforms=logical_fft, fft_points=fft_points,
            encoded_windows=encoded, bank_window_roles=sum(c['bank_windows'] for c in compiled)),
        parent_source_database_sha256=final['database']['sha256'], execution_ready=True,
        numerical_seconds=18000)
    lock['fingerprint'] = canonical(lock)
    save(OUT / 'SOURCE_PREPARATION_LOCK_V1.json', lock)
    return lock


if __name__ == '__main__':
    value = prepare_source()
    print(json.dumps(dict(fingerprint=value['fingerprint'], forecast=value['forecast'], limits=value['limits']), indent=2))
