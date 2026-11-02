"""Metadata-only binding for the five-arm compatible-set pilot."""
from datetime import datetime
from pathlib import Path
import source_rate_plan as old

ROOT = old.ROOT
OUT = ROOT / 'results/wfdg/ress_compatible_set'
ARMS = ('A0_STRONG_IN_MIXUP', 'A1_MATCHED_E0', 'A2_COMPATIBLE_MEAN',
        'C0_COMPATIBLE_WORST', 'N0_GRAM_MATCHED_RANDOM_WORST')
read, save, sha, canonical, require = old.read, old.save, old.sha, old.canonical, old.require
CODE = ('compatible_set_plan.py', 'compatible_set_core.py', 'compatible_set_source.py',
        'test_compatible_set.py', 'abduction_family_core.py', 'run_abduction_action.py',
        'abduction_action_plan.py', 'clock_real_core.py', 'clock_real_data.py',
        'natural_mapping_probe_plan.py', 'source_rate_plan.py')


def clock(scope):
    require(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']), 'original campaign deadline')


def verify(lock):
    clock(lock['campaign'])
    require(lock['fingerprint'] == canonical({k: v for k, v in lock.items() if k != 'fingerprint'}), 'complete immutable lock')
    for path, digest in {**lock['input_sha256'], **lock['code_sha256']}.items():
        require(sha(path) == digest, 'changed locked input/code: ' + path)


def source_lock():
    from natural_mapping_probe_plan import source_case
    campaign = read(OUT / 'CAMPAIGN_SCOPE_V1.json'); clock(campaign)
    parent = ROOT / 'results/wfdg/ress_source_rate_screen'
    paths = [parent / name for name in ('SOURCE_PREPARATION_LOCK_V1.json', 'SOURCE_BANKS_V1.json', 'NEURAL_LOCK_V2.json')]
    prep, banks, neural = map(read, paths)
    receipt = read(parent / 'CLOSED_RECEIPT_V1.json')
    require(receipt['files_sha256']['NEURAL_LOCK_V2.json'] == sha(paths[2]) and
            receipt['files_sha256']['SOURCE_BANKS_V1.json'] == sha(paths[1]), 'closed parent metadata and source banks')
    require(banks['fingerprint'] == prep['fingerprint'], 'qualified source preparation lineage')
    wanted = {tuple(t) for t in campaign['tasks']}
    cases = [dict(c, real_arms=list(ARMS), aliases={}) for c in prep['cases']
             if (c['task']['protocol'], c['task']['fold']) in wanted and c['seed'] in campaign['seeds']]
    require(len(cases) == 10 and len({c['key'] for c in cases}) == 10, 'ten fixed source cases')
    keys = {v['group'] for c in cases for v in c['bank']}
    groups = [g for g in prep['groups'] if g['key'] in keys]
    by_group = {g['key']: g for g in groups}
    for c in cases:
        original = neural['original_cases'][c['key']]
        require(canonical(original) == c['original_case_sha256'] and canonical(original['roles']) == c['original_roles_sha256'], 'same original source roles')
        for item in c['bank']:
            g = by_group[item['group']]
            for record in g['pair']: source_case(original, neural['records'], record, context=True)
            require(item['record'] == g['origin'] and item['domain'] in c['task']['sources'], 'current-case legal source bank')
            require(prep['assets'][g['inverse_identity']]['inverse']['pair'] == g['pair'], 'inverse belongs to bound source pair')
    identities = {g['inverse_identity'] for g in groups}
    assets = {k: prep['assets'][k] for k in sorted(identities)}
    anchors = {g['key']: banks['groups'][g['key']] for g in groups}
    bindings = [v for a in assets.values() for v in a.values()] + list(anchors.values())
    sizes = {str(Path(b['path']).resolve()): Path(b['path']).stat().st_size for b in bindings}
    n = len(groups)
    forecast = dict(cases=10, groups=n, inverse_pairs=len(assets), source_payload_decodes=len(bindings),
        source_payload_bytes=sum(sizes.values()), source_fft_1d_transforms_upper=1397*n,
        source_fft_points_upper=sum(1376*65536 + 21*g['window'] for g in groups),
        bank_window_roles=sum(c['bank_windows'] for c in cases))
    for field in ('source_payload_decodes', 'source_payload_bytes'):
        require(forecast[field] <= campaign['limits'][field], 'source resource fits campaign: ' + field)
    require(forecast['source_fft_1d_transforms_upper'] + 20000 <= campaign['limits']['source_fft_1d_transforms'] and
        forecast['source_fft_points_upper'] + 20000000 <= campaign['limits']['source_fft_points'], 'source FFT envelope including fixtures')
    lock = dict(kind='COMPATIBLE_SET_FULL_SOURCE_V1', campaign=campaign, cases=cases,
        original_cases={c['key']: neural['original_cases'][c['key']] for c in cases}, records=neural['records'],
        groups=groups, assets=assets, anchors=anchors, payload_sizes=sizes, forecast=forecast,
        limits=dict(campaign['limits'], owned_writes=n+10, owned_reads=n+10, owned_hashes=2*(n+10)),
        coefficient_l1_fraction=.01, ratios=['2/3', '3/2'], selection='joint_twoq_threequery_time_squared_norm_first_lexicographic',
        random_identity='COMPATIBLE_SET_GRAM_V1', gram_tolerance=dict(atol=5e-6, rtol=1e-4),
        input_sha256={str(v): sha(v) for v in paths + [parent/'CLOSED_RECEIPT_V1.json', OUT/'CAMPAIGN_SCOPE_V1.json', ROOT/'results/wfdg/ress_abduction_family_bound/NEXT_CLASSIFIER_DESIGN.md']},
        code_sha256={str(ROOT/'scripts/50_bearing_cnsr'/v): sha(ROOT/'scripts/50_bearing_cnsr'/v) for v in CODE})
    lock['fingerprint'] = canonical(lock); save(OUT/'SOURCE_LOCK_V2.json', lock)
    return lock


if __name__ == '__main__':
    import json
    value = source_lock(); print(json.dumps(dict(fingerprint=value['fingerprint'], forecast=value['forecast']), indent=2))
