"""Metadata-only source condition audit. Never import a training/data adapter.

No raw files, arrays, weights, old SQLite or target scores are opened. This
checks whether existing E0 factors equal actual *source-to-source* clock ratios;
it does not estimate how realistic any generated signal is.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_e0_natural_condition_design'
REGISTRY = 'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json'
OLD_FACTORS = (Fraction(2, 3), Fraction(3, 2))
PROTECTED = {'K006', 'KA08', 'KA09'}
# Explicit documented physical settings, not inferred bearing attributes.
PU = {'A': ('N15_M01_F10', Fraction(25), '0.1', '1000'),
      'B': ('N15_M07_F04', Fraction(25), '0.7', '400'),
      'C': ('N15_M07_F10', Fraction(25), '0.7', '1000'),
      'D': ('N09_M07_F10', Fraction(15), '0.7', '1000')}


def insist(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve(document, reference):
    value = document
    for field in reference['field']:
        value = value[field]
    if 'index' in reference:
        value = value[reference['index']]
    return value


def fraction_text(value):
    return f'{value.numerator}/{value.denominator}'


def csv_conditions(text, protocol):
    result = {}
    for row in csv.DictReader(io.StringIO(text.lstrip('\ufeff'))):
        domain = row['domain']
        if protocol == 'phm09_w3':
            value = {'clock': Fraction(row['hz']), 'load': row['load']}
        elif protocol == 'jnu_speed':
            value = {'clock': Fraction(row['rpm']) / 60,
                     'load': 'NOT_RECORDED_HERE'}
        else:
            raise ValueError('known protocol')
        insist(domain not in result or result[domain] == value,
               'inconsistent domain condition metadata')
        result[domain] = value
    return result


def condition(protocol, domain, record, tables):
    insist(record['protocol'] == protocol and record['domain'] == domain,
           'record belongs to the declared source condition')
    if protocol == 'pu_clock4':
        prefix, clock, torque, force = PU[domain]
        name = Path(record['path']).name
        # Accept Windows paths when run on a non-Windows host, too.
        name = name.replace('\\', '/').split('/')[-1]
        insist(name.startswith(prefix + '_'), 'explicit PU physical condition')
        value = {'clock': clock, 'load': (torque, force)}
    else:
        value = tables[protocol][domain]
    insist(math.isfinite(record['shaft_hz']) and abs(float(value['clock']) -
        record['shaft_hz']) <= 1e-10, 'manifest/source-provenance shaft clock')
    return value


def compare_conditions(protocol, left, right):
    ratio = right['clock'] / left['clock']
    if ratio == 1:
        kind = 'SAME_CLOCK_NONCLOCK_CONDITION_CHANGE'
    elif protocol == 'jnu_speed':
        kind = 'SPEED_CHANGE_OTHER_FACTORS_NOT_VERIFIED'
    elif left['load'] == right['load']:
        kind = 'RECORDED_SPEED_CHANGE_FIXED_RECORDED_LOAD'
    else:
        kind = 'SPEED_AND_RECORDED_LOAD_CHANGE'
    return dict(ratio=fraction_text(ratio), speed_changes=ratio != 1,
        transition_kind=kind, old_factor_exact_match=ratio in OLD_FACTORS,
        render_denominator_supported=ratio.denominator <= 12)


def source_support(case, records, tables):
    task, roles = case['task'], case['roles']
    protocol, sources = task['protocol'], task['sources']
    insist(task['target'] not in sources and len(sources) == len(set(sources)),
           'source/target disjoint and source domains unique')
    grouped = defaultdict(list)
    settings = {}
    for key, role in roles.items():
        record = records[key]
        insist(record['domain'] in sources and record['protocol'] == protocol,
               'no current-task target role admitted')
        insist(record.get('bearing') not in PROTECTED, 'no reserved bearing role')
        insist(set(role['tr']).isdisjoint(role['va']), 'disjoint window roles')
        if not role['tr'] and not role['va']:
            continue
        value = condition(protocol, record['domain'], record, tables)
        insist(record['domain'] not in settings or settings[record['domain']] == value,
               'one physical clock/load per domain')
        settings[record['domain']] = value
        grouped[(record['domain'], record['label'])].append((key, record, role))
    insist(set(settings) == set(sources), 'all source conditions have support')
    transitions = []
    for source in sources:
        for destination in sources:
            if source == destination:
                continue
            detail = compare_conditions(protocol, settings[source], settings[destination])
            for label in range(task['classes']):
                donors = [v for v in grouped[source, label] if v[2]['tr']]
                refs = [v for v in grouped[destination, label] if v[2]['va']]
                insist(donors, 'all source training classes retained')
                # JNU tr/va intervals belong to the same long file. Count this
                # support honestly, never turn it into independent recordings.
                if protocol == 'pu_clock4':
                    donor_ids = {v[1]['bearing'] for v in donors}
                    ref_ids = {v[1]['bearing'] for v in refs}
                    common = donor_ids & ref_ids
                    support_kind = 'SAME_BEARING_DIFFERENT_RECORDINGS'
                elif protocol == 'phm09_w3':
                    donor_ids = {v[1]['configuration'] for v in donors}
                    ref_ids = {v[1]['configuration'] for v in refs}
                    common = donor_ids & ref_ids
                    support_kind = 'SAME_CONFIGURATION_ENTITY_UNKNOWN'
                else:
                    common = {str(label)} if refs else set()
                    support_kind = 'CLASS_MATCHED_WITHIN_FILE_TIME_BLOCKS_ENTITY_UNKNOWN'
                transitions.append(dict(protocol=protocol, fold=task['fold'],
                    seed=case['seed'], source=source, destination=destination,
                    label=label, **detail, support_kind=support_kind,
                    donor_training_records=len(donors),
                    reference_validation_record_roles=len(refs),
                    reference_validation_windows=sum(len(v[2]['va']) for v in refs),
                    matched_recorded_identity_groups=len(common),
                    complete_reference_available=bool(common)))
    return transitions


def summarize(rows, cases):
    cases_by_key = {(v['task']['protocol'], v['task']['fold'], v['seed']): v for v in cases}
    by_case = defaultdict(list)
    for row in rows:
        by_case[row['protocol'], row['fold'], row['seed']].append(row)
    insist(set(by_case) == set(cases_by_key), 'all original task/seed cases covered')
    case_results = []
    for key, values in sorted(by_case.items()):
        speed = [v for v in values if v['speed_changes']]
        fixed_load = [v for v in speed if
            v['transition_kind'] == 'RECORDED_SPEED_CHANGE_FIXED_RECORDED_LOAD']
        case_results.append(dict(protocol=key[0], fold=key[1], seed=key[2],
            speed_transition_exists=bool(speed),
            fixed_recorded_load_speed_transition_exists=bool(fixed_load),
            transition_class_roles=len(values),
            speed_transition_class_roles=len(speed),
            all_speed_classes_have_original_val_reference=bool(speed) and all(
                v['complete_reference_available'] for v in speed),
            old_factor_exact_speed_roles=sum(v['old_factor_exact_match'] for v in speed),
            speed_roles_without_original_val_reference=sum(
                not v['complete_reference_available'] for v in speed)))
    protocol = {}
    for name in sorted({v['protocol'] for v in case_results}):
        selected = [v for v in case_results if v['protocol'] == name]
        protocol[name] = dict(cases=len(selected),
            tasks=len({v['fold'] for v in selected}),
            tasks_with_source_speed_transition=len({v['fold'] for v in selected
                if v['speed_transition_exists']}),
            tasks_with_fixed_recorded_load_speed_transition=len({v['fold'] for v in selected
                if v['fixed_recorded_load_speed_transition_exists']}),
            cases_with_all_original_val_speed_references=sum(
                v['all_speed_classes_have_original_val_reference'] for v in selected),
            speed_transition_class_roles=sum(v['speed_transition_class_roles'] for v in selected),
            speed_roles_without_original_val_reference=sum(
                v['speed_roles_without_original_val_reference'] for v in selected),
            exact_old_factor_speed_roles=sum(v['old_factor_exact_speed_roles'] for v in selected))
    return dict(protocols=protocol, cases=case_results,
        speed_ratios=sorted({v['ratio'] for v in rows if v['speed_changes']}),
        all_speed_ratios_fit_existing_max_denominator=all(
            v['render_denominator_supported'] for v in rows if v['speed_changes']),
        old_factors=[fraction_text(v) for v in OLD_FACTORS])


def run():
    scope = json.loads((OUT / 'SCOPE_V1.json').read_text(encoding='utf8'))
    insist(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']),
           'original research deadline; no clock restart')
    allowed = set(scope['metadata_input_allowlist'])
    documents, bindings, used = {}, [], 0
    for name in sorted(allowed):
        path = ROOT / name
        raw = path.read_bytes()
        used += len(raw)
        insist(used <= scope['metadata_bytes_per_pass_max'], 'finite metadata byte budget')
        bindings.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        text = raw.decode('utf-8-sig')
        documents[name] = json.loads(text) if name.endswith('.json') else text
    registry = documents[REGISTRY]
    insist(canonical({k: v for k, v in registry.items() if k != 'fingerprint'}) ==
           registry['fingerprint'], 'historical registry fingerprint')
    reference = registry['record_reference']
    records = resolve(documents[reference['path']], reference)
    tables = {p: csv_conditions(documents['data/processed/' + p + '/provenance.csv'], p)
              for p in ('phm09_w3', 'jnu_speed')}
    cases, rows = [], []
    for item in registry['cases']:
        ref = item['case_reference']
        case = resolve(documents[ref['path']], ref)
        insist(canonical(case) == item['case_sha256'] and
               canonical(case['roles']) == item['original_roles_sha256'],
               'unchanged case and original source-role fingerprint')
        insist(case['task'] == item['task'] and case['seed'] == item['seed'],
               'original case identity')
        cases.append(case)
        rows.extend(source_support(case, records, tables))
    result = dict(status='SOURCE_NATURAL_CONDITION_SUPPORT_ENUMERATED_NOT_VALIDATED',
        scope_sha256=hashlib.sha256((OUT / 'SCOPE_V1.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bindings=bindings, metadata_bytes=used,
        original_cases=len(cases), original_tasks=len({(c['task']['protocol'], c['task']['fold'])
                                                     for c in cases}),
        summary=summarize(rows, cases), rows=rows,
        new_scientific_array_reads=0, new_raw_reads=0, new_NN=0,
        new_FFT=0, new_fitting_steps=0, new_predictions=0,
        generated_at=datetime.now().astimezone().isoformat())
    insist(result['original_cases'] == 81 and result['original_tasks'] == 27,
           'full original 27-task/three-seed metadata coverage')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='exclusive new metadata report only')
    args = parser.parse_args()
    result = run()
    if args.write:
        with (OUT / 'SOURCE_SUPPORT_V1.json').open('x', encoding='utf8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'bindings')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
