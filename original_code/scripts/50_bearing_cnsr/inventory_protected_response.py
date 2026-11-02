"""Source-only preparation inventory: metadata/code/readonly SQL, no payloads.

Never import a training module, NumPy or Torch; do not open ZIP members, raw
signals, inverse arrays, feature arrays, probabilities or model packets.
Declared writer schemas are NOT a new per-file schema/numeric audit.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/wfdg/ress_protected_response_preparation'
SPAN = (65536, 98304)
DATABASE = ROOT / 'results/wfdg/ress_event_phm_main/control/CONTROL.sqlite'
PATHS = {
    'registry': 'results/wfdg/ress_e0_source_selection/INPUT_REGISTRY_V1.json',
    'old_lock': 'results/wfdg/ress_e0_source_selection/RUN_LOCK_V1.json',
    'core_cases': 'results/wfdg/ress_event_mechanism_factorial/DATA_MANIFEST.json',
    'phm_cases': 'results/wfdg/ress_event_phm_main/DATA_PROPOSAL.json',
    'old_inventory': 'results/wfdg/ress_event_validation_inventory/METHOD_IDENTITY_AND_COVERAGE.json',
    'contexts': 'results/wfdg/ress_calibrated_event_pilot/SOURCE_CONTEXTS.json',
}
WRITERS = (
    'scripts/50_bearing_cnsr/event_phm_main_source.py',
    'scripts/50_bearing_cnsr/event_mechanism_source.py',
    'scripts/50_bearing_cnsr/calibrated_event_source.py',
    'scripts/50_bearing_cnsr/clock_real_core.py',
    'scripts/50_bearing_cnsr/stable_channel_core.py',
)


def insist(value, message):
    if not value:
        raise AssertionError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    path = Path(path)
    if path.suffix.lower() not in ('.json', '.py', '.md', '.sqlite'):
        raise PermissionError('metadata/code SHA only, never scientific payloads')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if path.suffix.lower() != '.json':
        raise PermissionError('JSON reader only')
    return json.loads(path.read_text(encoding='utf-8'))


def intervals(offsets, window):
    insist(type(window) is int and window > 0, 'positive integer source window')
    insist(all(type(x) is int and x >= 0 for x in offsets) and len(set(offsets)) == len(offsets), 'unique nonnegative integer source offsets')
    result = []
    for start in sorted(offsets):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], start + window)
        else:
            result.append([start, start + window])
    return result


def covered(offsets, window, start=SPAN[0], stop=SPAN[1]):
    return any(a <= start < stop <= b for a, b in intervals(offsets, window))


def legal_record(case, records, record):
    if record not in records or record not in case['roles']:
        return False
    r, role = records[record], case['roles'][record]
    if r['protocol'] != case['task']['protocol'] or r['domain'] not in case['task']['sources']:
        return False
    return covered(role['tr'], r['window']) and not any(a < SPAN[1] and b > SPAN[0] for a, b in intervals(role['va'], r['window']))


def legal_measurement_pair(case, records, pair):
    # A stored two-row .npy member is loaded as a whole. A legal wanted row
    # does NOT authorize opening a parent member containing this case's target.
    return len(pair) == 2 and all(legal_record(case, records, r) for r in pair)


def source_key(case):
    return case['task']['protocol'] + '/' + case['task']['fold'] + '/s' + str(case['seed'])


def merge_record(left, right):
    # Libraries may carry extra provenance columns. Do not mistake absence of
    # a column for a physical discrepancy; conflicting shared values still fail.
    shared = left.keys() & right.keys()
    insist(all(left[k] == right[k] or k == 'expected_samples' and (left[k] is None or right[k] is None) for k in shared), 'same shared physical record metadata')
    merged = dict(left, **right)
    if right.get('expected_samples') is None and left.get('expected_samples') is not None:
        merged['expected_samples'] = left['expected_samples']
    known_lengths = [merged.get(k) for k in ('expected_samples', 'provenance_samples') if merged.get(k) is not None]
    insist(len(set(known_lengths)) <= 1, 'source-length provenance agrees with newly known length')
    return merged


def journal():
    query = "SELECT name,value FROM journal WHERE name GLOB 'INVERSE/*' OR name GLOB 'RAW/*' ORDER BY name"
    with sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        rows = [(name, json.loads(value)) for name, value in db.execute(query)]
    return dict(rows)


class Inventory:
    def __init__(self):
        self.inputs, self.presence, self.stats = {}, {}, collections.Counter()

    def load(self, relative):
        path = ROOT / relative
        self.inputs[str(path)] = file_sha(path)
        self.stats['json_parses'] += 1
        return read(path)

    def present(self, path):
        path = str(Path(path).resolve())
        if path not in self.presence:
            self.stats['scientific_path_presence_stats'] += 1
            self.presence[path] = Path(path).is_file()
        return self.presence[path]

    def build(self):
        docs = {key: self.load(path) for key, path in PATHS.items()}
        registry, old_lock = docs['registry'], docs['old_lock']
        insist(file_sha(ROOT / PATHS['registry']) == old_lock['registry_sha256'], 'same closed-comparison registry bytes')
        insist(digest({k: v for k, v in registry.items() if k != 'fingerprint'}) == registry['fingerprint'], 'registry canonical identity')
        for path in WRITERS:
            self.inputs[str(ROOT / path)] = file_sha(ROOT / path)
        self.inputs[str(DATABASE)] = file_sha(DATABASE)
        db = journal(); self.stats['readonly_journal_queries'] += 1
        db_fp = digest(db)
        old, phm = docs['core_cases'], docs['phm_cases']
        records = dict(old['records'])
        for key, row in phm['data']['records'].items():
            records[key] = merge_record(records[key], row) if key in records else row
        contexts = docs['contexts']['bindings']
        catalogs = {v['archive']: v for v in docs['old_inventory']['representative_inverse_ZIP_catalogs']}
        inverses, by_case, record_sha = {}, {}, {}
        for item in registry['cases']:
            ref = item['case_reference']
            document = docs['core_cases'] if ref['path'] == PATHS['core_cases'] else docs['phm_cases'] if ref['path'] == PATHS['phm_cases'] else None
            insist(document is not None, 'no unplanned case library')
            for name in ref['field']:
                document = document[name]
            case = document[ref['index']]
            insist(source_key(case) == item['key'] and digest(case) == item['case_sha256'] and digest(case['roles']) == item['original_roles_sha256'], 'unaltered complete source case and roles')
            required = []
            for pk, pair in case['pairs'].items():
                insist(legal_measurement_pair(case, records, pair), 'own inverse donor fully source-train, not source-val/target')
                if 'inventory' in case:
                    b = case['inventory']['required_inverse'][pk]
                    archive = case['inventory']['archive']
                    schema = catalogs[archive]['affine_context_stored']
                    entry = dict(path=b['path'], sha256=b['sha256'], pair=b['pair'], raw_sha256=b['raw_input_sha256'],
                        declared_affine_context=schema, declaration='inherited representative writer-lineage ZIP catalog, NOT all-file schema verification', writer=archive)
                else:
                    b = db['INVERSE/' + pk]
                    is_new_writer = b['reused'] is False
                    entry = dict(path=b['inverse_path'], sha256=b['inverse_sha256'], pair=b['pair'], raw_sha256=b['input_sha256'],
                        declared_affine_context=True if is_new_writer else None,
                        declaration='PHM source writer stores x/mu/scale' if is_new_writer else 'reused PHM entry: schema resolved only from identical already cataloged file',
                        writer='ress_event_phm_main' if is_new_writer else 'PHM_reused_parent')
                insist(entry['pair'] == pair and len(entry['raw_sha256']) == 2, 'inverse row order and input identities')
                identity = digest(dict(path=entry['path'], sha256=entry['sha256']))
                if identity in inverses:
                    prior = inverses[identity]
                    insist(all(prior[k] == entry[k] for k in ('path', 'sha256', 'pair', 'raw_sha256')), 'shared inverse is identical')
                else:
                    entry['present'] = self.present(entry['path'])
                    inverses[identity] = entry
                required.append(identity)
                for name, h in zip(pair, entry['raw_sha256']):
                    insist(name not in record_sha or record_sha[name] == h, 'same physical record bytes across pairs and cases')
                    record_sha[name] = h
            by_case[item['key']] = (case, item, sorted(set(required)))
        insist(len(by_case) == 81 and len({(c['task']['protocol'], c['task']['fold']) for c, _, _ in by_case.values()}) == 27, 'all81 cases and27 tasks, no favorable subset')
        # Borrow only a measured row, never another fit's s/h. Both rows in the
        # parent must be legal sources in the CURRENT case before member decode.
        measurement_parents = collections.defaultdict(list)
        for identity, b in inverses.items():
            if b['declared_affine_context'] is True and b['present']:
                for index, record in enumerate(b['pair']):
                    measurement_parents[record].append((identity, index))
        reports, totals, selected_contexts, selected_raw = [], collections.Counter(), {}, {}
        for key, (case, item, required) in by_case.items():
            task = case['task']; source = {r for edge in required for r in inverses[edge]['pair']}
            source_rows, counts = [], collections.Counter()
            history = math.ceil(.004 * records[next(iter(source))]['fs'])
            for record in sorted(source):
                r = records[record]
                insist(legal_record(case, records, record) and r.get('bearing') not in {'K006', 'KA08', 'KA09'}, 'lawful original source context, no protected bearing')
                candidates = []
                for identity, index in measurement_parents[record]:
                    b = inverses[identity]
                    if legal_measurement_pair(case, records, b['pair']):
                        candidates.append((identity not in required, identity, index))
                if candidates:
                    borrowed, identity, index = min(candidates)
                    route = dict(kind='OTHER_LEGAL_PAIR_MEASUREMENT' if borrowed else 'OWN_INVERSE_DECLARED_CONTEXT', inverse=identity, source_index=index)
                elif record in contexts and contexts[record]['input_sha256'] == record_sha[record] and self.present(contexts[record]['path']):
                    route = dict(kind='SINGLE_RECORD_MEASUREMENT_CACHE', record=record)
                    selected_contexts[record] = contexts[record]
                elif 'RAW/' + record in db and db['RAW/' + record]['raw']['sha256'] == record_sha[record] and self.present(db['RAW/' + record]['cache']['path']):
                    route = dict(kind='PHM_SOURCE_RAW_ARRAY_CACHE', record=record)
                    selected_raw[record] = db['RAW/' + record]
                else:
                    route = dict(kind='NO_DECLARED_MEASURED_CONTEXT_ROUTE')
                counts[route['kind']] += 1
                source_rows.append(dict(record=record, measurement=route, original_source_train_intervals=intervals(case['roles'][record]['tr'], r['window'])))
            for bank in case['bank']:
                r = records[bank['record']]
                insist(bank['record'] in source and bank['pair'] in case['pairs'] and case['pairs'][bank['pair']][bank['source']] == bank['record'], 'all bank source/row/pair identities')
                insist(bank['starts'] == [5792 + j*r['window'] for j in range(4)], 'unmodified four actual context-relative crops')
                kernel_history = math.ceil(.004*r['fs']) - 1
                for start in bank['starts']:
                    insist(start-history-kernel_history >= 0 and start+r['window'] <= SPAN[1]-SPAN[0], 'full causal channel and packet history inside measured source context')
            insist(item['expected_bank_windows'] == len(case['bank'])*4, 'unaltered bank denominator')
            totals.update(counts)
            reports.append(dict(key=key, protocol=task['protocol'], seed=case['seed'], source_domains=task['sources'], target_domain=task['target'],
                original_case_sha256=item['case_sha256'], original_role_sha256=item['original_roles_sha256'],
                original_bank_binding=item['bank_binding'], bank_windows=item['expected_bank_windows'],
                context_raw_span=list(SPAN), crop_starts_in_context=[5792+j*task['window'] for j in range(4)],
                crop_starts_in_raw_record=[SPAN[0]+5792+j*task['window'] for j in range(4)],
                channel_history_samples=history, metadata_geometry_pass=True, required_inverse=required,
                source_records=source_rows, measurement_route_counts=dict(counts)))
        # No probability/target metrics are input to this source-only inventory.
        insist(all(file_sha(path) == h for path, h in self.inputs.items()), 'all metadata/code and closed source DB bytes unchanged')
        insist(digest(journal()) == db_fp, 'read-only source journal semantic snapshot unchanged')
        self.stats['readonly_journal_queries'] += 1
        insist('numpy' not in sys.modules and 'torch' not in sys.modules, 'metadata inventory has no neural/numeric runtime')
        return dict(status='SOURCE_METADATA_COVERAGE_COMPLETE_PAYLOAD_UNVERIFIED' if not totals['NO_DECLARED_MEASURED_CONTEXT_ROUTE'] else 'SOURCE_MEASUREMENT_METADATA_GAPS_IDENTIFIED',
            training_authorized=False, scientific_effectiveness='NOT_TESTED', novelty='UNRESOLVED', cases=reports,
            inverse_catalog=inverses, source_record_metadata={k: dict(protocol=records[k]['protocol'], domain=records[k]['domain'], label=records[k]['label'], fs=records[k]['fs'], window=records[k]['window'], input_sha256=record_sha[k]) for k in sorted(record_sha)},
            selected_single_record_contexts=selected_contexts, selected_source_raw_array_caches=selected_raw,
            metadata_counts=dict(cases=len(reports), tasks=27, required_inverse_files=len(inverses), missing_inverse_files=sum(not b['present'] for b in inverses.values()),
                inverse_case_roles=sum(len(v['required_inverse']) for v in reports), unique_source_records=len(record_sha),
                source_record_case_roles=sum(len(v['source_records']) for v in reports), measurement_routes=dict(totals),
                existing_single_record_context_catalog_records=len(contexts)),
            fresh_payload_sha_checks=0, fresh_payload_schema_checks=0, science_payload_array_reads=0,
            raw_signal_reads=0, new_FFT=0, new_inverse_fits=0, neural_actions=0,
            metadata_IO_counts=dict(self.stats), source_journal_semantic_sha256=db_fp,
            input_sha256=self.inputs, code_sha256=file_sha(__file__), at_unix=time.time(),
            warning='writer/schema declarations and file presence are not real-array numeric qualification; any future loader must enforce current-case legality before opening a full multi-row member')


class Tests(unittest.TestCase):
    def test_train_gap_refused(self):
        self.assertFalse(covered([0, 10], 8, 0, 18))
    def test_touching_intervals_merge(self):
        self.assertTrue(covered([0, 8, 16], 8, 0, 24))
    def test_duplicate_offsets_refused(self):
        with self.assertRaises(AssertionError):
            intervals([0, 0], 8)
    def test_target_parent_row_refused(self):
        records = {k: dict(protocol='p', domain=k, window=32768) for k in ('A', 'B', 'C')}
        case = dict(task=dict(protocol='p', sources=['A', 'B']), roles={k: dict(tr=[65536], va=[]) for k in records})
        self.assertTrue(legal_measurement_pair(case, records, ['A', 'B']))
        self.assertFalse(legal_measurement_pair(case, records, ['A', 'C']))
    def test_validation_context_refused(self):
        records = {'A': dict(protocol='p', domain='A', window=32768)}
        case = dict(task=dict(protocol='p', sources=['A']), roles={'A': dict(tr=[65536], va=[65536])})
        self.assertFalse(legal_record(case, records, 'A'))
    def test_no_payload_hash_or_json(self):
        for action in (read, file_sha):
            with self.assertRaises(PermissionError):
                action('forbidden.npz')
    def test_optional_provenance_columns_preserved(self):
        self.assertEqual(merge_record(dict(fs=64000, file='a'), dict(fs=64000)), dict(fs=64000, file='a'))
    def test_physical_record_conflict_refused(self):
        with self.assertRaises(AssertionError):
            merge_record(dict(fs=64000), dict(fs=48000))
    def test_unknown_length_can_gain_archived_evidence(self):
        self.assertEqual(merge_record(dict(expected_samples=None, provenance_samples=123), dict(expected_samples=123))['expected_samples'], 123)
    def test_conflicting_length_provenance_refused(self):
        with self.assertRaises(AssertionError):
            merge_record(dict(expected_samples=None, provenance_samples=123), dict(expected_samples=124))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    elif sys.argv[1:] == ['write']:
        result = Inventory().build()
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT/'SOURCE_METADATA_V1.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(dict(status=result['status'], counts=result['metadata_counts'], sha256=file_sha(OUT/'SOURCE_METADATA_V1.json'))))
    elif not sys.argv[1:]:
        result = read(OUT/'SOURCE_METADATA_V1.json')
        insist(result['code_sha256'] == file_sha(__file__), 'unchanged inventory code')
        insist(all(file_sha(p) == h for p, h in result['input_sha256'].items()), 'unchanged bound metadata/source DB')
        insist(digest(journal()) == result['source_journal_semantic_sha256'], 'unchanged read-only journal')
        print(json.dumps(dict(status='PASS_BOUND_SOURCE_METADATA_ONLY', sha256=file_sha(OUT/'SOURCE_METADATA_V1.json'))))
    else:
        raise SystemExit('only --self-test, one-time write, or default metadata verification')
