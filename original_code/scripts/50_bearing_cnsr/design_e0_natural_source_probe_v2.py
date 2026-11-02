"""Three-crop metadata amendment; original four-crop proposal stays preserved.

The real ratios were not old augmentation factors. Finite-window support must
therefore be checked anew, not inferred from the denominator<=12 condition.
"""
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import check_e0_natural_source_support as support

ROOT, OUT = support.ROOT, support.OUT


def valid_support(length, kernel_size, ratio, starts, window):
    center = Fraction(length - 1, 2)
    return all(center + ratio * (start - (kernel_size - 1) - center) >= 0 and
               center + ratio * (start + window - 1 - center) <= length - 1
               for start in starts)


def centered_three(length, window):
    support.insist(length >= 3 * window, 'three complete fixed nonoverlapping windows')
    begin = (length - 3 * window) // 2
    return [begin + k * window for k in range(3)]


def amend(original, records):
    result = {k: v for k, v in original.items() if k not in
        ('units', 'code_sha256', 'generated_at', 'metadata_bindings', 'metadata_bytes')}
    result['units'] = []
    old_bad = 0
    for item in original['units']:
        unit = dict(item)
        record = records[unit['origin_record']]
        window, length = record['window'], unit['context'][1] - unit['context'][0]
        kernel = math.ceil(.004 * record['fs'])
        ratios = [Fraction(unit['correct_ratio']), Fraction(unit['reverse_ratio'])]
        old_ok = all(valid_support(length, kernel, r, unit['origin_query_starts'], window)
                     for r in ratios)
        old_bad += not old_ok
        starts = centered_three(length, window)
        support.insist(all(valid_support(length, kernel, r, starts, window) for r in ratios),
                       'every correct/reverse crop has original physical event support')
        reference = dict(unit['natural_reference'])
        reference['offsets'] = reference['offsets'][:3]
        support.insist(len(reference['offsets']) == 3 and reference['window'] == window,
                       'same three source-only full reference windows')
        unit.update(natural_reference=reference, origin_query_starts=starts,
            window=window, fs=record['fs'], kernel_length=kernel,
            source_context_support='BOTH_CORRECT_AND_REVERSE_PASS',
            old_four_crop_support=old_ok)
        result['units'].append(unit)
    result.update(status='THREE_CROP_SOURCE_PROBE_DESIGN_BOUND_NUMERIC_EXECUTION_PENDING',
        amendment='same274 units and exact natural ratios; all use three centered full windows',
        old_four_crop_units_without_both_ratio_support=old_bad,
        reference_window_roles=3 * len(result['units']),
        proposed_rendered_window_roles=3 * len(result['units']),
        new_scientific_payload_reads=0)
    return result


def main():
    scope = json.loads((OUT / 'SCOPE_V1.json').read_text(encoding='utf8'))
    support.insist(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']),
                   'original finite research deadline')
    raw = (OUT / 'PROBE_CATALOG_V1.json').read_bytes()
    original = json.loads(raw)
    name = 'results/wfdg/ress_mocd_complete/PART1_SOURCE_MANIFEST_V2.json'
    data = (ROOT / name).read_bytes()
    expected = next(v for v in original['metadata_bindings'] if v['path'] == name)
    digest = hashlib.sha256(data).hexdigest()
    support.insist(digest == expected['sha256'], 'unchanged original record metadata')
    result = amend(original, json.loads(data)['records'])
    result.update(parent_catalog_sha256=hashlib.sha256(raw).hexdigest(),
        record_metadata_sha256=digest, metadata_bytes=len(raw) + len(data),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generated_at=datetime.now().astimezone().isoformat())
    with (OUT / 'PROBE_CATALOG_V2.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('units', 'task_coverage')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
