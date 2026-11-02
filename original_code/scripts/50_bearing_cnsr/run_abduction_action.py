"""Finite zero-neural source probe on already bound archived inverse pairs."""
import argparse
import ast
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

from abduction_action_plan import ROOT, OUT, require, canonical, sha

CODE = ('abduction_action_plan.py', 'abduction_action_core.py', 'test_abduction_action.py',
        'run_abduction_action.py', 'clock_real_core.py', 'clock_real_data.py')


def save(name, value):
    with (OUT/name).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def code_lock():
    require(not (OUT/'CODE_LOCK_V1.json').exists(), 'no code-lock replay')
    scope = json.loads((OUT/'SCOPE_V1.json').read_text(encoding='utf8'))
    require(datetime.now().astimezone() < datetime.fromisoformat(scope['deadline']), 'original deadline')
    value = dict(input_lock_sha256=sha(OUT/'SOURCE_INPUT_LOCK_V1.json'),
        scope_sha256=sha(OUT/'SCOPE_V1.json'), amendment_sha256=sha(OUT/'PRE_PAYLOAD_AMENDMENT.md'),
        code_sha256={str(ROOT/'scripts/50_bearing_cnsr'/name): sha(ROOT/'scripts/50_bearing_cnsr'/name) for name in CODE},
        arrays_opened=0, at=datetime.now().astimezone().isoformat())
    value['fingerprint'] = canonical(value)
    save('CODE_LOCK_V1.json', value)
    print(json.dumps(dict(fingerprint=value['fingerprint'], arrays_opened=0, at=value['at'])))


class Runtime:
    def __init__(self, lock):
        self.lock = lock; self.scope = lock['scope']; self.counts = Counter(lock['counts'])
        self.allowed = {str(Path(b['path']).resolve()).casefold() for unit in lock['selected'] for b in unit['bindings'].values()}

    def clock(self):
        require(datetime.now().astimezone() < datetime.fromisoformat(self.scope['deadline']), 'original1h total deadline')

    def guard(self, event, args):
        if event == 'import' and isinstance(args[0], str) and args[0].split('.')[0] in ('torch', 'tensorflow', 'jax'):
            raise PermissionError('no neural library import')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in ('.mat', '.csv', '.npy', '.pt', '.pth', '.h5', '.pkl'):
            raise PermissionError('no raw, model or unregistered scientific payload')
        if path.suffix.lower() == '.npz':
            require(str(path).casefold() in self.allowed, 'only six fixed source-only payloads')

    def fft(self, name, value, n=None, axis=-1, **kwargs):
        import numpy as np
        self.clock()
        a = np.asarray(value); axis %= a.ndim
        number = a.size//a.shape[axis]
        length = n if n is not None else (2*(a.shape[axis]-1) if name == 'irfft' else a.shape[axis])
        self.counts['logical_real_or_synthetic_1d_FFTs'] += int(number)
        self.counts['FFT_points_total'] += int(number*length)
        require(self.counts['logical_real_or_synthetic_1d_FFTs'] <= self.scope['limits']['logical_real_or_synthetic_1d_FFTs'] and
                self.counts['FFT_points_total'] <= self.scope['limits']['FFT_points_total'], 'FFT quota before operation')
        self.counts['fft_calls'] += 1
        return getattr(np.fft, name)(value, n=n, axis=axis, **kwargs)

    def payload(self, binding, kind):
        import numpy as np
        self.clock(); size = Path(binding['path']).stat().st_size
        require(size == binding['size_bytes'], 'bound payload size')
        self.counts[kind+'_NPZ_opens'] += 1; self.counts['source_payload_bytes'] += size
        require(self.counts[kind+'_NPZ_opens'] <= 3 and self.counts['source_payload_bytes'] <= self.lock['payload_limit']['combined_bytes'], 'source decode quota before read')
        raw = Path(binding['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'exact previously qualified source payload SHA')
        names = ('s', 'params') if kind == 'inverse' else ('observed', 'u_normalized', 'mean', 'scale')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in names}


def physics(runtime):
    import numpy as np
    class FFT:
        def __getattr__(self, name):
            require(name in ('rfft', 'irfft'), 'only original real FFTs')
            return lambda value, n=None, axis=-1, **kw: runtime.fft(name, value, n=n, axis=axis, **kw)
    class Proxy:
        fft = FFT()
        def __getattr__(self, name):
            return getattr(np, name)
    namespace = dict(np=Proxy(), Fraction=Fraction)
    result = {}
    for filename, names in [('clock_real_core.py', ('kernel', 'render_phase', 'support')),
                            ('clock_real_data.py', ('spectrum',))]:
        path = ROOT/'scripts/50_bearing_cnsr'/filename
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        require({node.name for node in functions} == set(names) and all(not node.decorator_list for node in functions), 'exact old undecorated pure functions')
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        result.update({name: namespace[name] for name in names})
    return result


def source_probe(unit, rt, funcs):
    import numpy as np
    from abduction_action_core import null_family
    inv = rt.payload(unit['bindings']['inverse'], 'inverse')
    observed_pack = rt.payload(unit['bindings']['qualified_context'], 'qualified_context')
    group = unit['group']; source = group['source']; size = group['kernel_length']; n = 32768
    s, params = inv['s'], inv['params']
    require(s.shape == (2, 48, n) and params.shape == (2, 48, 3), 'unchanged full inverse axes')
    require(observed_pack['observed'].shape == observed_pack['u_normalized'].shape == (2, n), 'bound physical-unit complete context')
    observed, scale = observed_pack['observed'][source], float(observed_pack['scale'][source])
    require(np.isfinite(observed).all() and scale > 0, 'finite physical observation/scale')
    family = null_family(s[source], params[source], size, funcs['kernel'], rt.lock['coefficient_l1_addition_fraction'])
    originals = {}; deltas = {}; trivial = {}
    for text in rt.lock['ratios']:
        q = float(Fraction(text)); rt.clock()
        for start in group['starts']:
            funcs['support'](n, size, q, start, start+group['window'])
        originals[text] = funcs['render_phase'](s[source], params[source], q, size)
        deltas[text] = funcs['render_phase'](family['delta'], family['params'], q, size)
        trivial[text] = funcs['render_phase'](family['trivial_delta'], family['trivial_params'], q, size)
    original_error = float(np.max(abs(originals['1']-observed_pack['u_normalized'][source])))
    require(original_error <= 1e-8*max(1., float(np.max(abs(originals['1'])))), 'same original E0 component')
    factual_relative = float(np.linalg.norm(scale*deltas['1'])/max(1e-30, np.linalg.norm(observed)))
    require(factual_relative < 1e-10 and abs(family['addition_fraction']-.01) < 1e-12, 'same factual at fixed1% sparse cost')
    rows = []
    crops = lambda wave: np.stack([wave[t:t+group['window']] for t in group['starts']])
    factual_features = funcs['spectrum'](crops(observed)).astype(float)
    for text in ('2/3', '3/2'):
        old = observed+scale*(originals[text]-originals['1'])
        alternative = old+scale*deltas[text]
        alias = old+scale*trivial[text]
        old_features = funcs['spectrum'](crops(old)).astype(float)
        new_features = funcs['spectrum'](crops(alternative)).astype(float)
        alias_features = funcs['spectrum'](crops(alias)).astype(float)
        component_denominator = max(1e-30, np.linalg.norm(originals[text]-originals['1']))
        front_denominator = max(1e-30, np.linalg.norm(old_features-factual_features))
        old_mag = abs(rt.fft('rfft', old)); new_mag = abs(rt.fft('rfft', alternative))
        rows.append(dict(ratio=text, time_difference_l2=float(np.linalg.norm(scale*deltas[text])),
            difference_over_original_E0_event_edit=float(np.linalg.norm(deltas[text])/component_denominator),
            frontend_difference_l2=float(np.linalg.norm(new_features-old_features)),
            frontend_difference_over_original_E0_edit=float(np.linalg.norm(new_features-old_features)/front_denominator),
            trivial_null_time_l2=float(np.linalg.norm(scale*trivial[text])),
            trivial_null_frontend_l2=float(np.linalg.norm(alias_features-old_features)),
            full_context_magnitude_relative_difference=float(np.linalg.norm(new_mag-old_mag)/max(1e-30, np.linalg.norm(old_mag))),
            nonalias=bool(np.linalg.norm(new_features-old_features) > 1e-8)))
    return dict(protocol=unit['protocol'], case_key=unit['case_key'], source_record=group['origin'],
        source_pair=group['pair'], source_index=source, sample_size_unit='one deterministic existing source inverse origin, not dataset replication',
        original_u1_max_error=original_error, factual_relative_error=factual_relative,
        coefficient_l1_increase_fraction=family['addition_fraction'],
        sparse_penalty_increase_fraction=family['addition_fraction'],
        sparse_penalty_condition='old per-source penalty_s is equal across all48 coefficients',
        theoretical_full_objective_relative_upper_bound=.01,
        objective_bound_is_not_confidence_coverage_or_unchanged_optimizer_objective=True,
        no_new_kernel_or_event_clock=True, seed_interval=family['seed_interval'], rows=rows)


def run():
    require(not (OUT/'RESULT_V1.json').exists(), 'already complete, no payload replay')
    lock = json.loads((OUT/'SOURCE_INPUT_LOCK_V1.json').read_text(encoding='utf8'))
    codes = json.loads((OUT/'CODE_LOCK_V1.json').read_text(encoding='utf8'))
    require(lock['fingerprint'] == canonical({k: v for k, v in lock.items() if k != 'fingerprint'}), 'bound input lock')
    require(codes['input_lock_sha256'] == sha(OUT/'SOURCE_INPUT_LOCK_V1.json') and
        codes['scope_sha256'] == sha(OUT/'SCOPE_V1.json') and
        codes['amendment_sha256'] == sha(OUT/'PRE_PAYLOAD_AMENDMENT.md'), 'scope/input/amendment unchanged')
    for path, digest in codes['code_sha256'].items():
        require(sha(path) == digest, 'unchanged locked numerical code')
    rt = Runtime(lock); rt.clock(); sys.addaudithook(rt.guard)
    require('torch' not in sys.modules, 'zero neural at entry')
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_abduction_action'))
    require(tests.wasSuccessful() and tests.testsRun == 13, 'exact algebra before source decode')
    save('P0_V1.json', dict(status='PASS', tests=13, prior_cli_tests=12, cumulative_named_tests=25,
        output=stream.getvalue(), at=datetime.now().astimezone().isoformat(), source_arrays_opened=0,
        code_lock_fingerprint=codes['fingerprint']))
    from abduction_action_core import algebra
    funcs = physics(rt)
    results = [source_probe(unit, rt, funcs) for unit in lock['selected']]
    rt.clock(); require('torch' not in sys.modules, 'zero neural throughout')
    require(rt.counts['inverse_NPZ_opens'] == rt.counts['qualified_context_NPZ_opens'] == 3, 'exact three frozen source pairs')
    value = dict(execution='COMPLETE_NUMERICAL_QUALIFICATION',
        scientific='FACTUAL_NEAR_OBJECTIVE_AMBIGUITY_HAS_NONALIAS_INTERVENTIONS' if all(
            any(row['nonalias'] for row in item['rows']) for item in results) else 'SOURCE_INTERVENTION_AMBIGUITY_NOT_UNIFORMLY_OBSERVED',
        at=datetime.now().astimezone().isoformat(), deadline=lock['scope']['deadline'],
        input_fingerprint=lock['fingerprint'], code_fingerprint=codes['fingerprint'], counts=dict(rt.counts),
        algebra=[algebra(text) for text in lock['ratios']], source_results=results,
        new_neural_or_raw_or_inverse_fit_or_target_reads=0,
        real_classification_gain_or_label_preservation_or_physical_identification_proved=False,
        uncertainty_set_coverage_calibrated=False,
        no_actual_classifier_training_qualified_by_this_result_alone=True)
    save('RESULT_V1.json', value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'run'))
    args = parser.parse_args()
    code_lock() if args.mode == 'lock' else run()
