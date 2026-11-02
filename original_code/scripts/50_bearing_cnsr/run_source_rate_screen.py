"""One locked76-model comparison, actual CUDA P0, freeze, evaluation and audit."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import source_rate_plan as p
import source_rate_screen_plan as plan


def predict(rt, chain, path, arrays, checkpoint=None):
    import numpy as np
    from source_rate_neural import tensor_sha
    from source_rate_screen_data import check_vectors
    spec = rt.lock['predictions'][str(path)]
    p.require(spec['chain'] == chain.name, 'one preregistered prediction identity')
    p.require(spec['side'] == 'source' or rt.phase == 'TARGET' and rt.get('SOURCE_FREEZE'), 'no target before all-source freeze')
    check_vectors(arrays, rt.lock['vectors'][spec['expected_key']]); n = len(arrays['y'])
    if spec['side'] == 'source':
        p.require(chain.step_n == spec['step'], 'fixed source score state')
    else:
        state = rt.finished()[chain.name]; step = 2000 if spec['policy'] == 'last' else state['selected_step']
        p.require(chain.step_n == step and checkpoint == state[spec['policy']], 'source-only target policy selection')
    torch = chain.torch; rt.before_neural()
    rt.claim('source_prediction_windows' if spec['side'] == 'source' else 'target_prediction_windows', n)
    before = tensor_sha(chain.model.state_dict()); chain.model.eval(); parts = []
    with torch.no_grad():
        for lo in range(0, n, 512):
            rt.before_neural(); x = torch.as_tensor(arrays['x'][lo:lo + 512].copy(), device='cuda')
            logits = chain.model(x)['logits'].detach().cpu().numpy().astype(float)
            logits -= logits.max(1, keepdims=True); prob = np.exp(logits); parts.append(prob / prob.sum(1, keepdims=True))
    torch.cuda.synchronize(); p.require(before == tensor_sha(chain.model.state_dict()), 'frozen parameters and BN in prediction')
    payload = dict(prob=np.concatenate(parts), **{k: arrays[k] for k in ('y', 'g', 'domain', 'offset')},
        chain=np.array(chain.name), step=np.array(chain.step_n), model_tensor_sha256=np.array(before),
        fingerprint=np.array(rt.lock['fingerprint']), checkpoint_sha256=np.array('' if checkpoint is None else rt.artifact(checkpoint)['sha256']))
    rt.publish(str(path), arrays=payload)
    return payload


def source_score(rt, chain, source, qualification=False):
    import numpy as np
    arrays = source['va']
    if qualification: arrays = {k: v[:32] for k, v in arrays.items()}
    path = plan.paths(rt.lock['campaign']['store'], chain.name, 'source', chain.step_n)
    result = predict(rt, chain, path, arrays)
    if not qualification:
        score = float(np.mean([(result['prob'][arrays['domain'] == d].argmax(1) == arrays['y'][arrays['domain'] == d]).mean()
            for d in chain.case['task']['sources']]))
        p.require(np.isfinite(score), 'finite original domain-equal source score')
        chain.curve.append(dict(step=chain.step_n, score=score, probability=path, target_used=False))
        if chain.best is None or score > chain.best: chain.best, chain.selected = score, chain.step_n
    return result


def save_chain(rt, chain):
    path = plan.paths(rt.lock['campaign']['store'], chain.name, 'checkpoint', chain.step_n)
    rt.publish(path, packet=chain.packet()); return path


def scientific_packet(packet):
    value = copy.deepcopy(packet)
    # The only discarded fields are bookkeeping identities; no tensor/Adam/RNG is removed.
    value.pop('chain'); value.pop('arm'); return value


def qualify(rt, files):
    if rt.get('P0_PASS'): return
    from source_rate_neural import Chain, Data, exact
    started = time.time(); total_steps = 0; proofs = []; same_clock = {}
    for key in plan.P0_CASES:
        case = rt.lock['cases'][key]; source = files.source(case)
        data = Data(case, source, files.bank(case), rt)
        for arm in p.ARMS:
            output = []
            for branch in ('continuous', 'resume'):
                name = 'P0/' + key + '/' + arm + '/' + branch
                p.require(rt.db.execute('SELECT count(*) FROM steps WHERE chain=?', (name,)).fetchone()[0] == 0,
                    'incomplete P0 is not silently replayed')
                chain = Chain(rt, case, arm, name, data)
                initial = source_score(rt, chain, source, True)
                for _ in range(5):
                    chain.step(inspect=chain.step_n == 0); total_steps += 1
                    if branch == 'resume' and chain.step_n == 2:
                        path = save_chain(rt, chain); before = chain.packet(); del chain
                        chain = Chain(rt, case, arm, name, data, rt.reopen(path))
                        exact(before, chain.packet(), chain.torch, chain.np, 'P0 actual disk2 full-state restore')
                final = source_score(rt, chain, source, True); save_chain(rt, chain)
                packet = scientific_packet(chain.packet()); output.append((packet, initial['prob'], final['prob']))
                torch, np = chain.torch, chain.np; del chain
            for index in range(3): exact(output[0][index], output[1][index], torch, np, 'continuous5 vs actual disk2+3')
            if not case['source_speed_diversity'] and arm != p.ARMS[0]:
                if arm == p.ARMS[1]: same_clock[key] = output[0]
                else:
                    for index in range(3): exact(same_clock[key][index], output[0][index], torch, np, 'same-clock fixed/coupled/ring scientific alias')
            proofs.append(dict(case_key=key, arm=arm, exact=True, actual_disk_resume=True, finite_nonzero_gradient=True))
        del data
    duration = time.time() - started
    # Only an execution feasibility guard, not an effect/target gate. Generous2x updates plus fixed I/O slack.
    estimated_seconds = duration / total_steps * 152000 * 2 + 900
    start = rt.get('neural_start')['at_unix']; remaining = min(start + rt.lock['campaign']['neural_seconds'],
        __import__('datetime').datetime.fromisoformat(rt.lock['campaign']['deadline']).timestamp()) - time.time()
    # Most P0 time is48 model/disk/probe setup, not update cost; use separately measured step times below.
    update_times = rt.get('P0_UPDATE_TIMES') or []
    p.require(len(update_times) == 160, 'all actual qualification updates timed')
    estimated_seconds = sum(update_times) / len(update_times) * 152000 * 2 + 1200
    p.require(estimated_seconds < remaining, 'actual CUDA timing cannot fit remaining fixed budget')
    receipt = dict(status='PASS', proofs=proofs, updates=total_steps, elapsed_seconds=duration,
        estimated_remaining_seconds=estimated_seconds, remaining_seconds=remaining,
        all_same_clock_states_and_probabilities_exact=True, new_target_predictions=0, at_unix=time.time())
    p.save(p.OUT / 'CUDA_P0_V1.json', receipt)
    with rt.db: rt.set('P0_PASS', receipt)
    print('SOURCE_RATE_CUDA_P0_PASS', total_steps, round(duration, 2), flush=True)


def train(rt, files):
    from source_rate_neural import Chain, Data, exact
    p.require(rt.get('P0_PASS'), 'real locked CUDA qualification before formal models')
    for key, case in sorted(rt.lock['cases'].items()):
        source = files.source(case); data = None
        for arm in case['real_arms']:
            name = 'MAIN/' + key + '/' + arm
            if name in rt.finished(): continue
            if data is None: data = Data(case, source, files.bank(case), rt)
            artifacts = [(v['step'], path) for path, v in rt.lock['checkpoints'].items()
                if v['chain'] == name and rt.artifact(path)]
            if artifacts:
                step, path = max(artifacts)
                paid, done = rt.db.execute('SELECT count(*),coalesce(sum(done),0) FROM steps WHERE chain=?', (name,)).fetchone()
                p.require(paid == done == step, 'no implicit lost-tail optimization replay')
                chain = Chain(rt, case, arm, name, data, rt.reopen(path))
            else:
                p.require(rt.db.execute('SELECT count(*) FROM steps WHERE chain=?', (name,)).fetchone()[0] == 0, 'no unsaved paid prefix replay')
                chain = Chain(rt, case, arm, name, data); source_score(rt, chain, source); save_chain(rt, chain)
            while chain.step_n < 2000:
                chain.step(inspect=chain.step_n == 0)
                if chain.step_n in plan.STEPS: source_score(rt, chain, source)
                if chain.step_n % 250 == 0:
                    path = save_chain(rt, chain)
                    if chain.step_n == 1000:
                        before = chain.packet(); del chain
                        chain = Chain(rt, case, arm, name, data, rt.reopen(path))
                        exact(before, chain.packet(), chain.torch, chain.np, 'formal actual disk1000 all-state restore')
            result = dict(last=plan.paths(rt.lock['campaign']['store'], name, 'checkpoint', 2000),
                selected=plan.paths(rt.lock['campaign']['store'], name, 'checkpoint', chain.selected),
                selected_step=chain.selected, curve=chain.curve, losses=chain.losses, completed_at_unix=time.time(),
                actual_disk1000_restore_exact=True)
            rt.finish(name, result); del chain
            print('SOURCE_RATE_MODEL', len(rt.finished()), '/76', key, arm, flush=True)
        if data is not None: del data
    finished = rt.finished()
    p.require(len(finished) == 76 and rt.counts['formal_updates'] == 152000, 'all registered76 source models first')
    if not rt.get('SOURCE_FREEZE'):
        bound = {}
        for name, item in finished.items():
            for step in plan.STEPS:
                path = plan.paths(rt.lock['campaign']['store'], name, 'checkpoint', step)
                value = rt.artifact(path)
                p.require(value and rt.hash(path, 'checkpoint_hashes') == value['sha256'], 'all frozen critical state hashes')
                bound[path] = value['sha256']
        freeze = dict(at_unix=time.time(), models=76, logical_aliases=12, states=bound,
            selected={k: v['selected_step'] for k, v in finished.items()}, fingerprint=rt.lock['fingerprint'])
        p.save(p.OUT / 'FROZEN_SOURCE_V1.json', freeze)
        with rt.db: rt.set('SOURCE_FREEZE', freeze)


def evaluate(rt, files):
    from source_rate_neural import Chain
    p.require(rt.get('SOURCE_FREEZE'), 'all source states frozen before target')
    rt.phase = 'TARGET'
    for name, item in rt.finished().items():
        definition = rt.lock['chains'][name]; case = rt.lock['cases'][definition['case_key']]
        target = files.target(case)
        for policy in ('last', 'selected'):
            path = str(Path(rt.lock['campaign']['store']) / 'neural' / name / f'target_{policy}.npz')
            if rt.artifact(path): continue
            checkpoint = item[policy]; chain = Chain(rt, case, definition['arm'], name, packet=rt.reopen(checkpoint))
            predict(rt, chain, path, target, checkpoint); del chain
    with rt.db: rt.set('NEURAL_END', dict(at_unix=time.time()))
    rt.phase = 'AUDIT'


def decisions(rows, lock):
    from e0_selection_metrics import aggregate
    cases = [(c['task']['protocol'], c['task']['fold'], c['seed']) for c in lock['cases'].values()]
    methods = (*p.ARMS, 'E0_EVENTS', 'B1_IN_MIX')
    keys = [(v['protocol'], v['fold'], v['seed'], v['arm'], v['role']) for v in rows]
    wanted = {(*case, arm, policy) for case in cases for arm in methods for policy in ('last', 'selected')}
    p.require(len(keys) == len(wanted) and set(keys) == wanted, 'complete264 fixed metric roles')
    primary = aggregate(rows, cases, methods, 'last'); secondary = aggregate(rows, cases, methods, 'selected')
    per_seed = {str(s): aggregate(rows, cases, methods, 'last', seed=s) for s in (0, 1)}
    effect, specificity = [], []
    def check(destination, name, delta, bound, strict=False):
        destination.append(dict(name=name, delta=delta, bound=bound, passed=bool(delta > bound + 1e-12 if strict else delta + 1e-12 >= bound)))
    for anchor in (p.ARMS[1], 'E0_EVENTS', 'B1_IN_MIX'):
        for weighting in ('dataset_equal', 'task_equal'):
            for field in ('acc', 'macro_f1', 'recording_ba'):
                check(effect, f'{anchor}/{weighting}/{field}', primary[p.ARMS[2]][weighting][field] - primary[anchor][weighting][field], .003 if field == 'acc' else 0.)
            for seed, values in per_seed.items():
                check(effect, f'{anchor}/{weighting}/seed{seed}', values[p.ARMS[2]][weighting]['acc'] - values[anchor][weighting]['acc'], 0., True)
        for protocol in primary[anchor]['per_protocol']:
            check(effect, f'{anchor}/{protocol}', primary[p.ARMS[2]]['per_protocol'][protocol]['acc'] - primary[anchor]['per_protocol'][protocol]['acc'], -.002)
    for weighting in ('dataset_equal', 'task_equal'):
        check(specificity, weighting, primary[p.ARMS[2]][weighting]['acc'] - primary[p.ARMS[3]][weighting]['acc'], .001)
        for seed, values in per_seed.items():
            check(specificity, f'{weighting}/seed{seed}', values[p.ARMS[2]][weighting]['acc'] - values[p.ARMS[3]][weighting]['acc'], 0.)
    varied = [k for k in cases if lock['cases'][k[0] + '/' + k[1] + '/s' + str(k[2])]['source_speed_diversity'] and k[0] in ('pu_clock4', 'phm09_w3')]
    layer = aggregate(rows, varied, methods, 'last')
    deltas = {protocol: layer[p.ARMS[2]]['per_protocol'][protocol]['acc'] - layer[p.ARMS[3]]['per_protocol'][protocol]['acc'] for protocol in ('pu_clock4', 'phm09_w3')}
    check(specificity, 'varied_source_PU_or_PHM_positive', max(deltas.values()), 0., True)
    passed = all(v['passed'] for v in effect); specific = all(v['passed'] for v in specificity)
    return dict(status='CANDIDATE_SCREEN_FAILED' if not passed else 'GAIN_WITHOUT_SOURCE_RATE_SPECIFICITY' if not specific else 'SOURCE_COUPLED_RATE_DEVELOPMENT_SIGNAL',
        effect_gate=passed, specificity_gate=specific, effect_checks=effect, specificity_checks=specificity,
        primary=primary, secondary=secondary, per_seed=per_seed, varied_source_layer=layer,
        varied_source_PU_PHM_differences=deltas, novelty='UNRESOLVED', causal_latents_identified=False, global_SOTA_established=False)


def independent_metrics(prob, y, g, classes):
    import numpy as np
    groups, inverse = np.unique(g, return_inverse=True); total = np.zeros((len(groups), classes)); np.add.at(total, inverse, prob)
    prediction = (total / np.bincount(inverse)[:, None]).argmax(1)
    truth = np.array([y[np.flatnonzero(inverse == i)[0]] for i in range(len(groups))])
    p.require(np.array_equal(truth[inverse], y), 'independent recording-label constancy')
    window = np.array([[int(((y == a) & (prob.argmax(1) == b)).sum()) for b in range(classes)] for a in range(classes)])
    recording = np.array([[int(((truth == a) & (prediction == b)).sum()) for b in range(classes)] for a in range(classes)])
    denominator = window.sum(0) + window.sum(1)
    f1 = np.divide(2 * np.diag(window), denominator, out=np.zeros(classes), where=denominator > 0)
    observed = recording.sum(1) > 0
    return dict(acc=float(np.trace(window) / window.sum()), macro_f1=float(f1.mean()),
        recording_ba=float((np.diag(recording)[observed] / recording.sum(1)[observed]).mean()), windows=len(y), recordings=len(groups),
        window_confusion=window.tolist(), recording_confusion=recording.tolist())


def audit(rt):
    import numpy as np
    from source_rate_screen_data import check_vectors
    from e0_selection_metrics import from_probabilities
    rt.phase = 'AUDIT'; rows = []; source_scores = {}; max_error = 0.
    p.require(rt.get('NEURAL_END') and rt.get('SOURCE_FREEZE'), 'actual neural work complete before audit')
    for path, spec in rt.lock['predictions'].items():
        a = rt.reopen(path); check_vectors(a, rt.lock['vectors'][spec['expected_key']])
        expected_fingerprint = rt.lock.get('parent_p0_fingerprint', rt.lock['fingerprint']) if spec['qualification'] else rt.lock['fingerprint']
        p.require(str(a['fingerprint']) == expected_fingerprint and str(a['chain']) == spec['chain'], 'saved prediction scientific identity')
        case = rt.lock['cases'][spec['case_key']]; classes = case['task']['classes']
        metrics = from_probabilities(a['prob'], a['y'], a['g'], classes)
        independent = independent_metrics(a['prob'], a['y'], a['g'], classes)
        for field in ('acc', 'macro_f1', 'recording_ba'):
            error = abs(metrics[field] - independent[field]); max_error = max(max_error, error); p.require(error < 1e-12, 'independent probability metrics')
        p.require(metrics['window_confusion'] == independent['window_confusion'] and metrics['recording_confusion'] == independent['recording_confusion'], 'independent confusion matrices')
        if spec['side'] == 'source' and not spec['qualification']:
            source_scores.setdefault(spec['chain'], []).append((spec['step'], float(np.mean([
                (a['prob'][a['domain'] == d].argmax(1) == a['y'][a['domain'] == d]).mean() for d in case['task']['sources']]))))
        if spec['side'] == 'target':
            item = rt.finished()[spec['chain']]
            p.require(rt.artifact(path)['at_unix'] >= rt.get('SOURCE_FREEZE')['at_unix'] and
                str(a['checkpoint_sha256']) == rt.artifact(item[spec['policy']])['sha256'], 'frozen selected state lineage')
            definition = rt.lock['chains'][spec['chain']]
            rows.append(dict(protocol=case['task']['protocol'], fold=case['task']['fold'], seed=case['seed'],
                arm=definition['arm'], role=spec['policy'], metrics=metrics, fresh_this_campaign=True))
    finished = rt.finished()
    for name, item in finished.items():
        scores = sorted(source_scores[name]); selected = max(scores, key=lambda v: v[1])[0]
        p.require([v[0] for v in scores] == list(plan.STEPS) and selected == item['selected_step'], 'earliest strict source selection independently recomputed')
        for step in plan.STEPS:
            path = plan.paths(rt.lock['campaign']['store'], name, 'checkpoint', step)
            p.require(rt.hash(path, 'checkpoint_hashes') == rt.get('SOURCE_FREEZE')['states'][path], 'critical full states unchanged')
    for alias, origin in rt.lock['aliases'].items():
        key = alias.split('/')[1:4]; case_key = '/'.join(key); arm = alias.split('/')[-1]
        case = rt.lock['cases'][case_key]
        p.require(not case['source_speed_diversity'] and rt.get('P0_PASS')['all_same_clock_states_and_probabilities_exact'], 'alias only after actual CUDA proof')
        source_arm = rt.lock['chains'][origin]['arm']
        originals = [r for r in rows if (r['protocol'], r['fold'], r['seed'], r['arm']) == (case['task']['protocol'], case['task']['fold'], case['seed'], source_arm)]
        p.require(len(originals) == 2, 'two exact alias policies')
        rows.extend(dict(r, arm=arm, fresh_this_campaign=False, exact_alias_of=origin) for r in originals)
    archived = p.read(p.OUT / 'ARCHIVED_ANCHORS_V1.json'); rows.extend(archived['rows'])
    result = decisions(rows, rt.lock)
    paid = rt.db.execute('SELECT count(*),sum(done) FROM steps').fetchone()
    p.require(paid == (152160, 152160) and len(finished) == 76 and rt.counts['qualification_updates'] == 160 and
        rt.counts['formal_updates'] == 152000, 'all real qualification/formal paid and done, no unaccounted replay')
    p.require(rt.counts['source_prediction_windows'] == rt.lock['forecast']['source_prediction_windows'] and
        rt.counts['target_prediction_windows'] == rt.lock['forecast']['target_prediction_windows'] and
        rt.counts['training_forward_windows'] == rt.lock['forecast']['training_forward_windows'], 'complete forecast actual NN windows')
    p.require(rt.db.execute('SELECT count(*) FROM artifacts').fetchone()[0] == len(rt.lock['predictions']) + len(rt.lock['checkpoints']), 'all scheduled artifacts present')
    plan.verify(rt.lock)
    result.update(rows=rows, execution='COMPLETE', fingerprint=rt.lock['fingerprint'], counts=rt.counts,
        metric_max_error=max_error, all_source_freeze=rt.get('SOURCE_FREEZE')['at_unix'],
        neural_start=rt.get('neural_start'), neural_end=rt.get('NEURAL_END'), audit_at_unix=time.time(),
        audit_boundary='independent saved-probability arithmetic and source-selection/state-hash checks; actual optimizer replay tested at P0 and formal1000, not whole-training replay')
    p.save(p.OUT / 'SCREEN_RESULT_V1.json', result)
    p.save(p.OUT / 'AUDIT_V1.json', dict(status='PASS', fingerprint=rt.lock['fingerprint'],
        result_sha256=p.sha(p.OUT / 'SCREEN_RESULT_V1.json'), metric_max_error=max_error, counts=rt.counts,
        models=76, logical_aliases=12, probability_roles=len(rt.lock['predictions']), metric_roles=len(rows), at_unix=time.time()))
    with rt.db:
        rt.set('DONE', dict(at_unix=time.time(), scientific=result['status']))
        rt.set('owner', dict(state='DONE', native=rt.native, pid=rt.native['pid']))
    print('SOURCE_RATE_COMPLETE', result['status'], 'metric_error', max_error, flush=True)


def run():
    from source_rate_screen_runtime import Runtime, Guard
    from source_rate_screen_data import Files
    lock = p.read(plan.LOCK); plan.verify(lock)
    grant = p.read(p.OUT / lock.get('grant_filename', 'NEURAL_ACTUAL_GRANT_V1.json'))
    p.require(grant['fingerprint'] == lock['fingerprint'] and grant['authorization'] == 'standing_authorization' and
        grant['human_message_id'] == lock['campaign']['human_message_id'], 'actual recorded standing authorization for this finite comparison')
    rt = Runtime(lock)
    sys.addaudithook(Guard(rt)); files = Files(rt)
    try:
        qualify(rt, files); train(rt, files); evaluate(rt, files); audit(rt)
    except BaseException as error:
        with rt.db:
            rt.set('STOPPED', dict(at_unix=time.time(), kind=type(error).__name__, reason=str(error)))
            rt.set('owner', dict(state='STOPPED', native=rt.native, pid=rt.native['pid']))
        raise
    finally: rt.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('prepare', 'test', 'run'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        lock = plan.compile_neural(); p.save(plan.LOCK, lock); print(json.dumps(dict(fingerprint=lock['fingerprint'], forecast=lock['forecast'], limits=lock['limits']), indent=2))
    elif args.mode == 'test':
        import unittest
        unittest.main(module='test_source_rate_screen', argv=['source_rate_screen'], verbosity=2)
    else: run()
