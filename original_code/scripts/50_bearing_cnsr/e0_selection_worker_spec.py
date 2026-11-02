"""Pure metadata for real workers. No NumPy, Torch, model or payload access."""
import hashlib
import e0_selection_control as c

SCHEMA = 'E0_SELECTION_FULL_SCIENTIFIC_PACKET_V1'
PACKET_KEYS = frozenset(('schema','fingerprint','chain','kind','case_key','teacher_key',
    'external_arm','internal_arm','initialization_seed','data_identity','step',
    'model','optimizer','model_modes','sampler','augmentation','auxiliary',
    'torch_rng','cuda_rng','numpy_rng','python_rng','losses','curve','best','selected',
    'gradient_audit','initial_encoder_sha256','initial_tensor_sha256','AMP_GradScaler','scheduler'))


def main_seed(task, seed):
    # Same formula as historical run_context_evidence_screen.seed_for.
    token = f'CONTEXT_EVIDENCE|{task["protocol"]}|{task["fold"]}|{seed}'
    return int(hashlib.sha256(token.encode()).hexdigest()[:8],16)


def definition(registry, models, qualifications, chain):
    if chain in qualifications:
        q = qualifications[chain];reference = q['reference_chain']
        if not chain.startswith('P0/') or reference not in models:raise PermissionError('fixed qualification reference')
        underlying = models[reference];steps = q['steps'];forward = q['forward_windows']
        if forward != underlying['forward_windows']:raise PermissionError('qualification must use actual batch geometry')
    elif chain in models:
        reference = chain;underlying = models[chain];steps = underlying['steps'];forward = underlying['forward_windows']
    else:raise PermissionError('unknown worker chain')
    cases = {v['key']:v for v in registry['cases']};teachers = {v['key']:v for v in registry['teachers']}
    kind = underlying['kind']
    if kind == 'auxiliary':
        teacher = teachers[underlying['origin']]
        choices = [v for v in registry['cases'] if any(g['slot'] == teacher['slot'] for g in v['query_groups'])]
        if not choices:raise PermissionError('teacher has no legal original source query')
        case = min(choices,key=lambda v:v['key']);task = case['task']
        if task['classes'] != teacher['classes']:raise PermissionError('teacher class schema')
        init = teacher['initialization_seed'];external = 'AUXILIARY_FACTUAL_MIXUP';internal = 'B1_IN_MIX'
        teacher_key = teacher['key'];slot = teacher['slot']
    elif kind == 'main':
        case = cases[underlying['origin']];task = case['task'];external = underlying['arm']
        if external not in c.ARMS:raise PermissionError('fixed G/R/U main arm')
        init = main_seed(task,case['seed']);internal = 'E0_EVENTS';teacher_key = slot = None
    else:raise PermissionError('only auxiliary or main worker families')
    return dict(chain=chain,reference_chain=reference,kind=kind,case_key=case['key'],
        teacher_key=teacher_key,slot=slot,external_arm=external,internal_arm=internal,
        initialization_seed=init,task=task,classes=task['classes'],steps=steps,
        forward_windows=forward,qualification=(chain in qualifications))


def identity(info, fingerprint, data_identity):
    return dict(schema=SCHEMA,fingerprint=fingerprint,chain=info['chain'],kind=info['kind'],
        case_key=info['case_key'],teacher_key=info['teacher_key'],external_arm=info['external_arm'],
        internal_arm=info['internal_arm'],initialization_seed=info['initialization_seed'],data_identity=data_identity)


def check_packet(packet, info, fingerprint, data_identity):
    if set(packet) != PACKET_KEYS:raise PermissionError('complete scientific state keys, no implicit omissions')
    expected = identity(info,fingerprint,data_identity)
    if any(packet[k] != v for k,v in expected.items()):raise PermissionError('packet chain, source support or selection binding')
    if type(packet['step']) is not int or not 0 <= packet['step'] <= info['steps']:raise PermissionError('packet step')
    if packet['AMP_GradScaler'] != 'not_applicable_AMP_disabled' or packet['scheduler'] is not None:
        raise PermissionError('fixed FP32/no-scheduler recipe')
    if not isinstance(packet['model_modes'],dict) or not packet['model_modes'] or any(type(v) is not bool for v in packet['model_modes'].values()):
        raise PermissionError('all explicit module train/eval modes')
    return expected
