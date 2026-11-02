"""Future MOCD source interface: metadata/synthetic only until separately approved.

No Torch import, model, cache loader, execution entry or implicit authority.
The historical qualified engine and all historical fingerprints remain unchanged.
"""
from types import SimpleNamespace
import numpy as np
import event_author_sources as pu
import event_native_main_sources as other

PROTOCOLS={'pu_clock4':(3,3),'phm09_w3':(2,8),'jnu_speed':(2,4)}


def source_metadata(records,case):
    task=case['task'];protocol=task['protocol']
    if protocol not in PROTOCOLS or type(case['seed']) is not int or case['seed'] not in (0,1,2):raise ValueError('fixed three protocols and actual seed0/1/2')
    domains,classes=PROTOCOLS[protocol]
    if (len(task['sources']),task['classes'])!=(domains,classes) or len(set(task['sources']))!=domains or task['target'] in task['sources']:raise ValueError('fixed source/class/target declaration')
    for key in case['roles']:
        if key not in records:raise ValueError('unknown source recording')
        record=records[key]
        if record.get('bearing') in pu.PROTECTED:raise PermissionError('protected bearing identity')
        if record['protocol']!=protocol or record['domain'] not in task['sources']:raise PermissionError('no target or foreign record in source roles')
        if type(record['window']) is not int or record['window']<=0:raise ValueError('physical full-window length')
        if any(type(o) is not int or o<0 for o in record['offsets']):raise ValueError('actual nonnegative window offsets')
    result=(pu.inventory if protocol=='pu_clock4' else other.inventory)(records,case)
    if any(min(v['tr']['class_counts'])<5 for v in result.values()):raise ValueError('five real training candidates per domain/class, no borrowing')
    return result


def role_indices(records,case,domain,side):
    implementation=pu.role_indices if case['task']['protocol']=='pu_clock4' else other.role_indices
    return implementation(records,case,domain,side)


class CompleteSourceRoles:
    """Same engine get()/allowed/classes interface; explicit split semantics."""
    def __init__(self,case,train,val,fingerprint,inventory):
        self.allowed=tuple(case['task']['sources']);self.forbidden={case['task']['target']}
        self.train=train;self.val=val;self.classes=case['task']['classes'];self.fingerprint=fingerprint
        self.inventory=inventory
        self.validation_rule='TEMPORAL_BLOCK_FULL_INTERVAL_DISJOINT' if case['task']['protocol']=='jnu_speed' else 'GROUPED_RECORDING_DISJOINT'
        if not isinstance(fingerprint,str) or not fingerprint or set(train)!=set(self.allowed) or set(val)!=set(self.allowed):raise PermissionError('fully bound source roles')
    def get(self,domain,role):
        if domain not in self.allowed or domain in self.forbidden or role not in ('train','val'):raise PermissionError('declared source train/validation only')
        return (self.train if role=='train' else self.val)[domain]


def bind_sources(records,case,arrays,fingerprint,tensor_factory=None):
    """Real runner must provide already-verified arrays and torch.from_numpy later.

    Fixtures pass NumPy identity. No cap, resplit, record renaming or I/O exists.
    JNU retains the true recording key shared across disjoint physical intervals.
    """
    inventory=source_metadata(records,case)
    if set(arrays)!=set(('tr','va')):raise ValueError('source-only two-way input')
    build=(lambda x:x) if tensor_factory is None else tensor_factory;train={};val={}
    for side,destination in (('tr',train),('va',val)):
        a=arrays[side]
        if set(a)!=set(('x','y','g','domain')):raise ValueError('exact source array schema')
        if not all(isinstance(v,np.ndarray) for v in a.values()):raise ValueError('explicit NumPy source arrays')
        n=len(a['y'])
        if a['x'].dtype!=np.float32 or a['x'].shape!=(n,2048) or not np.isfinite(a['x']).all():raise ValueError('finite FP32 logFFT2048, no alternate preprocessing')
        if a['y'].shape!=(n,) or a['y'].dtype.kind not in 'iu' or any(a[k].shape!=(n,) or a[k].dtype.kind not in 'US' for k in ('g','domain')):raise ValueError('integer labels and string role provenance')
        if set(a['domain'])!=set(case['task']['sources']):raise ValueError('all and only actual source domains')
        for d in case['task']['sources']:
            expected=role_indices(records,case,d,side);mask=a['domain']==d
            if not np.array_equal(a['y'][mask],expected['y']) or not np.array_equal(a['g'][mask],expected['g']):raise ValueError('no row reorder, label or record substitution')
            destination[d]=SimpleNamespace(domain=d,role='train' if side=='tr' else 'val',
                x=build(np.ascontiguousarray(a['x'][mask])),y=expected['y'],groups=expected['g'],indices=expected['indices'],offsets=expected['offset'])
    return CompleteSourceRoles(case,train,val,fingerprint,inventory)


def random_stream_spec(seed):
    if type(seed) is not int or seed not in (0,1,2):raise ValueError('actual seed0/1/2 only')
    return dict(model_initialization=seed,global_python=seed,global_numpy=seed,global_torch_cpu=seed,global_torch_cuda=seed,
        main_batch_RandomState=seed,matching_matrix_and_sampling_RandomState=seed,
        matching_sampling_initial_state='CONTINUE_AFTER_strict_matching_indices_CONSTRUCTION_NO_RESET',
        same_task_seed_profiles_share_stream_initialization=True,
        seed0_legacy_stream_semantics_preserved=seed==0,
        future_engine_must_bind_and_restore_all_stream_states=True,
        status='POLICY_ONLY_NOT_YET_NEURAL_WIRED_OR_CUDA_QUALIFIED')


def main_index_trace(source_lengths,seed,steps):
    """Pure index oracle for the unchanged 32-draw StageB sampling algorithm."""
    random_stream_spec(seed)
    if len(source_lengths)<2 or any(type(n) is not int or n<=0 for n in source_lengths) or type(steps) is not int or steps<0:raise ValueError('nonempty actual source pools and steps')
    rng=np.random.RandomState(seed);trace=[]
    for _ in range(steps):
        pairs=[]
        for _ in range(32):
            domain=int(rng.randint(len(source_lengths)));index=int(rng.randint(source_lengths[domain]));pairs.append((domain,index))
        trace.append(pairs)
    return trace,rng.get_state()
