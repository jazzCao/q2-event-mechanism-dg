"""Complete-entry closure and immutable finite proposal; stdlib only."""
import ast
from pathlib import Path
import e0_selection_control as c

LOCK_PATH = c.OUT/'RUN_LOCK_V1.json'
ANCHOR_PATH = c.OUT/'ANCHOR_METADATA_V1.json'
READINESS_PATH = c.OUT/'CPU_READINESS_V1.json'


def local_closure(entry_paths):
    """Resolve local imports statically, including lazy neural/legacy imports.

    External packages are runtime dependencies, not imported here. The original
    neural/model code is hashed without constructing its classes.
    """
    base = Path(__file__).parent;seen = set();todo = list(entry_paths)
    while todo:
        path = Path(todo.pop()).resolve()
        if path in seen:continue
        if not path.is_file() or not path.is_relative_to(base):raise PermissionError('local dependency must exist inside scripts')
        seen.add(path);tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node,ast.Import):names = [v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names = [node.module]
            for name in names:
                parts = name.split('.');candidate = base.joinpath(*parts).with_suffix('.py')
                package = base.joinpath(*parts,'__init__.py')
                if candidate.is_file():todo.append(candidate)
                elif package.is_file():todo.append(package)
    return {str(path):c.sha(path) for path in sorted(seen)}


def make_lock(registry,proposal,anchors,gate,input_sha,code_sha,regression_sha):
    p = proposal;plan = p['schedule']
    if p['fingerprint'] != c.canonical({k:v for k,v in p.items() if k != 'fingerprint'}):raise PermissionError('unchanged schedule proposal')
    if plan['models'] != c.compile_models(registry):raise PermissionError('unchanged full formal models')
    if anchors['status'] != 'ALL81_MATCHED_CORE_METADATA_ELIGIBLE_PENDING_REAL_CUDA_REFERENCE':raise PermissionError('all historical core anchors')
    lock = dict(kind=c.KIND,execution_ready=True,registry_path=str(c.OUT/'INPUT_REGISTRY_V1.json'),
        registry_fingerprint=registry['fingerprint'],registry_sha256=c.sha(c.OUT/'INPUT_REGISTRY_V1.json'),
        **{k:plan[k] for k in ('models','qualification_models','qualification_pairs','legacy_reference_checks','prediction_specs','auxiliary_order','main_order','case_order')},
        limits=p['proposed_limits'],artifact_load_limits=p['proposed_artifact_load_limits'],
        neural_seconds=p['proposed_neural_seconds'],total_seconds=p['proposed_total_seconds'],
        qualification_steps=290,repair_reserve_attempts=60,repair_reserve_automatically_enabled=False,
        input_sha256=input_sha,code_sha256=code_sha,post_neural_source_bank_audit=True,
        legacy_anchor_path=str(ANCHOR_PATH),legacy_anchor_sha256=c.sha(ANCHOR_PATH),
        legacy_metric_path=anchors['canonical_metric_file'],legacy_metric_sha256=input_sha[anchors['canonical_metric_file']],
        schedule_proposal_sha256=c.sha(c.OUT/'SCHEDULE_PROPOSAL_V1.json'),cpu_regression_sha256=regression_sha,
        gate=gate,primary='last2000',secondary='earliest_strict_source_best',
        development_panel=True,new_entity_confirmation=False,training_authorized=False,
        **{k:0 for k in c.FORBIDDEN})
    lock['fingerprint'] = c.canonical(lock);return lock


def validate_closed(lock,registry):
    """Called before any production I/O; also usable with no grant."""
    if lock.get('fingerprint') != c.canonical({k:v for k,v in lock.items() if k != 'fingerprint'}):raise PermissionError('closed lock fingerprint')
    if not lock.get('execution_ready') or lock.get('training_authorized') is not False:raise PermissionError('readiness is not authorization')
    for name in ('code_sha256','input_sha256'):
        for path,digest in lock[name].items():
            if c.sha(c.ROOT/path) != digest:raise PermissionError('closed code/input drift: '+path)
    if c.sha(c.ROOT/lock['registry_path']) != lock['registry_sha256'] or registry['fingerprint'] != lock['registry_fingerprint']:
        raise PermissionError('exact closed registry')
    if c.sha(lock['legacy_anchor_path']) != lock['legacy_anchor_sha256']:raise PermissionError('closed anchor metadata drift')
    from e0_selection_files_v2 import Catalog
    from e0_selection_schedule import Compiler
    from e0_selection_metrics import GATE
    # This module is only imported here after the actual grant by production.
    # Offline preparation tests may import NumPy, but never Torch or real arrays.
    catalog = Catalog(registry,{path:c.read(c.ROOT/path) for path in
        {registry['record_reference']['path']}|{v['case_reference']['path'] for v in registry['cases']}|
        {v['support_reference']['path'] for v in registry['slots']}})
    plan = Compiler(registry,catalog).compile()
    for field in ('models','qualification_models','qualification_pairs','legacy_reference_checks','prediction_specs','auxiliary_order','main_order','case_order'):
        if lock[field] != plan[field]:raise PermissionError('complete fixed schedule: '+field)
    if lock['gate'] != GATE or lock.get('post_neural_source_bank_audit') is not True:raise PermissionError('closed scientific gate/CPU audit')
    if lock['limits']['optimizer_attempts'] != 1080350 or lock['limits']['qualification_attempts'] != 350:
        raise PermissionError('fixed formal scope plus nonautomatic repair reserve')
    if lock['neural_seconds'] != 28800 or lock['total_seconds'] != 57600:raise PermissionError('fixed proposed8h/16h')
    if any(lock.get(k) != 0 for k in c.FORBIDDEN):raise PermissionError('no raw FFT or inverse')
    return dict(status='CLOSED_CPU_ENTRY_VALIDATED_NOT_CUDA',models=len(lock['models']),prediction_roles=len(lock['prediction_specs']),training_authorized=False)
