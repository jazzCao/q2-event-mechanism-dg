"""Role-gated reuse of CachedStore, injectable without Torch or real arrays.

The real base is imported only after Runtime has accepted actual new authority.
This does not decode raw signals, recompute FFTs or fit new inverse models.
"""
from pathlib import Path
import time
from pcr_campaign_core import canonical,read,sha,write_new
import mocd_complete_source_adapter as adapter
import mocd_complete_control as control


def source_cache_class(base):
    class CompleteCache(base):
        def __init__(self,rt):
            super().__init__(rt);self.accesses=rt.get('CACHE_ACCESSES') or []
            self.seen={(v.get('case_key'),v['role'],v['domain'],v['phase']) for v in self.accesses if v.get('outcome')=='PASS'}

        def domain(self,protocol,domain,task,role):
            rt=self.rt
            if rt.active_case is None or task!=rt.active_case['task']:raise PermissionError('exact active source case before any cache use')
            rt.require_case(rt.active_case)
            if protocol!=task['protocol'] or (role=='source' and domain not in task['sources']) or (role=='target' and domain!=task['target']):raise PermissionError('exact protocol and source/target cache domain')
            if role=='target':rt.require_target(rt.active_case)
            elif role!='source' or rt.phase not in ('SOURCE_ADMISSION','QUALIFICATION','SOURCE') or rt.get('SOURCE_FREEZE') is not None:raise PermissionError('source-only cache phase, no source reads after freeze')
            before=len(self.accesses);ok=False
            try:
                result=super().domain(protocol,domain,task,role);ok=True;return result
            finally:
                # Historical loader removes this permission on success only.
                # A failed open/shape check must not leave a stale role permit.
                binding=getattr(self,'binding',{}).get((protocol,domain))
                if binding is not None:rt.revoke(binding['path'])
                # The old base appends once per recording while a complete
                # domain cache is reused. Keep one successful domain/role/case/
                # phase access, not megabytes of identical full-JSON rewrites.
                # Manifest + cache SHA still bind every actual record/offset.
                self.accesses[before:]=[]
                key=(control.m.case_key(rt.active_case),role,domain,rt.phase)
                if key not in self.seen or not ok:
                    self.accesses.append(dict(case_key=key[0],protocol=protocol,fold=task['fold'],seed=rt.active_case['seed'],
                        domain=domain,role=role,phase=rt.phase,outcome='PASS' if ok else 'FAILED',at_unix=time.time()))
                    if ok:self.seen.add(key)

        def flush_accesses(self):self.rt.put('CACHE_ACCESSES',self.accesses)

    return CompleteCache


def create(rt):
    if not rt.authority_verified:raise PermissionError('new actual finite authorization before real cached payloads/Torch imports')
    rt.clock()
    from clock_factual_data import CachedStore
    return source_cache_class(CachedStore)(rt)


def actual_sources(rt,cache,case,tensor_factory):
    rt.set_case(case)
    arrays=cache.source_case(case);cache.flush_accesses()
    return adapter.bind_sources(rt.scope['records'],case,arrays,case['new_source_fingerprint'],tensor_factory=tensor_factory)


def admit(rt,cache,tensor_factory):
    """All81 full source supports; an old same-lock receipt may only be reused."""
    old=rt.get('SOURCE_ADMISSION')
    if old is not None:
        if old.get('lock_fingerprint')!=rt.lock['fingerprint'] or old.get('cases')!=[control.m.case_key(v) for v in rt.scope['cases']]:raise PermissionError('all81 same-lock source admission')
        return old
    rt.phase='SOURCE_ADMISSION';rows=[]
    for case in rt.scope['cases']:
        rt.clock();source=actual_sources(rt,cache,case,tensor_factory)
        expected=case['source_inventory']
        for domain in source.allowed:
            for name,side in (('train','tr'),('val','va')):
                part=source.get(domain,name)
                if len(part.y)!=expected[domain][side]['windows']:raise PermissionError('all real full source windows required')
        rows.append(dict(case_key=control.m.case_key(case),source_fingerprint=source.fingerprint,source_validation_rule=source.validation_rule,
            training_windows=sum(len(source.get(d,'train').y) for d in source.allowed),validation_windows=sum(len(source.get(d,'val').y) for d in source.allowed)))
    value=dict(status='ALL81_COMPLETE_CACHED_SOURCE_SUPPORT_PASS_NO_NETWORK',lock_fingerprint=rt.lock['fingerprint'],
        cases=[v['case_key'] for v in rows],source_rows=rows,counters=rt.count(),at_unix=time.time())
    write_new(control.OUT/'SOURCE_ADMISSION.json',value);rt.put('SOURCE_ADMISSION',value)
    return value
