"""Whole scope, source semantics, counterfactual algebra and durable-control tests.

Renderer/front-end injections are unit fixtures, not a new physical qualification
or actual source FFT. No scientific payload or neural library is opened.
"""
from collections import Counter
from datetime import datetime,timedelta
import copy
import math
import sqlite3
import unittest
import numpy as np
import event_full_validation_plan as p
import event_full_validation_execution as e
from event_full_validation_schedule import forecast
from event_full_validation_source import make_group,compatible
from event_full_validation_decision import decide
from event_full_validation_run import aggregate
from event_full_validation_check import aggregate as independent_aggregate
from check_rate_impulse_screen import stats,close


class FullScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=p.read(p.OUT/'SOURCE_METADATA_V1.json');cls.original=p.parent.load_plan()

    def test_complete_original_source_roles_and_support(self):
        m=self.meta;self.assertEqual(len(m['cases']),81);self.assertEqual(len(m['groups']),3372)
        from natural_mapping_probe_plan import source_case
        for c in m['cases']:
            case=self.original['cases'][c['key']];self.assertEqual(p.canonical(case),c['original_case_sha256'])
            for pair in case['pairs'].values():
                for key in pair:source_case(case,self.original['records'],key,context=True)
        for g in m['groups']:
            self.assertEqual(g['kernel_length'],math.ceil(.004*g['fs']))
            for q in (1.,*p.Q):
                mid=(32768-1)/2
                self.assertGreaterEqual(mid+q*(min(g['starts'])-mid-g['kernel_length']+1),0)
                self.assertLessEqual(mid+q*(max(g['starts'])+g['window']-1-mid),32767)

    def test_complete_neural_schedule_and_forecast(self):
        m=self.meta;s=e.schedule(m,self.original,'D:/unit_metadata_only/not_created')
        f=forecast(m);self.assertEqual(f,p.read(p.OUT/'NEURAL_FORECAST_V1.json')['forecast'])
        self.assertEqual(sum(not v['qualification'] for v in s['chains'].values()),360)
        self.assertEqual(len(s['checkpoints']),3285);self.assertEqual(len(s['predictions']),1860)
        self.assertEqual(sum(len(c['real_arms']) for c in s['cases'].values()),360)
        self.assertEqual(set(s['cases']),set(self.original['cases']))

    def test_source_algebra_with_injected_renderer_not_science(self):
        class Mock:
            lock={'ratios':p.Q}
            counts=Counter()
            def charge(self,k,n=1):self.counts[k]+=n
        rt=Mock();n=32768;t=np.arange(n,dtype=float);b=.001*(1+np.sin(t*.01));observed=np.linspace(-.2,.2,n)
        inv={'s':np.zeros((2,48,n),dtype=np.float32),'params':np.zeros((2,48,3),dtype=float)}
        ctx={'observed':np.stack([observed,observed]),'scale':np.array([2.,2.])}
        g=dict(source=0,starts=[12000,15000,18000],window=2048,kernel_length=200,key='synthetic',origin='synthetic')
        fn={'spectrum':lambda a:a.astype(np.float32)}
        a=make_group(g,inv,ctx,rt,fn,renderer=lambda s,params,q,length:q*b)
        crop=lambda x:np.stack([x[i:i+2048] for i in g['starts']]).astype(np.float32)
        self.assertTrue(np.array_equal(a['clean'],crop(observed)))
        for i,q in enumerate(p.Q):
            np.testing.assert_allclose(a['coupled'][:,i],crop(observed+2*(q*q-1)*b),rtol=0,atol=3e-8)
            branch=a['fixed'][:,i] if i<2 else a['interior_fixed'][:,i-2]
            np.testing.assert_allclose(branch,crop(observed+2*(q-1)*b),rtol=0,atol=3e-8)
        self.assertEqual(rt.counts['renderer_calls'],5);self.assertEqual(rt.counts['source_fft_1d_transforms'],27)

    def test_reuse_rejects_semantic_or_numeric_mismatch(self):
        a={k:np.array([0,1,2]) for k in ('y','g','domain','offset')}
        a.update(clean=np.zeros((3,2048),np.float32),fixed=np.zeros((3,2,2048),np.float32),coupled=np.zeros((3,4,2048),np.float32))
        old={k:v.copy() for k,v in a.items()};old['coupled']=np.zeros((3,8,2048),np.float32)
        self.assertEqual(compatible(a,old)['status'],'EXACT_ORIGINAL_SOURCE_INPUTS')
        old['y'][0]=9
        with self.assertRaises(ValueError):compatible(a,old)
        old['y'][0]=0;old['coupled'][0,2,4]=np.float32(1e-6)
        with self.assertRaises(ValueError):compatible(a,old)

    def test_source_role_target_is_forbidden(self):
        from natural_mapping_probe_plan import source_case
        c=next(iter(self.original['cases'].values()))
        key=next(k for k,v in self.original['records'].items() if (v['protocol'],v['domain'])==(c['task']['protocol'],c['task']['target']))
        with self.assertRaises(ValueError):source_case(c,self.original['records'],key,context=True)

    def test_archive_cannot_open_before_common_new_source_freeze(self):
        from event_full_validation_neural import Runtime
        rt=Runtime.__new__(Runtime)
        for phase,frozen in (('SOURCE',True),('AUDIT',None)):
            rt.phase=phase;rt.get=lambda k:frozen
            with self.assertRaises(ValueError):rt.archive({'path':'D:/never_opened_unit_fixture.npz'})

    def test_durable_paid_prefix_refuses_implicit_replay(self):
        from source_rate_screen_runtime import Runtime
        rt=Runtime.__new__(Runtime);rt.phase='SOURCE';rt.native={'unit_test':True};rt.counts={}
        rt.lock=dict(campaign={'deadline':(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),'neural_seconds':3600},chains={'C':{'steps':2,'qualification':False}},limits={'optimizer_attempts':2,'formal_updates':2})
        rt.db=sqlite3.connect(':memory:');rt.db.executescript('create table meta(k text primary key,v text);create table counters(k text primary key,n integer);create table steps(chain text,step integer,done integer,primary key(chain,step));')
        try:
            rt.attempt('C',1);self.assertEqual(rt.db.execute('select step,done from steps').fetchall(),[(1,0)])
            with self.assertRaises(ValueError):rt.attempt('C',1)
            with self.assertRaises(ValueError):rt.attempt('C',2)
            rt.completed('C',1);rt.attempt('C',2);rt.completed('C',2)
            with self.assertRaises(ValueError):rt.attempt('C',3)
            self.assertEqual(rt.counts['optimizer_attempts'],2)
        finally:rt.db.close()

    def fixture_rows(self,level=None):
        levels=dict(zip(p.ARMS,(60,61,62,63,80)));levels.update(level or {});rows=[]
        for c in self.meta['cases']:
            k=c['task']['classes']
            for arm in p.ARMS:
                for role in ('last','selected'):
                    n=levels.get((arm,role),levels[arm]);cm=[[0]*k for _ in range(k)]
                    for i in range(k):cm[i][i]=n;cm[i][(i+1)%k]=100-n
                    m=stats(cm,cm);m.update(window_confusion=cm,recording_confusion=cm)
                    rows.append(dict(protocol=c['task']['protocol'],fold=c['task']['fold'],seed=c['seed'],arm=arm,role=role,metrics=m))
        return rows

    def test_two_independent_aggregators_and_full_positive_fixture(self):
        rows=self.fixture_rows();one=decide(rows,self.meta['cases'],aggregate);two=decide(rows,self.meta['cases'],independent_aggregate)
        close(one,two);self.assertEqual(one['status'],'FULL_SCOPE_EVENT_OPERATOR_SUPPORTED')
        self.assertFalse(one['global_SOTA_established']);self.assertEqual(len(one['leave_one_target_domain']),17)

    def test_component_control_equality_is_unresolved(self):
        value=decide(self.fixture_rows({'Q2_FIXED_IMPULSE':80}),self.meta['cases'],aggregate)
        self.assertEqual(value['status'],'FULL_SCOPE_PERFORMANCE_SUPPORTED_COMPONENT_UNRESOLVED')

    def test_secondary_policy_cannot_rescue_primary_failure(self):
        rows=self.fixture_rows({('Q2','last'):40,('Q2','selected'):99})
        self.assertEqual(decide(rows,self.meta['cases'],aggregate)['status'],'FULL_SCOPE_EVENT_OPERATOR_NOT_SUPPORTED')

    def test_missing_or_duplicate_role_cannot_pass(self):
        rows=self.fixture_rows()
        with self.assertRaises(ValueError):decide(rows[:-1],self.meta['cases'],aggregate)
        with self.assertRaises(ValueError):decide(rows[:-1]+[rows[0]],self.meta['cases'],aggregate)


if __name__=='__main__':unittest.main()
