"""Actual-scope metadata and synthetic adversarial tests; no payloads or CUDA."""
from collections import Counter
from datetime import datetime,timedelta
import copy
import sqlite3
import unittest
import numpy as np

import event_component_execution as e
from event_component_source import make_group,assemble
from event_component_decision import ALL_ARMS,aggregate,decide
from event_component_check import aggregate as independent_aggregate
from check_rate_impulse_screen import close,stats


class ComponentEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=e.read(e.OUT/'METADATA_PREFLIGHT_V1.json')
        cls.parent=e.load(e.old.OUT/e.old_execution.NN_NAME)

    def fixture(self):
        class Meter:
            def __init__(self,ratios):self.lock={'ratios':ratios};self.counts=Counter()
            def charge(self,k,n=1):self.counts[k]+=n
        ratios=self.meta['ratios'];rt=Meter(ratios);n=32768
        base=np.sin(np.arange(n)*.01)*.001;observed=.05+np.cos(np.arange(n)*.002)
        inv={'s':np.zeros((2,48,n),np.float32),'params':np.zeros((2,48,3))}
        ctx={'observed':np.stack([observed,observed]),'scale':np.array([2.,2.])}
        g=dict(source=0,starts=[8704,13824,18944],window=2048,kernel_length=200,key='invented',origin='invented')
        crop=lambda v:np.stack([v[a:a+2048] for a in g['starts']])
        coupled=np.zeros((3,4,2048),np.float32)
        for j,q in enumerate(ratios):coupled[:,j+2]=crop(observed+2*(q*(q*base)-base)).astype(np.float32)
        anchor=dict(group_key=np.array('invented'),origin=np.array('invented'),starts=np.array(g['starts']),
            clean=crop(observed).astype(np.float32),fixed=np.zeros((3,2,2048),np.float32),interior_fixed=np.zeros((3,2,2048),np.float32),coupled=coupled)
        return rt,g,inv,ctx,anchor,{'spectrum':lambda a:a.astype(np.float32)},lambda s,p,q,l:q*base,crop,observed,base

    def test_complete_actual_schedule_and_exact_forecast(self):
        schedule=e.schedule(self.meta['cases'],self.parent,'D:/unit_scope_fixture_not_created')
        self.assertEqual(len(schedule['cases']),81);self.assertEqual(len(schedule['chains']),174)
        self.assertEqual(sum(not x['qualification'] for x in schedule['chains'].values()),162)
        self.assertEqual(len(schedule['checkpoints']),1476);self.assertEqual(len(schedule['predictions']),834)
        self.assertEqual(e.meta.forecast(schedule['cases'],self.parent),self.meta['neural_forecast'])
        self.assertEqual({v['arm'] for v in schedule['chains'].values()},set(e.ARMS))
        self.assertEqual(set(schedule['cases']),set(self.parent['cases']))

    def test_nested_archive_provenance_is_preserved(self):
        from event_component_run import archived_rows
        parent=e.read(e.old.OUT/'SCREEN_RESULT_V1.json');rows=archived_rows(parent)
        self.assertEqual(len(rows),810);self.assertEqual(sum(v['original_origin']=='ARCHIVE' for v in rows),90)
        for before,after in zip(parent['rows'],rows):
            self.assertTrue(all(after[k]==v for k,v in before.items() if k!='origin'))
            self.assertEqual(after['closed_comparison_fingerprint'],parent['fingerprint'])
            self.assertEqual(after['original_origin'],before['origin'])

    def test_every_source_role_and_query_support(self):
        from natural_mapping_probe_plan import source_case
        groups={g['key']:g for g in self.meta['groups']}
        for key,c in self.meta['cases'].items():
            original=self.parent['original_cases'][key]
            self.assertEqual(e.canonical(original),c['original_case_sha256'])
            for item in c['bank']:
                g=groups[item['group']]
                self.assertEqual(item['record'],g['origin'])
                for record in g['pair']:source_case(original,self.parent['records'],record,context=True)
        for g in groups.values():
            m=(32768-1)/2
            for q in (1.,*self.meta['ratios']):
                self.assertGreaterEqual(m+q*(min(g['starts'])-m-g['kernel_length']+1),0)
                self.assertLessEqual(m+q*(max(g['starts'])+g['window']-1-m),32767)

    def test_existing_lock_unicode_uses_original_serialization(self):
        source=e.read(e.old.OUT/'SOURCE_LOCK_V1.json')
        self.assertEqual(e.canonical({k:v for k,v in source.items() if k!='fingerprint'}),source['fingerprint'])
        self.assertNotEqual(e.canonical({'condition':'工况'}),e.meta.closed.canonical({'condition':'工况'}))

    def test_source_algebra_and_21_frontend_queries(self):
        rt,g,inv,ctx,anchor,fn,renderer,crop,x,u=self.fixture()
        a=make_group(g,inv,ctx,anchor,rt,fn,renderer)
        for k in ('clean','fixed','interior_fixed','coupled'):np.testing.assert_array_equal(a[k],anchor[k])
        for i,q in enumerate(self.meta['ratios']):
            expected=crop(x)+(q-1)*2*crop(u)
            np.testing.assert_allclose(a['amplitude_only'][:,i],expected.astype(np.float32),rtol=0,atol=1e-7)
            self.assertTrue(np.all(a['moment_mean_error'][:,i]<1e-12));self.assertTrue(np.all(a['moment_centered_rms_error'][:,i]<1e-12))
        self.assertEqual(rt.counts['renderer_calls'],3);self.assertEqual(rt.counts['source_fft_1d_transforms'],21)
        self.assertEqual(rt.counts['source_fft_points'],21*g['window'])

    def test_one_anchor_byte_or_origin_mismatch_rejects_reuse(self):
        for field in ('coupled','origin'):
            rt,g,inv,ctx,a,fn,renderer,*_=self.fixture()
            if field=='coupled':a['coupled'][0,2,0]+=np.float32(1e-5)
            else:a['origin']=np.array('wrong_record')
            with self.assertRaises(ValueError):make_group(g,inv,ctx,a,rt,fn,renderer)

    def test_bank_order_and_role_vectors_not_cherry_picked(self):
        rt,g,inv,ctx,anchor,fn,renderer,*_=self.fixture();a=make_group(g,inv,ctx,anchor,rt,fn,renderer)
        case=dict(bank=[dict(group=g['key'],record=g['origin'],domain='A',label=2)],bank_windows=3)
        bank=assemble(case,{g['key']:a})
        self.assertEqual(bank['y'].tolist(),[2,2,2]);self.assertEqual(bank['g'].tolist(),[g['origin']]*3)
        self.assertEqual(bank['offset'].tolist(),[65536+x for x in g['starts']])
        case['bank'][0]['record']='wrong'
        with self.assertRaises(ValueError):assemble(case,{g['key']:a})

    def test_target_feature_access_blocked_before_freeze(self):
        from event_component_neural import Files
        files=Files.__new__(Files);case=next(iter(self.meta['cases'].values()))
        class Runtime:
            phase='SOURCE'
            def get(self,k):return None
        files.rt=Runtime();files.lock={}
        with self.assertRaises(ValueError):files.domain(case,case['task']['target'],'target')

    def test_raw_signal_guard_and_paid_prefix_refuse_implicit_replay(self):
        from event_component_neural import Runtime,Guard
        with self.assertRaises(PermissionError):Guard(None)('open',('D:/not_read.mat','rb'))
        rt=Runtime.__new__(Runtime);rt.phase='SOURCE';rt.native={'fixture':True};rt.counts={}
        rt.lock=dict(campaign=dict(deadline=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),neural_seconds=3600),
            chains={'C':dict(steps=2,qualification=False)},limits={'optimizer_attempts':2,'formal_updates':2})
        rt.db=sqlite3.connect(':memory:');rt.db.executescript('create table meta(k text primary key,v text);create table counters(k text primary key,n integer);create table steps(chain text,step integer,done integer,primary key(chain,step));')
        try:
            rt.attempt('C',1)
            with self.assertRaises(ValueError):rt.attempt('C',1)
            with self.assertRaises(ValueError):rt.attempt('C',2)
            rt.completed('C',1);rt.attempt('C',2);rt.completed('C',2)
            with self.assertRaises(ValueError):rt.attempt('C',3)
            self.assertEqual(rt.counts['optimizer_attempts'],2)
        finally:rt.db.close()

    def fixture_rows(self,levels=None):
        levels=levels or {};rows=[]
        for c in self.meta['cases'].values():
            k=c['task']['classes']
            for arm in ALL_ARMS:
                for role in ('last','selected'):
                    n=levels.get((arm,role),levels.get(arm,90 if arm=='Q2' else 70));cm=[[0]*k for _ in range(k)]
                    for i in range(k):cm[i][i]=n;cm[i][(i+1)%k]=100-n
                    m=stats(cm,cm);m.update(window_confusion=cm,recording_confusion=cm)
                    rows.append(dict(protocol=c['task']['protocol'],fold=c['task']['fold'],seed=c['seed'],arm=arm,role=role,metrics=m))
        return rows

    def test_full_positive_fixture_and_independent_confusion_aggregate(self):
        rows=self.fixture_rows();one=decide(rows,list(self.meta['cases'].values()));two=decide(rows,list(self.meta['cases'].values()),independent_aggregate)
        close(one,two);self.assertEqual(one['status'],'FIXED_Q2_COMPONENT_SPECIFICITY_SUPPORTED')
        self.assertEqual(len(one['leave_one_target_domain']),17)
        self.assertEqual(len(one['paired_factorial']['last']),81)
        self.assertEqual(len(one['paired_control_differences']['last']),162)
        self.assertFalse(one['global_SOTA_established']);self.assertEqual(one['original_scientific_status'],'FULL_SCOPE_EVENT_OPERATOR_SUPPORTED')

    def test_control_tie_or_selected_improvement_does_not_rescue_primary(self):
        rows=self.fixture_rows({(e.ARMS[0],'last'):90,(e.ARMS[0],'selected'):60})
        r=decide(rows,list(self.meta['cases'].values()))
        self.assertEqual(r['control_evidence'][e.ARMS[0]]['status'],'COMPONENT_SPECIFICITY_UNRESOLVED')
        self.assertTrue(r['control_evidence'][e.ARMS[0]]['control_covers_Q2_mean_Acc'])
        self.assertEqual(r['status'],'FIXED_Q2_COMPONENT_SPECIFICITY_UNRESOLVED')

    def test_missing_duplicate_or_extra_seed_roles_rejected(self):
        rows=self.fixture_rows()
        for bad in (rows[:-1],rows+[copy.deepcopy(rows[0])],[dict(rows[0],seed=8),*rows[1:]]):
            with self.assertRaises(ValueError):decide(bad,list(self.meta['cases'].values()))

    def test_known_factorial_and_negative_interaction_retained(self):
        r=decide(self.fixture_rows({'F0_UNEDITED_BANK':80,'Q2_FIXED_IMPULSE':84,'A0_IMPULSE_ONLY':85,'Q2':87}),list(self.meta['cases'].values()))
        effects=r['factorial']['last']['dataset_equal']['acc']
        self.assertAlmostEqual(effects['interaction'],-.02)
        self.assertAlmostEqual(effects['timing_given_amplitude'],.02)
        self.assertAlmostEqual(effects['amplitude_given_timing'],.03)

    def test_synthetic_draws_match_original_closed_sampler(self):
        from rate_impulse_screen_neural import draw_batch
        pools=[np.arange(5)+10*i for i in range(6)];banks=[np.arange(3)+10*i for i in range(6)]
        outputs=[]
        for arm in e.ARMS:
            a,b=np.random.RandomState(77),np.random.RandomState(88)
            outputs.append(draw_batch(pools,banks,2,2,3,a,b))
        for a,b in zip(outputs[0],outputs[1]):
            if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
            else:self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
