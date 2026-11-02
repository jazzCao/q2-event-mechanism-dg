"""Actual CUDA qualification,360 new source models, common freeze and810 roles."""
from datetime import datetime
from pathlib import Path
import sys
import time
import event_full_validation_plan as p
import event_full_validation_execution as e
from run_source_rate_screen import predict,source_score,save_chain,scientific_packet,independent_metrics


def aggregate(rows,policy):
    from e0_selection_metrics import aggregate as original
    cases=sorted({(r['protocol'],r['fold'],r['seed']) for r in rows})
    return original(rows,cases,p.ARMS,policy)


def qualify(rt,files):
    from event_full_validation_neural import Chain,Data
    from source_rate_neural import exact
    if rt.get('P0_PASS'):return
    proofs=[];updates=0
    for key in p.P0_CASES:
        case=rt.lock['cases'][key];source=files.source(case);data=Data(case,source,files.bank(case),rt)
        for arm in p.ARMS:
            output=[]
            for branch in ('continuous','resume'):
                name='P0/'+key+'/'+arm+'/'+branch
                p.need(rt.db.execute('select count(*) from steps where chain=?',(name,)).fetchone()[0]==0,'no implicit partial P0 replay')
                chain=Chain(rt,case,arm,name,data)
                if branch=='continuous' and arm in ('A0_STRONG_IN_MIXUP','A1_MATCHED_E0','Q2'):data.original_batch_check(chain)
                initial=source_score(rt,chain,source,True)
                for _ in range(5):
                    chain.step(inspect=chain.step_n==0);updates+=1
                    if branch=='resume' and chain.step_n==2:
                        path=save_chain(rt,chain);before=chain.packet();del chain
                        chain=Chain(rt,case,arm,name,data,rt.reopen(path));exact(before,chain.packet(),chain.torch,chain.np,'actual disk2 all-state restore')
                final=source_score(rt,chain,source,True);save_chain(rt,chain)
                output.append((scientific_packet(chain.packet()),initial['prob'],final['prob']));torch,np=chain.torch,chain.np;del chain
            for i in range(3):exact(output[0][i],output[1][i],torch,np,'continuous5 vs actual disk2+3')
            proofs.append(dict(case_key=key,arm=arm,full_restore_exact=True,finite_nonzero_gradients=True))
        del data
    draws,times=rt.get('P0_DRAWS'),rt.get('P0_UPDATE_TIMES')
    for key in p.P0_CASES:
        v=[draws['P0/'+key+'/'+a+'/'+b] for a in p.ARMS for b in ('continuous','resume')]
        p.need(all(x==v[0] and len(x)==5 for x in v),'all five arms actual complete random draws identical')
    p.need(updates==len(times)==150,'all150 actual source CUDA updates')
    estimate=sum(times)/len(times)*720000*1.2+900
    remaining=min(rt.get('neural_start')['at_unix']+rt.lock['campaign']['neural_seconds'],datetime.fromisoformat(rt.lock['campaign']['deadline']).timestamp())-time.time()
    p.need(estimate<remaining,'measured full360-model run does not fit remaining original clocks')
    value=dict(status='PASS',updates=updates,proofs=proofs,all_draws_exact=True,estimated_remaining_seconds=estimate,remaining_seconds=remaining,at_unix=time.time())
    p.save(p.OUT/'CUDA_P0_V1.json',value)
    with rt.db:rt.set('P0_PASS',value)
    print('FULL_EVENT_CUDA_P0_PASS',updates,'estimated',round(estimate,1),'remaining',round(remaining,1),flush=True)


def train(rt,files):
    from event_full_validation_neural import Chain,Data
    from source_rate_neural import exact
    p.need(rt.get('P0_PASS'),'complete real CUDA qualification')
    for key,case in sorted(rt.lock['cases'].items()):
        source=files.source(case);data=None
        for arm in case['real_arms']:
            name='MAIN/'+key+'/'+arm
            if name in rt.finished():continue
            if data is None:data=Data(case,source,files.bank(case),rt)
            saved=[(v['step'],path) for path,v in rt.lock['checkpoints'].items() if v['chain']==name and rt.artifact(path)]
            if saved:
                step,path=max(saved);paid,done=rt.db.execute('select count(*),coalesce(sum(done),0) from steps where chain=?',(name,)).fetchone()
                p.need(paid==done==step,'no free unsaved paid-tail replay');chain=Chain(rt,case,arm,name,data,rt.reopen(path))
            else:
                p.need(rt.db.execute('select count(*) from steps where chain=?',(name,)).fetchone()[0]==0,'no lost initial-state replay')
                chain=Chain(rt,case,arm,name,data);source_score(rt,chain,source);save_chain(rt,chain)
            while chain.step_n<2000:
                chain.step(inspect=chain.step_n==0)
                if chain.step_n in e.STEPS:source_score(rt,chain,source)
                if chain.step_n%250==0:
                    path=save_chain(rt,chain)
                    if chain.step_n==1000:
                        before=chain.packet();del chain;chain=Chain(rt,case,arm,name,data,rt.reopen(path))
                        exact(before,chain.packet(),chain.torch,chain.np,'formal actual disk1000 full restore')
            rt.finish(name,dict(last=e.paths(rt.lock['campaign']['store'],name,'checkpoint',2000),selected=e.paths(rt.lock['campaign']['store'],name,'checkpoint',chain.selected),
                selected_step=chain.selected,curve=chain.curve,losses=chain.losses,actual_disk1000_restore_exact=True,completed_at_unix=time.time()))
            del chain;print('FULL_EVENT_MODEL',len(rt.finished()),'/360',name,flush=True)
        del data
    p.need(len(rt.finished())==360 and rt.counts['formal_updates']==720000,'all360 new source models before any target')
    states={}
    for name in rt.finished():
        for step in e.STEPS:
            path=e.paths(rt.lock['campaign']['store'],name,'checkpoint',step);states[path]=rt.hash(path,'checkpoint_hashes')
            p.need(states[path]==rt.artifact(path)['sha256'],'all1080 new critical source state SHA')
    for path,h in rt.lock['archive_critical_states'].items():
        p.need(rt.hash(path,'archive_checkpoint_hashes')==h,'all135 archived source states still unchanged')
    value=dict(models=405,new_models=360,reused_models=45,states=states,archive_states=rt.lock['archive_critical_states'],fingerprint=rt.lock['fingerprint'],at_unix=time.time())
    p.save(p.OUT/'FROZEN_SOURCE_V1.json',value)
    with rt.db:rt.set('SOURCE_FREEZE',value)


def evaluate(rt,files):
    from event_full_validation_neural import Chain
    p.need(rt.get('SOURCE_FREEZE'),'common new-source freeze first');rt.phase='TARGET'
    for name,item in rt.finished().items():
        spec=rt.lock['chains'][name];case=rt.lock['cases'][spec['case_key']];target=files.target(case)
        for policy in ('last','selected'):
            state=item[policy];chain=Chain(rt,case,spec['arm'],name,packet=rt.reopen(state))
            path=str(Path(rt.lock['campaign']['store'])/'neural'/name/f'target_{policy}.npz');predict(rt,chain,path,target,state);del chain
    with rt.db:rt.set('NEURAL_END',dict(at_unix=time.time()))
    rt.phase='AUDIT'


def metrics(a,classes):
    from e0_selection_metrics import from_probabilities
    one=from_probabilities(a['prob'],a['y'],a['g'],classes);two=independent_metrics(a['prob'],a['y'],a['g'],classes)
    error=max(abs(one[f]-two[f]) for f in ('acc','macro_f1','recording_ba'))
    p.need(error<1e-12 and one['window_confusion']==two['window_confusion'] and one['recording_confusion']==two['recording_confusion'],'independent saved probability metrics')
    return one,error


def audit(rt):
    import numpy as np
    import torch
    from source_rate_neural import tensor_sha
    from source_rate_screen_data import check_vectors
    from event_full_validation_decision import decide
    rows=[];scores={};error=0.;rt.phase='AUDIT'
    for path,spec in rt.lock['predictions'].items():
        a=rt.reopen(path);check_vectors(a,rt.lock['vectors'][spec['expected_key']]);case=rt.lock['cases'][spec['case_key']]
        p.need(str(a['fingerprint'])==rt.lock['fingerprint'] and str(a['chain'])==spec['chain'],'all new saved prediction identities')
        m,d=metrics(a,case['task']['classes']);error=max(error,d)
        if spec['side']=='source' and not spec['qualification']:
            score=float(np.mean([(a['prob'][a['domain']==v].argmax(1)==a['y'][a['domain']==v]).mean() for v in case['task']['sources']]))
            scores.setdefault(spec['chain'],[]).append((spec['step'],score,str(a['model_tensor_sha256'])))
        elif spec['side']=='target':
            state=rt.finished()[spec['chain']][spec['policy']]
            p.need(str(a['checkpoint_sha256'])==rt.artifact(state)['sha256'] and rt.artifact(path)['at_unix']>=rt.get('SOURCE_FREEZE')['at_unix'],'new target source-only state and timing')
            rows.append(dict(protocol=case['task']['protocol'],fold=case['task']['fold'],seed=case['seed'],arm=rt.lock['chains'][spec['chain']]['arm'],role=spec['policy'],origin='NEW',metrics=m))
    critical=0
    for name,item in rt.finished().items():
        v=sorted(scores[name]);p.need([s for s,_,_ in v]==list(e.STEPS) and max(v,key=lambda z:z[1])[0]==item['selected_step'],'all360 original earliest strict source selection')
        for step,score,modelsha in v:
            path=e.paths(rt.lock['campaign']['store'],name,'checkpoint',step);packet=rt.reopen(path);critical+=1
            p.need(packet['step']==step and packet['chain']==name and packet['fingerprint']==rt.lock['fingerprint'] and tensor_sha(packet['model'])==modelsha,'all1080 critical CPU states/probability binding')
            p.need(all(torch.isfinite(z).all() for z in packet['model'].values()),'all new model weights/buffers finite')
            p.need(rt.hash(path,'checkpoint_hashes')==rt.get('SOURCE_FREEZE')['states'][path],'all new source states unchanged')
    for name,model in rt.lock['archive_models'].items():
        v=sorted(model['detail']['curve'],key=lambda a:a['step'])
        p.need([a['step'] for a in v]==list(e.STEPS) and max(v,key=lambda a:a['score'])['step']==model['detail']['selected_step'],'original45-model source selection preserved')
    for path,b in rt.lock['archive_predictions'].items():
        a=rt.archive(b);check_vectors(a,rt.lock['vectors'][b['expected_key']]);case=rt.lock['cases'][b['case_key']]
        p.need(str(a['fingerprint'])==b['parent_fingerprint'] and str(a['chain'])==b['chain'] and int(a['step'])==b['step'] and str(a['checkpoint_sha256'])==b['checkpoint_sha256'],'all90 archived original identities/policies')
        m,d=metrics(a,case['task']['classes']);error=max(error,d)
        rows.append(dict(protocol=case['task']['protocol'],fold=case['task']['fold'],seed=case['seed'],arm=b['arm'],role=b['policy'],origin='ARCHIVE',parent_fingerprint=b['parent_fingerprint'],metrics=m))
    p.need(critical==1080 and rt.db.execute('select count(*),sum(done) from steps').fetchone()==(720150,720150),'all actual critical states and paid/done steps')
    for k,n in rt.lock['forecast'].items():
        if k in rt.lock['limits']:p.need(rt.counts.get(k)==n,'complete neural exact resource '+k)
    e.verify(rt.lock);value=decide(rows,list(rt.lock['cases'].values()),aggregate)
    value.update(execution='COMPLETE',fingerprint=rt.lock['fingerprint'],rows=rows,counts=rt.counts,metric_max_error=error,
        neural_start=rt.get('neural_start'),neural_end=rt.get('NEURAL_END'),source_freeze=rt.get('SOURCE_FREEZE')['at_unix'],audit_at_unix=time.time())
    p.save(p.OUT/'SCREEN_RESULT_V1.json',value);p.save(p.OUT/'AUDIT_V1.json',dict(status='PASS',new_probability_files=1860,archived_probability_files=90,critical_CPU_new_states=1080,metric_max_error=error,result_sha256=p.sha(p.OUT/'SCREEN_RESULT_V1.json'),at_unix=time.time()))
    with rt.db:rt.set('DONE',dict(scientific=value['status'],at_unix=time.time()));rt.set('owner',dict(state='DONE',native=rt.native,pid=rt.native['pid']))
    print('FULL_EVENT_COMPLETE',value['status'],flush=True)


def run():
    from event_full_validation_neural import Runtime,Guard,Files
    lock=e.read(p.OUT/e.NN_NAME);e.verify(lock);grant=p.read(p.OUT/'NEURAL_ACTUAL_GRANT_V1.json')
    p.need(grant['fingerprint']==lock['fingerprint'] and grant['authorization']=='standing_authorization' and grant['human_message_id']==lock['campaign']['human_message_id'],'actual fixed full neural grant')
    rt=Runtime(lock);sys.addaudithook(Guard(rt));files=Files(rt)
    try:qualify(rt,files);train(rt,files);evaluate(rt,files);audit(rt)
    except BaseException as error:
        with rt.db:rt.set('STOPPED',dict(kind=type(error).__name__,reason=str(error),at_unix=time.time()));rt.set('owner',dict(state='STOPPED',native=rt.native,pid=rt.native['pid']))
        raise
    finally:rt.db.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','run'));args=parser.parse_args()
    if args.mode=='test':
        import unittest
        import event_full_validation_test as tests
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        p.save(p.OUT/e.TEST_NAME,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,torch_imported='torch' in sys.modules,
            code_sha256=e.local_closure([Path(__file__).with_name(n) for n in e.ENTRY]),at_unix=time.time()))
        p.need(result.wasSuccessful() and 'torch' not in sys.modules,'all actual zero-neural CPU tests')
    elif args.mode=='prepare':print('FULL_EVENT_SOURCE_LOCK',e.prepare_source()['fingerprint'],flush=True)
    else:run()
