"""Fixed two-control CUDA qualification, full source freeze, target and saved audit."""
from datetime import datetime
from pathlib import Path
import sys
import time

import event_component_execution as e
from run_source_rate_screen import predict,source_score,save_chain,scientific_packet
from event_full_validation_run import metrics


def qualify(rt,files):
    from event_component_neural import Chain,Data
    from source_rate_neural import exact
    if rt.get('P0_PASS'):return
    updates=0;proofs=[]
    for key in e.old.P0_CASES:
        case=rt.lock['cases'][key];source=files.source(case);data=Data(case,source,files.bank(case),rt)
        for arm in e.ARMS:
            outputs=[]
            for branch in ('continuous','resume'):
                name='P0/'+key+'/'+arm+'/'+branch
                e.need(rt.db.execute('select count(*) from steps where chain=?',(name,)).fetchone()[0]==0,'no implicit partial P0 replay')
                chain=Chain(rt,case,arm,name,data)
                if arm==e.ARMS[0] and branch=='continuous':data.reference_batch_check(chain)
                initial=source_score(rt,chain,source,True)
                for _ in range(5):
                    chain.step(inspect=chain.step_n==0);updates+=1
                    if branch=='resume' and chain.step_n==2:
                        path=save_chain(rt,chain);before=chain.packet();del chain
                        chain=Chain(rt,case,arm,name,data,rt.reopen(path))
                        exact(before,chain.packet(),chain.torch,chain.np,'actual component disk2 complete-state restore')
                final=source_score(rt,chain,source,True);save_chain(rt,chain)
                outputs.append((scientific_packet(chain.packet()),initial['prob'],final['prob']))
                torch,np=chain.torch,chain.np;del chain
            for i in range(3):exact(outputs[0][i],outputs[1][i],torch,np,'continuous5 versus actual disk2+3 complete model/Adam/RNG/predictions')
            proofs.append(dict(case_key=key,arm=arm,full_restore_exact=True,finite_nonzero_gradients=True))
        del data
    draws,times=rt.get('P0_DRAWS'),rt.get('P0_UPDATE_TIMES')
    for key in e.old.P0_CASES:
        records=[draws['P0/'+key+'/'+a+'/'+b] for a in e.ARMS for b in ('continuous','resume')]
        e.need(all(x==records[0] and len(x)==5 for x in records),'actual both-control draw streams identical')
    e.need(updates==len(times)==60,'all60 actual CUDA qualification steps')
    estimate=sum(times)/len(times)*324000*1.2+900
    remaining=min(rt.get('neural_start')['at_unix']+rt.lock['campaign']['neural_seconds'],datetime.fromisoformat(rt.lock['campaign']['deadline']).timestamp())-time.time()
    e.need(estimate<remaining,'measured remaining fixed work exceeds allocation; preserve P0 without replay')
    value=dict(status='PASS',updates=updates,proofs=proofs,all_draws_exact=True,reference_cases=3,
        estimated_remaining_seconds=estimate,remaining_seconds=remaining,at_unix=time.time())
    e.save(e.OUT/'CUDA_P0_V1.json',value)
    with rt.db:rt.set('P0_PASS',value)
    print('COMPONENT_CUDA_P0_PASS',updates,'estimated',round(estimate,1),flush=True)


def train(rt,files):
    from event_component_neural import Chain,Data
    from source_rate_neural import exact
    e.need(rt.get('P0_PASS'),'actual complete component P0 before formal source training')
    for key,case in sorted(rt.lock['cases'].items()):
        names=['MAIN/'+key+'/'+a for a in e.ARMS]
        if all(name in rt.finished() for name in names):continue
        source=files.source(case);data=Data(case,source,files.bank(case),rt)
        for arm,name in zip(e.ARMS,names):
            if name in rt.finished():continue
            saved=[(v['step'],path) for path,v in rt.lock['checkpoints'].items() if v['chain']==name and rt.artifact(path)]
            if saved:
                step,path=max(saved);prefix=rt.db.execute('select count(*),coalesce(sum(done),0) from steps where chain=?',(name,)).fetchone()
                e.need(prefix==(step,step),'paid unsaved formal tail needs a separately recorded finite engineering allowance')
                chain=Chain(rt,case,arm,name,data,rt.reopen(path))
            else:
                e.need(rt.db.execute('select count(*) from steps where chain=?',(name,)).fetchone()[0]==0,'lost initial state cannot replay implicitly')
                chain=Chain(rt,case,arm,name,data);source_score(rt,chain,source);save_chain(rt,chain)
            while chain.step_n<2000:
                chain.step(inspect=chain.step_n==0)
                if chain.step_n in e.STEPS:source_score(rt,chain,source)
                if chain.step_n%250==0:
                    path=save_chain(rt,chain)
                    if chain.step_n==1000:
                        before=chain.packet();del chain;chain=Chain(rt,case,arm,name,data,rt.reopen(path))
                        exact(before,chain.packet(),chain.torch,chain.np,'formal actual disk1000 full-state equality')
            rt.finish(name,dict(last=e.paths(rt.lock['campaign']['store'],name,'checkpoint',2000),
                selected=e.paths(rt.lock['campaign']['store'],name,'checkpoint',chain.selected),selected_step=chain.selected,
                curve=chain.curve,losses=chain.losses,actual_disk1000_restore_exact=True,completed_at_unix=time.time()))
            del chain;print('COMPONENT_MODEL',len(rt.finished()),'/162',name,flush=True)
        del data
    e.need(len(rt.finished())==162 and rt.counts['formal_updates']==324000,'all162 new source models before any new target')
    states={}
    for name in rt.finished():
        for step in e.STEPS:
            path=e.paths(rt.lock['campaign']['store'],name,'checkpoint',step);states[path]=rt.hash(path,'checkpoint_hashes')
            e.need(states[path]==rt.artifact(path)['sha256'],'all486 new critical states exact')
    value=dict(models=162,reused_closed_models=405,states=states,fingerprint=rt.lock['fingerprint'],at_unix=time.time())
    e.save(e.OUT/'FROZEN_SOURCE_V1.json',value)
    with rt.db:rt.set('SOURCE_FREEZE',value)


def evaluate(rt,files):
    from event_component_neural import Chain
    e.need(rt.get('SOURCE_FREEZE'),'common new-source freeze first');rt.phase='TARGET'
    # An interrupted target forward without a published output must not be silently charged twice.
    completed=sum(len(rt.lock['vectors'][v['expected_key']]['y']) for path,v in rt.lock['predictions'].items() if v['side']=='target' and rt.artifact(path))
    e.need(rt.counts.get('target_prediction_windows',0)==completed,'unsaved paid target forward needs separate recovery accounting')
    for name,info in rt.finished().items():
        case=rt.lock['cases'][rt.lock['chains'][name]['case_key']];target=None
        for policy in ('last','selected'):
            path=str(Path(rt.lock['campaign']['store'])/'neural'/name/f'target_{policy}.npz')
            if rt.artifact(path):continue
            if target is None:target=files.target(case)
            state=info[policy];chain=Chain(rt,case,rt.lock['chains'][name]['arm'],name,packet=rt.reopen(state))
            predict(rt,chain,path,target,state);del chain
    with rt.db:rt.set('NEURAL_END',dict(at_unix=time.time()))
    rt.phase='AUDIT'


def archived_rows(parent):
    return [dict(r,origin='CLOSED_405_MODEL_METRICS',original_origin=r['origin'],
                 closed_comparison_fingerprint=parent['fingerprint']) for r in parent['rows']]


def audit(rt):
    import numpy as np
    import torch
    from source_rate_neural import tensor_sha
    from source_rate_screen_data import check_vectors
    from event_component_decision import decide
    rows=[];scores={};max_error=0.;rt.phase='AUDIT'
    for path,spec in rt.lock['predictions'].items():
        v=rt.reopen(path);check_vectors(v,rt.lock['vectors'][spec['expected_key']]);case=rt.lock['cases'][spec['case_key']]
        e.need(str(v['fingerprint'])==rt.lock['fingerprint'] and str(v['chain'])==spec['chain'],'all actual prediction identities')
        m,error=metrics(v,case['task']['classes']);max_error=max(max_error,error)
        if spec['side']=='source' and not spec['qualification']:
            score=float(np.mean([(v['prob'][v['domain']==d].argmax(1)==v['y'][v['domain']==d]).mean() for d in case['task']['sources']]))
            scores.setdefault(spec['chain'],[]).append((spec['step'],score,str(v['model_tensor_sha256'])))
        elif spec['side']=='target':
            state=rt.finished()[spec['chain']][spec['policy']]
            e.need(str(v['checkpoint_sha256'])==rt.artifact(state)['sha256'] and rt.artifact(path)['at_unix']>=rt.get('SOURCE_FREEZE')['at_unix'],'target prediction bound to frozen source-only policy')
            rows.append(dict(protocol=case['task']['protocol'],fold=case['task']['fold'],seed=case['seed'],arm=rt.lock['chains'][spec['chain']]['arm'],role=spec['policy'],origin='NEW',metrics=m))
    critical=0
    for name,info in rt.finished().items():
        values=sorted(scores[name]);e.need([s for s,_,_ in values]==list(e.STEPS) and max(values,key=lambda z:z[1])[0]==info['selected_step'],'earliest strict source selection for all162')
        e.need(info['actual_disk1000_restore_exact'] and [v['step'] for v in info['losses']]==list(range(100,2001,100)),'complete source loss log and actual1000 restore')
        for step,score,model_sha in values:
            path=e.paths(rt.lock['campaign']['store'],name,'checkpoint',step);packet=rt.reopen(path);critical+=1
            e.need(packet['step']==step and packet['chain']==name and packet['fingerprint']==rt.lock['fingerprint'] and tensor_sha(packet['model'])==model_sha,'all486 actual CPU critical states')
            e.need(all(torch.isfinite(z).all() for z in packet['model'].values()),'all model parameters and buffers finite')
            e.need(rt.hash(path,'checkpoint_hashes')==rt.get('SOURCE_FREEZE')['states'][path],'all frozen state SHA preserved')
    e.need(critical==486 and rt.db.execute('select count(*),sum(done) from steps').fetchone()==(324060,324060),'all fixed paid steps including60 real P0')
    for key,limit in rt.lock['limits'].items():e.need(rt.counts.get(key,0)==limit,'exact full work counter '+key)
    e.need(e.sha(rt.lock['parent_result_path'])==rt.lock['parent_result_sha256'],'same closed405-model metric archive')
    parent=e.read(rt.lock['parent_result_path']);e.need(parent['status']=='FULL_SCOPE_EVENT_OPERATOR_SUPPORTED' and len(parent['rows'])==810,'original all810 archived roles retained')
    rows += archived_rows(parent)
    e.verify(rt.lock);result=decide(rows,list(rt.lock['cases'].values()))
    for policy,parent_key in (('primary','primary'),('secondary','secondary')):
        for arm in e.old.ARMS:
            for weight in ('dataset_equal','task_equal'):
                for field in ('acc','macro_f1','recording_ba'):
                    e.need(abs(result[policy][arm][weight][field]-parent[parent_key][arm][weight][field])<1e-12,'original archived aggregate unchanged')
    result.update(execution='COMPLETE',fingerprint=rt.lock['fingerprint'],rows=rows,counts=rt.counts,metric_max_error=max_error,
        neural_start=rt.get('neural_start'),neural_end=rt.get('NEURAL_END'),source_freeze=rt.get('SOURCE_FREEZE')['at_unix'],audit_at_unix=time.time())
    e.save(e.OUT/'SCREEN_RESULT_V1.json',result)
    e.save(e.OUT/'AUDIT_V1.json',dict(status='PASS',new_probability_files=834,critical_CPU_new_states=486,
        closed_metric_roles_reused=810,new_metric_roles=324,metric_max_error=max_error,result_sha256=e.sha(e.OUT/'SCREEN_RESULT_V1.json'),at_unix=time.time()))
    with rt.db:rt.set('DONE',dict(scientific=result['status'],at_unix=time.time()));rt.set('owner',dict(state='DONE',native=rt.native,pid=rt.native['pid']))
    print('COMPONENT_COMPLETE',result['status'],flush=True)


def run():
    from event_component_neural import Runtime,Files,Guard
    lock=e.load(e.OUT/e.NN_NAME);e.verify(lock);grant=e.read(e.OUT/'NEURAL_ACTUAL_GRANT_V1.json')
    e.need(grant['authorization']=='standing_authorization' and grant['human_message_id']==e.meta.closed.AUTH and grant['fingerprint']==lock['fingerprint'],'actual complete neural grant')
    rt=Runtime(lock);sys.addaudithook(Guard(rt));files=Files(rt)
    try:
        e.need(rt.get('NEURAL_END') is None,'partially repeated post-neural audit needs its own bounded recovery scope')
        qualify(rt,files)
        if not rt.get('SOURCE_FREEZE'):train(rt,files)
        evaluate(rt,files);audit(rt)
    except BaseException as error:
        with rt.db:rt.set('STOPPED',dict(kind=type(error).__name__,reason=str(error),at_unix=time.time()));rt.set('owner',dict(state='STOPPED',native=rt.native,pid=rt.native['pid']))
        raise
    finally:rt.db.close()


if __name__=='__main__':run()
