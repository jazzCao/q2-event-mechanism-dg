"""Finite source->locked CUDA->frozen evaluation coordinator; no candidate expansion."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import compatible_set_plan as p
import compatible_set_screen_plan as plan
from protected_response_journal import process_identity


def checked_git_stage(paths,message):
    """Only our exact files; refuse to mix any already staged work."""
    def git(*args,**kwargs):
        return subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],
            cwd=p.ROOT,check=True,text=True,encoding='utf8',**kwargs)
    pending=git('diff','--cached','--name-only',capture_output=True).stdout.strip()
    p.require(not pending,'unrelated staged changes; finite coordinator must not commit them')
    git('-c','core.autocrlf=false','add','--',*[str(v) for v in paths])
    actual=git('diff','--cached','--name-only',capture_output=True).stdout.splitlines()
    allowed={Path(v).resolve().relative_to(p.ROOT.resolve()).as_posix() for v in paths}
    p.require(set(actual)<=allowed and bool(actual),'exact scoped stage')
    git('-c','core.whitespace=cr-at-eol','diff','--cached','--check')
    stamp=datetime.now().astimezone().strftime('%m%d-%H%M')
    git('-c','user.name=jazzCao','-c','user.email=jazzcao@buaa.edu.cn','commit','-m',stamp+' '+message)
    env=dict(os.environ,GIT_SSH_COMMAND="ssh -oHostName=ssh.github.com -oHostKeyAlias='[ssh.github.com]:443' -oStrictHostKeyChecking=yes -oBatchMode=yes -oPort=443 -oConnectTimeout=8 -oIPQoS=none")
    git('push','origin','main',env=env,timeout=120)


def closeout(exit_receipt):
    p.require(exit_receipt['exit_code']==0,'neural producer did not complete')
    result=p.read(p.OUT/'SCREEN_RESULT_V1.json');audit=p.read(p.OUT/'AUDIT_V1.json')
    p.require(result['execution']=='COMPLETE' and audit['status']=='PASS' and audit['result_sha256']==p.sha(p.OUT/'SCREEN_RESULT_V1.json'),'actual independent saved-probability audit first')
    dbpath=p.OUT/'NEURAL_CONTROL.sqlite';db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        done=db.execute("SELECT v FROM meta WHERE k='DONE'").fetchone()
        p.require(done and db.execute('SELECT count(*) FROM finished').fetchone()[0]==50 and db.execute('SELECT count(*),sum(done) FROM steps').fetchone()==(100150,100150),'actual complete producer ledger')
    finally:db.close()
    scope=p.read(p.OUT/'CAMPAIGN_SCOPE_V2.json');p.clock(scope)
    text=['# Compatible counterfactual set: fixed pilot result','',
        'Execution COMPLETE. Scientific status '+result['status']+'.',
        '', 'Five fixed development tasks, two seeds, five arms: 50 newly trained models. Primary last2000; source-selected is secondary and cannot rescue the gate.', '',
        'Dataset-equal window accuracy / window macro-F1 / recording balanced accuracy:', '']
    for arm,value in result['primary'].items():
        v=value['dataset_equal'];text.append('- '+arm+': '+ ' / '.join(f'{100*v[k]:.4f}%' for k in ('acc','macro_f1','recording_ba')))
    text.extend(['','Failed prespecified checks:',''])
    failed=[v for v in result['checks'] if not v['passed']]
    for v in failed:text.append('- '+v['name']+f": delta {100*v['delta']:+.4f} pp, threshold {100*v['bound']:+.4f} pp.")
    if not failed:text.append('- None. A pilot signal is not SOTA confirmation.')
    text.extend(['','Complete protocol/seed/task rows, negative tasks and secondary policies are in SCREEN_RESULT_V1.json. All50 source models froze before target evaluation; the audit recomputed all310 probability roles using two confusion-matrix implementations.',
        '', 'Actual updates 100000 formal +150 qualification; neural and total clocks retain their original starts. All arrays/models are on D, small journals on C. No new raw, inverse fitting, datasets or reserved bearing reads.',
        '', 'The finite mechanical compatible set is a working-model sensitivity construction. It does not identify physical causal latents or certify true counterfactual labels. The global source-pool orthogonal control preserves cross-example geometry through Mixup in exact arithmetic; float32 mixed Gram errors on640 prespecified pairs per case are retained.',
        '', 'No full27-task expansion, parameter search, control promotion, selected-policy rescue or new candidate starts automatically. Overall publication/SOTA goal remains unachieved.'])
    with (p.OUT/'FINAL_ANALYSIS.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text)+'\n')
    names=('SOURCE_BANKS_V1.json','NEURAL_LOCK_V1.json','NEURAL_ACTUAL_GRANT_V1.json','CUDA_P0_V1.json','FROZEN_SOURCE_V1.json','SCREEN_RESULT_V1.json','AUDIT_V1.json','NEURAL_PRODUCER_EXIT_V1.json','FINAL_ANALYSIS.md')
    receipt=dict(execution='COMPLETE',scientific=result['status'],files_sha256={n:p.sha(p.OUT/n) for n in names},
        database=dict(path=str(dbpath),sha256=p.sha(dbpath)),producer=exit_receipt,counts=result['counts'],at_unix=time.time())
    p.save(p.OUT/'CLOSED_RECEIPT_V1.json',receipt)
    path=p.ROOT/'worklog/compatible_set_20261026_auto.md'
    with path.open('x',encoding='utf8',newline='\n') as f:
        f.write('# 兼容集合固定首筛自动闭合\n\n'+datetime.now().astimezone().isoformat()+'\n\n执行COMPLETE；科学'+result['status']+'。50模型/100150实际更新，全源冻结后评目标、310概率独立算术审计PASS。原起点/额度不重置，无扩项/控制改名/主表。总体方法与SOTA目标未完成。详细数值见ress_compatible_set/FINAL_ANALYSIS.md。\n')
    checked_git_stage([*[p.OUT/n for n in names[3:]],p.OUT/'CLOSED_RECEIPT_V1.json',path],
        'results: close fixed compatible-set classification pilot without expansion')


def run():
    native=p.read(p.OUT/'SOURCE_NATIVE_V1.json');scope=p.read(p.OUT/'CAMPAIGN_SCOPE_V2.json')
    p.save(p.OUT/'COORDINATOR_START_V2.json',dict(native=process_identity(os.getpid()),at_unix=time.time(),source_native=native,
        original_start=scope['actual_start'],deadline=scope['deadline']))
    while process_identity(native['pid'])==native:
        p.clock(scope);time.sleep(30)
    p.require((p.OUT/'SOURCE_BANKS_V1.json').is_file(),'source process exited without complete source banks; do not replay')
    lock=plan.compile_neural();p.verify(lock)
    p.save(p.OUT/'NEURAL_ACTUAL_GRANT_V1.json',dict(authorization='standing_authorization',human_message_id=scope['human_message_id'],
        fingerprint=lock['fingerprint'],at_unix=time.time(),all_source_generation_complete=True,all_neural_actions_before_grant=0,
        scope='only fixed50 models and real150-step P0; no expansion/automatic repair/next candidate'))
    p.save(p.OUT/'PRE_NEURAL_RECEIPT_V1.json',dict(fingerprint=lock['fingerprint'],forecast=lock['forecast'],at_unix=time.time(),
        source_native_absent=True,source_bank_sha256=p.sha(p.OUT/'SOURCE_BANKS_V1.json')))
    checked_git_stage([p.OUT/n for n in ('SOURCE_BANKS_V1.json','NEURAL_LOCK_V1.json','NEURAL_ACTUAL_GRANT_V1.json','PRE_NEURAL_RECEIPT_V1.json')],
        'research: lock complete compatible-set neural comparison before CUDA')
    p.clock(scope);p.verify(lock)
    path=p.OUT/'NEURAL_PRODUCER_V1.log'
    with path.open('x',encoding='utf8',newline='\n') as stream:
        process=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__).with_name('run_compatible_set.py')),'run'],
            cwd=p.ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        child_native=process_identity(process.pid)
        p.save(p.OUT/'NEURAL_NATIVE_V1.json',dict(native=child_native,at_unix=time.time()))
        code=process.wait()
    receipt=dict(exit_code=code,native=child_native,native_absent=process_identity(process.pid)!=child_native,at_unix=time.time())
    p.save(p.OUT/'NEURAL_PRODUCER_EXIT_V1.json',receipt)
    closeout(receipt)
    print('COMPATIBLE_COORDINATOR_COMPLETE',flush=True)


if __name__=='__main__':
    try:run()
    except BaseException as error:
        path=p.OUT/'COORDINATOR_STOPPED_V2.json'
        if not path.exists():p.save(path,dict(kind=type(error).__name__,reason=str(error),at_unix=time.time(),automatic_replay=False))
        raise
