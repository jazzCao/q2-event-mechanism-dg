"""Exact finite full-scope workload; no scientific payload decode or Torch."""
from collections import Counter
from pathlib import Path
import json
import event_full_validation_plan as p


def forecast(meta):
    original=p.parent.load_plan();cases=meta['cases'];old=set(meta['reuse']['case_keys']);anchors=set(meta['reuse']['arms'])
    p.need(p.canonical(original['records'])==meta['original_records_metadata_sha256'],'original full record metadata')
    target=Counter()
    for record in original['records'].values():target[record['protocol'],record['domain']]+=len(record['offsets'])
    forward=0;source_windows=0;target_windows=0;archive_windows=0;new=0;reuse=0;by_protocol=Counter()
    for c in cases:
        p.need(p.canonical(original['cases'][c['key']])==c['original_case_sha256'],'all81 original case bindings')
        batch=120//(len(c['task']['sources'])*c['task']['classes'])*len(c['task']['sources'])*c['task']['classes']
        nt=target[c['task']['protocol'],c['task']['target']]
        for arm in p.ARMS:
            if c['key'] in old and arm in anchors:
                reuse+=1;archive_windows+=2*nt;continue
            new+=1;by_protocol[c['task']['protocol']]+=1
            forward+=2000*6*batch;source_windows+=3*c['source_validation_windows'];target_windows+=2*nt
    per_case={c['key']:c for c in cases};proofs=len(p.P0_CASES)*len(p.ARMS)
    for key in p.P0_CASES:
        c=per_case[key];b=120//(len(c['task']['sources'])*c['task']['classes'])*len(c['task']['sources'])*c['task']['classes']
        forward+=len(p.ARMS)*10*6*b;source_windows+=len(p.ARMS)*4*32
    p.need((new,reuse)==(360,45),'all405 full-scope roles with45 explicit archives')
    return dict(real_new_models=new,reused_models=reuse,logical_models=new+reuse,new_models_by_protocol=dict(by_protocol),
        formal_updates=new*2000,qualification_updates=proofs*10,optimizer_attempts=new*2000+proofs*10,
        training_forward_windows=forward,source_prediction_windows=source_windows,target_prediction_windows=target_windows,
        archive_prediction_windows_read=archive_windows,feature_hashes=17,feature_decodes=17,bank_hashes=81,bank_decodes=81,
        neural_model_builds=4*new+3*proofs,checkpoint_writes=9*new+3*proofs,
        checkpoint_reads=6*new+proofs,checkpoint_hashes=21*new+4*proofs,
        probability_writes=5*new+4*proofs,probability_reads=5*new+4*proofs,probability_hashes=2*(5*new+4*proofs),
        archive_probability_reads=2*reuse,archive_probability_hashes=2*reuse,archive_checkpoint_hashes=3*reuse,
        critical_CPU_new_states=3*new,batch_reference_rows=12564,new_raw_reads=0,new_inverse_fits=0,protected_reads=0)


if __name__=='__main__':
    meta=p.read(p.OUT/'SOURCE_METADATA_V1.json');value=forecast(meta)
    p.need(value['optimizer_attempts']==meta['campaign']['optimizer_attempt_limit'],'whole budget and actual schedule')
    out=dict(status='EXACT_METADATA_FORECAST_NOT_EXECUTION_READY',forecast=value,
        metadata_sha256=p.sha(p.OUT/'SOURCE_METADATA_V1.json'),scientific_payload_reads=0,torch_imported='torch' in __import__('sys').modules,
        code_sha256={str(Path(__file__).resolve()):p.sha(__file__)})
    p.save(p.OUT/'NEURAL_FORECAST_V1.json',out);print(json.dumps(out['forecast'],ensure_ascii=False))
