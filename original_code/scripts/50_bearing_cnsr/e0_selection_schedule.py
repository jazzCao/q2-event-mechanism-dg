"""Compile fixed original source/target/P0 roles from JSON metadata only."""
import e0_selection_control as c
from e0_selection_worker_spec import definition

SCORE_STEPS = (0,1000,2000)
SOURCE_CASES = ('pu_clock4/ABCtoD/s0','phm09_w3/ext_IJ_A/s0','jnu_speed/A/s0')
AUXILIARY_CASES = ((SOURCE_CASES[0],0),('phm09_w3/adj_FH_G/s0',0),(SOURCE_CASES[1],1),(SOURCE_CASES[2],0))


def signature(metadata,view,case_key):
    values = {k:metadata[k] for k in ('y','g','domain','offset','factor') if k in metadata}
    values.update(view=view,case_key=case_key,shape=[len(metadata['y']),2048])
    return c.canonical(values)


def source_key(chain,step):return 'SOURCEVAL/'+chain+'/step'+str(step)
def target_key(chain,policy):return 'TARGET/'+chain+'/'+policy
def query_key(teacher,case):return 'QUERY/'+teacher+'|'+case
def factual_key(teacher):return 'AUX_FACT/'+teacher
def p0_key(chain):return 'P0_PRED/'+chain


class Compiler:
    def __init__(self,registry,catalog):
        self.registry,self.catalog = registry,catalog
        self.models = c.compile_models(registry);self.cases = {v['key']:v for v in registry['cases']}
        self.teachers = {v['key']:v for v in registry['teachers']};self.records = catalog.records
        self._metadata = {}

    def support_metadata(self,key,wanted):
        # Same sorted-record/declared-offset order as source_pack/main_source.
        result = dict(y=[],g=[],domain=[],offset=[])
        for name,offsets in sorted(wanted.items()):
            record = self.records[name];n = len(offsets)
            result['y'].extend([record['label']]*n);result['g'].extend([name]*n)
            result['domain'].extend([record['domain']]*n);result['offset'].extend(offsets)
        return result

    def source_metadata(self,key,side):
        cache = ('source',key,side)
        if cache not in self._metadata:
            _,case = self.catalog.case(key)
            self._metadata[cache] = self.support_metadata(key,{k:v[side] for k,v in case['roles'].items() if v[side]})
        return self._metadata[cache]

    def auxiliary_metadata(self,key,slot,side):
        cache = ('auxiliary',key,slot,side)
        if cache not in self._metadata:self._metadata[cache] = self.support_metadata(key,self.catalog.support(key,slot,side))
        return self._metadata[cache]

    def query_metadata(self,key,slot):
        item,case = self.catalog.case(key);groups = [g for g in item['query_groups'] if g['slot'] == slot]
        if len(groups) != 1:raise PermissionError('fixed source query group')
        group = groups[0];names = [v['record'] for v in case['bank']]
        labels = [self.records[name]['label'] for name in names for _ in range(4)]
        recordings = [name for name in names for _ in range(4)]
        domains = [self.records[name]['domain'] for name in names for _ in range(4)]
        offsets = [s for v in case['bank'] for s in v['starts']];rows = group['bank_window_rows'];n = len(rows)
        metadata = {k:[values[i] for i in rows]*3 for k,values in (('y',labels),('g',recordings),('domain',domains),('offset',offsets))}
        metadata.update(row_index=list(rows)*3,factor=[-1]*n+[0]*n+[1]*n)
        return metadata

    def target_metadata(self,key):
        item = self.cases[key];task = item['task'];cache = ('target',task['protocol'],task['target'])
        if cache not in self._metadata:
            result = dict(y=[],g=[],offset=[])
            for name,r in sorted(self.records.items()):
                if (r['protocol'],r['domain']) != (task['protocol'],task['target']):continue
                n = len(r['offsets']);result['y'].extend([r['label']]*n);result['g'].extend([name]*n);result['offset'].extend(r['offsets'])
            if not result['y']:raise ValueError('nonempty existing target metadata')
            self._metadata[cache] = result
        return self._metadata[cache]

    def qualification(self):
        models = {};pairs = [];references = []
        for key,index in AUXILIARY_CASES:
            slot = self.cases[key]['query_groups'][index]['slot']
            teachers = [t for t in self.teachers.values() if t['slot'] == slot and t['ensemble_member'] == 0]
            if len(teachers) != 1:raise PermissionError('fixed source P0 teacher')
            reference = 'AUX/'+teachers[0]['key'];base = 'P0/AUX/'+key
            left,right = base+'/continuous',base+'/resumed'
            for chain in (left,right):models[chain] = dict(stage='P0',kind='qualification',steps=10,forward_windows=240,reference_chain=reference,backend='new')
            pairs.append(dict(continuous=left,resumed=right,reference_chain=reference))
        for key in SOURCE_CASES:
            for arm in c.ARMS:
                reference = 'MAIN/'+arm+'/'+key;base = 'P0/MAIN/'+arm+'/'+key
                left,right = base+'/continuous',base+'/resumed'
                for chain in (left,right):models[chain] = dict(stage='P0',kind='qualification',steps=10,
                    forward_windows=self.models[reference]['forward_windows'],reference_chain=reference,backend='new')
                pairs.append(dict(continuous=left,resumed=right,reference_chain=reference))
                if arm == 'G_PAIRED':
                    chain = 'P0/REFERENCE/'+key
                    models[chain] = dict(stage='P0',kind='qualification',steps=10,forward_windows=self.models[reference]['forward_windows'],reference_chain=reference,backend='legacy_reference')
                    references.append(dict(chain=chain,compare_continuous=left))
        return models,pairs,references

    def compile(self):
        quals,pairs,references = self.qualification();predictions = {}
        def add(key,chain,stage,view,case,metadata,step,**extra):
            if key in predictions or not metadata['y']:raise ValueError('unique nonempty prediction role')
            predictions[key] = dict(chain=chain,stage=stage,view=view,case_key=case,windows=len(metadata['y']),
                input_identity_sha256=signature(metadata,view,case),expected_step=step,**extra)
        for chain in quals:
            info = definition(self.registry,self.models,quals,chain);case = info['case_key']
            metadata = (self.auxiliary_metadata(case,info['slot'],'train') if info['kind'] == 'auxiliary' else self.source_metadata(case,'tr'))
            subset = {k:v[:128] for k,v in metadata.items()}
            add(p0_key(chain),chain,'P0','qualification_source_training',case,subset,10,role_type='qualification')
        for teacher in self.registry['teachers']:
            chain = 'AUX/'+teacher['key'];info = definition(self.registry,self.models,{},chain);case = info['case_key']
            add(factual_key(teacher['key']),chain,'SOURCE_QUERY','auxiliary_factual_probe',case,
                self.auxiliary_metadata(case,teacher['slot'],'probe'),2000,role_type='auxiliary_factual',slot=teacher['slot'])
            for item in self.registry['cases']:
                if not any(g['slot'] == teacher['slot'] for g in item['query_groups']):continue
                add(query_key(teacher['key'],item['key']),chain,'SOURCE_QUERY','bank_clean_event0_event1',item['key'],
                    self.query_metadata(item['key'],teacher['slot']),2000,role_type='bank_query',slot=teacher['slot'])
        for chain,model in self.models.items():
            if model['kind'] != 'main':continue
            key = model['origin']
            for step in SCORE_STEPS:add(source_key(chain,step),chain,'MAIN','source_validation',key,self.source_metadata(key,'va'),step,role_type='source_validation')
            for policy in ('last','selected'):add(target_key(chain,policy),chain,'TARGET','target',key,self.target_metadata(key),policy,role_type='target',policy=policy)
        formal_windows = sum(v['steps']*v['forward_windows'] for v in self.models.values())
        p0_updates = sum(v['steps'] for v in quals.values());p0_windows = sum(v['steps']*v['forward_windows'] for v in quals.values())
        if (len(self.models),len(quals),len(predictions),p0_updates,p0_windows) != (540,29,2000,290,68060):raise AssertionError('fixed complete schedule arithmetic')
        source = sum(v['windows'] for v in predictions.values() if v['stage'] != 'TARGET')
        target = sum(v['windows'] for v in predictions.values() if v['stage'] == 'TARGET')
        if (formal_windows,source,target) != (253008000,4357030,2206224):raise AssertionError('exact unchanged full-panel workload')
        return dict(status='FIXED_SCHEDULE_NOT_EXECUTION_APPROVAL',models=self.models,qualification_models=quals,
            qualification_pairs=pairs,legacy_reference_checks=references,prediction_specs=predictions,
            auxiliary_order=[k for k,v in self.models.items() if v['kind'] == 'auxiliary'],
            main_order=[k for k,v in self.models.items() if v['kind'] == 'main'],
            case_order=[v['key'] for v in self.registry['cases']],
            totals=dict(formal_updates=1080000,qualification_updates=p0_updates,formal_training_windows=formal_windows,
                qualification_training_windows=p0_windows,source_prediction_windows=source,target_prediction_windows=target),
            training_authorized=False)
