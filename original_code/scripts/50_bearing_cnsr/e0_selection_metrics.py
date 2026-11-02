"""Independent confusion arithmetic and fixed development decision; no Torch."""
import numpy as np

FIELDS = ('acc','macro_f1','recording_ba')
WEIGHTS = ('dataset_equal','task_equal')
GATE = dict(version='E0_SELECTION_DEVELOPMENT_GATE_V1',primary='last',secondary='selected',
    minimum_acc_over_E0_and_IN=.003,protocol_acc_floor=-.002,all_three_seed_aggregate_positive=True,
    aggregate_macroF1_and_recordingBA_floor=0.,minimum_acc_over_R_and_U=.0015,
    minimum_protocols_with_nonalias_control=2,uniform_fp32_energy_tolerance=1e-4,
    numerical_comparison_epsilon=1e-12,target_policy_or_parameter_search=False,
    development_pass_is_not_causal_identification_or_publication_ready=True)


def confusion_stats(cm):
    a = np.asarray(cm)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.dtype.kind not in 'iu' or (a < 0).any() or not a.sum():
        raise ValueError('nonempty square integer confusion matrix')
    support,predicted = a.sum(1),a.sum(0);tp = np.diag(a)
    recall = np.divide(tp,support,out=np.zeros(len(a)),where=support > 0)
    f1 = np.divide(2*tp,support+predicted,out=np.zeros(len(a)),where=support+predicted > 0)
    return dict(acc=float(tp.sum()/a.sum()),macro_f1=float(f1.mean()),ba=float(recall[support > 0].mean()),
                recall=recall.tolist(),support=support.tolist())


def from_confusions(window,recording):
    w,r = confusion_stats(window),confusion_stats(recording)
    if len(window) != len(recording):raise ValueError('same declared classes')
    return dict(acc=w['acc'],macro_f1=w['macro_f1'],recording_ba=r['ba'],windows=int(np.asarray(window).sum()),
        recordings=int(np.asarray(recording).sum()),window_confusion=np.asarray(window).tolist(),
        recording_confusion=np.asarray(recording).tolist())


def from_probabilities(prob,y,g,classes):
    p,y,g = np.asarray(prob),np.asarray(y),np.asarray(g)
    if p.shape != (len(y),classes) or not len(y) or y.shape != g.shape or y.dtype.kind not in 'iu' or np.any((y < 0)|(y >= classes)):
        raise ValueError('exact classification vectors')
    if p.dtype.kind != 'f' or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1),1.,atol=1e-5,rtol=0):
        raise ValueError('finite saved class probabilities')
    w = np.zeros((classes,classes),np.int64);r = np.zeros_like(w)
    np.add.at(w,(y,p.argmax(1)),1)
    for recording in sorted(set(g.tolist())):
        selected = g == recording;truth = np.unique(y[selected])
        if len(truth) != 1:raise ValueError('recording label must be constant')
        r[int(truth[0]),int(p[selected].mean(0).argmax())] += 1
    return from_confusions(w,r)


def row_key(row):return row['protocol'],row['fold'],int(row['seed']),row['arm'],row['role']


def aggregate(rows,case_keys,methods,policy,*,seed=None):
    lookup = {}
    for row in rows:
        key = row_key(row)
        if key in lookup:raise ValueError('duplicate metric role')
        lookup[key] = row['metrics']
    protocols = sorted({p for p,f,s in case_keys});result = {}
    for arm in methods:
        tasks = {}
        for p,f,s in case_keys:
            if seed is not None and s != seed:continue
            key = (p,f,s,arm,policy)
            if key not in lookup:raise ValueError('complete unchanged panel required: '+str(key))
            tasks.setdefault((p,f),[]).append(lookup[key])
        means = {key:{field:float(np.mean([v[field] for v in values])) for field in FIELDS} for key,values in tasks.items()}
        per_protocol = {p:{field:float(np.mean([v[field] for (pp,f),v in means.items() if pp == p])) for field in FIELDS} for p in protocols}
        result[arm] = dict(per_protocol=per_protocol,
            dataset_equal={field:float(np.mean([v[field] for v in per_protocol.values()])) for field in FIELDS},
            task_equal={field:float(np.mean([v[field] for v in means.values()])) for field in FIELDS})
    return result


def decide(rows,case_keys,baseline_methods,control_audits):
    """Fixed primary gate; secondary scores are reported, never rescue failure."""
    methods = tuple(baseline_methods)+('G_PAIRED','R_COUNT','U_ENERGY')
    if len(set(methods)) != len(methods) or not {'E0_EVENTS','B1_IN_MIX','S0_INTACT'} <= set(baseline_methods):raise ValueError('all matched anchors required')
    if len(case_keys) != len(set(case_keys)) or len({s for p,f,s in case_keys}) != 3:raise ValueError('unchanged three-seed cases')
    primary = aggregate(rows,case_keys,methods,'last');secondary = aggregate(rows,case_keys,methods,'selected')
    eps = GATE['numerical_comparison_epsilon'];performance = [];specificity = [];details = []
    for anchor in ('E0_EVENTS','B1_IN_MIX'):
        for weighting in WEIGHTS:
            difference = {field:primary['G_PAIRED'][weighting][field]-primary[anchor][weighting][field] for field in FIELDS}
            performance.append(difference['acc']+eps >= GATE['minimum_acc_over_E0_and_IN'])
            performance.extend(difference[field]+eps >= 0 for field in ('macro_f1','recording_ba'))
            details.append(dict(anchor=anchor,weighting=weighting,difference=difference))
        for protocol in primary[anchor]['per_protocol']:
            performance.append(primary['G_PAIRED']['per_protocol'][protocol]['acc']-primary[anchor]['per_protocol'][protocol]['acc']+eps >= GATE['protocol_acc_floor'])
    per_seed = {}
    for seed in sorted({s for p,f,s in case_keys}):
        values = aggregate(rows,case_keys,methods,'last',seed=seed);per_seed[str(seed)] = values
        for anchor in ('E0_EVENTS','B1_IN_MIX'):
            for weighting in WEIGHTS:performance.append(values['G_PAIRED'][weighting]['acc']-values[anchor][weighting]['acc'] > eps)
    for control in ('R_COUNT','U_ENERGY'):
        for weighting in WEIGHTS:
            specificity.append(primary['G_PAIRED'][weighting]['acc']-primary[control][weighting]['acc']+eps >= GATE['minimum_acc_over_R_and_U'])
    protocol_control = set()
    all_energy = True
    for audit in control_audits:
        if audit.get('all_uniform_fp32_energy_groups_qualified') is not True:all_energy = False
        if (audit.get('actual_random_mask_hamming',0) > 0 and audit.get('random_gated_data_alias') is False
                and audit.get('uniform_gated_data_alias') is False):protocol_control.add(audit['case_key'].split('/')[0])
    expected_cases = {p+'/'+f+'/s'+str(seed) for p,f,seed in case_keys}
    if len(control_audits) != len(case_keys) or {a['case_key'] for a in control_audits} != expected_cases:
        raise ValueError('every fixed case control audit required exactly once')
    specificity.extend((all_energy,len(protocol_control) >= GATE['minimum_protocols_with_nonalias_control']))
    effect = all(performance);mechanism = all(specificity)
    route = ('NO_STABLE_SELECTION_INCREMENT' if not effect else 'PERFORMANCE_GAIN_WITHOUT_SEPARATED_SELECTION_EVIDENCE' if not mechanism else 'DEVELOPMENT_SELECTION_SIGNAL')
    compared_sota = all(primary['G_PAIRED'][weighting]['acc']+eps >= max(primary[arm][weighting]['acc'] for arm in methods if arm != 'G_PAIRED') for weighting in WEIGHTS)
    return dict(scientific_route=route,performance_gate=effect,selection_specificity_gate=mechanism,
        compared_panel_best_aggregate=compared_sota,gate=GATE,primary=primary,secondary=secondary,per_seed=per_seed,
        matched_anchor_differences=details,all_uniform_fp32_energy_groups_qualified=all_energy,
        protocols_with_nonalias_controls=sorted(protocol_control),true_counterfactual_labels_identified=False,
        causal_decomposition_identified=False,publication_ready=False,global_SOTA_established=False)
