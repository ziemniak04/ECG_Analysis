"""Frozen full-fold-9 comparison of A/B/C. Never builds a final-test loader."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from src import ptbxl_training as trainer
from src.models.ptbxl_cnn import PTBXLRhythmCNN
from src.ptbxl_decisions import (ValidationOnlyIndex, frozen_validation_predictions,
    sbrad_decisions, decision_metrics)
from src.ptbxl_metrics import CLASS_NAMES, classification_metrics
from src.ptbxl_model_config import RhythmCNNConfig
from src.ptbxl_optimization import BASELINE, SELECTION, select_candidate
from src.ptbxl_protocol import file_sha256


def write_csv(path, rows):
    with path.open('x', newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_preservation(root):
    root=Path(root)
    protected=json.loads((root/'preservation_before.json').read_text())
    changed=[name for name,sha in protected.items() if not Path(name).is_file() or file_sha256(name)!=sha]
    if changed:raise RuntimeError(f'Protected baseline/code/report changed: {changed}')
    before=json.loads((root/'dataset_stats_before.json').read_text())
    after={str(p):[p.stat().st_size,p.stat().st_mtime_ns]
        for folder in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1')
        for p in Path(folder).rglob('*') if p.is_file()}
    if before!=after:raise RuntimeError('Dataset files changed')
    return {'protected_sha256_unchanged':len(protected),'dataset_files_size_mtime_unchanged':len(before)}


def compare(root,b_run=None,c_run=None):
    root=Path(root); output=root/'comparison'
    if output.exists():raise FileExistsError('Comparison outputs already exist; never overwrite')
    plan=json.loads((root/'study_plan.json').read_text())
    if plan['selection_criterion']!=SELECTION or plan['threshold_grid']!=[0.,.65]:
        raise ValueError('Prespecified selection criterion changed')
    preservation=verify_preservation(root)
    runs={'A_balanced':BASELINE,'B_unweighted':Path(b_run) if b_run else root/'B_unweighted',
          'C_sqrt_balanced':Path(c_run) if c_run else root/'C_sqrt_balanced'}
    output.mkdir()
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    index=ValidationOnlyIndex();all_rows=[];class_rows=[];models={};histories={};reference_data=None
    reference_population=None
    for name,run in runs.items():
        result=json.loads((run/'result.json').read_text())
        if result['status'] not in ('completed','early_stopped') or result['test_fold_used']:
            raise ValueError(f'{name} is not a complete safe training run')
        selection=json.loads((run/'selection_manifest.json').read_text())
        if reference_population is None:reference_population=selection['partitions']
        elif selection['partitions']!=reference_population:raise ValueError('Populations differ')
        checkpoint=run/'best_model.pt';sha=file_sha256(checkpoint)
        if sha!=result['best_model_sha256']:raise ValueError('Checkpoint hash mismatch')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['classes']!=list(CLASS_NAMES) or payload['manifest_sha256']!=index.manifest_sha256:
            raise ValueError('Model class mapping or dataset identity differs')
        model=PTBXLRhythmCNN(RhythmCNNConfig(**payload['model_configuration']))
        if model.parameter_count()!=74116:raise ValueError('Architecture differs')
        model.load_state_dict(payload['model_state_dict'],strict=True)
        timings=[];start_forward=[None]
        def before(module,inputs):start_forward[0]=time.perf_counter()
        def after(module,inputs,result):timings.append(time.perf_counter()-start_forward[0])
        h1=model.register_forward_pre_hook(before);h2=model.register_forward_hook(after)
        wall_start=time.perf_counter()
        try:data=frozen_validation_predictions(model,index)
        finally:h1.remove();h2.remove()
        wall_seconds=time.perf_counter()-wall_start
        if reference_data is None:reference_data=data
        else:
            for key in ('labels','ecg_id','patient_id','strat_fold'):
                np.testing.assert_array_equal(data[key],reference_data[key])
        if len(data['labels'])!=1958 or set(data['strat_fold'].tolist())!={9}:
            raise ValueError('Full validation cohort required')
        old=json.loads((run/'best_validation_metrics.json').read_text())
        if (data['ecg_id'].tolist()!=old['ecg_ids'] or data['labels'].tolist()!=old['true_labels']
                or data['probabilities'].argmax(axis=1).tolist()!=old['predicted_labels']
                or not np.allclose(data['probabilities'],old['probabilities'],atol=1e-7,rtol=1e-6)):
            raise ValueError('Fresh validation predictions differ from selected epoch')
        baseline=classification_metrics(data['labels'],data['probabilities'])
        with (output/f'{name}_predictions.npz').open('xb') as target:np.savez_compressed(target,**data)
        trainer.write_json_new(output/f'{name}_probability_metrics.json',baseline)
        decision_rows=[]
        for threshold in (0.,.65):
            decisions=sbrad_decisions(data['probabilities'],threshold)
            m=decision_metrics(data['labels'],decisions);m['threshold']=threshold;decision_rows.append(m)
            per=m['per_class'];s=per[3]
            row={'candidate':f'{name}_'+('argmax' if threshold==0 else 'threshold_0.65'),
                 'model':name,'checkpoint':str(checkpoint),'checkpoint_sha256':sha,
                 'threshold':threshold,'validation_records':len(data['labels']),
                 'validation_patients':len(set(data['patient_id'].tolist())),
                 'argmax_accuracy_full_cohort':baseline['accuracy'],
                 'accuracy_all_cases_abstain_not_correct':sum(c['tp'] for c in per)/len(data['labels']),
                 'macro_f1_full_cohort':m['full_cohort_macro_f1_abstentions_as_misses'],
                 'macro_auprc_full_cohort':baseline['macro_auprc'],
                 'SBRAD_fp':s['fp'],'SBRAD_precision':s['precision'],'SBRAD_recall':s['recall'],
                 'SBRAD_f1':s['f1'],'AFIB_recall':per[1]['recall'],'STACH_recall':per[2]['recall'],
                 'SR_to_any_target_arrhythmia_fp':sum(m['confusion_matrix_with_abstain'][0][1:4]),
                 'abstained':m['abstained'],'overall_abstention_rate':m['abstention_rate'],
                 'coverage':m['coverage'],
                 'accepted_accuracy_coverage_conditional':m['accepted_accuracy_coverage_conditional'],
                 'accepted_macro_f1_coverage_conditional':m['accepted_macro_f1_coverage_conditional'],
                 **{f"{c['class']}_abstained":c['abstained'] for c in per},
                 **{f"{c['class']}_abstention_rate":c['abstention_rate'] for c in per},
                 'inference_forward_ms_per_record':1000*sum(timings)/len(data['labels']),
                 'validation_inference_wall_seconds':wall_seconds,'best_epoch':result['best_epoch']}
            all_rows.append(row)
            for c in per:
                class_rows.append({'candidate':row['candidate'],'model':name,'threshold':threshold,
                    **c,'probability_auprc_full_cohort':baseline['per_class'][c['class_index']]['auprc']})
        trainer.write_json_new(output/f'{name}_decision_metrics.json',decision_rows)
        checkpoint_history=trainer.load_checkpoint(result['last_completed_checkpoint'])
        history=checkpoint_history['history'];histories[name]=history
        for record in history:
            for partition,expected,folds in (('train',15689,set(range(1,9))),('validation',1958,{9})):
                metrics=record[partition]
                if (metrics['record_count']!=expected or len(set(metrics['ecg_ids']))!=expected
                        or set(metrics['ecg_ids'])!=set(selection['partitions'][partition]['ecg_ids'])
                        or not set(metrics['strat_folds'])<=folds or not np.isfinite(metrics['loss'])):
                    raise ValueError('Incomplete/leaking/invalid epoch history')
            if name!='A_balanced' and record['epoch']<=len(histories['A_balanced']):
                if record['train']['ecg_ids']!=histories['A_balanced'][record['epoch']-1]['train']['ecg_ids']:
                    raise ValueError('Training shuffle differs from original control')
        if name!='A_balanced':
            expected=json.loads((root/f'{name}_plan.json').read_text())
            if checkpoint_history['plan']!=expected:raise ValueError('Loss/config fingerprint differs')
            summary=json.loads((run/'full_run_summary.json').read_text())
        else:summary=json.loads((BASELINE/'full_run_summary.json').read_text())
        models[name]={'run':str(run),'checkpoint':str(checkpoint),'sha256':sha,
            'epochs':len(history),'best_epoch':result['best_epoch'],'status':result['status'],
            'training_seconds':summary['training_call_seconds'],
            'class_weights':selection['class_weights'],'best_validation_metrics':baseline,
            'inference_forward_ms_per_record':1000*sum(timings)/len(data['labels']),
            'validation_wall_seconds':wall_seconds,'model_state_unchanged':True}
        if file_sha256(checkpoint)!=sha:raise ValueError('Frozen checkpoint changed')
    for b,c in zip(histories['B_unweighted'],histories['C_sqrt_balanced']):
        if b['train']['ecg_ids']!=c['train']['ecg_ids']:raise ValueError('B/C shuffled order differs')
    recommendation=select_candidate(all_rows)
    selected=recommendation['selected']
    for row in all_rows:row['recommended']=selected is not None and row['candidate']==selected['candidate']
    write_csv(root/'PHASE18C_MODEL_COMPARISON.csv',all_rows)
    write_csv(root/'PHASE18C_CLASS_METRICS.csv',class_rows)
    trainer.write_json_new(root/'candidate_selection.json',recommendation)
    if selected:
        trainer.write_json_new(root/'frozen_candidate.json',{
            'recommendation_only':True,'checkpoint':selected['checkpoint'],'checkpoint_sha256':selected['checkpoint_sha256'],
            'class_names':list(CLASS_NAMES),'sbrad_threshold':selected['threshold'],
            'abstain_index':-1,'require_sbrad_argmax':True,'accept_equality':True,
            'never_assign_second_choice':True,'validation_selected_not_medically_validated':True,
            'preprocessing_unchanged':True,'test_fold_used':False})
    plot_comparison(root,models,histories,all_rows)
    preservation.update(verify_preservation(root))
    if index.blocked_reads or len(index.read_ecg_ids)!=3*1958:
        raise ValueError('Unexpected waveform reads')
    summary={'status':'passed','models':models,'selection':recommendation,
             'comparison_validation_records_per_model':1958,'patients':1763,
             'waveform_folds_read':[9],'test_waveforms_read':0,'blocked_reads':index.blocked_reads,
             'all_models_same_validation_ids_labels_patients':True,'same_training_order_verified':True,
             'baseline_retrained':False,'preservation':preservation}
    trainer.write_json_new(root/'evaluation_summary.json',summary)
    print(json.dumps({'status':'passed','selection':recommendation},indent=2),flush=True)
    return summary


def plot_comparison(root,models,histories,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        with (root/name).open('xb') as target:fig.savefig(target,format='png',dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for col,name in enumerate(models):
        metrics=json.loads((root/'comparison'/f'{name}_decision_metrics.json').read_text())
        for ax,m in zip(axes[:,col],metrics):
            cm=np.asarray(m['confusion_matrix_with_abstain']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=1667)
            for i in range(4):
                for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>830 else 'black')
            ax.set(xticks=range(5),xticklabels=(*CLASS_NAMES,'UNCERTAIN'),yticks=range(4),yticklabels=CLASS_NAMES,
                xlabel='Decision',ylabel='True label',title=f"{name}; "+('argmax' if m['threshold']==0 else 'threshold 0.65'))
            ax.tick_params(axis='x',labelrotation=30)
    fig.suptitle('Complete fold-9 validation; frozen models; no test data')
    save(fig,'COMPARATIVE_CONFUSION_MATRICES.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(name,history) in zip(axes,histories.items()):
        epochs=[r['epoch'] for r in history]
        ax.plot(epochs,[r['train']['loss'] for r in history],label='Train (online)')
        ax.plot(epochs,[r['validation']['loss'] for r in history],label='Validation')
        ax.set(xlabel='Epoch',ylabel='CE loss (variant-specific weights)',title=name);ax.legend();ax.grid(alpha=.25)
    save(fig,'COMPARATIVE_LOSS_CURVES.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,history in histories.items():
        for ax,key,label in zip(axes,('macro_f1','macro_auprc'),('Macro F1','Macro AUPRC')):
            ax.plot([r['epoch'] for r in history],[r['validation'][key] for r in history],marker='o',label=name)
            ax.set(xlabel='Completed epoch',ylabel='Full-validation '+label,ylim=(0,1));ax.legend();ax.grid(alpha=.25)
    save(fig,'VALIDATION_MACRO_F1_AUPRC_CURVES.png')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for name in models:
        points=[r for r in rows if r['model']==name and r['SBRAD_precision'] is not None]
        ax.plot([r['SBRAD_recall'] for r in points],[r['SBRAD_precision'] for r in points],'-',alpha=.5)
        for r in points:
            marker='o' if r['threshold']==0 else 's'
            ax.scatter(r['SBRAD_recall'],r['SBRAD_precision'],marker=marker,s=60,label=f"{name} t={r['threshold']:g}")
    ax.set(xlabel='SBRAD recall, all 63 true cases',ylabel='SBRAD precision',xlim=(0,1),ylim=(0,1),
           title='Two prespecified operating points per frozen model; fold 9')
    ax.legend(fontsize=8);ax.grid(alpha=.25);save(fig,'SBRAD_PRECISION_RECALL_COMPARISON.png')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for name in models:
        points=[r for r in rows if r['model']==name]
        ax.plot([r['coverage'] for r in points],[r['SBRAD_fp'] for r in points],'-',alpha=.5)
        for r,marker in zip(points,('o','s')):
            ax.scatter(r['coverage'],r['SBRAD_fp'],marker=marker,s=60,label=f"{name} t={r['threshold']:g}")
    ax.set(xlabel='Coverage (accepted / 1958)',ylabel='SBRAD false-positive records',
           title='Coverage versus false SBRAD alerts — validation only')
    ax.legend(fontsize=8);ax.grid(alpha=.25);save(fig,'COVERAGE_FALSE_POSITIVE_TRADEOFF.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/model_training/phase18c_optimization/run_001'))
    parser.add_argument('--b-run',type=Path);parser.add_argument('--c-run',type=Path)
    args=parser.parse_args();compare(args.root,args.b_run,args.c_run)
