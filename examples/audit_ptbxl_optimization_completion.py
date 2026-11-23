"""Read-only recovery audit of completed Phase-18C runs and saved fold-9 results.

Loads checkpoints and prediction artifacts, never ECG waveforms. Does not resume
an early-stopped run or overwrite any completed experiment artifact.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.ptbxl_protocol import file_sha256
from src.ptbxl_metrics import classification_metrics
from src.ptbxl_decisions import decision_metrics, sbrad_decisions
from src.ptbxl_optimization import BASELINE, select_candidate
from src.ptbxl_training import load_checkpoint, write_json_new
from examples.compare_ptbxl_loss_models import verify_preservation


def audit(root):
    root=Path(root); preservation=verify_preservation(root)
    summary=json.loads((root/'evaluation_summary.json').read_text())
    assert summary['status']=='passed' and summary['test_waveforms_read']==0
    assert summary['waveform_folds_read']==[9] and not summary['baseline_retrained']
    study=json.loads((root/'study_plan.json').read_text())
    runs={}; reference=None; common_orders={}; candidate_rows=[]
    for name in ('A_balanced','B_unweighted','C_sqrt_balanced'):
        run=BASELINE if name=='A_balanced' else root/name
        result=json.loads((run/'result.json').read_text())
        assert result['status'] in ('completed','early_stopped') and not result['test_fold_used']
        assert result['checkpoint_reload_identical_logits']
        full=json.loads((run/'full_run_summary.json').read_text())
        assert file_sha256(run/'best_model.pt')==full['best_model_sha256']==result['best_model_sha256']
        assert file_sha256(run/'last_resumable_checkpoint.pt')==full['last_checkpoint_sha256']
        last=load_checkpoint(run/'last_resumable_checkpoint.pt')
        assert len(last['history'])==last['completed_epoch']==result['completed_epochs']==full['epochs']
        assert last['early_stopped']==(result['status']=='early_stopped')
        if last['early_stopped']:assert last['bad_epochs']>=4
        assert last['optimizer_steps']==981*last['completed_epoch']
        best_row=max(last['history'],key=lambda row:row['validation']['macro_f1'])
        assert best_row['epoch']==result['best_epoch']==full['best_epoch']
        model=torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
        for key,value in last['best_model_state_dict'].items():
            assert torch.equal(value,model['model_state_dict'][key])
        plan=last['plan']
        if reference is None:reference=plan['partitions']
        else:assert plan['partitions']==reference
        train_labels=plan['partitions']['train']['ecg_ids']
        validation_labels=plan['partitions']['validation']['ecg_ids']
        orders=[]
        for row in last['history']:
            orders.append(row['train']['ecg_ids'])
            for partition,count,batches,folds,ids in (
                ('train',15689,981,set(range(1,9)),train_labels),
                ('validation',1958,123,{9},validation_labels)):
                m=row[partition]
                assert m['record_count']==count and m['batches']==batches
                assert len(set(m['ecg_ids']))==count and set(m['ecg_ids'])==set(ids)
                assert set(m['strat_folds'])<=folds and np.isfinite(m['loss'])
        common_orders[name]=orders
        if name!='A_balanced':
            assert plan==json.loads((root/f'{name}_plan.json').read_text())
            cp_files=sorted((run/'checkpoints').glob('epoch_*.pt'))
            assert len(cp_files)==result['completed_epochs']
            for epoch,path in enumerate(cp_files,1):
                cp=load_checkpoint(path)
                assert cp['completed_epoch']==epoch and cp['plan_sha256']==last['plan_sha256']
            execution=json.loads((root/(name+'_audit')/'execution.json').read_text())
            resource=json.loads((root/(name+'_audit')/'resource_usage.json').read_text())
            assert execution['blocked_test_reads']==0 and '10' not in execution['read_counts_by_fold']
            assert resource['violation'] is None
        with np.load(root/'comparison'/f'{name}_predictions.npz',allow_pickle=False) as data:
            y=data['labels'];prob=data['probabilities']
            assert len(y)==1958 and len(set(data['ecg_id'].tolist()))==1958
            assert data['ecg_id'].tolist()==validation_labels
            assert set(data['strat_fold'].tolist())=={9}
            assert len(set(data['patient_id'].tolist()))==1763
            computed=classification_metrics(y,prob)
            saved=json.loads((root/'comparison'/f'{name}_probability_metrics.json').read_text())
            assert computed==saved
            original=json.loads((run/'best_validation_metrics.json').read_text())
            assert computed['confusion_matrix']==original['confusion_matrix']
            decisions=json.loads((root/'comparison'/f'{name}_decision_metrics.json').read_text())
            for saved_decision in decisions:
                threshold=saved_decision['threshold']
                expected=decision_metrics(y,sbrad_decisions(prob,threshold));expected['threshold']=threshold
                assert expected==saved_decision
        runs[name]={'status':result['status'],'epochs':result['completed_epochs'],
            'best_epoch':result['best_epoch'],'best_sha256':full['best_model_sha256'],
            'last_checkpoint_valid':True,'training_repeated_or_resumed':False}
    for name,orders in common_orders.items():
        for other,other_orders in common_orders.items():
            for left,right in zip(orders,other_orders):assert left==right
    # Re-run only the pure selection function on persisted comparison rows.
    import csv
    for row in csv.DictReader((root/'PHASE18C_MODEL_COMPARISON.csv').open()):
        keys=('threshold','AFIB_recall','STACH_recall','SBRAD_recall','coverage',
              'macro_f1_full_cohort','macro_auprc_full_cohort')
        candidate_rows.append({**row,**{k:float(row[k]) for k in keys},'SBRAD_fp':int(row['SBRAD_fp'])})
    chosen=select_candidate(candidate_rows)
    assert chosen['selected']['candidate']==summary['selection']['selected']['candidate']
    result={'status':'passed','runs':runs,'training_resume_required':False,
        'training_repeated':False,'completed_model_comparison_preserved':True,
        'saved_probabilities_metrics_recomputed':True,'test_waveforms_read_by_audit':0,
        'selected_candidate':chosen['selected']['candidate'],'preservation':preservation,
        'baseline_sha256':study['baseline_sha256']}
    write_json_new(root/'resumption_audit_001.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/model_training/phase18c_optimization/run_001'))
    args=parser.parse_args();audit(args.root)
