"""Two frozen checkpoints, same full validation; no final-test data or training."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.mitbih_unweighted import BASELINE,BASELINE_SHA256,evaluate,loss_summary
from src.mitbih_protocol import MITBIHExportIndex,load_split
from src.mitbih_readiness import verify_frozen_protocol
from src.mitbih_training import loss_weights
from src.mitbih_full_run import summarize_run,validated_history,csv_new
from src.label_mapping import DEFAULT_MAPPING
from src.prepare_dataset import sha256_file
from src.ptbxl_training import write_json_new
from src.mitbih_torch import MITBIHBeatDataset



def frozen_train_losses(checkpoint,dataset,split,weights):
    """Loss-only inference on TRAIN, separating online/mode effects from generalization."""
    if dataset.partition!='train':raise PermissionError('Training-only loss audit')
    from src.models.mitbih_cnn import MITBIHBeatCNN
    from src.mitbih_torch import collate_mitbih
    from torch.utils.data import DataLoader
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=MITBIHBeatCNN();model.load_state_dict(saved['model_state_dict']);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    seen=set();ce=[];labels=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,collate_fn=collate_mitbih):
            if not set(batch['patient_group'])<=set(split['groups']['train']):raise PermissionError('Wrong group')
            logits=model(batch['X'])
            ce.extend(torch.nn.functional.cross_entropy(logits,batch['y'],reduction='none').tolist())
            labels.extend(batch['y'].tolist());seen.update(batch['global_source_index'].tolist())
    if len(seen)!=len(dataset) or len(labels)!=len(dataset):raise ValueError('Incomplete training inference')
    if not all(torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Model changed')
    return dict(beat_count=len(labels),mode='frozen eval/inference_mode; no updates',
        unweighted_ce=loss_summary(ce,np.asarray(labels,dtype=np.int64),None),
        sqrt_weighted_ce=loss_summary(ce,np.asarray(labels,dtype=np.int64),weights),parameters_unchanged=True)


def main(output,audit):
    output,audit=Path(output),Path(audit)
    target=output/'comparison'
    if target.exists():raise FileExistsError('Comparison already exists')
    if sha256_file(BASELINE/'best_model.pt')!=BASELINE_SHA256:raise ValueError('Baseline changed')
    summarize_run(output)
    target.mkdir()
    index=MITBIHExportIndex();split=load_split(index);verify_frozen_protocol(index,split)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    dataset=MITBIHBeatDataset(index,split,partition='validation',max_cached_shards=10)
    # Fill only the validation loader cache, same I/O conditions for both checkpoints.
    offset=0
    for shard in dataset.shards:dataset[offset];offset+=shard.count
    weights=loss_weights(index,split,'sqrt_balanced')
    results={};all_rows=[];class_rows=[];group_rows=[];symbol_rows=[];f_error_rows=[];predictions={}
    for name,folder in [('A_sqrt_balanced',BASELINE),('B_unweighted',output)]:
        metrics,groups,metadata,y,prob,ce=evaluate(folder/'best_model.pt',dataset,split,weights)
        results[name]=metrics;predictions[name]=(y,prob)
        write_json_new(target/(name+'_metrics.json'),metrics)
        write_json_new(target/(name+'_patient_groups.json'),groups)
        result,plan,history,best=validated_history(folder)
        np.testing.assert_array_equal(metrics['confusion_matrix'],best['validation']['confusion_matrix'])
        np.testing.assert_allclose(metrics['macro_f1'],best['validation']['macro_f1'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(metrics['macro_auprc'],best['validation']['macro_auprc'],rtol=1e-6,atol=1e-7)
        own='sqrt_weighted_validation_ce' if name.startswith('A') else 'unweighted_validation_ce'
        np.testing.assert_allclose(metrics[own]['loss'],best['validation']['loss'],rtol=3e-6,atol=3e-6)
        all_rows.append(dict(model=name,epochs=result['completed_epochs'],best_epoch=result['best_epoch'],
            accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],macro_auprc=metrics['macro_auprc'],
            nonzero_recall_classes=metrics['classes_with_nonzero_recall'],
            validation_unweighted_ce=metrics['unweighted_validation_ce']['loss'],
            validation_sqrt_weighted_ce=metrics['sqrt_weighted_validation_ce']['loss'],
            training_online_own_loss=best['train']['loss'],validation_own_loss=best['validation']['loss'],
            inference_seconds=metrics['forward_seconds'],inference_ms_per_beat=metrics['inference_ms_per_beat'],
            training_seconds=json.loads((folder/'execution.json').read_text())['wall_seconds'],
            checkpoint_sha256=metrics['checkpoint_sha256']))
        class_rows.extend(dict(model=name,**row) for row in metrics['per_class'])
        group_rows.extend(dict(model=name,patient_group=g,**row) for g,m in groups.items() for row in m['per_class'])
        pred=prob.argmax(1)
        for symbol in sorted({r['original_symbol'] for r in metadata}):
            mask=np.array([r['original_symbol']==symbol for r in metadata])
            for true in range(5):
                for predicted in range(5):
                    count=int(np.sum(mask&(y==true)&(pred==predicted)))
                    if count:symbol_rows.append(dict(model=name,original_symbol=symbol,
                        mapped_class='NSVFQ'[true],predicted_class='NSVFQ'[predicted],count=count))
        rows=[dict(**r,true_class='NSVFQ'[int(y[i])],predicted_class='NSVFQ'[int(pred[i])],
                   **{'p_'+c:float(prob[i,j]) for j,c in enumerate('NSVFQ')}) for i,r in enumerate(metadata)]
        csv_new(target/(name+'_validation_predictions.csv'),rows)
        with (target/(name+'_validation_arrays.npz')).open('xb') as f:
            np.savez_compressed(f,y=y,probabilities=prob.astype(np.float32),unweighted_ce=ce.astype(np.float32),
                source_indices=np.array([r['source_index'] for r in metadata],dtype=np.int64))
        fp_counts=Counter((r['record_id'],r['original_symbol'],'NSVFQ'[int(y[i])])
            for i,r in enumerate(metadata) if pred[i]==3 and y[i]!=3)
        f_error_rows.extend(dict(model=name,record_id=rid,original_symbol=symbol,true_class=c,F_false_positives=count)
            for (rid,symbol,c),count in sorted(fp_counts.items()))
        # Explicit 201/202 split, beyond their combined patient-group table.
        for rid in ('201','202'):
            mask=np.array([r['record_id']==rid for r in metadata]);cm=np.zeros((5,5),dtype=int)
            np.add.at(cm,(y[mask],pred[mask]),1)
            write_json_new(target/f'{name}_record_{rid}_confusion.json',dict(record_id=rid,
                patient_group='mitdb_group_201_202',beats=int(mask.sum()),confusion_matrix=cm.tolist(),
                F_false_positives=int(cm[:,3].sum()-cm[3,3])))
    train_dataset=MITBIHBeatDataset(index,split,partition='train',max_cached_shards=29)
    frozen_train={}
    for name,folder in [('A_sqrt_balanced',BASELINE),('B_unweighted',output)]:
        frozen_train[name]=frozen_train_losses(folder/'best_model.pt',train_dataset,split,weights)
    write_json_new(target/'FROZEN_TRAIN_LOSS_AUDIT.json',frozen_train)
    np.testing.assert_array_equal(predictions['A_sqrt_balanced'][0],predictions['B_unweighted'][0])
    csv_new(target/'MODEL_COMPARISON.csv',all_rows)
    csv_new(target/'CLASS_COMPARISON.csv',class_rows)
    csv_new(target/'PATIENT_GROUP_COMPARISON.csv',group_rows)
    csv_new(target/'ORIGINAL_SYMBOL_ERRORS.csv',symbol_rows)
    if f_error_rows:csv_new(target/'F_FALSE_POSITIVE_SYMBOL_AUDIT.csv',f_error_rows)
    with (audit/'original_symbol_distribution.csv').open() as f:source=list(csv.DictReader(f))
    totals={}
    for r in source:
        key=(r['partition'],r['original_symbol']);totals[key]=totals.get(key,0)+int(r['count'])
    csv_new(target/'ORIGINAL_SYMBOL_AAMI_AUDIT.csv',[dict(partition=p,original_symbol=s,
        mapped_class=DEFAULT_MAPPING[s],count=n) for (p,s),n in sorted(totals.items())])
    allowed=set(split['summary']['train']['record_ids']+split['summary']['validation']['record_ids'])
    if not set(index.read_records)<=allowed:raise RuntimeError('Unexpected final-test access')
    protected=json.loads((audit/'preservation_before.json').read_text())
    excluded=[p for p in protected if Path(p).is_relative_to(output.parent)]
    changed=[p for p,d in protected.items() if p not in excluded and (not Path(p).is_file() or sha256_file(p)!=d)]
    stats=json.loads((audit/'dataset_stats_before.json').read_text())
    changed_data=[p for p,v in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
    current={str(p) for root in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1') for p in Path(root).rglob('*') if p.is_file()}
    if changed or changed_data or current!=set(stats):raise ValueError(f'Preservation check failed: {changed}, {changed_data}')
    write_json_new(target/'integrity.json',dict(status='passed',prior_files_sha256_verified=len(protected)-len(excluded),
        dataset_files_stat_verified=len(stats),changed_prior_files=[],changed_dataset_files=[],
        mutable_new_run_logs_excluded=excluded,final_test_read=False,baseline_sha256=sha256_file(BASELINE/'best_model.pt'),
        evaluated_records=index.read_records,validation_populations_identical=True,
        training_loss_audit_only=True,
        common_loss_matches_original_epoch=True,model_parameters_unchanged=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,(name,m) in zip(axes,results.items()):
        cm=np.asarray(m['confusion_matrix']);norm=cm/cm.sum(1,keepdims=True)
        im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{cm[i,j]}\n{norm[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.5 else 'black',fontsize=8)
        ax.set_xticks(range(5),list('NSVFQ'));ax.set_yticks(range(5),list('NSVFQ'))
        ax.set(title=name,xlabel='Predicted',ylabel='True')
    fig.colorbar(im,ax=axes,label='Recall fraction within true class')
    with (target/'confusion_matrices.png').open('xb') as f:fig.savefig(f,format='png',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,folder in [('A sqrt balanced',BASELINE),('B unweighted',output)]:
        _,_,history,_=validated_history(folder);ep=[r['epoch'] for r in history]
        axes[0].plot(ep,[r['train']['loss'] for r in history],'o-',label=name+' train')
        axes[0].plot(ep,[r['validation']['loss'] for r in history],'o--',label=name+' val')
        axes[1].plot(ep,[r['validation']['macro_f1'] for r in history],'o-',label=name)
    axes[0].set(xlabel='Epoch',ylabel='Own-objective CE (not cross-model comparable)',title='Online train / frozen validation')
    axes[1].set(xlabel='Epoch',ylabel='Full-validation Macro F1',title='Same patient-separated validation')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
    with (target/'training_comparison.png').open('xb') as f:fig.savefig(f,format='png',dpi=300)
    plt.close(fig)
    print(json.dumps(all_rows,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args();main(args.output_dir,args.audit_dir)
