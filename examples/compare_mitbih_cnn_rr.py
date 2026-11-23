"""Frozen complete validation comparison and all-missing-RR diagnostic, once."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_training/phase24b_mitbih_cnn_rr')
if __name__=='__main__':
    with HoldoutAccessGuard(ROOT/'runtime_guard_comparison_001.json') as guard:
        guard.probe()
        import json,csv,time
        import numpy as np
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.mitbih_cnn_rr_evaluation import evaluate,verify_alignment,neutral_rr
        from src.mitbih_unweighted import evaluate as evaluate_baseline,BASELINE,BASELINE_SHA256
        from src.mitbih_protocol import MITBIHExportIndex,load_split
        from src.mitbih_rr_dataset import MITBIHRRDataset,DEFAULT_RR_ROOT
        from src.mitbih_training import loss_weights,fingerprint
        from src.mitbih_readiness import verify_frozen_protocol
        from src.mitbih_full_run import validated_history,csv_new
        from src.models.mitbih_cnn_rr import MITBIHBeatCNNRR
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        torch.set_num_threads(2)
        output=ROOT/'run_001';target=output/'comparison'
        if target.exists():raise FileExistsError('Preserve completed/partial comparison; inspect before rerun')
        result,plan,history,best=validated_history(output)
        if result['status'] not in ('completed','early_stopped'):raise ValueError('Training incomplete')
        index=MITBIHExportIndex();split=load_split(index);verify_frozen_protocol(index,split)
        ds=MITBIHRRDataset(index=index,split=split,partition='validation',max_cached_shards=10)
        offset=0
        for e in ds.base.shards:ds[offset];offset+=e.count
        weights=loss_weights(index,split,'sqrt_balanced')
        if sha256_file(BASELINE/'best_model.pt')!=BASELINE_SHA256:raise ValueError('Baseline changed')
        cp=output/'best_model.pt'
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        last=torch.load(result['last_checkpoint'],map_location='cpu',weights_only=True)
        if saved['plan_sha256']!=fingerprint(plan) or last['plan']!=plan or saved['best_epoch']!=best['epoch']:raise ValueError('Checkpoint identity mismatch')
        m1=MITBIHBeatCNNRR();m2=MITBIHBeatCNNRR()
        m1.load_state_dict(saved['model_state_dict']);m2.load_state_dict(last['best_model_state_dict'])
        m1.eval();m2.eval()
        # Technical reload verification on TRAIN only, not a second training experiment.
        train=MITBIHRRDataset(index=index,split=split,partition='train',max_cached_shards=1)
        samples=[train[i] for i in range(8)];x=torch.stack([s['X'] for s in samples]);rr=torch.stack([s['RR'] for s in samples])
        with torch.inference_mode():
            a=m1(x,rr);b=m2(x,rr)
        if not torch.equal(a,b):raise ValueError('Checkpoint reload logits differ')
        del train,m1,m2,last,samples
        models={}
        models['A_Phase22_CNN']=evaluate_baseline(BASELINE/'best_model.pt',ds.base,split,weights)
        models['CNN_RR']=evaluate(cp,ds,split,weights)
        models['CNN_RR_neutral_diagnostic']=evaluate(cp,ds,split,weights,neutral=True)
        reference=models['A_Phase22_CNN']
        old=json.loads((BASELINE/f"epoch_{json.loads((BASELINE/'result.json').read_text())['best_epoch']:03d}_metrics.json").read_text())['validation']
        for name,values in models.items():
            verify_alignment(reference[2],values[2]);np.testing.assert_array_equal(reference[3],values[3])
        np.testing.assert_array_equal(reference[0]['confusion_matrix'],old['confusion_matrix'])
        np.testing.assert_array_equal(models['CNN_RR'][0]['confusion_matrix'],best['validation']['confusion_matrix'])
        for key in ('accuracy','macro_f1','macro_auprc'):
            np.testing.assert_allclose(reference[0][key],old[key],rtol=0,atol=1e-7)
            np.testing.assert_allclose(models['CNN_RR'][0][key],best['validation'][key],rtol=0,atol=1e-7)
        np.testing.assert_allclose(reference[0]['sqrt_weighted_validation_ce']['loss'],old['loss'],rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose(models['CNN_RR'][0]['sqrt_weighted_validation_ce']['loss'],best['validation']['loss'],rtol=1e-5,atol=1e-7)
        target.mkdir()
        classrows=[];summary=[];grouprows=[];recordrows=[]
        for name,(metrics,groups,metadata,y,prob,ce) in models.items():
            predicted=prob.argmax(1);cm=np.asarray(metrics['confusion_matrix'])
            write_json_new(target/f'{name}_metrics.json',metrics)
            write_json_new(target/f'{name}_patient_metrics.json',groups)
            with (target/f'{name}_validation_arrays.npz').open('xb') as f:
                np.savez_compressed(f,y=y,probabilities=prob,record_ids=np.asarray([r['record_id'] for r in metadata]),patient_groups=np.asarray([r['patient_group'] for r in metadata]),source_indices=np.asarray([r['source_index'] for r in metadata]))
            csv_new(target/f'{name}_predictions.csv',[dict(**r,actual_class='NSVFQ'[int(y[i])],predicted_class='NSVFQ'[int(predicted[i])],uncalibrated_softmax_score=float(prob[i,predicted[i]])) for i,r in enumerate(metadata)])
            classrows.extend(dict(model=name,**r,predicted_count=int(cm[:,r['class_index']].sum())) for r in metrics['per_class'])
            summary.append(dict(model=name,scope='validation-only diagnostic of one trained model' if 'neutral' in name else 'complete frozen validation',beats=len(y),accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],macro_auprc=metrics['macro_auprc'],forward_seconds=metrics['forward_seconds'],inference_ms_per_beat=metrics['inference_ms_per_beat'],**{f'{c}_{k}':metrics['per_class']['NSVFQ'.index(c)][k] for c in 'SFQ' for k in ('fp','fn','recall','precision')}))
            for group,gm in groups.items():grouprows.extend(dict(model=name,patient_group=group,**r) for r in gm['per_class'])
            for rid in sorted({r['record_id'] for r in metadata}):
                mask=np.asarray([r['record_id']==rid for r in metadata]);ry=y[mask];rp=predicted[mask]
                for i,c in enumerate('NSVFQ'):
                    tp=int(((ry==i)&(rp==i)).sum());support=int((ry==i).sum());fp=int(((ry!=i)&(rp==i)).sum())
                    recordrows.append(dict(model=name,record_id=rid,patient_group=index.by_record[rid].patient_group,class_name=c,support=support,tp=tp,fp=fp,fn=support-tp,recall=tp/support if support else None))
        csv_new(output/'PHASE24B_MODEL_COMPARISON.csv',summary);csv_new(output/'PHASE24B_CLASS_METRICS.csv',classrows)
        csv_new(target/'patient_group_class_metrics.csv',grouprows);csv_new(target/'record_class_metrics.csv',recordrows)
        write_json_new(output/'training_history.json',history)
        csv_new(output/'training_history.csv',[dict(epoch=h['epoch'],seconds=h['seconds'],train_loss=h['train']['loss'],validation_loss=h['validation']['loss'],train_macro_f1=h['train']['macro_f1'],validation_macro_f1=h['validation']['macro_f1'],validation_macro_auprc=h['validation']['macro_auprc']) for h in history])
        plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
        fig,axes=plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
        for ax,(name,values) in zip(axes,models.items()):
            cm=np.asarray(values[0]['confusion_matrix']);image=ax.imshow(np.log1p(cm),cmap='Blues')
            for i in range(5):
                for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if np.log1p(cm[i,j])>6 else 'black')
            ax.set(xticks=range(5),yticks=range(5),xticklabels=list('NSVFQ'),yticklabels=list('NSVFQ'),xlabel='Predicted group',ylabel='Actual group',title=name.replace('_','\n'))
        fig.suptitle('Complete MIT-BIH validation: counts (log1p color only), n=23134; no test evaluation')
        fig.savefig(target/'validation_confusion_matrices.png');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True);epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['train']['loss'] for r in history],'-o',label='Training, online')
        axes[0].plot(epochs,[r['validation']['loss'] for r in history],'-o',label='Full validation')
        axes[0].set(xlabel='Epoch',ylabel='sqrt-weighted CE: sum(w*CE)/sum(w)',title='Losses');axes[0].legend()
        axes[1].plot(epochs,[r['validation']['macro_f1'] for r in history],'-o',label='Macro F1')
        axes[1].plot(epochs,[r['validation']['macro_auprc'] for r in history],'-o',label='Macro AUPRC')
        axes[1].axvline(best['epoch'],linestyle=':',color='grey',label='Selected epoch')
        axes[1].set(xlabel='Epoch',ylabel='Full validation metric',ylim=(0,1),title='Checkpoint selection: Macro F1 only');axes[1].legend()
        fig.savefig(target/'training_validation_curves.png');plt.close(fig)
        write_json_new(target/'integrity.json',dict(best_epoch=best['epoch'],checkpoint_reload_logits_exact=True,identical_validation_sources=23134,reproduced_baseline_metrics=True,reproduced_training_validation_metrics=True,baseline_sha256=BASELINE_SHA256,CNN_RR_sha256=sha256_file(cp),read_records=index.read_records,unexpected_guard_denials=len(guard.unexpected),neutral_RR_vector=neutral_rr(ds.transform,1)[0].tolist(),neutral_scope='one trained model ablation; not independently trained CNN; no selection or thresholds changed'))
        print(json.dumps(summary,indent=2),flush=True)
