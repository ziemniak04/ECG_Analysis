"""PTB-XL training infrastructure, with an explicitly bounded Phase-17 default.

Reuses the Phase-16 model/loader. Never instantiates a test-fold Dataset.
Checkpoints/runs are immutable; resume continues at an epoch boundary into a
NEW output directory. In-progress partial epochs are replayed from last checkpoint.
"""
from dataclasses import asdict,dataclass
import csv
import hashlib
import json
from itertools import islice
import logging
import math
import os
import platform
import sys
import importlib.metadata
from pathlib import Path
import random
import time
from uuid import uuid4
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .models.ptbxl_cnn import PTBXLRhythmCNN
from .ptbxl_model_config import RhythmCNNConfig
from .ptbxl_protocol import PTBXLExportIndex,PARTITION_FOLDS,training_class_counts,training_class_weights,file_sha256
from .ptbxl_torch import PTBXLRhythmDataset
from .ptbxl_metrics import CLASS_NAMES,classification_metrics

LOG=logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = 'short_baseline'
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 42
    device: str = 'cpu'
    cpu_threads: int = 1
    num_workers: int = 0
    train_per_class: int | None = 32
    validation_per_class: int | None = 16
    max_train_batches: int | None = 8
    max_validation_batches: int | None = 4
    class_weighted_loss: bool = False
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self):
        for name in ('epochs','batch_size','cpu_threads','patience'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or not 0<=self.seed<2**32:raise ValueError('Invalid seed')
        if isinstance(self.num_workers,bool) or not isinstance(self.num_workers,int) or self.num_workers<0:raise ValueError('Invalid workers')
        for name in ('train_per_class','validation_per_class','max_train_batches','max_validation_batches'):
            value=getattr(self,name)
            if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value<1):raise ValueError(f'Invalid {name}')
        if isinstance(self.learning_rate,bool) or not math.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError('Invalid learning rate')
        if isinstance(self.min_delta,bool) or not math.isfinite(self.min_delta) or self.min_delta<0:raise ValueError('Invalid min_delta')
        if not isinstance(self.class_weighted_loss,bool):raise ValueError('class_weighted_loss must be boolean')
        if self.device not in ('cpu','cuda'):raise ValueError('Device must be cpu or explicitly verified cuda')
        if self.mode not in ('short_baseline','full'):raise ValueError('Unknown experiment mode')
        if self.mode=='short_baseline':
            if self.epochs>2 or self.batch_size>32 or self.train_per_class is None or self.validation_per_class is None:
                raise ValueError('Short baseline requires <=2 epochs, batch<=32 and explicit small cohorts')
            if self.train_per_class>64 or self.validation_per_class>32:
                raise ValueError('Short baseline permits at most 256 train and 128 validation records')
            if self.max_train_batches is None or self.max_validation_batches is None or self.max_train_batches>16 or self.max_validation_batches>8:
                raise ValueError('Short baseline requires bounded training/validation batches')


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def select_indices(index,partition,per_class,seed):
    """Deterministic hash ordering within classes, then ascending source index.

    Does not select based on waveforms, quality or model scores. No record is
    duplicated. None means the entire official training/validation partition.
    """
    if partition not in ('train','validation'):raise ValueError('Phase 17 cannot select final test records')
    if per_class is not None and (isinstance(per_class,bool) or not isinstance(per_class,int) or per_class<1):raise ValueError('Invalid class quota')
    source=getattr(index.partitions,partition)
    if per_class is None:return tuple(source)
    result=[]
    for label in range(4):
        candidates=[i for i in source if index.entries[i].label==label]
        if len(candidates)<per_class:raise ValueError(f'Not enough {partition} records for class {CLASS_NAMES[label]}')
        ordered=sorted(candidates,key=lambda i:(_hash([seed,partition,index.entries[i].ecg_id]),index.entries[i].ecg_id))
        result.extend(ordered[:per_class])
    return tuple(sorted(result))


def verify_selection(index,train_indices,validation_indices):
    if not train_indices or not validation_indices:raise ValueError('Both partitions must be nonempty')
    for name,indices in [('train',train_indices),('validation',validation_indices)]:
        if len(set(indices))!=len(indices):raise ValueError('Duplicate subset indices')
        allowed=set(getattr(index.partitions,name))
        if any(i not in allowed for i in indices):raise ValueError(f'Invalid {name} indices')
    train_patients={index.entries[i].patient_id for i in train_indices}
    val_patients={index.entries[i].patient_id for i in validation_indices}
    if set(train_indices)&set(validation_indices) or train_patients&val_patients:raise ValueError('Patient or record leakage')
    if any(index.entries[i].strat_fold==10 for i in (*train_indices,*validation_indices)):raise ValueError('Final test fold is forbidden')


def build_plan(index,config):
    train=select_indices(index,'train',config.train_per_class,config.seed)
    validation=select_indices(index,'validation',config.validation_per_class,config.seed)
    verify_selection(index,train,validation)
    groups={}
    for name,indices,cap in [('train',train,config.max_train_batches),('validation',validation,config.max_validation_batches)]:
        batches=math.ceil(len(indices)/config.batch_size)
        groups[name]={'global_indices':list(indices),'ecg_ids':[index.entries[i].ecg_id for i in indices],
            'patient_ids':sorted({index.entries[i].patient_id for i in indices}),
            'patient_count':len({index.entries[i].patient_id for i in indices}),'records':len(indices),
            'class_counts':np.bincount([index.entries[i].label for i in indices],minlength=4).tolist(),
            'folds':sorted({index.entries[i].strat_fold for i in indices}),
            'available_batches':batches,'batches_per_epoch':min(batches,cap) if cap else batches,
            'maximum_records_per_epoch':min(len(indices),config.batch_size*(min(batches,cap) if cap else batches))}
    weights=training_class_weights(index,train).tolist() if config.class_weighted_loss else None
    plan={'schema':'ptbxl-training-plan-v1','configuration':asdict(config),'manifest_sha256':index.manifest_sha256,
        'manifest':str(index.manifest_path),'classes':list(CLASS_NAMES),'partitions':groups,
        'patient_overlap':False,'test_fold_used':False,'class_weights':weights,
        'class_weight_source':'selected_training_indices_only' if weights is not None else 'disabled',
        'model_configuration':RhythmCNNConfig().to_dict(),
        'code_sha256':{name:file_sha256(Path(__file__).parent/name) for name in
           ('ptbxl_training.py','ptbxl_metrics.py','ptbxl_torch.py','ptbxl_protocol.py','models/ptbxl_cnn.py','ptbxl_model_config.py')}}
    # JSON-normalize tuples for exact, version-independent checkpoint comparison.
    return json.loads(json.dumps(plan))


def prepare_device(config):
    if config.device=='cpu':return torch.device('cpu')
    if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use CPU')
    device=torch.device('cuda')
    probe=torch.ones((2,2),device=device);output=probe@probe
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all()):raise RuntimeError('CUDA probe failed')
    return device


def seed_everything(config):
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if config.device=='cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def _worker_seed(worker_id):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)


def make_loader(index,indices,partition,config,epoch):
    if partition not in ('train','validation'):raise ValueError('No final-test loader is available in Phase 17')
    dataset=PTBXLRhythmDataset(index,partition=partition)
    mapping={global_i:local_i for local_i,global_i in enumerate(dataset.source_indices)}
    local=[mapping[i] for i in indices]
    generator=torch.Generator().manual_seed(config.seed+epoch)
    return DataLoader(Subset(dataset,local),batch_size=config.batch_size,shuffle=partition=='train',
        num_workers=config.num_workers,generator=generator,worker_init_fn=_worker_seed,drop_last=False)


def _sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def epoch_pass(model,loader,device,*,partition,weights=None,optimizer=None,batch_cap=None):
    """Weighted loss aggregates exact numerator/denominator, not batch means.

    Validation forward timing excludes loading, transfer, loss, softmax and metrics.
    Training metrics use online pre-update logits with dropout/BN training enabled.
    """
    if partition not in ('train','validation') or (optimizer is not None)!=(partition=='train'):
        raise ValueError('Optimizer is allowed only for training; final test is forbidden')
    train=partition=='train';model.train(train)
    numerator=denominator=0.;labels=[];probabilities=[];ecg_ids=[];folds=[]
    batches=0;forward_seconds=0.;finite_gradients=True
    criterion=torch.nn.CrossEntropyLoss(weight=weights,reduction='sum')
    context=torch.enable_grad() if train else torch.inference_mode()
    with context:
        limited=loader if batch_cap is None else islice(loader,batch_cap)
        for batch_number,batch in enumerate(limited):
            if any(int(f) not in PARTITION_FOLDS[partition] for f in batch['strat_fold']):
                raise RuntimeError(f'Wrong official fold in {partition} batch')
            x,y=batch['X'].to(device),batch['y'].to(device)
            if tuple(x.shape[1:])!=(12,1000) or x.dtype!=torch.float32 or y.dtype!=torch.long:
                raise ValueError('Invalid training tensors')
            if not bool(((y>=0)&(y<4)).all()):raise ValueError('Invalid rhythm labels')
            if train:optimizer.zero_grad(set_to_none=True)
            _sync(device);start=time.perf_counter();logits=model(x);_sync(device)
            forward_seconds+=time.perf_counter()-start
            if logits.shape!=(len(y),4) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite or malformed logits')
            loss_sum=criterion(logits,y)
            normalizer=weights[y].sum() if weights is not None else y.new_tensor(len(y),dtype=torch.float32)
            loss=loss_sum/normalizer
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
            if train:
                loss.backward()
                gradients=[p.grad for p in model.parameters() if p.requires_grad]
                if not all(g is not None and bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError('Missing or nonfinite gradients')
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameter after update')
            numerator+=float(loss_sum.detach());denominator+=float(normalizer.detach())
            labels.extend(y.detach().cpu().tolist())
            probabilities.extend(torch.softmax(logits.detach(),dim=1).cpu().numpy().tolist())
            ecg_ids.extend(batch['ecg_id'].tolist());folds.extend(batch['strat_fold'].tolist());batches+=1
            expected=min(len(loader),batch_cap) if batch_cap else len(loader)
            if batches==1 or batches%10==0 or batches==expected:
                LOG.info('%s batch %s/%s loss %.6f',partition,batches,expected,float(loss.detach()))
    if not labels:raise ValueError('No batches were processed')
    metrics=classification_metrics(np.asarray(labels,dtype=np.int64),np.asarray(probabilities))
    metrics.update({'loss':numerator/denominator,'loss_numerator':numerator,'loss_denominator':denominator,
        'batches':batches,'ecg_ids':ecg_ids,'strat_folds':folds,
        'true_labels':labels,'predicted_labels':np.asarray(probabilities).argmax(axis=1).tolist(),
        'probabilities':probabilities,'finite_gradients':finite_gradients if train else None,
        'forward_seconds':forward_seconds,'inference_ms_per_record':1000*forward_seconds/len(labels) if not train else None,
        'metric_scope':'online pre-update training logits' if train else 'epoch-end eval/inference_mode validation logits'})
    return metrics


def rng_state(include_cuda=False):
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if include_cuda else []}


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def write_json_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def save_checkpoint_new(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/('.pending-'+uuid4().hex+'.pt')
    try:
        with temporary.open('xb') as target:
            torch.save(payload,target);target.flush();os.fsync(target.fileno())
        os.link(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def load_checkpoint(path):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!='ptbxl-training-checkpoint-v1':raise ValueError('Unsupported checkpoint schema')
    return checkpoint


def _plot_results(output,history,best_metrics,preliminary=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in history]
    title='PRELIMINARY — class-aware subset baseline' if preliminary else 'Validation-selected full training run'
    for name,ylabel,series in [('loss_curves','Cross-entropy loss',[(r['train']['loss'],r['validation']['loss']) for r in history]),
                               ('validation_macro_f1','Validation Macro F1',[r['validation']['macro_f1'] for r in history])]:
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        if name=='loss_curves':
            ax.plot(epochs,[s[0] for s in series],marker='o',label='Training (online)')
            ax.plot(epochs,[s[1] for s in series],marker='o',label='Validation (eval)');ax.legend()
        else:ax.plot(epochs,series,marker='o',color='tab:purple');ax.set_ylim(0,1)
        ax.set_xlabel('Completed epoch');ax.set_ylabel(ylabel);ax.set_xticks(epochs);ax.grid(alpha=.25)
        ax.set_title(title)
        with (output/(name+'.png')).open('xb') as f:fig.savefig(f,format='png',dpi=300)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    cm=np.asarray(best_metrics['confusion_matrix']);image=ax.imshow(cm,cmap='Blues');fig.colorbar(image,ax=ax,label='Record count')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks(range(4),CLASS_NAMES);ax.set_yticks(range(4),CLASS_NAMES)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title('PRELIMINARY — best validation subset' if preliminary else 'Best validation confusion matrix')
    with (output/'validation_confusion_matrix.png').open('xb') as f:fig.savefig(f,format='png',dpi=300)
    plt.close(fig)


def train_model(config,output_dir,*,index=None,resume_from=None,allow_full_training=False,epoch_budget=None):
    """Run finite epochs. Resume is accepted only for identical plan/code/config.

    epoch_budget is a smaller invocation budget for controlled interruption tests;
    it never increases config.epochs. Production default runs configured epochs.
    """
    if config.mode=='full' and allow_full_training is not True:
        raise PermissionError('Full training needs separate explicit approval; Phase 17 defaults to a short subset')
    if epoch_budget is not None and (isinstance(epoch_budget,bool) or not isinstance(epoch_budget,int) or epoch_budget<1):raise ValueError('Invalid invocation epoch budget')
    output=Path(output_dir)
    if output.exists():raise FileExistsError(f'Experiment output already exists: {output}; resume into a NEW directory')
    index=PTBXLExportIndex() if index is None else index
    plan=build_plan(index,config);fingerprint=_hash(plan)
    seed_everything(config);device=prepare_device(config)
    model=PTBXLRhythmCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    weights=torch.tensor(plan['class_weights'],dtype=torch.float32,device=device) if plan['class_weights'] is not None else None
    history=[];best_score=None;best_epoch=None;best_state=None;bad_epochs=0;stopping_score=None;steps=0;first_epoch=1
    if resume_from is not None:
        checkpoint=load_checkpoint(resume_from)
        if checkpoint['plan_sha256']!=fingerprint or checkpoint['plan']!=plan:raise ValueError('Resume plan/config/code/metadata differs')
        if checkpoint['early_stopped']:raise ValueError('Cannot resume a run already stopped by early stopping')
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history=checkpoint['history'];best_score=checkpoint['best_score'];best_epoch=checkpoint['best_epoch'];best_state=checkpoint['best_model_state_dict']
        bad_epochs=checkpoint['bad_epochs'];stopping_score=checkpoint['stopping_score'];steps=checkpoint['optimizer_steps']
        first_epoch=checkpoint['completed_epoch']+1
        if first_epoch>config.epochs:raise ValueError('Checkpoint already completed all configured epochs')
        restore_rng(checkpoint['rng_state'])
    output.mkdir(parents=True,exist_ok=False)
    write_json_new(output/'training_config.json',asdict(config));write_json_new(output/'selection_manifest.json',plan)
    write_json_new(output/'architecture.json',{'configuration':RhythmCNNConfig().to_dict(),'parameter_count':model.parameter_count()})
    initial_parameters=[p.detach().clone() for p in model.parameters()]
    status='completed';last_checkpoint=None;executed=0
    handler=logging.FileHandler(output/'training.log',mode='x')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOG.addHandler(handler);LOG.setLevel(logging.INFO)
    try:
        for epoch in range(first_epoch,config.epochs+1):
            if epoch_budget is not None and executed>=epoch_budget:
                status='invocation_epoch_budget_reached';break
            train_loader=make_loader(index,tuple(plan['partitions']['train']['global_indices']),'train',config,epoch)
            val_loader=make_loader(index,tuple(plan['partitions']['validation']['global_indices']),'validation',config,epoch)
            training=epoch_pass(model,train_loader,device,partition='train',weights=weights,optimizer=optimizer,batch_cap=config.max_train_batches)
            validation=epoch_pass(model,val_loader,device,partition='validation',weights=weights,batch_cap=config.max_validation_batches)
            score=validation['macro_f1']
            if score is None:raise ValueError('Validation Macro F1 undefined: absent class after batch cap; use all four classes')
            steps+=training['batches']
            record={'epoch':epoch,'optimizer_steps_cumulative':steps,'train':training,'validation':validation}
            history.append(record)
            if best_score is None or score>best_score:
                best_score=score;best_epoch=epoch
                best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if stopping_score is None or score>stopping_score+config.min_delta:
                stopping_score=score;bad_epochs=0
            else:bad_epochs+=1
            stopped=bad_epochs>=config.patience
            payload={'schema':'ptbxl-training-checkpoint-v1','plan':plan,'plan_sha256':fingerprint,
                'completed_epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'best_model_state_dict':best_state,'best_score':best_score,'best_epoch':best_epoch,
                'stopping_score':stopping_score,'bad_epochs':bad_epochs,'early_stopped':stopped,
                'optimizer_steps':steps,'history':history,'rng_state':rng_state(include_cuda=device.type=='cuda')}
            last_checkpoint=output/'checkpoints'/f'epoch_{epoch:03d}.pt'
            save_checkpoint_new(last_checkpoint,payload)
            write_json_new(output/f'epoch_{epoch:03d}_metrics.json',record)
            executed+=1
            LOG.info('epoch %s: train loss %.6f val loss %.6f val Macro F1 %.6f (%s updates)',epoch,training['loss'],validation['loss'],score,training['batches'])
            if stopped:status='early_stopped';break
    except KeyboardInterrupt:
        status='interrupted_partial_epoch_discarded'
        LOG.warning('Interrupted; resume from the last completed epoch checkpoint into a new directory')
    except Exception as exc:
        write_json_new(output/'failure.json',{'error':f'{type(exc).__name__}: {exc}','last_completed_checkpoint':str(last_checkpoint) if last_checkpoint else None})
        raise
    finally:
        LOG.removeHandler(handler);handler.close()
    updates_changed=any(not torch.equal(a,b.detach()) for a,b in zip(initial_parameters,model.parameters()))
    result={'status':status,'mode':config.mode,'completed_epochs':len(history),'epochs_executed_this_run':executed,
        'optimizer_steps':steps,'best_epoch':best_epoch,'best_validation_macro_f1':best_score,
        'parameters_changed_this_run':updates_changed,'device':str(device),'test_fold_used':False,
        'last_completed_checkpoint':str(last_checkpoint) if last_checkpoint else None,
        'resume_from':str(resume_from) if resume_from else None,
        'resume_checkpoint_sha256':file_sha256(resume_from) if resume_from else None,'plan_sha256':fingerprint,
        'preliminary_subset_experiment':config.mode=='short_baseline'}
    if history:
        with (output/'epoch_metrics.csv').open('x',newline='') as f:
            fields=['epoch','train_loss','validation_loss','train_accuracy','validation_accuracy','validation_macro_f1','validation_macro_auprc','train_batches','validation_batches']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in history:
                writer.writerow({'epoch':row['epoch'],'train_loss':row['train']['loss'],'validation_loss':row['validation']['loss'],
                    'train_accuracy':row['train']['accuracy'],'validation_accuracy':row['validation']['accuracy'],
                    'validation_macro_f1':row['validation']['macro_f1'],'validation_macro_auprc':row['validation']['macro_auprc'],
                    'train_batches':row['train']['batches'],'validation_batches':row['validation']['batches']})
        best_metrics=next(r['validation'] for r in history if r['epoch']==best_epoch)
        write_json_new(output/'best_validation_metrics.json',best_metrics)
        validation_entries={index.entries[i].ecg_id:index.entries[i] for i in plan['partitions']['validation']['global_indices']}
        with (output/'best_validation_predictions.csv').open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['ecg_id','patient_id','strat_fold','true_label','predicted_label',*['p_'+c for c in CLASS_NAMES]])
            for eid,y,pred,prob in zip(best_metrics['ecg_ids'],best_metrics['true_labels'],best_metrics['predicted_labels'],best_metrics['probabilities']):
                entry=validation_entries[eid]
                writer.writerow([eid,entry.patient_id,entry.strat_fold,y,pred,*prob])
        save_checkpoint_new(output/'best_model.pt',{'schema':'ptbxl-best-model-v1','model_state_dict':best_state,
            'model_configuration':RhythmCNNConfig().to_dict(),'classes':list(CLASS_NAMES),'best_epoch':best_epoch,
            'best_validation_macro_f1':best_score,'plan_sha256':fingerprint,'manifest_sha256':index.manifest_sha256})
        # Reload the selected best checkpoint on the SAME validation batch, never test.
        restored=torch.load(output/'best_model.pt',map_location='cpu',weights_only=True)
        reloaded=PTBXLRhythmCNN().to(device);reloaded.load_state_dict(restored['model_state_dict']);reloaded.eval()
        reference=PTBXLRhythmCNN().to(device);reference.load_state_dict(best_state);reference.eval()
        check_loader=make_loader(index,tuple(plan['partitions']['validation']['global_indices']),'validation',config,1)
        check_batch=next(iter(check_loader));check_x=check_batch['X'].to(device)
        if not all(int(fold)==9 for fold in check_batch['strat_fold']):raise RuntimeError('Checkpoint check must use validation only')
        result['checkpoint_reload_ecg_ids']=check_batch['ecg_id'].tolist()
        result['checkpoint_reload_strat_folds']=check_batch['strat_fold'].tolist()
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(check_x),reference(check_x),rtol=0,atol=0)
        result['checkpoint_reload_identical_logits']=True
        result['best_model_sha256']=file_sha256(output/'best_model.pt')
        _plot_results(output,history,best_metrics,preliminary=config.mode=='short_baseline')
    write_json_new(output/'result.json',result)
    write_json_new(output/'reproducibility.json',{'torch':str(torch.__version__),'numpy':np.__version__,
        'python':sys.version,'platform':platform.platform(),
        'package_versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        'seed':config.seed,'deterministic_algorithms':True,'threads':config.cpu_threads,'device':str(device),
        'plan_sha256':fingerprint,'manifest_sha256':index.manifest_sha256,
        'checkpoint_sha256':{p.name:file_sha256(p) for p in (output/'checkpoints').glob('*.pt')},
        'resume_boundary':'Completed epoch only; partial epoch is replayed; deterministic shuffle seed=seed+epoch'})
    return result
