"""Phase-23B read-only audit runner. No optimizers, fitting, or final-test access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import wfdb
from src.mitbih_error_audit import (require_record,predictions,verify_annotations,reconstruct,mapping_rows,
    select_examples,draw_examples,MODEL_PATHS,MODEL_HASHES,COMPARISON)
from src.mitbih_protocol import MITBIHExportIndex,load_split
from src.mitbih_readiness import verify_frozen_protocol
from src.mitbih_full_run import snapshot,csv_new
from src.models.mitbih_cnn import MITBIHBeatCNN
from src.data_loader import DEFAULT_DATA_DIR,load_recording
from src.prepare_dataset import sha256_file
from src.ptbxl_training import write_json_new
from src.mitbih_metrics import classification_metrics

RECONSTRUCT_RECORDS=('104','114','201','202','210','232')


def train232(index,split,arrays):
    """Frozen TRAIN inference only, explicitly in-sample; never use optimizer/grad."""
    from src.label_mapping import CLASS_NAMES
    result={};metrics={}
    for model,path in MODEL_PATHS.items():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        network=MITBIHBeatCNN();network.load_state_dict(saved['model_state_dict']);network.requires_grad_(False);network.eval()
        state={k:v.clone() for k,v in network.state_dict().items()};probs=[]
        with torch.inference_mode():
            for start in range(0,len(arrays['X']),32):
                logits=network(torch.from_numpy(arrays['X'][start:start+32].copy()))
                probs.extend(torch.softmax(logits,1).tolist())
        if not all(torch.equal(v,network.state_dict()[k]) for k,v in state.items()):raise ValueError('Inference changed model')
        metrics[model]=classification_metrics(arrays['y'],np.asarray(probs));rows={}
        for j,p in enumerate(probs):
            source=index.by_record['232'].global_start+j;pred=int(np.argmax(p))
            rows[source]=dict(source_index=source,partition='train',record_id='232',patient_group=index.by_record['232'].patient_group,
                original_symbol=str(arrays['original_symbols'][j]),true_class=CLASS_NAMES[int(arrays['y'][j])],
                predicted_class=CLASS_NAMES[pred],annotation_position=int(arrays['annotation_positions'][j]),
                score=float(max(p)),channel_0=str(arrays['channel_names'][0]),channel_1=str(arrays['channel_names'][1]),
                **{'p_'+c:float(p[i]) for i,c in enumerate(CLASS_NAMES)})
        result[model]=rows
    return result,metrics


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new non-overwriting audit directory is required')
    index=MITBIHExportIndex();split=load_split(index);verify_frozen_protocol(index,split)
    for model,path in MODEL_PATHS.items():
        if sha256_file(path)!=MODEL_HASHES[model]:raise ValueError('Frozen checkpoint differs')
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    pred=predictions(index,split)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    from collections import defaultdict
    raw_counts={'train':Counter(),'validation':Counter()};export_counts={'train':Counter(),'validation':Counter()}
    config_rows=[];record_rows=[];boundary_rows=[];recon_rows={};source_hashes={}
    population={};cached={};train_metrics=None;train_predictions=None
    # Every file opened below is explicitly in frozen TRAIN or VALIDATION.
    for entry in index.entries:
        rid=entry.record_id
        if entry.patient_group in split['groups']['test']:continue
        part=require_record(index,split,rid)
        for ext in ('hea','atr','dat'):
            path=DEFAULT_DATA_DIR/f'{rid}.{ext}';digest=sha256_file(path)
            if digest!=index.manifest['source_files_sha256'][path.name]:raise ValueError('Original source checksum differs')
            source_hashes[str(path)]=digest
        arrays=index.load_shard(rid,split,part)
        config=json.loads(str(arrays['preprocessing_config']))
        header=wfdb.rdheader(str(DEFAULT_DATA_DIR/rid));ann=wfdb.rdann(str(DEFAULT_DATA_DIR/rid),'atr')
        check=verify_annotations(arrays,ann,header,config)
        if tuple(header.units)!=tuple(index.manifest['shards'][list(index.by_record).index(rid)]['source_units']):
            raise ValueError('Raw physical units differ')
        raw_counts[part].update(ann.symbol);export_counts[part].update(map(str,arrays['original_symbols']))
        config_rows.append(dict(record_id=rid,partition=part,patient_group=entry.patient_group,
            channel_0=header.sig_name[0],channel_1=header.sig_name[1],raw_units=','.join(header.units),
            ADC_gain_0=header.adc_gain[0],ADC_gain_1=header.adc_gain[1],
            before_samples=config['before_samples'],after_samples=config['after_samples'],
            highpass_hz=config['filtering']['highpass_hz'],lowpass_hz=config['filtering']['lowpass_hz'],
            filter_order=config['filtering']['order'],notch=str(config['filtering']['notch_hz']),
            normalization_scope=config['normalization']['scope'],norm_scale_0=config['normalization']['scale'][0],
            norm_scale_1=config['normalization']['scale'][1],**check))
        for s,n in Counter(map(str,arrays['original_symbols'])).items():
            record_rows.append(dict(partition=part,record_id=rid,patient_group=entry.patient_group,original_symbol=s,count=n))
        for j,p in enumerate(arrays['annotation_positions']):
            if p-90<1800 or header.sig_len-(p+162)<1800:
                boundary_rows.append(dict(partition=part,record_id=rid,source_index=entry.global_start+j,
                    annotation_position=int(p),left_margin_samples=int(p-90),right_margin_samples=int(header.sig_len-(p+162)),
                    original_symbol=str(arrays['original_symbols'][j])))
            if part=='validation':
                source=entry.global_start+j
                for model in ('A','B'):
                    r=pred[model][source]
                    if (r['annotation_position']!=int(p) or r['original_symbol']!=str(arrays['original_symbols'][j])
                        or r['true_class']!='NSVFQ'[int(arrays['y'][j])] or r['patient_group']!=entry.patient_group):
                        raise ValueError('Saved prediction does not match original exported beat')
        if rid in RECONSTRUCT_RECORDS:
            record=load_recording(rid)
            recon_rows[rid]=reconstruct(record,arrays,config)
        if rid=='232':
            train_predictions,train_metrics=train232(index,split,arrays)
        # Keep only shards needed by later figures; majority freed before next record.
        cached[rid]=None
        print(f'Checked {part} {rid}: {len(arrays["y"])} beats; overlap pairs {check["overlap_pairs"]}',flush=True)
    csv_new(output/'MITBIH_SYMBOL_MAPPING_AUDIT.csv',mapping_rows(raw_counts,export_counts))
    csv_new(output/'PREPROCESSING_CONSISTENCY.csv',config_rows)
    csv_new(output/'ORIGINAL_SYMBOL_BY_RECORD.csv',record_rows)
    csv_new(output/'RECORD_EDGE_WINDOWS.csv',boundary_rows)
    write_json_new(output/'RECONSTRUCTION_CHECKS.json',recon_rows)
    write_json_new(output/'RAW_SOURCE_CHECKSUMS.json',source_hashes)
    write_json_new(output/'TRAIN_232_INSAMPLE_METRICS.json',train_metrics)
    for model in ('A','B'):csv_new(output/f'TRAIN_232_{model}_PREDICTIONS.csv',list(train_predictions[model].values()))
    # Preserve all per-class group rows; distinguish TRAIN 232 from unseen validation.
    patient_rows=[];record_metrics=[]
    for model,name in [('A','A_sqrt_balanced'),('B','B_unweighted')]:
        groups=json.loads((COMPARISON/(name+'_patient_groups.json')).read_text())
        for group,m in groups.items():
            records=[e.record_id for e in index.entries if e.patient_group==group]
            for row in m['per_class']:
                patient_rows.append(dict(model=model,partition='validation',patient_group=group,record_ids=','.join(records),
                    group_accuracy=m['accuracy'],**row))
        for row in train_metrics[model]['per_class']:
            patient_rows.append(dict(model=model,partition='train_in_sample',patient_group=index.by_record['232'].patient_group,
                record_ids='232',group_accuracy=train_metrics[model]['accuracy'],**row))
        for rid in split['summary']['validation']['record_ids']:
            chosen=[r for r in pred[model].values() if r['record_id']==rid]
            cm=np.zeros((5,5),dtype=int)
            for r in chosen:cm['NSVFQ'.index(r['true_class']),'NSVFQ'.index(r['predicted_class'])]+=1
            record_metrics.append(dict(model=model,record_id=rid,patient_group=chosen[0]['patient_group'],beats=len(chosen),
                accuracy=float(np.trace(cm)/cm.sum()),errors=int(cm.sum()-np.trace(cm)),F_FP=int(cm[:,3].sum()-cm[3,3]),
                S_FN=int(cm[1].sum()-cm[1,1]),Q_FN=int(cm[4].sum()-cm[4,4]),channel_0=chosen[0]['channel_0'],channel_1=chosen[0]['channel_1']))
    csv_new(output/'MITBIH_PATIENT_ERROR_ANALYSIS.csv',patient_rows)
    csv_new(output/'RECORD_ERROR_ANALYSIS.csv',record_metrics)
    # Select actual rows once and freeze the selection before reading waveform shapes.
    selected={};manifest=[]
    for category in ('S_to_N','Q_to_N','F_false_positive','V_correct','N_correct','F_correct'):
        items=[];seen=set()
        for model in ('A','B'):
            for row in select_examples(list(pred[model].values()),category,limit=3):
                if row['source_index'] not in seen:items.append(row);seen.add(row['source_index'])
        selected[category]=items
    trainrows=list(train_predictions['A'].values())
    s_correct=[r for r in trainrows if r['true_class']==r['predicted_class']=='S']
    if s_correct:
        median=float(np.median([r['score'] for r in s_correct]));sr=min(s_correct,key=lambda r:(abs(r['score']-median),r['source_index']))
        vr=next(r for r in selected['S_to_N'] if r['original_symbol']==sr['original_symbol'])
        selected['TRAIN_232_vs_VALIDATION_S']=[sr,vr]
    for model in ('A','B'):pred[model].update(train_predictions[model])
    # Add boundary and actual lead-order examples; no causal edge diagnosis implied.
    first=next(r for r in boundary_rows if r['partition']=='validation')
    selected['record_boundary_example']=[pred['A'][first['source_index']]]
    needed={r['record_id'] for items in selected.values() for r in items}
    for rid in needed:cached[rid]=index.load_shard(rid,split,require_record(index,split,rid))
    captions={
        'S_to_N':'S → N: actual validation errors, original A/a when available',
        'Q_to_N':'Q → N: paced /, paced-fusion f and unclassifiable Q subtypes',
        'F_false_positive':'False F: true non-F beats in records 201/202 and 210',
        'V_correct':'Correct V: available validation true positives',
        'N_correct':'Correct N: available validation true positives',
        'F_correct':'Rare F true positives: one available per model; not evidence of reliability',
        'TRAIN_232_vs_VALIDATION_S':'Same original S subtype: TRAIN 232 (in-sample) versus unseen validation error',
        'record_boundary_example':'Near-record-edge processed window: risk tag, not proof of a filtering artifact'}
    figures=output/'figures';figures.mkdir()
    for category,items in selected.items():
        # At most 3 columns per sheet to keep both leads and score labels readable.
        for start in range(0,len(items),3):
            subset=items[start:start+3];filename=f'{category}_{start//3+1:02d}.png'
            draw_examples(figures/filename,subset,cached,pred,captions[category])
            for row in subset:
                manifest.append(dict(figure=filename,category=category,partition=row['partition'],record_id=row['record_id'],
                    patient_group=row['patient_group'],source_index=row['source_index'],annotation_position=row['annotation_position'],
                    original_symbol=row['original_symbol'],true_class=row['true_class'],
                    A_prediction=pred['A'][row['source_index']]['predicted_class'],A_score=pred['A'][row['source_index']]['score'],
                    B_prediction=pred['B'][row['source_index']]['predicted_class'],B_score=pred['B'][row['source_index']]['score']))
    csv_new(output/'PLOT_EXAMPLE_MANIFEST.csv',manifest)
    write_json_new(output/'audit_result.json',dict(status='passed',records_checked=len(config_rows),
        exported_train_validation_beats_checked=sum(r['exported_beats'] for r in config_rows),
        raw_annotation_counts={k:dict(v) for k,v in raw_counts.items()},export_symbol_counts={k:dict(v) for k,v in export_counts.items()},
        reconstructed_records=list(recon_rows),figures=len(list(figures.glob('*.png'))),plot_examples=len(manifest),
        models_sha256=MODEL_HASHES,training_steps=0,final_test_read=False,ptbxl_data_read=False,
        records_read=index.read_records,example_selection='Deterministic record/symbol strata then closest-to-category-median score; no waveform-driven selection',
        edge_tag_seconds=5,edge_tag_is_not_an_artifact_diagnosis=True,annotation_index_in_window=90))
    print('Audit completed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();run(args.output_dir)
