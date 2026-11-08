"""Bounded Phase-13 pilot. Missing real waveforms are reported, never synthesized."""
import argparse
import json
from pathlib import Path
import shlex
import numpy as np
from src.ptbxl_metadata import DEFAULT_METADATA_DIR,load_ptb_metadata
from src.ptbxl_rhythm import (DEFAULT_WAVEFORM_ROOT,CLASS_NAMES,preprocessing_config,select_pilot,
    pilot_files,download_pilot,load_rhythm_record,preprocess_rhythm_record,export_pilot_record)
from src.prepare_dataset import sha256_file
from src.preprocessing_comparison import save_comparison_figure
import matplotlib.pyplot as plt


def write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as output:
        json.dump(data,output,indent=2,sort_keys=True,allow_nan=False)


def plot_pilot_record(processed):
    """Visualize actual supplied raw/filtered/normalized data; no medical diagnosis."""
    leads=('I','II','V1','V5')
    fig,axes=plt.subplots(len(leads),3,sharex=True,squeeze=False,figsize=(17,11))
    time=np.arange(1000)/processed.loaded.sampling_frequency
    for row,name in enumerate(leads):
        index=processed.loaded.channel_names.index(name)
        for col,(title,values,unit) in enumerate([
            ('Raw',processed.loaded.raw[:,index],'mV'),('Filtered',processed.filtered[:,index],'mV'),
            ('Z-score',processed.X[index],'dimensionless')]):
            ax=axes[row,col]
            ax.plot(time,values,linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel(f'{name} ({unit})')
            ax.set_xlabel('Time (s)')
            ax.set_xlim(0,10)
            ax.grid(alpha=0.25)
    identity=processed.loaded.identity
    fig.suptitle(f'PTB-XL ecg_id={identity["ecg_id"]} | metadata rhythm={identity["label"]} | fs=100 Hz | preprocessing inspection only')
    fig.tight_layout()
    return fig


def download_commands(selection,path):
    """Exact bounded curl commands for a terminal with network access; no overwrite."""
    lines=['#!/usr/bin/env bash','set -euo pipefail','# Run from the ECG-APP project root. Only the selected 16 files are requested.']
    for item in pilot_files(selection):
        target='data/ptbxl/'+item['relative_path']
        lines += [f'mkdir -p {shlex.quote(str(Path(target).parent))}',
                  f'if [ ! -e {shlex.quote(target)} ]; then',
                  '  curl -fL --retry 3 --connect-timeout 15 --max-time 90 --no-clobber '+
                  f'-o {shlex.quote(target)} {shlex.quote(item["url"])}','fi']
    with Path(path).open('x') as output:
        output.write('\n'.join(lines)+'\n')


def run_pilot(metadata_dir=DEFAULT_METADATA_DIR,waveform_root=DEFAULT_WAVEFORM_ROOT,
              output_dir=Path('data/processed/ptbxl_rhythm_pilot'),
              artifact_dir=Path('results/preprocessing/phase13_ptbxl_pilot'),report_dir=Path('.'),download=True):
    data=load_ptb_metadata(metadata_dir)
    selection=select_pilot(data,per_class=2)
    output_dir=Path(output_dir)
    artifact_dir=Path(artifact_dir)
    report_dir=Path(report_dir)
    # Refuse to overwrite any existing processed pilot directory.
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'recordings').mkdir()
    artifact_dir.mkdir(parents=True,exist_ok=True)
    (artifact_dir/'plots').mkdir(exist_ok=True)
    report_dir.mkdir(parents=True,exist_ok=True)
    config=preprocessing_config()
    write_json(report_dir/'PTBXL_PREPROCESSING_CONFIG.json',config)
    download_commands(selection,artifact_dir/'download_selected_pilot.sh')
    attempts=download_pilot(selection,waveform_root) if download else []
    records=[]
    plotted=set()
    loaded_count=0
    for identity in selection:
        item={'selection':identity}
        try:
            loaded=load_rhythm_record(data,identity['ecg_id'],waveform_root)
            loaded_count+=1
            processed=preprocess_rhythm_record(loaded)
            filename=f'recordings/{identity["ecg_id"]:05d}_rhythm.npz'
            export_pilot_record(processed,output_dir/filename)
            item.update({'status':'processed','file':filename,'sha256':sha256_file(output_dir/filename),
                         'shape':list(processed.X.shape),'source_hashes':loaded.source_hashes,
                         'normalization_parameters':processed.configuration['fitted_normalization']})
            if identity['label'] not in plotted:
                fig=plot_pilot_record(processed)
                try:
                    save_comparison_figure(fig,artifact_dir/'plots'/f'{identity["label"]}_{identity["ecg_id"]:05d}_preprocessing.png')
                finally:
                    plt.close(fig)
                plotted.add(identity['label'])
        except FileNotFoundError as exc:
            item.update({'status':'missing_waveforms','error':str(exc)})
        except Exception as exc:
            item.update({'status':'failed_validation_or_processing','error':f'{type(exc).__name__}: {exc}'})
        records.append(item)
        print(identity['ecg_id'],identity['label'],item['status'],flush=True)
    complete_pairs=sum(all(any(a['relative_path']==row['filename_lr']+suffix and a['status']=='downloaded'
                              for a in attempts) for suffix in ('.hea','.dat')) for row in selection)
    missing=[{**item,'local_path':str(Path(waveform_root)/item['relative_path'])}
             for item in pilot_files(selection) if not (Path(waveform_root)/item['relative_path']).is_file()]
    processed_count=sum(r['status']=='processed' for r in records)
    manifest={'schema_version':'ptbxl-rhythm-small-pilot-v1','dataset_version':'1.0.3',
        'status':'complete_small_pilot' if processed_count==len(selection) else
                 ('blocked_no_real_waveforms_processed' if processed_count==0 else 'partial_small_pilot'),
        'selection_rule':config['selection_rule'],'class_names':list(CLASS_NAMES),'selected_records':selection,
        'metadata_provenance':data.provenance,'configuration':config,'download_attempts':attempts,
        'records':records,'missing_files':missing,
        'summary':{'selected_waveforms':len(selection),'downloaded_files_this_run':sum(a['status']=='downloaded' for a in attempts),
            'complete_waveform_pairs_downloaded_this_run':complete_pairs,'real_waveforms_read':loaded_count,
            'successfully_processed':processed_count,'processed_classes':sorted({r['selection']['label'] for r in records if r['status']=='processed'}),
            'plots_generated':len(plotted),'expected_X_shape':[12,1000]},
        'waveform_root':str(Path(waveform_root).relative_to(Path.cwd()) if Path(waveform_root).is_absolute() and Path(waveform_root).is_relative_to(Path.cwd()) else waveform_root),
        'processed_output_dir':str(output_dir),'artifacts_dir':str(artifact_dir),
        'code_sha256':{name:sha256_file(Path(__file__).resolve().parents[1]/'src'/name)
                       for name in ('ptbxl_rhythm.py','ptbxl_metadata.py','preprocessing.py')},
        'scope':'Pilot only; no full dataset, no splits, no model training or evaluation',
        'real_waveform_verification_complete':processed_count==len(selection)}
    write_json(report_dir/'PTBXL_PILOT_MANIFEST.json',manifest)
    write_json(output_dir/'pilot_manifest.json',manifest)
    write_json(artifact_dir/'pilot_result.json',manifest)
    print(json.dumps(manifest['summary'],indent=2),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir',type=Path,default=DEFAULT_METADATA_DIR)
    parser.add_argument('--waveform-root',type=Path,default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument('--output-dir',type=Path,default=Path('data/processed/ptbxl_rhythm_pilot'))
    parser.add_argument('--artifact-dir',type=Path,default=Path('results/preprocessing/phase13_ptbxl_pilot'))
    parser.add_argument('--report-dir',type=Path,default=Path('.'))
    parser.add_argument('--no-download',action='store_true')
    args=parser.parse_args()
    run_pilot(args.metadata_dir,args.waveform_root,args.output_dir,args.artifact_dir,args.report_dir,not args.no_download)


if __name__=='__main__':
    main()
