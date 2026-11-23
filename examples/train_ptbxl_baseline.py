"""Phase-17 entry point: print preflight, then run only a bounded default baseline."""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
from src.ptbxl_protocol import PTBXLExportIndex
from src.ptbxl_training import TrainingConfig,build_plan,prepare_device,train_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/ptbxl_phase17_baseline.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/model_training/phase17_baseline'))
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--allow-full-training',action='store_true',help='For a separately approved future full experiment only')
    args=parser.parse_args()
    config=TrainingConfig(**json.loads(args.config.read_text()))
    if config.mode=='full' and not args.allow_full_training:
        parser.error('Full experiments require separately approved --allow-full-training')
    index=PTBXLExportIndex()
    plan=build_plan(index,config)
    # Print exact planned counts/device/config BEFORE training or writing outputs.
    print(json.dumps({'configuration':asdict(config),'device_verified':str(prepare_device(config)),
        'train':{k:v for k,v in plan['partitions']['train'].items() if k not in ('global_indices','ecg_ids','patient_ids')},
        'validation':{k:v for k,v in plan['partitions']['validation'].items() if k not in ('global_indices','ecg_ids','patient_ids')},
        'class_weights':plan['class_weights'],'test_fold_used':False},indent=2),flush=True)
    if args.dry_run:return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    result=train_model(config,args.output_dir,index=index,resume_from=args.resume_from,allow_full_training=args.allow_full_training)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
