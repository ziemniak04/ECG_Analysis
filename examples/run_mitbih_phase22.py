"""Execute the single explicitly approved Phase-22 experiment, with preflight."""
import argparse
import json
import logging
from pathlib import Path
from src.mitbih_training import MITBIHTrainingConfig
from src.mitbih_full_run import run_approved

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--approved',action='store_true')
    args=parser.parse_args()
    if not args.approved:parser.error('Separate explicit user approval required')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    config=MITBIHTrainingConfig(**json.loads(Path('PHASE21_MITBIH_TRAINING_CONFIG.json').read_text()))
    print(json.dumps(run_approved(config,args.audit_dir,args.output_dir),indent=2),flush=True)
