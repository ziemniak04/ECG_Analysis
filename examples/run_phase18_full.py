"""Approved full Phase-18 run, with preflight separated from long CPU execution."""
import argparse
import json
import logging
from pathlib import Path
from src.ptbxl_full_run import preflight,execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/ptbxl_phase18_full_v1.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/model_training/phase18_full/run_001'))
    parser.add_argument('--audit-dir',type=Path,default=Path('results/model_training/phase18_full/audit_run_001'))
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--execute-approved-full-training',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--prior-audit-dir',type=Path)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.preflight_only:result=preflight(args.config,args.output_dir,args.audit_dir)
    elif args.execute_approved_full_training:
        result=execute(args.config,args.output_dir,args.audit_dir,resume_from=args.resume_from,prior_audit=args.prior_audit_dir)
    else:parser.error('Choose --preflight-only or explicit --execute-approved-full-training')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
