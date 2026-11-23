"""Single explicitly authorized Phase-23A loss-only experiment."""
import argparse
import logging
from pathlib import Path
from src.mitbih_unweighted import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    print(run(args.output_dir,args.audit_dir),flush=True)
