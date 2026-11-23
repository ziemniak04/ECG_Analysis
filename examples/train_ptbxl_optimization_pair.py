"""Run exactly B then C after passed preflight/smokes; never retrain original A."""
import argparse
import json
import logging
from pathlib import Path
from src import ptbxl_training as trainer
from src.ptbxl_full_run import atomic_status
from src.ptbxl_optimization import VARIANTS, train_variant


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/model_training/phase18c_optimization/run_001'))
    args=parser.parse_args();root=args.root
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    for variant in VARIANTS:
        if json.loads((root/f'smoke_{variant}_audit.json').read_text())['status']!='passed':
            raise ValueError('Both real smoke tests must pass before execution')
    trainer.write_json_new(root/'pair_started.json',{'sequence':list(VARIANTS),'original_A_retrained':False})
    results={}
    try:
        for variant in VARIANTS:
            atomic_status(root,status='training',variant=variant)
            results[variant]=train_variant(root,variant)
        trainer.write_json_new(root/'pair_result.json',results)
        atomic_status(root,status='both_new_variants_completed',variants=list(results))
    except BaseException as exc:
        atomic_status(root,status='stopped',error=f'{type(exc).__name__}: {exc}',completed_variants=list(results))
        raise


if __name__=='__main__':main()
