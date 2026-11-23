"""Only Phase-18 Checkpoint A; this entry point cannot start a full training run."""
import argparse
import json
import logging
from pathlib import Path
from src.ptbxl_readiness import run_readiness


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/model_training/phase18_checkpoint_a'))
    parser.add_argument('--config-path',type=Path,default=Path('configs/ptbxl_phase18_full_v1.json'))
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    report=run_readiness(args.output_dir,args.config_path)
    print(json.dumps({k:report[k] for k in ('full_training_started','recommended_batch_size','recommended_cpu_threads',
        'recommended_projection','sampled_peak_rss_bytes','memory_2x_headroom_observed','disk_reserve_fits')},indent=2),flush=True)


if __name__=='__main__':main()
