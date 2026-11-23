"""Phase-21 bounded CPU benchmark; does not call the full trainer."""
import argparse
from pathlib import Path
from src.mitbih_readiness import benchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.output_dir)
