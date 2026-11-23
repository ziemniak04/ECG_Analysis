"""Phase-18C controlled loss study; no test-fold waveform access."""
import argparse
import json
import logging
from pathlib import Path
from src.ptbxl_optimization import prepare, smoke, train_variant, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'train'))
    parser.add_argument('--root', type=Path,
        default=Path('results/model_training/phase18c_optimization/run_001'))
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--destination')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--prior-audit', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'prepare':
        result = prepare(args.root)
    else:
        if args.variant is None:
            parser.error('--variant required; original A may never be retrained')
        if args.stage == 'smoke':
            result = smoke(args.root, args.variant)
        else:
            for variant in VARIANTS:
                path = args.root/f'smoke_{variant}_audit.json'
                if not path.is_file() or json.loads(path.read_text())['status'] != 'passed':
                    parser.error('Both loss smoke tests must pass before full training')
            result = train_variant(args.root, args.variant, destination=args.destination,
                                   resume_from=args.resume_from, prior_audit=args.prior_audit)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
