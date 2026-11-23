"""Future approved full MIT-BIH training. Phase 21 does not execute this command."""
import argparse
import json
import logging
from pathlib import Path
from src.mitbih_training import MITBIHTrainingConfig, train_model
from src.mitbih_protocol import MITBIHExportIndex, load_split
from src.mitbih_readiness import verify_frozen_protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--approve-full-training', action='store_true',
                        help='Use ONLY after separate user approval for Phase 22')
    args = parser.parse_args()
    if not args.approve_full_training:
        parser.error('Full training requires separate approval; readiness does not grant it')
    config = MITBIHTrainingConfig(**json.loads(args.config.read_text()))
    index = MITBIHExportIndex(); split = load_split(index)
    verify_frozen_protocol(index, split)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    result = train_model(config, args.output_dir, index=index, split=split,
                         resume_from=args.resume_from, allow_full_training=True)
    print(json.dumps(result, indent=2))
