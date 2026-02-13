import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from avdar.core.base_config import BaseConfig
from avdar.core.io import build_from_config
from avdar.auralization.engine import InferenceEngine
from avdar.auralization.audio import load_audio, save_audio, normalize_audio
from avdar.geometry.pathspace import SpecularPathSampler
from avdar.model.renderer import RirRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render RIRs and auralize audio at arbitrary positions'
    )
    parser.add_argument('--config_dir', type=str, required=True,
                        help='Hydra training run directory')
    parser.add_argument('--state_dict_name', type=str, default='weight_final.pt',
                        help='Checkpoint filename within config_dir')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--source_xyz', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'))
    parser.add_argument('--listener_xyz', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'))
    parser.add_argument('--source_orientation', type=float, nargs=4, default=None,
                        metavar=('QX', 'QY', 'QZ', 'QW'),
                        help='Quaternion [x,y,z,w], default: identity')

    parser.add_argument('--positions_file', type=str, default=None,
                        help='JSON file with list of position dicts')
    parser.add_argument('--input_audio', type=str, default=None,
                        help='Dry audio file to auralize')
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


def main(config_dir, state_dict_name, device, positions, input_audio_path,
         output_dir):

    # Load config (same pattern as evaluate.py)
    with hydra.initialize_config_dir(
        config_dir=str(pathlib.Path(config_dir).absolute()),
        version_base="1.2"
    ):
        config = hydra.compose(config_name='config', overrides=[
            f'device={device}',
            f'state_dict_path={pathlib.Path(config_dir) / state_dict_name}',
            'no_terminal=True',
        ])

    cache_dir = (
        pathlib.Path(config.working_dir)
        / (config.dataset.name + '_' + config.dataset.scene_name)
    )

    # Build model and datasets
    build_dict = build_from_config(
        config, working_dir=config.working_dir,
        cache_dir=cache_dir, resume=True, inference_only=True,
    )

    dataset = (
        build_dict.get('dataset_inference')
        or build_dict.get('dataset_test')
        or build_dict.get('dataset_val')
        or build_dict['dataset_train']
    )

    rir_renderer: RirRenderer = build_dict['rir_renderer'].to(device)
    rir_renderer.eval()

    # Path sampler
    max_path_length = config.train.max_bounce
    path_sampler = SpecularPathSampler.from_config(
        config.train['sampler_opts'],
        max_path_length,
        dataset.get_mesh_path(),
    )

    engine = InferenceEngine(
        config=config,
        renderer=rir_renderer,
        path_sampler=path_sampler,
        dataset=dataset,
        device=device,
    )

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load input audio if provided
    input_audio = None
    input_sr = None
    if input_audio_path is not None:
        input_audio, input_sr = load_audio(
            input_audio_path, target_sr=config.dataset.sample_rate
        )

    # Render
    all_metadata = []
    for i, pos in enumerate(positions):
        prefix = f"position_{i:04d}" if len(positions) > 1 else "output"
        logger.info(
            f"[{i+1}/{len(positions)}] "
            f"src={pos['source_xyz']} -> lst={pos['listener_xyz']}"
        )

        if input_audio is not None:
            auralized, rir, metadata = engine.auralize(
                source_xyz=pos['source_xyz'],
                listener_xyz=pos['listener_xyz'],
                input_audio=input_audio,
                sr=input_sr,
                source_orientation=pos.get('source_orientation'),
            )
            auralized_path = output_path / f"{prefix}_auralized.wav"
            save_audio(auralized, str(auralized_path), input_sr)
            metadata['auralized_audio_path'] = str(auralized_path)
        else:
            rir, metadata = engine.render_rir(
                source_xyz=pos['source_xyz'],
                listener_xyz=pos['listener_xyz'],
                source_orientation=pos.get('source_orientation'),
            )

        rir_path = output_path / f"{prefix}_rir.wav"
        save_audio(normalize_audio(rir), str(rir_path), config.dataset.sample_rate)
        metadata['rir_path'] = str(rir_path)
        all_metadata.append(metadata)

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"Done. {len(positions)} RIR(s) saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()

    has_single = args.source_xyz is not None and args.listener_xyz is not None
    has_batch = args.positions_file is not None

    if not has_single and not has_batch:
        print("Error: specify --source_xyz + --listener_xyz, or --positions_file",
              file=sys.stderr)
        sys.exit(1)
    if has_single and has_batch:
        print("Error: use --source_xyz/--listener_xyz or --positions_file, not both",
              file=sys.stderr)
        sys.exit(1)

    if has_single:
        positions = [{
            'source_xyz': args.source_xyz,
            'listener_xyz': args.listener_xyz,
            'source_orientation': args.source_orientation,
        }]
    else:
        with open(args.positions_file, 'r') as f:
            positions = json.load(f)

    main(
        config_dir=args.config_dir,
        state_dict_name=args.state_dict_name,
        device=args.device,
        positions=positions,
        input_audio_path=args.input_audio,
        output_dir=args.output_dir,
    )
