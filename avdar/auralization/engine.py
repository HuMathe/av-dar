import torch
import numpy as np
import logging

from scipy.spatial.transform import Rotation

from ..model.renderer import RirRenderer
from ..geometry.pathspace import SpecularPathSampler
from .audio import convolve_rir, normalize_audio

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps a trained RirRenderer for inference at arbitrary positions."""

    def __init__(self, config, renderer, path_sampler, dataset, device):
        self.config = config
        self.renderer = renderer
        self.path_sampler = path_sampler
        self.dataset = dataset
        self.device = device

        self.renderer.eval()

        self.sample_rate = config.dataset.sample_rate
        self.speed_of_sound = config.dataset.options.speed_of_sound

    @torch.no_grad()
    def render_rir(self, source_xyz, listener_xyz, source_orientation=None):
        """Render RIR at a source-listener position. Returns (rir, metadata)."""
        source_xyz_np = np.array(source_xyz, dtype=np.float32)
        listener_xyz_np = np.array(listener_xyz, dtype=np.float32)

        if source_orientation is None:
            source_quat = np.array([0, 0, 0, 1], dtype=np.float32)
        else:
            source_quat = np.array(source_orientation, dtype=np.float32)

        # Rotation matrix from quaternion
        rotation = None
        try:
            rot_mat = Rotation.from_quat(source_quat).as_matrix()
            rotation = torch.from_numpy(rot_mat.astype(np.float32)).to(self.device)
        except Exception:
            pass

        # Beam tracing
        mc_samples = self.path_sampler.fast_sample(source_xyz_np, listener_xyz_np)

        source_t = torch.tensor(source_xyz_np, dtype=torch.float32).to(self.device)
        listener_t = torch.tensor(listener_xyz_np, dtype=torch.float32).to(self.device)
        quat_t = torch.tensor(source_quat, dtype=torch.float32).to(self.device)

        # Forward pass — same signature as eval_step_rir in run.py
        pred_dict = self.renderer(
            None, None, None, None,
            rotation, source_t, listener_t, quat_t,
            mc_samples=mc_samples,
        )

        rir = pred_dict['rir_full'].detach().cpu().numpy()

        metadata = {
            'source_xyz': source_xyz_np.tolist(),
            'listener_xyz': listener_xyz_np.tolist(),
            'source_orientation': source_quat.tolist(),
            'sample_rate': self.sample_rate,
            'rir_length_samples': len(rir),
            'rir_duration_seconds': float(len(rir) / self.sample_rate),
        }

        logger.info(f"Rendered RIR: {len(rir)} samples ({len(rir)/self.sample_rate:.3f}s)")
        return rir, metadata

    @torch.no_grad()
    def auralize(self, source_xyz, listener_xyz, input_audio, sr,
                 source_orientation=None):
        """Convolve input audio with rendered RIR. Returns (auralized, rir, metadata)."""
        if sr != self.sample_rate:
            logger.warning(
                f"Input sr ({sr}) != model sr ({self.sample_rate}). "
                f"Ensure sample rates match for correct results."
            )

        rir, metadata = self.render_rir(source_xyz, listener_xyz, source_orientation)

        auralized = convolve_rir(input_audio, rir)
        auralized = normalize_audio(auralized)

        metadata['input_audio_samples'] = len(input_audio)
        metadata['output_audio_samples'] = len(auralized)

        return auralized, rir, metadata

    @torch.no_grad()
    def batch_render(self, positions):
        """Render RIRs for multiple source-listener pairs."""
        results = []
        for i, pos in enumerate(positions):
            logger.info(f"Rendering {i+1}/{len(positions)}")
            rir, metadata = self.render_rir(
                source_xyz=pos['source_xyz'],
                listener_xyz=pos['listener_xyz'],
                source_orientation=pos.get('source_orientation', None),
            )
            results.append({'rir': rir, 'metadata': metadata})
        return results
