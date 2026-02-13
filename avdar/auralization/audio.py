import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal

import librosa

import logging

logger = logging.getLogger(__name__)


def load_audio(path, target_sr=None):
    """Load audio file and optionally resample. Returns (audio, sr)."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    logger.info(f"Loaded {path} (sr={sr}, {len(audio)/sr:.2f}s)")
    return audio, sr


def save_audio(audio, path, sr):
    """Save audio signal as int16 .wav file."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    wavfile.write(path, sr, audio_int16)
    logger.info(f"Saved {path} (sr={sr}, {len(audio)/sr:.2f}s)")


def convolve_rir(audio, rir):
    """FFT-based convolution of audio with RIR, truncated to input length."""
    convolved = scipy.signal.fftconvolve(audio, rir, mode='full')
    return convolved[:len(audio)]


def normalize_audio(audio, target_peak=0.95):
    """Peak-normalize audio signal."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio * (target_peak / peak)
