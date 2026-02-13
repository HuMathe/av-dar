# Auralization Module

This module provides inference and auralization capabilities for trained AV-DAR models. Given a trained checkpoint, it renders Room Impulse Responses (RIRs) at arbitrary source-listener positions in the scene and optionally convolves them with input audio — enabling spatial audio rendering at novel positions without retraining.

## Directory Structure

```
avdar/auralization/
├── __init__.py     # package init
├── audio.py        # audio I/O: load, save, convolve, normalize
├── engine.py       # InferenceEngine: wraps RirRenderer for inference
└── README.md       # this file

auralize.py             # CLI entry point (at repo root, alongside train.py / evaluate.py)
```

## Quick Start

### Render a single RIR

```bash
python auralize.py \
    --config_dir ./outputs/HAA-Classroom-16K/2025-10-20_12-00-00 \
    --source_xyz 1.0 2.0 0.5 \
    --listener_xyz 3.0 1.5 0.5 \
    --output_dir ./auralized_output
```

This produces:
- `output_rir.wav` — the rendered RIR
- `metadata.json` — source/listener positions, sample rate, RIR length

### Auralize an input audio file

```bash
python auralize.py \
    --config_dir ./outputs/HAA-Classroom-16K/2025-10-20_12-00-00 \
    --source_xyz 1.0 2.0 0.5 \
    --listener_xyz 3.0 1.5 0.5 \
    --input_audio ./dry_speech.wav \
    --output_dir ./auralized_output
```

This additionally produces:
- `output_auralized.wav` — the input audio convolved with the rendered RIR

### Batch render multiple positions

Create a `positions.json` file:

```json
[
    {"source_xyz": [1.0, 2.0, 0.5], "listener_xyz": [3.0, 1.5, 0.5]},
    {"source_xyz": [1.0, 2.0, 0.5], "listener_xyz": [5.0, 3.0, 0.5]},
    {"source_xyz": [2.0, 1.0, 0.5], "listener_xyz": [4.0, 2.5, 0.5],
     "source_orientation": [0, 0, 0.707, 0.707]}
]
```

```bash
python auralize.py \
    --config_dir ./outputs/HAA-Classroom-16K/2025-10-20_12-00-00 \
    --positions_file ./positions.json \
    --output_dir ./auralized_output
```

Produces `position_0000_rir.wav`, `position_0001_rir.wav`, etc.

## CLI Reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config_dir` | Yes | — | Path to a Hydra training run directory |
| `--state_dict_name` | No | `weight_final.pt` | Checkpoint filename within `config_dir` |
| `--device` | No | `cuda:0` | Compute device |
| `--source_xyz X Y Z` | * | — | Source position in 3D |
| `--listener_xyz X Y Z` | * | — | Listener position in 3D |
| `--source_orientation QX QY QZ QW` | No | identity | Source rotation quaternion |
| `--positions_file` | * | — | JSON with list of position dicts |
| `--input_audio` | No | — | Dry audio file to convolve with RIR |
| `--output_dir` | Yes | — | Output directory for results |

\* Either `--source_xyz` + `--listener_xyz` or `--positions_file` is required (not both).

## Python API

The `InferenceEngine` class can be used directly in scripts:

```python
from avdar.auralization.engine import InferenceEngine
from avdar.auralization.audio import load_audio, save_audio

# ... build config, renderer, path_sampler, dataset as in auralize.py ...

engine = InferenceEngine(config, renderer, path_sampler, dataset, device)

# Render RIR
rir, metadata = engine.render_rir(
    source_xyz=[1.0, 2.0, 0.5],
    listener_xyz=[3.0, 1.5, 0.5]
)

# Auralize
audio, sr = load_audio("speech.wav", target_sr=16000)
auralized, rir, metadata = engine.auralize(
    source_xyz=[1.0, 2.0, 0.5],
    listener_xyz=[3.0, 1.5, 0.5],
    input_audio=audio, sr=sr
)
save_audio(auralized, "output.wav", sr)

# Batch render
results = engine.batch_render([
    {"source_xyz": [1,2,0.5], "listener_xyz": [3,1.5,0.5]},
    {"source_xyz": [1,2,0.5], "listener_xyz": [5,3,0.5]},
])
```

## Output Format

Each run produces:

| File | Description |
|---|---|
| `*_rir.wav` | Rendered Room Impulse Response (peak-normalized, int16) |
| `*_auralized.wav` | Input audio convolved with RIR (only if `--input_audio` provided) |
| `metadata.json` | Positions, sample rate, RIR length, file paths |

### metadata.json example

```json
[
    {
        "source_xyz": [1.0, 2.0, 0.5],
        "listener_xyz": [3.0, 1.5, 0.5],
        "source_orientation": [0, 0, 0, 1],
        "sample_rate": 16000,
        "rir_length_samples": 32000,
        "rir_duration_seconds": 2.0,
        "rir_path": "./auralized_output/output_rir.wav"
    }
]
```

## Architecture

The auralization pipeline uses the same trained components as the evaluation loop:

1. **Config loading** — Hydra `initialize_config_dir` + `compose` (same as `evaluate.py`)
2. **Model building** — `build_from_config()` with `inference_only=True`
3. **Beam tracing** — `SpecularPathSampler` traces specular reflection paths from source through the scene mesh to the listener
4. **Neural rendering** — `RirRenderer.forward()` combines early reflections (specular MLP), diffuse field (positional encoding network), and late reverberation to produce the full RIR
5. **Convolution** — FFT-based convolution of dry audio with the rendered RIR

No modifications to any existing files are required.
