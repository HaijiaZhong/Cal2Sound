# Cal2Sound

Calcium-aware Neural-to-Sound Reconstruction from Wide-field Auditory Cortical Imaging

Cal2Sound reconstructs sound from wide-field auditory cortical calcium activity and evaluates the preservation of pitch information in reconstructed audio.

**Status:** Repository skeleton for a planned ICASSP 2027 submission. Project and paper details are provisional; implementations and results are not included.

## Overview

This project studies neural-to-sound reconstruction from wide-field auditory cortical calcium imaging, with a focus on pitch/F0 preservation in reconstructed audio.

## Method

```text
Wide-field calcium activity
        ↓
Neural adaptor
        ↓
CalGRU
        ↓
ResTCN
        ↓
Acoustic latent representation
        ↓
Pretrained audio decoder
        ↓
Reconstructed sound
```

The `cal2sound/` package reserves model and decoder interfaces. Implementation will be migrated from the paper codebase.

## Audio Reconstruction Examples

| Example | Original | Reconstruction | F0 visualization |
|---|---|---|---|
| Melody example | TBD | TBD | TBD |
| Speech example | TBD | TBD | TBD |

Examples will be organized in `demos/`. `docs/` contains a placeholder project page; no audio or figures are included yet.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are minimal and unpinned; versions and additional packages will be confirmed during migration. The placeholder workflows are not functional yet.

## Inference

`scripts/inference.py` reserves the inference entry point. Input formats, decoder setup, checkpoint loading, and CLI arguments are TBD. Running it currently raises `NotImplementedError`.

## Training

`scripts/train.py` is a placeholder. `configs/cal2sound.yaml` reserves configuration fields, with unconfirmed parameters set to `null`. Training is not implemented.

## Pitch Evaluation

`evaluation/pitch_metrics.py` reserves PDA, median F0 error, and note accuracy interfaces. Definitions, units, alignment, voicing rules, and tolerances are TBD.

`evaluation/f0_estimators.py` reserves a common interface for CREPE, SwiftF0, PESTO, pYIN, and RMVPE, without importing or installing them. `evaluation/audio_metrics.py` and `scripts/evaluate.py` are also placeholders.

## Analysis

`analysis/` contains title-only notebooks for Figure 2 (Pitch Preservation Analysis) and Figure 3 (Factors Affecting Pitch Decodability). Analysis code, input schemas, and figures will be added later.

## Data

The raw wide-field calcium imaging dataset is not included in the repository at this stage.

Future additions may include preprocessing code, model implementation, evaluation code, and representative reconstruction examples. Dataset availability and sharing permissions remain to be confirmed; public release of raw neural data is not assumed. See `data/README.md`.

## Pretrained Models

No checkpoints or external pretrained models are included. Availability, decoder requirements, and loading instructions are TBD. See `checkpoints/README.md`.

## Citation

Provisional citation placeholder only; publication and acceptance are not confirmed. Title and authors are TBD.

```bibtex
@inproceedings{cal2sound2027,
  title     = {Cal2Sound: TBD},
  author    = {TBD},
  booktitle = {ICASSP},
  year      = {2027}
}
```

## License

Repository code is provided under the [MIT License](LICENSE). The copyright holder is TBD and must be confirmed before publication.

The license for the neural dataset and pretrained external models may be governed by their respective original licenses. This repository does not claim ownership of third-party models or datasets.
