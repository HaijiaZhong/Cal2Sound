# Cal2Sound

Cal2Sound reconstructs sound from wide-field auditory cortical calcium activity by mapping neural signals to Music2Latent acoustic latents. This repository contains the model, training and reconstruction code, paired audio demos, and F0 evaluation tools.

The release includes **700 original/reconstructed audio pairs from 100 ZJM folds** (`S1–S10 × M1–M10`), with seven conditions per fold: clean, three SM conditions, and three TM conditions. Audio files occupy approximately 583 MB in total.

Neural input datasets and trained adaptor weights are not distributed. Listening to the demos and evaluating their F0 does not require neural data or adaptor weights. Training and generating new reconstructions require prepared neural data; reconstruction also requires a trained adaptor and the external Music2Latent decoder.

## Repository layout

```text
Cal2Sound/
├── README.md
├── LICENSE
├── requirements.txt
├── adaptation/
│   ├── main.py                         # Fold training and reconstruction runner
│   ├── Train.py                        # Dataset loading, losses and training
│   ├── Recons.py                       # Neural input → latent → waveform
│   ├── model_ablation.py                # Config-driven model factory
│   ├── model_tcn_dynamics_v2.py          # Released CalciumGRU / TCN adaptor
│   ├── model_vanilla.py                 # Required import of the model factory
│   ├── config.json                     # Saved configuration of the 100-fold run
│   ├── utils/
│   │   ├── loss_function.py
│   │   ├── music2latent_mel.py
│   │   └── split_ids.py
│   └── Res_Ana_F0/
│       ├── f0_backends.py               # Six F0 estimator adapters
│       ├── pitch_analysis_shared.py     # Shared metrics, manifests and caches
│       ├── run_estimators.py           # Batch F0 cache generation
│       ├── test_pitch_metric_contract.py
│       ├── english_note_annotations.json
│       ├── given_2audio_pitch_ana.ipynb  # Evaluation of an arbitrary audio pair
│       └── single_case_pitch_ana.ipynb  # Fold-level playback and visualization
├── demos/
│   ├── README.md
│   ├── original/                       # 700 reference WAV files
│   └── reconstructed/                  # 700 reconstructed WAV files
└── docs/                               # Project-page scaffold
```

The source files retain their original names and relative layout. Earlier scaffold directories (`cal2sound/`, `scripts/`, `evaluation/`, `analysis/`, and `configs/`) remain in the repository but are not the implemented workflows. Use the files under `adaptation/` described here.

## Installation

The source environment is Conda **`marmoset`, Python 3.11.11**. `requirements.txt` records the relevant installed package versions, rather than exporting unrelated packages from that environment. It includes training, notebook support, and all six F0 estimators.

To use the existing environment:

```bash
conda activate marmoset
```

To create a separate environment, run from the repository root:

```bash
conda create -n cal2sound python=3.11.11
conda activate cal2sound
python -m pip install -r requirements.txt
```

The source environment uses PyTorch/Torchaudio 2.9.1 with CUDA 12.8 and `onnxruntime-gpu` 1.26.0. The requirements file uses the CPU `onnxruntime` distribution for the ONNX estimators; CUDA support is not needed to evaluate demos. Use only one ONNX Runtime distribution in an environment. On an existing `marmoset` installation, retain its working GPU runtime instead of installing the CPU runtime alongside it.

Selected estimators and Music2Latent may download external model assets on first use. These assets are not included here. Package imports and the metric contract tests have been checked in `marmoset`; a fresh environment installation and a complete training/F0 run have not been validated.

## Audio demos

All pairs are stored in [demos/original](demos/original/) and [demos/reconstructed](demos/reconstructed/), without nested fold directories. Original filenames identify the matching `S`, `M`, and condition.

| Condition | Original filename | Reconstructed filename |
|---|---|---|
| Clean | `English_S{s}_M{m}_SM(0.0).wav` | `recon_English_S{s}_M{m}.wav` |
| SM: 3.0, 1.8, 0.6 | `English_S{s}_M{m}_SM({value}).wav` | `recons_English_S{s}_M{m}_SM({value}).wav` |
| TM: 3.5, 2.0, 1.0 | `English_S{s}_M{m}_TM({value}).wav` | `recons_English_S{s}_M{m}_TM({value}).wav` |

For example, listen to the S1/M1 clean [original](demos/original/English_S1_M1_SM%280.0%29.wav) and [reconstruction](demos/reconstructed/recon_English_S1_M1.wav). See [the demo guide](demos/README.md) for notebook playback.

## Model and training

The released model implementation is `adaptation/model_tcn_dynamics_v2.py`. The model factory maps both `tcn_dynamics_v2` and `ours` to this implementation. Other registered architectures have not been copied into this release; select the supplied `tcn_dynamics_v2` configuration.

`adaptation/config.json` is an unchanged copy of `sweep_config.json` from the released 100-fold experiment. It still contains source-machine paths. Before training, update:

- `data.data_path` and the neural/latent dataset filenames;
- `data.mel_dataset` if using the precomputed spectral targets;
- `save_load.save_folder` to a new local output directory;
- `loss.spectral.checkpoint_path` if using a local Music2Latent decoder asset.

Input neural arrays must follow the dataset ordering expected by `utils/split_ids.py`; an arbitrary reordered array is not compatible with the same-types-out split. The supplied configuration uses 4,667 neural features and target latents of shape `[trial, 64, 52]`. Adjusting the neural feature count alone does not establish a compatible dataset.

From the repository root, inspect the fold plan, then run training:

```bash
cd adaptation
python main.py --config config.json --all-combinations --dry-run
python main.py --config config.json --all-combinations --gpu 0
```

`--dry-run` only prints folds and output locations; it does not validate the data or run a model. The full runner trains each fold and calls `Recons.py` to generate its seven waveforms. It creates weights, fold configurations, split records and logs in the configured output directory.

`Recons.py` also exposes `recons_main(config)`. Its standalone `__main__` block still points to a historical result directory; use the training runner above, or call the function with a configuration pointing to your own trained fold.

## F0 evaluation

### Evaluate a released audio pair

Start JupyterLab from the repository root:

```bash
python -m jupyterlab adaptation/Res_Ana_F0/given_2audio_pitch_ana.ipynb
```

In the notebook's parameter cell, replace `AUDIO_PATHS` with two absolute paths in this order:

```python
AUDIO_PATHS = [
    '/absolute/path/to/Cal2Sound/demos/original/English_S1_M1_SM(0.0).wav',
    '/absolute/path/to/Cal2Sound/demos/reconstructed/recon_English_S1_M1.wav',
]
```

Run the cells to extract F0 and evaluate Original ↔ Reconstructed agreement. This notebook uses CREPE, PESTO and SwiftF0 with estimator-specific confidence thresholds, a 150–750 Hz F0 range, and ±200/±100/±50-cent tolerances. It uses zero time shift and timestamp mapping. These settings are retained from the source analysis; assess their suitability before applying the notebook to other audio.

The primary PDA metric uses only frames where **both** Original and Reconstructed pass the confidence threshold. It is conditional pitch agreement, not accuracy against human annotations. Confidence scales differ across estimators; the same numeric threshold does not imply equivalent coverage.

### Shared metrics and fold-based tools

`f0_backends.py` supports CREPE (`torchcrepe`), PESTO, pYIN, RMVPE, SwiftF0 and SWIPE. `pitch_analysis_shared.py` implements Human ↔ Original RPA, Human ↔ Reconstructed RPA, Original ↔ Reconstructed PDA, and corresponding chroma metrics. Human-reference evaluation uses `english_note_annotations.json`.

`run_estimators.py` generates batch F0 caches; it does not compute final accuracy tables. It expects a full experiment directory containing `S{s}_M{m}/split.json` and reconstructed audio. It cannot consume the flat `demos/reconstructed/` directory directly. With your own compatible result directory, inspect the extraction plan using:

```bash
python adaptation/Res_Ana_F0/run_estimators.py \
  --monkey ZJM \
  --result-root /absolute/path/to/results \
  --reference-audio-dir /absolute/path/to/Cal2Sound/demos/original \
  --estimators crepe pesto swiftf0 \
  --conditions clean \
  --dry-run
```

Remove `--dry-run` to extract F0. The batch tool covers clean and SM conditions; the release also provides TM audio for listening and pairwise evaluation.

`single_case_pitch_ana.ipynb` provides playback, spectrograms and pitch visualization for one fold. It retains the original experiment-path settings and requires that fold's configuration and audio. For the files included in this repository, begin with `given_2audio_pitch_ana.ipynb` and the playback example in the demo guide.

### Check metric definitions

From the repository root:

```bash
cd adaptation/Res_Ana_F0
python -m unittest -v test_pitch_metric_contract.py
```

These synthetic tests verify metric denominators, confidence filtering and frame/note aggregation without downloading estimator models or using real audio.

## License

See [LICENSE](LICENSE) for the repository's current license text. External pretrained models and dependencies retain their respective licenses. Paper citation details will be added when finalized.
