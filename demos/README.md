# Audio demos

This directory contains 700 paired original and reconstructed WAV files from the ZJM 100-fold experiment (`S1–S10 × M1–M10`). Each fold provides clean audio, SM conditions 3.0/1.8/0.6, and TM conditions 3.5/2.0/1.0.

- `original/`: 700 reference audio files.
- `reconstructed/`: 700 decoded audio files.

Match files by `S`, `M` and condition. Clean originals use `English_S{s}_M{m}_SM(0.0).wav`; clean reconstructions use `recon_English_S{s}_M{m}.wav`. Other reconstructions use the `recons_` prefix followed by the original filename.

## Listen in a notebook

With the notebook working directory set to the repository root:

```python
from IPython.display import Audio, display

display(Audio(filename='demos/original/English_S1_M1_SM(0.0).wav'))
display(Audio(filename='demos/reconstructed/recon_English_S1_M1.wav'))
```

For pairwise F0 evaluation, open `adaptation/Res_Ana_F0/given_2audio_pitch_ana.ipynb` and set its `AUDIO_PATHS` to the absolute paths of an original/reconstructed pair. See the [main README](../README.md#f0-evaluation) for metric settings and the distinction between pairwise and fold-based tools.

No neural input data or adaptor weights are needed for playback or pairwise F0 evaluation. `figures/` is an unused placeholder; generated F0 plots are not included.
