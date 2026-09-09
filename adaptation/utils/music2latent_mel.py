"""Differentiable Music2Latent latent-to-Mel decoding for training."""

from contextlib import nullcontext

import librosa
import numpy as np
import torch
import torch.nn as nn

from music2latent import EncoderDecoder
from music2latent.hparams import (
    alpha_rescale,
    beta_rescale,
    data_channels,
    freq_downsample_list,
    hop,
    sigma_max,
)


class FrozenMusic2LatentMelDecoder(nn.Module):
    """Decode ``[B, 64, T]`` latents into compressed magnitude Mel spectra.

    Music2Latent's public ``decode`` path is inference-only and detaches its
    result. This module performs the equivalent one-step consistency-model
    forward pass without detaching, so gradients can flow back to the input
    latent while all decoder parameters remain frozen.
    """

    def __init__(
        self,
        device: torch.device,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 24000.0,
        latent_scale: float = 0.06,
        denoising_steps: int = 1,
        noise_seed: int = 3407,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        checkpoint_path=None,
        encoder_decoder=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.latent_scale = float(latent_scale)
        self.denoising_steps = int(denoising_steps)
        self.noise_seed = int(noise_seed)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"

        if self.n_fft != 4 * hop:
            raise ValueError(
                f"Expected n_fft == 4 * Music2Latent hop ({4 * hop}), "
                f"but got {self.n_fft}"
            )
        if self.denoising_steps != 1:
            raise ValueError(
                "The differentiable training path currently supports exactly "
                f"one denoising step, got {self.denoising_steps}"
            )
        if amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                f"amp_dtype must be 'float16' or 'bfloat16', got {amp_dtype!r}"
            )
        self.amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[amp_dtype]

        # Sweep 模式由 main.py 注入共享 EncoderDecoder；单 fold 独立运行时
        # 仍保留内部创建的兼容路径。
        if encoder_decoder is None:
            encoder_decoder = EncoderDecoder(
                load_path_inference=checkpoint_path,
                device=self.device,
            )
        else:
            encoder_decoder.device = self.device
            encoder_decoder.gen.to(self.device)
        self.decoder = encoder_decoder.gen
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        frequency_bins = hop * 2
        mel_filterbank = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            htk=False,
            norm="slaney",
        )[:, :frequency_bins].astype(np.float32)
        self.register_buffer(
            "mel_filterbank",
            torch.from_numpy(mel_filterbank).to(self.device),
            persistent=False,
        )

        self.frequency_bins = frequency_bins
        self.time_downscaling_factor = 2 ** freq_downsample_list.count(0)

    def train(self, mode: bool = True):
        """Keep the frozen decoder in evaluation mode."""
        super().train(False)
        self.decoder.eval()
        return self

    def _make_initial_noise(
        self,
        sample_ids: torch.Tensor,
        time_frames: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create deterministic, sample-specific noise directly on the GPU."""
        noises = []
        for sample_id in sample_ids.detach().cpu().tolist():
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.noise_seed + int(sample_id))
            noises.append(
                torch.randn(
                    (data_channels, self.frequency_bins, time_frames),
                    generator=generator,
                    device=self.device,
                    dtype=dtype,
                )
            )
        return torch.stack(noises, dim=0) * sigma_max

    def forward(
        self,
        latents: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return compressed magnitude Mel spectra with shape ``[B, 128, 416]``."""
        if latents.ndim != 3:
            raise ValueError(
                f"Expected latents with shape [B, C, T], got {tuple(latents.shape)}"
            )
        if sample_ids.ndim != 1 or sample_ids.numel() != latents.size(0):
            raise ValueError(
                "sample_ids must be a 1D tensor with one ID per latent: "
                f"latents={tuple(latents.shape)}, sample_ids={tuple(sample_ids.shape)}"
            )
        if latents.device != self.device:
            raise ValueError(
                f"Latents are on {latents.device}, but decoder is on {self.device}"
            )

        decoded_time_frames = latents.size(-1) * self.time_downscaling_factor
        initial_noise = self._make_initial_noise(
            sample_ids=sample_ids,
            time_frames=decoded_time_frames,
            dtype=latents.dtype,
        )

        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=True,
            )
            if self.use_amp
            else nullcontext()
        )
        with amp_context:
            # With one denoising step Music2Latent evaluates the consistency
            # UNet at sigma_max and returns this prediction directly.
            representation = self.decoder(
                latents * self.latent_scale,
                initial_noise,
                sigma_max,
            )

        # Undo the compressed linear magnitude before Mel projection: Mel and
        # the nonlinear 0.34 * magnitude**0.65 compression do not commute.
        compressed_linear_magnitude = torch.linalg.vector_norm(
            representation.float(), dim=1
        )
        linear_magnitude = (
            compressed_linear_magnitude.div(beta_rescale)
            .clamp_min(0.0)
            .pow(1.0 / alpha_rescale)
        )
        mel_magnitude = torch.matmul(
            self.mel_filterbank.unsqueeze(0),
            linear_magnitude,
        )
        return mel_magnitude.clamp_min(0.0).pow(alpha_rescale) * beta_rescale
