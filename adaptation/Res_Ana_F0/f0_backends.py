"""Unified F0 extractor adapters used by the frame- and note-level notebooks.

The public entry point is ``extract_f0(audio, sr, backend, config)``.  It
returns three one-dimensional NumPy arrays: F0 in Hz, confidence in [0, 1],
and frame timestamps in seconds.  Optional dependencies are imported only
when their backend is selected.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import warnings

import numpy as np


SUPPORTED_BACKENDS = ("crepe", "pesto", "pyin", "rmvpe", "swiftf0", "swipe")

CONFIDENCE_DEFINITIONS = {
    "crepe": "CREPE periodicity (confidence that a frame contains a periodic pitch)",
    "pesto": "PESTO confidence returned by pesto.predict",
    "pyin": "pYIN voiced_prob (voicing probability, not pitch uncertainty)",
    "rmvpe": "maximum RMVPE salience across pitch bins (not a calibrated probability)",
    "swiftf0": "SwiftF0 model confidence returned for each frame",
    "swipe": (
        "SWIPE' pitch strength: spectral match to the selected prime-harmonic "
        "pitch kernel (not a calibrated probability)"
    ),
}

_DISTRIBUTIONS = {
    "crepe": "torchcrepe",
    "pesto": "pesto-pitch",
    "pyin": "librosa",
    "rmvpe": "rmvpe-onnx",
    "swiftf0": "swift-f0",
    "swipe": "libf0",
}

_RMVPE_MODELS: dict[tuple, object] = {}
_SWIFTF0_MODELS: dict[tuple, object] = {}


def backend_package_version(backend: str) -> str:
    """Return the installed extractor package version for cache fingerprints."""
    backend = _normalise_backend(backend)
    distribution = _DISTRIBUTIONS[backend]
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def backend_runtime_metadata(backend: str, config: dict) -> dict:
    """Return JSON-serialisable backend metadata used by cache fingerprints."""
    backend = _normalise_backend(backend)
    if backend == "pesto":
        native_hop_seconds = float(config["step_size_ms"]) / 1000.0
    elif backend in {"crepe", "pyin", "swipe"}:
        native_hop_seconds = (
            float(config["hop_length"]) / float(config["sample_rate"])
        )
    elif backend == "rmvpe":
        native_hop_seconds = 160.0 / 16_000.0
    else:
        native_hop_seconds = 256.0 / 16_000.0
    return {
        "package": _DISTRIBUTIONS[backend],
        "package_version": backend_package_version(backend),
        "model_version": str(config.get("model_version", "package-default")),
        "native_hop_seconds": native_hop_seconds,
        "confidence_definition": CONFIDENCE_DEFINITIONS[backend],
    }


def extract_f0(
    audio: np.ndarray,
    sr: int,
    backend: str,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an F0 track through the selected backend.

    Parameters
    ----------
    audio:
        Mono floating-point waveform.
    sr:
        Waveform sample rate.
    backend:
        One of ``crepe``, ``pesto``, ``pyin``, ``rmvpe``, ``swiftf0`` or
        ``swipe``.
    config:
        Common parameters (sample_rate, hop_length, fmin, fmax) plus the
        selected backend's options.
    """
    backend = _normalise_backend(backend)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("audio must be a non-empty finite mono waveform")
    if int(sr) <= 0:
        raise ValueError("sr must be positive")

    extractor = {
        "crepe": extract_f0_crepe,
        "pesto": extract_f0_pesto,
        "pyin": extract_f0_pyin,
        "rmvpe": extract_f0_rmvpe,
        "swiftf0": extract_f0_swiftf0,
        "swipe": extract_f0_swipe,
    }[backend]
    f0_hz, confidence, time_sec = extractor(audio, int(sr), config)
    return _validate_track(f0_hz, confidence, time_sec, backend)


def extract_f0_crepe(audio: np.ndarray, sr: int, config: dict):
    """CREPE path kept equivalent to the original notebook implementation."""
    import torch
    import torchcrepe

    device = _torch_device(config.get("device", "cuda:0"))
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            wav,
            sr,
            int(config["hop_length"]),
            float(config["fmin"]),
            float(config["fmax"]),
            config.get("model", "full"),
            batch_size=int(config.get("batch_size", 8)),
            device=device,
            return_periodicity=True,
        )

    f0_hz = pitch.squeeze().detach().cpu().numpy().astype(np.float64)
    confidence = periodicity.squeeze().detach().cpu().numpy().astype(np.float64)
    time_sec = (
        np.arange(np.asarray(f0_hz).size, dtype=np.float64)
        * int(config["hop_length"])
        / sr
    )
    return f0_hz, confidence, time_sec


def extract_f0_pesto(audio: np.ndarray, sr: int, config: dict):
    """PESTO path kept equivalent to the original notebook implementation."""
    import torch
    import pesto

    device = _torch_device(config.get("device", "cuda:0"))
    wav = torch.as_tensor(audio, dtype=torch.float32, device=device).reshape(-1)
    with torch.inference_mode():
        timesteps_ms, pitch, confidence, _ = pesto.predict(
            wav,
            sr,
            step_size=float(config["step_size_ms"]),
            model_name=config.get("model_name", "mir-1k_g7"),
            reduction=config.get("reduction", "alwa"),
            num_chunks=int(config.get("num_chunks", 1)),
            convert_to_freq=True,
        )

    f0_hz = _tensor_to_numpy(pitch)
    confidence = _tensor_to_numpy(confidence)
    time_sec = _tensor_to_numpy(timesteps_ms) / 1000.0
    common_length = min(len(f0_hz), len(confidence), len(time_sec))
    f0_hz = f0_hz[:common_length].copy()
    confidence = confidence[:common_length]
    time_sec = time_sec[:common_length]
    return _mask_f0_range(f0_hz, config), confidence, time_sec


def extract_f0_pyin(audio: np.ndarray, sr: int, config: dict):
    """Use pYIN voiced probability as confidence.

    ``voiced_prob`` estimates whether a frame is voiced.  It is not a
    calibrated measure of uncertainty in the selected pitch value.
    """
    import librosa

    f0_hz, _, voiced_prob = librosa.pyin(
        audio,
        fmin=float(config["fmin"]),
        fmax=float(config["fmax"]),
        sr=sr,
        frame_length=int(config.get("frame_length", 2048)),
        hop_length=int(config["hop_length"]),
        fill_na=np.nan,
        center=bool(config.get("center", True)),
    )
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    confidence = np.asarray(voiced_prob, dtype=np.float64)
    time_sec = (
        np.arange(len(f0_hz), dtype=np.float64)
        * int(config["hop_length"])
        / sr
    )
    return f0_hz, confidence, time_sec


def extract_f0_rmvpe(audio: np.ndarray, sr: int, config: dict):
    """Extract RMVPE F0 and use maximum pitch-bin salience as confidence.

    RMVPE salience is a model activation, not a calibrated probability.  The
    community ``rmvpe-onnx`` wrapper follows the Dream-High/RVC decoder and
    exposes the full 360-bin activation needed for this definition.
    """
    model = _get_rmvpe_model(config)
    time_sec, f0_hz, _, activation = model.predict(audio=audio, sr=sr)
    activation = np.asarray(activation, dtype=np.float64)
    if activation.ndim != 2:
        raise ValueError(
            f"RMVPE activation must have shape (frames, bins), got {activation.shape}"
        )
    confidence = np.max(activation, axis=1)
    f0_hz = _mask_f0_range(np.asarray(f0_hz, dtype=np.float64), config)
    return f0_hz, confidence, np.asarray(time_sec, dtype=np.float64)


def extract_f0_swiftf0(audio: np.ndarray, sr: int, config: dict):
    """Extract SwiftF0 pitch and use its model confidence output directly."""
    detector = _get_swiftf0_model(config)
    result = detector.detect_from_array(audio, sr)
    f0_hz = _mask_f0_range(
        np.asarray(result.pitch_hz, dtype=np.float64),
        config,
    )
    return (
        f0_hz,
        np.asarray(result.confidence, dtype=np.float64),
        np.asarray(result.timestamps, dtype=np.float64),
    )


def extract_f0_swipe(audio: np.ndarray, sr: int, config: dict):
    """Extract SWIPE' F0 and use its pitch strength as confidence.

    ``libf0.swipe`` follows the multi-resolution SWIPE' implementation and
    returns the strength of the selected prime-harmonic pitch kernel.  The
    extraction threshold stays at zero so that notebook-level confidence
    thresholds can be applied later without recomputing the F0 trajectory.
    Pitch strength is an estimator-native spectral match, not a calibrated
    voicing or correctness probability.
    """
    try:
        import libf0
    except ImportError as exc:
        raise ImportError(
            "SWIPE' backend requires `pip install libf0==1.0.2`."
        ) from exc

    implementation = str(config.get("implementation", "full")).lower()
    common = {
        "x": np.asarray(audio, dtype=np.float64),
        "Fs": int(sr),
        "H": int(config["hop_length"]),
        "F_min": float(config["fmin"]),
        "F_max": float(config["fmax"]),
        "strength_threshold": float(config.get("strength_threshold", 0.0)),
    }
    if implementation == "full":
        f0_hz, time_sec, pitch_strength = libf0.swipe(
            **common,
            dlog2p=float(config.get("dlog2p", 1.0 / 96.0)),
            derbs=float(config.get("derbs", 0.1)),
        )
    elif implementation == "slim":
        f0_hz, time_sec, pitch_strength = libf0.swipe_slim(
            **common,
            R=float(config.get("resolution_cents", 10.0)),
        )
    else:
        raise ValueError("SWIPE' implementation must be 'full' or 'slim'")

    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    pitch_strength = np.asarray(pitch_strength, dtype=np.float64)
    invalid_strength = ~np.isfinite(pitch_strength)
    # libf0 can emit NaN at zero-padded boundaries and slightly negative
    # similarities for unpitched frames.  Both represent zero confidence.
    f0_hz[invalid_strength | (pitch_strength < 0.0)] = np.nan
    confidence = np.clip(
        np.nan_to_num(pitch_strength, nan=0.0, neginf=0.0, posinf=1.0),
        0.0,
        1.0,
    )
    return (
        _mask_f0_range(f0_hz, config),
        confidence,
        np.asarray(time_sec, dtype=np.float64),
    )


def clear_model_cache() -> None:
    """Clear lazy model singletons; mainly useful for focused tests."""
    _RMVPE_MODELS.clear()
    _SWIFTF0_MODELS.clear()


def _get_rmvpe_model(config: dict):
    try:
        from rmvpe_onnx import RMVPE
    except ImportError as exc:
        raise ImportError(
            "RMVPE backend requires `pip install rmvpe-onnx`. "
            "Install onnxruntime-gpu as well for CUDA inference."
        ) from exc

    device = _onnx_device(config.get("device", "cuda:0"))
    model_path = config.get("model_path")
    cache_key = (str(model_path), device)
    if cache_key not in _RMVPE_MODELS:
        _RMVPE_MODELS[cache_key] = RMVPE(
            model_path=None if model_path in (None, "") else str(model_path),
            device=device,
        )
    return _RMVPE_MODELS[cache_key]


def _get_swiftf0_model(config: dict):
    try:
        import onnxruntime
        import swift_f0
        from swift_f0 import SwiftF0
    except ImportError as exc:
        raise ImportError(
            "SwiftF0 backend requires `pip install swift-f0`. "
            "Install onnxruntime-gpu as well for CUDA inference."
        ) from exc

    device = _onnx_device(config.get("device", "cuda:0"))
    model_path_value = config.get("model_path")
    model_path = (
        Path(model_path_value).expanduser().resolve()
        if model_path_value
        else Path(swift_f0.__file__).resolve().parent / "model.onnx"
    )
    cache_key = (str(model_path), device, float(config["fmin"]), float(config["fmax"]))
    if cache_key in _SWIFTF0_MODELS:
        return _SWIFTF0_MODELS[cache_key]

    # The official package creates a CPU session in __init__.  Keep all its
    # preprocessing/timestamp logic, then replace only the ONNX session when
    # CUDA (or a custom model path) is requested.
    detector = SwiftF0(
        confidence_threshold=0.0,
        fmin=float(config["fmin"]),
        fmax=float(config["fmax"]),
    )
    if device != "cpu" or model_path_value:
        provider = (
            ("CUDAExecutionProvider", {"device_id": int(device.split(":")[-1])})
            if device.startswith("cuda")
            else "CPUExecutionProvider"
        )
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        detector.pitch_session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[provider],
        )
        detector.pitch_input_name = detector.pitch_session.get_inputs()[0].name

    _SWIFTF0_MODELS[cache_key] = detector
    return detector


def _normalise_backend(backend: str) -> str:
    value = str(backend).strip().lower().replace("-", "")
    aliases = {
        "librosapyin": "pyin",
        "swiftf0": "swiftf0",
        "swipep": "swipe",
        "swipeprime": "swipe",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported F0 backend {backend!r}; choose from {SUPPORTED_BACKENDS}"
        )
    return value


def _torch_device(requested: str) -> str:
    import torch

    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"CUDA is unavailable; falling back from {requested} to CPU")
        return "cpu"
    return requested


def _onnx_device(requested: str) -> str:
    requested = str(requested).lower()
    if not requested.startswith("cuda"):
        return "cpu"
    try:
        import onnxruntime

        # Load the CUDA/cuDNN libraries bundled with the active PyTorch
        # environment before constructing an ONNX CUDA session.
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            return requested
    except ImportError:
        pass
    warnings.warn(f"ONNX CUDA provider is unavailable; falling back from {requested} to CPU")
    return "cpu"


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ravel(np.asarray(value, dtype=np.float64))


def _mask_f0_range(f0_hz: np.ndarray, config: dict) -> np.ndarray:
    f0_hz = np.asarray(f0_hz, dtype=np.float64).copy()
    invalid = (
        ~np.isfinite(f0_hz)
        | (f0_hz < float(config["fmin"]))
        | (f0_hz > float(config["fmax"]))
    )
    f0_hz[invalid] = np.nan
    return f0_hz


def _validate_track(f0_hz, confidence, time_sec, backend):
    f0_hz = np.ravel(np.asarray(f0_hz, dtype=np.float64))
    confidence = np.ravel(np.asarray(confidence, dtype=np.float64))
    time_sec = np.ravel(np.asarray(time_sec, dtype=np.float64))
    if not (len(f0_hz) == len(confidence) == len(time_sec)):
        raise ValueError(
            f"{backend} returned mismatched lengths: "
            f"f0={len(f0_hz)}, confidence={len(confidence)}, time={len(time_sec)}"
        )
    if len(f0_hz) == 0:
        raise ValueError(f"{backend} returned an empty F0 track")
    if not np.isfinite(confidence).all():
        raise ValueError(f"{backend} confidence contains NaN/Inf")
    if np.any((confidence < -1e-7) | (confidence > 1.0 + 1e-7)):
        raise ValueError(f"{backend} confidence is outside [0, 1]")
    if not np.isfinite(time_sec).all() or np.any(np.diff(time_sec) <= 0):
        raise ValueError(f"{backend} timestamps must be finite and strictly increasing")
    invalid_f0 = np.isfinite(f0_hz) & (f0_hz <= 0)
    if invalid_f0.any():
        raise ValueError(f"{backend} returned non-positive finite F0 values")
    return f0_hz, confidence, time_sec
