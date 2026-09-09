"""Shared data and metric helpers for the factor-oriented pitch notebooks.

The notebooks own their scientific question and visual narrative.  This module
only centralises data discovery, F0 cache handling, frame/note metric
definitions, and small reconciliation helpers so that the five analyses use
the same denominators.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


SUPPORTED_LEVELS = ("frame", "note")
DEFAULT_THRESHOLDS = (0.1, 0.3, 0.5)
ESTIMATOR_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
SM_CONDITIONS = ("clean", "SM(3.0)", "SM(1.8)", "SM(0.6)")
PITCH_ACCURACY_METRICS = ("rpa_orig", "rpa_recon", "pda")
DEFAULT_TOLERANCE_HALF_WIDTH_CENTS = 50.0

# Project H/O/R metric contract.  RPA keeps the Human-voiced denominator.
# At frame level, PDA is conditional on both Original and Reconstructed passing
# the same estimator-native confidence threshold.  At note level, confidence is
# not used: all valid F0 frames in each note are aggregated independently.
PITCH_METRIC_SPECS = {
    "rpa_orig": {
        "denominator": "n_human_voiced",
        "pitch_correct": "n_rpa_orig_correct",
        "chroma_correct": "n_rca_orig_correct",
        "chroma_metric": "rca_orig",
        "candidate_recall": "original_voiced_recall",
        "candidate_pair": "n_human_original_voiced",
        "label": "RPA_orig (Human ↔ Original)",
        "chroma_label": "RCA_orig (Human ↔ Original)",
    },
    "rpa_recon": {
        "denominator": "n_human_voiced",
        "pitch_correct": "n_rpa_recon_correct",
        "chroma_correct": "n_rca_recon_correct",
        "chroma_metric": "rca_recon",
        "candidate_recall": "reconstructed_voiced_recall",
        "candidate_pair": "n_human_reconstructed_voiced",
        "label": "RPA_recon (Human ↔ Reconstructed)",
        "chroma_label": "RCA_recon (Human ↔ Reconstructed)",
    },
    "pda": {
        "denominator": "n_pda_evaluated",
        "pitch_correct": "n_pda_correct",
        "chroma_correct": "n_rca_pda_correct",
        "chroma_metric": "rca_pda",
        "candidate_recall": "pda_candidate_recall",
        "candidate_pair": "n_original_reconstructed_voiced",
        "label": "PDA (Original ↔ Reconstructed)",
        "chroma_label": "RCA_PDA (Original ↔ Reconstructed)",
    },
}


@dataclass
class F0Track:
    f0_hz: np.ndarray
    confidence: np.ndarray
    time_sec: np.ndarray
    backend: str


def find_analysis_dir() -> Path:
    """Find Res_Ana_F0 from the project root, adaptation, or the folder itself."""
    candidates = [
        Path.cwd(),
        Path.cwd() / "Res_Ana_F0",
        Path.cwd() / "adaptation" / "Res_Ana_F0",
    ]
    for candidate in candidates:
        if (candidate / "english_note_annotations.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find adaptation/Res_Ana_F0")


def ensure_level(level: str) -> str:
    level = str(level).strip().lower()
    if level not in SUPPORTED_LEVELS:
        raise ValueError(f"EVAL_LEVEL must be one of {SUPPORTED_LEVELS}")
    return level


def backend_configs(sr: int = 48_000, hop_length: int | None = None) -> dict:
    """Return the canonical configuration for all supported F0 estimators."""
    hop_length = int(sr / 200) if hop_length is None else int(hop_length)
    common = {
        "sample_rate": int(sr),
        "hop_length": hop_length,
        "fmin": 150.0,
        "fmax": 750.0,
    }
    return {
        "crepe": {**common, "model": "full", "batch_size": 8, "device": "cuda:0"},
        "pesto": {
            **common,
            "model_name": "mir-1k_g7",
            "reduction": "alwa",
            "num_chunks": 1,
            "device": "cuda:0",
            "step_size_ms": 1000.0 * hop_length / sr,
        },
        "pyin": {
            **common,
            "frame_length": 2048,
            "center": True,
            "model_version": "librosa-pyin",
        },
        "rmvpe": {
            **common,
            "model_path": None,
            "model_version": "rmvpe-onnx-0.2.3",
            "device": "cuda:0",
        },
        "swiftf0": {
            **common,
            "model_path": None,
            "model_version": "swift-f0-package-model",
            "device": "cuda:0",
        },
        "swipe": {
            **common,
            # Full multi-resolution libf0 implementation of SWIPE'.
            "implementation": "full",
            "dlog2p": 1.0 / 96.0,
            "derbs": 0.1,
            # Preserve native strength for later notebook-level filtering.
            "strength_threshold": 0.0,
            "model_version": "libf0-swipe-full",
        },
    }


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def hz_to_midi(f0_hz):
    values = np.asarray(f0_hz, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = 69.0 + 12.0 * np.log2(values[valid] / 440.0)
    return result


def pitch_error_cents(ref_hz, pred_hz):
    return 1200.0 * np.log2(
        np.asarray(pred_hz, dtype=float) / np.asarray(ref_hz, dtype=float)
    )


def chroma_error_cents_from_signed(signed_error):
    error = np.asarray(signed_error, dtype=float)
    return np.abs(np.mod(error + 600.0, 1200.0) - 600.0)


def valid_f0(values):
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values > 0)


def aggregate_f0(values, method: str):
    values = np.asarray(values, dtype=float)
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    raise ValueError("NOTE_F0_AGGREGATION must be 'mean' or 'median'")


def dataset_id_to_audio_condition(dataset_id: int) -> int:
    dataset_id = int(dataset_id)
    if 0 <= dataset_id < 100:
        return dataset_id
    if 100 <= dataset_id < 400:
        return (dataset_id - 100) // 3
    if 400 <= dataset_id < 700:
        return (dataset_id - 400) // 3
    raise ValueError(f"Dataset id outside [0, 699]: {dataset_id}")


def normalise_condition(condition: str) -> str:
    value = str(condition).strip()
    if value.lower() == "clean":
        return "clean"
    if re.fullmatch(r"SM\((?:0\.6|1\.8|3\.0)\)", value):
        return value
    raise ValueError(f"Unsupported condition: {condition!r}")


def condition_slug(condition: str) -> str:
    value = normalise_condition(condition)
    return "clean" if value == "clean" else value.lower().replace("(", "_").replace(")", "").replace(".", "p")


def infer_monkey_id(result_root: Path) -> str:
    """Infer the monkey identifier from a canonical results directory name."""
    name = Path(result_root).resolve().name
    match = re.match(r"^results_\d{8}_([^_]+)_", name)
    if match is None:
        raise ValueError(
            "Cannot infer monkey id from result directory name: "
            f"{name!r}. Expected results_YYYYMMDD_<MONKEY>_..."
        )
    return match.group(1).upper()


def fold_scope_token(pair_keys) -> str:
    """Return the stable fold-count/hash token used in canonical cache names."""
    keys = [list(map(int, key)) for key in pair_keys]
    key_token = hashlib.sha1(json.dumps(keys).encode()).hexdigest()[:8]
    return f"{len(keys)}folds-{key_token}"


def canonical_f0_cache_path(
    result_root: Path,
    backend: str,
    condition: str,
    pair_keys,
    monkey_id: str | None = None,
) -> Path:
    """Build a readable cache path for one Original/Reconstructed condition pair."""
    condition_token = condition_slug(condition).replace("_", "-")
    subject = infer_monkey_id(result_root) if monkey_id is None else str(monkey_id).upper()
    scope = fold_scope_token(pair_keys)
    filename = (
        f"f0_tracks__monkey-{subject}__condition-{condition_token}"
        f"__roles-original-reconstructed__estimator-{str(backend).lower()}"
        f"__scope-{scope}.npz"
    )
    return Path(result_root) / "f0_estimator_cache" / filename


def canonical_f0_metadata_path(cache_path: Path) -> Path:
    """Return the JSON sidecar path paired with a canonical NPZ cache."""
    return Path(cache_path).with_suffix(".json")


def _full_scope_canonical_cache_path(
    result_root: Path,
    backend: str,
    condition: str,
) -> Path | None:
    """Find the unique 100-fold cache that can safely serve a fold subset."""
    monkey_id = infer_monkey_id(result_root)
    condition_token = condition_slug(condition).replace("_", "-")
    pattern = (
        f"f0_tracks__monkey-{monkey_id}__condition-{condition_token}"
        f"__roles-original-reconstructed__estimator-{str(backend).lower()}"
        "__scope-100folds-*.npz"
    )
    candidates = sorted((Path(result_root) / "f0_estimator_cache").glob(pattern))
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple canonical 100-fold caches match the same estimator and "
            f"condition: {candidates}"
        )
    return candidates[0] if candidates else None


def build_manifest(
    analysis_dir: Path,
    result_root: Path,
    reference_audio_dir: Path,
    condition: str = "clean",
    max_folds: int | None = None,
):
    """Build pair records and fold-specific pitch exposure for one condition."""
    condition = normalise_condition(condition)
    annotation_path = Path(analysis_dir) / "english_note_annotations.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        recordings = json.load(file)["recordings"]

    base_rows = []
    for record in recordings:
        s_id, m_id = int(record["S"]), int(record["M"])
        for note in record["notes"]:
            midi = int(note["label_midi"])
            base_rows.append({
                "note_uid": f"S{s_id:02d}_M{m_id:02d}_N{int(note['note_index']):02d}",
                "S": s_id,
                "M": m_id,
                "note_index": int(note["note_index"]),
                "label_midi": midi,
                "label_hz": float(midi_to_hz(midi)),
                "start_time_s": max(0.0, float(note["start_time_s"])),
                "end_time_s": min(4.2, float(note["end_time_s"])),
            })
    note_table = pd.DataFrame(base_rows).sort_values(
        ["S", "M", "note_index"]
    ).reset_index(drop=True)
    audio_condition = (note_table["S"] - 1) * 10 + (note_table["M"] - 1)

    exposure_rows = []
    for s_id in range(1, 11):
        for m_id in range(1, 11):
            split_path = Path(result_root) / f"S{s_id}_M{m_id}" / "split.json"
            with split_path.open("r", encoding="utf-8") as file:
                train_ids = [int(value) for value in json.load(file)["train_ids"]]
            train_conditions = {
                dataset_id_to_audio_condition(value) for value in train_ids
            }
            train_notes = note_table.loc[audio_condition.isin(train_conditions)]
            counts = train_notes["label_midi"].value_counts()
            test_notes = note_table.loc[
                note_table["S"].eq(s_id) & note_table["M"].eq(m_id)
            ]
            for note in test_notes.itertuples(index=False):
                count = int(counts.get(note.label_midi, 0))
                exposure_rows.append({
                    "note_uid": note.note_uid,
                    "n_train_notes_same_pitch": count,
                    "train_pitch_probability": count / len(train_notes),
                    "seen_group": "unseen" if count == 0 else "seen",
                })
    note_table = note_table.merge(
        pd.DataFrame(exposure_rows), on="note_uid", validate="one_to_one"
    )
    note_lookup = note_table.set_index("note_uid").to_dict("index")

    pairs = []
    recordings = sorted(recordings, key=lambda r: (int(r["S"]), int(r["M"])))
    if max_folds is not None:
        recordings = recordings[: int(max_folds)]
    for record in recordings:
        s_id, m_id = int(record["S"]), int(record["M"])
        if condition == "clean":
            ref_name = f"English_S{s_id}_M{m_id}_SM(0.0).wav"
            pred_name = f"recon_English_S{s_id}_M{m_id}.wav"
        else:
            ref_name = f"English_S{s_id}_M{m_id}_{condition}.wav"
            pred_name = f"recons_English_S{s_id}_M{m_id}_{condition}.wav"
        notes = []
        for note in record["notes"]:
            note = dict(note)
            uid = f"S{s_id:02d}_M{m_id:02d}_N{int(note['note_index']):02d}"
            note["note_uid"] = uid
            note.update(note_lookup[uid])
            notes.append(note)
        pair = {
            "key": (s_id, m_id),
            "S": s_id,
            "M": m_id,
            "condition": condition,
            "notes": notes,
            "ref_path": Path(reference_audio_dir) / ref_name,
            "pred_path": Path(result_root) / f"S{s_id}_M{m_id}" / pred_name,
        }
        missing = [str(path) for path in [pair["ref_path"], pair["pred_path"]] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing audio files: {missing}")
        pairs.append(pair)
    return pairs, note_table


def pair_clean_reference_with_reconstruction(
    clean_pairs,
    result_root: Path,
    condition: str,
):
    """Pair the clean Original with a clean/SM ZJM reconstruction.

    The returned records retain the clean ``ref_path`` and note annotations.
    For SM conditions only the reconstruction path changes; an SM reference
    audio file is neither required nor inspected.
    """
    condition = normalise_condition(condition)
    result_root = Path(result_root)
    pairs = []
    for clean_pair in clean_pairs:
        pair = dict(clean_pair)
        pair["condition"] = condition
        if condition != "clean":
            s_id, m_id = int(pair["S"]), int(pair["M"])
            pair["pred_path"] = (
                result_root / f"S{s_id}_M{m_id}"
                / f"recons_English_S{s_id}_M{m_id}_{condition}.wav"
            )
        if not Path(pair["ref_path"]).is_file():
            raise FileNotFoundError(f"Missing clean reference audio: {pair['ref_path']}")
        if not Path(pair["pred_path"]).is_file():
            raise FileNotFoundError(f"Missing reconstructed audio: {pair['pred_path']}")
        pairs.append(pair)
    return pairs


def load_audio_mono(path: Path, target_sr: int = 48_000):
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != target_sr:
        divisor = np.gcd(file_sr, target_sr)
        audio = resample_poly(
            audio, target_sr // divisor, file_sr // divisor
        ).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    return np.asarray(audio, dtype=np.float32)


def _load_npz_tracks(path: Path, backend: str, wanted_keys=None):
    tracks = {}
    wanted = None if wanted_keys is None else set(wanted_keys)
    with np.load(path, allow_pickle=False) as cache:
        for index, pair_key in enumerate(cache["pair_keys"]):
            key = tuple(map(int, pair_key))
            if wanted is not None and key not in wanted:
                continue
            ref_n = int(cache["ref_lengths"][index])
            pred_n = int(cache["pred_lengths"][index])
            tracks[key] = {
                "reference": F0Track(
                    cache["ref_f0_hz"][index, :ref_n].copy(),
                    cache["ref_confidence"][index, :ref_n].copy(),
                    cache["ref_time_sec"][index, :ref_n].copy(), backend,
                ),
                "reconstructed": F0Track(
                    cache["pred_f0_hz"][index, :pred_n].copy(),
                    cache["pred_confidence"][index, :pred_n].copy(),
                    cache["pred_time_sec"][index, :pred_n].copy(), backend,
                ),
            }
    return tracks


def _pad_tracks(track_list, field, fill):
    lengths = np.asarray([len(getattr(track, field)) for track in track_list], dtype=np.int32)
    result = np.full((len(track_list), int(lengths.max())), fill, dtype=float)
    for index, track in enumerate(track_list):
        values = np.asarray(getattr(track, field), dtype=float)
        result[index, : len(values)] = values
    return result, lengths


def _save_npz_tracks(path: Path, tracks: dict, pair_keys):
    """Atomically save paired Original/Reconstructed F0 tracks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = [tracks[key]["reference"] for key in pair_keys]
    pred = [tracks[key]["reconstructed"] for key in pair_keys]
    ref_f0, ref_lengths = _pad_tracks(ref, "f0_hz", np.nan)
    ref_conf, _ = _pad_tracks(ref, "confidence", 0.0)
    ref_time, _ = _pad_tracks(ref, "time_sec", np.nan)
    pred_f0, pred_lengths = _pad_tracks(pred, "f0_hz", np.nan)
    pred_conf, _ = _pad_tracks(pred, "confidence", 0.0)
    pred_time, _ = _pad_tracks(pred, "time_sec", np.nan)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                pair_keys=np.asarray(pair_keys, dtype=np.int16),
                ref_f0_hz=ref_f0, ref_confidence=ref_conf,
                ref_time_sec=ref_time, ref_lengths=ref_lengths,
                pred_f0_hz=pred_f0, pred_confidence=pred_conf,
                pred_time_sec=pred_time, pred_lengths=pred_lengths,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_f0_cache_metadata(
    cache_path: Path,
    pairs,
    result_root: Path,
    backend: str,
    config: dict,
    condition: str,
    cache_status: str,
    migrated_from: Path | None = None,
):
    """Write an atomic, human-readable provenance sidecar for an F0 cache."""
    from f0_backends import backend_runtime_metadata

    monkey_id = infer_monkey_id(result_root)
    runtime = backend_runtime_metadata(backend, config)
    fingerprint_input = {
        "backend": backend,
        "backend_config": config,
        "backend_runtime": runtime,
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_input, sort_keys=True, separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    metadata = {
        "schema_version": "2.0",
        "cache_file": cache_path.name,
        "cache_path": str(cache_path.resolve()),
        "monkey_id": monkey_id,
        "result_root": str(Path(result_root).resolve()),
        "condition": condition,
        "track_roles": {
            "reference": f"{condition} Original audio",
            "reconstructed": f"{condition} Reconstructed audio",
        },
        "estimator": backend,
        "backend_config": config,
        "backend_runtime": runtime,
        "config_fingerprint": config_fingerprint,
        "scope": {
            "n_folds": len(pairs),
            "fold_token": fold_scope_token([pair["key"] for pair in pairs]),
            "fold_keys": [list(map(int, pair["key"])) for pair in pairs],
        },
        "audio_pairs": [
            {
                "S": int(pair["S"]), "M": int(pair["M"]),
                "original": str(Path(pair["ref_path"]).resolve()),
                "reconstructed": str(Path(pair["pred_path"]).resolve()),
            }
            for pair in pairs
        ],
        "cache_status_when_written": cache_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "adaptation/Res_Ana_F0/run_estimators.py",
    }
    if migrated_from is not None:
        metadata["migrated_from"] = str(Path(migrated_from).resolve())
        metadata["migration_note"] = (
            "Legacy cache arrays were reused without running an estimator; "
            "the recorded backend config is the current canonical config."
        )
    metadata_path = canonical_f0_metadata_path(cache_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, metadata_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _legacy_f0_cache_candidates(
    result_root: Path,
    backend: str,
    condition: str,
    keys,
):
    """Return old cache names for read-only compatibility during migration."""
    result_root = Path(result_root)
    key_token = hashlib.sha1(
        json.dumps([list(map(int, key)) for key in keys]).encode()
    ).hexdigest()[:8]
    if condition == "clean":
        candidates = []
        if len(keys) < 100:
            candidates.append(
                result_root / "pitch_notes_ana"
                / f"f0_trajectories_{backend}_{len(keys)}folds_{key_token}.npz"
            )
        # The historical canonical clean cache contains all 100 folds and may
        # safely serve a requested subset because pair_keys are validated.
        candidates.append(
            result_root / "pitch_notes_ana" / f"f0_trajectories_{backend}.npz"
        )
        return candidates
    old_scope = f"{len(keys)}folds_{key_token}"
    return [
        result_root / "pitch_factor_cache"
        / f"f0_{condition_slug(condition)}_{backend}_{old_scope}.npz"
    ]


def migrate_legacy_f0_cache(
    pairs,
    result_root: Path,
    backend: str,
    config: dict,
    condition: str = "clean",
):
    """Copy one exact legacy cache into the canonical namespace without F0 inference."""
    backend = str(backend).lower()
    condition = normalise_condition(condition)
    keys = [pair["key"] for pair in pairs]
    canonical_path = canonical_f0_cache_path(
        result_root, backend, condition, keys
    )
    if canonical_path.is_file():
        tracks = _load_npz_tracks(canonical_path, backend, keys)
        if set(tracks) != set(keys):
            raise ValueError(
                f"Canonical cache lacks requested fold keys: {canonical_path}"
            )
        return tracks, canonical_path, "cache_hit"

    for legacy_path in _legacy_f0_cache_candidates(
        result_root, backend, condition, keys
    ):
        if not legacy_path.is_file():
            continue
        tracks = _load_npz_tracks(legacy_path, backend, keys)
        if set(tracks) != set(keys):
            raise ValueError(
                f"Legacy cache lacks requested fold keys: {legacy_path}"
            )
        _save_npz_tracks(canonical_path, tracks, keys)
        _write_f0_cache_metadata(
            canonical_path,
            pairs,
            result_root,
            backend,
            config,
            condition,
            "migrated_legacy",
            migrated_from=legacy_path,
        )
        return tracks, canonical_path, "migrated_legacy"
    return None, canonical_path, "legacy_missing"


def load_or_extract_tracks(
    pairs,
    result_root: Path,
    backend: str,
    config: dict,
    condition: str = "clean",
    allow_extraction: bool = False,
    force_recompute: bool = False,
):
    """Load or extract one condition's paired Original/Reconstructed tracks.

    Canonical caches live under ``RESULT_ROOT/f0_estimator_cache`` and encode
    monkey, condition, track roles, estimator and fold scope in the filename.
    Legacy paths remain read-only fallbacks; ``run_estimators.py`` owns normal
    batch extraction and migration.
    """
    backend = str(backend).lower()
    condition = normalise_condition(condition)
    keys = [pair["key"] for pair in pairs]
    path = canonical_f0_cache_path(
        result_root, backend, condition, keys,
    )
    if path.is_file() and not force_recompute:
        tracks = _load_npz_tracks(path, backend, keys)
        if set(tracks) != set(keys):
            raise ValueError(f"Canonical cache lacks requested fold keys: {path}")
        return tracks, path, "cache_hit"

    # Development runs commonly request only the first few folds.  Reuse and
    # slice the official 100-fold cache instead of rerunning an estimator.
    if len(keys) < 100 and not force_recompute:
        full_scope_path = _full_scope_canonical_cache_path(
            result_root, backend, condition
        )
        if full_scope_path is not None:
            tracks = _load_npz_tracks(full_scope_path, backend, keys)
            if set(tracks) != set(keys):
                raise ValueError(
                    f"100-fold cache lacks requested fold keys: {full_scope_path}"
                )
            return tracks, full_scope_path, "cache_hit_100fold_subset"

    if not force_recompute:
        for legacy_path in _legacy_f0_cache_candidates(
            result_root, backend, condition, keys
        ):
            if not legacy_path.is_file():
                continue
            tracks = _load_npz_tracks(legacy_path, backend, keys)
            if set(tracks) != set(keys):
                raise ValueError(
                    f"Legacy cache lacks requested fold keys: {legacy_path}"
                )
            return tracks, legacy_path, "legacy_cache_hit"

    if not allow_extraction:
        raise FileNotFoundError(
            f"Missing canonical F0 cache: {path}. Run run_estimators.py for "
            "this monkey/estimator/condition, or explicitly allow extraction."
        )

    analysis_dir = find_analysis_dir()
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    from f0_backends import extract_f0

    partial_path = path.with_suffix(".partial.npz")
    tracks = {}
    if partial_path.is_file() and not force_recompute:
        tracks = _load_npz_tracks(partial_path, backend)
        unexpected_keys = set(tracks) - set(keys)
        if unexpected_keys:
            raise ValueError(
                f"Partial cache contains unexpected fold keys: {unexpected_keys}"
            )
        print(
            f"{condition} / {backend}: resumed {len(tracks)}/{len(pairs)} folds "
            f"from {partial_path.name}"
        )
    for index, pair in enumerate(pairs, start=1):
        if pair["key"] in tracks:
            continue
        item = {}
        for source, path_value in [
            ("reference", pair["ref_path"]),
            ("reconstructed", pair["pred_path"]),
        ]:
            audio = load_audio_mono(path_value, int(config["sample_rate"]))
            f0_hz, confidence, time_sec = extract_f0(
                audio, int(config["sample_rate"]), backend, config
            )
            item[source] = F0Track(f0_hz, confidence, time_sec, backend)
        tracks[pair["key"]] = item
        # A full job can take hours.  Keep an atomic fold-level checkpoint so
        # an interrupted process never has to restart completed folds.
        completed_keys = [key for key in keys if key in tracks]
        _save_npz_tracks(partial_path, tracks, completed_keys)
        print(f"{condition} / {backend}: extracted {index}/{len(pairs)} folds")
    _save_npz_tracks(path, tracks, keys)
    _write_f0_cache_metadata(
        path, pairs, result_root, backend, config, condition, "recomputed"
    )
    if partial_path.is_file():
        partial_path.unlink()
    return tracks, path, "recomputed"


def load_or_extract_reconstruction_tracks(
    pairs,
    clean_reference_tracks,
    result_root: Path,
    backend: str,
    config: dict,
    condition: str,
    allow_extraction: bool = False,
    force_recompute: bool = False,
):
    """Pair clean Original tracks with a condition's cached Reconstruction.

    The canonical on-disk cache always stores the matching condition Original
    and condition Reconstruction.  This function replaces only its Original
    half in memory, avoiding duplicate ``clean Original + SM Reconstruction``
    cache files.
    """
    backend = str(backend).lower()
    condition = normalise_condition(condition)
    if condition == "clean":
        raise ValueError(
            "Use load_or_extract_tracks(..., condition='clean') for clean reconstruction"
        )

    keys = [pair["key"] for pair in pairs]
    if set(clean_reference_tracks) != set(keys):
        raise ValueError("Clean reference tracks do not match the requested fold keys")

    condition_pairs = []
    for pair in pairs:
        s_id, m_id = int(pair["S"]), int(pair["M"])
        condition_pair = dict(pair)
        condition_pair["ref_path"] = (
            Path(pair["ref_path"]).parent
            / f"English_S{s_id}_M{m_id}_{condition}.wav"
        )
        if not condition_pair["ref_path"].is_file():
            raise FileNotFoundError(
                f"Missing condition Original audio: {condition_pair['ref_path']}"
            )
        condition_pairs.append(condition_pair)

    condition_tracks, path, status = load_or_extract_tracks(
        condition_pairs,
        result_root,
        backend,
        config,
        condition=condition,
        allow_extraction=allow_extraction,
        force_recompute=force_recompute,
    )
    tracks = {
        key: {
            "reference": clean_reference_tracks[key]["reference"],
            "reconstructed": condition_tracks[key]["reconstructed"],
        }
        for key in keys
    }
    return tracks, path, status


def _aligned_arrays(track_pair):
    ref = track_pair["reference"]
    pred = track_pair["reconstructed"]
    n = min(len(ref.f0_hz), len(pred.f0_hz))
    ref_time = np.asarray(ref.time_sec[:n], dtype=float)
    pred_time = np.asarray(pred.time_sec[:n], dtype=float)
    if not np.allclose(ref_time, pred_time, atol=1e-8, rtol=0.0):
        raise AssertionError("Reference and reconstruction F0 grids are not aligned")
    return (
        ref_time,
        np.asarray(ref.f0_hz[:n], dtype=float),
        np.asarray(pred.f0_hz[:n], dtype=float),
        np.asarray(ref.confidence[:n], dtype=float),
        np.asarray(pred.confidence[:n], dtype=float),
    )


def ensure_pitch_metric(metric: str) -> str:
    """Validate a canonical H/O/R pitch-accuracy metric name."""
    metric = str(metric).strip().lower()
    if metric not in PITCH_ACCURACY_METRICS:
        raise ValueError(f"metric must be one of {PITCH_ACCURACY_METRICS}")
    return metric


def pitch_metric_label(metric: str, chroma: bool = False) -> str:
    """Return a reader-facing metric label."""
    spec = PITCH_METRIC_SPECS[ensure_pitch_metric(metric)]
    return spec["chroma_label" if chroma else "label"]


def _safe_rate(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast(numerator, denominator).shape, np.nan, dtype=float),
        where=denominator > 0,
    )


def _threshold_pairs(original_thresholds, reconstructed_thresholds=None):
    original = [float(value) for value in original_thresholds]
    if reconstructed_thresholds is None:
        reconstructed = original.copy()
    elif np.isscalar(reconstructed_thresholds):
        reconstructed = [float(reconstructed_thresholds)] * len(original)
    else:
        reconstructed = [float(value) for value in reconstructed_thresholds]
    if len(original) != len(reconstructed):
        raise ValueError("Original/Reconstructed threshold lists must have equal length")
    return list(zip(original, reconstructed))


def _pair_error(reference_hz, candidate_hz):
    if not (np.isfinite(reference_hz) and reference_hz > 0):
        return np.nan, np.nan
    if not (np.isfinite(candidate_hz) and candidate_hz > 0):
        return np.nan, np.nan
    signed = float(pitch_error_cents(reference_hz, candidate_hz))
    chroma = float(chroma_error_cents_from_signed(signed))
    return signed, chroma


def _attach_pitch_contract_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Derive all H/O/R rates and temporary legacy aliases from counts."""
    result = table.copy()
    result["rpa_orig"] = _safe_rate(
        result["n_rpa_orig_correct"], result["n_human_voiced"]
    )
    result["rca_orig"] = _safe_rate(
        result["n_rca_orig_correct"], result["n_human_voiced"]
    )
    result["rpa_recon"] = _safe_rate(
        result["n_rpa_recon_correct"], result["n_human_voiced"]
    )
    result["rca_recon"] = _safe_rate(
        result["n_rca_recon_correct"], result["n_human_voiced"]
    )
    result["pda"] = _safe_rate(
        result["n_pda_correct"], result["n_pda_evaluated"]
    )
    result["rca_pda"] = _safe_rate(
        result["n_rca_pda_correct"], result["n_pda_evaluated"]
    )
    result["original_voiced_recall"] = _safe_rate(
        result["n_human_original_voiced"], result["n_human_voiced"]
    )
    result["reconstructed_voiced_recall"] = _safe_rate(
        result["n_human_reconstructed_voiced"], result["n_human_voiced"]
    )
    result["pda_candidate_recall"] = _safe_rate(
        result["n_original_reconstructed_voiced"], result["n_original_voiced"]
    )
    result["original_unvoiced_reconstructed_voiced_rate"] = _safe_rate(
        result["n_original_unvoiced_reconstructed_voiced"],
        result["n_original_unvoiced"],
    )

    # Compatibility aliases: the historical code called O↔R pitch accuracy
    # ``rpa``/``rca``.  Keep it runnable while all active notebooks migrate to
    # the explicit PDA names.
    result["n_eval"] = result["n_pda_evaluated"]
    result["n_rpa_correct"] = result["n_pda_correct"]
    result["n_rca_correct"] = result["n_rca_pda_correct"]
    result["n_missing_prediction"] = 0
    result["rpa"] = result["pda"]
    result["rca"] = result["rca_pda"]
    result["coverage"] = result["original_voiced_recall"]
    result["voiced_recall"] = result["pda_candidate_recall"]
    result["hallucination_rate"] = (
        result["original_unvoiced_reconstructed_voiced_rate"]
    )
    return result


def select_pitch_metric(metrics: pd.DataFrame, metric: str = "pda") -> pd.DataFrame:
    """Add generic selected-metric columns without discarding other metrics."""
    metric = ensure_pitch_metric(metric)
    spec = PITCH_METRIC_SPECS[metric]
    result = metrics.copy()
    result["pitch_accuracy_metric"] = metric
    result["pitch_accuracy"] = result[metric]
    result["chroma_accuracy"] = result[spec["chroma_metric"]]
    result["n_accuracy_denominator"] = result[spec["denominator"]]
    result["n_pitch_correct"] = result[spec["pitch_correct"]]
    result["n_chroma_correct"] = result[spec["chroma_correct"]]
    result["n_candidate_voiced"] = result[spec["candidate_pair"]]
    result["candidate_voiced_recall"] = result[spec["candidate_recall"]]
    return result


def evaluate_pitch(
    pairs,
    tracks,
    level: str,
    thresholds=DEFAULT_THRESHOLDS,
    mask_type: str | None = None,
    tolerance_cents: float = DEFAULT_TOLERANCE_HALF_WIDTH_CENTS,
    note_aggregation: str = "median",
    min_confident_frames: int = 1,
    reconstructed_thresholds=None,
    pda_frame_mask: str = "both",
):
    """Compute RPA_orig, RPA_recon, and PDA under the H/O/R contract.

    ``tolerance_cents`` is the non-negative half-width applied as
    ``abs(cents_error) <= tolerance_cents``.  The formal project criterion is
    therefore displayed as ``tolerance = ±50 cents``.

    Human voicing is currently the binary annotated-note interval because the
    dataset has no framewise human voicing-confidence track. At frame level,
    ``thresholds`` controls Original confidence and
    ``reconstructed_thresholds`` controls Reconstructed confidence; PDA uses
    the frame mask selected by ``pda_frame_mask``. At note level, confidence thresholds are
    metadata only: every valid F0 frame in the note interval is used, and each
    track is independently reduced to one representative F0 by
    ``note_aggregation``.

    ``min_confident_frames`` is retained for API compatibility. At note level
    it is the minimum number of valid F0 frames, regardless of confidence.

    At frame level, ``pda_frame_mask='both'`` requires both Original and
    Reconstructed confidence to exceed the threshold.  With
    ``pda_frame_mask='original'``, Original confidence alone selects the PDA
    denominator; a missing/invalid Reconstructed F0 is then counted as an
    incorrect prediction.  This option does not affect note-level PDA or any
    RPA metric.

    ``mask_type`` is accepted only so historical notebooks still run.  It no
    longer changes a metric denominator: RPA uses H, while PDA requires both O
    and R at the selected evaluation level.
    """
    level = ensure_level(level)
    pda_frame_mask = str(pda_frame_mask).strip().lower()
    if pda_frame_mask not in {"both", "original"}:
        raise ValueError("pda_frame_mask must be 'both' or 'original'")
    min_valid_note_frames = int(min_confident_frames)
    threshold_pairs = _threshold_pairs(thresholds, reconstructed_thresholds)
    if level == "frame" and pda_frame_mask == "both" and any(
        not np.isclose(original, reconstructed)
        for original, reconstructed in threshold_pairs
    ):
        raise ValueError(
            "PDA requires the same confidence threshold for Original and "
            "Reconstructed tracks"
        )
    if mask_type not in {None, "pitch_acc_contract"}:
        warnings.warn(
            "mask_type is deprecated for named pitch metrics; the H/O/R "
            "contract now fixes each denominator explicitly",
            FutureWarning,
            stacklevel=2,
        )

    rows = []
    for pair in pairs:
        time, original_f0, reconstructed_f0, original_conf, reconstructed_conf = (
            _aligned_arrays(tracks[pair["key"]])
        )
        original_valid = valid_f0(original_f0)
        reconstructed_valid = valid_f0(reconstructed_f0)

        for note in pair["notes"]:
            start = max(0.0, float(note["start_time_s"]))
            end = min(4.2, float(note["end_time_s"]))
            note_mask = (time >= start) & (time < end)
            label_hz = float(note["label_hz"])
            human_valid = bool(np.isfinite(label_hz) and label_hz > 0)
            human_mask = note_mask & human_valid
            n_note_frames = int(note_mask.sum())
            common = {
                "S": pair["S"], "M": pair["M"],
                "condition": pair["condition"],
                "note_uid": note["note_uid"],
                "note_index": int(note["note_index"]),
                "label_midi": int(note["label_midi"]),
                "label_hz": label_hz,
                "seen_group": note["seen_group"],
                "n_train_notes_same_pitch": int(note["n_train_notes_same_pitch"]),
                "train_pitch_probability": float(note["train_pitch_probability"]),
                "level": level,
                "human_voicing_source": "binary_note_annotation",
                "mask_type": "pitch_acc_contract",
                "pda_frame_mask": (
                    pda_frame_mask if level == "frame" else "not_applicable"
                ),
                "legacy_mask_type": mask_type,
                "n_note_frames": n_note_frames,
            }

            for original_threshold, reconstructed_threshold in threshold_pairs:
                original_confident_mask = (
                    note_mask & original_valid & np.isfinite(original_conf)
                    & (original_conf > original_threshold)
                )
                reconstructed_confident_mask = (
                    note_mask & reconstructed_valid
                    & np.isfinite(reconstructed_conf)
                    & (reconstructed_conf > reconstructed_threshold)
                )
                if level == "frame":
                    original_mask = original_confident_mask
                    reconstructed_mask = reconstructed_confident_mask
                else:
                    # Note-level analysis deliberately does not use confidence:
                    # aggregate every valid F0 frame in the annotated interval.
                    original_mask = note_mask & original_valid
                    reconstructed_mask = note_mask & reconstructed_valid
                both_confident = (
                    original_confident_mask & reconstructed_confident_mask
                )
                original_low = human_mask & ~original_mask
                original_unconfident = human_mask & ~original_confident_mask
                confident_fill_in = (
                    original_unconfident & reconstructed_confident_mask
                )

                threshold_metadata = {
                    "original_confidence_threshold": original_threshold,
                    "reconstructed_confidence_threshold": reconstructed_threshold,
                    "confidence_threshold": (
                        original_threshold
                        if np.isclose(original_threshold, reconstructed_threshold)
                        else np.nan
                    ),
                }
                frame_diagnostics = {
                    # Confidence diagnostics remain available at both levels,
                    # but note-level RPA/PDA never consume these masks.
                    "n_ref_confident_frames": int(original_confident_mask.sum()),
                    "n_pred_confident_frames": int(
                        reconstructed_confident_mask.sum()
                    ),
                    "n_both_confident_frames": int(both_confident.sum()),
                    "n_ref_unconfident_frames": int(original_unconfident.sum()),
                    "n_hallucinated_frames": int(confident_fill_in.sum()),
                }

                if level == "frame":
                    h_o_pair = human_mask & original_mask
                    h_r_pair = human_mask & reconstructed_mask
                    # PDA denominator selection is independent of the RPA masks.
                    # In Original-only mode, low Reconstructed confidence is
                    # ignored, but an invalid Reconstructed F0 remains an error.
                    pda_eval_mask = (
                        original_mask & reconstructed_mask
                        if pda_frame_mask == "both"
                        else original_mask
                    )
                    o_r_pair = pda_eval_mask & reconstructed_valid

                    error_h_o = np.full(len(time), np.nan)
                    error_h_r = np.full(len(time), np.nan)
                    error_o_r = np.full(len(time), np.nan)
                    error_h_o[h_o_pair] = pitch_error_cents(label_hz, original_f0[h_o_pair])
                    error_h_r[h_r_pair] = pitch_error_cents(
                        label_hz, reconstructed_f0[h_r_pair]
                    )
                    error_o_r[o_r_pair] = pitch_error_cents(
                        original_f0[o_r_pair], reconstructed_f0[o_r_pair]
                    )
                    chroma_h_o = chroma_error_cents_from_signed(error_h_o)
                    chroma_h_r = chroma_error_cents_from_signed(error_h_r)
                    chroma_o_r = chroma_error_cents_from_signed(error_o_r)

                    counts = {
                        "n_human_voiced": int(human_mask.sum()),
                        "n_original_voiced": int(original_mask.sum()),
                        "n_reconstructed_voiced": int(reconstructed_mask.sum()),
                        "n_human_original_voiced": int(h_o_pair.sum()),
                        "n_human_reconstructed_voiced": int(h_r_pair.sum()),
                        "n_original_reconstructed_voiced": int(o_r_pair.sum()),
                        "n_pda_evaluated": int(pda_eval_mask.sum()),
                        "n_original_unvoiced": int(original_low.sum()),
                        "n_original_unvoiced_reconstructed_voiced": int(
                            confident_fill_in.sum()
                        ),
                        "n_rpa_orig_correct": int(
                            (h_o_pair & (np.abs(error_h_o) <= tolerance_cents)).sum()
                        ),
                        "n_rca_orig_correct": int(
                            (h_o_pair & (chroma_h_o <= tolerance_cents)).sum()
                        ),
                        "n_rpa_recon_correct": int(
                            (h_r_pair & (np.abs(error_h_r) <= tolerance_cents)).sum()
                        ),
                        "n_rca_recon_correct": int(
                            (h_r_pair & (chroma_h_r <= tolerance_cents)).sum()
                        ),
                        "n_pda_correct": int(
                            (o_r_pair & (np.abs(error_o_r) <= tolerance_cents)).sum()
                        ),
                        "n_rca_pda_correct": int(
                            (o_r_pair & (chroma_o_r <= tolerance_cents)).sum()
                        ),
                    }
                    representatives = {
                        "human_representative_f0": label_hz,
                        "original_representative_f0": np.nan,
                        "reconstructed_representative_f0": np.nan,
                        "signed_cents_error_orig": np.nan,
                        "signed_cents_error_recon": np.nan,
                        "signed_cents_error_pda": np.nan,
                        "absolute_cents_error_orig": np.nan,
                        "absolute_cents_error_recon": np.nan,
                        "absolute_cents_error_pda": np.nan,
                    }
                else:
                    original_values = original_f0[original_mask]
                    reconstructed_values = reconstructed_f0[reconstructed_mask]
                    has_human = human_valid
                    has_original = len(original_values) >= min_valid_note_frames
                    has_reconstructed = (
                        len(reconstructed_values) >= min_valid_note_frames
                    )
                    has_pda_pair = has_original and has_reconstructed
                    human_rep = label_hz if has_human else np.nan
                    original_rpa_rep = (
                        aggregate_f0(original_values, note_aggregation)
                        if has_original else np.nan
                    )
                    reconstructed_rpa_rep = (
                        aggregate_f0(reconstructed_values, note_aggregation)
                        if has_reconstructed else np.nan
                    )
                    # RPA and PDA use the same per-track note representatives.
                    # In particular, PDA does not intersect confidence masks (or
                    # time masks beyond the shared annotated note interval).
                    original_pda_rep = original_rpa_rep if has_pda_pair else np.nan
                    reconstructed_pda_rep = (
                        reconstructed_rpa_rep if has_pda_pair else np.nan
                    )
                    signed_h_o, chroma_h_o = _pair_error(human_rep, original_rpa_rep)
                    signed_h_r, chroma_h_r = _pair_error(
                        human_rep, reconstructed_rpa_rep
                    )
                    signed_o_r, chroma_o_r = _pair_error(
                        original_pda_rep, reconstructed_pda_rep
                    )

                    counts = {
                        "n_human_voiced": int(has_human),
                        "n_original_voiced": int(has_original),
                        "n_reconstructed_voiced": int(has_reconstructed),
                        "n_human_original_voiced": int(has_human and has_original),
                        "n_human_reconstructed_voiced": int(has_human and has_reconstructed),
                        "n_original_reconstructed_voiced": int(has_pda_pair),
                        "n_pda_evaluated": int(has_pda_pair),
                        "n_original_unvoiced": int(has_human and not has_original),
                        "n_original_unvoiced_reconstructed_voiced": int(
                            has_human and not has_original and has_reconstructed
                        ),
                        "n_rpa_orig_correct": int(
                            has_human and has_original
                            and abs(signed_h_o) <= tolerance_cents
                        ),
                        "n_rca_orig_correct": int(
                            has_human and has_original
                            and chroma_h_o <= tolerance_cents
                        ),
                        "n_rpa_recon_correct": int(
                            has_human and has_reconstructed
                            and abs(signed_h_r) <= tolerance_cents
                        ),
                        "n_rca_recon_correct": int(
                            has_human and has_reconstructed
                            and chroma_h_r <= tolerance_cents
                        ),
                        "n_pda_correct": int(
                            has_pda_pair
                            and abs(signed_o_r) <= tolerance_cents
                        ),
                        "n_rca_pda_correct": int(
                            has_pda_pair
                            and chroma_o_r <= tolerance_cents
                        ),
                    }
                    representatives = {
                        "human_representative_f0": human_rep,
                        # These generic O/R representatives form the PDA pair.
                        "original_representative_f0": original_pda_rep,
                        "reconstructed_representative_f0": reconstructed_pda_rep,
                        "signed_cents_error_orig": signed_h_o,
                        "signed_cents_error_recon": signed_h_r,
                        "signed_cents_error_pda": signed_o_r,
                        "absolute_cents_error_orig": abs(signed_h_o),
                        "absolute_cents_error_recon": abs(signed_h_r),
                        "absolute_cents_error_pda": abs(signed_o_r),
                    }

                row = {
                    **common,
                    **threshold_metadata,
                    **frame_diagnostics,
                    **counts,
                    **representatives,
                }
                # Temporary O/R aliases used by existing error-distribution code.
                row["signed_cents_error"] = row["signed_cents_error_pda"]
                row["absolute_cents_error"] = row["absolute_cents_error_pda"]
                row["ref_representative_f0"] = row["original_representative_f0"]
                row["pred_representative_f0"] = row["reconstructed_representative_f0"]
                rows.append(row)

    result = _attach_pitch_contract_rates(pd.DataFrame(rows))
    for pitch_col, chroma_col in [
        ("n_rpa_orig_correct", "n_rca_orig_correct"),
        ("n_rpa_recon_correct", "n_rca_recon_correct"),
        ("n_pda_correct", "n_rca_pda_correct"),
    ]:
        if (result[chroma_col] < result[pitch_col]).any():
            raise AssertionError(f"{chroma_col} cannot be smaller than {pitch_col}")
    return select_pitch_metric(result, "pda")


_COUNT_COLUMNS = [
    "n_human_voiced", "n_original_voiced", "n_reconstructed_voiced",
    "n_human_original_voiced", "n_human_reconstructed_voiced",
    "n_original_reconstructed_voiced", "n_pda_evaluated", "n_original_unvoiced",
    "n_original_unvoiced_reconstructed_voiced",
    "n_rpa_orig_correct", "n_rca_orig_correct",
    "n_rpa_recon_correct", "n_rca_recon_correct",
    "n_pda_correct", "n_rca_pda_correct", "n_note_frames",
    "n_ref_confident_frames", "n_pred_confident_frames",
    "n_both_confident_frames", "n_ref_unconfident_frames",
    "n_hallucinated_frames",
]


def _aggregate_metric_counts(metrics: pd.DataFrame, keys, extra_aggregations=None):
    aggregations = {column: (column, "sum") for column in _COUNT_COLUMNS}
    if extra_aggregations:
        aggregations.update(extra_aggregations)
    result = metrics.groupby(keys, as_index=False).agg(**aggregations)
    result["confidence_threshold"] = np.where(
        np.isclose(
            result["original_confidence_threshold"],
            result["reconstructed_confidence_threshold"],
        ),
        result["original_confidence_threshold"],
        np.nan,
    )
    result["mask_type"] = "pitch_acc_contract"
    return _attach_pitch_contract_rates(result)


def aggregate_fold_metrics(
    metrics: pd.DataFrame,
    metric: str = "pda",
) -> pd.DataFrame:
    """Aggregate rows per fold using each named metric's own denominator."""
    keys = [
        "S", "M", "condition", "level",
        "original_confidence_threshold", "reconstructed_confidence_threshold",
    ]
    result = _aggregate_metric_counts(metrics, keys)
    return select_pitch_metric(result, metric)


def aggregate_pitch_metrics(
    metrics: pd.DataFrame,
    metric: str = "pda",
) -> pd.DataFrame:
    """Aggregate rows per label MIDI under the H/O/R metric contract."""
    keys = [
        "label_midi", "label_hz", "level",
        "original_confidence_threshold", "reconstructed_confidence_threshold",
    ]
    result = _aggregate_metric_counts(
        metrics,
        keys,
        extra_aggregations={"n_notes": ("note_uid", "size")},
    )
    return select_pitch_metric(result, metric)


def confidence_table(pairs, tracks, max_points_per_source: int | None = None):
    """Collect Original/Reconstructed confidence with deterministic downsampling."""
    rows = []
    for pair in pairs:
        for source_key, source_label in [
            ("reference", "Original"), ("reconstructed", "Reconstructed")
        ]:
            values = np.asarray(tracks[pair["key"]][source_key].confidence, dtype=float)
            values = values[np.isfinite(values)]
            if max_points_per_source and len(values) > max_points_per_source:
                index = np.linspace(0, len(values) - 1, max_points_per_source, dtype=int)
                values = values[index]
            rows.extend({
                "S": pair["S"], "M": pair["M"],
                "source": source_label, "confidence": float(value),
            } for value in values)
    return pd.DataFrame(rows)


def collect_pitch_pairs(
    pairs,
    tracks,
    level: str,
    threshold: float,
    mask_type: str | None = None,
    tolerance_cents: float = DEFAULT_TOLERANCE_HALF_WIDTH_CENTS,
    note_aggregation: str = "median",
    reference_midi: str = "metric",
    metric: str = "pda",
    reconstructed_threshold: float | None = None,
    pda_frame_mask: str = "both",
):
    """Collect valid cents-error pairs for one H/O/R metric.

    At frame level, confidence masks select candidate pairs. At note level,
    confidence is ignored and all valid F0 frames in each annotated note are
    independently aggregated for Original and Reconstructed.
    """
    level = ensure_level(level)
    metric = ensure_pitch_metric(metric)
    pda_frame_mask = str(pda_frame_mask).strip().lower()
    if pda_frame_mask not in {"both", "original"}:
        raise ValueError("pda_frame_mask must be 'both' or 'original'")
    if reference_midi not in {"label", "estimated", "metric"}:
        raise ValueError("reference_midi must be label, estimated, or metric")
    if mask_type not in {None, "pitch_acc_contract"}:
        warnings.warn(
            "mask_type is ignored by collect_pitch_pairs; metric selects the H/O/R pair",
            FutureWarning,
            stacklevel=2,
        )
    original_threshold = float(threshold)
    reconstructed_threshold = (
        original_threshold
        if reconstructed_threshold is None else float(reconstructed_threshold)
    )
    if level == "frame" and pda_frame_mask == "both" and not np.isclose(
        original_threshold, reconstructed_threshold
    ):
        raise ValueError(
            "PDA/cents-pair analysis requires the same confidence threshold "
            "for Original and Reconstructed tracks"
        )
    rows = []
    for pair in pairs:
        time, original_f0, reconstructed_f0, original_conf, reconstructed_conf = (
            _aligned_arrays(tracks[pair["key"]])
        )
        original_valid = valid_f0(original_f0)
        reconstructed_valid = valid_f0(reconstructed_f0)
        for note in pair["notes"]:
            note_mask = (
                (time >= max(0.0, float(note["start_time_s"])))
                & (time < min(4.2, float(note["end_time_s"])))
            )
            label_hz = float(note["label_hz"])
            human_mask = note_mask & np.isfinite(label_hz) & (label_hz > 0)
            if level == "frame":
                original_mask = (
                    note_mask & original_valid & np.isfinite(original_conf)
                    & (original_conf > original_threshold)
                )
                reconstructed_mask = (
                    note_mask & reconstructed_valid
                    & np.isfinite(reconstructed_conf)
                    & (reconstructed_conf > reconstructed_threshold)
                )
            else:
                # No confidence filtering at note level.
                original_mask = note_mask & original_valid
                reconstructed_mask = note_mask & reconstructed_valid

            if level == "frame":
                if metric == "rpa_orig":
                    selected = human_mask & original_mask
                    reference_values = np.full(len(time), label_hz)
                    candidate_values = original_f0
                    reference_source, candidate_source = "Human", "Original"
                elif metric == "rpa_recon":
                    selected = human_mask & reconstructed_mask
                    reference_values = np.full(len(time), label_hz)
                    candidate_values = reconstructed_f0
                    reference_source, candidate_source = "Human", "Reconstructed"
                else:
                    selected = original_mask & reconstructed_valid
                    if pda_frame_mask == "both":
                        selected &= reconstructed_mask
                    reference_values = original_f0
                    candidate_values = reconstructed_f0
                    reference_source, candidate_source = "Original", "Reconstructed"
                indices = np.flatnonzero(selected)
                for index in indices:
                    signed = float(
                        pitch_error_cents(reference_values[index], candidate_values[index])
                    )
                    ref_midi_value = (
                        int(note["label_midi"])
                        if reference_midi == "label"
                        else int(np.rint(hz_to_midi(reference_values[index])))
                    )
                    rows.append({
                        "S": pair["S"], "M": pair["M"], "note_uid": note["note_uid"],
                        "label_midi": int(note["label_midi"]),
                        "pitch_accuracy_metric": metric,
                        "reference_source": reference_source,
                        "candidate_source": candidate_source,
                        "reference_midi": ref_midi_value,
                        "predicted_midi": int(np.rint(hz_to_midi(candidate_values[index]))),
                        "signed_cents_error": signed,
                        "absolute_cents_error": abs(signed),
                        "pitch_correct": abs(signed) <= tolerance_cents,
                        "chroma_correct": (
                            float(chroma_error_cents_from_signed(signed))
                            <= tolerance_cents
                        ),
                        "seen_group": note["seen_group"],
                    })
            else:
                if metric == "rpa_orig":
                    original_rep = (
                        aggregate_f0(original_f0[original_mask], note_aggregation)
                        if original_mask.any() else np.nan
                    )
                    reference_value, candidate_value = label_hz, original_rep
                    reference_source, candidate_source = "Human", "Original"
                elif metric == "rpa_recon":
                    reconstructed_rep = (
                        aggregate_f0(
                            reconstructed_f0[reconstructed_mask], note_aggregation
                        )
                        if reconstructed_mask.any() else np.nan
                    )
                    reference_value, candidate_value = label_hz, reconstructed_rep
                    reference_source, candidate_source = "Human", "Reconstructed"
                else:
                    original_rep = (
                        aggregate_f0(original_f0[original_mask], note_aggregation)
                        if original_mask.any() else np.nan
                    )
                    reconstructed_rep = (
                        aggregate_f0(
                            reconstructed_f0[reconstructed_mask], note_aggregation
                        )
                        if reconstructed_mask.any() else np.nan
                    )
                    reference_value, candidate_value = original_rep, reconstructed_rep
                    reference_source, candidate_source = "Original", "Reconstructed"
                if not (
                    np.isfinite(reference_value) and reference_value > 0
                    and np.isfinite(candidate_value) and candidate_value > 0
                ):
                    continue
                signed = float(pitch_error_cents(reference_value, candidate_value))
                ref_midi_value = (
                    int(note["label_midi"]) if reference_midi == "label"
                    else int(np.rint(hz_to_midi(reference_value)))
                )
                rows.append({
                    "S": pair["S"], "M": pair["M"], "note_uid": note["note_uid"],
                    "label_midi": int(note["label_midi"]),
                    "pitch_accuracy_metric": metric,
                    "reference_source": reference_source,
                    "candidate_source": candidate_source,
                    "reference_midi": ref_midi_value,
                    "predicted_midi": int(np.rint(hz_to_midi(candidate_value))),
                    "signed_cents_error": signed,
                    "absolute_cents_error": abs(signed),
                    "pitch_correct": abs(signed) <= tolerance_cents,
                    "chroma_correct": (
                        float(chroma_error_cents_from_signed(signed)) <= tolerance_cents
                    ),
                    "seen_group": note["seen_group"],
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["rpa_correct"] = result["pitch_correct"]
        result["rca_correct"] = result["chroma_correct"]
        result["original_confidence_threshold"] = original_threshold
        result["reconstructed_confidence_threshold"] = reconstructed_threshold
        result["pda_frame_mask"] = (
            pda_frame_mask if level == "frame" else "not_applicable"
        )
    return result


def shift_track_to_grid(ref_times, pred_track: F0Track, shift_sec: float):
    """Shift reconstructed timestamps and map nearest values to a reference grid."""
    ref_times = np.asarray(ref_times, dtype=float)
    pred_times = np.asarray(pred_track.time_sec, dtype=float)
    source_times = ref_times - float(shift_sec)
    right = np.searchsorted(pred_times, source_times, side="left")
    left = np.clip(right - 1, 0, len(pred_times) - 1)
    right = np.clip(right, 0, len(pred_times) - 1)
    nearest = np.where(
        np.abs(pred_times[left] - source_times) <= np.abs(pred_times[right] - source_times),
        left, right,
    )
    step = float(np.median(np.diff(pred_times)))
    matched = np.abs(pred_times[nearest] - source_times) <= step / 2.0 + 1e-9
    f0_hz = np.full(len(ref_times), np.nan)
    confidence = np.zeros(len(ref_times))
    source_f0 = np.asarray(pred_track.f0_hz, dtype=float)
    source_conf = np.asarray(pred_track.confidence, dtype=float)
    f0_hz[matched] = source_f0[nearest[matched]]
    confidence[matched] = source_conf[nearest[matched]]
    return F0Track(f0_hz, confidence, ref_times.copy(), pred_track.backend)


def scan_time_shift(
    pairs,
    tracks,
    level: str,
    shifts_sec,
    threshold: float,
    mask_type: str | None = None,
    tolerance_cents: float = DEFAULT_TOLERANCE_HALF_WIDTH_CENTS,
    note_aggregation: str = "median",
    metric: str = "pda",
    reconstructed_threshold: float | None = None,
    pda_frame_mask: str = "both",
):
    """Return fold-level named pitch accuracy across reconstruction shifts."""
    metric = ensure_pitch_metric(metric)
    rows = []
    for shift in shifts_sec:
        shifted = {}
        for pair in pairs:
            original = tracks[pair["key"]]["reference"]
            shifted[pair["key"]] = {
                "reference": original,
                "reconstructed": shift_track_to_grid(
                    original.time_sec, tracks[pair["key"]]["reconstructed"], shift
                ),
            }
        metrics = evaluate_pitch(
            pairs, shifted, level, [threshold], mask_type,
            tolerance_cents, note_aggregation,
            reconstructed_thresholds=reconstructed_threshold,
            pda_frame_mask=pda_frame_mask,
        )
        fold = aggregate_fold_metrics(metrics, metric=metric)
        fold["time_shift_sec"] = float(shift)
        rows.append(fold)
    return pd.concat(rows, ignore_index=True)


def level_note(level: str) -> str:
    return (
        "Frame level: each eligible F0 frame is scored; note intervals only define regions."
        if ensure_level(level) == "frame"
        else "Note level: all valid F0 frames inside each note are aggregated without confidence filtering."
    )
