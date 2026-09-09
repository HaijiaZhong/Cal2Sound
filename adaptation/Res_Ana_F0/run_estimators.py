#!/usr/bin/env python3
"""Precompute F0 trajectories for all Original/Reconstructed audio pairs.

This is the only batch entry point that should run expensive F0 estimators.
Analysis notebooks should normally set ``ALLOW_F0_EXTRACTION=False`` and read
the caches produced here through ``pitch_analysis_shared.load_or_extract_*``.

Canonical output
----------------
Each ``condition × estimator × fold scope`` produces one NPZ plus one JSON
sidecar under::

    <RESULT_ROOT>/f0_estimator_cache/

Example::

    f0_tracks__monkey-M102D__condition-sm-0p6\
    __roles-original-reconstructed__estimator-crepe\
    __scope-100folds-08e865d0.npz

The NPZ contains both the condition Original and matching condition
Reconstructed tracks.  The JSON records monkey, condition, estimator config,
package/runtime metadata, fold keys and exact audio paths.  A temporary file is
written first and atomically renamed, so interrupted jobs cannot masquerade as
complete caches.  A ``*.partial.npz`` fold checkpoint is updated during long
jobs; restarting the same command resumes completed folds.  Completed caches
are skipped.

Typical use
-----------
Run from the Neu2Sound repository root.  Quote SM condition names in shells.

1. Inspect planned jobs without inference or writes::

       python adaptation/Res_Ana_F0/run_estimators.py \
         --monkey M102D --dry-run

2. Compute every estimator for clean and all three SM conditions::

       python adaptation/Res_Ana_F0/run_estimators.py --monkey M102D

3. Compute selected estimators/conditions::

       python adaptation/Res_Ana_F0/run_estimators.py \
         --monkey M160E \
         --estimators crepe pesto swiftf0 \
         --conditions clean 'SM(3.0)' 'SM(1.8)' 'SM(0.6)'

4. Migrate old exact-scope caches into the new namespace without inference::

       python adaptation/Res_Ana_F0/run_estimators.py \
         --monkey ZJM --migrate-only

Important options
-----------------
``--result-root`` overrides the built-in monkey directory mapping.
``--reference-audio-dir`` overrides ``/data/zhj/audio_VAE/data``.
``--device`` updates GPU/CPU-capable backend configs (default ``cuda:0``;
backend adapters already implement their supported fallback behavior).
``--max-folds`` is only for development caches; omit it for the official 100
fold analysis. ``--overwrite`` explicitly recomputes canonical files and should
only be used after an intentional estimator/config/version change.

This file does not calculate RPA/PDA or create figures.  It only extracts and
stores ``f0_hz``, estimator-native ``confidence`` and ``time_sec`` arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ANALYSIS_DIR = Path(__file__).resolve().parent
ADAPTATION_DIR = ANALYSIS_DIR.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from f0_backends import SUPPORTED_BACKENDS  # noqa: E402
from pitch_analysis_shared import (  # noqa: E402
    SM_CONDITIONS,
    backend_configs,
    build_manifest,
    canonical_f0_cache_path,
    infer_monkey_id,
    load_or_extract_tracks,
    migrate_legacy_f0_cache,
    normalise_condition,
)


DEFAULT_RESULT_ROOTS = {
    "ZJM": ADAPTATION_DIR / "results_20260721_ZJM_pixelLevel_100folds",
    "M102D": ADAPTATION_DIR / "results_20260722_M102D_pixelLevel_100folds",
    "M160E": ADAPTATION_DIR / "results_20260814_M160E_pixelLevel_100folds",
}
DEFAULT_REFERENCE_AUDIO_DIR = Path("/data/zhj/audio_VAE/data")


def _condition_argument(value: str) -> str:
    """Accept canonical names plus convenient unparenthesized CLI aliases."""
    compact = value.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "clean": "clean",
        "sm3.0": "SM(3.0)", "sm3p0": "SM(3.0)", "sm(3.0)": "SM(3.0)",
        "sm1.8": "SM(1.8)", "sm1p8": "SM(1.8)", "sm(1.8)": "SM(1.8)",
        "sm0.6": "SM(0.6)", "sm0p6": "SM(0.6)", "sm(0.6)": "SM(0.6)",
    }
    if compact not in aliases:
        raise argparse.ArgumentTypeError(
            f"Unsupported condition {value!r}; use clean, SM(3.0), SM(1.8), SM(0.6)"
        )
    return normalise_condition(aliases[compact])


def _estimator_arguments(values: list[str]) -> tuple[str, ...]:
    requested = [value.strip().lower() for value in values]
    if requested == ["all"]:
        return tuple(SUPPORTED_BACKENDS)
    unknown = sorted(set(requested) - set(SUPPORTED_BACKENDS))
    if unknown:
        raise ValueError(
            f"Unknown estimator(s): {unknown}; choose from {SUPPORTED_BACKENDS}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Estimator list contains duplicates")
    return tuple(requested)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute canonical Original/Reconstructed F0 caches."
    )
    parser.add_argument(
        "--monkey", required=True, type=str.upper,
        help=(
            "Monkey/model identifier used in the cache filename. ZJM, M102D "
            "and M160E have built-in result-root mappings; use --result-root "
            "for a future monkey."
        ),
    )
    parser.add_argument(
        "--result-root", type=Path,
        help="Override the built-in results directory for the selected monkey.",
    )
    parser.add_argument(
        "--reference-audio-dir", type=Path,
        default=DEFAULT_REFERENCE_AUDIO_DIR,
        help="Directory containing English_S*_M*_SM(...).wav Original audio.",
    )
    parser.add_argument(
        "--estimators", nargs="+", default=["all"],
        help="Estimator names, or 'all' (default).",
    )
    parser.add_argument(
        "--conditions", nargs="+", type=_condition_argument,
        default=list(SM_CONDITIONS),
        help="Conditions to process (default: clean and all SM conditions).",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device string applied to backends that expose a device option.",
    )
    parser.add_argument(
        "--max-folds", type=int,
        help="Development-only fold limit; omit for the official 100 folds.",
    )
    parser.add_argument(
        "--migrate-only", action="store_true",
        help="Copy exact legacy caches to canonical names; never run estimators.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print canonical jobs and cache presence; do not write or infer F0.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Intentionally recompute existing canonical caches.",
    )
    args = parser.parse_args(argv)
    if args.max_folds is not None and not 1 <= args.max_folds <= 100:
        parser.error("--max-folds must be within 1..100")
    if args.migrate_only and args.overwrite:
        parser.error("--migrate-only and --overwrite cannot be combined")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.result_root is None and args.monkey not in DEFAULT_RESULT_ROOTS:
        raise ValueError(
            f"No built-in result root for {args.monkey}; pass --result-root"
        )
    result_root = (
        args.result_root.resolve()
        if args.result_root is not None
        else DEFAULT_RESULT_ROOTS[args.monkey].resolve()
    )
    reference_audio_dir = args.reference_audio_dir.resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")
    if not reference_audio_dir.is_dir():
        raise FileNotFoundError(
            f"Reference audio directory does not exist: {reference_audio_dir}"
        )
    inferred_monkey = infer_monkey_id(result_root)
    if inferred_monkey != args.monkey:
        raise ValueError(
            f"--monkey={args.monkey} conflicts with result root monkey "
            f"{inferred_monkey}: {result_root}"
        )

    estimators = _estimator_arguments(args.estimators)
    conditions = tuple(dict.fromkeys(args.conditions))
    configs = backend_configs()
    for backend in estimators:
        if "device" in configs[backend]:
            configs[backend]["device"] = args.device

    manifests = {
        condition: build_manifest(
            ANALYSIS_DIR,
            result_root,
            reference_audio_dir,
            condition=condition,
            max_folds=args.max_folds,
        )[0]
        for condition in conditions
    }
    jobs = [(backend, condition) for backend in estimators for condition in conditions]
    print(f"Monkey: {args.monkey}")
    print(f"Result root: {result_root}")
    print(f"Reference audio: {reference_audio_dir}")
    print(f"Estimators: {', '.join(estimators)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Folds per job: {len(next(iter(manifests.values())))}")
    print(f"Jobs: {len(jobs)}")

    failures = []
    migration_misses = 0
    for job_index, (backend, condition) in enumerate(jobs, start=1):
        pairs = manifests[condition]
        canonical_path = canonical_f0_cache_path(
            result_root,
            backend,
            condition,
            [pair["key"] for pair in pairs],
            monkey_id=args.monkey,
        )
        prefix = f"[{job_index:02d}/{len(jobs):02d}] {backend} / {condition}"
        if args.dry_run:
            state = "exists" if canonical_path.is_file() else "missing"
            print(f"{prefix}: {state} -> {canonical_path}")
            continue

        try:
            if not args.overwrite:
                _, migrated_path, migration_status = migrate_legacy_f0_cache(
                    pairs,
                    result_root,
                    backend,
                    configs[backend],
                    condition=condition,
                )
                if migration_status in {"cache_hit", "migrated_legacy"}:
                    print(f"{prefix}: {migration_status} -> {migrated_path}")
                    continue
            if args.migrate_only:
                print(f"{prefix}: no exact legacy cache -> {canonical_path}")
                migration_misses += 1
                continue

            _, cache_path, status = load_or_extract_tracks(
                pairs,
                result_root,
                backend,
                configs[backend],
                condition=condition,
                allow_extraction=True,
                force_recompute=args.overwrite,
            )
            print(f"{prefix}: {status} -> {cache_path}")
        except Exception as error:  # Continue independent jobs, fail at the end.
            failures.append((backend, condition, error))
            print(f"{prefix}: FAILED: {error}", file=sys.stderr)

    if failures:
        print("\nFailed jobs:", file=sys.stderr)
        for backend, condition, error in failures:
            print(f"- {backend} / {condition}: {error}", file=sys.stderr)
        return 1
    if args.migrate_only:
        print(
            "\nMigration finished without estimator inference; "
            f"{migration_misses} job(s) had no canonical or exact legacy cache."
        )
    else:
        print("\nAll requested cache jobs completed or were already available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
