import argparse
import copy
import gc
import json
import os
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
"""
使用案例：
cd /home/zhj/projs/Neu2Sound/adaptation
conda activate marmoset

python main.py     --gpu 0     --s-start 1     --s-end 5     --m-start 1     --m-end 10

python main.py     --gpu 1     --s-start 6     --s-end 10     --m-start 1     --m-end 10

# Table II 小规模 benchmark（示例：两个 worker 各负责四个实验）
python main.py --gpu 0 --models linear mlp gru tcn --s-start 1 --s-end 1 --m-start 1 --m-end 10
python main.py --gpu 1 --models transformer vanilla vanilla_wo_dilation vanilla_wo_attention --s-start 1 --s-end 1 --m-start 1 --m-end 10

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run independent same-types-out (S, M) folds on one GPU."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU index. The selected card is exposed as cuda:0.",
    )
    parser.add_argument("--s-start", type=int, default=1)
    parser.add_argument("--s-end", type=int, default=10)
    parser.add_argument("--m-start", type=int, default=1)
    parser.add_argument("--m-end", type=int, default=10)
    parser.add_argument(
        "--diagonal-only",
        action="store_true",
        help="Run only matched folds (S1,M1), ..., (S10,M10).",
    )
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Override config diagonal_only and run the Cartesian product of S and M ranges.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model names handled sequentially by this GPU worker. When provided, "
            "each model is isolated under save_folder/<model_name>/; otherwise "
            "the single model in config.json uses save_folder directly."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the fold list and output directories.",
    )
    parser.add_argument(
        "--flat-single-fold",
        action="store_true",
        help=(
            "Save one requested fold directly in save_folder instead of "
            "save_folder/Sx_My. Requires exactly one S and one M index."
        ),
    )
    return parser.parse_args()


def validate_range(name, start, end):
    if not 1 <= start <= end <= 10:
        raise ValueError(
            f"{name} range must satisfy 1 <= start <= end <= 10, "
            f"got {start}..{end}"
        )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def write_json_once(path, payload):
    """并行 worker 安全地写入一份不可变的 sweep 配置。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    try:
        os.link(temporary_path, path)
    except FileExistsError:
        with path.open(encoding="utf-8") as f:
            existing_payload = json.load(f)
        if existing_payload != payload:
            raise ValueError(
                f"Existing sweep config differs from the requested config: {path}"
            )
    finally:
        temporary_path.unlink(missing_ok=True)


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_run_config(base_config, sweep_root, s_idx, m_idx, *, flat_single_fold=False):
    run_config = copy.deepcopy(base_config)
    run_dir = sweep_root if flat_single_fold else sweep_root / f"S{s_idx}_M{m_idx}"
    run_config["data_loader"]["S_idx"] = s_idx
    run_config["data_loader"]["M_idx"] = m_idx
    run_config["save_load"]["save_folder"] = str(run_dir.resolve())
    return run_config, run_dir


def main():
    args = parse_args()
    validate_range("S", args.s_start, args.s_end)
    validate_range("M", args.m_start, args.m_end)
    if args.flat_single_fold and (
        args.s_start != args.s_end or args.m_start != args.m_end
    ):
        raise ValueError("--flat-single-fold requires exactly one S index and one M index")

    # 必须在导入 torch、Train 和 Music2Latent 前限制可见设备。
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from music2latent import EncoderDecoder

    from Recons import recons_main
    from Train import load_shared_datasets, set_random_seed, train_main
    from model_ablation import (
        ARCHITECTURE_MODELS,
        COMPONENT_VARIANTS,
        MODEL_REGISTRY,
        normalize_model_name,
    )
    from utils.music2latent_mel import FrozenMusic2LatentMelDecoder

    with Path(args.config).open(encoding="utf-8") as f:
        base_config = json.load(f)

    runtime_config = base_config.get("runtime", {})
    cpu_threads = int(runtime_config.get("cpu_threads", 1))
    cpu_interop_threads = int(runtime_config.get("cpu_interop_threads", 1))
    if cpu_threads < 1 or cpu_interop_threads < 1:
        raise ValueError("runtime CPU thread counts must be positive")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_interop_threads)
    print(
        f"Worker CPU thread limits: intra_op={cpu_threads}, "
        f"interop={cpu_interop_threads}"
    )

    benchmark_mode = args.models is not None
    requested_models = args.models or [base_config["model"]["model_name"]]
    selected_models = [normalize_model_name(name) for name in requested_models]
    if len(selected_models) != len(set(selected_models)):
        raise ValueError(f"Duplicate model names after normalization: {selected_models}")
    benchmark_models = set((*ARCHITECTURE_MODELS, *COMPONENT_VARIANTS))
    unknown_models = [
        name for name in selected_models
        if name not in MODEL_REGISTRY
    ]
    if unknown_models:
        raise ValueError(f"Unknown registered models: {unknown_models}")
    if benchmark_mode:
        unsupported_benchmark_models = [
            name for name in selected_models if name not in benchmark_models
        ]
        if unsupported_benchmark_models:
            raise ValueError(
                f"Unsupported benchmark models {unsupported_benchmark_models}; "
                f"choose from {sorted(benchmark_models)}"
            )

    output_root = Path(base_config["save_load"]["save_folder"])
    if args.diagonal_only and args.all_combinations:
        raise ValueError("--diagonal-only and --all-combinations cannot be used together")
    diagonal_only = (
        False
        if args.all_combinations
        else args.diagonal_only or bool(base_config.get("benchmark", {}).get("diagonal_only", False))
    )
    if diagonal_only:
        if (args.s_start, args.s_end) != (args.m_start, args.m_end):
            raise ValueError(
                "Diagonal folds require identical S and M ranges, got "
                f"S={args.s_start}..{args.s_end}, M={args.m_start}..{args.m_end}"
            )
        folds = [(index, index) for index in range(args.s_start, args.s_end + 1)]
    else:
        folds = [
            (s_idx, m_idx)
            for s_idx in range(args.s_start, args.s_end + 1)
            for m_idx in range(args.m_start, args.m_end + 1)
        ]
    model_roots = {
        model_name: output_root / model_name if benchmark_mode else output_root
        for model_name in selected_models
    }
    total_tasks = len(selected_models) * len(folds)

    print(
        f"Worker models={selected_models}, S={args.s_start}..{args.s_end}, "
        f"M={args.m_start}..{args.m_end}, total={total_tasks}, "
        f"physical_gpu={args.gpu}, benchmark_mode={benchmark_mode}, "
        f"diagonal_only={diagonal_only}"
    )
    for model_name in selected_models:
        for s_idx, m_idx in folds:
            preview_dir = (
                model_roots[model_name]
                if args.flat_single_fold
                else model_roots[model_name] / f"S{s_idx}_M{m_idx}"
            )
            print(
                f"  {model_name}/S{s_idx}_M{m_idx} -> "
                f"{preview_dir}"
            )
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    if benchmark_mode:
        benchmark_config = base_config.get("benchmark", {})
        planned_models = [
            normalize_model_name(name)
            for name in benchmark_config.get("models", sorted(benchmark_models))
        ]
        benchmark_manifest = {
            "experiment_type": "table2_model_ablation",
            "models": planned_models,
            "fold_scope": {
                # This is the benchmark's declared scope, not the current
                # worker's subset.  A supplemental non-diagonal worker must
                # therefore not invalidate the shared benchmark manifest.
                "diagonal_only": bool(benchmark_config.get("diagonal_only", False)),
                "s_start": int(benchmark_config.get("s_start", args.s_start)),
                "s_end": int(benchmark_config.get("s_end", args.s_end)),
                "m_start": int(benchmark_config.get("m_start", args.m_start)),
                "m_end": int(benchmark_config.get("m_end", args.m_end)),
            },
            "data": base_config["data"],
            "seed": base_config["seed"],
            "whether_seed": base_config["whether_seed"],
            "base_config": base_config,
        }
        # A Table II result root can contain independent model groups produced
        # at different times. Preserve an existing group manifest instead of
        # rejecting a new, non-overlapping group such as TimeMixer + Mamba.
        # The group-specific manifest remains immutable on resume.
        try:
            write_json_once(output_root / "benchmark_manifest.json", benchmark_manifest)
        except ValueError:
            model_group = "-".join(selected_models)
            group_manifest_path = (
                output_root / f"benchmark_manifest_{model_group}.json"
            )
            write_json_once(group_manifest_path, benchmark_manifest)
            print(
                "Existing benchmark manifest belongs to a different model "
                f"group; using {group_manifest_path.name} for this sweep."
            )

    worker_manifest = {
        "gpu": args.gpu,
        "models": selected_models,
        "diagonal_only": diagonal_only,
        "s_start": args.s_start,
        "s_end": args.s_end,
        "m_start": args.m_start,
        "m_end": args.m_end,
        "fold_count_per_model": len(folds),
        "task_count": total_tasks,
        "folds": [f"S{s}_M{m}" for s, m in folds],
    }
    model_label = "-".join(selected_models)
    worker_file_stem = (
        f"worker_gpu{args.gpu}_models-{model_label}_"
        f"S{args.s_start}-{args.s_end}_M{args.m_start}-{args.m_end}"
    )
    write_json(
        output_root / f"{worker_file_stem}.json",
        worker_manifest,
    )
    worker_status_path = output_root / f"{worker_file_stem}.status.json"
    worker_status = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": timestamp(),
        "gpu": args.gpu,
        "models": selected_models,
        "folds": worker_manifest["folds"],
        "current_task": None,
    }
    write_json(worker_status_path, worker_status)

    active_run = {"run_dir": None, "status": None, "task_label": None}
    worker_interrupted = {"value": False}

    def handle_termination(signum, _frame):
        signal_name = signal.Signals(signum).name
        worker_interrupted["value"] = True
        interruption = {
            "status": "interrupted",
            "signal": signal_name,
            "received_at": timestamp(),
            "pid": os.getpid(),
            "task_label": active_run["task_label"],
        }
        if active_run["status"] is not None:
            write_json(
                active_run["run_dir"] / "INTERRUPTED.json",
                {**active_run["status"], **interruption},
            )
            write_json(
                active_run["run_dir"] / "status.json",
                {**active_run["status"], **interruption},
            )
        write_json(
            worker_status_path,
            {**worker_status, **interruption, "current_task": active_run["task_label"]},
        )
        print(
            f"Received {signal_name}; interruption metadata written for "
            f"{active_run['task_label']}.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(128 + signum)

    for termination_signal in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(termination_signal, handle_termination)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Music2Latent sweep")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each sweep worker must see exactly one GPU. Use --gpu N or set "
            "CUDA_VISIBLE_DEVICES to one device before launching."
        )
    device = torch.device("cuda:0")
    print(f"Worker CUDA device: {torch.cuda.get_device_name(device)} ({device})")

    # 完整数据和 Music2Latent 在每个 GPU worker 中都只初始化一次。
    datasets = load_shared_datasets(base_config)
    set_random_seed(base_config)
    spectral_config = base_config["loss"]["spectral"]
    # 关闭 spectral loss 时，训练阶段不实例化 Music2Latent decoder。重建仍
    # 需要它，因此在首个 fold 训练完成后才懒加载一次并在同一 worker 内复用。
    encoder_decoder = None
    mel_decoder = None
    if spectral_config["enabled"]:
        encoder_decoder = EncoderDecoder(
            load_path_inference=spectral_config.get("checkpoint_path"),
            device=device,
        )
        encoder_decoder.gen.eval()
        encoder_decoder.gen.requires_grad_(False)
        mel_decoder = FrozenMusic2LatentMelDecoder(
            device=device,
            sample_rate=spectral_config["sample_rate"],
            n_fft=spectral_config["n_fft"],
            n_mels=spectral_config["n_mels"],
            f_min=spectral_config["f_min"],
            f_max=spectral_config["f_max"],
            latent_scale=spectral_config["latent_scale"],
            denoising_steps=spectral_config["denoising_steps"],
            noise_seed=spectral_config["noise_seed"],
            use_amp=spectral_config["use_amp"],
            amp_dtype=spectral_config["amp_dtype"],
            checkpoint_path=spectral_config.get("checkpoint_path"),
            encoder_decoder=encoder_decoder,
        ).to(device)

    task_index = 0
    failed_tasks = []
    for model_name in selected_models:
        model_config = copy.deepcopy(base_config)
        model_config["model"]["model_name"] = model_name
        model_root = model_roots[model_name]
        model_config["save_load"]["save_folder"] = str(model_root.resolve())
        write_json_once(model_root / "sweep_config.json", model_config)

        for s_idx, m_idx in folds:
            task_index += 1
            run_config, run_dir = build_run_config(
                model_config,
                model_root,
                s_idx,
                m_idx,
                flat_single_fold=args.flat_single_fold,
            )
            completed_path = run_dir / "COMPLETED"
            failed_path = run_dir / "FAILED.json"
            interrupted_path = run_dir / "INTERRUPTED.json"
            task_label = f"{model_name}/S{s_idx}_M{m_idx}"
            if completed_path.exists():
                print(f"[{task_index}/{total_tasks}] Skip completed {task_label}")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            if failed_path.exists():
                failed_path.unlink()
            if interrupted_path.exists():
                interrupted_path.unlink()
            write_json(run_dir / "config.json", run_config)
            running_status = {
                "status": "running",
                "model_name": model_name,
                "S_idx": s_idx,
                "M_idx": m_idx,
                "pid": os.getpid(),
                "started_at": timestamp(),
            }
            write_json(run_dir / "status.json", running_status)
            active_run.update(
                {"run_dir": run_dir, "status": running_status, "task_label": task_label}
            )
            worker_status["current_task"] = task_label
            write_json(worker_status_path, worker_status)

            print(f"[{timestamp()}] [{task_index}/{total_tasks}] Start {task_label}")
            try:
                training_summary = train_main(
                    config=run_config,
                    datasets=datasets,
                    mel_decoder=mel_decoder,
                    device=device,
                )
                if encoder_decoder is None:
                    print(
                        "Spectral loss is disabled; initializing Music2Latent "
                        "decoder for reconstruction only."
                    )
                    encoder_decoder = EncoderDecoder(
                        load_path_inference=spectral_config.get("checkpoint_path"),
                        device=device,
                    )
                    encoder_decoder.gen.eval()
                    encoder_decoder.gen.requires_grad_(False)
                output_files = recons_main(
                    config=run_config,
                    datasets=datasets,
                    encoder_decoder=encoder_decoder,
                    device=device,
                )
                summary = {
                    **training_summary,
                    "status": "completed",
                    "reconstruction_files": output_files,
                }
                write_json(run_dir / "summary.json", summary)
                write_json(
                    run_dir / "status.json",
                    {**running_status, "status": "completed", "completed_at": timestamp()},
                )
                completed_path.touch()
                print(f"[{timestamp()}] [{task_index}/{total_tasks}] Completed {task_label}")
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "model_name": model_name,
                    "S_idx": s_idx,
                    "M_idx": m_idx,
                    "failed_at": timestamp(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                traceback.print_exc()
                write_json(failed_path, failure)
                write_json(run_dir / "status.json", failure)
                failed_tasks.append(failure)
                print(
                    f"[{timestamp()}] [{task_index}/{total_tasks}] Failed {task_label}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                active_run.update({"run_dir": None, "status": None, "task_label": None})
                if not worker_interrupted["value"]:
                    worker_status["current_task"] = None
                    write_json(worker_status_path, worker_status)

    if failed_tasks:
        failed_labels = [
            f"{item['model_name']}/S{item['S_idx']}_M{item['M_idx']}"
            for item in failed_tasks
        ]
        write_json(
            worker_status_path,
            {
                **worker_status,
                "status": "failed",
                "failed_at": timestamp(),
                "failed_tasks": failed_labels,
            },
        )
        raise RuntimeError(
            f"{len(failed_tasks)} benchmark task(s) failed: {failed_labels}"
        )

    write_json(
        worker_status_path,
        {**worker_status, "status": "completed", "completed_at": timestamp()},
    )


if __name__ == "__main__":
    main()
