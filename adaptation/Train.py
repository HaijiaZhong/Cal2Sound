import os
import shutil
import random
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, random_split,DataLoader
import json
from utils.loss_function import *
from utils.music2latent_mel import FrozenMusic2LatentMelDecoder
from utils.split_ids import get_target_sm_indices, sametypes_out_ids, target_condition_ids
from adamp import AdamP, SGDP
from model_ablation import (
    build_model,
    get_model_identity,
    get_model_source_path,
    get_selected_model_config,
)

# 读取项目文件配置
# config = json.load(open("config.json"))

# 自定义数据集类
class CustomDataset(Dataset):
    """通过样本索引复用完整数组，避免每个 fold 复制整份训练数据。"""

    def __init__(self, input_data, label_data, mel_data, sample_ids):
        self.input_data = input_data
        self.label_data = label_data
        self.mel_data = mel_data
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)

        dataset_sizes = {len(self.input_data), len(self.label_data)}
        if self.mel_data is not None:
            dataset_sizes.add(len(self.mel_data))
        if len(dataset_sizes) != 1:
            raise ValueError(
                "Neural, latent and Mel datasets must have the same length, "
                f"got {sorted(dataset_sizes)}"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = int(self.sample_ids[idx])
        input_sample = torch.as_tensor(
            self.input_data[sample_id], dtype=torch.float32
        )
        label_sample = torch.as_tensor(
            self.label_data[sample_id], dtype=torch.float32
        )
        if self.mel_data is None:
            # spectral loss 未启用时避免 collate 无用的 [128, 416] Mel 数据。
            return input_sample, label_sample

        mel_sample = torch.as_tensor(self.mel_data[sample_id], dtype=torch.float32)
        return input_sample, label_sample, mel_sample, sample_id


def load_shared_datasets(config):
    """一次加载当前 loss 配置实际需要的数据，供多个 fold 复用。"""
    data_config = config["data"]
    data_path = Path(data_config["data_path"])
    datasets = {
        "neu": np.load(
            data_path / data_config["neu_dataset"], allow_pickle=False
        ),
        "latent": np.load(
            data_path / data_config["latent_dataset"], allow_pickle=False
        ),
    }
    neu_dataset = datasets["neu"]
    latent_dataset = datasets["latent"]
    if neu_dataset.shape[0] != latent_dataset.shape[0]:
        raise ValueError(
            "Neural and latent datasets must have the same number of trials: "
            f"neu={neu_dataset.shape[0]}, latent={latent_dataset.shape[0]}, "
        )
    spectral_config = config["loss"]["spectral"]
    if not spectral_config["enabled"]:
        print("Spectral loss disabled: Mel dataset is not loaded for training.")
        return datasets

    mel_dataset = np.load(data_path / data_config["mel_dataset"], allow_pickle=False)
    datasets["mel"] = mel_dataset
    if neu_dataset.shape[0] != mel_dataset.shape[0]:
        raise ValueError(
            "Neural, latent and Mel datasets must have the same number of trials: "
            f"neu={neu_dataset.shape[0]}, latent={latent_dataset.shape[0]}, "
            f"mel={mel_dataset.shape[0]}"
        )
    expected_mel_shape = (
        spectral_config["n_mels"],
        config["model"]["latent_T_dim"] * 8,
    )
    if mel_dataset.shape[1:] != expected_mel_shape:
        raise ValueError(
            "Unexpected Mel dataset shape: expected "
            f"[N, {expected_mel_shape[0]}, {expected_mel_shape[1]}], "
            f"got {mel_dataset.shape}"
        )
    if not np.isfinite(mel_dataset).all() or np.any(mel_dataset < 0):
        raise ValueError("Mel dataset must contain only finite, non-negative values")
    return datasets

class LatentLossFunction(nn.Module):
    """
    总损失 = lambda_huber * Huber
           + lambda_time  * Time Dynamic Loss
           + lambda_std   * Frame-wise Std Loss

    输入:
        pred   : [B, C, T]
        target : [B, C, T]

        B: batch size
        C: 通道/特征维度
        T: 时间维度

    返回:
        total_loss, loss_huber, loss_time, loss_std
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        lambda_huber: float = 1.0,
        lambda_time: float = 0.5,
        lambda_std: float = 0.1,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.lambda_huber = lambda_huber
        self.lambda_time = lambda_time
        self.lambda_std = lambda_std

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred   : [B, C, T]，当前任务中为 [B, 64, 52]
        target : [B, C, T]，当前任务中为 [B, 64, 52]
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if pred.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, C, T], but got {pred.dim()}D tensor")

        if pred.size(-1) < 2:
            raise ValueError(
                f"Time dimension T must be at least 2, but got shape {pred.shape}"
            )

        # -------------------------------------------------
        # 1) 主损失：Huber loss
        # -------------------------------------------------
        loss_huber = F.huber_loss(
            pred,
            target,
            delta=self.huber_delta,
            reduction="mean"
        )

        # -------------------------------------------------
        # 2) 时间动态损失：比较相邻时间帧的一阶差分
        #    delta_z[t] = z[t+1] - z[t]
        # -------------------------------------------------
        pred_delta = pred[:, :, 1:] - pred[:, :, :-1]       # [B, C, T-1]
        target_delta = target[:, :, 1:] - target[:, :, :-1] # [B, C, T-1]

        loss_time = F.l1_loss(
            pred_delta,
            target_delta,
            reduction="mean"
        )

        # -------------------------------------------------
        # 3) 逐帧方差损失：
        #    对每个时间帧，在64个通道上算 std
        # -------------------------------------------------
        pred_std = torch.std(pred, dim=1, unbiased=False)       # [B, T]
        target_std = torch.std(target, dim=1, unbiased=False)   # [B, T]

        loss_std = F.l1_loss(
            pred_std,
            target_std,
            reduction="mean"
        )

        # -------------------------------------------------
        # 4) 总损失
        # -------------------------------------------------
        total_loss = (
            self.lambda_huber * loss_huber
            + self.lambda_time * loss_time
            + self.lambda_std * loss_std
        )

        return total_loss, loss_huber, loss_time, loss_std


class MetricTracker:
    """在设备端累计按样本加权的标量指标，并在 epoch 结束时统一取值。"""

    def __init__(self):
        self.sums = {}
        self.counts = {}

    @torch.no_grad()
    def update(self, metrics, weight):
        if weight <= 0:
            raise ValueError(f"Metric weight must be positive, got {weight}")
        for name, value in metrics.items():
            if value.ndim != 0:
                raise ValueError(
                    f"Metric {name!r} must be a scalar tensor, got {value.shape}"
                )
            weighted_value = value.detach() * weight
            if name not in self.sums:
                self.sums[name] = weighted_value
                self.counts[name] = weight
            else:
                self.sums[name] += weighted_value
                self.counts[name] += weight

    def compute(self):
        return {
            name: (value / self.counts[name]).cpu().item()
            for name, value in self.sums.items()
        }


@torch.no_grad()
def update_best_state_on_cpu(best_state, model):
    """将当前模型状态复制到可复用的 CPU 缓冲区，不执行磁盘写入。"""
    current_state = model.state_dict()
    if best_state is None:
        return {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in current_state.items()
        }

    if best_state.keys() != current_state.keys():
        raise RuntimeError("Model state keys changed during training")
    for name, value in current_state.items():
        best_state[name].copy_(value.detach(), non_blocking=False)
    return best_state


def log_train_val_losses(
    train_writer,
    val_writer,
    train_metrics,
    val_metrics,
    epoch,
):
    """向两个 run 写入相同 loss tags，使 train/val 显示在同一张图。"""
    logged_loss_names = ("total", "huber", "time", "std", "spec")
    for name in logged_loss_names:
        tag = f"loss/{name}"
        train_writer.add_scalar(tag, train_metrics[name], epoch)
        val_writer.add_scalar(tag, val_metrics[name], epoch)


def save_training_source_snapshot(save_folder, model_config):
    """保存本次训练使用的核心源码，便于追踪实验结果。"""
    adaptation_dir = Path(__file__).resolve().parent
    source_paths = [
        adaptation_dir / "main.py",
        adaptation_dir / "model_ablation.py",
        get_model_source_path(model_config),
        Path(__file__).resolve(),
        adaptation_dir / "Recons.py",
        adaptation_dir / "utils" / "music2latent_mel.py",
        adaptation_dir / "utils" / "split_ids.py",
    ]
    save_dir = Path(save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for source_path in source_paths:
        target_path = save_dir / source_path.name
        if target_path.exists() and target_path.read_bytes() != source_path.read_bytes():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = save_dir / f"{source_path.stem}_{timestamp}{source_path.suffix}"
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"训练源码已保存: {target_path}")
        saved_paths.append(target_path)
    return saved_paths


def set_random_seed(config):
    """为每个 fold 重置同一随机状态，保证初始化和抽样可复现。"""
    if not config.get("whether_seed", False):
        return
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_main(config, datasets=None, mel_decoder=None, device=None):
    """训练一个独立的 (S, M) fold，并返回该 fold 的训练摘要。"""

    set_random_seed(config)

    ## 定义数据集：
    # 我这里需要有两种划分数据集的方式，一种是只有S1_M1进行测试，其他所有的用于训练。
    # 第二种是，S1_Mj和Si_M1都不参与训练。
    
    if datasets is None:
        datasets = load_shared_datasets(config)
    neu_dataset = datasets["neu"]       # [700, 90, 4667]
    latent_dataset = datasets["latent"] # [700, 64, 52]
    spectral_config = config["loss"]["spectral"]
    spectral_loss_enabled = spectral_config["enabled"]
    mel_dataset = datasets.get("mel")
    if spectral_loss_enabled and mel_dataset is None:
        raise ValueError("Spectral loss is enabled but the Mel dataset is unavailable")

    s_idx, m_idx = get_target_sm_indices(config)

    # 划分数据集的索引逻辑
    if config["data"]["data_splite_types"]=="sametypes_out":
        excluded_ids = sametypes_out_ids(s_idx, m_idx, neu_dataset.shape[0])
    else:
        excluded_ids = target_condition_ids(s_idx, m_idx, neu_dataset.shape[0])

    val_ids = target_condition_ids(s_idx, m_idx, neu_dataset.shape[0])

    # 确定idx后，划分训练集和验证集。Dataset 只保存索引，不复制完整数组。
    all_ids = np.arange(neu_dataset.shape[0])
    train_ids = np.delete(all_ids, excluded_ids, axis=0)

    # 打印训练集和验证集的大小
    print(f"训练集大小: {len(train_ids)}")
    print(f"验证集大小: {len(val_ids)}")

    # 由 config.model.model_name / variant 选择架构或受控组件消融。
    model = build_model(config["model"])
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    

    # 每个 worker 只使用一张由 main.py 指定的 GPU，不启用 DataParallel。
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)

    # 是否启用 Mel spectrogram loss。关闭时不会初始化或调用 Music2Latent decoder。
    batch_size = int(config["data_loader"]["batch_size"])
    if batch_size < 1:
        raise ValueError("data_loader.batch_size must be at least 1")

    if spectral_loss_enabled:
        # 保留频谱损失所需的 CPU Mel DataLoader 路径。
        train_dataset = CustomDataset(neu_dataset, latent_dataset, mel_dataset, train_ids)
        val_dataset = CustomDataset(neu_dataset, latent_dataset, mel_dataset, val_ids)
        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(int(config["seed"]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=config["data_loader"]["shuffle"],
            generator=data_loader_generator,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        # 无频谱损失时，neural / latent 在每个 fold 初始化时仅传输一次到 GPU；
        # 后续 epoch 的 shuffle、索引和 batch 切分全部在 GPU 完成。
        train_inputs_gpu = torch.as_tensor(
            np.ascontiguousarray(neu_dataset[train_ids]),
            dtype=torch.float32,
            device=device,
        )
        train_latents_gpu = torch.as_tensor(
            np.ascontiguousarray(latent_dataset[train_ids]),
            dtype=torch.float32,
            device=device,
        )
        val_inputs_gpu = torch.as_tensor(
            np.ascontiguousarray(neu_dataset[val_ids]),
            dtype=torch.float32,
            device=device,
        )
        val_latents_gpu = torch.as_tensor(
            np.ascontiguousarray(latent_dataset[val_ids]),
            dtype=torch.float32,
            device=device,
        )
        gpu_shuffle_generator = torch.Generator(device=device)
        gpu_shuffle_generator.manual_seed(int(config["seed"]))
        print(
            "Spectral loss disabled: neural and latent fold data are resident on "
            f"{device}; batch_size={batch_size}."
        )

    # 提前检查频谱损失的关键参数，避免开始训练后才因非法配置中断。
    # batch_size 表示“每次送入 Music2Latent decoder 的样本数”，不是主 DataLoader
    # 的 batch size；lambda 是 spec loss 在总损失中的权重。
    if spectral_config["batch_size"] < 1:
        raise ValueError("loss.spectral.batch_size must be at least 1")
    if spectral_config["lambda"] < 0:
        raise ValueError("loss.spectral.lambda must be non-negative")
    if spectral_config["smooth_l1_beta"] <= 0:
        raise ValueError("loss.spectral.smooth_l1_beta must be positive")
    model_latent_scale = float(config["model"].get("sigma_rescale", 0.06))
    decoder_latent_scale = float(spectral_config["latent_scale"])
    if model_latent_scale != decoder_latent_scale:
        raise ValueError(
            "model.sigma_rescale and loss.spectral.latent_scale describe the "
            "same Music2Latent normalization and must match, got "
            f"{model_latent_scale} and {decoder_latent_scale}"
        )

    if spectral_loss_enabled and mel_decoder is None:
        # 这里只初始化一次 Decoder。FrozenMusic2LatentMelDecoder 内部会：
        # 1. 加载 Music2Latent decoder 到 device；
        # 2. 将其参数 requires_grad 设为 False，并始终保持 eval 模式；
        # 3. 保留 decoder 对输入 latent 的可微计算图；
        # 4. 只输出压缩 magnitude Mel spectrogram，不重建 waveform。
        # 因此训练过程中 Decoder 参数不会更新，但 spec loss 仍可通过 Decoder
        # 回传到 adap_latent，并进一步更新 adaptor model。
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
        )

    if spectral_loss_enabled:

        # Decoder 和 Mel filterbank 在整个训练期间常驻同一设备，不在 batch
        # 循环中重复加载，也不在 CPU/GPU 之间往返搬运。
        mel_decoder.to(device)

        # predicted Mel 与预处理 target Mel 形状均为 [B_spec, 128, 416]。
        mel_loss_function = nn.SmoothL1Loss(
            beta=spectral_config["smooth_l1_beta"],
            reduction="mean",
        )

        # 该生成器只负责从主训练 batch 中抽取频谱子 batch。
        # 它与 Decoder 内按 original sample_id 生成固定噪声的 seed 相互独立。
        spec_subset_generator = torch.Generator()
        spec_subset_generator.manual_seed(spectral_config["subset_seed"])
        print(
            "Using shared frozen Music2Latent decoder on "
            f"{device}; spec_batch_size={spectral_config['batch_size']}"
        )

    ## 定义损失函数
    # 定义CKA loss
    if config["loss"]["type"] == "huber":
        loss_function = nn.HuberLoss(delta=config["loss"]["huber"]["delta"],
                                    reduction=config["loss"]["huber"]["reduction"])
    elif config["loss"]["type"] == "cross":
        loss_function = nn.CrossEntropyLoss(reduction=config["loss"]["cross_entropy"]["reduction"])
    elif config["loss"]["type"] == "PseudoHuber":
        loss_function = PseudoHuberLoss(c=config["loss"]["PseudoHuber"]["c"]) 
    elif config["loss"]["type"] == "mse":
        loss_function = nn.MSELoss(reduction='mean')
    elif config["loss"]["type"] == "complex":
        loss_function = LatentLossFunction(
            huber_delta=config["loss"]["complex"]["huber_delta"],
            lambda_huber=config["loss"]["complex"]["lambda_huber"],
            lambda_time=config["loss"]["complex"]["lambda_time"],
            lambda_std=config["loss"]["complex"]["lambda_std"])
        

    ## 定义优化器：
    if config["optimizer"]["type"] == "sgd": 
        optimizer = optim.SGD(params=model.parameters(),
                                lr=config["optimizer"]["sgd"]["lr"],
                                momentum=config["optimizer"]["sgd"]["momentum"],
                                nesterov=True,
                                weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"],
                    config["optimizer"]["adam"]["beta2"]),
            eps=config["optimizer"]["adam"]["eps"],
            weight_decay=config["optimizer"]["adam"]["weight_decay"])
    elif config["optimizer"]["type"] == "radam":
        optimizer = optim.RAdam(
            params=model.parameters(),
            lr=config["optimizer"]["radam"]["lr"],
            weight_decay=config["optimizer"]["radam"]["weight_decay"])
    elif config["optimizer"]["type"] == "sgdp":
        optimizer = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimizer = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"],
                    config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )

    ## 定义学习率调整策略
    if config["scheduler"]["type"] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config["scheduler"]["step"]["step_size"],
                                                gamma=config["scheduler"]["step"]["gamma"])
    elif config["scheduler"]["type"] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=config["scheduler"]["exp"]["gamma"])
    elif config["scheduler"]["type"] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=int(config["scheduler"]["cos"]["T_max_of_epoch"]*config["train"]["epochs"]),
                                                        eta_min=config["scheduler"]["cos"]["eta_min"])
    elif config["scheduler"]["type"] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode=config["scheduler"]["plateau"]["mode"],
                                                            factor=config["scheduler"]["plateau"]["factor"], 
                                                            verbose=config["scheduler"]["plateau"]["verbose"])
    else:
        scheduler = None

    # 处理好了所有的数据，开始进行训练过程
    res_path = Path(config["save_load"]["save_folder"])
    res_path.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "S_idx": s_idx,
        "M_idx": m_idx,
        "train_ids": train_ids.tolist(),
        "val_ids": list(val_ids),
        "test_ids": list(val_ids),
        "excluded_ids": list(excluded_ids),
    }
    with (res_path / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)
    
    # 保存
    save_training_source_snapshot(res_path, config["model"])

    # TensorBoard 将 loss_train 和 loss_val 识别为两个独立 run。两个 run
    # 写入相同的 loss tags，因此同一成分的 train/val 会在一张图中显示为
    # 两条不同颜色的曲线。
    tensorboard_dir = Path(res_path) / config["logging"]["tensorboard_subdir"]
    train_writer = SummaryWriter(str(tensorboard_dir / "loss_train"))
    val_writer = SummaryWriter(str(tensorboard_dir / "loss_val"))
    model_path = res_path / config["save_load"]["model_path"]

    best_val_loss=float("inf")
    best_epoch_point=-1
    best_val_metrics = None
    best_model_state = None
    training_start_time = time.perf_counter()
    last_epoch = -1
    # 验证没有提升的轮数
    val_no_impove=0
    for epoch in tqdm(range(config["train"]["epochs"])):
        last_epoch = epoch
        # 记录完整 epoch（训练 + 验证）的耗时，以及该 epoch 训练时实际使用
        # 的学习率。scheduler.step() 在训练后执行，因此需提前保存当前值。
        epoch_start_time = time.perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]

        # 开始进入训练
        model.train()
        train_tracker = MetricTracker()
        if spectral_loss_enabled:
            train_batches = train_loader
        else:
            if config["data_loader"]["shuffle"]:
                batch_order = torch.randperm(
                    train_inputs_gpu.size(0),
                    generator=gpu_shuffle_generator,
                    device=device,
                )
            else:
                batch_order = torch.arange(train_inputs_gpu.size(0), device=device)
            train_batches = (
                (
                    train_inputs_gpu.index_select(0, batch_order[start:stop]),
                    train_latents_gpu.index_select(0, batch_order[start:stop]),
                )
                for start in range(0, train_inputs_gpu.size(0), batch_size)
                for stop in (min(start + batch_size, train_inputs_gpu.size(0)),)
            )

        for batch in train_batches:
            if spectral_loss_enabled:
                inputs, label_latents, target_mels, sample_ids = batch
                inputs = inputs.to(device, non_blocking=True)
                label_latents = label_latents.to(device, non_blocking=True)
            else:
                inputs, label_latents = batch

            adap_latent = model(inputs)

            optimizer.zero_grad(set_to_none=True)
            latent_loss, loss_huber, loss_time, loss_std = loss_function(
                adap_latent, label_latents
            )

            if spectral_loss_enabled:
                # 主训练 batch 当前为128。由于可微 Decoder 需要保存反向传播所需的激活，只从中抽取 batch_size 个样本计算 spec loss。
                # 如果最后一个主 batch 小于配置值，则使用该 batch 的实际大小。
                spec_batch_size = min(
                    spectral_config["batch_size"], adap_latent.size(0)
                )

                # 在 CPU 上生成本 batch 的随机位置。spec_subset_generator 使用固定
                # seed，因此在数据顺序相同的情况下，重新训练可复现抽样结果。
                spec_positions = torch.randperm(
                    adap_latent.size(0), generator=spec_subset_generator
                )[:spec_batch_size]

                # predicted latent 位于 GPU，因此索引也复制到 GPU；sample_ids 保留
                # 在 CPU，Decoder 用它为每个原始样本确定固定的扩散噪声。
                spec_positions_device = spec_positions.to(device)

                # 这里是一次批量 Decoder 调用：输入形状为
                # [B_spec, 64, 52]，输出形状为 [B_spec, 128, 416]。
                # 不是逐样本调用 Decoder。
                pred_mels = mel_decoder(
                    adap_latent.index_select(0, spec_positions_device),
                    sample_ids.index_select(0, spec_positions),
                )

                # target Mel 已经离线预处理完成。这里只选择与 predicted latent
                # 相同位置的目标并传到 GPU，不会解码 label latent 或重算频谱。
                selected_target_mels = target_mels.index_select(
                    0, spec_positions
                ).to(device, non_blocking=True)

                # 显式检查可以尽早发现样本对应关系或频谱预处理 shape 出错。
                if pred_mels.shape != selected_target_mels.shape:
                    raise ValueError(
                        "Predicted and target Mel shapes do not match: "
                        f"pred={tuple(pred_mels.shape)}, "
                        f"target={tuple(selected_target_mels.shape)}"
                    )

                # 对整个频谱子 batch 的所有 Mel bin 和时间帧取 mean。
                spec_loss = mel_loss_function(pred_mels, selected_target_mels)
            else:
                # 保持后续总损失和 MetricTracker 逻辑统一；该零标量与 latent loss
                # 位于同一设备，不参与任何有效梯度更新。
                spec_loss = latent_loss.new_zeros(())

            loss = latent_loss + spectral_config["lambda"] * spec_loss

            loss.backward()
            optimizer.step()

            train_tracker.update(
                {
                    "latent": latent_loss,
                    "huber": loss_huber,
                    "time": loss_time,
                    "std": loss_std,
                },
                weight=inputs.size(0),
            )
            train_tracker.update(
                {"spec": spec_loss},
                weight=spec_batch_size if spectral_loss_enabled else inputs.size(0),
            )

        train_metrics = train_tracker.compute()
        train_metrics["total"] = (
            train_metrics["latent"]
            + spectral_config["lambda"] * train_metrics["spec"]
        )

        if scheduler is not None:
            scheduler.step()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         cp.add_histogram(f'{name}.grad', param.grad, epoch)
        # 验证循环
        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            if spectral_loss_enabled:
                val_batches = val_loader
            else:
                val_batches = ((val_inputs_gpu, val_latents_gpu),)

            for batch in val_batches:
                if spectral_loss_enabled:
                    inputs, label_latents, target_mels, sample_ids = batch
                    inputs = inputs.to(device, non_blocking=True)
                    label_latents = label_latents.to(device, non_blocking=True)
                else:
                    inputs, label_latents = batch
                adap_latent = model(inputs)
                latent_loss, loss_huber, loss_time, loss_std = loss_function(
                    adap_latent, label_latents
                )

                if spectral_loss_enabled:
                    # 验证阶段不随机抽取子集，而是计算当前验证 batch 中的全部样本。
                    # spec_loss_sum 用于合并多个 Decoder 子 batch 的损失。
                    spec_loss_sum = latent_loss.new_zeros(())

                    # 这里按 spectral.batch_size 对完整验证 batch 做“分块批处理”。
                    # 例如验证集有7个样本、batch_size=4时，Decoder 被批量调用两次：
                    # 第一次输入 [4, 64, 52]，第二次输入 [3, 64, 52]。
                    # 该循环不是把7个样本逐个送入 Decoder；分块只是为了限制
                    # Decoder 单次调用的显存占用。
                    for start in range(0, adap_latent.size(0), spectral_config["batch_size"]):
                        # 最后一个分块可能小于 spectral.batch_size。
                        stop = min(
                            start + spectral_config["batch_size"],
                            adap_latent.size(0),
                        )

                        # 一次性解码当前 latent 分块，输出
                        # [stop-start, 128, 416] 的 predicted Mel。
                        pred_mels = mel_decoder(
                            adap_latent[start:stop], sample_ids[start:stop]
                        )

                        # 从预处理数据中取得同一分块的 target Mel，并传到 GPU。
                        target_mel_chunk = target_mels[start:stop].to(
                            device, non_blocking=True
                        )

                        # 每个 chunk_loss 是当前分块内部的 mean loss。
                        chunk_loss = mel_loss_function(pred_mels, target_mel_chunk)

                        # 乘以当前分块样本数，避免最后一个较小分块与完整分块获得
                        # 相同权重；累加后再除以验证 batch 总样本数。
                        spec_loss_sum += chunk_loss * (stop - start)

                    # 得到当前完整验证 batch 上按样本加权的平均 spec loss。
                    spec_loss = spec_loss_sum / adap_latent.size(0)
                else:
                    # 未启用频谱损失时使用同设备零标量，保持指标记录接口不变。
                    spec_loss = latent_loss.new_zeros(())

                val_tracker.update(
                    {
                        "latent": latent_loss,
                        "huber": loss_huber,
                        "time": loss_time,
                        "std": loss_std,
                        "spec": spec_loss,
                    },
                    weight=inputs.size(0),
                )

            val_metrics = val_tracker.compute()
            val_metrics["total"] = (
                val_metrics["latent"]
                + spectral_config["lambda"] * val_metrics["spec"]
            )

            # loss_train 和 loss_val 写入相同 tags；TensorBoard 会按 run 将
            # 同一 loss 的 train/val 画成两条不同颜色的曲线。
            log_train_val_losses(
                train_writer,
                val_writer,
                train_metrics,
                val_metrics,
                epoch,
            )

            # 运行时间和学习率属于训练过程指标，只记录在 loss_train run。
            train_writer.add_scalar(
                "runtime/epoch_seconds",
                time.perf_counter() - epoch_start_time,
                epoch,
            )
            train_writer.add_scalar(
                "optimizer/learning_rate",
                epoch_learning_rate,
                epoch,
            )

        # validation 改善时只更新 CPU 中的最佳参数，不在 epoch 循环内写盘。
        if val_metrics["total"]<best_val_loss:
            val_no_impove=0
            best_epoch_point=epoch
            best_val_loss=val_metrics["total"]
            best_val_metrics = dict(val_metrics)
            best_model_state = update_best_state_on_cpu(best_model_state, model)
        else:
                val_no_impove += 1  # 统计没有提升的次数
            # 如果训练 es_num 个 epoch 没有提升, 结束训练
                if  val_no_impove >= config["train"]["es_num"] and config["train"]["early_stop"]:
                    print("No improvement for {:,} epochs, early stopping.".format(config["train"]["es_num"]))
                    print("epoch:", epoch)
                    if config["train"]["final_save"]:
                        torch.save(
                            model.state_dict(),
                            res_path / f"model_epoch_{epoch}.pth",
                        )
                    break
    train_writer.close()
    val_writer.close()

    if best_model_state is None:
        raise RuntimeError("Training finished without a valid best model state")
    # 一个 fold 结束后只执行一次 checkpoint 磁盘写入。
    torch.save(best_model_state, model_path)

    model_label, model_variant = get_model_identity(config["model"])
    selected_model_config = get_selected_model_config(config["model"])
    return {
        "S_idx": s_idx,
        "M_idx": m_idx,
        "model_name": model_label,
        "model_variant": model_variant,
        "model_display_name": selected_model_config["display_name"],
        "experiment_group": selected_model_config["experiment_group"],
        "model_parameter_count": model_parameter_count,
        "best_epoch": best_epoch_point,
        "last_epoch": last_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "training_seconds": time.perf_counter() - training_start_time,
        "checkpoint": str(model_path),
    }

if __name__=="__main__":
    configs = json.load(open("config.json"))
    os.makedirs(configs["save_load"]["save_folder"],exist_ok=True)
    with open(configs["save_load"]["save_folder"]+"/config.json", 'w') as f:
        json.dump(configs, f)

    train_main(config=configs)
