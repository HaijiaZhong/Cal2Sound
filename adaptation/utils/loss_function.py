import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 导入参数

def centering(K):
    # 中心化矩阵计算 (支持GPU)
    n = K.size(0)
    unit = torch.ones(n, n, device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return H @ K @ H  # 链式矩阵乘法优化计算顺序

def rbf(X, sigma=None):
    # 计算RBF核矩阵
    GX = torch.matmul(X, X.T)
    diag_GX = torch.diag(GX)  # 提取 GX 的对角线元素
    KX = diag_GX - GX + (diag_GX - GX).T
    # 自动计算sigma（基于中位数）
    if sigma is None:
        sigma = torch.median(KX).sqrt()
    
    K = torch.exp(-KX / (2 * sigma**2 ))
    return K

def kernel_HSIC(X, Y, sigma):
    # 计算HSIC核
    K = centering(rbf(X, sigma))
    L = centering(rbf(Y, sigma))
    return torch.sum(K * L)  # 保持梯度传播

def CKA_loss(X, Y, sigma=None):
    # 重构输入维度
    X_flat = X.view(X.size(0), -1)
    Y_flat = Y.view(Y.size(0), -1)
    
    # 计算HSIC和方差
    hsic = kernel_HSIC(X_flat, Y_flat, sigma)
    var_X = torch.sqrt(kernel_HSIC(X_flat, X_flat, sigma))
    var_Y = torch.sqrt(kernel_HSIC(Y_flat, Y_flat, sigma))
    
    return hsic / (var_X * var_Y)

def compute_rms(signal, frame_length=960):
    """_summary_

    Args:
        signal (_type_): [batch_size, samples] 多个音频采样后，组合而成的一个batch
        frame_length (int, optional): _description_. Defaults to 960.
        hop_length (int, optional): _description_. Defaults to 480.

    Returns:
        _type_: rms包络
    """
    # 将一个batch的信号分割成帧
    hop_length=int(frame_length//2)
    frames = signal.unfold(1, frame_length,hop_length)
    # 计算每帧的均方根能量
    rms = torch.sqrt(torch.mean(frames**2, dim=2))
    return rms


def contrastive_loss(source_latents,label_latents, temperature=0.1):
    # 归一化向量（余弦相似度等价于点积）
    batch_size=label_latents.size(0)
    label_latents=label_latents.reshape(batch_size,-1)
    source_latents=source_latents.reshape(batch_size,-1)
    # label_latents = F.normalize(label_latents, p=2, dim=1)
    # source_latents = F.normalize(source_latents, p=2, dim=1)
    
    # 计算相似度矩阵 [64, 64]
    similarity_matrix = torch.mm(label_latents, source_latents.T) / temperature
    
    # 标签为对角线位置（正样本对）
    labels = torch.arange(similarity_matrix.size(0), device=label_latents.device)
    
    # 交叉熵损失（最大化正样本对的相似度）
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def KL_loss(adaptor_latents,label_latents):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss_val = kl_loss(F.log_softmax(adaptor_latents,dim=-1),F.softmax(label_latents,dim=-1))
    return kl_loss_val

def pearson_loss(recons_wvs,label_envs,config):
    recons_envs=compute_rms(recons_wvs,frame_length=int(config["loss"]["window_size"])).to(device)
    batch_size=label_envs.size(0)

    # 拼接两组数据
    combined = torch.cat([recons_envs, label_envs], dim=0)  # 形状为 (2*batch_size, envs_dim)
    # 计算相关系数矩阵
    corr_matrix = torch.corrcoef(combined)
    # 提取特征与标签之间的相关系数矩阵
    feature_label_corr = corr_matrix[0:batch_size, batch_size:]  # 形状为 (batch_size, batch_size)
    mean_corr = torch.mean(torch.diagonal(feature_label_corr))
    
    return mean_corr

# def loss_function(adaptor_latents,label_latents,recons_wvs,label_envs,alpha=-40):
#     """20250305 zhj
#     """
#     kl_val = KL_loss(adaptor_latents,label_latents)
#     alpha = config["loss"]["corr"]["alpha"]
#     corr_val = alpha*(pearson_loss(recons_wvs,label_envs)-1)
#     loss = kl_val + corr_val
#     return kl_val,corr_val,loss

def joint_loss(adaptor_latents,label_latents,configs,alpha=20):
    """20250306 zhj 
    """
    kl_val = KL_loss(adaptor_latents,label_latents)
    huber_loss=nn.HuberLoss(delta=configs["loss"]["huber"]["delta"],
                            reduction=configs["loss"]["huber"]["reduction"])
    huber_val = huber_loss(adaptor_latents,label_latents) * alpha
    loss = kl_val + huber_val
    return kl_val,huber_val,loss


# class PseudoHuberLoss(nn.Module):
#     """The Pseudo-Huber loss.
#     20250425 by zhj
#     """

#     reductions = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}
    
#     def __init__(self, beta=1, reduction='mean'):
#         super().__init__()
#         self.beta = beta
#         self.reduction = reduction

#     def extra_repr(self):
#         return f'beta={self.beta:g}, reduction={self.reduction!r}'

#     def forward(self, input, target):
#         output = self.beta**2 * input.sub(target).div(self.beta).pow(2).add(1).sqrt().sub(1)
#         return self.reductions[self.reduction](output)

class PseudoHuberLoss(nn.Module):
    def __init__(self, c=0.00054, mean=True, use_dim_scaling=True):
        """
        基于 Music2Latent 源码重构的 Pseudo-Huber Loss。
        edited by zhj 20251223
        
        Args:
            c (float): 这里的 c 是基础阈值参数。默认值 0.00054 是 Music2Latent 的原始设置。
                       如果你发现模型收敛太慢或对细节捕捉不够，可以适当调大这个值（例如 0.01 - 0.1）。
            mean (bool): 是否返回整个 Batch 的平均 Loss。默认为 True。
            use_dim_scaling (bool): 是否启用 Music2Latent 特有的维度缩放技巧。
                                    即 c_final = c * sqrt(dim)。
                                    建议保持 True 以复现原论文逻辑；
                                    若设为 False，则 c 直接作为 L1/L2 的分界阈值。
        """
        super(PseudoHuberLoss, self).__init__()
        self.c = c
        self.mean = mean
        self.use_dim_scaling = use_dim_scaling

    def forward(self, input, target, w=None):
        """
        Args:
            input:  学生网络的输出 (Neuro Latent), 形状 [Batch, 64, 49]
            target: 教师/目标网络的输出 (Sound Latent), 形状 [Batch, 64, 49]
            w:      可选的权重 (Weight), 形状 [Batch] 或 [Batch, 1], 默认为 None。
        """
        # 1. 计算平方差
        # input/target shape: [B, 64, 49]
        # diff shape: [B, 64, 49]
        diff_sq = (input - target) ** 2

        # 2. 展平除 Batch 外的所有维度
        # Music2Latent 原逻辑：将 Frequency 和 Time 展平在一起处理
        # flatten shape: [B, 64*49] = [B, 3136]
        diff_flat = torch.flatten(diff_sq, start_dim=1)
        
        # 3. 处理参数 c
        if self.use_dim_scaling:
            # 获取特征总维度 (64 * 49 = 3136)
            data_dim = diff_flat.shape[-1]
            # 动态调整 c: c_scaled = c * sqrt(dim)
            # 这一步是为了让阈值匹配“求和”后的误差量级
            c_factor = self.c * torch.sqrt(torch.tensor(data_dim, device=input.device, dtype=input.dtype))
        else:
            # 如果不使用缩放，直接使用传入的 c (适合你明确知道想要 0.1 或 0.2 阈值的情况)
            c_factor = torch.tensor(self.c, device=input.device, dtype=input.dtype)

        # 4. 求和 (Sum over dim)
        # 这一步将所有特征维度的误差累加，得到每个样本的总平方误差
        # shape: [B]
        diff_sum = torch.sum(diff_flat, -1)

        # 5. Pseudo-Huber 核心公式 (Charbonnier 形式)
        # Loss = sqrt(Sum + c^2) - c
        loss = torch.sqrt(diff_sum + c_factor**2) - c_factor

        # 6. 处理 NaN (数值稳定性)
        loss = torch.nan_to_num(loss)

        # 7. 应用样本权重 (可选)
        if w is not None:
            loss = loss * w.squeeze()

        # 8. 返回结果
        if self.mean:
            return loss.mean()
        else:
            return loss
