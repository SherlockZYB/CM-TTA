from matplotlib import image
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms


def avg_entropy(outputs, eps=1e-8):
    p = torch.clamp(outputs, eps, 1.0 - eps)
    entropy_map = -p * torch.log(p)  # (N, H, W)
    entropies = entropy_map.view(outputs.size(0), -1).mean(dim=1)  # (N,)
    return entropies.sum(dim=-1)


def select_confident_samples(logits, top, eps=1e-8):
    p = torch.clamp(logits, eps, 1.0 - eps)
    entropy_map = -p * torch.log(p)  # (N, H, W)
    batch_entropy = entropy_map.view(logits.size(0), -1).mean(dim=1)  # (N,)
    K = max(1, int(batch_entropy.size()[0] * top))
    idx = torch.argsort(batch_entropy, descending=False)[:K]
    return logits[idx], idx

def select_confident_samples_contrast_entropy(contrast_scores, logits, top, eps=1e-8):
    """
    基于 contrast_entropy_rank 排名融合策略筛选高置信样本。
    contrast 排名 + entropy 排名，综合排名最小的 top 比例被选中。
    """
    p = torch.clamp(logits, eps, 1.0 - eps)
    entropy_map = -p * torch.log(p)  # (N, H, W)
    batch_entropy = entropy_map.view(logits.size(0), -1).mean(dim=1)  # (N,)

    # 排名融合
    entropy_ranks = batch_entropy.argsort().argsort().float()            # 低 entropy → 小排名
    contrast_ranks = (-contrast_scores).argsort().argsort().float()      # 高 contrast → 小排名
    combined_ranks = entropy_ranks + contrast_ranks                      # 综合排名，越小越好

    K = max(1, int(logits.size(0) * top))
    idx = torch.argsort(combined_ranks, descending=False)[:K]
    return logits[idx], idx


# ================================================================
#  多种不确定性指标筛选函数 (用于 SelectionOnly 消融)
# ================================================================

def select_by_variance(logits, top):
    """
    基于像素级方差筛选: 选取预测方差最低的视图 (预测越确定, 方差越低)。
    方差 = mean( p * (1-p) ), 对二值分割等价于 Bernoulli 方差。
    """
    p = logits  # (N, H, W), 值域 [0,1]
    pixel_var = p * (1 - p)  # (N, H, W)
    batch_var = pixel_var.view(logits.size(0), -1).mean(dim=1)  # (N,)
    K = max(1, int(logits.size(0) * top))
    idx = torch.argsort(batch_var, descending=False)[:K]  # 低方差 = 高置信
    return logits[idx], idx


def select_by_cross_view_consistency(logits, top):
    """
    基于跨视图一致性筛选: 每个视图与所有视图均值的 Dice 相似度。
    与集体一致性越高的视图越可信。
    """
    mean_pred = logits.mean(dim=0, keepdim=True)  # (1, H, W)
    # 计算每个视图与均值的 soft Dice
    mean_pred_expanded = mean_pred.expand_as(logits)  # (N, H, W)
    intersection = (logits * mean_pred_expanded).view(logits.size(0), -1).sum(dim=1)
    sum_pred = logits.view(logits.size(0), -1).sum(dim=1)
    sum_mean = mean_pred_expanded.view(logits.size(0), -1).sum(dim=1)
    dice_scores = (2 * intersection + 1e-6) / (sum_pred + sum_mean + 1e-6)  # (N,)
    K = max(1, int(logits.size(0) * top))
    idx = torch.argsort(dice_scores, descending=True)[:K]  # 高 dice = 高一致性
    return logits[idx], idx


def select_by_average_prob(logits, top):
    """
    基于平均概率筛选: 每个视图所有像素的平均预测概率。
    平均概率越高 → 模型对前景更自信 → 预测更可靠。
    适用于前景目标明确的医学分割任务 (如器官 / 病灶检测),
    排除因增强扰动导致前景信号被抑制的低质量视图。
    """
    batch_avg_prob = logits.view(logits.size(0), -1).mean(dim=1)  # (N,)
    K = max(1, int(logits.size(0) * top))
    idx = torch.argsort(batch_avg_prob, descending=True)[:K]  # 高平均概率 = 高置信
    return logits[idx], idx


def select_by_kl_divergence(logits, top, eps=1e-8):
    """
    基于 KL 散度筛选: 每个视图与所有视图均值分布之间的 KL 散度。
    KL(p_i || p_mean) 越小 → 该视图越接近集体共识 → 越可信。

    对二值分割, 逐像素计算 KL 散度:
      KL(p_i || q) = p_i * log(p_i / q) + (1-p_i) * log((1-p_i) / (1-q))

    Ref: Malinin & Gales, "Predictive Uncertainty Estimation via Prior Networks", NeurIPS 2018.
    """
    p = torch.clamp(logits, eps, 1.0 - eps)           # (N, H, W)
    q = torch.clamp(p.mean(dim=0, keepdim=True), eps, 1.0 - eps)  # (1, H, W)

    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))  # (N, H, W)
    batch_kl = kl.view(logits.size(0), -1).mean(dim=1)  # (N,)

    K = max(1, int(logits.size(0) * top))
    idx = torch.argsort(batch_kl, descending=False)[:K]  # 低 KL = 高共识
    return logits[idx], idx

def compute_fg_bg_similarity(text_features, text_mask, vision_features, ori_seg, fg_threshold=0.5):
    """
    计算前景和背景区域的图文相似度，并返回综合分数。
    
    好的 prompt 应该满足：
    1. 前景区域与文本相似度高 (fg_sim 高)
    2. 背景区域与文本相似度低 (bg_sim 低)
    3. 前景和背景的相似度差异大 (fg_sim - bg_sim 大)
    
    Args:
        text_features: (seq_len, batch_size, dim) - 文本特征
        text_mask: (batch_size, seq_len) - 文本 attention mask
        vision_features: (batch_size, dim, H, W) - 视觉特征图
        ori_seg: (batch_size, H_seg, W_seg) 或 (batch_size, 1, H_seg, W_seg) - 分割结果
        fg_threshold: 前景阈值
        
    Returns:
        combined_scores: (batch_size,) - 综合相似度分数 (越高越好)
        fg_sim: (batch_size,) - 前景相似度
        bg_sim: (batch_size,) - 背景相似度
        similarity_map: (batch_size, H, W) - 相似度图
        fg_mask: (batch_size, H, W) - 前景 mask
    """
    batch_size = vision_features.size(0)
    dim = vision_features.size(1)
    H, W = vision_features.size(2), vision_features.size(3)
    
    # --- 1. 提取文本特征 ---
    text_features = text_features.permute(1, 0, 2)  # (batch_size, seq_len, dim)
    
    if text_mask.sum() > text_mask.numel() // 2:
        valid_mask = ~text_mask
    else:
        valid_mask = text_mask
    
    mask_expanded = valid_mask.unsqueeze(-1).float()
    mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)
    text_pooled = (text_features * mask_expanded).sum(dim=1) / mask_sum  # (batch_size, dim)
    
    # --- 2. 计算相似度图 ---
    text_norm = F.normalize(text_pooled, dim=-1)
    vision_norm = F.normalize(vision_features, dim=1)
    text_norm_expanded = text_norm.unsqueeze(-1).unsqueeze(-1)
    similarity_map = (vision_norm * text_norm_expanded).sum(dim=1)  # (batch_size, H, W)
    
    # --- 3. 生成前景和背景 mask ---
    if ori_seg.dim() == 4:
        ori_seg = ori_seg.squeeze(1)
    
    ori_seg_prob = ori_seg.sigmoid()
    if ori_seg_prob.shape[-2:] != (H, W):
        seg_resized = F.interpolate(
            ori_seg_prob.unsqueeze(1).float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    else:
        seg_resized = ori_seg_prob.float()
    
    # 前景和背景 mask
    fg_mask = (seg_resized > fg_threshold).float()  # (batch_size, H, W)
    bg_mask = (seg_resized <= fg_threshold).float()  # (batch_size, H, W)
    
    # --- 4. 计算前景和背景的平均相似度 ---
    fg_pixel_count = fg_mask.sum(dim=(-2, -1)).clamp(min=1.0)
    bg_pixel_count = bg_mask.sum(dim=(-2, -1)).clamp(min=1.0)
    
    fg_sim = (similarity_map * fg_mask).sum(dim=(-2, -1)) / fg_pixel_count  # (batch_size,)
    bg_sim = (similarity_map * bg_mask).sum(dim=(-2, -1)) / bg_pixel_count  # (batch_size,)
    
    # --- 5. 计算综合分数 ---
    # 方法1: 前景-背景差异 (对比度)
    contrast_score = fg_sim - bg_sim  # 越大越好
    
    # 方法2: 加权组合 - 前景高 + 背景低
    # combined_scores = fg_sim - 0.5 * bg_sim
    
    # 方法3: 比值 (前景/背景)
    # ratio_score = fg_sim / (bg_sim + 1e-6)
    
    # 使用对比度分数作为默认
    combined_scores = contrast_score
    
    return combined_scores, fg_sim, bg_sim, similarity_map, fg_mask

def dice_loss(pred, target, smooth=1e-5):
    """
    计算 Dice Loss。

    Args:
        pred:   预测概率图, (N, H, W), 值域 [0, 1]
        target: 目标概率图, (N, H, W), 值域 [0, 1]（来自 teacher，已 detach）
        smooth: 平滑项

    Returns:
        1 - Dice coefficient（标量）
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class AugParamMemory:
    """
    增强参数记忆库: 跨 case 维护一组"高质量"增强参数组合 (而非图像本身)。

    核心思想:
      - 只存储增强变换的参数 (blur sigma, contrast, brightness, noise std, …)
      - 每个 TTA step 结束后，把被筛选中的增强参数连同其质量分数存入 buffer
      - 下一个 case 时，从 buffer 中采样高质量增强参数，应用到当前图像上
      - 避免了跨 case 图像混合的问题 — 增强参数天然与图像内容无关

    质量分数 = contrast_score - entropy
    """

    def __init__(self, max_size=32):
        self.max_size = max_size
        self.params = []    # list of dict (增强参数)
        self.scores = []    # list of float

    def __len__(self):
        return len(self.params)

    def push(self, param_dicts, quality_scores):
        """
        存入一批增强参数及质量分数。
        Args:
            param_dicts:     list of dict (每个 dict 是一组增强参数)
            quality_scores:  list/tensor of float
        """
        if isinstance(quality_scores, torch.Tensor):
            quality_scores = quality_scores.detach().cpu().tolist()

        for i, params in enumerate(param_dicts):
            score = float(quality_scores[i])
            if len(self.params) < self.max_size:
                self.params.append(params)
                self.scores.append(score)
            else:
                min_idx = int(np.argmin(self.scores))
                if score > self.scores[min_idx]:
                    self.params[min_idx] = params
                    self.scores[min_idx] = score

    def sample(self, n):
        """采样 n 组质量最高的增强参数。返回 list of dict, 或空 list。"""
        if len(self.params) == 0:
            return []
        k = min(n, len(self.params))
        top_indices = np.argsort(self.scores)[-k:][::-1]
        return [self.params[i] for i in top_indices]

    def reset(self):
        self.params.clear()
        self.scores.clear()

    def stats(self):
        if len(self.scores) == 0:
            return {"size": 0, "min_score": None, "max_score": None, "mean_score": None}
        return {
            "size": len(self.scores),
            "min_score": round(float(np.min(self.scores)), 4),
            "max_score": round(float(np.max(self.scores)), 4),
            "mean_score": round(float(np.mean(self.scores)), 4),
        }


def apply_parameterized_aug(images, params):
    """
    根据参数字典对图像应用确定性增强。
    Args:
        images: (1, C, H, W) tensor
        params: dict with keys: blur_sigma, auto_contrast, brightness, contrast,
                saturation, hue, sharpness, noise_std
    Returns:
        (1, C, H, W) augmented tensor
    """
    aug_img = images.clone()

    if params.get('blur_sigma', 0) > 0:
        aug_img = transforms.functional.gaussian_blur(
            aug_img, kernel_size=3, sigma=params['blur_sigma'])

    if params.get('auto_contrast', False):
        aug_img = transforms.functional.autocontrast(aug_img)

    b = params.get('brightness', 1.0)
    c = params.get('contrast', 1.0)
    s = params.get('saturation', 1.0)
    h = params.get('hue', 0.0)
    if b != 1.0 or c != 1.0 or s != 1.0 or h != 0.0:
        aug_img = transforms.functional.adjust_brightness(aug_img, b)
        aug_img = transforms.functional.adjust_contrast(aug_img, c)
        aug_img = transforms.functional.adjust_saturation(aug_img, s)
        aug_img = transforms.functional.adjust_hue(aug_img, h)

    sf = params.get('sharpness', 1.0)
    if sf != 1.0:
        aug_img = transforms.functional.adjust_sharpness(aug_img, sf)

    noise_std = params.get('noise_std', 0.0)
    if noise_std > 0:
        noise = torch.randn_like(aug_img) * noise_std
        aug_img = aug_img + noise

    return torch.clamp(aug_img, 0, 1)


def sample_aug_params():
    """
    随机采样一组增强参数 (与原始 aug_transform 对应)。
    返回 dict。
    """
    params = {}

    # GaussianBlur
    if torch.rand(1).item() < 0.5:
        params['blur_sigma'] = float(torch.empty(1).uniform_(0.1, 2.0).item())
    else:
        params['blur_sigma'] = 0.0

    # AutoContrast
    params['auto_contrast'] = (torch.rand(1).item() < 0.5)

    # ColorJitter
    if torch.rand(1).item() < 0.5:
        params['brightness'] = float(torch.empty(1).uniform_(0.8, 1.2).item())
        params['contrast'] = float(torch.empty(1).uniform_(0.8, 1.2).item())
        params['saturation'] = float(torch.empty(1).uniform_(0.8, 1.2).item())
        params['hue'] = float(torch.empty(1).uniform_(-0.1, 0.1).item())
    else:
        params['brightness'] = 1.0
        params['contrast'] = 1.0
        params['saturation'] = 1.0
        params['hue'] = 0.0

    # Sharpness
    if torch.rand(1).item() < 0.5:
        params['sharpness'] = 2.0
    else:
        params['sharpness'] = 1.0

    # Noise
    if torch.rand(1).item() < 0.5:
        params['noise_std'] = 0.05
    else:
        params['noise_std'] = 0.0

    return params


class PromptMemory:
    """
    Prompt 特征记忆库: 跨 case 存储历史最优 prompt embedding 快照。

    核心思想:
      - 每个 case TTA 结束后，保存当前 prompt embedding 和对应的质量分数
      - 下一个 case 开始 TTA 前，用历史最优 prompt 做加权融合初始化
      - 相比存图像: prompt embedding 维度小 (~16 × 768), 天然跨 case 通用
      - 适用于 online 模式 (prompt 状态天然跨 case 传递)

    用法:
      1. 每个 case 结束时: memory.push(prompt_embedding, quality_score)
      2. 每个 case 开始时: memory.get_init_prompt() → 加权融合的初始 prompt
    """

    def __init__(self, max_size=16, fusion_momentum=0.3):
        """
        Args:
            max_size:         记忆库最大容量
            fusion_momentum:  融合时历史 prompt 的权重 (0=不用历史, 1=完全用历史)
        """
        self.max_size = max_size
        self.fusion_momentum = fusion_momentum
        self.prompts = []   # list of (n_ctx, dim) CPU tensors
        self.scores = []    # list of float

    def __len__(self):
        return len(self.prompts)

    def push(self, prompt_embedding, quality_score):
        """
        存入一个 prompt embedding 快照。
        Args:
            prompt_embedding: (n_ctx, dim) 或 (batch, n_ctx, dim) tensor
            quality_score:    float
        """
        emb = prompt_embedding.detach().cpu()
        if emb.dim() == 3:
            emb = emb[0]  # 取第一个 batch
        score = float(quality_score)

        if len(self.prompts) < self.max_size:
            self.prompts.append(emb)
            self.scores.append(score)
        else:
            min_idx = int(np.argmin(self.scores))
            if score > self.scores[min_idx]:
                self.prompts[min_idx] = emb
                self.scores[min_idx] = score

    def get_init_prompt(self, current_prompt, device='cuda'):
        """
        基于历史最优 prompt 和当前 prompt 做加权融合。

        融合策略: prompt_init = (1 - m) * current + m * weighted_history
        历史 prompt 按质量分数 softmax 加权平均。

        Args:
            current_prompt: (n_ctx, dim) 或 (batch, n_ctx, dim) tensor
            device: 目标设备

        Returns:
            融合后的 prompt tensor, 与输入 shape 相同
        """
        if len(self.prompts) == 0:
            return current_prompt

        # 按质量分数 softmax 加权平均历史 prompt
        scores_arr = np.array(self.scores)
        scores_arr = scores_arr - scores_arr.max()
        weights = np.exp(scores_arr)
        weights = weights / weights.sum()

        weighted_prompt = torch.zeros_like(self.prompts[0])
        for w, p in zip(weights, self.prompts):
            weighted_prompt += w * p
        weighted_prompt = weighted_prompt.to(device)

        # 融合
        if current_prompt.dim() == 3:
            fused = ((1 - self.fusion_momentum) * current_prompt +
                     self.fusion_momentum * weighted_prompt.unsqueeze(0).expand_as(current_prompt))
        else:
            fused = ((1 - self.fusion_momentum) * current_prompt +
                     self.fusion_momentum * weighted_prompt)

        return fused

    def reset(self):
        self.prompts.clear()
        self.scores.clear()

    def stats(self):
        if len(self.scores) == 0:
            return {"size": 0, "min_score": None, "max_score": None, "mean_score": None}
        return {
            "size": len(self.scores),
            "min_score": round(float(np.min(self.scores)), 4),
            "max_score": round(float(np.max(self.scores)), 4),
            "mean_score": round(float(np.mean(self.scores)), 4),
        }


class AdaptiveLR:
    """
    自适应学习率调度器: 基于历史 case 的质量分数动态调整学习率和 EMA 动量。

    核心思想:
      - 维护历史 case 质量分数的滑动均值和标准差
      - 当前 case 质量 > 均值 → 模型适应良好 → 用更大学习率做更大步适应
      - 当前 case 质量 < 均值 → 模型不稳定 → 减小学习率保守适应
      - 同时动态调整 EMA 动量: 质量差时增大动量（更依赖 teacher 惯性）

    用法:
      - 每个 case 的 TTA step 中调用 get_lr_scale() 获取缩放因子
      - 每个 case 结束后调用 update(quality_score) 更新历史
    """

    def __init__(self, base_lr, base_momentum=0.99,
                 lr_scale_range=(0.3, 2.0),
                 momentum_range=(0.95, 0.999),
                 history_momentum=0.9):
        """
        Args:
            base_lr:          基础学习率
            base_momentum:    基础 EMA 动量
            lr_scale_range:   学习率缩放范围 (min_scale, max_scale)
            momentum_range:   EMA 动量范围 (min_momentum, max_momentum)
            history_momentum: 历史质量分数的滑动平均动量
        """
        self.base_lr = base_lr
        self.base_momentum = base_momentum
        self.lr_scale_min, self.lr_scale_max = lr_scale_range
        self.momentum_min, self.momentum_max = momentum_range
        self.history_momentum = history_momentum

        self.quality_baseline = None
        self.quality_std = None
        self.case_count = 0
        self.history_scores = []

    def update(self, quality_score):
        """
        每个 case 结束后调用，更新历史统计。
        Args:
            quality_score: float — 当前 case 的质量分数 (contrast - entropy)
        """
        score = float(quality_score)
        self.history_scores.append(score)
        if len(self.history_scores) > 50:
            self.history_scores = self.history_scores[-50:]

        if self.quality_baseline is None:
            self.quality_baseline = score
            self.quality_std = 0.1
        else:
            self.quality_baseline = (self.history_momentum * self.quality_baseline +
                                     (1 - self.history_momentum) * score)
            if len(self.history_scores) >= 3:
                self.quality_std = max(float(np.std(self.history_scores[-20:])), 0.01)

        self.case_count += 1

    def get_lr_scale(self, current_quality=None):
        """
        获取当前学习率缩放因子。
        Args:
            current_quality: 当前 step 的质量分数 (可选)
        Returns:
            float — 学习率缩放因子 (乘以 base_lr 使用)
        """
        if self.quality_baseline is None or current_quality is None:
            return 1.0

        z = (current_quality - self.quality_baseline) / max(self.quality_std, 0.01)
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
        scale = self.lr_scale_min + sigmoid_z * (self.lr_scale_max - self.lr_scale_min)
        return float(scale)

    def get_momentum(self, current_quality=None):
        """
        获取动态 EMA 动量。
        质量差 → 动量大 (更依赖 teacher 惯性)
        质量好 → 动量小 (student 更新更多地传递给 teacher)
        """
        if self.quality_baseline is None or current_quality is None:
            return self.base_momentum

        z = (current_quality - self.quality_baseline) / max(self.quality_std, 0.01)
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
        momentum = self.momentum_max - sigmoid_z * (self.momentum_max - self.momentum_min)
        return float(momentum)

    def stats(self):
        return {
            "case_count": self.case_count,
            "quality_baseline": round(self.quality_baseline, 4) if self.quality_baseline is not None else None,
            "quality_std": round(self.quality_std, 4) if self.quality_std is not None else None,
            "n_history": len(self.history_scores),
        }


class TPT_MT():
    """
    TPT + Mean Teacher 架构。

    在原始 TPT（最小化增强视图的 entropy）基础上新增：
      1. EMA Teacher 模型  —  每步用 student 参数指数滑动平均更新
      2. Dice 一致性 Loss  —  student 对增强图像的输出 与 teacher 对原图输出 的 Dice Loss
      3. 总 Loss = entropy_loss + w_dice * dice_loss

    跨 case 信息共享策略（互斥, 通过 CLI 开关选择）:
      A. 增强参数记忆库 (--use_aug_param_memory): 存储高质量增强参数组合
      B. Prompt 特征记忆库 (--use_prompt_memory): 存储历史最优 prompt embedding
      C. 自适应学习率 (--use_adaptive_lr): 基于历史质量动态调整 lr 和 momentum
    """

    def __init__(self, model, device):
        self.model = model           # student
        self.device = device
        self.teacher_model = None    # 在 prepare 阶段创建

    # ----------------------------------------------------------------
    #  prepare
    # ----------------------------------------------------------------
    def prepare_model_and_optimization(self, args):
        self.model.eval()
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1000)

        # EMA 超参
        self.ema_momentum = getattr(args, 'ema_momentum', 0.99)
        self.w_dice = getattr(args, 'w_dice', 1.0)
        self.w_contrast = getattr(args, 'w_contrast', 1.0)
        self.w_entropy = getattr(args, 'w_entropy', 1.0)

        # 消融开关
        self.use_contrast_selection = bool(getattr(args, 'use_contrast_selection', 1))
        self.use_contrast_loss = bool(getattr(args, 'use_contrast_loss', 1))
        self.use_dice_loss = bool(getattr(args, 'use_dice_loss', 1))
        self.use_entropy_loss = bool(getattr(args, 'use_entropy_loss', 1))
        self.use_teacher = bool(getattr(args, 'use_teacher', 1))
        # 消融: 不用 EMA Teacher / prompt memory，但用 student 自身对选中视图的 no_grad
        # 输出作为 dice 监督目标（自一致性 Dice）
        self.use_self_dice = bool(getattr(args, 'use_self_dice', 0))
        self.online = bool(getattr(args, 'online', 0))
        self.use_bbox_prompt = bool(getattr(args, 'use_bbox_prompt', 0))
        self.use_point_prompt = bool(getattr(args, 'use_point_prompt', 0))
        self.n_fg_points = getattr(args, 'n_fg_points', 3)
        self.n_bg_points = getattr(args, 'n_bg_points', 3)
        self.point_conf_thresh = getattr(args, 'point_conf_thresh', 0.7)
        self.use_loss = self.use_contrast_loss or self.use_dice_loss or self.use_entropy_loss or self.use_self_dice

        # 多尺度监督: 在 ori_seg（未 resize 的原始分辨率）上也计算 dice loss
        self.use_multiscale_dice = bool(getattr(args, 'use_multiscale_dice', 0))
        self.w_dice_ori = getattr(args, 'w_dice_ori', 0.5)

        # 医学先验: 前景占比合理范围 (例如 prostate 约 1%~10%)
        self.fg_prior_min = getattr(args, 'fg_prior_min', 0.01)
        self.fg_prior_max = getattr(args, 'fg_prior_max', 0.10)

        # 增强视图数量（默认 9 张增强 + 1 原图 = 10）
        self.num_aug_views = getattr(args, 'num_aug_views', 9)

        # 最终推理模式: 'original'=用原图推理, 'selected'=用筛选后增强视图取均值
        self.infer_mode = getattr(args, 'infer_mode', 'original')

        # ---------- 跨 case 信息共享策略 (三选一) ----------
        # 方案 A: 增强参数记忆库
        self.use_aug_param_memory = bool(getattr(args, 'use_aug_param_memory', 0))
        self.aug_param_memory_size = getattr(args, 'aug_param_memory_size', 32)
        self.n_memory_params = getattr(args, 'n_memory_params', 3)
        self.aug_param_buffer = (AugParamMemory(max_size=self.aug_param_memory_size)
                                 if self.use_aug_param_memory else None)

        # 方案 B: Prompt 特征记忆库
        self.use_prompt_memory = bool(getattr(args, 'use_prompt_memory', 0))
        self.prompt_memory_size = getattr(args, 'prompt_memory_size', 16)
        self.prompt_fusion_momentum = getattr(args, 'prompt_fusion_momentum', 0.3)
        self.prompt_memory = (PromptMemory(max_size=self.prompt_memory_size,
                                           fusion_momentum=self.prompt_fusion_momentum)
                              if self.use_prompt_memory else None)

        # 方案 C: 自适应学习率
        self.use_adaptive_lr = bool(getattr(args, 'use_adaptive_lr', 0))
        self.lr_scale_min = getattr(args, 'lr_scale_min', 0.3)
        self.lr_scale_max = getattr(args, 'lr_scale_max', 2.0)

        # ---------- Debug 日志 ----------
        self.debug_logs = []  # 每个 case 每个 step 的诊断信息
        self.output_dir = getattr(args, 'output_dir', '.')
        self.save_debug_vis = bool(getattr(args, 'save_debug_vis', 0))

        # 创建 Teacher（深拷贝 student，冻结梯度）
        if self.use_teacher:
            self.teacher_model = deepcopy(self.model)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
            self.teacher_init_state = deepcopy(self.teacher_model.state_dict())
        else:
            self.teacher_model = None
            self.teacher_init_state = None

        # ---------- 收集可训练参数（仅 prompt_learner）----------
        trainable_param = [p for p in self.model.prompt_learner.parameters() if p.requires_grad]

        # Create optimizer
        lr = getattr(args, 'lr', 1e-4)
        self.base_lr = lr
        if len(trainable_param) == 0:
            self.optimizer = torch.optim.AdamW([], lr=lr)
        else:
            self.optimizer = torch.optim.AdamW(trainable_param, lr=lr)

        self.optim_state = deepcopy(self.optimizer.state_dict())

        # 初始化 AdaptiveLR (需要 lr)
        self.adaptive_lr = (AdaptiveLR(
            base_lr=lr,
            base_momentum=self.ema_momentum,
            lr_scale_range=(self.lr_scale_min, self.lr_scale_max),
            momentum_range=(0.95, 0.999),
            history_momentum=0.9,
        ) if self.use_adaptive_lr else None)

        # --- 汇总 ---
        active_strategy = "none"
        if self.use_aug_param_memory:
            active_strategy = f"aug_param_memory(size={self.aug_param_memory_size}, n={self.n_memory_params})"
        elif self.use_prompt_memory:
            active_strategy = f"prompt_memory(size={self.prompt_memory_size}, fusion_m={self.prompt_fusion_momentum})"
        elif self.use_adaptive_lr:
            active_strategy = f"adaptive_lr(scale=[{self.lr_scale_min}, {self.lr_scale_max}])"

        print(f"[TPT_MT] 配置总结: aug_views={self.num_aug_views}+1, "
              f"trainable_params={sum(p.numel() for p in trainable_param)}, "
              f"fg_prior=[{self.fg_prior_min}, {self.fg_prior_max}], "
              f"point_prompt={self.use_point_prompt} (n_fg={self.n_fg_points}, n_bg={self.n_bg_points}), "
              f"multiscale_dice={self.use_multiscale_dice} (w_dice_ori={self.w_dice_ori}), "
              f"infer_mode={self.infer_mode}, "
              f"cross_case_strategy={active_strategy}")

    # ----------------------------------------------------------------
    #  pre_adaptation（每个样本 / case 前重置）
    # ----------------------------------------------------------------
    def pre_adaptation(self, args=None):
        if self.online:
            # online 模式: 不 reset，跨 case 持续适应
            # 方案 B: 用历史 prompt 融合初始化当前 prompt
            if self.prompt_memory is not None and len(self.prompt_memory) > 0:
                with torch.no_grad():
                    current_ctx = self.model.prompt_learner.ctx.data
                    fused_ctx = self.prompt_memory.get_init_prompt(
                        current_ctx, device=current_ctx.device)
                    self.model.prompt_learner.ctx.data.copy_(fused_ctx)
            return
        with torch.no_grad():
            self.model.reset()  # reset prompt_learner
        self.optimizer.load_state_dict(self.optim_state)
        # instance 模式下清空跨 case 记忆
        if self.aug_param_buffer is not None:
            self.aug_param_buffer.reset()
        if self.prompt_memory is not None:
            self.prompt_memory.reset()

    # ----------------------------------------------------------------
    #  EMA 更新 teacher（支持 selective update）
    # ----------------------------------------------------------------
    @torch.no_grad()
    def update_teacher(self, momentum=None):
        """
        teacher_param = m * teacher_param + (1-m) * student_param
        Args:
            momentum: 可选的动态动量值 (由 AdaptiveLR 提供)
        """
        if self.teacher_model is None:
            return

        m = momentum if momentum is not None else self.ema_momentum
        for tp, sp in zip(self.teacher_model.parameters(), self.model.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)

    # ----------------------------------------------------------------
    #  Visualization helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _compute_dice_np(pred, gt):
        """计算两个 binary numpy mask 的 Dice 系数"""
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)
        inter = (pred * gt).sum()
        denom = pred.sum() + gt.sum()
        if denom == 0:
            return 1.0 if inter == 0 else 0.0
        return round(2.0 * inter / denom, 4)

    @staticmethod
    def _get_bbox_from_mask(mask_np, margin_ratio=0.15, min_margin=10):
        """
        从 binary mask 中提取 bbox (xyxy) 并添加 margin。
        返回 (x0, y0, x1, y1) 或 None（无前景）。
        """
        mask_np = np.squeeze(mask_np)  # 确保 2D
        if mask_np.ndim != 2:
            return None
        ys, xs = np.where(mask_np > 0.5)
        if len(ys) == 0:
            return None
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        H, W = mask_np.shape
        bw, bh = x1 - x0, y1 - y0
        mx = max(bw * margin_ratio, min_margin)
        my = max(bh * margin_ratio, min_margin)
        x0 = max(x0 - mx, 0)
        y0 = max(y0 - my, 0)
        x1 = min(x1 + mx, W - 1)
        y1 = min(y1 + my, H - 1)
        return (x0, y0, x1, y1)

    def _extract_point_prompts(self, prob_maps, n_fg=3, n_bg=3, conf_thresh=0.7):
        """
        从 student 概率图中提取可靠的前景点和背景点。

        策略:
          - 前景点: 从高置信度前景区域 (prob > conf_thresh) 中采样
            优先选择概率最高的位置（最可靠的前景证据）
          - 背景点: 从高置信度背景区域 (prob < 1-conf_thresh) 中采样
            优先选择概率最低的位置（最可靠的背景证据）

        Args:
            prob_maps: (K, H, W) student 概率图（多个增强视图的筛选结果）
            n_fg: 前景点数量
            n_bg: 背景点数量
            conf_thresh: 置信度阈值

        Returns:
            fg_points: (N_fg, 2) 归一化坐标 [x, y]，或 None（无可靠前景）
            bg_points: (N_bg, 2) 归一化坐标 [x, y]，或 None
            diag: dict 诊断信息
        """
        device = prob_maps.device
        # 用所有选中视图的平均概率作为置信度
        avg_prob = prob_maps.detach().mean(dim=0)  # (H, W)
        H, W = avg_prob.shape

        diag = {"n_fg_requested": n_fg, "n_bg_requested": n_bg, "conf_thresh": conf_thresh}

        # --- 前景点: prob > conf_thresh 的区域 ---
        fg_mask = (avg_prob > conf_thresh)
        fg_count = fg_mask.sum().item()
        diag["fg_candidates"] = int(fg_count)

        fg_points = None
        if fg_count > 0:
            # 取概率最高的 n_fg 个位置
            fg_probs = avg_prob.clone()
            fg_probs[~fg_mask] = 0.0
            fg_flat = fg_probs.reshape(-1)
            k_fg = min(n_fg, fg_count)
            topk_vals, topk_idx = torch.topk(fg_flat, k_fg)
            fg_ys = (topk_idx // W).float()
            fg_xs = (topk_idx % W).float()
            # 归一化到 [0, 1]
            fg_points = torch.stack([fg_xs / float(W), fg_ys / float(H)], dim=-1)  # (k_fg, 2)
            diag["fg_points_pixel"] = list(zip(fg_xs.cpu().tolist(), fg_ys.cpu().tolist()))
            diag["fg_points_yx"] = list(zip(fg_ys.cpu().tolist(), fg_xs.cpu().tolist()))  # for visualization
            diag["fg_points_conf"] = topk_vals.cpu().tolist()
            diag["n_fg_actual"] = k_fg

        # --- 背景点: prob < (1 - conf_thresh) 的区域 ---
        bg_mask = (avg_prob < (1.0 - conf_thresh))
        bg_count = bg_mask.sum().item()
        diag["bg_candidates"] = int(bg_count)

        bg_points = None
        if bg_count > 0:
            # 取概率最低的 n_bg 个位置
            bg_probs = avg_prob.clone()
            bg_probs[~bg_mask] = 1.0
            bg_flat = bg_probs.reshape(-1)
            k_bg = min(n_bg, bg_count)
            _, botk_idx = torch.topk(-bg_flat, k_bg)  # 最小概率
            bg_ys = (botk_idx // W).float()
            bg_xs = (botk_idx % W).float()
            bg_points = torch.stack([bg_xs / float(W), bg_ys / float(H)], dim=-1)  # (k_bg, 2)
            diag["bg_points_pixel"] = list(zip(bg_xs.cpu().tolist(), bg_ys.cpu().tolist()))
            diag["bg_points_yx"] = list(zip(bg_ys.cpu().tolist(), bg_xs.cpu().tolist()))  # for visualization
            diag["bg_points_conf"] = bg_probs.reshape(-1)[botk_idx].cpu().tolist()
            diag["n_bg_actual"] = k_bg

        return fg_points, bg_points, diag

    def _save_debug_vis(self, img_np, gt_np, student_np, teacher_np,
                        student_bbox_xyxy, gt_bbox_xyxy,
                        use_bbox, case_name, step, extra_info=None,
                        fg_points_yx=None, bg_points_yx=None, prompt_mode="text"):
        """
        保存一张 4-panel 调试图:
          [0] 原图 + 提示 (点/bbox) + GT bbox
          [1] GT mask 叠加
          [2] Student mask 叠加
          [3] Teacher mask 叠加 (标注 prompt 模式)

        每个 mask panel 标注 Dice vs GT。
        """
        vis_dir = os.path.join(self.output_dir, "debug_vis")
        os.makedirs(vis_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # --- Panel 0: 原图 + prompts ---
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f"Image + Prompts ({prompt_mode})", fontsize=10)
        if gt_bbox_xyxy is not None:
            x0, y0, x1, y1 = gt_bbox_xyxy
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            axes[0].add_patch(rect)
            axes[0].text(x0, max(y0 - 5, 0), 'GT', color='red', fontsize=9, fontweight='bold')

        # 画点提示
        if fg_points_yx is not None and len(fg_points_yx) > 0:
            for pt in fg_points_yx:
                y_px, x_px = pt[0], pt[1]
                axes[0].plot(x_px, y_px, 'o', color='lime', markersize=8,
                             markeredgecolor='white', markeredgewidth=1.5)
            axes[0].text(5, 15, f'FG pts: {len(fg_points_yx)}', color='lime', fontsize=9,
                         fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        if bg_points_yx is not None and len(bg_points_yx) > 0:
            for pt in bg_points_yx:
                y_px, x_px = pt[0], pt[1]
                axes[0].plot(x_px, y_px, 'x', color='red', markersize=8,
                             markeredgecolor='white', markeredgewidth=1.5)
            axes[0].text(5, 35, f'BG pts: {len(bg_points_yx)}', color='red', fontsize=9,
                         fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # 画 bbox (兼容 bbox 模式)
        if student_bbox_xyxy is not None:
            x0, y0, x1, y1 = student_bbox_xyxy
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(x0, max(y0 - 5, 0), 'Student', color='lime', fontsize=9, fontweight='bold')

        # --- Panel 1: GT mask ---
        axes[1].imshow(img_np, cmap='gray')
        if gt_np is not None:
            gt_overlay = np.zeros((*gt_np.shape, 4))
            gt_overlay[gt_np > 0.5] = [1, 0, 0, 0.4]  # 红色半透明
            axes[1].imshow(gt_overlay)
            gt_fg = (gt_np > 0.5).sum() / gt_np.size
            axes[1].set_title(f"GT (fg: {gt_fg:.3f})", fontsize=10)
        else:
            axes[1].set_title("GT (N/A)", fontsize=10)

        # --- Panel 2: Student mask ---
        axes[2].imshow(img_np, cmap='gray')
        stu_overlay = np.zeros((*student_np.shape, 4))
        stu_overlay[student_np > 0.5] = [0, 1, 0, 0.4]  # 绿色半透明
        axes[2].imshow(stu_overlay)
        dice_stu = self._compute_dice_np(student_np, gt_np) if gt_np is not None else -1
        axes[2].set_title(f"Student (Dice: {dice_stu:.3f})", fontsize=10)

        # --- Panel 3: Teacher mask ---
        axes[3].imshow(img_np, cmap='gray')
        tea_overlay = np.zeros((*teacher_np.shape, 4))
        tea_overlay[teacher_np > 0.5] = [0, 0.5, 1, 0.4]  # 蓝色半透明
        axes[3].imshow(tea_overlay)
        dice_tea = self._compute_dice_np(teacher_np, gt_np) if gt_np is not None else -1
        axes[3].set_title(f"Teacher/{prompt_mode} (Dice: {dice_tea:.3f})", fontsize=10)

        for ax in axes:
            ax.axis('off')

        # 额外信息标题
        info_parts = [f"Case: {case_name}  Step: {step}"]
        if extra_info:
            for k, v in extra_info.items():
                info_parts.append(f"{k}={v}")
        fig.suptitle("  |  ".join(info_parts), fontsize=9, y=0.02)

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        save_path = os.path.join(vis_dir, f"{case_name}_step{step}.png")
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return {"dice_student_gt": dice_stu, "dice_teacher_gt": dice_tea,
                "prompt_mode": prompt_mode}

    # ----------------------------------------------------------------
    #  adaptation_process
    # ----------------------------------------------------------------
    def adaptation_process(self, images, args, basenames=None, gt_masks=None):
        assert args.tta_steps > 0
        selected_idx = None
        case_name = basenames[0] if basenames else "unknown"

        # ---------- 准备 GT 用于可视化 ----------
        gt_np = None
        gt_bbox_xyxy = None
        img_np = None
        if gt_masks is not None:
            # gt_masks 可能是 (B, H, W) 或 (B, 1, H, W) — 取第一张并 squeeze 到 2D
            gt_np = gt_masks[0].cpu().numpy().astype(np.float32)
            gt_np = gt_np.squeeze()  # 确保 (H, W)
            gt_bbox_xyxy = self._get_bbox_from_mask(gt_np)
            # 原图: images 是 (1, 3, H, W) 归一化 [0,1]
            img_np = images[0, 0].cpu().numpy()  # 取单通道灰度

        gt_fg_ratio = round((gt_np > 0.5).sum() / gt_np.size, 6) if gt_np is not None else None

        # ---------- 数据增强 ----------
        # 方案 A: 增强参数记忆库 — 用参数化增强替代随机 transform
        if self.use_aug_param_memory:
            # 从记忆库采样历史高质量增强参数
            memory_params = (self.aug_param_buffer.sample(self.n_memory_params)
                             if self.aug_param_buffer is not None and len(self.aug_param_buffer) > 0
                             else [])
            n_random = max(1, self.num_aug_views - len(memory_params))

            # 生成所有增强参数 (记忆库参数 + 新随机参数)
            all_aug_params = []
            augmented_images = []

            # 先用记忆库中的参数 (应用到当前图像)
            for p in memory_params:
                aug_img = apply_parameterized_aug(images, p)
                augmented_images.append(aug_img)
                all_aug_params.append(p)

            # 再随机采样新参数
            for _ in range(n_random):
                p = sample_aug_params()
                aug_img = apply_parameterized_aug(images, p)
                augmented_images.append(aug_img)
                all_aug_params.append(p)

            aug_images = torch.cat([images] + augmented_images, dim=0)
        else:
            # 默认随机增强
            aug_transform = transforms.Compose([
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomAutocontrast()
                ], p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomAdjustSharpness(sharpness_factor=2)
                ], p=0.5),
            ])

            augmented_images = []
            for i in range(self.num_aug_views):
                aug_img = aug_transform(images)
                if torch.rand(1) > 0.5:
                    noise = torch.randn_like(aug_img) * 0.05
                    aug_img = aug_img + noise
                aug_img = torch.clamp(aug_img, 0, 1)
                augmented_images.append(aug_img)

            aug_images = torch.cat([images] + augmented_images, dim=0)
            all_aug_params = None  # 标记: 非参数化模式

        # ---------- TTA 迭代 ----------
        for j in range(args.tta_steps):
            step_debug = {"case": case_name, "step": j, "gt_fg_ratio": gt_fg_ratio}

            with torch.amp.autocast('cuda'):
                # --- Student: 对增强图像前向 ---
                ori_seg, student_output, text_features, text_mask, vision_features, presence_scores = self.model(aug_images)

                student_output = torch.stack(student_output, dim=0)  # (N_aug, H, W)
                ori_seg = torch.stack(ori_seg, dim=0)

                # --- 记录 student 统计 ---
                with torch.no_grad():
                    stu_probs = student_output.detach()
                    step_debug["student_mean_prob"] = round(stu_probs.mean().item(), 4)
                    step_debug["student_fg_ratio"] = round((stu_probs > 0.5).float().mean().item(), 4)
                    # 原图 (idx=0) 的统计
                    step_debug["student_orig_mean_prob"] = round(stu_probs[0].mean().item(), 4)
                    step_debug["student_orig_fg_ratio"] = round((stu_probs[0] > 0.5).float().mean().item(), 4)

                # --- 计算 contrast scores (筛选 / loss 都可能用到) ---
                if self.use_contrast_selection or self.use_contrast_loss:
                    contrast_scores, fg_sim, bg_sim, similarity_map, fg_mask = compute_fg_bg_similarity(
                        text_features=text_features,
                        text_mask=text_mask,
                        vision_features=vision_features,
                        ori_seg=ori_seg,
                        fg_threshold=0.5
                    )
                else:
                    contrast_scores = None

                # --- 置信度筛选（仅第一步计算 idx，后续复用）---
                if selected_idx is not None:
                    student_selected = student_output[selected_idx]
                else:
                    if self.use_contrast_selection:
                        # contrast_entropy rank fusion
                        student_selected, selected_idx = select_confident_samples_contrast_entropy(
                            contrast_scores, student_output, args.selection_p)
                    else:
                        # 仅 entropy 筛选 (原始 TPT)
                        student_selected, selected_idx = select_confident_samples(
                            student_output, args.selection_p)
                
                step_debug["selected_idx"] = selected_idx.cpu().tolist()
                step_debug["num_selected"] = len(selected_idx)
                del student_output

                # --- 医学先验门控: 检查 student 前景占比是否合理 ---
                with torch.no_grad():
                    stu_fg_ratio = (student_selected.detach() > 0.5).float().mean().item()
                prior_valid = (self.fg_prior_min <= stu_fg_ratio <= self.fg_prior_max)
                step_debug["stu_selected_fg_ratio"] = round(stu_fg_ratio, 6)
                step_debug["prior_valid"] = prior_valid
                if not prior_valid:
                    step_debug["prior_skip_reason"] = (
                        f"fg_ratio={stu_fg_ratio:.4f} outside [{self.fg_prior_min}, {self.fg_prior_max}]"
                    )

                # --- Teacher: 生成伪标签 (仅在先验通过时) ---
                teacher_output = None
                teacher_ori_seg = None  # 多尺度: 未 resize 的原始 logits
                if self.use_teacher and self.use_dice_loss and prior_valid:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        imgs_sel = aug_images[selected_idx]

                        use_prompt_this_step = False
                        prompt_mode = "text"
                        # 诊断信息
                        prompt_diag = {
                            "point_enabled": self.use_point_prompt,
                            "bbox_enabled": self.use_bbox_prompt,
                            "gate_passed": False,
                            "gate_reason": "prompt_disabled",
                            "fg_ratio": 0.0,
                            "fg_conf": 0.0,
                        }

                        if self.use_point_prompt:
                            # --- 点提示模式: 从学生概率图提取可靠的前景/背景点 ---
                            fg_pts, bg_pts, pt_diag = self._extract_point_prompts(
                                student_selected.detach(),
                                n_fg=self.n_fg_points,
                                n_bg=self.n_bg_points,
                                conf_thresh=self.point_conf_thresh,
                            )
                            prompt_diag.update(pt_diag)

                            if fg_pts is not None and fg_pts.size(0) > 0:
                                use_prompt_this_step = True
                                prompt_mode = "point"
                                prompt_diag["gate_passed"] = True
                                prompt_diag["gate_reason"] = "passed"

                                K_sel = imgs_sel.size(0)
                                # fg_points: (N_fg, 2) → (N_fg, K, 2)
                                fg_pts_exp = fg_pts.to(imgs_sel.device).unsqueeze(1).expand(-1, K_sel, -1)
                                if bg_pts is not None and bg_pts.size(0) > 0:
                                    bg_pts_exp = bg_pts.to(imgs_sel.device).unsqueeze(1).expand(-1, K_sel, -1)
                                else:
                                    bg_pts_exp = None

                                teacher_result = self.teacher_model.inference_with_point_prompt(
                                    imgs_sel, fg_pts_exp, bg_pts_exp,
                                    return_ori_seg=self.use_multiscale_dice)
                                if self.use_multiscale_dice:
                                    teacher_ori_list, teacher_masks = teacher_result
                                    teacher_ori_seg = torch.stack(teacher_ori_list, dim=0)
                                else:
                                    teacher_masks = teacher_result
                                teacher_output = torch.stack(teacher_masks, dim=0)
                            else:
                                prompt_diag["gate_reason"] = "no_confident_fg_points"

                        elif self.use_bbox_prompt:
                            # --- bbox prompt 模式 (保留兼容) ---
                            masks_sel = student_selected.detach()
                            binary_mask = (masks_sel > 0.5).float()
                            union_mask = binary_mask.sum(dim=0).clamp(max=1.0)
                            fg_pixels = union_mask.sum().item()
                            total_pixels = union_mask.numel()
                            fg_ratio = fg_pixels / total_pixels
                            prompt_diag["fg_ratio"] = round(fg_ratio, 6)

                            if 0.005 < fg_ratio < 0.1:
                                fg_conf = masks_sel[:, union_mask > 0.5].mean().item()
                                prompt_diag["fg_conf"] = round(fg_conf, 4)
                                if fg_conf > 0.6:
                                    ys, xs = torch.where(union_mask > 0.5)
                                    if ys.numel() > 0:
                                        use_prompt_this_step = True
                                        prompt_mode = "bbox"
                                        prompt_diag["gate_passed"] = True
                                        prompt_diag["gate_reason"] = "passed"
                                        y0, y1 = ys.min().item(), ys.max().item()
                                        x0, x1 = xs.min().item(), xs.max().item()
                                        H_mask, W_mask = union_mask.shape
                                        cx = min(max((x0 + x1) / 2.0 / W_mask, 0), 1)
                                        cy = min(max((y0 + y1) / 2.0 / H_mask, 0), 1)
                                        bw = min(max((x1 - x0) / W_mask, 0.01), 1)
                                        bh = min(max((y1 - y0) / H_mask, 0.01), 1)
                                        K_sel = imgs_sel.size(0)
                                        pos_boxes = torch.tensor(
                                            [[cx, cy, bw, bh]], device=imgs_sel.device, dtype=torch.float
                                        ).expand(K_sel, -1)
                                        teacher_result = self.teacher_model.inference_with_bbox_prompt(
                                            imgs_sel, pos_boxes, neg_boxes=None,
                                            return_ori_seg=self.use_multiscale_dice)
                                        if self.use_multiscale_dice:
                                            teacher_ori_list, teacher_masks = teacher_result
                                            teacher_ori_seg = torch.stack(teacher_ori_list, dim=0)
                                        else:
                                            teacher_masks = teacher_result
                                        teacher_output = torch.stack(teacher_masks, dim=0)

                        if not use_prompt_this_step:
                            # fallback 到纯文本模式
                            t_ori_seg_list, t_masks, _, _, _, _ = self.teacher_model(imgs_sel)
                            teacher_output = torch.stack(t_masks, dim=0)
                            if self.use_multiscale_dice:
                                teacher_ori_seg = torch.stack(t_ori_seg_list, dim=0)

                        # --- 记录 teacher 统计 ---
                        t_probs = teacher_output.detach()
                        prompt_diag["teacher_mean_prob"] = round(t_probs.mean().item(), 4)
                        prompt_diag["teacher_fg_ratio"] = round((t_probs > 0.5).float().mean().item(), 4)
                        prompt_diag["teacher_max_prob"] = round(t_probs.max().item(), 4)
                        prompt_diag["teacher_min_prob"] = round(t_probs.min().item(), 4)
                        # student vs teacher 一致性
                        s_bin = (student_selected.detach() > 0.5).float()
                        t_bin = (t_probs > 0.5).float()
                        inter = (s_bin * t_bin).sum()
                        union = s_bin.sum() + t_bin.sum()
                        st_dice = (2 * inter / (union + 1e-8)).item()
                        prompt_diag["student_teacher_dice"] = round(st_dice, 4)
                        prompt_diag["prompt_mode"] = prompt_mode

                        step_debug["prompt_diag"] = prompt_diag

                # --- 自一致性 Dice: student 自身 no_grad 推理选中视图 ---
                # 消融用途: use_teacher=0, use_prompt_memory=0, use_self_dice=1
                # 监督信号 = 当前 prompt + 选中增强视图 → no_grad forward → 伪标签
                self_teacher_output = None
                if self.use_self_dice and prior_valid:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        imgs_sel = aug_images[selected_idx]
                        _, st_masks, _, _, _, _ = self.model(imgs_sel)
                        self_teacher_output = torch.stack(st_masks, dim=0).detach()

                # --- 保存可视化 debug 图（每步都保存）---
                if self.save_debug_vis and gt_np is not None and img_np is not None:
                    with torch.no_grad():
                        # student: 取原图(idx=0)的输出
                        stu_prob_np = student_selected[0].detach().cpu().numpy()
                        # teacher: 取第一张
                        tea_prob_np = teacher_output[0].detach().cpu().numpy() if teacher_output is not None else np.zeros_like(gt_np)
                        pd = step_debug.get("prompt_diag", {})

                        # 准备点/bbox 可视化信息
                        fg_pts_vis = pd.get("fg_points_yx", None)  # list of [y, x] pixel coords
                        bg_pts_vis = pd.get("bg_points_yx", None)
                        stu_bbox = None  # bbox 模式兼容
                        if pd.get("prompt_mode") == "bbox":
                            stu_bbox = pd.get("bbox_pixel_xyxy", None)
                            if stu_bbox is not None:
                                stu_bbox = tuple(stu_bbox)

                        extra = {
                            "mode": pd.get("prompt_mode", "text"),
                            "fg_ratio": pd.get("fg_ratio", "N/A"),
                            "gate": pd.get("gate_reason", "N/A"),
                        }
                        vis_result = self._save_debug_vis(
                            img_np=img_np, gt_np=gt_np,
                            student_np=stu_prob_np, teacher_np=tea_prob_np,
                            student_bbox_xyxy=stu_bbox, gt_bbox_xyxy=gt_bbox_xyxy,
                            use_bbox=(pd.get("prompt_mode") == "bbox"),
                            case_name=case_name, step=j,
                            extra_info=extra,
                            fg_points_yx=fg_pts_vis,
                            bg_points_yx=bg_pts_vis,
                            prompt_mode=pd.get("prompt_mode", "text"),
                        )
                        step_debug["dice_student_gt"] = vis_result["dice_student_gt"]
                        step_debug["dice_teacher_gt"] = vis_result["dice_teacher_gt"]

                # --- 组装 Loss ---
                loss = torch.tensor(0.0, device=self.device)

                # Loss 1: Entropy（来自 TPT）
                if self.use_entropy_loss:
                    loss_entropy = avg_entropy(student_selected)
                    loss = loss + self.w_entropy * loss_entropy
                    step_debug["loss_entropy"] = round(loss_entropy.item(), 6)

                # Loss 2: Dice 一致性（student vs teacher，仅先验通过时）
                if self.use_dice_loss and teacher_output is not None and prior_valid:
                    loss_dice = dice_loss(student_selected, teacher_output.detach())
                    loss = loss + self.w_dice * loss_dice
                    step_debug["loss_dice"] = round(loss_dice.item(), 6)

                # Loss 2c: 自一致性 Dice（student vs student no_grad，无 EMA Teacher）
                if self.use_self_dice and self_teacher_output is not None and prior_valid:
                    loss_self_dice = dice_loss(student_selected, self_teacher_output)
                    loss = loss + self.w_dice * loss_self_dice
                    step_debug["loss_self_dice"] = round(loss_self_dice.item(), 6)

                # Loss 2b: 多尺度 Dice（ori_seg 原始分辨率上的监督）
                if (self.use_multiscale_dice and teacher_ori_seg is not None
                        and prior_valid and self.use_dice_loss):
                    # student ori_seg: 原始 logits → sigmoid
                    # 仅取 selected_idx 对应的 ori_seg
                    student_ori_selected = ori_seg[selected_idx].sigmoid()
                    teacher_ori_detached = teacher_ori_seg.detach().sigmoid()
                    # 确保尺寸一致（student 和 teacher 的 seg head 分辨率应该相同，
                    # 但以防万一做一次对齐）
                    if student_ori_selected.shape != teacher_ori_detached.shape:
                        target_h, target_w = teacher_ori_detached.shape[-2:]
                        student_ori_selected = F.interpolate(
                            student_ori_selected.unsqueeze(1),
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    loss_dice_ori = dice_loss(student_ori_selected, teacher_ori_detached)
                    loss = loss + self.w_dice_ori * loss_dice_ori
                    step_debug["loss_dice_ori"] = round(loss_dice_ori.item(), 6)
                    step_debug["ori_seg_shape"] = list(student_ori_selected.shape[-2:])

                # Loss 3: Contrast（最大化 fg-bg 图文对比度）
                if self.use_contrast_loss and contrast_scores is not None:
                    loss_contrast = -contrast_scores[selected_idx].mean()
                    # loss_contrast = -contrast_scores.mean()
                    loss = loss + self.w_contrast * loss_contrast
                    step_debug["loss_contrast"] = round(loss_contrast.item(), 6)

                step_debug["loss_total"] = round(loss.item(), 6)

                # --- 方案 C: 自适应学习率 — 在 backward 前调整 lr ---
                quality_score = None
                if self.adaptive_lr is not None or self.use_aug_param_memory or self.use_prompt_memory:
                    with torch.no_grad():
                        q_entropy = avg_entropy(student_selected.detach())
                        if contrast_scores is not None:
                            q_contrast = contrast_scores[selected_idx].mean()
                            quality_score = (q_contrast - q_entropy).item()
                        else:
                            quality_score = -q_entropy.item()

                if self.adaptive_lr is not None and quality_score is not None:
                    lr_scale = self.adaptive_lr.get_lr_scale(quality_score)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.base_lr * lr_scale
                    step_debug["lr_scale"] = round(lr_scale, 4)
                    step_debug["effective_lr"] = round(self.base_lr * lr_scale, 8)

                if(self.use_loss):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # --- 方案 A: 存入增强参数到记忆库 ---
                if self.aug_param_buffer is not None and all_aug_params is not None:
                    with torch.no_grad():
                        sel_probs = student_selected.detach()
                        per_view_entropy = -(sel_probs * torch.log(sel_probs.clamp(min=1e-8)))
                        per_view_entropy = per_view_entropy.view(sel_probs.size(0), -1).mean(dim=1)
                        if contrast_scores is not None:
                            per_view_quality = contrast_scores[selected_idx].detach() - per_view_entropy
                        else:
                            per_view_quality = -per_view_entropy
                        # 只存增强视图 (排除原图 idx=0)
                        aug_mask = selected_idx > 0
                        if aug_mask.any():
                            aug_sel_local = selected_idx[aug_mask] - 1  # all_aug_params 的 index
                            aug_sel_quality = per_view_quality[aug_mask]
                            selected_params = [all_aug_params[i.item()] for i in aug_sel_local
                                               if i.item() < len(all_aug_params)]
                            if selected_params:
                                self.aug_param_buffer.push(selected_params, aug_sel_quality[:len(selected_params)])
                    step_debug["aug_param_buffer"] = self.aug_param_buffer.stats()

                del student_selected, teacher_output, teacher_ori_seg, loss

            # --- 计算动态 EMA 动量 ---
            dynamic_momentum = None
            if self.adaptive_lr is not None and quality_score is not None:
                dynamic_momentum = self.adaptive_lr.get_momentum(quality_score)
                step_debug["dynamic_momentum"] = round(dynamic_momentum, 6)

            # --- EMA 更新 Teacher（仅先验通过时）---
            if self.use_teacher and prior_valid:
                self.update_teacher(momentum=dynamic_momentum)

            # 保存 step 日志
            self.debug_logs.append(step_debug)

        # ---------- 跨 case 信息保存 (TTA 循环结束后) ----------
        # 方案 B: 保存当前 prompt 到记忆库
        if self.prompt_memory is not None:
            with torch.no_grad():
                current_ctx = self.model.prompt_learner.ctx.data.clone()
                # 用最后一步的 quality_score
                q = quality_score if quality_score is not None else 0.0
                self.prompt_memory.push(current_ctx, q)

        # 方案 C: 更新自适应学习率的历史统计
        if self.adaptive_lr is not None:
            q = quality_score if quality_score is not None else 0.0
            self.adaptive_lr.update(q)

        # ---------- 最终推理 ----------
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                if self.infer_mode == 'selected' and selected_idx is not None:
                    # 用筛选后的增强视图推理，取均值作为最终输出
                    imgs_sel = aug_images[selected_idx]
                    _, sel_outputs, _, _, _, _ = self.model(imgs_sel)
                    sel_outputs = torch.stack(sel_outputs, dim=0)  # (K, H, W)
                    avg_pred = sel_outputs.mean(dim=0)             # (H, W)
                    tta_outputs = [avg_pred]
                else:
                    # 默认: 用原图推理
                    _, tta_outputs, _, _, _, _ = self.model(images)

        return_dict = {
            'tta_outputs': tta_outputs,
        }

        return return_dict

    def save_debug_logs(self, path=None):
        """保存所有诊断日志到 JSON + CSV 文件。在实验结束后调用。"""
        if path is None:
            path = os.path.join(self.output_dir, "prompt_debug_logs.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 汇总统计 (按 prompt 模式分组)
        prior_pass_count = sum(1 for log in self.debug_logs if log.get("prior_valid", False))
        prior_skip_count = len(self.debug_logs) - prior_pass_count
        mode_counts = {"text": 0, "point": 0, "bbox": 0}
        metrics_by_mode = {
            m: {"st_dice": [], "stu_gt_dice": [], "tea_gt_dice": []}
            for m in ["text", "point", "bbox"]
        }
        gate_reasons = {}

        for log in self.debug_logs:
            pd = log.get("prompt_diag", {})
            mode = pd.get("prompt_mode", "text")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if "student_teacher_dice" in pd:
                metrics_by_mode[mode]["st_dice"].append(pd["student_teacher_dice"])
            if "dice_student_gt" in log:
                metrics_by_mode[mode]["stu_gt_dice"].append(log["dice_student_gt"])
            if "dice_teacher_gt" in log:
                metrics_by_mode[mode]["tea_gt_dice"].append(log["dice_teacher_gt"])
            reason = pd.get("gate_reason", "N/A")
            gate_reasons[reason] = gate_reasons.get(reason, 0) + 1

        summary = {
            "total_steps": len(self.debug_logs),
            "prior_pass_count": prior_pass_count,
            "prior_skip_count": prior_skip_count,
            "mode_counts": mode_counts,
            "gate_reasons": gate_reasons,
        }
        for mode in ["text", "point", "bbox"]:
            for metric_key in ["st_dice", "stu_gt_dice", "tea_gt_dice"]:
                vals = metrics_by_mode[mode][metric_key]
                summary[f"avg_{metric_key}_{mode}"] = round(np.mean(vals), 4) if vals else None

        output = {"summary": summary, "logs": self.debug_logs}
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # 同时保存简洁 CSV（每行一个 step）
        csv_path = path.replace(".json", ".csv")
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["case", "step", "prompt_mode", "prior_valid", "gt_fg_ratio", "stu_fg_ratio",
                             "dice_student_gt", "dice_teacher_gt", "st_dice",
                             "fg_ratio", "gate_reason", "n_fg_pts", "n_bg_pts"])
            for log in self.debug_logs:
                pd = log.get("prompt_diag", {})
                writer.writerow([
                    log.get("case", ""),
                    log.get("step", ""),
                    pd.get("prompt_mode", "text"),
                    log.get("prior_valid", ""),
                    log.get("gt_fg_ratio", ""),
                    log.get("stu_selected_fg_ratio", ""),
                    log.get("dice_student_gt", ""),
                    log.get("dice_teacher_gt", ""),
                    pd.get("student_teacher_dice", ""),
                    pd.get("fg_ratio", ""),
                    pd.get("gate_reason", ""),
                    pd.get("n_fg_actual", ""),
                    pd.get("n_bg_actual", ""),
                ])

        print(f"[TPT_MT] Debug logs saved to {path}")
        print(f"[TPT_MT] Debug CSV  saved to {csv_path}")
        print(f"[TPT_MT] Summary: prior_pass={summary['prior_pass_count']}, "
              f"prior_skip={summary['prior_skip_count']}")
        print(f"[TPT_MT] Mode counts: {mode_counts}")
        for mode in ["text", "point", "bbox"]:
            if mode_counts.get(mode, 0) > 0:
                st = summary.get(f'avg_st_dice_{mode}')
                sg = summary.get(f'avg_stu_gt_dice_{mode}')
                tg = summary.get(f'avg_tea_gt_dice_{mode}')
                print(f"[TPT_MT]   {mode}: S-T={st}, Stu-GT={sg}, Tea-GT={tg}")
        print(f"[TPT_MT] Gate reasons: {gate_reasons}")
        if self.debug_logs:
            vis_dir = os.path.join(self.output_dir, "debug_vis")
            print(f"[TPT_MT] Vis images in: {vis_dir}/")


class TPT_MT_SelectionOnly():
    """
    纯筛选消融实验: 生成增强视图 → 筛选高置信样本 → 取均值作为最终输出。
    不进行任何梯度更新，仅评估筛选策略本身的效果。

    通过 --selection_strategy 控制筛选策略:
      entropy            → 仅用 entropy 排名筛选 (低 entropy = 高置信)
      contrast_entropy   → entropy + fg-bg contrast 综合排名
      variance           → 像素级 Bernoulli 方差 (低方差 = 高置信)
      consistency        → 跨视图一致性 (与均值 Dice 高 = 可信)
      average_prob       → 平均预测概率 (高平均概率 = 前景信号强)
      kl_divergence      → KL(p_i || p_mean) (低 KL = 高共识)
      no_selection       → 不筛选, 直接对所有视图取均值 (baseline)
    """

    SUPPORTED_STRATEGIES = [
        'entropy', 'contrast_entropy', 'variance',
        'consistency', 'average_prob', 'kl_divergence', 'no_selection',
    ]

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def prepare_model_and_optimization(self, args):
        self.model.eval()
        self.strategy = getattr(args, 'selection_strategy', 'entropy')
        # 兼容旧的 use_contrast_selection 开关
        self.num_aug_views = getattr(args, 'num_aug_views', 9)
        self.selection_p = getattr(args, 'selection_p', 0.1)
        print(f"[SelectionOnly] 筛选策略={self.strategy}, aug_views={self.num_aug_views}, "
              f"selection_p={self.selection_p}")

    def pre_adaptation(self, args=None):
        with torch.no_grad():
            self.model.reset()

    def _select(self, outputs, ori_seg, text_features, text_mask, vision_features):
        """根据 self.strategy 分派到不同的筛选函数。"""
        top = self.selection_p

        if self.strategy == 'entropy':
            return select_confident_samples(outputs, top)

        elif self.strategy == 'contrast_entropy':
            contrast_scores, _, _, _, _ = compute_fg_bg_similarity(
                text_features=text_features, text_mask=text_mask,
                vision_features=vision_features, ori_seg=ori_seg, fg_threshold=0.5)
            return select_confident_samples_contrast_entropy(contrast_scores, outputs, top)

        elif self.strategy == 'variance':
            return select_by_variance(outputs, top)

        elif self.strategy == 'consistency':
            return select_by_cross_view_consistency(outputs, top)

        elif self.strategy == 'average_prob':
            return select_by_average_prob(outputs, top)

        elif self.strategy == 'kl_divergence':
            return select_by_kl_divergence(outputs, top)

        elif self.strategy == 'no_selection':
            idx = torch.arange(outputs.size(0), device=outputs.device)
            return outputs, idx

        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}. "
                             f"Supported: {self.SUPPORTED_STRATEGIES}")

    def adaptation_process(self, images, args, basenames=None, gt_masks=None):
        """
        1. 生成增强视图 (原图 + N 张增强)
        2. 前向推理所有视图
        3. 仅对增强视图按指定策略筛选 top-K (排除原图, 避免所有策略都选到原图导致结果一致)
        4. 将原图预测与筛选出的增强预测加权融合作为最终分割结果
        """
        # ---------- 数据增强 ----------
        aug_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAutocontrast()
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2)
            ], p=0.5),
        ])

        augmented_images = []
        for _ in range(self.num_aug_views):
            aug_img = aug_transform(images)
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(aug_img) * 0.05
                aug_img = aug_img + noise
            aug_img = torch.clamp(aug_img, 0, 1)
            augmented_images.append(aug_img)

        aug_images = torch.cat([images] + augmented_images, dim=0)

        # ---------- 前向推理 (无梯度) ----------
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                ori_seg, outputs, text_features, text_mask, vision_features, presence_scores = self.model(aug_images)
                outputs = torch.stack(outputs, dim=0)        # (N_total, H, W)  N_total = 1 + num_aug_views
                ori_seg = torch.stack(ori_seg, dim=0)


                if self.strategy == 'no_selection':
                    # 不筛选: 使用全部增强视图
                    selected = outputs
                else:
                    selected, idx = self._select(
                        outputs, ori_seg,
                        text_features, text_mask, vision_features)

                # ---------- 融合: 原图 + 筛选出的增强视图均值 ----------
                aug_mean = selected.mean(dim=0, keepdim=True)  # (1, H, W)
                

        return_dict = {
            'tta_outputs': [aug_mean.squeeze(0)],  # 与其他 algorithm 返回格式一致
        }
        return return_dict
