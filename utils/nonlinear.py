import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import comb  # 用于组合数计算
from PIL import Image

class LearnableBezierTransform(nn.Module):
    def __init__(self, num_control_points=4):
        """
        初始化 LearnableBezierTransform 模块。
        
        Args:
            num_control_points (int): 贝塞尔曲线的控制点数量，默认是4（用于三次贝塞尔曲线）。
        """
        super(LearnableBezierTransform, self).__init__()
        self.num_control_points = num_control_points
        
        # 初始化三组控制点，每组控制点对应一次三次贝塞尔曲线变换
        # 控制点的 y 坐标在 [0,1] 范围内，确保映射的连续性
        # 使用 sigmoid 激活确保控制点始终在 [0,1] 范围内
        # 下述初始值表示大致的线性映射，可根据需要进行微调
        self.control_points_1 = nn.Parameter(torch.tensor([
            0.0,
            0.33,
            0.66,
            1.0
        ], dtype=torch.float32))
        
        self.control_points_2 = nn.Parameter(torch.tensor([
            0.0,
            0.2,
            0.8,
            1.0
        ], dtype=torch.float32))

        self.control_points_3 = nn.Parameter(torch.tensor([
            0.0,
            0.25,
            0.75,
            1.0
        ], dtype=torch.float32))
        self.control_points_11 = nn.Parameter(torch.tensor([
            0.0,
            0.33,
            0.66,
            1.0
        ], dtype=torch.float32))
        
        self.control_points_22 = nn.Parameter(torch.tensor([
            0.0,
            0.2,
            0.8,
            1.0
        ], dtype=torch.float32))

        self.control_points_33 = nn.Parameter(torch.tensor([
            0.0,
            0.25,
            0.75,
            1.0
        ], dtype=torch.float32))

    def forward(self, x, visual=False):
        """
        前向传播函数。

        输入:
            x (torch.Tensor): 输入图像，形状为 [1, H, W] 或 [3, H, W]，值范围 [0, 1]。
        
        输出:
            torch.Tensor: 
                - 若输入为单通道 [1, H, W]，则输出为 [3, H, W] 的三通道图像；
                - 若输入为三通道 [3, H, W]，则输出也为 [3, H, W]（分别对每个通道变换）。
        """
        assert x.dim() == 3, "输入张量应为 3 维，形状为 [C, H, W]"
        # print(x.shape,'108')
        # print(x[0].sum(),x[1].sum(),x[2].sum(),'109')
        c, h, w = x.shape
        assert c in [1, 3], "仅支持单通道 [1, H, W] 或三通道 [3, H, W] 输入"

        # 使用 sigmoid 确保控制点在 [0,1] 范围内
        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]
        # print(cp1)
        # print(cp2)
        # print(cp3)

        if c == 1:
            # 原逻辑：单通道输入 -> 三条曲线 -> 拼接为三通道输出
            f1 = self.bezier_curve(cp1, x)  # [1, H, W]
            f2 = self.bezier_curve(cp2, x)  # [1, H, W]
            f3 = self.bezier_curve(cp3, x)  # [1, H, W]
            
            output = torch.cat([f1, f2, f3], dim=0)  # [3, H, W]

            # 可视化保存：原图 + 3 条通道变换
            # self.save_image_single_channel(x[0], f1[0], f2[0], f3[0], 'transformed_single')
            # self.save_image(x[0], f1[0], f2[0], f3[0], base_name)
            if visual:
                combined_output = self.visualize(x, output)
            else:
                combined_output = None
            return output, combined_output  # 返回两次变换的结果，便于后续处理

        else:  # c == 3
            # 三通道输入：分别对三个通道使用不同的 Bezier 曲线
            # 例如：通道0 -> cp1, 通道1 -> cp2, 通道2 -> cp3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]
            
            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]
            combined_output = self.visualize(x, output)
            # self.save_image(x[0], out_c0[0], out_c1[0], out_c2[0], base_name)

            # 可视化保存：原图的三通道 + 变换后的三通道
            # self.save_image_three_channel(x, output, 'transformed_rgb')
            return output, combined_output  # 返回两次变换的结果，便于后续处理
    def forward_2(self, x):
        """
        前向传播函数。

        输入:
            x (torch.Tensor): 输入图像，形状为 [1, H, W] 或 [3, H, W]，值范围 [0, 1]。
        
        输出:
            torch.Tensor: 
                - 若输入为单通道 [1, H, W]，则输出为 [3, H, W] 的三通道图像；
                - 若输入为三通道 [3, H, W]，则输出也为 [3, H, W]（分别对每个通道变换）。
        """
        assert x.dim() == 3, "输入张量应为 3 维，形状为 [C, H, W]"
        c, h, w = x.shape
        assert c in [1, 3], "仅支持单通道 [1, H, W] 或三通道 [3, H, W] 输入"

        # 使用 sigmoid 确保控制点在 [0,1] 范围内
        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]
        cp11 = torch.sigmoid(self.control_points_11)  # [4]
        cp22 = torch.sigmoid(self.control_points_22)  # [4]
        cp33 = torch.sigmoid(self.control_points_33)  # [4]

        if c == 1:
            # 原逻辑：单通道输入 -> 三条曲线 -> 拼接为三通道输出
            f1 = self.bezier_curve(cp1, x)  # [1, H, W]
            f2 = self.bezier_curve(cp2, x)  # [1, H, W]
            f3 = self.bezier_curve(cp3, x)  # [1, H, W]
            f11 = self.bezier_curve(cp11, x)  # [1, H, W]
            f22 = self.bezier_curve(cp22, x)  # [1, H, W]
            f33 = self.bezier_curve(cp33, x)  # [1, H, W]
            
            output = torch.cat([f1, f2, f3], dim=0)  # [3, H, W]
            output_2 = torch.cat([f11, f22, f33], dim=0)  # [3, H, W]

            # 可视化保存：原图 + 3 条通道变换
            # self.save_image_single_channel(x[0], f1[0], f2[0], f3[0], 'transformed_single')
            return output, output_2
        
        else:  # c == 3
            # 三通道输入：分别对三个通道使用不同的 Bezier 曲线
            # 例如：通道0 -> cp1, 通道1 -> cp2, 通道2 -> cp3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]
            f11 = self.bezier_curve(cp11, x[0:1])  # [1, H, W]
            f22 = self.bezier_curve(cp22, x[1:2])  # [1, H, W]
            f33 = self.bezier_curve(cp33, x[2:3])  # [1, H, W]
            
            
            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]
            output_2 = torch.cat([f11, f22, f33], dim=0)  # [3, H, W]

            # 可视化保存：原图的三通道 + 变换后的三通道
            # self.save_image_three_channel(x, output, 'transformed_rgb')
            return output, output_2
    
    
    def bezier_curve(self, control_points, x):
        """
        计算贝塞尔曲线的非线性变换。

        Args:
            control_points (torch.Tensor): 形状为 [4] 的一维控制点 (P0, P1, P2, P3)。
            x (torch.Tensor): 输入图像，形状为 [1, H, W]，值范围 [0,1]。
        
        Returns:
            torch.Tensor: 变换后的图像，形状为 [1, H, W]。
        """
        # 控制点 (P0, P1, P2, P3) 都是标量，表示 y 坐标
        P0, P1, P2, P3 = control_points
        
        # 定义 t 和 (1 - t)
        t = x                 # [1, H, W]
        one_minus_t = 1 - t   # [1, H, W]
        
        # 三次贝塞尔权重
        B0 = comb(3, 0) * (one_minus_t ** 3)      # [1, H, W]
        B1 = comb(3, 1) * (t ** 1) * (one_minus_t ** 2)
        B2 = comb(3, 2) * (t ** 2) * (one_minus_t ** 1)
        B3 = comb(3, 3) * (t ** 3) * (one_minus_t ** 0)
        
        # 计算像素映射值
        f_t = B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3
        
        # 限制在 [0,1] 范围
        f_t = torch.clamp(f_t, 0.0, 1.0)
        
        return f_t  # [1, H, W]
    
    def save_image(self, img, img1, img2, img3, filename):
        """
        保存 3 张灰度图为横向拼接的长图。

        Args:
            img1, img2, img3 (torch.Tensor): 分别为 [H, W] 的灰度图。
            filename (str): 保存文件名（不包含后缀 .png）。
        """
        # 转为 CPU 上的 NumPy 数组
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        img3_np = img3.detach().cpu().numpy()
        img_np = img.detach().cpu().numpy()

        # 规范化到 [0,255]
        img1_np = self.normalize_image(img1_np)
        img2_np = self.normalize_image(img2_np)
        img3_np = self.normalize_image(img3_np)
        img_np = self.normalize_image(img_np)

        # 横向拼接 => [H, 3*W]
        combined = np.concatenate([img_np, img1_np, img2_np, img3_np], axis=1)

        # 转为单通道 PIL 图像
        image = Image.fromarray(combined, mode='L')
        image.save(f"/home/data/SAM/wesam/show_4img/{filename}.png")
        print(f"Image saved as '{filename}.png'")

    @staticmethod
    def normalize_image(img):
        """
        将图像规范化到 [0,255] 并转换为 uint8。

        Args:
            img (np.ndarray): 输入图像 (H, W)。
        
        Returns:
            np.ndarray: 规范化并转换后的图像 (H, W)，dtype=uint8。
        """
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        return (img * 255).astype(np.uint8)
    
    def visualize(self, gray, out):
        """
        gray : torch.Tensor  [1,1,H,W]  或 [C,H,W]
        out  : torch.Tensor  [1,N,H,W]  或 [N,H,W]
        返回 : np.uint8  [H, W*(N+2), 3]
        """
        # -------- 0) 去掉 batch 维 --------
        if gray.dim() == 4:         # [B,C,H,W]
            gray = gray.squeeze(0)  # -> [C,H,W]
        if out.dim() == 4:          # [B,N,H,W]
            out = out.squeeze(0)    # -> [N,H,W]

        # ---------- 1) 工具函数 ----------
        def to_rgb(img2d):
            return np.stack([img2d]*3, axis=-1)    # [H,W,3]

        def norm(img):
            img = img.astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
            else:
                img = np.zeros_like(img)
            return (img*255).astype(np.uint8)
            # ---------- 2) 原图 ----------
        gray_np = gray.detach().cpu().numpy()      # [C,H,W]
        if gray_np.shape[0] == 1:
            base = norm(gray_np[0])                # [H,W] uint8
            base = to_rgb(base)                    # [H,W,3]
        else:
            chans = [norm(c) for c in gray_np]     # 每通道独立归一化
            base = np.stack(chans, axis=-1)        # [H,W,3]
        canvases = [base]

        # ---------- 3) Bézier 输出各灰度 ----------
        out_np = out.detach().cpu().numpy()        # [N,H,W]
        for i in range(out_np.shape[0]):
            canvases.append(to_rgb(norm(out_np[i])))

        # ---------- 4) 合成 RGB 彩色图 (最多取前 3 通道) ----------
        if out_np.shape[0] >= 3:
            rgb_comp = np.stack([norm(out_np[j]) for j in range(3)], axis=-1)
        else:  # 不足 3 通道时循环复用最后一通道
            last = out_np[-1]
            chans = [norm(out_np[j] if j < out_np.shape[0] else last)
                     for j in range(3)]
            rgb_comp = np.stack(chans, axis=-1)
        canvases.append(rgb_comp)

        # ---------- 5) 横向拼接并返回 ----------
        combined_output = np.concatenate(canvases, axis=1)  # [H, W*(N+2), 3]
        return combined_output




import torch
import torch.nn as nn
import numpy as np
from math import comb
from typing import Optional


class RandomBezierTransform(nn.Module):
    """
    随机 Bézier 灰度曲线变换（不学习，用于消融）
    - 输入:  [C,H,W], C ∈ {1,3}, 值域 [0,1]
    - 输出:  [3,H,W] （单通道会扩成三通道；三通道分别随机变换后再拼成3通道）
    - 每次 forward 随机采样控制点，可设置 seed 保持可复现
    """
    def __init__(
        self,
        num_control_points: int = 4,       # 三次 Bézier -> 4 点
        monotonic: bool = True,            # 是否强制单调（建议 True 以避免色带/反转）
        strength: float = 0.25,            # 随机扰动幅度 0~1
        mode: str = "medium",              # "mild" (0.15), "medium"(0.25), "strong"(0.4)
        share_across_rgb: bool = False,    # 对 RGB 是否共享一套曲线
        seed = None       # 设定后可复现；None 表示每次不同
    ):
        super().__init__()
        assert num_control_points == 4, "本版仅支持三次 Bézier（4 控制点）"
        self.n = num_control_points
        self.monotonic = monotonic
        self.share = share_across_rgb
        if mode == "mild":   strength = 0.15
        if mode == "strong": strength = 0.40
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.base_cp = torch.tensor([0.0, 1/3, 2/3, 1.0])  # 近似线性起点
        self.fixed_rng = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            self.fixed_rng = g

    @torch.no_grad()
    def _sample_cp(self, H: int = 0, W: int = 0, device=None):
        """
        采样一组控制点 y 值（长度=4，范围[0,1]，首尾固定 0/1）
        - monotonic=True: 中间两点按强度在基线附近扰动，并排序保证递增
        - monotonic=False: 自由扰动并裁剪
        """
        g = self.fixed_rng
        noise = torch.empty(2, device=device).uniform_(-self.strength, self.strength, generator=g)
        mid = self.base_cp[1:3].to(device) + noise
        if self.monotonic:
            mid = torch.sort(torch.clamp(mid, 0.0, 1.0)).values
            cp = torch.tensor([0.0, mid[0].item(), mid[1].item(), 1.0], device=device)
        else:
            cp = torch.tensor([0.0, mid[0].clamp(0,1).item(), mid[1].clamp(0,1).item(), 1.0], device=device)
        return cp

    @staticmethod
    def _bezier_eval(cp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        cp: [4] (P0..P3), x: [1,H,W] ∈ [0,1]
        返回: [1,H,W]
        """
        P0, P1, P2, P3 = cp
        t = x
        omt = 1.0 - t
        # 三次 Bézier 基函数
        B0 = comb(3,0) * (omt ** 3)
        B1 = comb(3,1) * (t   ) * (omt ** 2)
        B2 = comb(3,2) * (t**2) * (omt)
        B3 = comb(3,3) * (t**3)
        y = B0*P0 + B1*P1 + B2*P2 + B3*P3
        return torch.clamp(y, 0.0, 1.0)

    @staticmethod
    def _to_rgb_strip(arr_list):
        """把若干 [H,W] uint8 横向拼接为一张 [H, W*, 3]（可选用于可视化）"""
        import numpy as np
        imgs = []
        for a in arr_list:
            a = a.astype(np.float32)
            mn, mx = a.min(), a.max()
            if mx > mn: a = (a - mn) / (mx - mn)
            else:       a = np.zeros_like(a)
            a = (a*255).astype(np.uint8)
            imgs.append(np.stack([a,a,a], axis=-1))
        return np.concatenate(imgs, axis=1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, visual: bool = True):
        """
        x: [C,H,W], C∈{1,3}, 值域[0,1]
        return:
          output: [3,H,W]  (float, [0,1])
          viz   : 可视化条（np.uint8，[H, W*(N+1), 3]），若 visual=False 则为 None
        """
        assert x.dim() == 3 and x.size(0) in (1,3), "输入需为 [C,H,W]，C∈{1,3}"
        C, H, W = x.shape
        device = x.device

        # 采样三条曲线用于输出 3 个通道
        if C == 1:
            cp_list = [self._sample_cp(H, W, device) for _ in range(3)]
            outs = [self._bezier_eval(cp_list[i], x) for i in range(3)]  # 每条用同一灰度
        else:
            # 3 通道输入：可选择共享或各通道独立采样
            if self.share:
                cp = self._sample_cp(H, W, device)
                outs = [self._bezier_eval(cp, x[i:i+1]) for i in range(3)]
            else:
                outs = [self._bezier_eval(self._sample_cp(H, W, device), x[i:i+1]) for i in range(3)]

        out = torch.cat(outs, dim=0)  # [3,H,W]

        if not visual:
            return out, None

        # 生成简单可视化（原图+三通道变换）
        x_np = x.detach().cpu().numpy()
        out_np = out.detach().cpu().numpy()
        if C == 1:
            base = x_np[0]
        else:
            # RGB 合成展示：把原 RGB 各通道独立归一化再叠
            base = np.stack([
                ((x_np[i] - x_np[i].min()) / (np.ptp(x_np[i]) + 1e-6)) if np.ptp(x_np[i]) > 0
                else np.zeros_like(x_np[i])
                for i in range(3)
            ], axis=0).mean(0)
        strip = self._to_rgb_strip([base, out_np[0], out_np[1], out_np[2]])
        return out, strip
