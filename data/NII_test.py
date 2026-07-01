import os
import cv2
import random
import glob
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage.draw import polygon2mask

from data.datautils import AugMixAugmenter
from data.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_, decode_mask


import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2
import pandas as pd

def expand_bbox(x, y, w, h, width, height, expand_ratio=0.25):
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    
    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)
    
    return new_x, new_y, new_x2, new_y2

class NIIDataset_Promise(Dataset):
    def __init__(self, cfg, root_dir, list_file, transform=None, if_self_training=False):
        self.cfg = cfg
        df = pd.read_csv(os.path.join(list_file), encoding='gbk')
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()
        self.root_dir = root_dir
        self.transform = transform
        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # Get image name and path
        name = self.name_list[idx]
        basename = name.split('/')[1] + '-' + name.split('/')[2]
        image_path = os.path.join(self.root_dir, name)

        img_nii = nib.load(image_path)
        image = img_nii.get_fdata().transpose(2,0,1)  # Shape could be (H, W) or (D, H, W)

        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
        elif image.ndim == 3:
            image = image[0, :, :] 
            image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
        
        image = (image-image.min())/(image.max()-image.min())
        image = (image*255).astype(np.uint8)
        height, width, _ = image.shape
        # If configuration has 'get_prompt', return additional image information
        # if self.cfg.get_prompt:
        #     image_info = {}
        #     height, width, _ = image.shape
        #     image_info["file_path"] = image_path
        #     image_info["height"] = height
        #     image_info["width"] = width
        #     return idx, image_info, image

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)

        gt_nii = nib.load(gt_path)
        gt_mask = gt_nii.get_fdata().transpose(2,0,1)[0]  # Shape could be (H, W) or (D, H, W)
        gt_mask = (gt_mask*255).astype(np.uint8)

        masks = []
        bboxes = []
        bboxes_corse = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            masks.append(mask)
            x, y, w, h = cv2.boundingRect(mask)
            # bboxes.append([x, y, x + w, y + h])
            new_x, new_y, new_x2, new_y2 = expand_bbox(x, y, w, h, width, height, expand_ratio=0.0)
            new_x_1, new_y_1, new_x2_1, new_y2_1 = expand_bbox(x, y, w, h, width, height, expand_ratio=0.0)
            bboxes.append([new_x, new_y, new_x2, new_y2])
            bboxes_corse.append([new_x_1, new_y_1, new_x2_1, new_y2_1])
            # bboxes_corse = bboxes
            categories.append("0")

        # If self-training, apply transformations (augmented images)
        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)
                ###############
                # print(image.shape,bboxes,'111111', bboxes_corse)
                image_coarse = image
                masks_coarse = masks
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
                _, _, bboxes_corse = self.transform(image_coarse, masks_coarse, np.array(bboxes_corse))
                # print(image_coarse.shape, bboxes,'222222', bboxes_corse, self.transform)
            
            bboxes = np.stack(bboxes, axis=0)
            bboxes_corse = np.stack(bboxes_corse, axis=0)

            masks = np.stack(masks, axis=0)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            # return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()
            
            return image, torch.tensor(bboxes),torch.tensor(bboxes_corse), torch.tensor(masks).float(), basename, image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        # elif self.cfg.visual:
        #     file_name = os.path.splitext(os.path.basename(name))[0]
        #     origin_image = image
        #     origin_bboxes = bboxes
        #     origin_masks = masks
        #     if self.transform:
        #         padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

        #     bboxes = np.stack(bboxes, axis=0)
        #     masks = np.stack(masks, axis=0)
        #     origin_bboxes = np.stack(origin_bboxes, axis=0)
        #     origin_masks = np.stack(origin_masks, axis=0)
        #     return file_name, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()
        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float(), basename, img_nii
        
def load_datasets_promise(cfg, img_size, augmix=False, n_views=2):
    base_transform = ResizeAndPad(img_size)
    root_dir = cfg.data_dir
    all_list = os.path.join(root_dir,'all.csv')
    if(augmix):
        data_transform = AugMixAugmenter(base_transform, n_views=n_views)
    else:
        data_transform = base_transform
    all = NIIDataset_Promise(
        cfg,
        root_dir=root_dir,
        list_file=all_list,
        transform=data_transform,
        # if_self_training=True,
    )

    all_dataloader = DataLoader(
        all,
        batch_size=cfg.batch_size,
        shuffle=False, # 先把一个病人的slice读完再进行下一个
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    
    return all_dataloader


class NIIDataset_Promise_Aug(NIIDataset_Promise):
    """
    NIIDataset_Promise 的增强版本：在 __getitem__ 中预生成 num_aug_views 个增强视图。
    增强在 DataLoader worker 中并行执行，从而将 CPU 增强开销从 adaptation_process 中解耦出来。

    返回格式（在 else 分支，即非 if_self_training 模式）:
        image       : Tensor (C, H, W)              — 原始图像 (经过 ResizeAndPad)
        bboxes      : Tensor (N, 4)
        masks       : Tensor (N, H, W)
        basename    : str
        img_nii     : nibabel.Nifti1Image
        aug_images  : Tensor (num_aug_views, C, H, W) — 预生成的增强视图 (不含原图)
    """

    # 与 tpt_mt.py 中 adaptation_process 保持一致的增强变换
    _aug_transform = None  # 懒加载，每个 worker 各自初始化

    def __init__(self, cfg, root_dir, list_file, transform=None,
                 if_self_training=False, num_aug_views=9):
        super().__init__(cfg, root_dir, list_file, transform, if_self_training)
        self.num_aug_views = num_aug_views

    @staticmethod
    def _build_aug_transform():
        import torchvision.transforms as T
        return T.Compose([
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.RandomApply([T.RandomAutocontrast()], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
        ])

    def __getitem__(self, idx):
        # 调用父类获取原始样本
        result = super().__getitem__(idx)

        if self.if_self_training:
            # self_training 模式不走增强 dataloader 路径，直接透传
            return result

        # result = (image, bboxes, masks, basename, img_nii)
        image, bboxes, masks, basename, img_nii = result

        # 懒加载增强变换（在 worker 进程中初始化，避免跨进程 pickle 问题）
        aug_transform = self._build_aug_transform()

        # image 已经是 Tensor (C, H, W)，值域 [0, 1]（由 ResizeAndPad 保证）
        aug_views = []
        for _ in range(self.num_aug_views):
            aug_img = aug_transform(image)
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(aug_img) * 0.05
                aug_img = aug_img + noise
            aug_img = torch.clamp(aug_img, 0, 1)
            aug_views.append(aug_img)

        aug_images = torch.stack(aug_views, dim=0)  # (num_aug_views, C, H, W)

        return image, bboxes, masks, basename, img_nii, aug_images


def collate_fn_aug(batch):
    """
    针对 NIIDataset_Promise_Aug 的 collate_fn。
    batch 中每条样本: (image, bboxes, masks, basename, img_nii, aug_images)
    返回:
        images     : Tensor (B, C, H, W)
        bboxes     : list[Tensor]
        masks      : list[Tensor]
        basenames  : list[str]
        img_niis   : list[nib.Nifti1Image]
        aug_images : Tensor (B, num_aug_views, C, H, W)
    """
    images, bboxes, masks, basenames, img_niis, aug_images = zip(*batch)
    images = torch.stack(images, dim=0)
    aug_images = torch.stack(aug_images, dim=0)
    return images, list(bboxes), list(masks), list(basenames), list(img_niis), aug_images


def load_datasets_promise_aug(cfg, img_size, num_aug_views=9):
    """
    加载带有预生成增强视图的 Promise 数据集（供 tpt_mt_2 使用）。

    DataLoader 返回元组：
        (images, bboxes, masks, basenames, img_niis, aug_images)
    其中 aug_images 形状为 (B, num_aug_views, C, H, W)。
    """
    base_transform = ResizeAndPad(img_size)
    root_dir = cfg.data_dir
    all_list = os.path.join(root_dir, 'all.csv')

    dataset = NIIDataset_Promise_Aug(
        cfg,
        root_dir=root_dir,
        list_file=all_list,
        transform=base_transform,
        num_aug_views=num_aug_views,
    )

    all_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_aug,
        pin_memory=True,
    )

    return all_dataloader


class ISIC2018Dataset(Dataset):
    """
    ISIC2018 皮肤病变分割数据集
    图像格式: ISIC_XXXXXXX.jpg
    掩码格式: ISIC_XXXXXXX_segmentation.png
    """
    def __init__(self, cfg, root_dir, list_file=None, transform=None, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.if_self_training = if_self_training
        
        if list_file is not None and os.path.exists(list_file):
            # 从 CSV 文件读取图像列表
            df = pd.read_csv(list_file, encoding='utf-8')
            self.image_list = df.iloc[:, 0].tolist()
            if df.shape[1] > 1:
                self.label_list = df.iloc[:, 1].tolist()
            else:
                # 自动推断标签路径
                self.label_list = [img.replace('.jpg', '_segmentation.png') for img in self.image_list]
        else:
            # 自动扫描目录中的所有图像
            all_files = os.listdir(root_dir)
            self.image_list = sorted([f for f in all_files if f.endswith('.jpg') and not f.startswith('.')])
            self.label_list = [img.replace('.jpg', '_segmentation.png') for img in self.image_list]
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        from PIL import Image as PILImage
        
        # 获取图像名称和路径
        image_name = self.image_list[idx]
        basename = os.path.splitext(image_name)[0]  # e.g., ISIC_0024306
        
        # 使用 PIL 读取图像 (更稳定，避免 libpng 问题)
        image_path = os.path.join(self.root_dir, image_name)
        try:
            pil_image = PILImage.open(image_path).convert('RGB')
            image = np.array(pil_image)
        except Exception as e:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}, 错误: {e}")
        height, width, _ = image.shape
        
        # 加载分割掩码 (使用 PIL)
        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
        try:
            pil_mask = PILImage.open(gt_path).convert('L')
            gt_mask = np.array(pil_mask)
        except Exception as e:
            raise FileNotFoundError(f"无法读取掩码文件: {gt_path}，错误: {e}")
        
        # 二值化掩码 (前景=1, 背景=0)，转换为 0-255 格式
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        
        masks = []
        bboxes = []
        bboxes_corse = []
        categories = []
        
        # 解码掩码（可能包含多个连通区域）
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        
        for mask in gt_masks:
            masks.append(mask)
            x, y, w, h = cv2.boundingRect(mask)
            new_x, new_y, new_x2, new_y2 = expand_bbox(x, y, w, h, width, height, expand_ratio=0.0)
            new_x_1, new_y_1, new_x2_1, new_y2_1 = expand_bbox(x, y, w, h, width, height, expand_ratio=0.0)
            bboxes.append([new_x, new_y, new_x2, new_y2])
            bboxes_corse.append([new_x_1, new_y_1, new_x2_1, new_y2_1])
            categories.append("0")
        
        # 如果是自训练模式，应用增强变换
        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            
            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)
                
                image_coarse = image
                masks_coarse = masks
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
                _, _, bboxes_corse = self.transform(image_coarse, masks_coarse, np.array(bboxes_corse))
            
            bboxes = np.stack(bboxes, axis=0)
            bboxes_corse = np.stack(bboxes_corse, axis=0)
            masks = np.stack(masks, axis=0)
            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            
            return image, torch.tensor(bboxes), torch.tensor(bboxes_corse), torch.tensor(masks).float(), basename, image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()
        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            # 返回格式与 NIIDataset_Promise 类似，但不包含 img_nii（因为不是 NIfTI 格式）
            return image, torch.tensor(bboxes), torch.tensor(masks).float(), basename, None


def load_datasets_isic2018(cfg, img_size, augmix=False, n_views=2):
    """
    加载 ISIC2018 皮肤病变分割数据集
    
    Args:
        cfg: 配置对象，需要包含 data_dir 和 batch_size 属性
        img_size: 图像大小
        augmix: 是否使用 AugMix 数据增强
        n_views: AugMix 视图数量
    
    Returns:
        all_dataloader: 数据加载器
    """
    base_transform = ResizeAndPad(img_size)
    root_dir = cfg.data_dir
    
    # 检查是否存在 CSV 文件
    all_list = os.path.join(root_dir, 'all.csv')
    if not os.path.exists(all_list):
        all_list = None  # 将自动扫描目录
    
    if augmix:
        data_transform = AugMixAugmenter(base_transform, n_views=n_views)
    else:
        data_transform = base_transform
    
    all_dataset = ISIC2018Dataset(
        cfg,
        root_dir=root_dir,
        list_file=all_list,
        transform=data_transform,
    )
    
    all_dataloader = DataLoader(
        all_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    
    return all_dataloader
