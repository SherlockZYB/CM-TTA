import os
import time
import random

import numpy as np

import shutil
from enum import Enum
from medpy import metric
from typing import Tuple
import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device != 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg

def calculate_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    # Convert to numpy and binarize
    pred_mask_np = (pred_mask.cpu().numpy() > threshold).astype(np.bool_)
    gt_mask_np = (gt_mask.cpu().numpy() > threshold).astype(np.bool_)

    # Remove singleton dimensions
    pred_mask_np = np.squeeze(pred_mask_np)
    gt_mask_np = np.squeeze(gt_mask_np)

    # Align shapes when one has an extra leading channel dim
    if pred_mask_np.shape != gt_mask_np.shape:
        if pred_mask_np.shape == gt_mask_np.shape[1:]:
            gt_mask_np = np.squeeze(gt_mask_np, axis=0)
        elif gt_mask_np.shape == pred_mask_np.shape[1:]:
            pred_mask_np = np.squeeze(pred_mask_np, axis=0)

    # If still mismatched, return safe defaults to avoid metric crash
    if pred_mask_np.shape != gt_mask_np.shape:
        print("Shape mismatch between prediction and ground truth masks")
        return 0.0, 50.0, 50.0

    # 处理两者都为空的情况
    if np.sum(pred_mask_np) == 0 and np.sum(gt_mask_np) == 0:
        return 100.0, 0.0, 0.0

    # 处理两者都有内容的情况
    elif np.sum(pred_mask_np) > 0 and np.sum(gt_mask_np) > 0:
        batch_dice = metric.binary.dc(pred_mask_np, gt_mask_np)
        # batch_asd = min(metric.binary.asd(pred_mask_np, gt_mask_np), 50.0)
        batch_asd = metric.binary.asd(pred_mask_np, gt_mask_np)
        # batch_asd = 50
        batch_hd95 = metric.binary.hd95(pred_mask_np, gt_mask_np)
        # batch_hd95 = 50
        return batch_dice, batch_asd, batch_hd95
        # return batch_dice, 50, 50

    # 其中之一为空
    else:
        return metric.binary.dc(pred_mask_np, gt_mask_np), 50.0, 50.0
    
import SimpleITK as sitk
import numpy as np

import SimpleITK as sitk
import numpy as np

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
import torch
import warnings
warnings.filterwarnings('ignore')

def resize_and_save(pred_mask_np, save_path, img_nii, interpolation='nearest'):
    """
    将网络输出的掩码调整到原始图像尺寸并保存为NIfTI文件
    
    Args:
        pred_mask_np: 网络输出的掩码，形状为(1, 1008, 1008)或(1008, 1008)
        save_path: 保存路径
        img_nii: 原始的NIfTI图像对象（已用nibabel加载）
        interpolation: 插值方法，可选'nearest'（最近邻，适用于二值掩码）、
                      'linear'、'cubic'等
    """
    
    # 处理维度：去掉batch维度
    if pred_mask_np.ndim == 3 and pred_mask_np.shape[0] == 1:
        pred_mask = pred_mask_np[0]  # 形状变为(1008, 1008)
    elif pred_mask_np.ndim == 2:
        pred_mask = pred_mask_np
    else:
        raise ValueError(f"输入掩码的形状应为(1, H, W)或(H, W)，但得到{pred_mask_np.shape}")
    
    # 2. 获取原始图像信息
    img_data = img_nii.get_fdata()
    original_shape = img_data.shape
    affine = img_nii.affine
    header = img_nii.header
    
    # 3. 确定目标形状
    # 如果原始图像是3D，我们假设要调整到前两个维度
    if len(original_shape) == 3:
        # 原始是3D，目标调整到前两个维度，保持第三维为1
        target_shape_2d = original_shape[:2]
        target_shape_3d = (target_shape_2d[0], target_shape_2d[1], 1)
    elif len(original_shape) == 2:
        # 原始是2D
        target_shape_2d = original_shape
        target_shape_3d = original_shape
    else:
        raise ValueError(f"不支持的图像维度: {len(original_shape)}")
    
    # 4. 调整掩码大小
    # 选择合适的插值方法
    if interpolation == 'nearest':
        order = 0
    elif interpolation == 'linear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    else:
        order = 0
    
    # 使用scipy的zoom函数进行插值
    # 计算缩放因子
    zoom_factors = [target_shape_2d[0] / pred_mask.shape[0], 
                   target_shape_2d[1] / pred_mask.shape[1]]
    

    # 应用缩放
    resized_mask = ndimage.zoom(pred_mask, zoom_factors, order=order, 
                                mode='constant', cval=0.0)
    
    # 5. 创建3D NIfTI图像
    if len(target_shape_3d) == 3:
        # 如果原始是3D，创建一个3D数组
        final_mask_3d = np.zeros(target_shape_3d, dtype=resized_mask.dtype)
        if target_shape_3d[2] == 1:
            # 单切片
            final_mask_3d[:, :, 0] = resized_mask
        else:
            # 多切片，可以放在中间切片
            middle_slice = target_shape_3d[2] // 2
            final_mask_3d[:, :, middle_slice] = resized_mask
    else:
        # 如果原始是2D
        final_mask_3d = resized_mask

    
    # 6. 创建NIfTI图像对象
    # 使用原始图像的仿射矩阵和头部信息
    nii_mask = nib.Nifti1Image(final_mask_3d, affine=affine, header=header.copy())
    
    # 7. 设置数据类型
    # 根据掩码值的范围设置合适的数据类型
    # if np.issubdtype(resized_mask.dtype, np.integer):
    #     # 如果是整数类型，保持原样
    #     pass
    # else:
    #     # 如果是浮点数，可能需要转换为合适的类型
    #     if resized_mask.max() <= 1.0 and resized_mask.min() >= 0.0:
    #         # 概率值，转换为uint8
    #         nii_mask = nib.Nifti1Image((final_mask_3d * 255).astype(np.uint8), 
    #                                  affine=affine, header=header.copy())
    #     else:
    #         # 转换为float32
    #         nii_mask = nib.Nifti1Image(final_mask_3d.astype(np.float32), 
    #                                  affine=affine, header=header.copy())
    
    nib.save(nii_mask, save_path)
    
    return nii_mask, resized_mask
