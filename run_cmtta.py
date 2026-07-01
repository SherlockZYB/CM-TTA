import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

from data.NII_test import (
    load_datasets_isic2018,
    load_datasets_promise,
)
from method.tpt_mt import TPT_MT
from sam3.custom_sam3 import get_coop
from utils.tools import (
    AverageMeter,
    ProgressMeter,
    Summary,
    calculate_metrics,
    resize_and_save,
    set_random_seed,
)


CLASS_NAMES = {
    "promise": ["prostate"],
    "isic2018": ["skin lesion"],
}


def build_loader(args):
    if args.dataset == "promise":
        return load_datasets_promise(args, args.resolution, augmix=False, n_views=args.aug_views)
    if args.dataset == "isic2018":
        return load_datasets_isic2018(args, args.resolution, augmix=False, n_views=args.aug_views)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def save_prediction(pred_mask_np, save_path, img_nii):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if img_nii is not None:
        resize_and_save(pred_mask_np, save_path, img_nii)
        return

    if pred_mask_np.ndim == 3:
        pred_mask_np = np.squeeze(pred_mask_np)
    pred_uint8 = (pred_mask_np > 0.5).astype(np.uint8) * 255
    if not save_path.endswith(".png"):
        save_path = os.path.splitext(save_path)[0] + ".png"
    Image.fromarray(pred_uint8).save(save_path)


def print_args(args):
    rows = ["=========================================="]
    for key, value in sorted(vars(args).items()):
        rows.append(f"{key}: {value}")
    return "\n".join(rows)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    cudnn.benchmark = True

    output_root = Path(args.output_dir)
    args.output_dir = str(output_root / f"{args.dataset}_seed_{args.seed}" / args.ctx_init / args.exm_suffix)
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "log.txt")
    args.out_file = open(log_path, "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()

    if not torch.cuda.is_available():
        raise RuntimeError("CM-TTA requires CUDA for SAM3 inference/adaptation.")
    torch.cuda.set_device(args.gpu)

    data_loader = build_loader(args)
    classnames = CLASS_NAMES[args.dataset]

    model = get_coop(args.gpu, classnames, args.batch_size, args.n_ctx, args.ctx_init)
    model = model.cuda(args.gpu)

    cmtta = TPT_MT(model, args.gpu)
    cmtta.prepare_model_and_optimization(args)
    cmtta.model.eval()

    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    adapt_time = AverageMeter("Adapt", ":6.3f", Summary.NONE)
    dice_scores = AverageMeter("Dice")
    assd_scores = AverageMeter("ASSD")
    hd95_scores = AverageMeter("HD95")
    progress = ProgressMeter(len(data_loader), [data_time, adapt_time], prefix="Test: ")

    case_results = []
    end = time.time()
    for i, batch in enumerate(data_loader):
        images, bboxes, gt_masks, basenames, img_niis = batch
        images = images.to(args.gpu, non_blocking=True)
        data_end = time.time()

        cmtta.pre_adaptation(args)
        result = cmtta.adaptation_process(images, args, basenames=basenames, gt_masks=gt_masks)
        pred_masks = result["tta_outputs"]
        adapt_end = time.time()

        for pred_mask, gt_mask, basename, img_nii in zip(pred_masks, gt_masks, basenames, img_niis):
            pred_mask = (pred_mask > 0.5).float()
            gt_mask = (gt_mask > 0.5).float()
            dice, assd, hd95 = calculate_metrics(pred_mask, gt_mask)

            dice_scores.update(dice, args.batch_size)
            assd_scores.update(assd, args.batch_size)
            hd95_scores.update(hd95, args.batch_size)

            suffix = ".nii.gz" if img_nii is not None else ".png"
            save_path = os.path.join(args.output_dir, "segs", f"{basename}{suffix}")
            save_prediction(pred_mask.squeeze().detach().cpu().numpy(), save_path, img_nii)

            case_results.append({
                "basename": basename,
                "Dice": dice,
                "ASSD": assd,
                "HD95": hd95,
            })

        data_time.update(data_end - end)
        adapt_time.update(adapt_end - data_end)

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(data_loader):
            line = (
                f"iter:{i + 1}/{len(data_loader)}, data={data_time.avg:.4f}, "
                f"adapt={adapt_time.avg:.4f}, dice={dice_scores.avg:.4f}, "
                f"hd95={hd95_scores.avg:.4f}, assd={assd_scores.avg:.4f}"
            )
            print(line)
            args.out_file.write(line + "\n")
            args.out_file.flush()
            progress.display(i)

        end = time.time()

    if hasattr(cmtta, "save_debug_logs"):
        cmtta.save_debug_logs()

    avg_row = {
        "basename": "Average",
        "Dice": dice_scores.avg,
        "ASSD": assd_scores.avg,
        "HD95": hd95_scores.avg,
    }
    pd.DataFrame([avg_row] + case_results).to_csv(
        os.path.join(args.output_dir, "final-results.csv"), index=False)

    summary = f"=> Results: Dice {dice_scores.avg} / HD95 {hd95_scores.avg} / ASSD {assd_scores.avg}"
    print(summary)
    args.out_file.write(summary + "\n")
    args.out_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="CM-TTA for SAM3 medical image segmentation")
    parser.add_argument("--dataset", default="promise", choices=["promise", "isic2018"])
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--resolution", default=1008, type=int)
    parser.add_argument("--aug_views", default=15, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=1377, type=int)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--exm_suffix", default="cmtta", type=str)

    parser.add_argument("--n_ctx", default=1, type=int)
    parser.add_argument("--ctx_init", default="The", type=str)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--selection_p", default=0.1, type=float)
    parser.add_argument("--tta_steps", default=1, type=int)

    parser.add_argument("--ema_momentum", default=0.99, type=float)
    parser.add_argument("--w_contrast", default=1.0, type=float)
    parser.add_argument("--w_entropy", default=0.1, type=float)
    parser.add_argument("--w_dice", default=1.0, type=float)

    parser.add_argument("--use_contrast_selection", default=1, type=int)
    parser.add_argument("--use_contrast_loss", default=1, type=int)
    parser.add_argument("--use_dice_loss", default=1, type=int)
    parser.add_argument("--use_entropy_loss", default=1, type=int)
    parser.add_argument("--use_teacher", default=1, type=int)

    parser.add_argument("--online", default=1, type=int)
    parser.add_argument("--use_prompt_memory", default=1, type=int)
    parser.add_argument("--prompt_memory_size", default=16, type=int)
    parser.add_argument("--prompt_fusion_momentum", default=0.3, type=float)

    parser.add_argument("--use_point_prompt", default=1, type=int)
    parser.add_argument("--n_fg_points", default=3, type=int)
    parser.add_argument("--n_bg_points", default=3, type=int)
    parser.add_argument("--point_conf_thresh", default=0.7, type=float)
    parser.add_argument("--use_bbox_prompt", default=0, type=int)

    parser.add_argument("--use_multiscale_dice", default=0, type=int)
    parser.add_argument("--w_dice_ori", default=0.5, type=float)
    parser.add_argument("--fg_prior_min", default=0.01, type=float)
    parser.add_argument("--fg_prior_max", default=0.10, type=float)
    parser.add_argument("--num_aug_views", default=9, type=int)
    parser.add_argument("--infer_mode", default="original", choices=["original", "selected"])
    parser.add_argument("--save_debug_vis", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
