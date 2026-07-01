# CM-TTA

This repository contains the code for our paper:

**Concept Alignment Contrast and Long-Short Prompt Memory for Test-Time Adaptation of SAM3 in Medical Image Segmentation**

Paper: https://arxiv.org/abs/2606.22963

CM-TTA is a test-time adaptation method for SAM3 on medical image segmentation tasks. The main implementation is in `instance_method/tpt_mt.py`; `run_cmtta.py` is the entry script I use for evaluation.

## What is Included

```text
CM-TTA/
  run_cmtta.py
  data/
  instance_method/tpt_mt.py
  sam3/
  utils/
  requirements.txt
```

The released code includes:

- concept alignment contrast for choosing reliable augmented views;
- long-short prompt memory for online test-time adaptation;
- dense pseudo-label supervision from teacher predictions;
- loaders for Promise and ISIC2018.

Large files are not included. Please put the SAM3 checkpoint here:

```text
checkpoints/sam3.pt
```

You can also use a custom path:

```bash
export SAM3_CHECKPOINT=/path/to/sam3.pt
```

The default BPE vocabulary is already included in `sam3/assets/`. If needed:

```bash
export SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
```

## Environment

```bash
cd CM-TTA
conda create -n cmtta python=3.12 -y
conda activate cmtta
pip install -r requirements.txt
```

The code was developed with CUDA. Running CM-TTA on CPU is not recommended.

## Data

For Promise, `--data_dir` should contain `all.csv`. The first column is the image path and the second column is the mask path. Both paths are relative to `--data_dir`.

```text
Promise/
  all.csv
  ...
```

For ISIC2018, either provide `all.csv`, or put images and masks in the same folder. The loader expects image names like `ISIC_0000000.jpg` and masks like `ISIC_0000000_segmentation.png`.

```text
ISIC2018/
  ISIC_0000000.jpg
  ISIC_0000000_segmentation.png
```

## Evaluation

Promise:

```bash
python run_cmtta.py \
  --dataset promise \
  --data_dir /path/to/Promise \
  --gpu 0
```

ISIC2018:

```bash
python run_cmtta.py \
  --dataset isic2018 \
  --data_dir /path/to/ISIC2018 \
  --gpu 0
```

Some useful options:

```bash
--tta_steps 1
--num_aug_views 9
--online 1
--use_prompt_memory 1
--use_contrast_selection 1
--use_contrast_loss 1
--use_dice_loss 1
--use_entropy_loss 1
```

Results are saved under:

```text
results/<dataset>_seed_<seed>/<ctx_init>/<exm_suffix>/
```

The folder contains `final-results.csv`, `log.txt`, predicted masks, and debug logs.

## Citation

```bibtex
@article{zhou2026cmtta,
  title={Concept Alignment Contrast and Long-Short Prompt Memory for Test-Time Adaptation of SAM3 in Medical Image Segmentation},
  author={Zhou, Yubo and Wu, Jianghao and Ye, Ping and Zhang, Shaoting and Wang, Guotai},
  journal={arXiv preprint arXiv:2606.22963},
  year={2026}
}
```
