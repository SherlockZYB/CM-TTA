
from typing import Tuple
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
from sam3.model.data_misc import FindStage, interpolate
from sam3.model.tokenizer_ve import SimpleTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend

import copy
import math


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BPE_PATH = Path(__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
_DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints" / "sam3.pt"


def _sam3_bpe_path():
    return os.environ.get("SAM3_BPE_PATH", str(_DEFAULT_BPE_PATH))


def _sam3_checkpoint_path():
    return os.environ.get("SAM3_CHECKPOINT", str(_DEFAULT_CHECKPOINT_PATH))


def _build_cm_sam3_model(bpe_path):
    return build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=_sam3_checkpoint_path(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=False,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

class SAM3Tuning(nn.Module):
    def __init__(self, device, classnames, batch_size,
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(SAM3Tuning, self).__init__()
        bpe_path = _sam3_bpe_path()
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=_sam3_checkpoint_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=False,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # We use text prompts, not point/box
            compile=False,
        )
        self.class_names = classnames
        
        print("Initializing the contect with given words: [{}]".format(ctx_init))
        ctx_init = ctx_init.replace("_", " ")
        ctx_init = ctx_init.replace("{", "[")
        ctx_init = ctx_init.replace("}", "]")
        if '[CLS]' in ctx_init:
            self.prompts = ctx_init.replace('[CLS]', classnames[0])
        else:
            self.prompts = ctx_init + ' ' + classnames[0]
            if(ctx_init=='a'):
                self.prompts = 'a shoe'  # debug code
                print("debug prompt:", self.prompts)
        print("Using prompt:", self.prompts)
        # By default, freeze everything, 这里不存储SAM3的梯度信息，来节省显存
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        self.device = device
        # prompt tuning
        self.ini_model_state = copy.deepcopy(self.model.state_dict())
        
        

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.model.load_state_dict(self.ini_model_state, strict=True)
        

    def forward(self, images):

        """
        Forward pass for SAM3 with text prompts.
        
        Args:
            images: Tensor of shape [B, C, H, W]
            prompts: List of text prompts (noun phrases), one per image in batch
        
        Returns:
            image_embeddings: Backbone features (for compatibility)
            pred_masks: List of predicted masks, one per image
            ious: List of IoU predictions (dummy for now, SAM3 doesn't output per-mask IoU)
            res_masks: List of low-res masks
        """
        batch_size = images.size(0)
        device = images.device
        
        
        # if len(self.prompts) != batch_size:
        #     if len(self.prompts) == 1:
        #         self.prompts = self.prompts * batch_size
        # self.model.backbone.forward_text(prompts, device=device)
        
        # Get image embeddings from backbone
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))
        
        # Get text embeddings
        text_outputs = self.model.backbone.forward_text(self.prompts, device=device)
        backbone_out.update(text_outputs)
        
        
        def _slice_batch(obj, idx):
            """Slice batch dimension for tensors/lists/dicts to build per-image inputs."""
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx : idx + 1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = []
                for v in obj:
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
                        sliced.append(v[idx : idx + 1])
                    else:
                        sliced.append(v)
                return type(obj)(sliced)
            return obj

        # Forward through the model
        pred_masks = []
        ious = []
        res_masks = []
        image_embeddings = []
        
        # Process each image separately (SAM3 expects batched but we handle one at a time for compatibility)
        for i in range(batch_size):
            # Slice backbone/text outputs to a single-item batch and reset ids to 0
            single_backbone_out = _slice_batch(backbone_out, i)
            # Create empty geometric prompt for this image (we're using text only)
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device=device),
                box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            )
            # Forward grounding
            # For text-only prompts, we always use eval mode to avoid _compute_matching
            # (which requires find_target, but we don't have labels)
            was_training = self.model.training
            self.model.eval()  # Always use eval mode for text-only inference
            
            try:
                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,  # No target - we're using text-only prompts
                    geometric_prompt=geometric_prompt,
                )
            finally:
                # Restore training mode if it was training (for gradient computation)
                if was_training:
                    self.model.train()
            
            _, orig_h, orig_w = images[i].shape
            seg = output["semantic_seg"]  # Shape: [1, num_queries, H, W] or [1, num_queries, 1, H, ]
            if seg.dim() > 2:
                seg = seg.squeeze()
            if seg.shape != (orig_h, orig_w):
                    seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
            
            image_embeddings.append(seg)
            pred_masks.append(seg.sigmoid())
            
            # Extract outputs
            # pred_masks: [1, num_queries, H, W] or [1, num_queries, 1, H, W]
            # out_masks = output["pred_masks"]  # Shape: [1, num_queries, H, W] or [1, num_queries, 1, H, W]
            # out_logits = output["pred_logits"]  # Shape: [1, num_queries, 1]
            # out_probs = out_logits.sigmoid()  # [1, num_queries, 1]
            
            # # Apply presence score (like official Sam3Processor)
            # # This helps filter out low-quality predictions
            # if "presence_logit_dec" in output:
            #     presence_score = output["presence_logit_dec"].sigmoid().unsqueeze(1)  # [1, 1, 1]
            #     out_probs = (out_probs * presence_score).squeeze(-1)  # [1, num_queries]
            # else:
            #     out_probs = out_probs.squeeze(-1)  # [1, num_queries]
            
            # # Get the best mask (highest score)
            # if out_probs.size(1) > 0 and out_probs.max() > 0:
            #     best_idx = out_probs.argmax(1).item()
            #     best_mask = out_masks[0, best_idx]  # [H, W] or [1, H, W]
            #     if best_mask.dim() == 3:
            #         best_mask = best_mask.squeeze(0)
                
            #     # Get original image size for this sample
            #     _, orig_h, orig_w = images[i].shape
                
            #     # Interpolate to original size if needed
            #     if best_mask.shape != (orig_h, orig_w):
            #         best_mask = interpolate(
            #             best_mask.unsqueeze(0).unsqueeze(0),
            #             (orig_h, orig_w),
            #             mode="bilinear",
            #             align_corners=False,
            #         ).squeeze(0).squeeze(0)
                
            #     # Apply sigmoid (masks are already logits)
            #     best_mask = best_mask.sigmoid()
            #     pred_masks.append(best_mask)
                
            #     # Dummy IoU (SAM3 doesn't provide per-mask IoU in the same way)
            #     ious.append(out_probs[0, best_idx].unsqueeze(0))
                
            #     # Low-res mask (before interpolation) - store original shape
            #     low_res_mask = out_masks[0, best_idx]
            #     if low_res_mask.dim() == 3:
            #         low_res_mask = low_res_mask.squeeze(0)
            #     res_masks.append(low_res_mask)
            # else:
            #     # No predictions or all scores are 0
            #     _, _, H, W = images.shape
            #     dummy_mask = torch.zeros(H, W, device=device)
            #     pred_masks.append(dummy_mask)
            #     ious.append(torch.tensor(0.0, device=device))
            #     res_masks.append(torch.zeros(64, 64, device=device))
            
            # Store backbone features for compatibility
            # sam2_out = single_backbone_out.get("sam2_backbone_out")
            # if sam2_out is not None and isinstance(sam2_out, dict):
            #     # Use SAM2 backbone features if available
            #     fpn_feats = sam2_out.get("backbone_fpn", [])
            #     if isinstance(fpn_feats, (list, tuple)) and len(fpn_feats) > 0:
            #         image_embeddings.append(fpn_feats[-1][0])  # Remove batch dim
            #     else:
            #         image_embeddings.append(torch.zeros(256, 64, 64, device=device))
            # else:
            #     # Fallback: use visual features or create dummy
            #     if "visual_features" in single_backbone_out:
            #         feat = single_backbone_out["visual_features"]
            #         if feat.dim() == 4 and feat.size(0) == 1:
            #             image_embeddings.append(feat[0])
            #         else:
            #             image_embeddings.append(feat)
            #     else:
            #         # Create dummy embedding
            #         image_embeddings.append(torch.zeros(256, 64, 64, device=device))
        
        return image_embeddings, pred_masks
        
    def forward_features(self, images, prompts):
        # image_features = self.model.backbone.forward_image(images)["backbone_fpn"]
        # text_features = self.model.backbone.forward_text(prompts, device=images.device)["language_features"]
        image_features = self.model.backbone.forward_image(images)
        text_features = self.model.backbone.forward_text(prompts, device=images.device)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, text_features


def get_sam3(device, classnames, batch_size, n_ctx, ctx_init, learned_cls=False):

    model = SAM3Tuning(device, classnames, batch_size=batch_size, n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model



class PromptLearner(nn.Module):
    def __init__(self, sam3_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, bpe_path=None):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.weight.dtype
        self.dtype = dtype
        self.device = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.weight.device
        ctx_dim = sam3_model.backbone.language_backbone.encoder.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.bpe_path = bpe_path
        tokenizer = SimpleTokenizer(bpe_path,context_length=32) #根据SAM3的VETextEncoder的tokenizer设置context_length=32

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)
        
        
        if ctx_init:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            ctx_init = ctx_init.replace("{", "[")
            ctx_init = ctx_init.replace("}", "]")
            if ' [CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
                self.split_idx = split_idx
            elif '[CLS]' in ctx_init:
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "front"
            
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenizer(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = sam3_model.backbone.language_backbone.encoder.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(self.batch_size, 1, 1)  #(N, L, D)
            
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name for name in classnames]
            if(prompt_prefix=='a'):
                prompts = ['a shoe']
                print("debug prompt:", prompts)
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized
            
        

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]).to(self.device)
        self.text_attention_mask = (tokenized_prompts != 0).bool().ne(1)
        with torch.no_grad():
            embedding = sam3_model.backbone.language_backbone.encoder.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)
            
    def reset_classnames(self, classnames, arch):
        tokenizer = SimpleTokenizer(self.bpe_path)
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]).to(self.device)

        sam3 = build_sam3_image_model(
            bpe_path=self.bpe_path,
            checkpoint_path=_sam3_checkpoint_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=False,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # We use text prompts, not point/box
            compile=False,
        )

        with torch.no_grad():
            embedding = sam3.backbone.language_backbone.encoder.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames
        
    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        
        # ！！！这个条件是为了在batch_size=1且n_cls=1时，扩展成(n_cls, n_ctx, dim)的形式，方便后续拼接
        if(ctx.size()[0] == 1 and self.n_cls == 1):
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:,i : i + 1, :, :] #!!! 注意，因为我多了batch维度，所以前面加了个:，front和end同理
                class_i = suffix[:,i : i + 1, :name_len, :]
                suffix_i = suffix[:,i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[:,i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[:,i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=-2,  #!!! 注意，因为我多了batch维度，所以这里dim要设置为-2，front和end同理
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:,i : i + 1, :, :]
                class_i = suffix[:,i : i + 1, :name_len, :]
                suffix_i = suffix[:,i : i + 1, name_len:, :]
                ctx_i = ctx[:,i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class SAM3TestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size,
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(SAM3TestTimeTuning, self).__init__()
        bpe_path = _sam3_bpe_path()
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=_sam3_checkpoint_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=False,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # We use text prompts, not point/box
            compile=False,
        )
        # By default, freeze everything, 这里不存储SAM3的梯度信息，来节省显存
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        self.device = device
        # prompt tuning
        self.prompt_learner = PromptLearner(self.model, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, bpe_path)
        # self.ini_model_state = copy.deepcopy(self.model.state_dict())

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()
        # self.model.load_state_dict(self.ini_model_state, strict=True)
        

    def inference(self, images):

        """
        Forward pass for SAM3 with text prompts.
        
        Args:
            images: Tensor of shape [B, C, H, W]
            prompts: List of text prompts (noun phrases), one per image in batch
        
        Returns:
            image_embeddings: Backbone features (for compatibility)
            pred_masks: List of predicted masks, one per image
            ious: List of IoU predictions (dummy for now, SAM3 doesn't output per-mask IoU)
            res_masks: List of low-res masks
        """
        batch_size = images.size(0)
        device = images.device
        input_embeds = self.prompt_learner() #(batch_size, 1， seq_len(32), dim(1024))
        input_embeds = input_embeds.squeeze(1) #(batch_size, seq_len(32), dim(1024)) SAM3只能分单类，所以就不管n_cls的信息了
        
        # 这里是针对TPT这样的方法，由于输入多个增强图像的缘故，images的batch_size可能会发生变化
        if(input_embeds.shape[0]!=batch_size):
            input_embeds = input_embeds.repeat(int(batch_size/input_embeds.shape[0]),1,1)
        # self.model.backbone.forward_text(prompts, device=device)
        
        # Get image embeddings from backbone
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))
        
        # Get text embeddings
        text_outputs = {}
        
        text_attention_mask = self.prompt_learner.text_attention_mask
        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )
        with sdpa_context:
            _,text_memory = self.model.backbone.language_backbone.encoder(input_embeds)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = text_memory.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        text_embeds = input_embeds.transpose(0, 1)
        
        text_memory = text_memory_resized[:, : batch_size]
        text_attention_mask = text_attention_mask[: batch_size]
        text_embeds = text_embeds[:, : batch_size]
        text_outputs["language_features"] = text_memory
        text_outputs["language_mask"] = text_attention_mask
        text_outputs["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )
        backbone_out.update(text_outputs)
        
        
        def _slice_batch(obj, idx):
            """Slice batch dimension for tensors/lists/dicts to build per-image inputs."""
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx : idx + 1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = []
                for v in obj:
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
                        sliced.append(v[idx : idx + 1])
                    else:
                        sliced.append(v)
                return type(obj)(sliced)
            return obj

        # Forward through the model
        pred_masks = []
        pred_logits = []
        presence_scores = []    
        
        # Process each image separately (SAM3 expects batched but we handle one at a time for compatibility)
        for i in range(images.size(0)):
            # Slice backbone/text outputs to a single-item batch and reset ids to 0
            single_backbone_out = _slice_batch(backbone_out, i)
            # Create empty geometric prompt for this image (we're using text only)
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device=device),
                box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            )
            # Forward grounding
            # For text-only prompts, we always use eval mode to avoid _compute_matching
            # (which requires find_target, but we don't have labels)
            was_training = self.model.training
            self.model.eval()  # Always use eval mode for text-only inference
            
            try:
                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,  # No target - we're using text-only prompts
                    geometric_prompt=geometric_prompt,
                )
            finally:
                # Restore training mode if it was training (for gradient computation)
                if was_training:
                    self.model.train()
            
            _, orig_h, orig_w = images[i].shape
            seg = output["semantic_seg"]  # Shape: [1, num_queries, H, W] or [1, num_queries, 1, H, ]
            if seg.dim() > 2:
                seg = seg.squeeze()
            if seg.shape != (orig_h, orig_w):
                    resized_seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
            
            pred_logits.append(seg)
            pred_masks.append(resized_seg.sigmoid())
            
            out_logits = output["pred_logits"]  # Shape: [1, num_queries, 1]
            out_probs = out_logits.sigmoid()  # [1, num_queries, 1]
            presence_score = output["presence_logit_dec"].sigmoid().unsqueeze(1)  # [1, 1, 1]
            scores = (out_probs * presence_score).squeeze(-1)  # [1, num_queries]
            presence_scores.append(scores[0])
            
        
        return pred_logits, pred_masks, backbone_out['language_features'], backbone_out['language_mask'], backbone_out['vision_features'],presence_scores


    def forward(self, input):
        return self.inference(input)

    @torch.no_grad()
    def inference_with_bbox_prompt(self, images, pos_boxes, neg_boxes=None, return_ori_seg=False):
        """
        使用 bbox 作为 geometric prompt 来引导分割（用于 teacher 生成伪标签）。

        仍然使用 prompt_learner 生成文本特征，但额外传入正/负 bounding-box
        作为 geometric prompt，帮助模型定位前景/背景。

        Args:
            images:    (B, C, H, W) 输入图像
            pos_boxes: (B, 4)  正样本 bbox，格式 [cx, cy, w, h] 归一化到 [0,1]
                       如果只有 1 个 box 但 B>1，会自动广播
            neg_boxes: (B_neg, 4) 或 None  负样本 bbox（可选），同样 [cx, cy, w, h] 归一化
                       如果为 None，则不使用负 box
            return_ori_seg: 是否同时返回未 resize 的原始 logits

        Returns:
            如果 return_ori_seg=False:
                pred_masks: list of (H, W) sigmoid 概率图，长度 = B
            如果 return_ori_seg=True:
                (pred_logits, pred_masks):
                    pred_logits: list of (H_seg, W_seg) 原始 logits
                    pred_masks:  list of (H, W) sigmoid 概率图
        """
        batch_size = images.size(0)
        device = images.device

        # ---------- 1. 构建 backbone_out（图像 + 文本） ----------
        input_embeds = self.prompt_learner()          # (batch, 1, seq_len, dim)
        input_embeds = input_embeds.squeeze(1)        # (batch, seq_len, dim)
        if input_embeds.shape[0] != batch_size:
            input_embeds = input_embeds.repeat(
                int(batch_size / input_embeds.shape[0]), 1, 1)

        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))

        # 文本 —— 走 prompt_learner 路线（与 inference 一致）
        text_outputs = {}
        text_attention_mask = self.prompt_learner.text_attention_mask
        sdpa_context = sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
        )
        with sdpa_context:
            _, text_memory = self.model.backbone.language_backbone.encoder(input_embeds)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        text_embeds = input_embeds.transpose(0, 1)

        text_memory = text_memory_resized[:, :batch_size]
        text_attention_mask = text_attention_mask[:batch_size]
        text_embeds = text_embeds[:, :batch_size]
        text_outputs["language_features"] = text_memory
        text_outputs["language_mask"] = text_attention_mask
        text_outputs["language_embeds"] = text_embeds
        backbone_out.update(text_outputs)

        # ---------- 2. 组装 geometric prompt ----------
        # pos_boxes: (B, 4) → (1, B, 4) 即 N_boxes=1, B=batch
        if pos_boxes.dim() == 1:
            pos_boxes = pos_boxes.unsqueeze(0)          # (1, 4)
        if pos_boxes.dim() == 2:
            pos_boxes = pos_boxes.unsqueeze(0)          # (1, B, 4)
        # 如果只有 1 份 box 但 B>1，广播
        if pos_boxes.size(1) == 1 and batch_size > 1:
            pos_boxes = pos_boxes.expand(-1, batch_size, -1)

        n_pos = pos_boxes.size(0)  # 正 box 数量（通常 1）

        if neg_boxes is not None:
            if neg_boxes.dim() == 1:
                neg_boxes = neg_boxes.unsqueeze(0)
            if neg_boxes.dim() == 2:
                neg_boxes = neg_boxes.unsqueeze(0)      # (N_neg, 1, 4) or (N_neg, B, 4)
            if neg_boxes.size(1) == 1 and batch_size > 1:
                neg_boxes = neg_boxes.expand(-1, batch_size, -1)
            n_neg = neg_boxes.size(0)
            all_boxes = torch.cat([pos_boxes, neg_boxes], dim=0)  # (N_pos+N_neg, B, 4)
            all_labels = torch.cat([
                torch.ones(n_pos, batch_size, device=device, dtype=torch.long),
                torch.zeros(n_neg, batch_size, device=device, dtype=torch.long),
            ], dim=0)
        else:
            all_boxes = pos_boxes                                  # (N_pos, B, 4)
            all_labels = torch.ones(n_pos, batch_size, device=device, dtype=torch.long)

        n_total = all_boxes.size(0)
        box_mask = torch.zeros(batch_size, n_total, device=device, dtype=torch.bool)

        # ---------- 3. 逐图推理 ----------
        def _slice_batch(obj, idx):
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx:idx+1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = [
                    v[idx:idx+1] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size else v
                    for v in obj
                ]
                return type(obj)(sliced)
            return obj

        pred_masks = []
        pred_logits = []
        was_training = self.model.training
        self.model.eval()
        try:
            for i in range(batch_size):
                single_backbone_out = _slice_batch(backbone_out, i)
                # boxes for this image: (N_total, 1, 4)
                geo_boxes = all_boxes[:, i:i+1, :]
                geo_labels = all_labels[:, i:i+1]
                geo_mask = box_mask[i:i+1, :]       # (1, N_total)

                geometric_prompt = Prompt(
                    box_embeddings=geo_boxes,   # (N_total, 1, 4)
                    box_mask=geo_mask,          # (1, N_total)
                    box_labels=geo_labels,      # (N_total, 1)
                )

                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,
                    geometric_prompt=geometric_prompt,
                )

                _, orig_h, orig_w = images[i].shape
                seg = output["semantic_seg"]
                if seg.dim() > 2:
                    seg = seg.squeeze()
                # 保存原始 logit（未 resize、未 sigmoid）
                pred_logits.append(seg)
                if seg.shape != (orig_h, orig_w):
                    resized_seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                else:
                    resized_seg = seg
                pred_masks.append(resized_seg.sigmoid())
        finally:
            if was_training:
                self.model.train()

        if return_ori_seg:
            return pred_logits, pred_masks
        return pred_masks

    @torch.no_grad()
    def inference_with_point_prompt(self, images, fg_points, bg_points=None, return_ori_seg=False):
        """
        使用点提示（前景点 + 背景点）作为 geometric prompt 来引导分割。

        仍然使用 prompt_learner 生成文本特征，但额外传入前景/背景点
        作为 geometric prompt，帮助模型定位前景/背景。

        Args:
            images:    (B, C, H, W) 输入图像
            fg_points: (N_fg, B, 2) 前景点，归一化坐标 [x, y] ∈ [0,1]
                       或 (N_fg, 2) 如果所有图像共享相同点（会自动广播）
            bg_points: (N_bg, B, 2) 或 (N_bg, 2) 或 None
                       背景点，归一化坐标 [x, y] ∈ [0,1]
            return_ori_seg: 是否同时返回未 resize 的原始 logits

        Returns:
            如果 return_ori_seg=False:
                pred_masks: list of (H, W) sigmoid 概率图，长度 = B
            如果 return_ori_seg=True:
                (pred_logits, pred_masks):
                    pred_logits: list of (H_seg, W_seg) 原始 logits（未 resize、未 sigmoid）
                    pred_masks:  list of (H, W) sigmoid 概率图
        """
        batch_size = images.size(0)
        device = images.device

        # ---------- 1. 构建 backbone_out（图像 + 文本） ----------
        input_embeds = self.prompt_learner()
        input_embeds = input_embeds.squeeze(1)
        if input_embeds.shape[0] != batch_size:
            input_embeds = input_embeds.repeat(
                int(batch_size / input_embeds.shape[0]), 1, 1)

        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))

        text_outputs = {}
        text_attention_mask = self.prompt_learner.text_attention_mask
        sdpa_context = sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
        )
        with sdpa_context:
            _, text_memory = self.model.backbone.language_backbone.encoder(input_embeds)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        text_embeds = input_embeds.transpose(0, 1)

        text_memory = text_memory_resized[:, :batch_size]
        text_attention_mask = text_attention_mask[:batch_size]
        text_embeds = text_embeds[:, :batch_size]
        text_outputs["language_features"] = text_memory
        text_outputs["language_mask"] = text_attention_mask
        text_outputs["language_embeds"] = text_embeds
        backbone_out.update(text_outputs)

        # ---------- 2. 组装点 prompt ----------
        # fg_points 归一化: (N_fg, B, 2) 或 (N_fg, 2)
        if fg_points.dim() == 2:
            fg_points = fg_points.unsqueeze(1).expand(-1, batch_size, -1)  # (N_fg, B, 2)
        n_fg = fg_points.size(0)

        if bg_points is not None:
            if bg_points.dim() == 2:
                bg_points = bg_points.unsqueeze(1).expand(-1, batch_size, -1)
            n_bg = bg_points.size(0)
            all_points = torch.cat([fg_points, bg_points], dim=0)       # (N_fg+N_bg, B, 2)
            all_labels = torch.cat([
                torch.ones(n_fg, batch_size, device=device, dtype=torch.long),
                torch.zeros(n_bg, batch_size, device=device, dtype=torch.long),
            ], dim=0)
        else:
            all_points = fg_points
            all_labels = torch.ones(n_fg, batch_size, device=device, dtype=torch.long)

        n_total = all_points.size(0)
        point_mask = torch.zeros(batch_size, n_total, device=device, dtype=torch.bool)

        # ---------- 3. 逐图推理 ----------
        def _slice_batch(obj, idx):
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx:idx+1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = [
                    v[idx:idx+1] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size else v
                    for v in obj
                ]
                return type(obj)(sliced)
            return obj

        pred_masks = []
        pred_logits = []
        was_training = self.model.training
        self.model.eval()
        try:
            for i in range(batch_size):
                single_backbone_out = _slice_batch(backbone_out, i)
                geo_points = all_points[:, i:i+1, :]    # (N_total, 1, 2)
                geo_labels = all_labels[:, i:i+1]       # (N_total, 1)
                geo_mask = point_mask[i:i+1, :]          # (1, N_total)

                geometric_prompt = Prompt(
                    point_embeddings=geo_points,    # (N_total, 1, 2)
                    point_mask=geo_mask,            # (1, N_total)
                    point_labels=geo_labels,        # (N_total, 1)
                )

                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,
                    geometric_prompt=geometric_prompt,
                )

                _, orig_h, orig_w = images[i].shape
                seg = output["semantic_seg"]
                if seg.dim() > 2:
                    seg = seg.squeeze()
                # 保存原始 logit（未 resize、未 sigmoid）
                pred_logits.append(seg)
                if seg.shape != (orig_h, orig_w):
                    resized_seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                else:
                    resized_seg = seg
                pred_masks.append(resized_seg.sigmoid())
        finally:
            if was_training:
                self.model.train()

        if return_ori_seg:
            return pred_logits, pred_masks
        return pred_masks

    def forward_features(self, images, prompts):
        # image_features = self.model.backbone.forward_image(images)["backbone_fpn"]
        # text_features = self.model.backbone.forward_text(prompts, device=images.device)["language_features"]
        image_features = self.model.backbone.forward_image(images)
        text_features = self.model.backbone.forward_text(prompts, device=images.device)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, text_features


def get_coop(device, classnames, batch_size, n_ctx, ctx_init, learned_cls=False):

    model = SAM3TestTimeTuning(device, classnames, batch_size=batch_size, n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model


##################
class _LoRA_qkv(nn.Module):
    """LoRA wrapper for qkv layer in SAM3 ViT.
    In SAM3 ViT, it's implemented as:
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        # x shape: [B, L, C] or [B, H, W, C]; only the last dim matters here.
        qkv = self.qkv(x)  # [..., 3*C]

        new_q = self.linear_b_q(self.linear_a_q(x))  # [..., C]
        new_v = self.linear_b_v(self.linear_a_v(x))  # [..., C]

        # Slice on the last dimension to avoid assuming a flattened sequence shape.
        q = qkv[..., : self.dim] + new_q
        k = qkv[..., self.dim : 2 * self.dim]
        v = qkv[..., 2 * self.dim :] + new_v
        return torch.cat([q, k, v], dim=-1)
    
class SAM3LoraTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(SAM3LoraTestTimeTuning, self).__init__()
        bpe_path = _sam3_bpe_path()
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=_sam3_checkpoint_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=False,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # We use text prompts, not point/box
            compile=False,
        )
        # By default, freeze everything, 这里不存储SAM3的梯度信息，来节省显存
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.prompt_learner = PromptLearner(self.model, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, bpe_path)
        dtype = self.model.backbone.vision_backbone.trunk.patch_embed.proj.weight.dtype
        self.dtype = dtype
        self.w_As = []  # LoRA A matrices
        self.w_Bs = []  # LoRA B matrices

    def add_lora(self, layer_indices=[9,10,11], r=16, lora_alpha=32, lora_dropout=0.05):
        if layer_indices is None:
            layer_indices = list(range(len(self.model.backbone.vision_backbone.trunk.blocks)))
        print(f"Applying LoRA to SAM3 visual backbone layers: {layer_indices}")
        for t_layer_i, blk in enumerate(self.model.backbone.vision_backbone.trunk.blocks):
            if t_layer_i not in layer_indices:
                continue
            # Get the attention module
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                
                # Create LoRA adapters for q and v
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                
                # Replace qkv with LoRA wrapper
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            else:
                print(f"Warning: Block {t_layer_i} doesn't have attn.qkv, skipping")
        
        self.init_w_As = copy.deepcopy(self.w_As)

    def reset(self):
        for idx, w_A in enumerate(self.w_As):
            w_A.weight.data.copy_(self.init_w_As[idx].weight.data)
        for w_B in self.w_Bs:
            w_B.weight.data.zero_()
    
    def get_lora_params(self):
        params = []
        for w_A in self.w_As:
            params.append({'params': w_A.weight})
        for w_B in self.w_Bs:
            params.append({'params': w_B.weight})
        # for module in self.model.modules():
        #     if isinstance(module, _LoRA_qkv):
        #         params.append({'params': module.A})
        #         params.append({'params': module.B})
        return params
    
    def forward(self, images):
        batch_size = images.size(0)
        device = images.device
        input_embeds = self.prompt_learner() #(batch_size, 1， seq_len(32), dim(1024))
        input_embeds = input_embeds.squeeze(1) #(batch_size, seq_len(32), dim(1024)) SAM3只能分单类，所以就不管n_cls的信息了
        
        # 这里是针对TPT这样的方法，由于输入多个增强图像的缘故，images的batch_size可能会发生变化
        if(input_embeds.shape[0]!=batch_size):
            input_embeds = input_embeds.expand(int(batch_size/input_embeds.shape[0]),-1,-1)
        # self.model.backbone.forward_text(prompts, device=device)
        
        # Get image embeddings from backbone
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))
        
        # Get text embeddings
        text_outputs = {}
        
        text_attention_mask = self.prompt_learner.text_attention_mask
        _,text_memory = self.model.backbone.language_backbone.encoder(input_embeds)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = text_memory.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        text_embeds = input_embeds.transpose(0, 1)
        
        text_memory = text_memory_resized[:, : batch_size]
        text_attention_mask = text_attention_mask[: batch_size]
        text_embeds = text_embeds[:, : batch_size]
        text_outputs["language_features"] = text_memory
        text_outputs["language_mask"] = text_attention_mask
        text_outputs["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )
        backbone_out.update(text_outputs)
        
        
        def _slice_batch(obj, idx):
            """Slice batch dimension for tensors/lists/dicts to build per-image inputs."""
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx : idx + 1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = []
                for v in obj:
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
                        sliced.append(v[idx : idx + 1])
                    else:
                        sliced.append(v)
                return type(obj)(sliced)
            return obj

        # Forward through the model
        pred_masks = []
        pred_res = []
        pred_logits = []
        presence_scores = []    
        
        # Process each image separately (SAM3 expects batched but we handle one at a time for compatibility)
        for i in range(images.size(0)):
            # Slice backbone/text outputs to a single-item batch and reset ids to 0
            single_backbone_out = _slice_batch(backbone_out, i)
            # Create empty geometric prompt for this image (we're using text only)
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device=device),
                box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            )
            # Forward grounding
            # For text-only prompts, we always use eval mode to avoid _compute_matching
            # (which requires find_target, but we don't have labels)
            was_training = self.model.training
            self.model.eval()  # Always use eval mode for text-only inference
            
            try:
                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,  # No target - we're using text-only prompts
                    geometric_prompt=geometric_prompt,
                )
            finally:
                # Restore training mode if it was training (for gradient computation)
                if was_training:
                    self.model.train()
            
            _, orig_h, orig_w = images[i].shape
            seg = output["semantic_seg"]  # Shape: [1, num_queries, H, W] or [1, num_queries, 1, H, ]
            if seg.dim() > 2:
                seg = seg.squeeze()
            if seg.shape != (orig_h, orig_w):
                    resized_seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
            
            pred_logits.append(seg)
            pred_masks.append(resized_seg.sigmoid())
            
            out_logits = output["pred_logits"]  # Shape: [1, num_queries, 1]
            out_probs = out_logits.sigmoid()  # [1, num_queries, 1]
            presence_score = output["presence_logit_dec"].sigmoid().unsqueeze(1)  # [1, 1, 1]
            scores = (out_probs * presence_score).squeeze(-1)  # [1, num_queries]
            presence_scores.append(scores[0])
            
        
        return pred_logits, pred_masks, presence_scores

    
    def forward_features(self, input):
        pass

def get_sam3_lora(classnames, device, batch_size, n_ctx, ctx_init, learned_cls=False):
    model = SAM3LoraTestTimeTuning(device, classnames, batch_size, n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)
    return model


class PromptLearner_Dynamic(nn.Module):
    def __init__(self, sam3_model, classnames, batch_size=None, n_ctx=4, ctx_init=None, ctx_position='end', learned_cls=False, bpe_path=None, num_p=1
                 ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.weight.dtype
        self.dtype = dtype
        self.device = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.weight.device
        ctx_dim = sam3_model.backbone.vision_backbone.trunk.patch_embed.proj.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.bpe_path = bpe_path
        tokenizer = SimpleTokenizer(bpe_path,context_length=32)
        self.num_p = num_p
        
        if ctx_init:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            ctx_init = ctx_init.replace("{", "[")
            ctx_init = ctx_init.replace("}", "]")
            if ' [CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
                self.split_idx = split_idx
            elif '[CLS]' in ctx_init:
                split_idx = None
                ctx_position = "front"
                self.split_idx = split_idx
            
            n_ctx0 = len(ctx_init.split(" "))
            prompt = tokenizer(ctx_init).to(self.device)
            self.text_attention_mask = (prompt != 0).bool().ne(1)
            with torch.no_grad():
                embedding = sam3_model.backbone.language_backbone.encoder.token_embedding(prompt).type(dtype)
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            # ！！！这里是主要改动之一，目前还不太清楚用意
            ctx_vectors = torch.zeros(embedding[0, 1 + (n_ctx0-n_ctx) : 1 + n_ctx0, :].size(), dtype=dtype)
            self.init_ctx = embedding[0, 1 + (n_ctx0-n_ctx) : 1 + n_ctx0, :].clone()
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
            self.text_attention_mask = self.text_attention_mask.repeat(batch_size, 1)  #(N, L)
            
        ##### multiple prompts #####
        if num_p > 1:
            ctx_vectors = ctx_vectors.unsqueeze(0).repeat(num_p, 1, 1)  # (N, L, D)
            ctx_vectors = ctx_vectors.permute(1, 0, 2)  # (L, N, D)
        ##### feature initialization #####
        if num_p > 1:
            ctx_vectors = ctx_vectors.permute(1, 0, 2)  # (N, L, D)
        
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        self.ctx_order = list(range(num_p))
        self.ctx_use = [0] * num_p

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = sam3_model.backbone.language_backbone.encoder.token_embedding(tokenized_prompts).type(dtype)
            

        self.register_buffer("token_prefix", embedding[:, :1 + n_ctx0 - n_ctx, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx0 :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_ctx0 = n_ctx0
        self.classnames = classnames


    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        assert False

    def forward(self, init=None):
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx + self.init_ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        else:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        
        # ！！！这个条件是为了在batch_size=1且n_cls=1时，扩展成(n_cls, n_ctx, dim)的形式，方便后续拼接
        if(ctx.size()[0] == 1 and self.n_cls == 1):
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)
        # ！！！这里有个改动，感觉是不是不能随便设置batch_size了
        elif self.num_p > 1:
            prefix = prefix.unsqueeze(0).repeat(self.num_p, 1, 1, 1)
            suffix = suffix.unsqueeze(0).repeat(self.num_p, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:,i : i + 1, :, :] #!!! 注意，因为我多了batch维度，所以前面加了个:，front和end同理
                class_i = suffix[:,i : i + 1, :name_len, :]
                suffix_i = suffix[:,i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[:,i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[:,i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=-2,  #!!! 注意，因为我多了batch维度，所以这里dim要设置为-2，front和end同理
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:,i : i + 1, :, :]
                class_i = suffix[:,i : i + 1, :name_len, :]
                suffix_i = suffix[:,i : i + 1, name_len:, :]
                ctx_i = ctx[:,i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class SAM3TestTimeTuning_Dynamic(nn.Module):
    def __init__(self, device, classnames, batch_size,
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, num_p=1):
        super(SAM3TestTimeTuning_Dynamic, self).__init__()
        bpe_path = _sam3_bpe_path()
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=_sam3_checkpoint_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=False,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # We use text prompts, not point/box
            compile=False,
        )
        # By default, freeze everything, 这里不存储SAM3的梯度信息，来节省显存
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        self.device = device
        # prompt tuning
        self.num_p = num_p
        self.prompt_learner = PromptLearner_Dynamic(self.model, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, bpe_path, num_p)
        # self.ini_model_state = copy.deepcopy(self.model.state_dict())

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()
        # self.model.load_state_dict(self.ini_model_state, strict=True)
        

    def inference(self, images, prompt_idx=None):
        """
        Forward pass for SAM3 with text prompts.
        
        Args:
            images: Tensor of shape [B, C, H, W]
            prompts: List of text prompts (noun phrases), one per image in batch
        
        Returns:
            image_embeddings: Backbone features (for compatibility)
            pred_masks: List of predicted masks, one per image
            ious: List of IoU predictions (dummy for now, SAM3 doesn't output per-mask IoU)
            res_masks: List of low-res masks
        """
        batch_size = images.size(0)
        device = images.device
        input_embeds = self.prompt_learner() #(batch_size, 1， seq_len(32), dim(1024))
        input_embeds = input_embeds.squeeze(1) #(batch_size, seq_len(32), dim(1024)) SAM3只能分单类，所以就不管n_cls的信息了
        
        # 这里是针对TPT这样的方法，由于输入多个增强图像的缘故，images的batch_size可能会发生变化
        if(input_embeds.shape[0]<batch_size):
            input_embeds = input_embeds.repeat(int(batch_size/input_embeds.shape[0]),1,1)
        
        # 这里是针对DynaPrompt这样的方法，会得到多个Prompt，来输入
        if(input_embeds.shape[0]>batch_size):
            images = images.repeat(int(input_embeds.shape[0]/batch_size),1,1,1)
            batch_size = input_embeds.shape[0]
        # self.model.backbone.forward_text(prompts, device=device)
        
        if(prompt_idx is not None):
            images = images[prompt_idx]
            input_embeds = input_embeds[prompt_idx]
        # Get image embeddings from backbone
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))
        
        # Get text embeddings
        text_outputs = {}
        
        text_attention_mask = self.prompt_learner.text_attention_mask
        _,text_memory = self.model.backbone.language_backbone.encoder(input_embeds)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = text_memory.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        text_embeds = input_embeds.transpose(0, 1)
        
        text_memory = text_memory_resized[:, : batch_size]
        text_attention_mask = text_attention_mask[: batch_size]
        text_embeds = text_embeds[:, : batch_size]
        text_outputs["language_features"] = text_memory
        text_outputs["language_mask"] = text_attention_mask
        text_outputs["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )
        backbone_out.update(text_outputs)
        
        
        def _slice_batch(obj, idx):
            """Slice batch dimension for tensors/lists/dicts to build per-image inputs."""
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx : idx + 1]
                elif obj.dim() > 0 and obj.size(1) == batch_size:
                    return obj[:, idx : idx + 1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = []
                for v in obj:
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
                        sliced.append(v[idx : idx + 1])
                    elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(1) == batch_size:
                        sliced.append(v[:, idx : idx + 1])
                    else:
                        sliced.append(v)
                return type(obj)(sliced)
            return obj

        # Forward through the model
        pred_masks = []
        pred_logits = []
        
        # Process each image separately (SAM3 expects batched but we handle one at a time for compatibility)
        for i in range(images.size(0)):
            # Slice backbone/text outputs to a single-item batch and reset ids to 0
            single_backbone_out = _slice_batch(backbone_out, i)
            # Create empty geometric prompt for this image (we're using text only)
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device=device),
                box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            )
            # Forward grounding
            # For text-only prompts, we always use eval mode to avoid _compute_matching
            # (which requires find_target, but we don't have labels)
            was_training = self.model.training
            self.model.eval()  # Always use eval mode for text-only inference
            
            try:
                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,  # No target - we're using text-only prompts
                    geometric_prompt=geometric_prompt,
                )
            finally:
                # Restore training mode if it was training (for gradient computation)
                if was_training:
                    self.model.train()
            
            _, orig_h, orig_w = images[i].shape
            seg = output["semantic_seg"]  # Shape: [1, num_queries, H, W] or [1, num_queries, 1, H, ]
            if seg.dim() > 2:
                seg = seg.squeeze()
            if seg.shape != (orig_h, orig_w):
                    seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
            
            pred_logits.append(seg)
            pred_masks.append(seg.sigmoid())
            
        
        return pred_logits, pred_masks

    def forward(self, input, prompt_idx=None):
        return self.inference(input, prompt_idx)
        
    def forward_features(self, images, prompts):
        # image_features = self.model.backbone.forward_image(images)["backbone_fpn"]
        # text_features = self.model.backbone.forward_text(prompts, device=images.device)["language_features"]
        image_features = self.model.backbone.forward_image(images)
        text_features = self.model.backbone.forward_text(prompts, device=images.device)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

def get_sam3_dynamic(classnames, device, batch_size, n_ctx, ctx_init, learned_cls=False, num_p=1):

    model = SAM3TestTimeTuning_Dynamic(device, classnames, batch_size, 
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_p=num_p)

    return model

class SAM3TestTimeTuning_Multi_Dynamic(nn.Module):
    def __init__(self, device, classnames, batch_size,
                        n_ctx=16, ctx_init_list=None, ctx_position='end', learned_cls=False, num_p=1):
        super(SAM3TestTimeTuning_Multi_Dynamic, self).__init__()
        bpe_path = _sam3_bpe_path()
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=_sam3_checkpoint_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=False,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # We use text prompts, not point/box
            compile=False,
        )
        # By default, freeze everything, 这里不存储SAM3的梯度信息，来节省显存
        self.model.requires_grad_(False)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        self.device = device
        # prompt tuning
        self.prompt_learner_list = [PromptLearner(self.model, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, bpe_path) \
            for ctx_init in ctx_init_list]
        # self.ini_model_state = copy.deepcopy(self.model.state_dict())
        self.w_As = []  # LoRA A matrices
        self.w_Bs = []  # LoRA B matrices

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        for prompt_learner in self.prompt_learner_list:
            prompt_learner.reset()
        # self.model.load_state_dict(self.ini_model_state, strict=True)
        for idx, w_A in enumerate(self.w_As):
            w_A.weight.data.copy_(self.init_w_As[idx].weight.data)
        for w_B in self.w_Bs:
            w_B.weight.data.zero_()
    
    def add_lora(self, layer_indices=[9,10,11], r=16, lora_alpha=32, lora_dropout=0.05):
        if layer_indices is None:
            layer_indices = list(range(len(self.model.backbone.vision_backbone.trunk.blocks)))
        print(f"Applying LoRA to SAM3 visual backbone layers: {layer_indices}")
        for t_layer_i, blk in enumerate(self.model.backbone.vision_backbone.trunk.blocks):
            if t_layer_i not in layer_indices:
                continue
            # Get the attention module
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                
                # Create LoRA adapters for q and v
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                
                # Replace qkv with LoRA wrapper
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            else:
                print(f"Warning: Block {t_layer_i} doesn't have attn.qkv, skipping")
        
        self.init_w_As = copy.deepcopy(self.w_As)

    def get_lora_params(self):
        params = []
        for w_A in self.w_As:
            params+=list(w_A.parameters())
        for w_B in self.w_Bs:
            params+=list(w_B.parameters())
        # for module in self.model.modules():
        #     if isinstance(module, _LoRA_qkv):
        #         params.append({'params': module.A})
        #         params.append({'params': module.B})
        return params
    
    def inference(self, images, prompt_idx=None):
        """
        Forward pass for SAM3 with text prompts.
        
        Args:
            images: Tensor of shape [B, C, H, W]
            prompts: List of text prompts (noun phrases), one per image in batch
        
        Returns:
            image_embeddings: Backbone features (for compatibility)
            pred_masks: List of predicted masks, one per image
            ious: List of IoU predictions (dummy for now, SAM3 doesn't output per-mask IoU)
            res_masks: List of low-res masks
        """
        batch_size = images.size(0)
        device = images.device
        input_embeds = []
        text_attention_mask = []
        for prompt_learner in self.prompt_learner_list:
            input_embeds.append(prompt_learner())
            text_attention_mask.append(prompt_learner.text_attention_mask)
        input_embeds = torch.cat(input_embeds,dim=0)
        text_attention_mask = torch.cat(text_attention_mask,dim=0)
        input_embeds = input_embeds.squeeze(1) #(batch_size, seq_len(32), dim(1024)) SAM3只能分单类，所以就不管n_cls的信息了
        
        # 这里会得到多个Prompt，来输入
        if(input_embeds.shape[0]>batch_size):
            images = images.repeat(int(input_embeds.shape[0]/batch_size),1,1,1)
            batch_size = input_embeds.shape[0]
        # self.model.backbone.forward_text(prompts, device=device)
        
        if(prompt_idx is not None):
            images = images[prompt_idx]
            input_embeds = input_embeds[prompt_idx]
            if(images.dim()==3):
                images = images.unsqueeze(0)
                input_embeds = input_embeds.unsqueeze(0)
        # Get image embeddings from backbone
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))
        
        # Get text embeddings
        text_outputs = {}
        
        _,text_memory = self.model.backbone.language_backbone.encoder(input_embeds)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = text_memory.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        text_embeds = input_embeds.transpose(0, 1)
        
        text_memory = text_memory_resized[:, : batch_size]
        text_attention_mask = text_attention_mask[: batch_size]
        text_embeds = text_embeds[:, : batch_size]
        text_outputs["language_features"] = text_memory
        text_outputs["language_mask"] = text_attention_mask
        text_outputs["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )
        backbone_out.update(text_outputs)
        
        
        def _slice_batch(obj, idx):
            """Slice batch dimension for tensors/lists/dicts to build per-image inputs."""
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and obj.size(0) == batch_size:
                    return obj[idx : idx + 1]
                elif obj.dim() > 0 and obj.size(1) == batch_size:
                    return obj[:, idx : idx + 1]
                return obj
            if isinstance(obj, dict):
                return {k: _slice_batch(v, idx) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                sliced = []
                for v in obj:
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
                        sliced.append(v[idx : idx + 1])
                    elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(1) == batch_size:
                        sliced.append(v[:, idx : idx + 1])
                    else:
                        sliced.append(v)
                return type(obj)(sliced)
            return obj

        # Forward through the model
        pred_masks = []
        pred_logits = []
        presence_scores = []    
        
        # Process each image separately (SAM3 expects batched but we handle one at a time for compatibility)
        for i in range(images.size(0)):
            # Slice backbone/text outputs to a single-item batch and reset ids to 0
            single_backbone_out = _slice_batch(backbone_out, i)
            # Create empty geometric prompt for this image (we're using text only)
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device=device),
                box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            )
            # Forward grounding
            # For text-only prompts, we always use eval mode to avoid _compute_matching
            # (which requires find_target, but we don't have labels)
            was_training = self.model.training
            self.model.eval()  # Always use eval mode for text-only inference
            
            try:
                output = self.model.forward_grounding(
                    backbone_out=single_backbone_out,
                    find_input=FindStage(
                        img_ids=torch.tensor([0], device=device, dtype=torch.long),
                        text_ids=torch.tensor([0], device=device, dtype=torch.long),
                        input_boxes=None,
                        input_boxes_mask=None,
                        input_boxes_label=None,
                        input_points=None,
                        input_points_mask=None,
                    ),
                    find_target=None,  # No target - we're using text-only prompts
                    geometric_prompt=geometric_prompt,
                )
            finally:
                # Restore training mode if it was training (for gradient computation)
                if was_training:
                    self.model.train()
            
            _, orig_h, orig_w = images[i].shape
            seg = output["semantic_seg"]  # Shape: [1, num_queries, H, W] or [1, num_queries, 1, H, ]
            if seg.dim() > 2:
                seg = seg.squeeze()
            if seg.shape != (orig_h, orig_w):
                    resized_seg = interpolate(
                        seg.unsqueeze(0).unsqueeze(0),
                        (orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
            
            pred_logits.append(seg)
            pred_masks.append(resized_seg.sigmoid())
            
            out_logits = output["pred_logits"]  # Shape: [1, num_queries, 1]
            out_probs = out_logits.sigmoid()  # [1, num_queries, 1]
            presence_score = output["presence_logit_dec"].sigmoid().unsqueeze(1)  # [1, 1, 1]
            scores = (out_probs * presence_score).squeeze(-1)  # [1, num_queries]
            presence_scores.append(scores[0])
            
        
        return pred_logits, pred_masks, backbone_out['language_features'], backbone_out['language_mask'], backbone_out['vision_features'],presence_scores

    def forward(self, input, prompt_idx=None):
        return self.inference(input, prompt_idx)
        
    def forward_features(self, images, prompts):
        # image_features = self.model.backbone.forward_image(images)["backbone_fpn"]
        # text_features = self.model.backbone.forward_text(prompts, device=images.device)["language_features"]
        image_features = self.model.backbone.forward_image(images)
        text_features = self.model.backbone.forward_text(prompts, device=images.device)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, text_features
    
    def forward_negtive_text_features(self, prompts):
        text_features = self.model.backbone.forward_text(prompts, device='cuda:0')
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return text_features["language_features"], text_features["language_mask"]

def get_sam3_multi_dynamic(classnames, device, batch_size, n_ctx, ctx_init_list, learned_cls=False, num_p=1):

    model = SAM3TestTimeTuning_Multi_Dynamic(device, classnames, batch_size, 
                            n_ctx=n_ctx, ctx_init_list=ctx_init_list, learned_cls=learned_cls, num_p=num_p)

    return model
