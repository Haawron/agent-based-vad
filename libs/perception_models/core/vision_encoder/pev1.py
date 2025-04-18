# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE.PE file in the root directory of this source tree.

import copy
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint

from core.vision_encoder.pos_embed import RotaryEmbedding, apply_rotary_emb

logger = getLogger()


class Rope2D:
    def __init__(self, dim, use_cls_token=False, class_token_in_front=True):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.class_token_in_front = class_token_in_front
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        self.rope = RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)

            self.rope = self.rope.to(device)

            if self.use_cls_token:
                # +1 to leave space for the cls token to be (0, 0)
                grid_y_range = torch.arange(grid_h, device=device) + 1
                grid_x_range = torch.arange(grid_w, device=device) + 1
            else:
                grid_y_range = torch.arange(grid_h, device=device)
                grid_x_range = torch.arange(grid_w, device=device)

            freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                if self.class_token_in_front:
                    freq = torch.cat(
                        [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                    )
                else:
                    freq = torch.cat(
                        [freq, torch.zeros(1, freq.shape[-1], device=device)], dim=0
                    )

            self.freq = freq[None, ...]

        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        # batch, heads, seq, dim = q.shape
        q = apply_rotary_emb(self.freq[:, None, :, :], q)
        k = apply_rotary_emb(self.freq[:, None, :, :], k)

        return q, k


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def init_tensors(self):
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # To make this compatibile with nn.MultiHeadAttention
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.scale = self.head_dim ** (-0.5)

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None):
        batch, seq, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rope:
            self.attn = SelfAttention(d_model, n_head, rope=rope)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def _call_attn(
        self,
        q_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        if isinstance(self.attn, SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask)
        else:
            return self.attn(q_x, q_x, q_x, attn_mask=attn_mask, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    rope=rope,
                )
                for _ in range(layers)
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        for _, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

        return x

    def get_attn(self, x: torch.Tensor):
        attns = []
        for r in self.resblocks:
            x, attn = r(x, need_weights=True)
            attns.append(attn)
        return attns


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        ls_init_value: float = None,
        drop_path: float = 0.0,
        use_ln_pre: bool = True,
        abs_pos_embed: bool = True,
        image_size: int = 448,
        embed_cls_token: bool = False,
        class_token_in_front: bool = True,
        remove_class_token_output: bool = False,
        use_rope2d: bool = True,
        output_dim: int = 1280,
        attn_pooler_heads: int = 8,
        pool_type: str = "attn",
        use_ln_post: bool = True,
        load_ckpt: bool = False,
        ckpt_path: str = None,
        freeze_vision_model: bool = False,
        vision_select_layer: int = -1,
        vision_select_feature: str = "pooled",  # "patch" or "pooled
        remove_unused_layers: bool = False,
    ):
        super().__init__()
        assert pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size

        self.output_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers

        self.use_abs_posemb = abs_pos_embed
        self.use_cls_token = embed_cls_token
        self.use_rope2d = use_rope2d
        self.image_size = image_size

        self.freeze_vision_model = freeze_vision_model
        self.vision_select_layer = vision_select_layer
        self.vision_select_feature = vision_select_feature
        self.embed_cls_token = embed_cls_token  # a bug, but we live with it for now.
        self.class_token_in_front = class_token_in_front
        self.remove_class_token_output = remove_class_token_output

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.rope = (
            Rope2D(
                dim=width // heads,
                use_cls_token=self.use_cls_token,
                class_token_in_front=self.class_token_in_front,
            )
            if self.use_rope2d
            else None
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()

        if remove_unused_layers:
            assert vision_select_layer < 0
            logger.info("Removing unused layers from vision encoder...")
            layers = layers + vision_select_layer + 1
            self.vision_select_layer = -1
            logger.info(f"Number of transformer layers: {layers}")

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.init_tensors()
        if load_ckpt and ckpt_path:
            self.load_ckpt(ckpt_path)

    def init_tensors(self):
        def init_submodule_tensors(module):
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    logger.debug(f"Initializing tensors for submodule: {name}")
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        self.rope.init_tensors()

        # class embeddings and positional embeddings
        init_scale = self.width**-0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                init_scale
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2, self.width
                )
            )

        if self.vision_select_feature == "pooled":
            self.proj = nn.Parameter(
                init_scale * torch.randn(self.width, self.output_dim)
            )

    def load_ckpt(self, ckpt_path: str):
        _sd = torch.load(ckpt_path, weights_only=True)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        # for backwards compatibility
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        if any(k.startswith("visual.") for k in _sd):
            _sd = {k.replace("visual.", ""): v for k, v in _sd.items() if "visual" in k}

        m, u = self.load_state_dict(_sd, strict=False)
        logger.info(f"Missing keys for loading vision encoder: {m}")
        logger.info(f"Unexpected keys for loading vision encoder: {u}")

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.set_grad_checkpointing(enable=enable)

    def _sample_abs_posemb(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            # Keep it this way even if class_token_in_front is False for backwards compatibility
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def _pool(self, x: torch.Tensor):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        elif self.pool_type == "none":
            return x
        else:
            raise NotImplementedError

    def forward_features(self, x: torch.Tensor, norm: bool = False):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            if self.class_token_in_front:
                x = torch.cat(
                    [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                    dim=1,
                )
            else:
                x = torch.cat(
                    [x, self.class_embedding.view(1, 1, -1).expand(batch, -1, -1)],
                    dim=1,
                )

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)

        if self.use_rope2d:
            self.rope.update_grid(x.device, grid_h, grid_w)

        x = self.ln_pre(x)
        x = self.transformer(x)

        if norm:
            x = self.ln_post(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ):

        x = self.forward_features(x, norm=True)
        if self.vision_select_feature == "patch":
            if self.remove_class_token_output:
                if self.class_token_in_front:
                    return x[:, 1:]
                else:
                    return x[:, :-1]
            else:
                return x
        elif self.vision_select_feature == "pooled":
            return self._pool(x) @ self.proj
        else:
            raise NotImplementedError(
                f"Unsupported vision select feature: {self.vision_select_feature}"
            )


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 72,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 1280,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        output_tokens: bool = False,
        parallel_transformer: bool = False,
        normalize_qk: bool = False,
        use_ln_post: bool = True,
        #####
        load_ckpt: bool = True,
        ckpt_path: str = None,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none", "attn", "attn_eos")
        self.pool_type = pool_type
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.layers = layers

        self.token_embedding = nn.Embedding(vocab_size, width)

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        # self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity() # text doesn't use ln_pre

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )

        if pool_type == "attn" or pool_type == "attn_eos":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:  # argmax
            self.attn_pool = None

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        # self.embed_cls_token = embed_cls_token # a bug, but we live with it for now.
        if load_ckpt and ckpt_path:
            self.init_tensors(ckpt_path)

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init_tensors(self, ckpt_path: str):
        _sd = torch.load(ckpt_path)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        _sd_new = {}
        for key, value in _sd.items():
            if key.startswith(
                "module."
            ):  # extract the visual tower and remove 'module.' prefix
                new_key = key.replace("module.", "")
                _sd_new[new_key] = value
        _sd = _sd_new

        m, u = self.load_state_dict(_sd, strict=False)
        # print(f"m: {m}\n\n\nu: {u}")
        # assert len(m) == 0, f"Missing keys: {m}"
        logger.info(f"Missing keys for loading vision encoder: {m}")
        logger.info(f"Unexpected keys for loading vision encoder: {u}")
        print(f"Missing keys for loading vision encoder: {m}")
        print(f"Unexpected keys for loading vision encoder: {u}")

    def build_cls_mask(self, text):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def text_global_pool(
        self, x, text: Optional[torch.Tensor] = None, pool_type: str = "argmax"
    ):
        if pool_type == "first":
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == "last":
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == "argmax":
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, text):
        seq_len = text.shape[1]
        x = self.token_embedding(
            text
        )  # .to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text)
            if attn_mask is not None:
                attn_mask = (
                    attn_mask[None, :seq_len, :seq_len]
                    + cls_mask[:, :seq_len, :seq_len]
                )
        elif attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.pool_type == "attn":
            tokens = self.ln_final(x)
            pooled = self.attn_pool(tokens).squeeze(1)
        elif self.pool_type == "attn_eos":
            # Only use the eos token for the clip token, but emulate going through a self-attn layer
            # so that the text distribution can more closely match the vision distribution
            tokens = self.ln_final(x)
            pool_x, _ = self.text_global_pool(tokens, text, pool_type="argmax")
            pooled = self.attn_pool(pool_x.unsqueeze(1)).squeeze(1)
        elif self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = self.text_global_pool(x, pool_type="last")
            pooled = self.ln_final(
                pooled
            )  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = self.text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled
