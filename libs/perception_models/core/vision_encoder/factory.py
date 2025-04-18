# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE.PE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenCLIP: https://github.com/mlfoundations/open_clip
# --------------------------------------------------------

import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from core.vision_encoder.model import (CLIP, resize_pos_embed,
                                       resize_text_pos_embed,
                                       set_model_preprocess_cfg)
from core.vision_encoder.tokenizer import (DEFAULT_CONTEXT_LENGTH,
                                           SimpleTokenizer)
from core.vision_encoder.transform import (AugmentationCfg, PreprocessCfg,
                                           image_transform_v2,
                                           merge_preprocess_dict,
                                           merge_preprocess_kwargs)

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def print_model_param(model):
    """Print model parameters."""
    if int(os.environ.get("RANK", 0)) == 0:
        for name, sub_module in model.named_children():
            print(f"Submodule: {name}")
            print("Number of parameters:")
            total_params = sum(p.numel() for p in sub_module.parameters())
            print(f"{total_params / 1e9:.2f} Billion")


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(
    model_name: str = "",
    context_length: Optional[int] = None,
    **kwargs,
):
    config = get_model_config(model_name)
    assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get("text_cfg", {})
    if "tokenizer_kwargs" in text_config:
        tokenizer_kwargs = dict(text_config["tokenizer_kwargs"], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get("context_length", DEFAULT_CONTEXT_LENGTH)

    tokenizer = SimpleTokenizer(
        context_length=context_length,
        **tokenizer_kwargs,
    )

    return tokenizer


def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=False):
    if Path(checkpoint_path).suffix in (".npz", ".npy"):
        from .big_vision import load_big_vision_weights

        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if "positional_embedding" in state_dict and not hasattr(
        model, "positional_embedding"
    ):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # resize_pos_embed(state_dict, model)
    # resize_text_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    print(incompatible_keys)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    force_vision_cfg: Optional[Dict[str, Any]] = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: Optional[str] = None,
    output_dict: Optional[bool] = None,
    require_pretrained: bool = False,
    **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    model_name = model_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names
    checkpoint_path = None
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f"Loaded {model_name} model config.")
    else:
        logging.error(
            f"Model config for {model_name} not found; available models {list_models()}."
        )
        raise RuntimeError(f"Model config for {model_name} not found.")

    model_cfg = dict(
        model_cfg, **model_kwargs
    )  # merge cfg dict w/ kwargs (kwargs overrides cfg)

    print(model_cfg)
    logging.info(f"Initializing CLIP model.")
    model = CLIP(**model_cfg)  # , cast_dtype=cast_dtype)
    model.to(device=device, dtype=torch.float32)
    pretrained_loaded = False
    if pretrained:
        checkpoint_path = pretrained
        if checkpoint_path:
            logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
            load_checkpoint(model, checkpoint_path)
        else:
            error_str = (
                f"Pretrained weights ({pretrained}) not found for model {model_name}."
                f" Available pretrained tags ({list_pretrained_tags_by_model(model_name)}."
            )
            logging.warning(error_str)
            raise RuntimeError(error_str)
        pretrained_loaded = True

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, "image_size", None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg["size"] = model.visual.image_size
    set_model_preprocess_cfg(
        model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg)
    )

    print_model_param(model)
    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = (0.5, 0.5, 0.5),
    image_std: Optional[Tuple[float, ...]] = (0.5, 0.5, 0.5),
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: Optional[str] = None,
    output_dict: Optional[bool] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = {},
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        force_preprocess_cfg,
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,  # bernie: check this later
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val
