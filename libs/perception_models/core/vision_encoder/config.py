# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Include all available vision encoder configurations.
"""

PEV1_SETTINGS = {
    # CORE
    "pev1_G14_448": {
        "image_size": 448,
        "patch_size": 14,
        "width": 1536,
        'output_dim': 1280,
        "layers": 50,
        "heads": 16,
        "embed_cls_token": False,
        "abs_pos_embed": True,
        "mlp_ratio": 5.833333334, 
    },
    "pev1_L14_336": {
        "image_size": 336,
        "patch_size": 14,
        "width": 1024,
        'output_dim': 1024,
        "layers": 24,
        "heads": 16,
        "embed_cls_token": True,
        "abs_pos_embed": True,
        "mlp_ratio": 4.0, 
    },
    "pev1_B16_224": {
        "image_size": 224,
        "patch_size": 16,
        "width": 768,
        'output_dim': 1024,
        "layers": 12,
        "heads": 12,
        "mlp_ratio": 4.0,
        "embed_cls_token": True,
        "abs_pos_embed": True,
    },

    # LANG
    "pev1_lang_G14_448": {
        "image_size": 448,
        "patch_size": 14,
        "width": 1536,
        "layers": 47,
        "heads": 16,
        "embed_cls_token": False,
        "abs_pos_embed": True,
        "mlp_ratio": 5.833333334,
        "pool_type": "none",
        "use_ln_post": False,
        "vision_select_feature": "patch",
        "ls_init_value": 0.1,
    },

    "pev1_lang_L14_448": {
        "image_size": 448
        "patch_size": 14
        "width": 1024
        "layers": 23
        "heads": 16
        "embed_cls_token": True
        "abs_pos_embed": True
        "mlp_ratio": 4.0
        "ls_init_value": 0.1
        "vision_select_feature": "patch"
        "use_ln_post": False
        "pool_type": "none"
        "remove_class_token_output": True
    }



    # SPATIAL
    "pev1_spatial_G14_448": {
        "image_size": 448,
        "patch_size": 14,
        "width": 1536,
        "layers": 50,
        "heads": 16,
        "embed_cls_token": False,
        "abs_pos_embed": True,
        "mlp_ratio": 5.833333334,
        "pool_type": "none",
        "use_ln_post": False,
        "vision_select_feature": "patch",
        "ls_init_value": 0.1,
    },


}


PEV1_CLIP_SETTINGS = {
    "pev1_G14_448": {
        'vision':{
            "image_size": 448,
            "patch_size": 14,
            "width": 1536,
            'output_dim': 1280,
            "layers": 50,
            "heads": 16,
            "mlp_ratio": 5.833333334,
            "embed_cls_token": False,
            "abs_pos_embed": True,
        },
        'text':{
            "context_length":72,
            "vocab_size" 49408,
            "width": 1280,
            'output_dim': 1280,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4.0,
            "pool_type": "argmax",
        },
    },
    "pev1_L14_336": {
        'vision':{
            "image_size": 336,
            "patch_size": 14,
            "width": 1024,
            'output_dim': 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4.0,
            "embed_cls_token": True,
            "abs_pos_embed": True,
        },
        'text':{
            "context_length":32,
            "vocab_size" 49408,
            "width": 1024,
            'output_dim': 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4.0,
            "embed_cls_token": False,
            "pool_type": "argmax",
        },
    },
    "pev1_B16_224": {
        'vision':{
            "image_size": 224,
            "patch_size": 16,
            "width": 768,
            'output_dim': 1024,
            "layers": 12,
            "heads": 12,
            "mlp_ratio": 4.0,
            "embed_cls_token": True,
            "abs_pos_embed": True,
        },
        'text':{
            "context_length":32,
            "vocab_size" 49408,
            "width": 1024,
            'output_dim': 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4.0,
            "embed_cls_token": False,
            "pool_type": "argmax",
        },
    },      
}

