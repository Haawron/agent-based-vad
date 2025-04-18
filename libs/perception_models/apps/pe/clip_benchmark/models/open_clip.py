# import open_clip
from core.vision_encoder.factory import (create_model_and_transforms,
                                         get_tokenizer)


def load_open_clip(
    args,
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str = "laion400m_e32",
    cache_dir: str = None,
    device="cpu",
):
    model, _, transform = create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_preprocess_cfg=args.force_preprocess_cfg,
    )
    print(transform)
    model = model.to(device)
    tokenizer = get_tokenizer(model_name)
    return model, transform, tokenizer
