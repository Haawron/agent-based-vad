from typing import Union

import torch

from .open_clip import load_open_clip

TYPE2FUNC = {
    "open_clip": load_open_clip,
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
    args,
    model_type: str,
    model_name: str,
    pretrained: str,
    cache_dir: str,
    device: Union[str, torch.device] = "cuda",
):
    assert model_type in ["open_clip"], f"model_type={model_type} is invalid!"
    load_func = TYPE2FUNC[model_type]
    if model_type == "open_clip":
        return load_open_clip(
            args,
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            device=device,
        )
    else:
        assert False, f"model_type={model_type} is not supported yet!"
