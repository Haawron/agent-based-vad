from typing import Dict

import numpy as np
import torch
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPModel,
                          CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPVisionModel, CLIPVisionModelWithProjection)


class HuggingFaceTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        t = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")[
            "input_ids"
        ]
        return torch.Tensor(t)

    def __len__(self):
        return len(self.tokenizer)


class HuggingFaceProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        images = self.processor(images)
        pixels = images["pixel_values"]
        pixels = np.stack(pixels)
        tensor = torch.Tensor(pixels).squeeze()
        return tensor


class HuggingFaceCLIPWrapper(torch.nn.Module):
    """
    Wraps encode_text and encode_image around the huggingface CLIPModel
    """

    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

    def encode_text(self, tensor):
        outputs = self.text_encoder(tensor)
        return outputs.text_embeds

    def encode_image(self, image):
        outputs = self.vision_encoder(image)
        return outputs.image_embeds

    def eval(self):
        pass


def load_huggingface_clip(pretrained: str, device="cpu", **kwargs):
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained).to(
        device
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained).to(device)
    processor = CLIPImageProcessor.from_pretrained(pretrained)
    auto_tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = HuggingFaceCLIPWrapper(vision_encoder, text_encoder)
    tokenizer = HuggingFaceTokenizerWrapper(auto_tokenizer)
    transforms = HuggingFaceProcessorWrapper(processor)
    return model, transforms, tokenizer
