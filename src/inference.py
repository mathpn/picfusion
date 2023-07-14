"""
Inference functions.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ram import get_transform
from ram.models import ram


def create_clip_extractor(device: str = "cpu"):
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    torch_device = torch.device(device)
    model = model.to(torch_device)

    @torch.no_grad()
    def img_extractor(images: list[Image.Image]):
        inputs = processor(images=images, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        feat = model.get_image_features(**inputs)
        feat = feat.cpu().numpy()
        feat /= np.linalg.norm(feat, axis=-1, keepdims=True)
        return feat

    @torch.no_grad()
    def text_extractor(text: list[str]) -> np.ndarray:
        inputs = processor(text=text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        feat = model.get_text_features(**inputs)
        feat = feat.cpu().numpy()
        feat /= np.linalg.norm(feat, axis=-1, keepdims=True)
        return feat

    return img_extractor, text_extractor


def create_ram_extractor(model_path: str, image_size: int = 384, device: str = "cpu"):
    transform = get_transform(image_size=image_size)
    model = ram(pretrained=model_path, image_size=image_size, vit="swin_l")
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    @torch.no_grad()
    def extractor(images: list[Image.Image]) -> list[set[str]]:
        image = torch.stack([transform(image) for image in images], dim=0).to(torch_device)
        res = model.generate_tag(image, model)
        return [{x.strip() for x in tags.split("|")} for tags in res[0]]

    return extractor


if __name__ == "__main__":
    ram_extractor = create_ram_extractor("./models/ram_swin_large_14m.pth")
    clip_extractor = create_clip_extractor()

    img, _ = Image.open("./34845106.png")
    out = clip_extractor(img)
    print(out)
    out = ram_extractor([img])
    print(out)
