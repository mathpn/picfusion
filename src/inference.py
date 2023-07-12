"""
Inference functions.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from ram import get_transform
from ram.models import ram


def create_clip_extractor(device: str = "cpu"):
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    torch_device = torch.device(device)
    model.to(torch_device)

    @torch.no_grad()
    def extractor(images: list[Image.Image]):
        inputs = processor(images=images, return_tensors="pt")
        feat = model.get_image_features(**inputs)
        return feat.numpy()

    return extractor


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

    img = Image.open("./34845106.png")
    out = clip_extractor(img)
    print(out)
    out = ram_extractor([img])
    print(out)
