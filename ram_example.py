import argparse

import numpy as np
import torch
from PIL import Image

from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram


def main():
    parser = argparse.ArgumentParser(description="Tag2Text inferece for tagging and captioning")
    parser.add_argument("image")
    parser.add_argument(
        "--image-size", default=384, type=int, metavar="N", help="input image size (default: 448)"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform(image_size=args.image_size)

    model = ram(pretrained="./models/ram_swin_large_14m.pth", image_size=args.image_size, vit="swin_l")
    model.eval()

    model = model.to(device)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    res = inference(image, model)
    print(res[0])


if __name__ == "__main__":
    main()
