import argparse

from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    image = Image.open(args.image_path)
    inputs = processor(images=image, return_tensors="pt")
    print(inputs)
    feat = model.get_image_features(**inputs)

    print(feat.shape)


if __name__ == "__main__":
    main()
