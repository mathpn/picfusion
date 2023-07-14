import argparse
import os
import base64

import torch
from PIL import Image

from src.inference import create_ram_extractor, create_clip_extractor
from src.db import StorageDB


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image-mode", action="store_const", dest="type", const="image", default="image")
    group.add_argument("--text-mode", action="store_const", dest="type", const="text")
    parser.add_argument("--image", type=str, help="path to image file")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--tags", type=str, nargs="+", help="tags to query nearest neighbors. Use --show-tags to print options.")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--show-tags", action="store_true", help="show all available tags")
    parser.add_argument("--model-path", type=str, default=None, help="path to RAM model (.pth) file")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    with open("./ram/data/ram_tag_list.txt", "r", encoding="utf-8") as f:
        valid_tags = set(f.read().splitlines())

    if args.show_tags:
        print("--> available tags:")
        print("\n".join(sorted(valid_tags)))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db = StorageDB(args.db)
    index = db.create_index()

    if args.type == "image":
        if args.image is None:
            raise ValueError("--image must be provided")
        if args.model_path is None:
            raise ValueError("--model-path must be provided")

        img = Image.open(args.image)
        extractor = create_ram_extractor(args.model_path, device=device)
        tags = extractor([img])[0]
        input_tags = set(args.tags or [])
        tags = tags | (input_tags & valid_tags)
        top_ids = index.find_knn_tags(tags, args.top_k)
        imgs = [db.retrieve_img(img_id) for img_id, _ in top_ids]
        with open("output.html", "w") as out_html:
            for extension, img_bytes in imgs:
                out_html.write(f"<img src='data:image/{extension};base64,{base64.b64encode(img_bytes).decode('utf-8')}' /> <hr>\n")
        print("saved HTML with results to ./output.html")

    else:
        _, clip_extractor = create_clip_extractor(device)
        text_feats = clip_extractor([args.text])
        top_ids = index.find_knn_clip(text_feats, args.top_k)
        imgs = [db.retrieve_img(img_id) for img_id, _ in top_ids]
        with open("output.html", "w") as out_html:
            for extension, img_bytes in imgs:
                out_html.write(f"<img src='data:image/{extension};base64,{base64.b64encode(img_bytes).decode('utf-8')}' /> <hr>\n")
        print("saved HTML with results to ./output.html")


if __name__ == '__main__':
    main()
