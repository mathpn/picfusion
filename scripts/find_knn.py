import argparse
import base64

import torch
from PIL import Image

from src.db import StorageDB
from src.inference import create_clip_extractor, create_ram_extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="path to image file")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="tags to query nearest neighbors. Use --show-tags to print options.",
    )
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--show-tags", action="store_true", help="show all available tags")
    parser.add_argument(
        "--model-path", type=str, default=None, help="path to RAM model (.pth) file"
    )
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
    clip_img_extractor, clip_txt_extractor = create_clip_extractor(device)

    if args.image is None and args.text is None and args.tags is None:
        raise ValueError("at least one of --image --text or --tags must be provided")
    if args.model_path is None:
        raise ValueError("--model-path must be provided")

    tags = set()
    feat, img_feat, txt_feat = None, None, None
    if args.image is not None:
        img = Image.open(args.image)
        ram_extractor = create_ram_extractor(args.model_path, device=device)
        tags = ram_extractor([img])[0]
        img_feat = clip_img_extractor([img])[0]

    if args.text is not None:
        txt_feat = clip_txt_extractor([args.text])

    if img_feat is not None and txt_feat is not None:
        print("a")
        feat = (img_feat + txt_feat) / 2
    elif img_feat is not None and txt_feat is None:
        print("b")
        feat = img_feat
    else:
        print("c")
        feat = txt_feat

    input_tags = set(args.tags or [])
    tags = tags | (input_tags & valid_tags)
    if tags and feat is not None:
        print(1)
        top_ids = index.find_knn_combined(tags, feat, args.top_k)
    elif not tags and feat is not None:
        print(2)
        top_ids = index.find_knn_clip(feat, args.top_k)
    else:
        print(3)
        top_ids = index.find_knn_tags(tags, args.top_k)

    with open("output.html", "w", encoding="utf-8") as out_html:
        for img_id, score in top_ids:
            extension, img_bytes = db.retrieve_img(img_id)
            out_html.write(
                f"<img src='data:image/{extension};base64,{base64.b64encode(img_bytes).decode('utf-8')}' height='200'/> {score:.3f} <hr>\n"
            )
    print("saved HTML with results to ./output.html")


if __name__ == "__main__":
    main()
