import argparse
import os
from io import BytesIO

import torch
from PIL import Image

from src.inference import create_ram_extractor
from src.db import StorageDB


def process_batch(batch, extractor, db: StorageDB) -> None:
    img_ids = []
    imgs = []
    for img_path, img_bytes, extension in batch:
        try:
            img = Image.open(BytesIO(img_bytes))
        except Exception as exc:
            print(f"image file {img_path} failed to open: {exc}")
        img_id = db.insert_image(img_bytes, extension)
        img_ids.append(img_id)
        imgs.append(img)

    tags = extractor(imgs)
    for img_id, tag in zip(img_ids, tags):
        db.insert_tags(img_id, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-folder", type=str, required=True)
    parser.add_argument("--db-path", type=str, default="./storage.db")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db = StorageDB(args.db_path)
    extractor = create_ram_extractor(args.model_path, device=device)

    file_paths = os.listdir(args.img_folder)
    batch = []
    try:
        for i, file_path in enumerate(file_paths):
            img_path = f"{args.img_folder}/{file_path}"
            extension = os.path.splitext(img_path)[-1]
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            batch.append((img_path, img_bytes, extension))
            if len(batch) >= args.batch_size:
                process_batch(batch, extractor, db)
                batch = []
                print(f"processed {i}/{len(file_paths)} images")

        if batch:
            process_batch(batch, extractor, db)

        print("finished")

    except KeyboardInterrupt:
        db.close()


if __name__ == "__main__":
    main()