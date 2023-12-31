import argparse
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Callable

import torch
from PIL import Image, ImageOps
from PIL.ExifTags import Base

from src.db import StorageDB
from src.inference import create_clip_extractor, create_ram_extractor


def process_batch(
    batch,
    ram_extractor: Callable,
    clip_extractor: Callable,
    db: StorageDB,
    small_img_height: int = 400,
) -> None:
    img_ids = []
    imgs = []
    for img_path, img_bytes, extension in batch:
        try:
            img = Image.open(BytesIO(img_bytes))
            exif_metadata = img.getexif() or {}
            img = ImageOps.exif_transpose(img)
            timestamp = None
            timestamp_str = exif_metadata.get(Base.DateTime.value)
            if timestamp_str is not None:
                timestamp = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")

            width, height = img.size
            small_img = img
            if height > small_img_height:
                width = int(small_img_height / height * width)
                small_img = img.resize((width, small_img_height))
            small_img_buffer = BytesIO()
            small_img.save(small_img_buffer, format="JPEG")
            small_img_buffer.seek(0)
        except Exception as exc:
            print(f"image file {img_path} failed to open: {exc}")
            continue
        img_id = db.insert_image(img_bytes, small_img_buffer.read(), extension, timestamp)
        img_ids.append(img_id)
        imgs.append(img)

    tags = ram_extractor(imgs)
    features = clip_extractor(imgs)
    for img_id, tag, feat in zip(img_ids, tags, features):
        db.insert_tags(img_id, tag)
        db.insert_features(img_id, feat)
    db.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-folder", type=str, required=True)
    parser.add_argument("--db-path", type=str, default="./storage.db")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-path", type=str, default="./models/ram_swin_large_14m.pth")
    parser.add_argument("--small-img-height", type=int, default=400)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db = StorageDB(args.db_path)
    ram_extractor = create_ram_extractor(args.model_path, device=device)
    clip_extractor, _ = create_clip_extractor(device)

    file_paths = os.listdir(args.img_folder)
    batch = []
    try:
        init = time.perf_counter()
        for i, file_path in enumerate(file_paths):
            img_path = f"{args.img_folder}/{file_path}"
            extension = os.path.splitext(img_path)[-1]
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            batch.append((img_path, img_bytes, extension))
            if len(batch) >= args.batch_size:
                process_batch(batch, ram_extractor, clip_extractor, db, args.small_img_height)
                batch = []
                print(
                    f"processed {i}/{len(file_paths)} images in {time.perf_counter() - init:.2f} s"
                )
                init = time.perf_counter()

        if batch:
            process_batch(batch, ram_extractor, clip_extractor, db)

        print("finished")

    except KeyboardInterrupt:
        pass
    finally:
        db.close()


if __name__ == "__main__":
    main()
