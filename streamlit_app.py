import io

import streamlit as st
import torch
from PIL import Image, ImageOps

from src.db import StorageDB
from src.inference import create_clip_extractor, create_ram_extractor

MODEL_PATH = "./models/ram_swin_large_14m.pth"  # FIXME hardcoded
TAG_FILE_PATH = "./ram/data/ram_tag_list.txt"


@st.cache_resource
def get_valid_tags(tag_file_path: str) -> set[str]:
    with open(tag_file_path, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())


@st.cache_resource
def get_searcher(model_path: str, valid_tags: set[str], device):
    clip_img_extractor, clip_txt_extractor = create_clip_extractor(device)
    ram_extractor = create_ram_extractor(model_path, device=device)
    db = StorageDB("./storage.db", read_mode=True)  # FIXME hardcoded
    index = db.create_index()

    def do_search(img_file=None, input_tags=None, text=None, top_k: int = 10):
        tags = set()
        feat, img_feat, txt_feat = None, None, None
        if img_file is not None:
            img = Image.open(img_file)
            tags = ram_extractor([img])[0]
            img_feat = clip_img_extractor([img])[0]

        if text:
            txt_feat = clip_txt_extractor([text])

        if img_feat is not None and txt_feat is not None:
            feat = (img_feat + txt_feat) / 2
        elif img_feat is not None and txt_feat is None:
            feat = img_feat
        else:
            feat = txt_feat

        input_tags = set(input_tags or [])
        tags = tags | (input_tags & valid_tags)
        if tags and feat is not None:
            top_ids = index.find_knn_combined(tags, feat, top_k)
        elif not tags and feat is not None:
            top_ids = index.find_knn_clip(feat, top_k)
        else:
            top_ids = index.find_knn_tags(tags, top_k)

        img_info = [(*db.retrieve_img(img_id), score) for img_id, score in top_ids]
        return img_info

    return do_search


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_tags = get_valid_tags(TAG_FILE_PATH)
    searcher = get_searcher(MODEL_PATH, valid_tags, device)

    file_upload = st.file_uploader("Upload image:", accept_multiple_files=False)
    tags_selector = st.multiselect("Desired tags", options=sorted(valid_tags))
    txt_descriptor = st.text_input("Text description:", max_chars=256)
    top_k = st.slider("Number of images to retrieve:", min_value=1, max_value=100, value=12)
    search_button = st.button("Search Images")
    img_info = []
    if search_button:
        img_info = searcher(file_upload, tags_selector, txt_descriptor, top_k)

    groups = []
    n = 4
    for i in range(0, len(img_info), n):
        groups.append(img_info[i : i + n])

    for group in groups:
        cols = st.columns(n)
        for i, (_, img_bytes, score) in enumerate(group):
            img = Image.open(io.BytesIO(img_bytes))
            img = ImageOps.exif_transpose(img)
            cols[i].image(img, caption=f"{score:.3f}", output_format="JPEG")


if __name__ == "__main__":
    main()
