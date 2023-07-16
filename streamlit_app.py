import torch
import streamlit as st
from PIL import Image

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
    db = StorageDB("./storage.db")  # FIXME hardcoded
    index = db.create_index()
    print("hey")

    def do_search(img_file=None, input_tags=None, text=None, top_k: int = 10):
        print("searching")

        tags = set()
        feat, img_feat, txt_feat = None, None, None
        if img_file is not None:
            print(img_file)
            img = Image.open(img_file)
            print(img)
            tags = ram_extractor([img])[0]
            img_feat = clip_img_extractor([img])[0]

        if text:
            txt_feat = clip_txt_extractor([text])

        if img_feat is not None and txt_feat is not None:
            print("a")
            feat = (img_feat + txt_feat) / 2
        elif img_feat is not None and txt_feat is None:
            print("b")
            feat = img_feat
        else:
            print("c")
            feat = txt_feat

        input_tags = set(input_tags or [])
        tags = tags | (input_tags & valid_tags)
        if tags and feat is not None:
            print(1)
            top_ids = index.find_knn_combined(tags, feat, top_k)
        elif not tags and feat is not None:
            print(2)
            top_ids = index.find_knn_clip(feat, top_k)
        else:
            print(3)
            top_ids = index.find_knn_tags(tags, top_k)
        return top_ids

    return do_search


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_tags = get_valid_tags(TAG_FILE_PATH)
    searcher = get_searcher(MODEL_PATH, valid_tags, device)

    file_upload = st.file_uploader("Upload image:")
    tags_selector = st.multiselect("Desired tags", options=sorted(valid_tags))
    txt_descriptor = st.text_input("Text description:")
    search_button = st.button("Search")
    if search_button:
        result = searcher(file_upload, tags_selector, txt_descriptor)
        print(result)
        # TODO image grid


if __name__ == "__main__":
    main()
