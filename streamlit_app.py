import base64
import io
import json
import math

import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageOps

from src.db import StorageDB
from src.inference import create_clip_extractor, create_ram_extractor

MODEL_PATH = "./models/ram_swin_large_14m.pth"
TAG_FILE_PATH = "./ram/data/ram_tag_list.txt"
DB_PATH = "./storage.db"


@st.cache_resource
def get_valid_tags(tag_file_path: str) -> set[str]:
    with open(tag_file_path, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())


@st.cache_resource
def get_searcher(model_path: str, db_path: str, valid_tags: set[str], device):
    clip_img_extractor, clip_txt_extractor = create_clip_extractor(device)
    ram_extractor = create_ram_extractor(model_path, device=device)
    db = StorageDB(db_path, read_mode=True)
    index = db.create_index()

    def do_search(img_file=None, input_tags=None, text=None, top_k: int = 10):
        if img_file is None and not input_tags and not text:
            return []

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

        img_info = [(img_id, *db.retrieve_small_img(img_id), score) for img_id, score in top_ids]
        return img_info

    return db, do_search


# check https://discuss.streamlit.io/t/automatic-download-select-and-download-file-with-single-button-click/15141/4
def download_button(object_to_download, download_filename):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    """
    if isinstance(object_to_download, bytes):
        pass

    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    dl_link = f"""
    <html>
    <head>
    <title>Start Auto Download file</title>
    <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script>
    $('<a href="data:image/jpeg;base64,{b64}" download="{download_filename}">')[0].click()
    </script>
    </head>
    </html>
    """
    return dl_link


def download_img(db: StorageDB, img_id: str):
    result = db.retrieve_img(img_id)
    if result is None:
        return
    extension, img_bytes = result
    components.html(
        download_button(img_bytes, f"{img_id}{extension}"),
        height=0,
    )
    print(img_id)


def build_callback(db: StorageDB, img_id: str):
    return lambda: download_img(db, img_id)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_tags = get_valid_tags(TAG_FILE_PATH)
    db, searcher = get_searcher(MODEL_PATH, DB_PATH, valid_tags, device)

    if "page" not in st.session_state:
        st.session_state["page"] = 0

    if "results" not in st.session_state:
        st.session_state["results"] = []

    n_cols = 4
    n_rows = 3
    items_per_page = n_cols * n_rows

    file_upload = st.file_uploader("Upload image:", accept_multiple_files=False)
    tags_selector = st.multiselect("Desired tags", options=sorted(valid_tags))
    txt_descriptor = st.text_input("Text description:", max_chars=256)
    top_k = st.slider(
        "Number of images to retrieve:", min_value=1, max_value=100, value=items_per_page
    )

    results = searcher(file_upload, tags_selector, txt_descriptor, top_k)
    st.session_state["results"] = results

    n_pages = math.ceil(len(st.session_state["results"]) / items_per_page)
    prev_col, _, next_col = st.columns([3, 10, 2.25])

    if next_col.button("Next page"):
        st.session_state["page"] = min(n_pages - 1, st.session_state["page"] + 1)

    if prev_col.button("Previous page"):
        st.session_state["page"] = max(0, st.session_state["page"] - 1)

    groups = []
    current_page = st.session_state["page"]
    img_info = st.session_state["results"]
    start_idx = current_page * items_per_page
    end_idx = min((current_page + 1) * items_per_page, len(img_info))
    page_img_info = img_info[start_idx:end_idx]
    for i in range(0, len(page_img_info), n_cols):
        groups.append(page_img_info[i : i + n_cols])

    for group in groups:
        cols = st.columns(n_cols)
        button_cols = st.columns(n_cols)
        for i, (img_id, _, img_bytes, score) in enumerate(group):
            img = Image.open(io.BytesIO(img_bytes))
            img = ImageOps.exif_transpose(img)
            cols[i].image(img, caption=f"{score:.3f}")
            callback = build_callback(db, img_id)
            button_cols[i].button(
                "Download", key=img_id, on_click=callback, use_container_width=True
            )


if __name__ == "__main__":
    main()
