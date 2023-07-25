# Picfusion

Picfusion allows you to search through your images using other images as query, a set of tags, pure text, or any combination of them.

## How does it work?

This project is quite simple in its form and was built keeping the [KISS](https://en.wikipedia.org/wiki/KISS_principle) principle in mind. It's built using [PyTorch](https://pytorch.org/), [streamlit](https://streamlit.io/) and two models:

- this [CLIP model](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) available through [🤗 transformers](https://github.com/huggingface/transformers)
- the [Recognize Anything Model (RAM)](https://github.com/xinyu1205/recognize-anything) from [xinyu1205](https://github.com/xinyu1205)

RAM is used to label each images with a set of tags, while CLIP is used to extract a vector for each image. Both tags and features are stores in a local SQLite database. All the code needed for tag and feature extraction both from images and text is on _src/app.py_. All the code needed to store and load data from the SQLite database is on _src/db.py_.

At inference time, an in-memory index is built using sets to obtain a set overlap score for the tags and simple numpy is used to obtain cosine similarity for the features (image or text). All the indexing code is on _src/index.py_. When an image is provided, both CLIP features and RAM tags are extracted and used. Each one produces a score for each image, and the scores are combined into one before ranking through simple averaging. Of course, more elaborate strategies could be used. When only tags are provided, then only the set overlap score is used for ranking. Finally, when only text is provided, features are extracted using the CLIP model and the cosine similarity metric is used to rank the images.

A nice inference interface was build using [streamlit](https://streamlit.io/). Results are paginated and downsized version of the images are displayed to improve performance. When the _Download_ button is clicked, the original image file is downloaded.


## How to run it

### Requirements

Install the requirements:

```bash
pip install -r requirements.txt
```

Having a GPU is recommended, since both models are quite big (especially RAM). It's still possible to run it on CPU-only PyTorch installation, but adding images to the database will be quite slow.

1. Download the [RAM pretrained model](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) from huggingface and save the file to ./models/ram_swin_large_14m.pth
2. Run scripts/insert_images.py to insert images from a folder to the local database:

```bash
python scripts/insert_images.py --img-folder images/example --batch-size 16
```

you may have to adjust the batch size depending on your GPU/CPU memory.

3. Run the streamlit app

```bash
streamlit run streamlit_app.py
```

Please be aware that since all indexing is done in-memory, the script may crash if the database has a huge number of images.
