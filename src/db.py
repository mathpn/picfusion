import sqlite3
import os
from hashlib import sha1

from PIL import Image


class StorageDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, extension TEXT, content BLOB)"
        )
        # NOTE separate table for performance purposes
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY, tags TEXT, FOREIGN KEY (id) REFERENCES images (id)
            )
            """
        )

    def insert_image(self, img_path: str):
        extension = os.path.splitext(img_path)[-1]
        with open(img_path, "rb") as binary_file:
            img_bytes = open(binary_file, "rb")
        img_id = sha1(img_bytes).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO images (id, extension, content) VALUES (?, ?, ?)",
            (img_id, extension, img_bytes),
        )
        return img_id
