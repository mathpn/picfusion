import json
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
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY, tags TEXT, FOREIGN KEY (id) REFERENCES images (id)
            )
            """
        )

    def insert_image(self, img_bytes: bytes, extension: str) -> str:
        img_id = sha1(img_bytes).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO images (id, extension, content) VALUES (?, ?, ?)",
            (img_id, extension, img_bytes),
        )
        return img_id

    def insert_tags(self, img_id: str, tags: list[str]):
        tags_json = json.dumps(tags)
        self.conn.execute(
            "INSERT INTO tags (id, tags) VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET tags = ?",
            (img_id, tags_json, tags_json),
        )

    def close(self):
        self.conn.commit()
