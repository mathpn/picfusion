import json
import sqlite3
import os
from hashlib import sha1
from typing import Optional

import numpy as np

from src.index import TagIndex, ClipIndex


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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY, features BLOB, FOREIGN KEY (id) REFERENCES images (id)
            )
            """
        )
        self.conn.commit()

    def insert_image(self, img_bytes: bytes, extension: str) -> str:
        img_id = sha1(img_bytes).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO images (id, extension, content) VALUES (?, ?, ?)",
            (img_id, extension, img_bytes),
        )
        return img_id

    def insert_tags(self, img_id: str, tags: set[str]):
        tags_json = json.dumps(list(tags))
        self.conn.execute(
            "INSERT INTO tags (id, tags) VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET tags = ?",
            (img_id, tags_json, tags_json),
        )

    def insert_features(self, img_id: str, features: np.ndarray) -> None:
        feature_bytes = features.tobytes()
        self.conn.execute(
            "INSERT INTO features (id, features) VALUES (?, ?) ON CONFLICT DO UPDATE SET features = ?",
            (img_id, feature_bytes, feature_bytes),
        )

    def create_tag_index(self) -> TagIndex:
        cur = self.conn.execute("SELECT id, tags FROM tags")
        img_ids, img_tags = [], []
        for img_id, tags in cur:
            img_ids.append(img_id)
            img_tags.append(set(json.loads(tags)))
        return TagIndex(img_ids, img_tags)

    def create_clip_index(self) -> ClipIndex:
        cur = self.conn.execute("SELECT id, features FROM features")
        img_ids, img_features = [], []
        for img_id, blob in cur:
            # XXX dtype
            feat = np.frombuffer(blob, dtype=np.float32)
            img_ids.append(img_id)
            img_features.append(feat)
        return ClipIndex(img_ids, np.stack(img_features))

    def retrieve_img(self, img_id: str) -> Optional[tuple[str, bytes]]:
        cur = self.conn.execute("SELECT extension, content FROM images WHERE id = ?", (img_id,))
        return cur.fetchone()

    def close(self):
        self.commit()
        self.conn.close()

    def commit(self):
        self.conn.commit()
