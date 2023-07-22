import json
import os
import sqlite3
from datetime import datetime
from hashlib import sha1
from typing import Optional

import numpy as np

from src.index import CompositeIndex


class StorageDB:
    def __init__(self, db_path: str, read_mode: bool = False):
        db_path = os.path.abspath(db_path)
        if read_mode:
            db_path = f"file:{db_path}?mode=ro"
        self.conn = sqlite3.connect(db_path, check_same_thread=not read_mode, uri=read_mode)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, extension TEXT, timestamp TEXT, bytes BLOB, small_bytes BLOB
            )
            """
        )
        # NOTE separate table for performance reasons
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

    # TODO insert full size and downsized versions
    def insert_image(
        self,
        img_bytes: bytes,
        small_img_bytes: bytes,
        extension: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        img_id = sha1(img_bytes).hexdigest()
        timestamp = timestamp.isoformat() if timestamp is not None else None
        self.conn.execute(
            "INSERT OR IGNORE INTO images (id, extension, timestamp, bytes, small_bytes) VALUES (?, ?, ?, ?, ?)",
            (img_id, extension, timestamp, img_bytes, small_img_bytes),
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

    def create_index(self) -> CompositeIndex:
        cur = self.conn.execute(
            "SELECT tags.id, tags, features FROM tags JOIN features ON tags.id = features.id"
        )
        img_ids, img_tags, features = [], [], []
        for img_id, tags, blob in cur:
            img_ids.append(img_id)
            img_tags.append(set(json.loads(tags)))
            features.append(np.frombuffer(blob, dtype=np.float32))
        features = np.stack(features)
        return CompositeIndex(img_ids, img_tags, features)

    def retrieve_small_img(self, img_id: str) -> Optional[tuple[str, bytes]]:
        cur = self.conn.execute("SELECT extension, small_bytes FROM images WHERE id = ?", (img_id,))
        return cur.fetchone()

    def retrieve_img(self, img_id: str) -> Optional[tuple[str, bytes]]:
        cur = self.conn.execute("SELECT extension, bytes FROM images WHERE id = ?", (img_id,))
        return cur.fetchone()

    def close(self):
        self.commit()
        self.conn.close()

    def commit(self):
        self.conn.commit()
