from torch.utils.data import Dataset
import sqlite3
import os
import numpy as np

__all__ = ["TranslateDataset"]


_db_connection: sqlite3.Connection | None = None


def _db_connect(db_path: str | None = None):
    global _db_connection
    if _db_connection is None:
        if db_path is None:
            db_path = os.environ["DATASET_DB"]
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _db_connection = conn
    return _db_connection


class TranslateDataset(Dataset):
    def __init__(self):
        conn = _db_connect()
        cur = conn.execute("SELECT id FROM vectors")
        self.data: list[str] = [row[0] for row in cur.fetchall()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]
        conn = _db_connect()
        cur = conn.execute(
            "SELECT src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg FROM vectors WHERE id = ?",
            (id,),
        )
        src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg = cur.fetchone()

        src_ja_anc = np.frombuffer(src_ja_anc, dtype=np.float32).copy()
        src_ja_pos = np.frombuffer(src_ja_pos, dtype=np.float32).copy()
        src_en_pos = np.frombuffer(src_en_pos, dtype=np.float32).copy()
        dst_en_pos = np.frombuffer(dst_en_pos, dtype=np.float32).copy()
        dst_en_neg = np.frombuffer(dst_en_neg, dtype=np.float32).copy()
        return src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg


if __name__ == "__main__":
    dataset = TranslateDataset()
    print(len(dataset))
    # print(dataset[0])
    src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg = dataset[0]
    print(
        f"{src_ja_anc.shape=}, {src_ja_pos.shape=}, {src_en_pos.shape=}, {dst_en_pos.shape=}, {dst_en_neg.shape=}"
    )
