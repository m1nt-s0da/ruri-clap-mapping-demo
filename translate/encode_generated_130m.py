from dotenv import load_dotenv
import os
from tqdm import tqdm
from sqlite3 import connect
import asyncio
from ._encoder import ruriv3_encoder_130m as ruriv3_encoder, clap_text_encoder
import numpy as np
import json

load_dotenv()

DATASET_DB = os.environ["DATASET_DB"]


def open_db():
    conn = connect(DATASET_DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    conn.execute(
        "CREATE TABLE IF NOT EXISTS vectors_130m ("
        "id TEXT PRIMARY KEY NOT NULL, "
        "src_ja_anc BLOB NOT NULL, "
        "src_ja_pos BLOB NOT NULL, "
        "src_en_pos BLOB NOT NULL, "
        "dst_en_pos BLOB NOT NULL, "
        "dst_en_neg BLOB NOT NULL, "
        "FOREIGN KEY (id) REFERENCES samples(id) ON DELETE CASCADE"
        ") WITHOUT ROWID;"
    )
    return conn


async def main():
    conn = open_db()
    cur = conn.execute("SELECT id FROM samples")

    with open(".var/generated-prompt.json", "r", encoding="utf-8") as f:
        generated_data: list = json.load(f)

    src_rows = [
        (q["id"], q["ja_anc"], row["ja_pos"], row["en_pos"], q["en_neg"])
        for row in generated_data
        for q in row.get("queries", [])
    ]

    for row_id, ja_anc, ja_pos, en_pos, en_neg in tqdm(src_rows):
        with conn:
            cur = conn.execute("SELECT id FROM vectors_130m WHERE id = ?", (row_id,))
            if cur.fetchone():
                continue

            ruriv3 = ruriv3_encoder()
            clap_text = clap_text_encoder()

            [src_tokens, dst_tokens] = await asyncio.gather(
                ruriv3.tokenize(
                    [f"クエリ: {ja_anc}", f"文章: {ja_pos}", f"文章: {en_pos}"]
                ),
                clap_text.tokenize([en_pos, en_neg]),
            )

            [
                [
                    src_ja_anc,
                    src_ja_pos,
                    src_en_pos,
                ],
                [
                    dst_en_pos,
                    dst_en_neg,
                ],
            ] = await asyncio.gather(
                ruriv3.encode(src_tokens.input_ids, src_tokens.attention_mask),
                clap_text.encode(dst_tokens.input_ids, dst_tokens.attention_mask),
            )
            src_ja_anc: np.ndarray
            src_ja_pos: np.ndarray
            src_en_pos: np.ndarray
            dst_en_pos: np.ndarray
            dst_en_neg: np.ndarray

            conn.execute(
                "INSERT INTO samples (id, source, original_id, ja_anc, ja_pos, en_pos, en_neg) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row_id,
                    "gemma-generated",
                    f"gemma-generated:{row_id}",
                    ja_anc,
                    ja_pos,
                    en_pos,
                    en_neg,
                ),
            )

            conn.execute(
                "INSERT INTO vectors_130m (id, src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row_id,
                    src_ja_anc.astype(np.float32).tobytes(),
                    src_ja_pos.astype(np.float32).tobytes(),
                    src_en_pos.astype(np.float32).tobytes(),
                    dst_en_pos.astype(np.float32).tobytes(),
                    dst_en_neg.astype(np.float32).tobytes(),
                ),
            )

    print("Encoding complete.")


if __name__ == "__main__":
    asyncio.run(main())
