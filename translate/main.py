from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from sqlite3 import connect
from uuid import uuid7
import asyncio
from random import shuffle
from ._encoder import ruriv3_encoder, clap_text_encoder
import numpy as np

load_dotenv()

DATASET_DB = os.environ["DATASET_DB"]
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.environ["OPENAI_TRANSLATE_MODEL"]


def open_db():
    conn = connect(DATASET_DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    conn.execute(
        "CREATE TABLE IF NOT EXISTS samples ("
        "id TEXT PRIMARY KEY NOT NULL, "
        "source TEXT NOT NULL, "
        "original_id TEXT NOT NULL UNIQUE, "
        "ja_anc TEXT NOT NULL, "
        "ja_pos TEXT NOT NULL, "
        "en_pos TEXT NOT NULL, "
        "en_neg TEXT NOT NULL"
        ") WITHOUT ROWID;"
    )

    conn.execute(
        "CREATE TABLE IF NOT EXISTS vectors ("
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
    keys = [
        "auto-wiki-qa-nemotron",
        "jaquad",
        "jqara",
        "jsquad",
        "miracl",
        "mkqa",
        "mr-tydi",
        "nli",
        "quiz-no-mori",
        "quiz-works",
    ]
    datasets: dict[str, DatasetDict] = {}
    rows = 0
    for key in tqdm(keys):
        dataset: DatasetDict = load_dataset("cl-nagoya/ruri-v3-dataset-ft", key)
        rows += dataset["train"].num_rows
        datasets[key] = dataset
    all_rows = [
        (key, row) for key in keys for row in range(datasets[key]["train"].num_rows)
    ]
    shuffle(all_rows)
    print(f"Total rows: {rows}")

    client = AsyncOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    with open(os.path.dirname(__file__) + "/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    conn = open_db()

    async def translate(ja: str):
        completion = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=prompt,
            input=ja,
            reasoning={
                "effort": "none"  # Disables reasoning entirely on supported models
            },
        )

        return completion.output_text

    for dataset_key, row_id in tqdm(all_rows):
        dataset = datasets[dataset_key]

        row = dataset["train"][row_id]
        id: str = row["id"]
        ja_anc: str = row["anc"]
        ja_pos: str = row["pos"]
        negs: list[str] = row["neg"]
        cur = conn.execute("SELECT id FROM samples WHERE original_id = ?", (id,))
        query_row = cur.fetchone()
        if query_row:
            continue

        shuffle(negs)
        ja_neg: str = negs[0]

        [
            pos_en,
            neg_en,
        ] = await asyncio.gather(
            translate(ja_pos),
            translate(ja_neg),
        )

        ruriv3 = ruriv3_encoder()
        clap_text = clap_text_encoder()

        [src_tokens, dst_tokens] = await asyncio.gather(
            ruriv3.tokenize(
                [f"クエリ: {ja_anc}", f"文章: {ja_pos}", f"文章: {pos_en}"]
            ),
            clap_text.tokenize([pos_en, neg_en]),
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

        with conn:
            query_id = uuid7()
            conn.execute(
                "INSERT INTO samples (id, source, original_id, ja_anc, ja_pos, en_pos, en_neg) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(query_id), dataset_key, id, ja_anc, ja_pos, pos_en, neg_en),
            )

            conn.execute(
                "INSERT INTO vectors (id, src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(query_id),
                    src_ja_anc.astype(np.float32).tobytes(),
                    src_ja_pos.astype(np.float32).tobytes(),
                    src_en_pos.astype(np.float32).tobytes(),
                    dst_en_pos.astype(np.float32).tobytes(),
                    dst_en_neg.astype(np.float32).tobytes(),
                ),
            )


if __name__ == "__main__":
    asyncio.run(main())
