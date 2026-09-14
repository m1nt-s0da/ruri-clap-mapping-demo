from argparse import ArgumentParser
import asyncio
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from translate._encoder import clap_text_encoder, ruriv3_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / ".var" / "output" / "model.onnx"


async def run_demo(
    model_path: Path, japanese_texts: list[str], english_texts: list[str]
) -> np.ndarray:
    if len(japanese_texts) != len(english_texts):
        raise ValueError("Japanese and English text counts must match")

    ruri = ruriv3_encoder()
    clap = clap_text_encoder()
    ruri_tokens, clap_tokens = await asyncio.gather(
        ruri.tokenize([f"文章: {text}" for text in japanese_texts]),
        clap.tokenize(english_texts),
    )
    ruri_embeddings, clap_embeddings = await asyncio.gather(
        ruri.encode(ruri_tokens.input_ids, ruri_tokens.attention_mask),
        clap.encode(clap_tokens.input_ids, clap_tokens.attention_mask),
    )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    ruri_embedding = np.asarray(ruri_embeddings, dtype=np.float32)
    clap_embedding = np.asarray(clap_embeddings, dtype=np.float32)
    projected_embedding = cast(
        np.ndarray,
        session.run([model_output.name], {model_input.name: ruri_embedding})[0],
    )
    projected_embedding_normalized = projected_embedding / np.linalg.norm(
        projected_embedding, axis=-1, keepdims=True
    )
    clap_embedding_normalized = clap_embedding / np.linalg.norm(
        clap_embedding, axis=-1, keepdims=True
    )
    return np.sum(projected_embedding_normalized * clap_embedding_normalized, axis=-1)


def main() -> None:
    parser = ArgumentParser(
        description="Calculate cosine similarity for a Japanese Ruri-v3 to English CLAP-text mapping."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the ONNX model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--ja",
        action="append",
        default=None,
        help="Japanese source text; repeat for a batch",
    )
    parser.add_argument(
        "--en",
        action="append",
        default=None,
        help="English target text; repeat for a batch",
    )
    args = parser.parse_args()
    japanese_texts = args.ja or ["今日は良い天気です。"]
    english_texts = args.en or ["The weather is nice today."]

    if not args.model.is_file():
        parser.error(
            f"ONNX model not found: {args.model}. Run `python -m convert.onnx` first."
        )
    if len(japanese_texts) != len(english_texts):
        parser.error("--ja and --en must be provided the same number of times")

    cosine_similarity = asyncio.run(run_demo(args.model, japanese_texts, english_texts))
    print(f"model: {args.model}")
    print(f"batch size: {len(japanese_texts)}")
    for index, (japanese_text, english_text) in enumerate(
        zip(japanese_texts, english_texts, strict=True), start=1
    ):
        print(f"pair {index} ruri-v3 input: 文章: {japanese_text}")
        print(f"pair {index} clap-text input: {english_text}")
        print(
            f"pair {index} mapping cosine similarity: {cosine_similarity[index - 1]:.8f}"
        )


if __name__ == "__main__":
    main()
