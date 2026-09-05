from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize PG-19 books into fixed-length next-token evaluation chunks."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-file", type=Path)
    inputs.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_book(
    input_file: Path,
    output_directory: Path,
    context_length: int,
    tokenizer: tiktoken.Encoding,
    tokenizer_name: str,
    overwrite: bool,
) -> dict[str, int | str]:
    binary_path = output_directory / f"{input_file.stem}.bin"
    metadata_path = output_directory / f"{input_file.stem}.meta.json"
    if not overwrite and (binary_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Output already exists for {input_file.stem}; pass --overwrite to replace it"
        )

    text = input_file.read_text(encoding="utf-8")
    token_ids = tokenizer.encode_ordinary(text)
    chunk_length = context_length + 1
    chunk_count = len(token_ids) // chunk_length
    written_token_count = chunk_count * chunk_length
    dropped_token_count = len(token_ids) - written_token_count

    temporary_path = binary_path.with_suffix(binary_path.suffix + ".tmp")
    with temporary_path.open("wb") as output:
        for start in range(0, written_token_count, chunk_length):
            chunk = np.asarray(token_ids[start : start + chunk_length], dtype=np.uint16)
            chunk.tofile(output)
    temporary_path.replace(binary_path)

    metadata: dict[str, int | str] = {
        "source_file": str(input_file),
        "book_id": input_file.stem,
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.n_vocab,
        "dtype": "uint16",
        "context_length": context_length,
        "chunk_length": chunk_length,
        "source_tokens": len(token_ids),
        "num_chunks": chunk_count,
        "written_tokens": written_token_count,
        "dropped_tokens": dropped_token_count,
        "stride": chunk_length,
        "overlap": 0,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    output_directory = args.output_dir.resolve()
    if args.context_length <= 0:
        raise ValueError("context-length must be positive")

    if args.input_file is not None:
        input_file = args.input_file.resolve()
        if not input_file.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        input_files = [input_file]
        source = input_file
    else:
        input_directory = args.input_dir.resolve()
        if not input_directory.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
        input_files = sorted(input_directory.glob("*.txt"), key=lambda path: path.name)
        if not input_files:
            raise ValueError(f"No .txt files found in {input_directory}")
        source = input_directory

    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.get_encoding(args.tokenizer)
    if tokenizer.n_vocab > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocabulary does not fit in uint16")

    book_metadata = []
    total_source_tokens = 0
    total_written_tokens = 0
    total_dropped_tokens = 0
    total_chunks = 0
    books_with_chunks = 0

    for index, input_file in enumerate(input_files, start=1):
        metadata = prepare_book(
            input_file=input_file,
            output_directory=output_directory,
            context_length=args.context_length,
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer,
            overwrite=args.overwrite,
        )
        book_metadata.append(metadata)
        total_source_tokens += int(metadata["source_tokens"])
        total_written_tokens += int(metadata["written_tokens"])
        total_dropped_tokens += int(metadata["dropped_tokens"])
        total_chunks += int(metadata["num_chunks"])
        books_with_chunks += int(int(metadata["num_chunks"]) > 0)
        print(
            f"[{index}/{len(input_files)}] {input_file.name}: "
            f"tokens={int(metadata['source_tokens']):,}, "
            f"chunks={int(metadata['num_chunks']):,}, "
            f"dropped={int(metadata['dropped_tokens']):,}"
        )

    dataset_metadata = {
        "source": str(source),
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.n_vocab,
        "dtype": "uint16",
        "context_length": args.context_length,
        "chunk_length": args.context_length + 1,
        "book_count": len(input_files),
        "books_with_chunks": books_with_chunks,
        "total_source_tokens": total_source_tokens,
        "total_chunks": total_chunks,
        "total_written_tokens": total_written_tokens,
        "total_dropped_tokens": total_dropped_tokens,
        "stride": args.context_length + 1,
        "overlap": 0,
    }
    (output_directory / "dataset.meta.json").write_text(
        json.dumps(dataset_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("--- Dataset summary ---")
    print(f"Source: {source}")
    print(f"Output: {output_directory}")
    print(f"Books: {len(input_files):,}")
    print(f"Books with chunks: {books_with_chunks:,}")
    print(f"Source tokens: {total_source_tokens:,}")
    print(f"Context length: {args.context_length:,}")
    print(f"Full chunks: {total_chunks:,}")
    print(f"Written tokens: {total_written_tokens:,}")
    print(f"Dropped tail tokens: {total_dropped_tokens:,}")


if __name__ == "__main__":
    main()
