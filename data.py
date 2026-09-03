from __future__ import annotations

import gzip
import json
import random
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import tiktoken
import torch
from tqdm import tqdm


SUPPORTED_SUFFIXES = {".txt", ".md", ".jsonl"}


@dataclass
class DatasetMetadata:
    tokenizer: str
    vocab_size: int
    dtype: str
    train_tokens: int
    val_tokens: int
    source_directory: str
    validation_fraction: float
    seed: int
    source_type: str = "local"
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None


def _logical_suffix(path: Path) -> str:
    if path.suffix == ".gz":
        return Path(path.stem).suffix.lower()
    return path.suffix.lower()


def discover_files(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and _logical_suffix(path) in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_documents(paths: Iterable[Path], json_text_field: str = "text") -> Iterator[str]:
    for path in paths:
        suffix = _logical_suffix(path)
        with _open_text(path) as source:
            if suffix == ".jsonl":
                for line_number, line in enumerate(source, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                    text = record.get(json_text_field)
                    if not isinstance(text, str):
                        raise ValueError(
                            f"Expected string field {json_text_field!r} at {path}:{line_number}"
                        )
                    if text.strip():
                        yield text
            else:
                for line in source:
                    text = line.strip()
                    if text:
                        yield text


def _explicit_split_files(input_directory: Path) -> tuple[list[Path], list[Path]] | None:
    train_directory = input_directory / "train"
    validation_directory = input_directory / "validation"
    if not validation_directory.exists():
        validation_directory = input_directory / "val"
    if train_directory.is_dir() and validation_directory.is_dir():
        return discover_files(train_directory), discover_files(validation_directory)

    all_files = discover_files(input_directory)
    train_files = [path for path in all_files if path.stem.lower().startswith("train")]
    validation_files = [
        path
        for path in all_files
        if path.stem.lower().startswith(("val", "validation"))
    ]
    if train_files and validation_files:
        return train_files, validation_files
    return None


class TokenWriter:
    def __init__(self, path: Path, typecode: str = "H", flush_tokens: int = 1_000_000) -> None:
        self.path = path
        self.handle = path.open("wb")
        self.buffer = array(typecode)
        self.flush_tokens = flush_tokens
        self.token_count = 0

    def write(self, token_ids: list[int]) -> None:
        self.buffer.extend(token_ids)
        self.token_count += len(token_ids)
        if len(self.buffer) >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.buffer.tofile(self.handle)
            self.buffer = array(self.buffer.typecode)

    def close(self) -> None:
        self.flush()
        self.handle.close()

    def __enter__(self) -> "TokenWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def prepare_dataset(
    input_directory: str | Path,
    output_directory: str | Path,
    tokenizer_name: str = "gpt2",
    validation_fraction: float = 0.001,
    seed: int = 1337,
    json_text_field: str = "text",
) -> DatasetMetadata:
    input_directory = Path(input_directory).resolve()
    output_directory = Path(output_directory).resolve()
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    if tokenizer.n_vocab > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocabulary does not fit in uint16")
    output_directory.mkdir(parents=True, exist_ok=True)
    train_path = output_directory / "train.bin"
    validation_path = output_directory / "val.bin"
    train_temp = output_directory / "train.bin.tmp"
    validation_temp = output_directory / "val.bin.tmp"

    explicit_splits = _explicit_split_files(input_directory)
    random_generator = random.Random(seed)
    document_count = 0

    with TokenWriter(train_temp) as train_writer, TokenWriter(validation_temp) as validation_writer:
        if explicit_splits is not None:
            train_files, validation_files = explicit_splits
            if not train_files or not validation_files:
                raise ValueError("Explicit train/validation split contains no supported files")
            split_sources = (
                (iter_documents(train_files, json_text_field), train_writer),
                (iter_documents(validation_files, json_text_field), validation_writer),
            )
            for documents, writer in split_sources:
                for document in documents:
                    token_ids = tokenizer.encode_ordinary(document)
                    token_ids.append(tokenizer.eot_token)
                    writer.write(token_ids)
                    document_count += 1
        else:
            files = discover_files(input_directory)
            if not files:
                raise ValueError(f"No supported data files found under {input_directory}")
            for document in iter_documents(files, json_text_field):
                token_ids = tokenizer.encode_ordinary(document)
                token_ids.append(tokenizer.eot_token)
                writer = validation_writer if random_generator.random() < validation_fraction else train_writer
                writer.write(token_ids)
                document_count += 1

        train_tokens = train_writer.token_count
        validation_tokens = validation_writer.token_count

    if document_count == 0:
        raise ValueError("No non-empty documents were found")
    if train_tokens == 0 or validation_tokens == 0:
        train_temp.unlink(missing_ok=True)
        validation_temp.unlink(missing_ok=True)
        raise ValueError(
            "Both train and validation splits must contain tokens. "
            "Use explicit train/val files or increase validation_fraction."
        )

    train_temp.replace(train_path)
    validation_temp.replace(validation_path)
    metadata = DatasetMetadata(
        tokenizer=tokenizer_name,
        vocab_size=tokenizer.n_vocab,
        dtype="uint16",
        train_tokens=train_tokens,
        val_tokens=validation_tokens,
        source_directory=str(input_directory),
        validation_fraction=validation_fraction,
        seed=seed,
        source_type="local",
    )
    (output_directory / "meta.json").write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def prepare_hf_dataset(
    dataset_name: str,
    output_directory: str | Path,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    text_field: str = "text",
    tokenizer_name: str = "gpt2",
    validation_fraction: float = 0.001,
    seed: int = 1337,
    streaming: bool = True,
    cache_directory: str | Path | None = None,
    token: str | None = None,
    max_documents: int | None = None,
) -> DatasetMetadata:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face loading requires the 'datasets' package. "
            "Install project dependencies before using --hf-dataset."
        ) from error

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    if tokenizer.n_vocab > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocabulary does not fit in uint16")

    load_kwargs = {
        "path": dataset_name,
        "split": dataset_split,
        "streaming": streaming,
    }
    if dataset_config:
        load_kwargs["name"] = dataset_config
    if cache_directory:
        load_kwargs["cache_dir"] = str(cache_directory)
    if token:
        load_kwargs["token"] = token
    print(
        f"Loading Hugging Face dataset {dataset_name!r} "
        f"split={dataset_split!r}, streaming={streaming}...",
        flush=True,
    )
    dataset = load_dataset(**load_kwargs)
    print("Dataset metadata resolved. Starting network streaming and CPU tokenization...", flush=True)

    train_path = output_directory / "train.bin"
    validation_path = output_directory / "val.bin"
    train_temp = output_directory / "train.bin.tmp"
    validation_temp = output_directory / "val.bin.tmp"
    random_generator = random.Random(seed)
    document_count = 0

    try:
        with TokenWriter(train_temp) as train_writer, TokenWriter(validation_temp) as validation_writer:
            progress = tqdm(
                dataset,
                total=max_documents,
                desc="Tokenizing HF documents",
                unit="doc",
                dynamic_ncols=True,
            )
            for record in progress:
                text = record.get(text_field)
                if not isinstance(text, str):
                    raise ValueError(
                        f"Dataset record does not contain a string field named {text_field!r}. "
                        f"Available fields: {sorted(record)}"
                    )
                if not text.strip():
                    continue
                token_ids = tokenizer.encode_ordinary(text)
                token_ids.append(tokenizer.eot_token)
                writer = validation_writer if random_generator.random() < validation_fraction else train_writer
                writer.write(token_ids)
                document_count += 1
                if document_count % 100 == 0:
                    progress.set_postfix(
                        train_tokens=f"{train_writer.token_count:,}",
                        val_tokens=f"{validation_writer.token_count:,}",
                        refresh=False,
                    )
                if max_documents is not None and document_count >= max_documents:
                    break
            progress.close()

            train_tokens = train_writer.token_count
            validation_tokens = validation_writer.token_count
    except Exception:
        train_temp.unlink(missing_ok=True)
        validation_temp.unlink(missing_ok=True)
        raise

    if document_count == 0:
        train_temp.unlink(missing_ok=True)
        validation_temp.unlink(missing_ok=True)
        raise ValueError("The Hugging Face dataset yielded no non-empty documents")
    if train_tokens == 0 or validation_tokens == 0:
        train_temp.unlink(missing_ok=True)
        validation_temp.unlink(missing_ok=True)
        raise ValueError(
            "Both train and validation splits must contain tokens. "
            "For a small test, increase --validation-fraction or --max-documents."
        )

    train_temp.replace(train_path)
    validation_temp.replace(validation_path)
    metadata = DatasetMetadata(
        tokenizer=tokenizer_name,
        vocab_size=tokenizer.n_vocab,
        dtype="uint16",
        train_tokens=train_tokens,
        val_tokens=validation_tokens,
        source_directory=f"hf://{dataset_name}",
        validation_fraction=validation_fraction,
        seed=seed,
        source_type="huggingface",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
    )
    (output_directory / "meta.json").write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


class BinaryTokenDataset:
    def __init__(self, directory: str | Path, block_size: int) -> None:
        self.directory = Path(directory)
        metadata_path = self.directory / "meta.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dtype_name = self.metadata.get("dtype", "uint16")
        if dtype_name not in {"uint16", "uint32"}:
            raise ValueError(f"Unsupported token dtype: {dtype_name}")
        self.dtype = np.uint16 if dtype_name == "uint16" else np.uint32
        self.block_size = block_size
        self.train = np.memmap(self.directory / "train.bin", dtype=self.dtype, mode="r")
        self.val = np.memmap(self.directory / "val.bin", dtype=self.dtype, mode="r")
        for split_name, split in (("train", self.train), ("val", self.val)):
            if len(split) <= block_size:
                raise ValueError(
                    f"{split_name} split has {len(split)} tokens; more than {block_size} are required"
                )

    def close(self) -> None:
        for split in (self.train, self.val):
            memory_map = getattr(split, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __enter__(self) -> "BinaryTokenDataset":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        starts = torch.randint(
            0,
            len(data) - self.block_size,
            (batch_size,),
            generator=generator,
        )
        inputs = torch.stack(
            [
                torch.from_numpy(np.array(data[index : index + self.block_size], dtype=np.int64))
                for index in starts.tolist()
            ]
        )
        targets = torch.stack(
            [
                torch.from_numpy(
                    np.array(data[index + 1 : index + 1 + self.block_size], dtype=np.int64)
                )
                for index in starts.tolist()
            ]
        )
        if device.type == "cuda":
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        return inputs, targets
