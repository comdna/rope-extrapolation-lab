from __future__ import annotations

import json
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tiktoken
import torch
from tqdm import tqdm


@dataclass
class OnlineValidationMetadata:
    tokenizer: str
    vocab_size: int
    dtype: str
    validation_tokens: int
    validation_documents: int
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    text_field: str
    shuffle_seed: int
    shuffle_buffer_size: int


def _load_stream(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    cache_directory: str | Path | None,
    token: str | None,
    shuffle_seed: int,
    shuffle_buffer_size: int,
):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Online Hugging Face training requires the 'datasets' package."
        ) from error

    load_kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": dataset_split,
        "streaming": True,
    }
    if dataset_config:
        load_kwargs["name"] = dataset_config
    if cache_directory:
        load_kwargs["cache_dir"] = str(cache_directory)
    if token:
        load_kwargs["token"] = token
    dataset = load_dataset(**load_kwargs)
    return dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)


def prepare_online_validation(
    dataset_name: str,
    output_directory: str | Path,
    target_tokens: int,
    block_size: int,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    text_field: str = "text",
    tokenizer_name: str = "gpt2",
    shuffle_seed: int = 1337,
    shuffle_buffer_size: int = 10_000,
    tokenizer_batch_documents: int = 64,
    cache_directory: str | Path | None = None,
    token: str | None = None,
) -> OnlineValidationMetadata:
    if target_tokens <= block_size:
        raise ValueError("target_tokens must be greater than block_size")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / "online_val.bin"
    temporary_path = output_directory / "online_val.bin.tmp"
    metadata_path = output_directory / "online_meta.json"
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    stream = _load_stream(
        dataset_name,
        dataset_config,
        dataset_split,
        cache_directory,
        token,
        shuffle_seed,
        shuffle_buffer_size,
    )
    iterator = iter(stream)
    token_buffer = array("H")
    validation_documents = 0
    progress = tqdm(total=target_tokens, desc="Preparing fixed validation tokens", unit="tok", dynamic_ncols=True)

    try:
        while len(token_buffer) < target_tokens:
            texts: list[str] = []
            while len(texts) < tokenizer_batch_documents:
                record = next(iterator)
                text = record.get(text_field)
                if not isinstance(text, str):
                    raise ValueError(
                        f"Dataset record does not contain string field {text_field!r}. "
                        f"Available fields: {sorted(record)}"
                    )
                if text.strip():
                    texts.append(text)
            encoded_documents = tokenizer.encode_ordinary_batch(texts)
            previous_count = len(token_buffer)
            for token_ids in encoded_documents:
                token_buffer.extend(token_ids)
                token_buffer.append(tokenizer.eot_token)
                validation_documents += 1
            progress.update(len(token_buffer) - previous_count)
        with temporary_path.open("wb") as output:
            token_buffer.tofile(output)
        temporary_path.replace(final_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        progress.close()

    metadata = OnlineValidationMetadata(
        tokenizer=tokenizer_name,
        vocab_size=tokenizer.n_vocab,
        dtype="uint16",
        validation_tokens=len(token_buffer),
        validation_documents=validation_documents,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        text_field=text_field,
        shuffle_seed=shuffle_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


class HFOnlineBatchProvider:
    supports_train_evaluation = False

    def __init__(
        self,
        data_directory: str | Path,
        block_size: int,
        batch_size: int,
        rank: int,
        world_size: int,
        cache_directory: str | Path | None = None,
        token: str | None = None,
        tokenizer_batch_documents: int = 64,
    ) -> None:
        self.data_directory = Path(data_directory)
        self.metadata = json.loads(
            (self.data_directory / "online_meta.json").read_text(encoding="utf-8")
        )
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.cache_directory = cache_directory
        self.token = token
        self.tokenizer_batch_documents = tokenizer_batch_documents
        self.tokenizer = tiktoken.get_encoding(self.metadata["tokenizer"])
        self.validation = np.memmap(
            self.data_directory / "online_val.bin",
            dtype=np.uint16,
            mode="r",
        )
        if len(self.validation) <= block_size:
            raise ValueError("Online validation cache is shorter than block_size")
        self.token_buffer: list[int] = []
        self.token_offset = 0
        self.stream_iterator = self._new_train_iterator()

    def _new_train_iterator(self) -> Iterator[dict[str, Any]]:
        dataset = _load_stream(
            self.metadata["dataset_name"],
            self.metadata.get("dataset_config"),
            self.metadata["dataset_split"],
            self.cache_directory,
            self.token,
            int(self.metadata["shuffle_seed"]),
            int(self.metadata["shuffle_buffer_size"]),
        )
        dataset = dataset.skip(int(self.metadata["validation_documents"]))
        if self.world_size > 1:
            dataset = dataset.shard(
                num_shards=self.world_size,
                index=self.rank,
                contiguous=False,
            )
        return iter(dataset)

    def _append_documents(self) -> None:
        texts: list[str] = []
        while len(texts) < self.tokenizer_batch_documents:
            try:
                record = next(self.stream_iterator)
            except StopIteration:
                self.stream_iterator = self._new_train_iterator()
                record = next(self.stream_iterator)
            text = record.get(self.metadata["text_field"])
            if not isinstance(text, str):
                raise ValueError(
                    f"Dataset record does not contain string field {self.metadata['text_field']!r}"
                )
            if text.strip():
                texts.append(text)
        for token_ids in self.tokenizer.encode_ordinary_batch(texts):
            self.token_buffer.extend(token_ids)
            self.token_buffer.append(self.tokenizer.eot_token)

    def _next_sequence(self) -> tuple[list[int], list[int]]:
        required = self.block_size + 1
        while len(self.token_buffer) - self.token_offset < required:
            self._append_documents()
        sequence = self.token_buffer[self.token_offset : self.token_offset + required]
        self.token_offset += self.block_size
        if self.token_offset >= 1_000_000:
            del self.token_buffer[: self.token_offset]
            self.token_offset = 0
        return sequence[:-1], sequence[1:]

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            if batch_size != self.batch_size:
                raise ValueError("Online training batch size changed after provider initialization")
            sequences = [self._next_sequence() for _ in range(batch_size)]
            inputs = torch.tensor([sequence[0] for sequence in sequences], dtype=torch.long)
            targets = torch.tensor([sequence[1] for sequence in sequences], dtype=torch.long)
        elif split == "val":
            starts = torch.randint(
                0,
                len(self.validation) - self.block_size,
                (batch_size,),
                generator=generator,
            )
            inputs = torch.stack(
                [
                    torch.from_numpy(
                        np.array(
                            self.validation[index : index + self.block_size],
                            dtype=np.int64,
                        )
                    )
                    for index in starts.tolist()
                ]
            )
            targets = torch.stack(
                [
                    torch.from_numpy(
                        np.array(
                            self.validation[index + 1 : index + 1 + self.block_size],
                            dtype=np.int64,
                        )
                    )
                    for index in starts.tolist()
                ]
            )
        else:
            raise ValueError(f"Unknown split: {split}")

        if device.type == "cuda":
            return (
                inputs.pin_memory().to(device, non_blocking=True),
                targets.pin_memory().to(device, non_blocking=True),
            )
        return inputs.to(device), targets.to(device)

    def close(self) -> None:
        memory_map = getattr(self.validation, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
