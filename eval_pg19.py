from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig, RoPETransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a RoPE checkpoint on fixed-length PG-19 chunks."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--book-id")
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks per book; omit to evaluate every chunk.",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--training-context", type=int, default=1024)
    parser.add_argument(
        "--rope-scaling",
        choices=("none", "layerwise"),
        default="none",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def remove_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        (name[len(prefix) :] if name.startswith(prefix) else name): value
        for name, value in state_dict.items()
    }


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 80.0 else float("inf")


def discover_metadata(data_directory: Path, book_id: str | None) -> list[Path]:
    if book_id is not None:
        metadata_path = data_directory / f"{book_id}.meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata does not exist: {metadata_path}")
        return [metadata_path]
    paths = sorted(
        path
        for path in data_directory.glob("*.meta.json")
        if path.name != "dataset.meta.json"
    )
    if not paths:
        raise FileNotFoundError(f"No book metadata files found in {data_directory}")
    return paths


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    data_directory = args.data_dir.resolve()
    result_directory = args.result_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not data_directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("max-chunks must be positive")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")

    metadata_paths = discover_metadata(data_directory, args.book_id)
    metadata_entries = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    context_lengths = {int(metadata["context_length"]) for metadata in metadata_entries}
    if len(context_lengths) != 1:
        raise ValueError(f"Mixed context lengths in {data_directory}: {sorted(context_lengths)}")
    context_length = context_lengths.pop()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model_config.block_size = context_length
    model_config.rope_scaling_type = args.rope_scaling
    model_config.rope_training_context = args.training_context
    model = RoPETransformer(model_config)
    model.load_state_dict(remove_compile_prefix(checkpoint["model"]), strict=True)
    model.eval()
    model.to(device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    total_loss = 0.0
    total_tokens = 0
    in_distribution_loss = 0.0
    in_distribution_tokens = 0
    extrapolated_loss = 0.0
    extrapolated_tokens = 0
    chunk_results: list[dict[str, float | int | str]] = []
    book_results: list[dict[str, float | int | str | None]] = []
    evaluated_book_count = 0
    skipped_book_count = 0
    global_chunk_index = 0

    total_available_chunks = sum(
        min(int(metadata["num_chunks"]), args.max_chunks)
        if args.max_chunks is not None
        else int(metadata["num_chunks"])
        for metadata in metadata_entries
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"Data directory: {data_directory}")
    print(f"Books: {len(metadata_entries):,}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Context length: {context_length:,}")
    print(f"RoPE scaling: {args.rope_scaling}")
    layer_scales = [
        block.attention.rotary.scale_for_length(context_length)
        for block in model.blocks
    ]
    print("Layer scales: " + ", ".join(f"{scale:.4f}" for scale in layer_scales))
    print(f"Chunks to evaluate: {total_available_chunks:,}")

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode():
        for book_number, (metadata_path, metadata) in enumerate(
            zip(metadata_paths, metadata_entries), start=1
        ):
            book_id = str(metadata.get("book_id", metadata_path.name.removesuffix(".meta.json")))
            available_chunks = int(metadata["num_chunks"])
            chunks_to_evaluate = (
                min(available_chunks, args.max_chunks)
                if args.max_chunks is not None
                else available_chunks
            )
            binary_path = data_directory / f"{book_id}.bin"
            if chunks_to_evaluate == 0:
                skipped_book_count += 1
                book_results.append(
                    {
                        "book_id": book_id,
                        "chunks": 0,
                        "tokens": 0,
                        "loss": None,
                        "perplexity": None,
                    }
                )
                print(f"Book {book_number}/{len(metadata_entries)} {book_id}: no full chunks, skipped")
                continue
            if not binary_path.is_file():
                raise FileNotFoundError(f"Binary data does not exist: {binary_path}")

            chunk_length = int(metadata["chunk_length"])
            expected_values = available_chunks * chunk_length
            tokens = np.memmap(binary_path, dtype=np.uint16, mode="r")
            if tokens.size != expected_values:
                raise ValueError(
                    f"{binary_path}: token count {tokens.size} does not match metadata {expected_values}"
                )
            chunks = tokens.reshape(available_chunks, chunk_length)
            book_loss = 0.0
            book_token_count = 0

            for chunk_index in range(chunks_to_evaluate):
                chunk = torch.from_numpy(
                    np.asarray(chunks[chunk_index], dtype=np.int64)
                ).unsqueeze(0)
                inputs = chunk[:, :-1].to(device, non_blocking=True)
                targets = chunk[:, 1:].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=autocast_enabled,
                ):
                    logits, _ = model(inputs)
                    token_losses = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="none",
                    ).reshape(1, context_length)

                loss_sum = token_losses.sum().item()
                token_count = token_losses.numel()
                total_loss += loss_sum
                total_tokens += token_count
                book_loss += loss_sum
                book_token_count += token_count
                split_position = min(args.training_context, context_length)
                if split_position > 0:
                    in_distribution_loss += token_losses[:, :split_position].sum().item()
                    in_distribution_tokens += split_position
                if split_position < context_length:
                    extrapolated_loss += token_losses[:, split_position:].sum().item()
                    extrapolated_tokens += context_length - split_position

                chunk_loss = token_losses.mean().item()
                chunk_results.append(
                    {
                        "book_id": book_id,
                        "chunk_index": chunk_index,
                        "loss": chunk_loss,
                        "perplexity": perplexity(chunk_loss),
                    }
                )
                global_chunk_index += 1
                if global_chunk_index == 1 or global_chunk_index % args.log_interval == 0:
                    print(
                        f"Progress {global_chunk_index}/{total_available_chunks}: "
                        f"book={book_id}, chunk={chunk_index}, "
                        f"loss={chunk_loss:.6f}, ppl={perplexity(chunk_loss):.4f}"
                    )

            evaluated_book_count += 1
            mean_book_loss = book_loss / book_token_count
            book_results.append(
                {
                    "book_id": book_id,
                    "chunks": chunks_to_evaluate,
                    "tokens": book_token_count,
                    "loss": mean_book_loss,
                    "perplexity": perplexity(mean_book_loss),
                }
            )
            del chunks
            del tokens

    if total_tokens == 0:
        raise ValueError("No chunks were evaluated")

    mean_loss = total_loss / total_tokens
    overall_perplexity = perplexity(mean_loss)
    in_distribution_metrics = None
    extrapolated_metrics = None
    peak_gb = None
    print(f"Overall: loss={mean_loss:.6f}, ppl={overall_perplexity:.4f}")
    if in_distribution_tokens:
        mean = in_distribution_loss / in_distribution_tokens
        in_distribution_metrics = {
            "start_position": 1,
            "end_position": min(args.training_context, context_length),
            "tokens": in_distribution_tokens,
            "loss": mean,
            "perplexity": perplexity(mean),
        }
        print(
            f"Positions 1-{min(args.training_context, context_length)}: "
            f"loss={mean:.6f}, ppl={perplexity(mean):.4f}"
        )
    if extrapolated_tokens:
        mean = extrapolated_loss / extrapolated_tokens
        extrapolated_metrics = {
            "start_position": args.training_context + 1,
            "end_position": context_length,
            "tokens": extrapolated_tokens,
            "loss": mean,
            "perplexity": perplexity(mean),
        }
        print(
            f"Positions {args.training_context + 1}-{context_length}: "
            f"loss={mean:.6f}, ppl={perplexity(mean):.4f}"
        )
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Peak CUDA memory: {peak_gb:.3f} GiB")

    result_directory.mkdir(parents=True, exist_ok=True)
    scope = args.book_id if args.book_id is not None else "pg19"
    scaling_suffix = "" if args.rope_scaling == "none" else f"_{args.rope_scaling}"
    result_stem = f"{scope}_ctx{context_length}_{checkpoint_path.stem}{scaling_suffix}"
    summary_path = result_directory / f"{result_stem}_summary.json"
    chunks_path = result_directory / f"{result_stem}_chunks.csv"
    books_path = result_directory / f"{result_stem}_books.csv"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "data_directory": str(data_directory),
        "book_id": args.book_id,
        "context_length": context_length,
        "training_context": args.training_context,
        "rope_scaling": args.rope_scaling,
        "layer_scales": layer_scales,
        "book_count": len(metadata_entries),
        "evaluated_book_count": evaluated_book_count,
        "skipped_book_count": skipped_book_count,
        "evaluated_chunks": global_chunk_index,
        "evaluated_tokens": total_tokens,
        "device": str(device),
        "dtype": str(dtype),
        "overall": {
            "loss": mean_loss,
            "perplexity": overall_perplexity,
        },
        "in_distribution": in_distribution_metrics,
        "extrapolated": extrapolated_metrics,
        "peak_cuda_memory_gb": peak_gb,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with chunks_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("book_id", "chunk_index", "loss", "perplexity"),
        )
        writer.writeheader()
        writer.writerows(chunk_results)
    with books_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("book_id", "chunks", "tokens", "loss", "perplexity"),
        )
        writer.writeheader()
        writer.writerows(book_results)

    print(f"Summary saved to: {summary_path}")
    print(f"Chunk metrics saved to: {chunks_path}")
    print(f"Book metrics saved to: {books_path}")


if __name__ == "__main__":
    main()
