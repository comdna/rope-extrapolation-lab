from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data import BinaryTokenDataset, prepare_dataset, prepare_hf_dataset
from metrics import MetricsLogger, perplexity
from model import ModelConfig, RoPETransformer


@dataclass
class TrainingConfig:
    global_batch_size: int = 64
    micro_batch_size: int = 4
    max_iters: int = 100000
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_iters: int = 1000
    decay_iters: int = 100000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 10
    checkpoint_interval: int = 1000
    dtype: str = "auto"
    compile: bool = False
    tensorboard: bool = True
    save_checkpoints: bool = True
    seed: int = 1337


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    master: bool
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a 124M RoPE Transformer")
    parser.add_argument("--config", default="configs/rope_124m.json")
    parser.add_argument("--input-dir", help="Raw text/JSONL directory; used when binary data is absent")
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--data-dir", default="data/openwebtext")
    parser.add_argument("--out-dir", default="out/rope_124m")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument("--json-text-field", default="text")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_configs(path: str | Path) -> tuple[ModelConfig, TrainingConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model_keys = {field.name for field in fields(ModelConfig)}
    training_keys = {field.name for field in fields(TrainingConfig)}
    unknown_model = set(raw.get("model", {})) - model_keys
    unknown_training = set(raw.get("training", {})) - training_keys
    if unknown_model or unknown_training:
        raise ValueError(
            f"Unknown config keys: model={sorted(unknown_model)}, training={sorted(unknown_training)}"
        )
    return ModelConfig(**raw["model"]), TrainingConfig(**raw["training"])


def initialize_distributed(requested_device: str) -> DistributedContext:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed pretraining expects CUDA GPUs")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            master=rank == 0,
            device=torch.device("cuda", local_rank),
        )

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    return DistributedContext(False, 0, 0, 1, True, device)


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": None,
        "float32": None,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def learning_rate_at(iteration: int, config: TrainingConfig) -> float:
    if iteration < config.warmup_iters:
        return config.learning_rate * (iteration + 1) / max(1, config.warmup_iters)
    if iteration >= config.decay_iters:
        return config.min_learning_rate
    decay_ratio = (iteration - config.warmup_iters) / max(
        1, config.decay_iters - config.warmup_iters
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= context.world_size
    return value


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: BinaryTokenDataset,
    micro_batch_size: int,
    eval_iters: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    generator: torch.Generator,
    context: DistributedContext,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for index in range(eval_iters):
            inputs, targets = dataset.get_batch(split, micro_batch_size, device, generator)
            with autocast_context(device, amp_dtype):
                _, loss = model(inputs, targets)
            losses[index] = loss.detach()
        mean_loss = reduce_mean(losses.mean(), context)
        results[split] = mean_loss.item()
    model.train()
    return results


def save_checkpoint(
    path: Path,
    model: RoPETransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    iteration: int,
    best_val_loss: float,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "iteration": iteration,
            "best_val_loss": best_val_loss,
            "model_config": model.config_dict(),
            "training_config": vars(training_config),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    model_config, training_config = load_configs(args.config)
    if args.max_iters is not None:
        training_config.max_iters = args.max_iters
    if args.global_batch_size is not None:
        training_config.global_batch_size = args.global_batch_size
    if args.micro_batch_size is not None:
        training_config.micro_batch_size = args.micro_batch_size
    if args.block_size is not None:
        model_config.block_size = args.block_size
    if args.eval_interval is not None:
        training_config.eval_interval = args.eval_interval
    if args.eval_iters is not None:
        training_config.eval_iters = args.eval_iters
    if args.checkpoint_interval is not None:
        training_config.checkpoint_interval = args.checkpoint_interval
    if args.compile is not None:
        training_config.compile = args.compile
    if args.tensorboard is not None:
        training_config.tensorboard = args.tensorboard
    if args.save_checkpoints is not None:
        training_config.save_checkpoints = args.save_checkpoints
    model_config.validate()

    context = initialize_distributed(args.device)
    torch.manual_seed(training_config.seed + context.rank)
    if context.device.type == "cuda":
        torch.cuda.manual_seed(training_config.seed + context.rank)
    torch.set_float32_matmul_precision("high")

    data_directory = Path(args.data_dir)
    required_data_files = [data_directory / "train.bin", data_directory / "val.bin", data_directory / "meta.json"]
    if not all(path.exists() for path in required_data_files):
        if args.input_dir is not None and args.hf_dataset is not None:
            raise ValueError("Use either --input-dir or --hf-dataset, not both")
        if args.input_dir is None and args.hf_dataset is None:
            raise FileNotFoundError(
                "Prepared dataset is missing. Supply --input-dir or --hf-dataset."
            )
        if context.master:
            if args.hf_dataset is not None:
                metadata = prepare_hf_dataset(
                    dataset_name=args.hf_dataset,
                    dataset_config=args.hf_config,
                    dataset_split=args.hf_split,
                    text_field=args.hf_text_field,
                    output_directory=data_directory,
                    validation_fraction=args.validation_fraction,
                    seed=training_config.seed,
                    streaming=args.hf_streaming,
                    cache_directory=args.hf_cache_dir,
                    token=os.environ.get(args.hf_token_env),
                    max_documents=args.max_documents,
                )
            else:
                assert args.input_dir is not None
                metadata = prepare_dataset(
                    args.input_dir,
                    data_directory,
                    validation_fraction=args.validation_fraction,
                    seed=training_config.seed,
                    json_text_field=args.json_text_field,
                )
            print(
                f"Prepared dataset: train={metadata.train_tokens:,}, "
                f"val={metadata.val_tokens:,} tokens"
            )
        if context.enabled:
            dist.barrier()

    dataset = BinaryTokenDataset(data_directory, model_config.block_size)
    if int(dataset.metadata["vocab_size"]) != model_config.vocab_size:
        raise ValueError(
            f"Dataset vocabulary {dataset.metadata['vocab_size']} does not match "
            f"model vocabulary {model_config.vocab_size}"
        )

    denominator = training_config.micro_batch_size * context.world_size
    if training_config.global_batch_size % denominator != 0:
        raise ValueError(
            "global_batch_size must be divisible by micro_batch_size * world_size"
        )
    gradient_accumulation_steps = training_config.global_batch_size // denominator

    model = RoPETransformer(model_config).to(context.device)
    model.gradient_checkpointing = args.gradient_checkpointing
    optimizer = model.configure_optimizer(
        training_config.learning_rate,
        training_config.weight_decay,
        (training_config.beta1, training_config.beta2),
        context.device.type,
    )
    amp_dtype = resolve_amp_dtype(training_config.dtype, context.device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=context.device.type == "cuda" and amp_dtype == torch.float16,
    )

    output_directory = Path(args.out_dir)
    metrics_logger = MetricsLogger(output_directory, tensorboard=training_config.tensorboard) if context.master else None
    automatic_resume = output_directory / "latest.pt"
    resume_path = Path(args.resume) if args.resume else automatic_resume
    start_iteration = 0
    best_val_loss = float("inf")
    if resume_path.exists():
        checkpoint_data = torch.load(resume_path, map_location=context.device, weights_only=False)
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        scaler.load_state_dict(checkpoint_data.get("scaler", {}))
        start_iteration = int(checkpoint_data["iteration"]) + 1
        best_val_loss = float(checkpoint_data.get("best_val_loss", best_val_loss))
        if context.master:
            print(f"Resumed from {resume_path} at iteration {start_iteration}")

    training_model: torch.nn.Module = model
    if training_config.compile:
        training_model = cast(torch.nn.Module, torch.compile(training_model))
    if context.enabled:
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[context.local_rank],
        )

    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(training_config.seed + 1000 + context.rank)
    if context.master:
        print(f"Model parameters: {model.parameter_count():,}")
        print(f"Device: {context.device}; AMP dtype: {amp_dtype}")
        print(f"World size: {context.world_size}")
        print(f"Micro batch per GPU: {training_config.micro_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Global batch size: {training_config.global_batch_size}")
        print(
            f"Tokens per optimizer step: "
            f"{training_config.global_batch_size * model_config.block_size:,}"
        )

    training_model.train()
    run_start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_logged_iteration = start_iteration - 1
    for iteration in range(start_iteration, training_config.max_iters):
        learning_rate = learning_rate_at(iteration, training_config)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=context.device)
        for micro_step in range(gradient_accumulation_steps):
            inputs, targets = dataset.get_batch(
                "train",
                training_config.micro_batch_size,
                context.device,
                random_generator,
            )
            synchronization_context = nullcontext()
            if context.enabled and micro_step < gradient_accumulation_steps - 1:
                synchronization_context = cast(DistributedDataParallel, training_model).no_sync()
            with synchronization_context:
                with autocast_context(context.device, amp_dtype):
                    _, loss = training_model(inputs, targets)
                    scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.detach() / gradient_accumulation_steps

        grad_norm = None
        if training_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip).item()
            )
        scaler.step(optimizer)
        scaler.update()

        if iteration % training_config.log_interval == 0:
            mean_loss = reduce_mean(accumulated_loss, context)
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            current_time = time.perf_counter()
            elapsed = current_time - last_time
            last_time = current_time
            if context.master:
                assert metrics_logger is not None
                logged_iterations = iteration - last_logged_iteration
                tokens = (
                    training_config.global_batch_size
                    * model_config.block_size
                    * logged_iterations
                )
                mean_loss_value = mean_loss.item()
                train_perplexity = perplexity(mean_loss_value)
                tokens_per_second = tokens / max(elapsed, 1e-9)
                print(
                    f"iter {iteration:06d} | loss {mean_loss_value:.4f} | "
                    f"ppl {train_perplexity:.2f} | "
                    f"lr {learning_rate:.3e} | {elapsed * 1000:.1f} ms | "
                    f"{tokens_per_second:,.0f} tok/s"
                )
                memory_metrics = {}
                if context.device.type == "cuda":
                    memory_metrics = {
                        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(context.device) / 2**30,
                        "gpu_memory_reserved_gb": torch.cuda.memory_reserved(context.device) / 2**30,
                        "gpu_peak_memory_allocated_gb": torch.cuda.max_memory_allocated(context.device) / 2**30,
                    }
                    torch.cuda.reset_peak_memory_stats(context.device)
                metrics_logger.log(
                    iteration,
                    "train",
                    loss=mean_loss_value,
                    perplexity=train_perplexity,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                    tokens_seen=(iteration + 1) * training_config.global_batch_size * model_config.block_size,
                    tokens_per_second=tokens_per_second,
                    elapsed_seconds=time.perf_counter() - run_start_time,
                    **memory_metrics,
                )
            last_logged_iteration = iteration

        should_evaluate = iteration % training_config.eval_interval == 0 or iteration == training_config.max_iters - 1
        if should_evaluate:
            losses = estimate_loss(
                training_model,
                dataset,
                training_config.micro_batch_size,
                training_config.eval_iters,
                context.device,
                amp_dtype,
                random_generator,
                context,
            )
            if context.master:
                assert metrics_logger is not None
                train_eval_perplexity = perplexity(losses["train"])
                validation_perplexity = perplexity(losses["val"])
                print(
                    f"eval {iteration:06d} | train {losses['train']:.4f} | "
                    f"train ppl {train_eval_perplexity:.2f} | "
                    f"val {losses['val']:.4f} | val ppl {validation_perplexity:.2f}"
                )
                metrics_logger.log(
                    iteration,
                    "train_eval",
                    loss=losses["train"],
                    perplexity=train_eval_perplexity,
                    learning_rate=learning_rate,
                    tokens_seen=(iteration + 1) * training_config.global_batch_size * model_config.block_size,
                    elapsed_seconds=time.perf_counter() - run_start_time,
                )
                metrics_logger.log(
                    iteration,
                    "validation",
                    loss=losses["val"],
                    perplexity=validation_perplexity,
                    learning_rate=learning_rate,
                    tokens_seen=(iteration + 1) * training_config.global_batch_size * model_config.block_size,
                    elapsed_seconds=time.perf_counter() - run_start_time,
                )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if training_config.save_checkpoints:
                        save_checkpoint(
                            output_directory / "best.pt",
                            model,
                            optimizer,
                            scaler,
                            iteration,
                            best_val_loss,
                            model_config,
                            training_config,
                        )

        should_checkpoint = (
            iteration % training_config.checkpoint_interval == 0
            or iteration == training_config.max_iters - 1
        )
        if context.master and training_config.save_checkpoints and should_checkpoint:
            save_checkpoint(
                output_directory / "latest.pt",
                model,
                optimizer,
                scaler,
                iteration,
                best_val_loss,
                model_config,
                training_config,
            )

    dataset.close()
    if metrics_logger is not None:
        metrics_logger.close()
    if context.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
