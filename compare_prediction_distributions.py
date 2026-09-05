from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model import ModelConfig, RoPETransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and layer-wise RoPE next-token distributions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--lengths", type=int, nargs="+", default=(1024, 2048, 4096, 8192))
    parser.add_argument("--training-context", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
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


def read_chunk_losses(path: Path) -> dict[tuple[str, int], float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["book_id"], int(row["chunk_index"])): float(row["loss"])
            for row in csv.DictReader(handle)
        }


def select_chunk(
    result_directory: Path,
    checkpoint_stem: str,
    length: int,
    training_context: int,
) -> tuple[str, int, float]:
    baseline_path = result_directory / f"pg19_ctx{length}_{checkpoint_stem}_chunks.csv"
    layerwise_path = result_directory / f"pg19_ctx{length}_{checkpoint_stem}_layerwise_chunks.csv"
    if not baseline_path.is_file() or not layerwise_path.is_file():
        raise FileNotFoundError(
            f"Missing chunk-level evaluation files for context length {length}: "
            f"{baseline_path} and {layerwise_path}"
        )

    baseline = read_chunk_losses(baseline_path)
    layerwise = read_chunk_losses(layerwise_path)
    common = sorted(baseline.keys() & layerwise.keys())
    if not common:
        raise ValueError(f"No matching chunks for context length {length}")

    if length <= training_context:
        book_id, chunk_index = common[0]
    else:
        book_id, chunk_index = max(
            common,
            key=lambda item: baseline[item] - layerwise[item],
        )
    return book_id, chunk_index, baseline[(book_id, chunk_index)] - layerwise[(book_id, chunk_index)]


def load_chunk(data_directory: Path, book_id: str, chunk_index: int) -> torch.Tensor:
    metadata_path = data_directory / f"{book_id}.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunk_length = int(metadata["chunk_length"])
    num_chunks = int(metadata["num_chunks"])
    if chunk_index >= num_chunks:
        raise IndexError(f"Chunk {chunk_index} is outside {book_id}, which has {num_chunks} chunks")
    tokens = np.memmap(data_directory / f"{book_id}.bin", dtype=np.uint16, mode="r")
    chunks = tokens.reshape(num_chunks, chunk_length)
    return torch.from_numpy(np.asarray(chunks[chunk_index], dtype=np.int64)).unsqueeze(0)


def set_scaling(model: RoPETransformer, scaling_type: str) -> None:
    model.config.rope_scaling_type = scaling_type
    for block in model.blocks:
        block.attention.rotary.scaling_type = scaling_type


def token_losses_and_logits(
    model: RoPETransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        logits, _ = model(inputs)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
    return losses.float().cpu(), logits


def target_rank(probabilities: torch.Tensor, target_id: int) -> int:
    return int((probabilities > probabilities[target_id]).sum().item()) + 1


def display_token(tokenizer: tiktoken.Encoding, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    return repr(text)[1:-1].replace("|", "\\|")


def context_excerpt(tokenizer: tiktoken.Encoding, token_ids: torch.Tensor, position: int) -> str:
    start = max(0, position - 48)
    text = tokenizer.decode(token_ids[0, start : position + 2].tolist())
    return text.replace("\r", "").replace("\n", "\\n")


def distribution_metrics(
    baseline_probabilities: torch.Tensor,
    layerwise_probabilities: torch.Tensor,
) -> tuple[float, float, float]:
    epsilon = torch.finfo(torch.float32).tiny
    baseline = baseline_probabilities.clamp_min(epsilon)
    layerwise = layerwise_probabilities.clamp_min(epsilon)
    midpoint = 0.5 * (baseline + layerwise)
    js_divergence = 0.5 * (
        torch.sum(baseline * (baseline.log() - midpoint.log()))
        + torch.sum(layerwise * (layerwise.log() - midpoint.log()))
    )
    baseline_entropy = -torch.sum(baseline * baseline.log())
    layerwise_entropy = -torch.sum(layerwise * layerwise.log())
    return (
        float(js_divergence.item()),
        float(baseline_entropy.item()),
        float(layerwise_entropy.item()),
    )


def write_distribution_svg(result: dict[str, object], output_path: Path) -> None:
    rows = result["tokens"]
    width = 1100
    left = 250
    plot_width = 720
    row_height = 42
    height = 120 + len(rows) * row_height

    def bar_width(probability: float) -> float:
        log_probability = max(-12.0, math.log10(max(probability, 1e-12)))
        return (log_probability + 12.0) / 12.0 * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif}.label{font-size:13px}.small{font-size:11px;fill:#444}.title{font-size:20px;font-weight:700}</style>',
        f'<text x="20" y="30" class="title">Context {result["context_length"]}: Baseline vs V2</text>',
        '<text x="20" y="53" class="small">Bar length uses log10(probability), clipped at 1e-12. Blue: baseline; orange: V2.</text>',
    ]
    for tick in range(-12, 1, 3):
        x = left + (tick + 12) / 12 * plot_width
        svg.append(f'<line x1="{x:.1f}" y1="70" x2="{x:.1f}" y2="{height - 25}" stroke="#dddddd"/>')
        svg.append(f'<text x="{x:.1f}" y="67" text-anchor="middle" class="small">10^{tick}</text>')

    for index, row in enumerate(rows):
        y = 86 + index * row_height
        token = html.escape(str(row["token"]).replace("\\|", "|"))
        target_marker = " ★" if row["is_target"] else ""
        baseline_probability = float(row["baseline_probability"])
        layerwise_probability = float(row["layerwise_probability"])
        svg.extend(
            [
                f'<text x="20" y="{y + 16}" class="label">{token}{target_marker}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width(baseline_probability):.1f}" height="13" fill="#4C78A8"/>',
                f'<rect x="{left}" y="{y + 17}" width="{bar_width(layerwise_probability):.1f}" height="13" fill="#F58518"/>',
                f'<text x="{left + plot_width + 10}" y="{y + 11}" class="small">{baseline_probability:.3g}</text>',
                f'<text x="{left + plot_width + 10}" y="{y + 28}" class="small">{layerwise_probability:.3g}</text>',
            ]
        )
    svg.append('</svg>')
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    result_directory = args.result_dir.resolve()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = remove_compile_prefix(checkpoint["model"])
    tokenizer = tiktoken.get_encoding("gpt2")
    results: list[dict[str, object]] = []

    for context_length in args.lengths:
        data_directory = data_root / f"pg19_{context_length}"
        book_id, chunk_index, chunk_loss_improvement = select_chunk(
            result_directory,
            checkpoint_path.stem,
            context_length,
            args.training_context,
        )
        chunk = load_chunk(data_directory, book_id, chunk_index)
        inputs = chunk[:, :-1].to(device, non_blocking=True)
        targets = chunk[:, 1:].to(device, non_blocking=True)

        model_config = ModelConfig(**checkpoint["model_config"])
        model_config.block_size = context_length
        model_config.rope_training_context = args.training_context
        model_config.rope_scaling_type = "none"
        model = RoPETransformer(model_config)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device=device, dtype=dtype)

        set_scaling(model, "none")
        baseline_losses, baseline_logits = token_losses_and_logits(
            model, inputs, targets, device, dtype
        )
        del baseline_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

        set_scaling(model, "layerwise")
        layerwise_losses, layerwise_logits = token_losses_and_logits(
            model, inputs, targets, device, dtype
        )
        if context_length > args.training_context:
            candidate_start = args.training_context
            improvements = baseline_losses[candidate_start:] - layerwise_losses[candidate_start:]
            selected_position = candidate_start + int(torch.argmax(improvements).item())
        else:
            selected_position = context_length - 1

        layerwise_probabilities = torch.softmax(
            layerwise_logits[0, selected_position].float(), dim=-1
        ).cpu()
        del layerwise_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

        set_scaling(model, "none")
        _, baseline_logits = token_losses_and_logits(model, inputs, targets, device, dtype)
        baseline_probabilities = torch.softmax(
            baseline_logits[0, selected_position].float(), dim=-1
        ).cpu()
        del baseline_logits
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        target_id = int(targets[0, selected_position].item())
        js_divergence, baseline_entropy, layerwise_entropy = distribution_metrics(
            baseline_probabilities,
            layerwise_probabilities,
        )
        top_ids = set(torch.topk(baseline_probabilities, args.top_k).indices.tolist())
        top_ids.update(torch.topk(layerwise_probabilities, args.top_k).indices.tolist())
        top_ids.add(target_id)
        token_rows = []
        for token_id in sorted(
            top_ids,
            key=lambda item: max(
                float(baseline_probabilities[item]),
                float(layerwise_probabilities[item]),
            ),
            reverse=True,
        ):
            token_rows.append(
                {
                    "token_id": token_id,
                    "token": display_token(tokenizer, token_id),
                    "is_target": token_id == target_id,
                    "baseline_probability": float(baseline_probabilities[token_id]),
                    "layerwise_probability": float(layerwise_probabilities[token_id]),
                    "probability_delta": float(
                        layerwise_probabilities[token_id] - baseline_probabilities[token_id]
                    ),
                    "baseline_rank": target_rank(baseline_probabilities, token_id),
                    "layerwise_rank": target_rank(layerwise_probabilities, token_id),
                }
            )

        result = {
            "context_length": context_length,
            "book_id": book_id,
            "chunk_index": chunk_index,
            "chunk_mean_loss_improvement": chunk_loss_improvement,
            "prediction_position": selected_position + 1,
            "is_extrapolated": selected_position >= args.training_context,
            "target_id": target_id,
            "target_token": display_token(tokenizer, target_id),
            "context_excerpt": context_excerpt(tokenizer, chunk, selected_position),
            "baseline": {
                "target_loss": float(baseline_losses[selected_position]),
                "target_probability": float(baseline_probabilities[target_id]),
                "target_rank": target_rank(baseline_probabilities, target_id),
                "entropy_nats": baseline_entropy,
            },
            "layerwise": {
                "target_loss": float(layerwise_losses[selected_position]),
                "target_probability": float(layerwise_probabilities[target_id]),
                "target_rank": target_rank(layerwise_probabilities, target_id),
                "entropy_nats": layerwise_entropy,
            },
            "target_loss_improvement": float(
                baseline_losses[selected_position] - layerwise_losses[selected_position]
            ),
            "jensen_shannon_divergence_nats": js_divergence,
            "tokens": token_rows,
        }
        results.append(result)
        print(
            f"length={context_length}, book={book_id}, chunk={chunk_index}, "
            f"position={selected_position + 1}, target={result['target_token']!r}, "
            f"baseline_p={result['baseline']['target_probability']:.6g}, "
            f"layerwise_p={result['layerwise']['target_probability']:.6g}"
        )

    result_directory.mkdir(parents=True, exist_ok=True)
    json_path = result_directory / "pg19_baseline_vs_layerwise_prediction_samples.json"
    markdown_path = result_directory / "pg19_baseline_vs_layerwise_prediction_samples.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Baseline 与 V2 的预测分布样本",
        "",
        "对于超过训练长度的上下文，首先选择 V2 相对 baseline 平均 loss 改善最大的 chunk，再从外推区域选择目标 token loss 改善最大的位置。1024 长度没有发生缩放，因此使用第一个 chunk 的最后一个预测位置作为不变性对照。",
        "",
        "> 这些样本是为了观察两种方法差异而有意挑选的诊断样本，并不代表数据集上的平均表现。",
        "",
        "| 上下文长度 | 预测位置 | 真实 token | Baseline 概率/排名 | V2 概率/排名 | JS divergence |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        baseline = result["baseline"]
        layerwise = result["layerwise"]
        markdown_lines.append(
            f"| {result['context_length']} | {result['prediction_position']} | "
            f"`{result['target_token']}` | "
            f"{baseline['target_probability']:.6g} / {baseline['target_rank']} | "
            f"{layerwise['target_probability']:.6g} / {layerwise['target_rank']} | "
            f"{result['jensen_shannon_divergence_nats']:.6f} |"
        )
    markdown_lines.append("")
    for result in results:
        baseline = result["baseline"]
        layerwise = result["layerwise"]
        svg_name = f"pg19_ctx{result['context_length']}_baseline_vs_layerwise_distribution.svg"
        write_distribution_svg(result, result_directory / svg_name)
        markdown_lines.extend(
            [
                f"## 上下文长度 {result['context_length']}",
                "",
                f"- 书籍/chunk：`{result['book_id']}` / `{result['chunk_index']}`",
                f"- 预测位置：`{result['prediction_position']}`",
                f"- 是否属于外推位置：`{result['is_extrapolated']}`",
                f"- 真实下一个 token：`{result['target_token']}`（ID `{result['target_id']}`）",
                f"- Baseline 目标概率/排名：`{baseline['target_probability']:.6g}` / `{baseline['target_rank']}`",
                f"- V2 目标概率/排名：`{layerwise['target_probability']:.6g}` / `{layerwise['target_rank']}`",
                f"- 目标 token loss 改善：`{result['target_loss_improvement']:.6f}` nats",
                f"- Jensen-Shannon divergence：`{result['jensen_shannon_divergence_nats']:.6f}` nats",
                f"- 上下文片段：`{str(result['context_excerpt']).replace('`', '\\`')}`",
                f"- 分布图：[{svg_name}]({svg_name})",
                "",
                "| Token | ID | 真实目标 | Baseline 概率 | V2 概率 | V2 - baseline | Baseline 排名 | V2 排名 |",
                "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["tokens"]:
            markdown_lines.append(
                f"| `{row['token']}` | {row['token_id']} | "
                f"{'是' if row['is_target'] else ''} | "
                f"{row['baseline_probability']:.6g} | "
                f"{row['layerwise_probability']:.6g} | "
                f"{row['probability_delta']:+.6g} | "
                f"{row['baseline_rank']} | {row['layerwise_rank']} |"
            )
        markdown_lines.append("")

    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
