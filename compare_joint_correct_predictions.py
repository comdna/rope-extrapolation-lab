from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tiktoken
import torch

from compare_prediction_distributions import (
    display_token,
    distribution_metrics,
    load_chunk,
    remove_compile_prefix,
    resolve_device,
    resolve_dtype,
    set_scaling,
    target_rank,
    token_losses_and_logits,
    write_distribution_svg,
)
from model import ModelConfig, RoPETransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find positions where baseline and layer-wise RoPE both predict the target token."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--lengths", type=int, nargs="+", default=(1024, 2048, 4096, 8192))
    parser.add_argument("--training-context", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-search-chunks", type=int, default=20)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    return parser.parse_args()


def read_ranked_chunks(
    result_directory: Path,
    checkpoint_stem: str,
    context_length: int,
) -> list[tuple[str, int]]:
    paths = (
        result_directory / f"pg19_ctx{context_length}_{checkpoint_stem}_chunks.csv",
        result_directory / f"pg19_ctx{context_length}_{checkpoint_stem}_layerwise_chunks.csv",
    )
    tables: list[dict[tuple[str, int], float]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing chunk evaluation file: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            tables.append(
                {
                    (row["book_id"], int(row["chunk_index"])): float(row["loss"])
                    for row in csv.DictReader(handle)
                }
            )
    common = tables[0].keys() & tables[1].keys()
    return sorted(common, key=lambda item: max(tables[0][item], tables[1][item]))


def is_meaningful_token(tokenizer: tiktoken.Encoding, token_id: int) -> bool:
    if token_id == tokenizer.eot_token:
        return False
    text = tokenizer.decode([token_id])
    return any(character.isalnum() for character in text)


def model_outputs(
    model: RoPETransformer,
    scaling_type: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_scaling(model, scaling_type)
    losses, logits = token_losses_and_logits(model, inputs, targets, device, dtype)
    top_ids = logits.argmax(dim=-1).squeeze(0).cpu()
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    target_log_probabilities = (
        target_logits.float() - torch.logsumexp(logits.float(), dim=-1)
    ).squeeze(0).cpu()
    return losses, top_ids, target_log_probabilities


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.max_search_chunks <= 0:
        raise ValueError("top-k and max-search-chunks must be positive")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("tail-fraction must be in (0, 1]")

    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    result_directory = args.result_dir.resolve()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = remove_compile_prefix(checkpoint["model"])
    tokenizer = tiktoken.get_encoding("gpt2")
    results: list[dict[str, object]] = []

    for context_length in args.lengths:
        model_config = ModelConfig(**checkpoint["model_config"])
        model_config.block_size = context_length
        model_config.rope_training_context = args.training_context
        model_config.rope_scaling_type = "none"
        model = RoPETransformer(model_config)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device=device, dtype=dtype)

        selected: dict[str, object] | None = None
        ranked_chunks = read_ranked_chunks(
            result_directory,
            checkpoint_path.stem,
            context_length,
        )
        search_start = max(
            args.training_context if context_length > args.training_context else 0,
            int(context_length * (1.0 - args.tail_fraction)),
        )
        for book_id, chunk_index in ranked_chunks[: args.max_search_chunks]:
            chunk = load_chunk(data_root / f"pg19_{context_length}", book_id, chunk_index)
            inputs = chunk[:, :-1].to(device, non_blocking=True)
            targets = chunk[:, 1:].to(device, non_blocking=True)

            baseline_losses, baseline_top_ids, baseline_target_log_probs = model_outputs(
                model, "none", inputs, targets, device, dtype
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            layerwise_losses, layerwise_top_ids, layerwise_target_log_probs = model_outputs(
                model, "layerwise", inputs, targets, device, dtype
            )

            target_ids = targets.squeeze(0).cpu()
            jointly_correct = (
                (baseline_top_ids == target_ids)
                & (layerwise_top_ids == target_ids)
            )
            if search_start:
                jointly_correct[:search_start] = False
            meaningful = torch.tensor(
                [is_meaningful_token(tokenizer, int(token_id)) for token_id in target_ids],
                dtype=torch.bool,
            )
            candidate_indices = torch.nonzero(jointly_correct & meaningful).flatten()
            if candidate_indices.numel() == 0:
                del inputs, targets
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            confidence = torch.minimum(
                baseline_target_log_probs[candidate_indices],
                layerwise_target_log_probs[candidate_indices],
            )
            position = int(candidate_indices[torch.argmax(confidence)].item())

            set_scaling(model, "layerwise")
            _, layerwise_logits = token_losses_and_logits(model, inputs, targets, device, dtype)
            layerwise_probabilities = torch.softmax(
                layerwise_logits[0, position].float(), dim=-1
            ).cpu()
            del layerwise_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

            set_scaling(model, "none")
            _, baseline_logits = token_losses_and_logits(model, inputs, targets, device, dtype)
            baseline_probabilities = torch.softmax(
                baseline_logits[0, position].float(), dim=-1
            ).cpu()
            del baseline_logits

            selected = {
                "book_id": book_id,
                "chunk_index": chunk_index,
                "position": position,
                "chunk": chunk,
                "target_id": int(target_ids[position]),
                "baseline_loss": float(baseline_losses[position]),
                "layerwise_loss": float(layerwise_losses[position]),
                "baseline_probabilities": baseline_probabilities,
                "layerwise_probabilities": layerwise_probabilities,
            }
            del inputs, targets
            break

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if selected is None:
            raise RuntimeError(
                f"No meaningful jointly correct token found for length {context_length} "
                f"in the first {args.max_search_chunks} ranked chunks"
            )

        baseline_probabilities = selected["baseline_probabilities"]
        layerwise_probabilities = selected["layerwise_probabilities"]
        target_id = int(selected["target_id"])
        js_divergence, baseline_entropy, layerwise_entropy = distribution_metrics(
            baseline_probabilities,
            layerwise_probabilities,
        )
        total_variation = 0.5 * torch.sum(
            torch.abs(baseline_probabilities - layerwise_probabilities)
        ).item()
        overlap_coefficient = torch.minimum(
            baseline_probabilities, layerwise_probabilities
        ).sum().item()
        baseline_top = torch.topk(baseline_probabilities, args.top_k).indices.tolist()
        layerwise_top = torch.topk(layerwise_probabilities, args.top_k).indices.tolist()
        top_k_intersection = len(set(baseline_top) & set(layerwise_top))
        top_k_union = len(set(baseline_top) | set(layerwise_top))

        token_ids = set(baseline_top) | set(layerwise_top) | {target_id}
        token_rows = []
        for token_id_row in sorted(
            token_ids,
            key=lambda item: max(
                float(baseline_probabilities[item]),
                float(layerwise_probabilities[item]),
            ),
            reverse=True,
        ):
            token_rows.append(
                {
                    "token_id": token_id_row,
                    "token": display_token(tokenizer, token_id_row),
                    "is_target": token_id_row == target_id,
                    "baseline_probability": float(baseline_probabilities[token_id_row]),
                    "layerwise_probability": float(layerwise_probabilities[token_id_row]),
                    "probability_delta": float(
                        layerwise_probabilities[token_id_row]
                        - baseline_probabilities[token_id_row]
                    ),
                    "baseline_rank": target_rank(baseline_probabilities, token_id_row),
                    "layerwise_rank": target_rank(layerwise_probabilities, token_id_row),
                }
            )

        position = int(selected["position"])
        chunk = selected["chunk"]
        excerpt_start = max(0, position - 48)
        excerpt = tokenizer.decode(chunk[0, excerpt_start : position + 2].tolist())
        result = {
            "context_length": context_length,
            "book_id": selected["book_id"],
            "chunk_index": selected["chunk_index"],
            "prediction_position": position + 1,
            "is_extrapolated": position >= args.training_context,
            "target_id": target_id,
            "target_token": display_token(tokenizer, target_id),
            "context_excerpt": excerpt.replace("\r", "").replace("\n", "\\n"),
            "baseline": {
                "target_loss": selected["baseline_loss"],
                "target_probability": float(baseline_probabilities[target_id]),
                "target_rank": target_rank(baseline_probabilities, target_id),
                "entropy_nats": baseline_entropy,
            },
            "layerwise": {
                "target_loss": selected["layerwise_loss"],
                "target_probability": float(layerwise_probabilities[target_id]),
                "target_rank": target_rank(layerwise_probabilities, target_id),
                "entropy_nats": layerwise_entropy,
            },
            "matching": {
                "both_top1_correct": True,
                "top_k": args.top_k,
                "top_k_intersection": top_k_intersection,
                "top_k_jaccard": top_k_intersection / top_k_union,
                "probability_overlap": overlap_coefficient,
                "total_variation_distance": total_variation,
                "jensen_shannon_divergence_nats": js_divergence,
            },
            "tokens": token_rows,
        }
        results.append(result)
        print(
            f"length={context_length}, book={result['book_id']}, chunk={result['chunk_index']}, "
            f"position={position + 1}, target={result['target_token']!r}, "
            f"baseline_p={result['baseline']['target_probability']:.6g}, "
            f"layerwise_p={result['layerwise']['target_probability']:.6g}, "
            f"top{args.top_k}_overlap={top_k_intersection}/{args.top_k}"
        )

    result_directory.mkdir(parents=True, exist_ok=True)
    stem = "pg19_baseline_vs_layerwise_joint_correct_samples"
    json_path = result_directory / f"{stem}.json"
    markdown_path = result_directory / f"{stem}.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Baseline 与 V2 共同预测正确的样本",
        "",
        "每个样本都满足：baseline 和 V2 的最高概率 token 均为真实下一个 token；目标 token 不是 EOT，也不是纯空白或其他无可读内容的 token。为避免只选到刚刚超过训练边界的位置，所有样本均从对应 chunk 的最后 25% 区域中搜索；长度超过 1024 时，这些位置也全部属于外推区域。",
        "",
        "| 长度 | 位置 | 目标 token | Baseline 概率 | V2 概率 | Top-k 重合 | 概率重合度 | TV 距离 | JS 散度 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        baseline = result["baseline"]
        layerwise = result["layerwise"]
        matching = result["matching"]
        lines.append(
            f"| {result['context_length']} | {result['prediction_position']} | "
            f"`{result['target_token']}` | {baseline['target_probability']:.6f} | "
            f"{layerwise['target_probability']:.6f} | "
            f"{matching['top_k_intersection']}/{matching['top_k']} | "
            f"{matching['probability_overlap']:.6f} | "
            f"{matching['total_variation_distance']:.6f} | "
            f"{matching['jensen_shannon_divergence_nats']:.6f} |"
        )
    lines.extend(
        [
            "",
            "匹配指标说明：概率重合度越接近 1 越一致；TV 距离和 JS 散度越接近 0 越一致。",
            "",
        ]
    )

    for result in results:
        svg_name = f"pg19_ctx{result['context_length']}_joint_correct_distribution.svg"
        write_distribution_svg(result, result_directory / svg_name)
        baseline = result["baseline"]
        layerwise = result["layerwise"]
        matching = result["matching"]
        lines.extend(
            [
                f"## 上下文长度 {result['context_length']}",
                "",
                f"- 书籍/chunk：`{result['book_id']}` / `{result['chunk_index']}`",
                f"- 预测位置：`{result['prediction_position']}`",
                f"- 是否属于外推位置：`{result['is_extrapolated']}`",
                f"- 真实 token 与双方 top-1：`{result['target_token']}`",
                f"- Baseline 目标概率：`{baseline['target_probability']:.6f}`",
                f"- V2 目标概率：`{layerwise['target_probability']:.6f}`",
                f"- Top-{matching['top_k']} 交集：`{matching['top_k_intersection']}/{matching['top_k']}`",
                f"- 概率重合度：`{matching['probability_overlap']:.6f}`",
                f"- TV 距离：`{matching['total_variation_distance']:.6f}`",
                f"- JS 散度：`{matching['jensen_shannon_divergence_nats']:.6f}`",
                f"- 上下文片段：`{str(result['context_excerpt']).replace('`', '\\`')}`",
                f"- 分布图：[{svg_name}]({svg_name})",
                "",
                "| Token | ID | 真实目标 | Baseline 概率 | V2 概率 | V2 - baseline | Baseline 排名 | V2 排名 |",
                "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["tokens"]:
            lines.append(
                f"| `{row['token']}` | {row['token_id']} | "
                f"{'是' if row['is_target'] else ''} | "
                f"{row['baseline_probability']:.6g} | "
                f"{row['layerwise_probability']:.6g} | "
                f"{row['probability_delta']:+.6g} | "
                f"{row['baseline_rank']} | {row['layerwise_rank']} |"
            )
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
