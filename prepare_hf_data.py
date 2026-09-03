from __future__ import annotations

import argparse
import os

from data import prepare_hf_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face Hub dataset for pretraining")
    parser.add_argument("--dataset", default="Skylion007/openwebtext")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_hf_dataset(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.split,
        text_field=args.text_field,
        output_directory=args.output_dir,
        tokenizer_name=args.tokenizer,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        streaming=args.streaming,
        cache_directory=args.cache_dir,
        token=os.environ.get(args.hf_token_env),
        max_documents=args.max_documents,
    )
    print(
        f"Prepared {metadata.dataset_name}: train={metadata.train_tokens:,}, "
        f"val={metadata.val_tokens:,} tokens in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
