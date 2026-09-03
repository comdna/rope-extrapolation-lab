from __future__ import annotations

import argparse

from data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenWebText-style data for pretraining")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-text-field", default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_dataset(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        tokenizer_name=args.tokenizer,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        json_text_field=args.json_text_field,
    )
    print(
        f"Prepared {metadata.train_tokens:,} train tokens and "
        f"{metadata.val_tokens:,} validation tokens in {args.output_dir}"
    )


if __name__ == "__main__":
    main()

