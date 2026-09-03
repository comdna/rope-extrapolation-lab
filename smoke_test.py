from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from data import BinaryTokenDataset, prepare_dataset
from model import ModelConfig, RoPETransformer


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        raw = root / "raw"
        prepared = root / "prepared"
        raw.mkdir()
        (raw / "train.txt").write_text(
            "RoPE rotates query and key vectors using token positions.\n" * 32,
            encoding="utf-8",
        )
        (raw / "validation.txt").write_text(
            "Validation confirms that the complete pretraining pipeline works.\n" * 16,
            encoding="utf-8",
        )
        metadata = prepare_dataset(raw, prepared)
        config = ModelConfig(
            vocab_size=metadata.vocab_size,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
        )
        with BinaryTokenDataset(prepared, config.block_size) as dataset:
            generator = torch.Generator(device="cpu").manual_seed(11)
            model = RoPETransformer(config).to(device)
            optimizer = model.configure_optimizer(1e-3, 0.01, (0.9, 0.95), device.type)

            initial_loss = None
            final_loss = None
            for _ in range(2):
                inputs, targets = dataset.get_batch("train", 2, device, generator)
                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(inputs, targets)
                if initial_loss is None:
                    initial_loss = loss.item()
                if logits.shape != (2, config.block_size, config.vocab_size):
                    raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
                if not math.isfinite(loss.item()):
                    raise AssertionError("Loss is not finite")
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

        if initial_loss is None or final_loss is None:
            raise AssertionError("Smoke test did not execute any optimization steps")

        print(
            "SMOKE TEST PASSED | "
            f"device={device} | parameters={model.parameter_count():,} | "
            f"initial_loss={initial_loss:.4f} | final_loss={final_loss:.4f}"
        )


if __name__ == "__main__":
    main()
