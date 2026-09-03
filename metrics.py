from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_COLUMNS = [
    "timestamp",
    "iteration",
    "split",
    "loss",
    "perplexity",
    "learning_rate",
    "grad_norm",
    "tokens_seen",
    "tokens_per_second",
    "elapsed_seconds",
    "gpu_memory_allocated_gb",
    "gpu_memory_reserved_gb",
    "gpu_peak_memory_allocated_gb",
]


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 80.0 else float("inf")


class MetricsLogger:
    def __init__(self, output_directory: str | Path, tensorboard: bool = True) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_directory / "metrics.jsonl"
        self.csv_path = self.output_directory / "metrics.csv"
        self.csv_needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self.tensorboard_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "TensorBoard logging is enabled but tensorboard is not installed. "
                    "Install requirements.txt or pass --no-tensorboard."
                ) from error
            self.tensorboard_writer = SummaryWriter(self.output_directory / "tensorboard")

    def log(self, iteration: int, split: str, **metrics: Any) -> None:
        record: dict[str, Any] = {column: None for column in METRIC_COLUMNS}
        record.update(metrics)
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["iteration"] = iteration
        record["split"] = split

        with self.jsonl_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
        with self.csv_path.open("a", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=METRIC_COLUMNS)
            if self.csv_needs_header:
                writer.writeheader()
                self.csv_needs_header = False
            writer.writerow(record)

        if self.tensorboard_writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    self.tensorboard_writer.add_scalar(f"{split}/{name}", value, iteration)
            self.tensorboard_writer.flush()

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
