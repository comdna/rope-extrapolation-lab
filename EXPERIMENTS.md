# RoPE 124M Pretraining Pipeline

[Project Overview](README.md) | **English Experiments** | [中文实验](EXPERIMENTS_CN.md)

This project implements a GPT-style decoder-only Transformer trained from scratch, with a configuration aligned with the 124M baseline described in the paper:

- 12 Transformer layers
- Hidden size of 768
- 12 attention heads
- Context length of 1024
- GPT-2 tokenizer with a vocabulary size of 50,257
- RMSNorm
- RoPE positional encoding with base 10,000
- AdamW with a global batch size of 64
- OpenWebText-style autoregressive pretraining

The project does not use pretrained GPT-2 weights. The model is randomly initialized and only reuses the GPT-2 tokenizer.

## Installation

Linux or WSL2, Python 3.10+, and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` does not pin a CPU or CUDA build of PyTorch. Install the appropriate build for your machine first. For example, for CUDA 13.0:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Format

Place the raw data in any directory, for example:

```text
raw_openwebtext/
  shard-00000.jsonl
  shard-00001.jsonl.gz
  article-00001.txt
  article-00002.md
```

Supported formats:

- `.jsonl` / `.jsonl.gz`: one JSON object per line; the `text` field is read by default.
- `.txt` / `.txt.gz`.
- `.md` / `.md.gz`.

If `train/` and `val/` (or `validation/`) subdirectories exist, they are treated as an explicit split. Otherwise, documents are split randomly, with a default validation ratio of 0.1%.

## Run the Full Pipeline with One Command

```bash
python train.py \
  --config configs/rope_124m.json \
  --input-dir /path/to/raw_openwebtext \
  --data-dir data/openwebtext \
  --out-dir out/rope_124m
```

If `data/openwebtext/train.bin` and `val.bin` do not exist, `train.py` automatically performs GPT-2 tokenization and generates the binary datasets. If they already exist, training starts or resumes immediately.

## Use Hugging Face OpenWebText Directly

No manual dataset download is required. The following command streams `Skylion007/openwebtext`, tokenizes it with the GPT-2 tokenizer, generates local binary token data, and then starts training:

```powershell
python train.py `
  --config configs/rope_124m.json `
  --hf-dataset Skylion007/openwebtext `
  --hf-streaming `
  --data-dir data/openwebtext `
  --out-dir out/rope_124m
```

### Train While Streaming Data

This mode is recommended for the first training run on a server. With `--hf-online`, the program no longer waits for the entire OpenWebText dataset to be converted into `train.bin`. It first creates a fixed validation cache, then downloads documents from the Hugging Face stream on demand, tokenizes them in batches, concatenates them into continuous training sequences, and immediately sends them to the GPU:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_DISABLE_XET=1

python train.py \
  --config configs/rope_124m.json \
  --hf-dataset Skylion007/openwebtext \
  --hf-online \
  --online-validation-tokens 4000000 \
  --hf-shuffle-buffer 10000 \
  --tokenizer-batch-documents 64 \
  --hf-cache-dir /root/autodl-tmp/huggingface \
  --data-dir /root/autodl-tmp/openwebtext_online \
  --out-dir /root/autodl-tmp/rope_124m
```

In online mode, `data-dir` stores only:

```text
openwebtext_online/
  online_val.bin
  online_meta.json
```

- `online-validation-tokens` controls the target number of tokens in the fixed validation cache. The actual number may be slightly larger because data is written in document batches.
- `hf-shuffle-buffer` controls the buffer used for approximate random shuffling of the stream. Larger values generally improve randomness but require more host memory and increase initial sampling time.
- `tokenizer-batch-documents` controls the number of documents tokenized in each batch. On a slow network, the first training step may still wait for a complete Parquet shard to download.
- The console, `metrics.jsonl`, `metrics.csv`, and TensorBoard continue to record training loss/PPL, validation loss/PPL, throughput, and GPU memory usage.
- When online training resumes from a checkpoint, the model, optimizer, and iteration are restored, but the exact Hugging Face stream position is not. The stream restarts from the same randomized order and skips the validation documents again, so some training samples may be repeated. Use the offline `train.bin` mode when strict reproducibility is required.

To prepare the data separately:

```powershell
python prepare_hf_data.py `
  --dataset Skylion007/openwebtext `
  --streaming `
  --output-dir data/openwebtext
```

The first full run iterates through and tokenizes the dataset, so data preparation may take a long time before training begins. Later runs reuse the generated `train.bin` and `val.bin` files.

To quickly verify that Hugging Face downloads and dataset fields work correctly:

```powershell
python prepare_hf_data.py `
  --dataset Skylion007/openwebtext `
  --output-dir .hf_smoke_data `
  --validation-fraction 0.2 `
  --max-documents 20
```

## Training Metrics

During training, the console reports:

- Training loss and training PPL.
- Validation loss and validation PPL.
- Learning rate.
- Gradient norm.
- Tokens processed and tokens per second.
- Allocated, reserved, and peak GPU memory.

The following files are also written:

```text
out/rope_124m/
  metrics.jsonl
  metrics.csv
  tensorboard/
  best.pt
  latest.pt
```

Start TensorBoard with:

```powershell
tensorboard --logdir out/rope_124m/tensorboard
```

To disable TensorBoard:

```powershell
python train.py ... --no-tensorboard
```

## Multi-GPU Training

Use `torchrun` to launch DistributedDataParallel training:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/rope_124m.json \
  --input-dir /path/to/raw_openwebtext \
  --data-dir data/openwebtext \
  --out-dir out/rope_124m
```

The program automatically calculates gradient accumulation from the GPU count and `micro_batch_size` so that the global batch size remains 64. The following condition must hold:

```text
global_batch_size % (micro_batch_size * world_size) == 0
```

## Out of GPU Memory

Reduce `micro_batch_size` in the configuration, for example from 4 to 2 or 1. The global batch size remains unchanged, and the program automatically increases gradient accumulation.

Activation checkpointing can also be enabled:

```bash
python train.py ... --gradient-checkpointing
```

## Resume from a Checkpoint

By default, the program automatically loads `out-dir/latest.pt`. A checkpoint can also be specified explicitly:

```bash
python train.py ... --resume out/rope_124m/latest.pt
```

## Prepare Data Separately

```bash
python prepare_data.py \
  --input-dir /path/to/raw_openwebtext \
  --output-dir data/openwebtext
```

Output:

```text
data/openwebtext/
  train.bin
  val.bin
  meta.json
```

## Quick Tests

The smoke test uses a small model and random tokens to perform a forward pass, backward pass, and optimizer step on a CPU or GPU:

```bash
python smoke_test.py
```

To test the real data pipeline:

```bash
python train.py \
  --config configs/smoke.json \
  --input-dir sample_data \
  --data-dir .smoke_data \
  --out-dir .smoke_out
```

`configs/smoke.json` is only intended for pipeline testing, not full training.

## RoPE Length Extrapolation on PG-19

The `rope_124m/best.pt` checkpoint is evaluated on PG-19 test texts for length extrapolation. This checkpoint was saved at training iteration 12,500, and its training context length was 1024.

At evaluation time, `data/pg19` contains 10 `.txt` files. Each book is encoded separately with the GPT-2 tokenizer, without concatenating tokens across books. For a context length $L$, each raw chunk contains $L+1$ tokens: the first $L$ tokens are used as input, and the same sequence shifted by one token provides the $L$ next-token prediction labels. Windows do not overlap, and the final portion of a book is discarded if it contains fewer than $L+1$ tokens.

### Data Splitting

Generate data for a specific context length with:

```powershell
python prepare_pg19_eval.py `
  --input-dir data\pg19 `
  --output-dir data\pg19_2048 `
  --context-length 2048 `
  --overwrite
```

Replace `2048` with `1024`, `4096`, or `8192` to generate the corresponding data directory.

| Context length | Books | Complete chunks | Written tokens | Discarded tail tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 10 | 1,093 | 1,120,325 | 5,993 |
| 2048 | 10 | 545 | 1,116,705 | 9,613 |
| 4096 | 10 | 270 | 1,106,190 | 20,128 |
| 8192 | 10 | 133 | 1,089,669 | 36,649 |

### Evaluation Command

```powershell
python eval_pg19.py `
  --checkpoint ..\rope_124m\best.pt `
  --data-dir data\pg19_2048 `
  --training-context 1024 `
  --device auto `
  --dtype auto `
  --result-dir result
```

The evaluation was run on an NVIDIA GeForce RTX 5060 Laptop GPU using `bfloat16`. The results are shown below:

| Test length | Chunks | Overall PPL | Positions 1–1024 PPL | Extrapolated-position PPL | Peak CUDA memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1,093 | 149.01 | 149.01 | — | 0.63 GiB |
| 2048 | 545 | 157.72 | 150.39 | 165.40 | 1.01 GiB |
| 4096 | 270 | 241.16 | 151.16 | 281.79 | 1.78 GiB |
| 8192 | 133 | 358.10 | 149.45 | 405.72 | 3.31 GiB |

“Extrapolated-position PPL” refers to token perplexity over positions 1025 through the current test length. The PPL over the first 1024 positions remains nearly stable across test lengths, whereas the PPL beyond the training context increases substantially with test length. This indicates clear length-extrapolation degradation in the RoPE checkpoint.

### V2: Progressively Stronger Layer-Wise RoPE Scaling

Without modifying checkpoint parameters or performing additional training, different fixed RoPE scales are applied to Transformer layers at different depths. Let the training context length be $L_0=1024$, the current test length be $L$, and the model contain $N$ layers. The scaling factor for layer $l$ is defined as

$$
s_l=\left(\frac{L}{L_0}\right)^{\frac{l}{N-1}},
\qquad l=0,1,\ldots,N-1.
$$

The RoPE rotation angle is changed from $m\theta_i$ to

$$
\frac{m\theta_i}{s_l}.
$$

The shallowest layer therefore always uses $s_0=1$, the deepest layer uses the full length-extension ratio $s_{N-1}=L/L_0$, and the intermediate layers transition smoothly on a geometric schedule. This method adds no learnable parameters and does not alter the state dict, so the original RoPE checkpoint can be reused directly.

Run the layer-wise scaling evaluation with:

```powershell
python eval_pg19.py `
  --checkpoint ..\rope_124m\best.pt `
  --data-dir data\pg19_4096 `
  --training-context 1024 `
  --rope-scaling layerwise `
  --device auto `
  --dtype auto `
  --result-dir result
```

At length 1024, every layer has a scale of 1. The maximum output difference between the layer-wise scaling model and the baseline is 0, confirming checkpoint compatibility and unchanged computation within the training length. Complete results at different lengths are shown below:

| Test length | Baseline overall PPL | V2 overall PPL | Baseline first-1024 PPL | V2 first-1024 PPL | Baseline extrapolated PPL | V2 extrapolated PPL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 149.01 | 149.01 | 149.01 | 149.01 | — | — |
| 2048 | 157.72 | 164.34 | 150.39 | 171.77 | 165.40 | **157.23** |
| 4096 | 241.16 | 243.81 | 151.16 | 224.21 | 281.79 | **250.71** |
| 8192 | 358.10 | **346.51** | 149.45 | 299.20 | 405.72 | **353.86** |

The relative improvement of V2 in the extrapolated region is:

| Test length | Extrapolated PPL improvement |
| ---: | ---: |
| 2048 | 4.94% |
| 4096 | 11.03% |
| 8192 | 12.78% |

The results show that progressively stronger layer-wise scaling improves prediction performance beyond the training length, with larger gains at longer test lengths. At length 8192, the overall PPL also decreases from 358.10 to 346.51.

However, the current V2 calculates every layer's scale directly from the full input length. As a result, even the first 1024 positions of a long sequence use scaled RoPE. Their PPL degrades substantially as the test length increases, reaching 224.21 at length 4096 and 299.20 at length 8192. Simple layer-wise scaling therefore alleviates long-range extrapolation degradation but disrupts the positional distribution learned within the training context.

This result suggests that future variants should satisfy both of the following conditions:

- Preserve the original RoPE as much as possible in shallow layers or within the training-length region.
- Increase scaling gradually only in deeper layers, at lower frequencies, or at positions beyond 1024.

### Baseline vs. V2 Prediction Distributions

To inspect the behavior of the two methods at individual token positions, one sample was selected for each context length: 1024, 2048, 4096, and 8192. Every selected sample satisfies the following conditions:

- The highest-probability token from both the baseline and V2 is the actual next token.
- The target is not EOT, whitespace-only, or another non-readable token.
- The sample is selected from the final 25% of its chunk. For lengths greater than 1024, the selected position is in the extrapolated region.

| Context length | Prediction position | Target token | Baseline probability | V2 probability | Top-10 overlap | Probability overlap | TV distance | JS divergence |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 930 | ` to` | 0.976002 | 0.976002 | 10/10 | 1.000000 | 0.000000 | 0.000000 |
| 2048 | 1996 | `'t` | 0.870992 | 0.898657 | 6/10 | 0.920057 | 0.079943 | 0.026531 |
| 4096 | 3162 | `'t` | 0.746491 | 0.832525 | 4/10 | 0.844164 | 0.155836 | 0.046777 |
| 8192 | 6151 | `'t` | 0.890414 | 0.919635 | 7/10 | 0.958715 | 0.041286 | 0.007115 |

A probability overlap closer to 1 indicates more similar full-vocabulary distributions, while smaller total variation (TV) distance and Jensen-Shannon (JS) divergence indicate stronger agreement.

- At length 1024, every V2 scale is 1, so the two prediction distributions are identical.
- In the selected extrapolated samples at lengths 2048, 4096, and 8192, both models predict the correct token and V2 assigns it a higher probability.
- The length-4096 sample shows the largest difference. Only 4 of the top-10 tokens overlap, and the probability overlap is 0.844164. Although the top-1 prediction is unchanged, the ranking and probability allocation of the remaining candidates differ substantially.
- At length 8192, the probability overlap is 0.958715, so the two distributions remain relatively close at this position. These deliberately selected diagnostic samples do not represent average dataset behavior.

Full contexts, target-token ranks, and candidate probabilities are available in [`result/pg19_baseline_vs_layerwise_joint_correct_samples.md`](result/pg19_baseline_vs_layerwise_joint_correct_samples.md). Distribution plots for each context length are available here:

- [`1024`](result/pg19_ctx1024_joint_correct_distribution.svg)
- [`2048`](result/pg19_ctx2048_joint_correct_distribution.svg)
- [`4096`](result/pg19_ctx4096_joint_correct_distribution.svg)
- [`8192`](result/pg19_ctx8192_joint_correct_distribution.svg)

Result files are stored in `result`:

```text
result/
  pg19_best_length_summary.csv
  pg19_best_length_summary.json
  pg19_ctx1024_best_summary.json
  pg19_ctx1024_best_chunks.csv
  pg19_ctx1024_best_books.csv
  pg19_ctx2048_best_summary.json
  pg19_ctx2048_best_layerwise_summary.json
  pg19_best_baseline_vs_layerwise.csv
  pg19_best_baseline_vs_layerwise.json
  pg19_best_layerwise_length_summary.csv
  pg19_best_layerwise_length_summary.json
  pg19_baseline_vs_layerwise_joint_correct_samples.md
  pg19_baseline_vs_layerwise_joint_correct_samples.json
  pg19_ctx1024_joint_correct_distribution.svg
  pg19_ctx2048_joint_correct_distribution.svg
  pg19_ctx4096_joint_correct_distribution.svg
  pg19_ctx8192_joint_correct_distribution.svg
  ...
```

- `*_summary.json`: overall, within-training-length, and extrapolated-region loss/PPL, together with memory information.
- `*_chunks.csv`: loss and PPL for each chunk.
- `*_books.csv`: aggregated loss and PPL for each book.
- `pg19_best_length_summary.*`: comparison across the four context lengths.
- `*_layerwise_summary.json`: overall, within-training-length, and extrapolated-region metrics for V2 layer-wise scaling.
- `pg19_best_baseline_vs_layerwise.*`: direct comparison and relative changes between the baseline and V2.
- `pg19_best_layerwise_length_summary.*`: V2 summary across the four context lengths.
- `pg19_baseline_vs_layerwise_joint_correct_samples.*`: prediction probabilities, ranks, and distribution-agreement metrics when both models have the correct top-1 token.
- `pg19_ctx*_joint_correct_distribution.svg`: token-probability plots for the four context lengths.

Note that this checkpoint was trained for only 12,500 steps rather than the configured 100,000 steps. These results are therefore intended primarily to validate the evaluation pipeline and reveal length-degradation trends; they should not be treated as the final performance of a fully trained RoPE model.

## Notes for Full Training Runs

- The paper does not report the exact GPU, training precision, gradient accumulation, or random seed. These settings are configurable in this project.
- `batch size=64` is interpreted as the global batch size, making the implementation suitable for both single-GPU and multi-GPU reproduction.
- Do not use `--block-size` to override the paper's context length of 1024 during a full training run.
- Checkpoints store the model, optimizer, GradScaler, iteration number, and configuration, allowing interrupted runs to resume.
- Data preparation writes incrementally and does not require loading all of OpenWebText into memory.
