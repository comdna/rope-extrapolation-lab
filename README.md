# RoPE 124M Pretraining Pipeline

这是一个从头训练的 GPT-style decoder-only Transformer，配置与论文中的 124M baseline 对齐：

- 12 Transformer layers
- hidden size 768
- 12 attention heads
- context length 1024
- GPT-2 tokenizer，vocabulary size 50,257
- RMSNorm
- RoPE positional encoding，base 10,000
- AdamW，global batch size 64
- OpenWebText 风格自回归预训练

项目不依赖现成 GPT-2 权重。模型随机初始化，只使用 GPT-2 tokenizer。

## 安装

建议使用 Linux 或 WSL2、Python 3.10+ 和支持 CUDA 的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 不固定 PyTorch 的 CPU/CUDA 构建。请先根据机器安装对应版本，例如 CUDA 13.0：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据格式

将原始数据放进任意目录，例如：

```text
raw_openwebtext/
  shard-00000.jsonl
  shard-00001.jsonl.gz
  article-00001.txt
  article-00002.md
```

支持：

- `.jsonl` / `.jsonl.gz`：每行一个 JSON object，默认读取 `text` 字段；
- `.txt` / `.txt.gz`；
- `.md` / `.md.gz`。

如果存在 `train/` 和 `val/`（或 `validation/`）子目录，则按显式划分处理。否则按 document 随机划分，默认验证集比例为 0.1%。

## 一条命令跑完整 pipeline

```bash
python train.py \
  --config configs/rope_124m.json \
  --input-dir /path/to/raw_openwebtext \
  --data-dir data/openwebtext \
  --out-dir out/rope_124m
```

如果 `data/openwebtext/train.bin` 和 `val.bin` 不存在，`train.py` 会自动完成 GPT-2 tokenization 和二进制数据生成；如果已经存在，则直接开始或恢复训练。

## 直接使用 Hugging Face OpenWebText

无需手工下载数据文件。下面的命令会流式读取 `Skylion007/openwebtext`，使用 GPT-2 tokenizer 生成本地二进制 token 数据，然后开始训练：

```powershell
python train.py `
  --config configs/rope_124m.json `
  --hf-dataset Skylion007/openwebtext `
  --hf-streaming `
  --data-dir data/openwebtext `
  --out-dir out/rope_124m
```

如需单独完成数据准备：

```powershell
python prepare_hf_data.py `
  --dataset Skylion007/openwebtext `
  --streaming `
  --output-dir data/openwebtext
```

第一次正式运行会遍历并 tokenize 数据集，因此开始训练前的数据准备可能持续较长时间。生成 `train.bin` 和 `val.bin` 后，后续启动会直接复用。

快速验证 Hugging Face 下载和字段是否正确：

```powershell
python prepare_hf_data.py `
  --dataset Skylion007/openwebtext `
  --output-dir .hf_smoke_data `
  --validation-fraction 0.2 `
  --max-documents 20
```

## 训练指标

训练期间控制台会输出：

- train loss 和 train PPL；
- validation loss 和 validation PPL；
- learning rate；
- gradient norm；
- tokens processed 和 tokens/second；
- GPU allocated、reserved 和 peak memory。

同时写入：

```text
out/rope_124m/
  metrics.jsonl
  metrics.csv
  tensorboard/
  best.pt
  latest.pt
```

启动 TensorBoard：

```powershell
tensorboard --logdir out/rope_124m/tensorboard
```

如不需要 TensorBoard：

```powershell
python train.py ... --no-tensorboard
```

## 多 GPU

使用 `torchrun` 启动 DDP：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/rope_124m.json \
  --input-dir /path/to/raw_openwebtext \
  --data-dir data/openwebtext \
  --out-dir out/rope_124m
```

程序根据 GPU 数量和 `micro_batch_size` 自动计算 gradient accumulation，使 global batch size 保持为 64。必须满足：

```text
global_batch_size % (micro_batch_size * world_size) == 0
```

## 显存不足

减小配置中的 `micro_batch_size`，例如从 4 改为 2 或 1。global batch size 不变，程序会自动增加 gradient accumulation。

也可以启用 activation checkpointing：

```bash
python train.py ... --gradient-checkpointing
```

## 从 checkpoint 恢复

默认会自动读取 `out-dir/latest.pt`。也可以明确指定：

```bash
python train.py ... --resume out/rope_124m/latest.pt
```

## 单独准备数据

```bash
python prepare_data.py \
  --input-dir /path/to/raw_openwebtext \
  --output-dir data/openwebtext
```

输出：

```text
data/openwebtext/
  train.bin
  val.bin
  meta.json
```

## 快速测试

快速测试使用小模型和随机 token，在 CPU 或 GPU 上完成前向、反向和 optimizer step：

```bash
python smoke_test.py
```

测试真实数据 pipeline：

```bash
python train.py \
  --config configs/smoke.json \
  --input-dir sample_data \
  --data-dir .smoke_data \
  --out-dir .smoke_out
```

`configs/smoke.json` 只用于 pipeline 测试，不是正式训练配置。

## 正式运行注意事项

- 论文没有报告具体 GPU、训练精度、gradient accumulation 或随机种子，本项目将这些部分实现为可配置项。
- `batch size=64` 按 global batch size 处理，更适合单卡和多卡复现。
- 正式训练不要使用 `--block-size` 覆盖论文的 1024。
- checkpoint 保存模型、optimizer、GradScaler、迭代数和配置，可以中断恢复。
- 数据准备采用流式写入，不需要把 OpenWebText 全部加载到内存。
