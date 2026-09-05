# RoPE 124M Pretraining Pipeline

[English](README.md) | **中文**

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

### 边加载边训练（推荐用于服务器首次训练）

加入 `--hf-online` 后，程序不再等待整个 OpenWebText 转换为 `train.bin`。它只先生成一个固定验证集缓存，然后从 Hugging Face 流中按需下载文档、批量分词、连续拼接为训练序列，并立即送入 GPU：

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

在线模式的 `data-dir` 只保存：

```text
openwebtext_online/
  online_val.bin
  online_meta.json
```

- `online-validation-tokens` 控制固定验证缓存的目标 token 数；实际数量可能略大，因为程序按一批文档写入。
- `hf-shuffle-buffer` 控制流式近似随机打乱的缓冲区；数值越大，随机性通常越好，但占用更多主机内存并延长首次取样时间。
- `tokenizer-batch-documents` 控制每次批量分词的文档数。网络较慢时首个训练 step 仍可能等待一个 parquet 分片下载完成。
- 控制台及 `metrics.jsonl`、`metrics.csv`、TensorBoard 仍会记录 train loss/PPL、validation loss/PPL、吞吐率和 GPU 显存。
- 在线训练恢复 checkpoint 时会恢复模型、optimizer 和 iteration，但不会精确恢复 Hugging Face 流的位置；数据流会从同一随机顺序开头重新跳过验证文档，部分训练样本可能重复。若要求严格可复现，请继续使用离线 `train.bin` 模式。

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

## RoPE 在 PG-19 上的长度外推实验

使用 `rope_124m/best.pt` 对 PG-19 test 文本进行长度外推评测。该 checkpoint 对应训练迭代数 12,500，训练时的上下文长度为 1024。

执行实验时，`data/pg19` 中共有 10 个 `.txt` 文件。每本书使用 GPT-2 tokenizer 单独编码，不跨书拼接。对于上下文长度 $L$，每个原始 chunk 包含 $L+1$ 个 token：前 $L$ 个 token 作为输入，后移一位后的 $L$ 个 token 作为 next-token prediction 标签。窗口之间不重叠，不足 $L+1$ token 的书籍尾部被丢弃。

### 数据切分

使用以下命令生成不同长度的数据：

```powershell
python prepare_pg19_eval.py `
  --input-dir data\pg19 `
  --output-dir data\pg19_2048 `
  --context-length 2048 `
  --overwrite
```

将 `2048` 替换为 `1024`、`4096` 或 `8192`，即可生成对应的数据目录。

| 上下文长度 | 书籍数 | 完整 chunks | 写入 token 数 | 丢弃尾部 token 数 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 10 | 1,093 | 1,120,325 | 5,993 |
| 2048 | 10 | 545 | 1,116,705 | 9,613 |
| 4096 | 10 | 270 | 1,106,190 | 20,128 |
| 8192 | 10 | 133 | 1,089,669 | 36,649 |

### 评测命令

```powershell
python eval_pg19.py `
  --checkpoint ..\rope_124m\best.pt `
  --data-dir data\pg19_2048 `
  --training-context 1024 `
  --device auto `
  --dtype auto `
  --result-dir result
```

评测使用 NVIDIA GeForce RTX 5060 Laptop GPU 和 `bfloat16`。结果如下：

| 测试长度 | chunks | 总体 PPL | 位置 1–1024 PPL | 外推位置 PPL | 峰值 CUDA 显存 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1,093 | 149.01 | 149.01 | — | 0.63 GiB |
| 2048 | 545 | 157.72 | 150.39 | 165.40 | 1.01 GiB |
| 4096 | 270 | 241.16 | 151.16 | 281.79 | 1.78 GiB |
| 8192 | 133 | 358.10 | 149.45 | 405.72 | 3.31 GiB |

其中，“外推位置 PPL”表示位置 1025 至当前测试长度范围内的 token 困惑度。可以看到，前 1024 个位置的 PPL 在不同测试长度下基本稳定，而超出训练上下文后的 PPL 随测试长度显著增加，说明该 RoPE checkpoint 出现了明显的长度外推退化。

### V2：逐层增强 RoPE 缩放

在不修改 checkpoint 参数、不进行额外训练的情况下，对不同深度的 Transformer 层使用不同的固定 RoPE scale。设训练上下文长度为 $L_0=1024$，当前测试长度为 $L$，模型共有 $N$ 层，则第 $l$ 层的缩放系数定义为

$$
s_l=\left(\frac{L}{L_0}\right)^{\frac{l}{N-1}},
\qquad l=0,1,\ldots,N-1.
$$

RoPE 旋转角由 $m\theta_i$ 改为

$$
\frac{m\theta_i}{s_l}.
$$

因此，最浅层始终保持 $s_0=1$，最深层使用完整的长度扩展比例 $s_{N-1}=L/L_0$，中间层按照几何形式平滑过渡。该方法不增加可学习参数，也不改变 state dict，可以直接复用原始 RoPE checkpoint。

使用以下参数运行逐层缩放评测：

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

在 1024 长度下，所有层的 scale 均为 1，逐层缩放模型与 baseline 的最大输出差为 0，说明 checkpoint 兼容且未改变训练长度内的基础计算。不同长度下的完整结果如下：

| 测试长度 | Baseline 总体 PPL | V2 总体 PPL | Baseline 前 1024 PPL | V2 前 1024 PPL | Baseline 外推 PPL | V2 外推 PPL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 149.01 | 149.01 | 149.01 | 149.01 | — | — |
| 2048 | 157.72 | 164.34 | 150.39 | 171.77 | 165.40 | **157.23** |
| 4096 | 241.16 | 243.81 | 151.16 | 224.21 | 281.79 | **250.71** |
| 8192 | 358.10 | **346.51** | 149.45 | 299.20 | 405.72 | **353.86** |

V2 对外推区域 PPL 的相对改善为：

| 测试长度 | 外推 PPL 改善 |
| ---: | ---: |
| 2048 | 4.94% |
| 4096 | 11.03% |
| 8192 | 12.78% |

结果表明，逐层增强缩放能够改善超出训练长度区域的预测性能，并且测试长度越长，改善越明显。在 8192 长度下，总体 PPL 也从 358.10 降至 346.51。

但当前 V2 直接根据完整输入长度计算每层 scale，因此即使是长序列中的前 1024 个位置，也会使用缩放后的 RoPE。随着测试长度增长，这部分 PPL 明显恶化：4096 长度下增加到 224.21，8192 长度下增加到 299.20。说明简单的逐层缩放虽然缓解了远距离位置外推问题，却破坏了模型在训练长度范围内已经学到的位置分布。

这一结果提示后续变体需要同时满足两个条件：

- 浅层或训练长度范围内尽可能保留原始 RoPE；
- 只在较深层、较低频率或超出 1024 的位置上逐渐增强缩放。

结果文件保存在 `result`：

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
  ...
```

- `*_summary.json`：总体、训练长度内和外推区域的 loss/PPL，以及显存信息；
- `*_chunks.csv`：每个 chunk 的 loss 和 PPL；
- `*_books.csv`：每本书的汇总 loss 和 PPL；
- `pg19_best_length_summary.*`：四种上下文长度的横向比较。
- `*_layerwise_summary.json`：V2 逐层缩放的总体、训练长度内和外推区域指标；
- `pg19_best_baseline_vs_layerwise.*`：baseline 与 V2 的直接对比及相对变化；
- `pg19_best_layerwise_length_summary.*`：V2 在四种上下文长度下的汇总结果。

需要注意，该 checkpoint 只训练到 12,500 steps，尚未完成配置中的 100,000 steps。因此，这组结果主要用于验证评测管线和观察长度退化趋势，不应直接视为完整训练后 RoPE 模型的最终性能。

## 正式运行注意事项

- 论文没有报告具体 GPU、训练精度、gradient accumulation 或随机种子，本项目将这些部分实现为可配置项。
- `batch size=64` 按 global batch size 处理，更适合单卡和多卡复现。
- 正式训练不要使用 `--block-size` 覆盖论文的 1024。
- checkpoint 保存模型、optimizer、GradScaler、迭代数和配置，可以中断恢复。
- 数据准备采用流式写入，不需要把 OpenWebText 全部加载到内存。
