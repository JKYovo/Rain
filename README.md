# Rain: End-to-End 0.1B Chinese LLM Training

Rain 是一个从零实现的 0.1B 级中文 Decoder-only LLM 项目，覆盖 Tokenizer、Pretraining、SFT、GRPO、Evaluation 和 Inference 的完整训练链路。项目重点不是封装黑盒训练框架，而是用尽量清晰的 PyTorch 代码展示一个小型中文语言模型从数据处理到强化对齐的完整实验过程。

This project is designed for learning, research, and small-scale experimentation with Chinese language model training.

## Highlights

- **End-to-end pipeline**：Tokenizer → Pretraining → SFT → GRPO → Evaluation → Inference。
- **Custom Tokenizer**：基于 Byte-level BPE 训练 15K 词表，支持中文文本、chat template 和 `<think>` reasoning format。
- **Decoder-only Transformer**：实现 RMSNorm、RoPE、SwiGLU、Grouped Query Attention、KV Cache 和 embedding/LM head weight tying。
- **Training engineering**：支持 mixed precision、gradient accumulation、gradient clipping、warmup + cosine learning rate schedule、DDP、torch.compile 和 checkpoint resume。
- **Data pipeline**：支持 pretrain JSONL 转 `.bin/.meta`，SFT multi-turn conversation JSONL，以及 GRPO prompt-only JSONL。
- **GRPO alignment**：实现 Group Relative Policy Optimization，结合 reference model KL penalty、format reward 和 DeepSeek Judge reward。
- **Evaluation loop**：支持 C3/XCOPA multiple-choice benchmark，以及 mini benchmark + LLM-as-Judge 生成式评测。

## Results

| Stage | Metric | Result |
| --- | --- | --- |
| Pretraining | validation loss | `~8.0 -> 2.28` |
| Benchmark | C3 / XCOPA accuracy | `0.40 / 0.55` |
| GRPO | reasoning QA F1 | `0.45 -> 0.60` |

以上结果来自当前实验记录，实际效果会随数据规模、训练步数、模型配置、采样参数和评测方式变化。

## Model Architecture

模型配置见 `model/config.py`，核心实现见 `model/model_rain.py`。

```text
Input IDs
  ↓
Token Embedding
  ↓
12 x Transformer Blocks
  ├── RMSNorm
  ├── Grouped Query Attention
  │   ├── RoPE
  │   ├── SDPA / Flash Attention path
  │   └── KV Cache
  ├── Residual
  ├── RMSNorm
  ├── SwiGLU FFN
  └── Residual
  ↓
Final RMSNorm
  ↓
LM Head
```

默认配置：

| Parameter | Default | Description |
| --- | --- | --- |
| `vocab_size` | `15000` | 词表大小 |
| `hidden_size` | `768` | hidden dimension |
| `num_hidden_layers` | `12` | Transformer layers |
| `num_attention_heads` | `12` | query heads |
| `num_key_value_heads` | `4` | key/value heads |
| `intermediate_size` | `2048` | FFN hidden size |
| `max_position_embeddings` | `32768` | max position length |

## Repository Structure

```text
.
├── model/
│   ├── config.py              # RainConfig
│   └── model_rain.py          # Rain model and CausalLM wrapper
├── train/
│   ├── train_tokenizer.py     # Tokenizer training and validation
│   ├── pretrain.py            # Pretraining script with DDP support
│   ├── train_sft.py           # SFT training script
│   ├── train_grpo.py          # GRPO training script
│   └── utils.py               # training utilities
├── dataset/
│   ├── preprocess_data.py     # pretrain JSONL -> bin/meta
│   ├── pretrain_dataset.py    # pretraining dataset
│   ├── sft_dataset.py         # SFT dataset
│   └── grpo_dataset.py        # GRPO dataset
├── benchmark/
│   ├── evaluator.py           # C3 / XCOPA evaluation
│   └── mini_benchmark/        # mini benchmark + Judge
├── tokenizer_15k/             # trained 15K tokenizer
├── GRPO_train.jsonl           # sample GRPO prompts
├── eval.py                    # interactive inference
└── requirements.txt
```

## Installation

建议使用 Python 3.10+。GPU 训练时请根据本机 CUDA 版本安装合适的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果启用 DeepSeek Judge，请配置自己的 API key：

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

## Data Format

### Pretraining Data

原始 pretraining 数据为 JSONL，每行包含 `text` 字段：

```json
{"text": "这里是一段用于语言模型预训练的中文文本。"}
```

预处理为二进制训练数据：

```bash
python dataset/preprocess_data.py \
  --input data/pretrain.jsonl \
  --output data/pretrain_512 \
  --tokenizer tokenizer_15k \
  --seq_len 512 \
  --num_workers 16
```

输出文件：

- `data/pretrain_512.bin`
- `data/pretrain_512.meta`

### SFT Data

SFT 数据为 multi-turn conversation JSONL。训练时只计算 assistant response 部分的 loss。

```json
{"conversations":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好，我是 Rain。"}]}
```

### GRPO Data

GRPO 数据只需要 prompt。仓库中的 `GRPO_train.jsonl` 提供了示例数据。

```json
{"id": 153, "category": "指令与逻辑", "prompt": "计算5减2等于几。"}
```

## Training

### Tokenizer

仓库已包含 `tokenizer_15k/`，可以直接用于训练和推理。如果需要重新训练 tokenizer，请先在 `train/train_tokenizer.py` 中配置 `DATA_PATH` 和 `TOKENIZER_DIR`，并打开脚本末尾的 `train_tokenizer(...)` 调用。

```bash
python train/train_tokenizer.py
```

### Pretraining

单卡示例：

```bash
python train/pretrain.py \
  --data_path data/pretrain_512.bin \
  --save_dir out/pretrain \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --use_swanlab 0 \
  --eval_bench 0
```

DDP 多卡示例：

```bash
torchrun --nproc_per_node=2 train/pretrain.py \
  --data_path data/pretrain_512.bin \
  --save_dir out/pretrain \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --use_swanlab 0 \
  --eval_bench 0
```

Resume checkpoint：

```bash
python train/pretrain.py \
  --data_path data/pretrain_512.bin \
  --save_dir out/pretrain \
  --from_resume 1 \
  --use_swanlab 0 \
  --eval_bench 0
```

### SFT

```bash
python train/train_sft.py \
  --data_path data/sft.jsonl \
  --tokenizer_path tokenizer_15k \
  --from_weight out/pretrain/global_step_x/pretrain_768.pth \
  --save_dir out/sft \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --use_swanlab 0 \
  --enable_eval 0
```

启用 mini benchmark + DeepSeek Judge：

```bash
python train/train_sft.py \
  --data_path data/sft.jsonl \
  --tokenizer_path tokenizer_15k \
  --from_weight out/pretrain/global_step_x/pretrain_768.pth \
  --save_dir out/sft \
  --enable_eval 1 \
  --judge_api_key "$DEEPSEEK_API_KEY" \
  --use_swanlab 0
```

### GRPO

```bash
python train/train_grpo.py \
  --data_path GRPO_train.jsonl \
  --tokenizer_path tokenizer_15k \
  --sft_model_path out/sft/global_step_x/sft_768.pth \
  --save_dir out/grpo \
  --batch_size 16 \
  --num_generations 4 \
  --beta 0.05 \
  --judge_api_key "$DEEPSEEK_API_KEY" \
  --use_swanlab 0
```

GRPO 训练会保存 rollout logs，便于分析每条回答的 format check、Judge score 和 final reward。

## Inference

单轮对话：

```bash
python eval.py \
  --model_path out/grpo/global_step_x/grpo_768.pth \
  --tokenizer_path tokenizer_15k \
  --model_type sft \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --temperature 0.3 \
  --top_p 0.7
```

多轮对话：

```bash
python eval.py \
  --model_path out/grpo/global_step_x/grpo_768.pth \
  --tokenizer_path tokenizer_15k \
  --model_type sft \
  --multi_turn
```

## Evaluation

- `benchmark/evaluator.py`：C3 / XCOPA multiple-choice evaluation，通过比较选项 perplexity 选择答案。
- `benchmark/mini_benchmark/eval.py`：生成式 mini benchmark，使用 DeepSeek Judge 评估 fluency、factuality 和 instruction following。
- `train/train_sft.py`：支持训练过程中定期运行 generation evaluation。
- `train/train_grpo.py`：记录 rollout、reward、format check 和 Judge score。

## Checkpoints

训练输出目录会按实验配置自动组织：

```text
out/{stage}/h768_l12_bs128_lr0.001/
├── global_step_1000/
│   ├── pretrain_768.pth
│   └── resume.pth
└── global_step_2000/
    ├── pretrain_768.pth
    └── resume.pth
```

- `.pth`：模型权重文件，可用于 inference 或下一阶段训练初始化。
- `resume.pth`：完整训练状态，包含 model、optimizer、scaler、epoch、step、global_step 等信息，用于断点续训。

## Notes

- 训练脚本中部分默认路径来自个人实验环境，实际运行时建议通过命令行参数覆盖。
- 快速验证 pretraining 流程时，建议设置 `--eval_bench 0`，避免因评测路径或 tokenizer 路径不完整导致中断。
- GRPO 和 mini benchmark 会调用外部 Judge API，请确认网络、额度和密钥配置。
- 加载已有权重时，模型维度、层数和 tokenizer 必须与训练时保持一致。
- 当前仓库尚未声明开源许可证，如需公开发布或商业使用，建议补充 `LICENSE` 文件。
