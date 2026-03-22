# 0.1B 中文大语言模型端到端构建与优化

从零实现并优化一个 ~0.1B 参数的中文 Dense 模型，完整覆盖 Tokenizer → Pretraining → SFT → 原创架构创新 → 自研强化对齐全链路。

**项目周期**：2025.12 – 2026.03  
**最终成果**：  
- Validation loss 从初始 ~8 降至 **2.28**  
- C3 / XCOPA 评测：**0.40 / 0.55**  
- GRPO 强化后推理 F1 score：**0.45 → 0.60**（格式遵守能力显著提升）

## 项目亮点

- 自研 **BBPE Tokenizer**（15K+ 词表），适配中文高效压缩
- 1.5B tokens 多领域高质量语料（MinHash 去重）
- 现代 Decoder-only 架构：**GQA (8/2)**、**SwiGLU**、**RoPE**、**Flash Attention 2**、**KV Cache**
- 工程优化：**bfloat16** 全链路 + **torch.compile** + **DDP**（2× GPU），训练效率高
- **原创创新 1**：**Gated Attention**（Head 级 query-dependent 门控），有效缓解 attention sink，loss 进一步下降 0.02
- **原创创新 2**：**自研 GRPO**（Group Relative Policy Optimization + per-token KL 约束 + 混合奖励），小模型 RL 格式崩坏问题得到有效解决

## 模型架构概览

```text
Input IDs
    ↓
Token Embedding (15,360 × 512) + RoPE
    ↓
×12 Transformer Blocks
    ├─ Pre-RMSNorm
    ├─ Gated Grouped-Query Attention (8 heads / 2 kv heads + 自研 Gate)
    │   └─ Flash Attention 2 + RoPE + KV Cache
    ├─ Residual
    ├─ Pre-RMSNorm
    ├─ SwiGLU FFN (中间维度 1365)
    └─ Residual
    ↓
Final RMSNorm
    ↓
LM Head (weight tied with Embedding)
