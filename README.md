# HunyuanOCR GRPO experiments

Sanitized training and data-generation scripts for GRPO experiments on OCR/VLM tasks.

## Contents

- `swift_grpo_train_qwen.sh`, `qwen25_vl_grpo_train*.py`: GRPO training entry points.
- `generate_grpo*.py`: candidate generation and reward dataset preparation.
- `GRPO_TRAINING_GUIDE.md`: setup notes.

Model, checkpoint, and dataset paths are placeholders. Set paths for your own environment before running. HTTP generation scripts read `MODEL_API_BASE_URL` and `MODEL_API_KEY` from environment variables. Keep credentials and private datasets out of Git.

This repository does not include model weights, checkpoints, OCR outputs, or training/evaluation datasets.

---

## GRPO Training Guide

# Qwen2.5-VL GRPO 强化学习训练指南

## 概述

本指南说明如何基于SFT训练好的Qwen2.5-VL模型，使用GRPO（Group Relative Policy Optimization）算法进行强化学习进一步优化OCR效果。

## 流程

```
SFT训练好的LoRA模型
       ↓
生成GRPO数据集（多候选+奖励分数）
       ↓
GRPO强化学习训练
       ↓
得到优化后的RL模型
```

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `generate_grpo_dataset.py` | GRPO数据集生成脚本 |
| `qwen25_vl_grpo_train.py` | Python版本GRPO训练脚本 |
| `swift_grpo_train_qwen.sh` | 命令行版本GRPO训练脚本 |
| `swift_grpo_train_simple.sh` | 一键运行脚本（生成数据+训练） |

## 1. 环境要求

```bash
# 确保ModelScope Swift版本支持RL/GRPO
pip install ms-swift[rl] --upgrade

# 检查Swift版本
swift --version
```

## 2. 生成GRPO数据集

### 2.1 配置参数

编辑 `generate_grpo_dataset.py` 中的配置：

```python
MODEL_PATH = '/path/to/model'
LORA_CHECKPOINT = '/path/to/model'
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

NUM_CANDIDATES = 4  # 每个样本生成的候选数量
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]  # 不同温度增加多样性
```

### 2.2 运行生成脚本

```bash
python generate_grpo_dataset.py
```

### 2.3 数据集格式

生成的GRPO数据集格式（兼容SFT数据格式）：

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "提取图片中的字段..."}
    ],
    "images": "/path/to/image.jpg",
    "candidates": ["候选结果1", "候选结果2", "候选结果3", "候选结果4"],
    "rewards": [0.95, 0.82, 0.75, 0.60],
    "reference": "标准答案JSON"
  }
]
```

### 2.4 奖励计算逻辑

- **字段准确率 (80%)**: 每个字段完全匹配得1分，相似度>90%得0.5分
- **格式合规性 (20%)**: 输出是否为有效JSON格式
- **总奖励**: 字段准确率 * 0.8 + 格式合规性 * 0.2

## 3. GRPO训练

### 方法一：使用Python脚本（推荐）

编辑 `qwen25_vl_grpo_train.py` 配置后运行：

```bash
python qwen25_vl_grpo_train.py
```

训练参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `LEARNING_RATE` | 5e-6 | RL训练通常用比SFT小的学习率 |
| `NUM_TRAIN_EPOCHS` | 3-5 | RL训练不需要太多epoch |
| `BATCH_SIZE` | 1 | 多模态训练建议使用小batch |
| `LORA_RANK` | 16 | 保持与SFT一致 |
| `BETA` | 0.04 | KL散度惩罚系数 |
| `NUM_CANDIDATES` | 4 | 每个prompt的候选数量 |

### 方法二：使用命令行（需要Swift支持RL）

```bash
bash swift_grpo_train_qwen.sh
```

### 方法三：一键运行

```bash
bash swift_grpo_train_simple.sh
```

## 4. 训练优化建议

### 4.1 奖励函数调优

根据实际业务调整奖励权重：

```python
# 在 generate_grpo_dataset.py 中修改
def calculate_total_reward(pred_str, ref_str, fields=None):
    field_reward = calculate_field_level_reward(pred_str, ref_str, fields)
    format_reward = calculate_format_reward(pred_str)
    # 可根据业务调整权重，例如更看重统一社会信用代码：
    return field_reward * 0.9 + format_reward * 0.1
```

### 4.2 候选多样性

- 调整 `TEMPERATURES` 数组来控制候选多样性
- 可以尝试添加top_p、top_k等采样参数
- 建议温度范围：0.3~1.5

### 4.3 训练稳定性

1. **学习率**: RL训练建议使用更小的学习率 (1e-6 ~ 1e-5)
2. **梯度裁剪**: 建议启用gradient_clipping
3. **KL惩罚**: BETA参数控制与参考策略的距离，防止发散
4. **EMA**: advantage_ema平滑优势估计

## 5. 验证和推理

训练完成后，使用训练好的LoRA进行推理：

```python
from swift.llm import get_model_tokenizer, get_template, PtEngine
from swift.tuners import Swift

MODEL_PATH = '/path/to/model'
GRPO_CHECKPOINT = './output_grpo_227/checkpoint-XXX'

model, tokenizer = get_model_tokenizer(MODEL_PATH, dtype=torch.bfloat16)
model = Swift.from_pretrained(model, GRPO_CHECKPOINT)
model = model.to('cuda')

template = get_template(model.model_meta.template, tokenizer)
engine = PtEngine.from_model_template(model, template, max_batch_size=1)
```

## 6. 常见问题

### Q1: Swift版本不支持RL训练？
A: 确保安装最新版本：`pip install ms-swift[rl] --upgrade`

### Q2: 训练时OOM？
A: 
- 减小batch_size
- 启用gradient_checkpointing
- 使用更小的模型

### Q3: 奖励分数一直很低？
A:
- 检查prompt是否与SFT一致
- 调整温度参数，候选多样性不够
- 检查奖励函数是否合理

### Q4: 如何从SFT checkpoint继续训练？
A: 脚本已配置自动加载SFT的LoRA权重，在此基础上继续RL训练。

## 7. 注意事项

1. **数据集质量**: GRPO效果很大程度上取决于候选质量和奖励函数设计
2. **训练时间**: RL训练通常比SFT慢2-4倍
3. **模型保存**: 训练后的LoRA可以直接用于推理，不需要合并
4. **评估**: 训练过程中建议定期在验证集上评估效果

## 8. 参考资源

- [ModelScope Swift官方文档](https://modelscope.cn/docs/llm-training-and-inference)
- [GRPO论文: Group Relative Policy Optimization](https://arxiv.org/abs/xxxx)
- [Qwen2.5-VL官方文档](https://github.com/QwenLM/Qwen-VL)
