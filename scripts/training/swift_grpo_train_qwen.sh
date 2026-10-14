#!/bin/bash
# Qwen2.5-VL GRPO训练脚本

# 设置环境变量
export MAX_PIXELS=12845056
export CUDA_VISIBLE_DEVICES=0

# 训练配置
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="/path/to/dataset"
OUTPUT_DIR="output_grpo_227"

# 从SFT checkpoint继续训练
SFT_CHECKPOINT="/path/to/model"

# GRPO训练参数
NUM_TRAIN_EPOCHS=5
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=5e-6
LORA_RANK=16
LORA_ALPHA=64

# GRPO特有参数
NUM_CANDIDATES=4  # 每个prompt生成的候选数量
BETA=0.04         # KL散度惩罚系数
ADVANTAGE_EMA=0.95
TEMPERATURE=0.7

echo "开始GRPO训练..."
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "输出目录: $OUTPUT_DIR"

# 使用Swift进行GRPO训练
# 注意：Swift版本需支持多模态RL训练
swift rl \
    --model $MODEL \
    --dataset $DATASET \
    --train_type lora \
    --rl_type grpo \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --freeze_vit false \
    --freeze_aligner false \
    --target_modules all-linear \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --num_candidates $NUM_CANDIDATES \
    --beta $BETA \
    --advantage_ema $ADVANTAGE_EMA \
    --temperature $TEMPERATURE

echo "GRPO训练完成！"
