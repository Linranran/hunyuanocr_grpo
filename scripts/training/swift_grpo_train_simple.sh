#!/bin/bash
# Qwen2.5-VL GRPO训练脚本 - 简化版
# 使用自定义Python脚本进行RL训练

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MAX_PIXELS=12845056
export CUDA_VISIBLE_DEVICES=0

echo "=================================================="
echo "Qwen2.5-VL GRPO 强化学习训练"
echo "=================================================="

# 步骤1: 生成GRPO数据集（如果还没生成）
if [ ! -f "/path/to/dataset" ]; then
    echo "步骤1: 生成GRPO数据集..."
    python "$SCRIPT_DIR/../data_generation/generate_grpo_dataset.py"
else
    echo "步骤1: GRPO数据集已存在，跳过生成"
fi

# 步骤2: 运行GRPO训练
echo ""
echo "步骤2: 开始GRPO训练..."
python "$SCRIPT_DIR/qwen25_vl_grpo_train.py"

echo ""
echo "=================================================="
echo "训练完成！"
echo "输出目录: output_grpo_227"
echo "=================================================="
