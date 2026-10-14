"""
Qwen2.5-VL GRPO 强化学习训练脚本
使用Swift框架的RL API
"""
import os
import json
import torch

os.environ['MAX_PIXELS'] = "12845056"
os.environ['VIDEO_MAX_PIXELS'] = "50176"
os.environ['FPS_MAX_FRAMES'] = "12"

# ==================== 配置 ====================
MODEL_PATH = '/path/to/model'
SFT_CHECKPOINT = '/path/to/model'
GRPO_DATASET = '/path/to/dataset'
OUTPUT_DIR = './output_grpo_227'

# 训练参数
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
LORA_RANK = 16
LORA_ALPHA = 64
TARGET_MODULES = 'all-linear'

# GRPO特有参数
NUM_CANDIDATES = 4
BETA = 0.04  # KL散度惩罚系数
ADVANTAGE_EMA = 0.95
TEMPERATURE = 0.7
MAX_TOKENS = 1000

def train_grpo():
    from swift.llm import RLArguments, RLTrainer, get_model_tokenizer, get_template
    from swift.tuners import Swift, LoraConfig

    print(f"加载模型: {MODEL_PATH}")
    model, tokenizer = get_model_tokenizer(MODEL_PATH, torch_dtype=torch.bfloat16)

    print(f"加载SFT LoRA权重: {SFT_CHECKPOINT}")
    model = Swift.from_pretrained(model, SFT_CHECKPOINT)

    template = get_template(model.model_meta.template, tokenizer)

    # 加载GRPO数据集
    print(f"加载GRPO数据集: {GRPO_DATASET}")
    with open(GRPO_DATASET, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"数据集大小: {len(dataset)} 条")

    # RL训练参数
    training_args = RLArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_steps=50,
        save_total_limit=5,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        dataloader_num_workers=1,
        remove_unused_columns=False,
        # GRPO参数
        rl_type='grpo',
        num_candidates=NUM_CANDIDATES,
        beta=BETA,
        advantage_ema=ADVANTAGE_EMA,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # 创建RL Trainer
    trainer = RLTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        template=template,
        train_dataset=dataset,
        # 自定义奖励函数（也可以使用数据集中预先计算的rewards）
        compute_metrics=None,
    )

    print("开始GRPO训练...")
    trainer.train()

    print(f"训练完成，模型保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_grpo()
