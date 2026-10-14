"""
基于TRL的Qwen2.5-VL GRPO强化学习训练脚本
使用Transformers + TRL库，不依赖Swift框架
"""
import os
import json
import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
MODEL_PATH = '/path/to/model'
SFT_CHECKPOINT = '/path/to/model'
GRPO_DATASET = '/path/to/dataset'
OUTPUT_DIR = './output_grpo_trl_227'

# 训练参数
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
LORA_RANK = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# GRPO特有参数
NUM_CANDIDATES = 4
BETA = 0.04  # KL散度惩罚系数
MAX_NEW_TOKENS = 512

# ==================== 数据处理函数 ====================
def format_sample_for_grpo(sample):
    """将样本格式化为GRPO Trainer需要的格式"""
    # 提取user prompt
    user_prompt = ""
    for msg in sample['messages']:
        if msg['role'] == 'user':
            user_prompt = msg['content']
            break

    return {
        "prompt": user_prompt,
        "images": sample['images'],
        "candidates": sample['candidates'],
        "rewards": sample['rewards'],
        "reference": sample['reference']
    }

def load_grpo_dataset():
    """加载GRPO数据集"""
    with open(GRPO_DATASET, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为HuggingFace Dataset格式
    formatted_data = [format_sample_for_grpo(sample) for sample in data]
    dataset = Dataset.from_list(formatted_data)

    print(f"加载GRPO数据集: {len(dataset)} 条")
    return dataset

# ==================== 自定义奖励函数 ====================
def ocr_reward_fn(completions, **kwargs):
    """
    自定义OCR奖励函数
    completions: 模型生成的结果列表
    kwargs: 包含prompts等信息
    """
    import re
    from difflib import SequenceMatcher

    def extract_json_from_text(text):
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        return None

    rewards = []
    for completion in completions:
        # 格式奖励
        try:
            json_obj = extract_json_from_text(completion)
            format_reward = 1.0 if json_obj else 0.0
        except:
            format_reward = 0.0

        # 这里可以结合reference计算更精确的字段奖励
        # 注意：如果数据集中已有rewards，GRPO会优先使用
        rewards.append(format_reward)

    return rewards

# ==================== 训练函数 ====================
def train_grpo():
    print("="*60)
    print("Qwen2.5-VL GRPO 强化学习训练 (TRL)")
    print("="*60)

    # 1. 加载数据集
    dataset = load_grpo_dataset()

    # 2. 加载processor
    print(f"\n加载Processor: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    # 3. 加载基础模型
    print(f"加载基础模型: {MODEL_PATH}")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. 加载SFT LoRA权重（作为RL训练的起点）
    print(f"加载SFT LoRA权重: {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)

    # 5. LoRA配置（继续在SFT基础上训练新的LoRA或继续训练现有LoRA）
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. GRPO训练配置
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="qwen25-vl-grpo-ocr",
        # 训练参数
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # 日志和保存
        logging_steps=5,
        save_steps=50,
        save_total_limit=5,
        # 精度设置
        fp16=False,
        bf16=True,
        # GRPO特有参数
        beta=BETA,
        num_generations=NUM_CANDIDATES,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        # 多模态相关
        remove_unused_columns=False,
        dataloader_num_workers=0,
        # 优化设置
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    # 7. 自定义collate函数（处理多模态数据）
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        images = [item['images'] for item in batch]

        # 构建Qwen2.5-VL格式的消息
        messages_list = []
        pil_images = []
        for prompt, img_path in zip(prompts, images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            messages_list.append(messages)
            pil_images.append(Image.open(img_path).convert('RGB'))

        # 应用聊天模板并编码
        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        inputs = processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt"
        )

        # 添加参考答案（用于奖励计算）
        inputs["references"] = [item['reference'] for item in batch]

        return inputs

    # 8. 创建GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_config,
        reward_fn=ocr_reward_fn,
        data_collator=collate_fn,
    )

    # 9. 开始训练
    print("\n开始GRPO训练...")
    trainer.train()

    # 10. 保存模型
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\n" + "="*60)
    print(f"训练完成！模型已保存至: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Qwen2.5-VL环境变量
    os.environ['MAX_PIXELS'] = '1003520'
    os.environ['VIDEO_MAX_PIXELS'] = '50176'

    # 安装依赖检查
    try:
        import trl
    except ImportError:
        print("请先安装TRL: pip install trl")
        exit(1)

    train_grpo()
