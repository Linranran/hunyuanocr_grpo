"""
简化版GRPO训练脚本 - 适用于已有预计算奖励的数据集
基于LoRA继续训练，不需要在线采样
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ==================== 配置 ====================
MODEL_PATH = '/path/to/model'
SFT_CHECKPOINT = '/path/to/model'
GRPO_DATASET = '/path/to/dataset'
OUTPUT_DIR = './output_grpo_simple_227'

# 训练参数
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
GRADIENT_ACCUMULATION = 16
MAX_SEQ_LENGTH = 2048
BETA = 0.04  # KL惩罚系数

# ==================== 数据集类 ====================
class GRPODataset(Dataset):
    def __init__(self, dataset_path, processor):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 提取user prompt
        user_prompt = ""
        for msg in item['messages']:
            if msg['role'] == 'user':
                user_prompt = msg['content']
                break

        return {
            "prompt": user_prompt,
            "image_path": item['images'],
            "candidates": item['candidates'],
            "rewards": torch.tensor(item['rewards'], dtype=torch.float32),
        }

# ==================== 训练函数 ====================
def compute_grpo_loss(model, processor, batch, device):
    """
    简化版GRPO损失计算
    基于预生成的candidates和rewards进行策略更新
    """
    prompt = batch['prompt'][0]
    image_path = batch['image_path'][0]
    candidates = batch['candidates'][0]
    rewards = batch['rewards'][0].to(device)

    # 加载图片
    image = Image.open(image_path).convert('RGB')

    # 构建messages并编码
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    losses = []
    log_probs_list = []

    # 对每个候选计算log_prob并加权
    for i, candidate in enumerate(candidates):
        if not candidate.strip():
            continue

        # 编码prompt + candidate
        full_text = text + candidate

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        candidate_tokens = processor.tokenizer(
            candidate,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)

        # 前向传播获取logits
        with torch.no_grad():
            # 获取参考策略log_probs (KL penalty用)
            ref_outputs = model(**inputs)
            ref_logits = ref_outputs.logits[:, -1:, :]

        # 计算当前策略log_prob
        inputs_with_candidate = processor(
            text=[text + candidate],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs_with_candidate)
        logits = outputs.logits

        # 计算候选部分的log_prob
        shift_logits = logits[:, -candidate_tokens.input_ids.shape[1]-1:-1, :]
        shift_labels = candidate_tokens.input_ids

        log_probs = -nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        ).mean()

        # 优势函数归一化
        advantage = (rewards[i] - rewards.mean()) / (rewards.std() + 1e-8)

        # 策略梯度损失
        policy_loss = -log_probs * advantage.detach()

        # KL散度惩罚（简化）
        with torch.no_grad():
            kl_div = 0.0  # 可根据需要添加完整KL计算

        loss = policy_loss + BETA * kl_div
        losses.append(loss)
        log_probs_list.append(log_probs.detach())

    if not losses:
        return None

    return sum(losses) / len(losses)

def train():
    print("="*60)
    print("简化版GRPO训练 - OCR优化")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载Processor和Model
    print(f"\n加载模型: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 加载SFT LoRA
    print(f"加载SFT LoRA: {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model.train()

    # 3. 新LoRA配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 数据集
    dataset = GRPODataset(GRPO_DATASET, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. 训练循环
    print(f"\n开始训练，共 {EPOCHS} 轮...")
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        pbar = tqdm(dataloader, desc=f"训练")
        total_loss = 0

        for step, batch in enumerate(pbar):
            try:
                loss = compute_grpo_loss(model, processor, batch, device)

                if loss is not None:
                    loss = loss / GRADIENT_ACCUMULATION
                    loss.backward()

                    total_loss += loss.item() * GRADIENT_ACCUMULATION

                    if (step + 1) % GRADIENT_ACCUMULATION == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        pbar.set_postfix({
                            "loss": f"{total_loss / (step + 1):.4f}",
                            "step": global_step
                        })

                        # 定期保存
                        if global_step % 50 == 0:
                            save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                            model.save_pretrained(save_path)
                            print(f"\n已保存模型: {save_path}")

            except Exception as e:
                print(f"训练步骤异常: {e}")
                continue

        # Epoch结束保存
        epoch_save_path = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
        model.save_pretrained(epoch_save_path)

    # 最终保存
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    print("\n" + "="*60)
    print(f"训练完成！模型保存至: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MAX_PIXELS'] = '1003520'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train()
