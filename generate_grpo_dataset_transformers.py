"""
基于Transformers的GRPO数据集生成脚本
适用于Qwen2.5-VL模型，不依赖Swift框架
"""
import os
import json
import re
import torch
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
MODEL_PATH = '/path/to/model'
LORA_CHECKPOINT = '/path/to/model'
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

NUM_CANDIDATES = 4  # 每个样本生成的候选数量
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]  # 不同温度用于生成多样性候选
MAX_NEW_TOKENS = 1000
DTYPE = torch.bfloat16

# 奖励字段配置
FIELD_CONFIGS = [
    ['统一社会信用代码', '名称', '经营者'],
    ['统一社会信用代码', '名称', '类型', '经营者', '注册资本', '成立日期', '经营期限', '住址']
]

# ==================== 工具函数 ====================
def clean_repeated_substrings(text):
    """清理文本中重复的子串（防止模型输出重复内容）"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]
    return text

def extract_json_from_text(text):
    """从文本中提取JSON内容"""
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def get_fields_from_prompt(prompt_content):
    """从prompt中提取要求的字段列表"""
    for fields in FIELD_CONFIGS:
        for field in fields:
            if field in prompt_content:
                return fields
    return FIELD_CONFIGS[0]  # 默认3字段

# ==================== 奖励函数 ====================
def calculate_field_level_reward(pred_str, ref_str, fields=None):
    """计算字段级别的奖励分数"""
    pred_json = extract_json_from_text(pred_str)
    ref_json = extract_json_from_text(ref_str)

    if pred_json is None or ref_json is None:
        return 0.0

    if fields is None:
        fields = get_fields_from_prompt(ref_str)

    correct_fields = 0
    total_fields = len(fields)

    for field in fields:
        pred_val = str(pred_json.get(field, '')).strip()
        ref_val = str(ref_json.get(field, '')).strip()

        if pred_val == ref_val:
            correct_fields += 1
        elif pred_val and ref_val:
            similarity = SequenceMatcher(None, pred_val, ref_val).ratio()
            if similarity > 0.9:
                correct_fields += 0.5

    return correct_fields / total_fields if total_fields > 0 else 0.0

def calculate_format_reward(text):
    """计算格式合规性奖励"""
    try:
        json_obj = extract_json_from_text(text)
        if json_obj:
            return 1.0
    except:
        pass
    return 0.0

def calculate_total_reward(pred_str, ref_str, fields=None):
    """计算总奖励：字段准确率 * 0.8 + 格式合规性 * 0.2"""
    field_reward = calculate_field_level_reward(pred_str, ref_str, fields)
    format_reward = calculate_format_reward(pred_str)
    return field_reward * 0.8 + format_reward * 0.2

# ==================== 模型加载 ====================
def load_model_and_processor():
    """加载基础模型、LoRA和处理器"""
    print(f"加载基础模型: {MODEL_PATH}")

    # 加载processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    # 加载基础模型 - 使用AutoModelForVision2Seq自动适配Qwen2.5-VL类型
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载LoRA权重
    print(f"加载LoRA权重: {LORA_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, LORA_CHECKPOINT)

    model.eval()
    device = next(model.parameters()).device
    print(f"模型已加载到设备: {device}")

    return model, processor, device

# ==================== 单样本推理 ====================
def generate_single_candidate(model, processor, device, image_path, prompt, temperature):
    """生成单个候选结果"""
    try:
        # 打开图片
        image = Image.open(image_path).convert('RGB')

        # 构建对话模板 - Qwen2.5-VL格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 应用聊天模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 预处理
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = inputs.to(device)

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # 处理生成结果
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 清理重复内容
        output_text = clean_repeated_substrings(output_text)

        return output_text.strip()

    except Exception as e:
        print(f"生成失败: {e}")
        return ""

# ==================== 数据集生成 ====================
def load_sft_dataset():
    """加载SFT训练数据集"""
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_dataset(dataset):
    """保存数据集"""
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存至: {OUTPUT_DATASET_PATH}")

def generate_grpo_dataset(model, processor, device):
    """生成完整的GRPO数据集"""
    sft_data = load_sft_dataset()
    print(f"加载SFT训练数据: {len(sft_data)} 条")

    grpo_dataset = []

    for idx, item in enumerate(tqdm(sft_data, desc="生成GRPO数据集")):
        messages = item['messages']
        images = item.get('images', '')

        # 提取ground truth（assistant的回答）和user prompt
        ref_answer = None
        user_prompt = None
        for msg in messages:
            if msg['role'] == 'assistant':
                ref_answer = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']

        if ref_answer is None or not images:
            continue

        image_path = images if isinstance(images, str) else images[0]

        # 确定字段列表
        fields = get_fields_from_prompt(user_prompt or "")

        # 生成多个候选结果
        candidates = []
        rewards = []

        for temp in TEMPERATURES[:NUM_CANDIDATES]:
            pred_answer = generate_single_candidate(
                model, processor, device,
                image_path, user_prompt, temp
            )

            reward = calculate_total_reward(pred_answer, ref_answer, fields)

            candidates.append(pred_answer)
            rewards.append(reward)

        # 构建GRPO样本（保持与SFT一致的格式）
        grpo_item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            "images": images,
            "candidates": candidates,
            "rewards": rewards,
            "reference": ref_answer
        }

        grpo_dataset.append(grpo_item)

        # 每50条保存一次并清理显存
        if (idx + 1) % 50 == 0:
            save_dataset(grpo_dataset)
            torch.cuda.empty_cache()

    save_dataset(grpo_dataset)
    print(f"GRPO数据集生成完成，共 {len(grpo_dataset)} 条")
    return grpo_dataset

# ==================== 主函数 ====================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 设置环境变量（Qwen2.5-VL需要）
    os.environ['MAX_PIXELS'] = '1003520'
    os.environ['VIDEO_MAX_PIXELS'] = '50176'

    print("="*60)
    print("基于Transformers的GRPO数据集生成")
    print("="*60)

    # 加载模型
    model, processor, device = load_model_and_processor()

    # 生成数据集
    grpo_dataset = generate_grpo_dataset(model, processor, device)

    print("="*60)
    print("数据集生成完成！")
    print(f"输出文件: {OUTPUT_DATASET_PATH}")
    print("="*60)
