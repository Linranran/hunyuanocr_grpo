"""
GRPO数据集生成脚本
基于SFT训练好的Qwen2.5-VL模型生成候选结果，并计算奖励分数
"""
import os
import json
import re
import torch
from tqdm import tqdm
from difflib import SequenceMatcher

os.environ['MAX_PIXELS'] = "1003520"
os.environ['VIDEO_MAX_PIXELS'] = "50176"
os.environ['FPS_MAX_FRAMES'] = "12"

# ==================== 配置 ====================
MODEL_PATH = '/path/to/model'
LORA_CHECKPOINT = '/path/to/model'
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

PROMPT = "提取图片中的: ['统一社会信用代码','名称','类型', '经营者','注册资本','成立日期','经营期限','住址'] 的字段内容，并按照 JSON 格式返回。"

NUM_CANDIDATES = 4  # 每个样本生成的候选数量
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]  # 不同温度用于生成多样性候选
MAX_TOKENS = 1000
DTYPE = torch.bfloat16
BATCH_SIZE = 1

# 奖励字段配置（支持3字段和8字段两种格式）
FIELD_CONFIGS = [
    ['统一社会信用代码', '名称', '经营者'],
    ['统一社会信用代码', '名称', '类型', '经营者', '注册资本', '成立日期', '经营期限', '住址']
]

# ==================== 奖励函数 ====================
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
def load_model():
    from swift.llm import get_model_tokenizer, get_template, PtEngine
    from swift.tuners import Swift

    print(f"加载模型: {MODEL_PATH}")
    model, tokenizer = get_model_tokenizer(MODEL_PATH, dtype=DTYPE)

    print(f"加载LoRA权重: {LORA_CHECKPOINT}")
    model = Swift.from_pretrained(model, LORA_CHECKPOINT)
    model = model.to('cuda')

    template = get_template(model.model_meta.template, tokenizer)
    engine = PtEngine.from_model_template(model, template, max_batch_size=BATCH_SIZE)

    print("模型加载完成")
    return engine

# ==================== 数据集生成 ====================
def load_sft_dataset():
    """加载SFT训练数据集"""
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_grpo_dataset(engine):
    from swift.llm import RequestConfig, InferRequest

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
        fields = get_fields_from_prompt(user_prompt or PROMPT)

        # 生成多个候选结果
        candidates = []
        rewards = []

        for temp in TEMPERATURES[:NUM_CANDIDATES]:
            try:
                request_config = RequestConfig(
                    max_tokens=MAX_TOKENS,
                    temperature=temp,
                    top_p=0.9
                )

                infer_request = InferRequest(
                    messages=[{'role': 'user', 'content': user_prompt or PROMPT}],
                    images=[image_path]
                )

                response = engine.infer([infer_request], request_config)[0]
                pred_answer = response.choices[0].message.content.strip()

                reward = calculate_total_reward(pred_answer, ref_answer, fields)

                candidates.append(pred_answer)
                rewards.append(reward)

            except Exception as e:
                print(f"生成候选失败: {e}")
                candidates.append("")
                rewards.append(0.0)

        # 构建GRPO样本（保持与SFT一致的格式，添加GRPO特有字段）
        grpo_item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt or PROMPT}
            ],
            "images": images,
            "candidates": candidates,
            "rewards": rewards,
            "reference": ref_answer
        }

        grpo_dataset.append(grpo_item)

        # 每100条保存一次
        if (idx + 1) % 100 == 0:
            save_dataset(grpo_dataset)

    save_dataset(grpo_dataset)
    print(f"GRPO数据集生成完成，共 {len(grpo_dataset)} 条")
    return grpo_dataset

def save_dataset(dataset):
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存到: {OUTPUT_DATASET_PATH}")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    engine = load_model()
    grpo_dataset = generate_grpo_dataset(engine)
