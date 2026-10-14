"""
基于HunyuanOCR的GRPO数据集生成脚本
使用Swift框架加载HunyuanOCR模型 + LoRA权重
"""
import os
import json
import re
import time
import torch
import gc
from tqdm import tqdm
from multiprocessing import cpu_count
from difflib import SequenceMatcher

# ===================== 环境变量优化 =====================
os.environ['MAX_PIXELS'] = "501760"
os.environ['VIDEO_MAX_PIXELS'] = "25088"
os.environ['FPS_MAX_FRAMES'] = "8"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ===================== 配置 =====================
MODEL_PATH = '/path/to/model'
LORA_CHECKPOINT = '/path/to/model'

# SFT训练数据路径
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

DTYPE = torch.bfloat16
MAX_TOKENS = 200

# GRPO配置
NUM_CANDIDATES = 4  # 每个样本生成的候选数量
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]  # 不同温度生成多样性候选

# 奖励字段配置
FIELD_CONFIGS = [
    ['统一社会信用代码', '名称', '经营者'],
    ['统一社会信用代码', '名称', '类型', '经营者', '注册资本', '成立日期', '经营期限', '住址']
]

# ===================== 工具函数 =====================
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
    return FIELD_CONFIGS[0]

def print_gpu_memory(stage=""):
    """打印 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU 显存: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB")

def clear_gpu_memory():
    """彻底清理 GPU 显存"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    time.sleep(0.5)

# ===================== 奖励函数 =====================
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

# ===================== 模型加载 =====================
def load_model():
    from swift.llm import get_model_tokenizer, get_template, PtEngine
    from swift.tuners import Swift

    print(f"加载模型: {MODEL_PATH}")
    model, tokenizer = get_model_tokenizer(
        MODEL_PATH,
        dtype=DTYPE,
        use_fast=False
    )

    print(f"加载LoRA: {LORA_CHECKPOINT}")
    model = Swift.from_pretrained(model, LORA_CHECKPOINT)
    model = model.to('cuda')

    template_type = model.model_meta.template
    template = get_template(template_type, tokenizer)
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)

    print_gpu_memory("模型加载完成")
    return engine

# ===================== 数据集加载 =====================
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
    print(f"\n数据集已保存至: {OUTPUT_DATASET_PATH}")
    print_gpu_memory("保存数据集后")

# ===================== GRPO数据集生成 =====================
def generate_grpo_dataset(engine):
    from swift.llm import RequestConfig, InferRequest

    sft_data = load_sft_dataset()
    print(f"加载SFT训练数据: {len(sft_data)} 条")
    print_gpu_memory("数据加载完成")

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
            try:
                request_config = RequestConfig(
                    max_tokens=MAX_TOKENS,
                    temperature=temp
                )

                # 构建推理请求
                req = InferRequest(
                    messages=[{'role': 'user', 'content': user_prompt}],
                    images=[image_path]
                )

                # 执行推理
                resp = engine.infer([req], request_config)[0]
                pred_answer = resp.choices[0].message.content.strip()

                # 计算奖励
                reward = calculate_total_reward(pred_answer, ref_answer, fields)

                candidates.append(pred_answer)
                rewards.append(reward)

            except torch.cuda.OutOfMemoryError as e:
                print(f"\n⚠️ OOM 错误 {image_path}: {e}")
                clear_gpu_memory()
                candidates.append("")
                rewards.append(0.0)

            except Exception as e:
                print(f"\n❌ 错误 {image_path}: {str(e)}")
                candidates.append("")
                rewards.append(0.0)

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

        # 每20条保存一次并清理显存
        if (idx + 1) % 20 == 0:
            save_dataset(grpo_dataset)
            clear_gpu_memory()

        # 每100条打印统计
        if (idx + 1) % 100 == 0:
            avg_reward = sum(sum(item['rewards']) for item in grpo_dataset[-100:]) / (100 * NUM_CANDIDATES)
            print(f"\n📊 最近100条平均奖励: {avg_reward:.4f}")

    # 最终保存
    save_dataset(grpo_dataset)
    clear_gpu_memory()

    # 最终统计
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    print(f"\n✅ GRPO数据集生成完成！")
    print(f"   总样本数: {len(grpo_dataset)}")
    print(f"   总候选数: {len(all_rewards)}")
    print(f"   平均奖励: {sum(all_rewards) / len(all_rewards):.4f}")
    print(f"   奖励分布: 0-0.2={sum(1 for r in all_rewards if 0<=r<0.2)}, 0.2-0.4={sum(1 for r in all_rewards if 0.2<=r<0.4)}, 0.4-0.6={sum(1 for r in all_rewards if 0.4<=r<0.6)}, 0.6-0.8={sum(1 for r in all_rewards if 0.6<=r<0.8)}, 0.8-1.0={sum(1 for r in all_rewards if 0.8<=r<=1.0)}")

    return grpo_dataset

# ===================== 主函数 =====================
if __name__ == '__main__':
    print("="*70)
    print("HunyuanOCR GRPO 数据集生成脚本")
    print("="*70)
    print(f"基础模型: {MODEL_PATH}")
    print(f"LoRA权重: {LORA_CHECKPOINT}")
    print(f"训练数据: {SFT_TRAIN_DATA}")
    print(f"输出路径: {OUTPUT_DATASET_PATH}")
    print(f"候选数量: {NUM_CANDIDATES}")
    print(f"温度参数: {TEMPERATURES}")
    print("="*70)

    print(f"\nCPU核心数: {cpu_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

    # 加载模型
    engine = load_model()

    # 生成数据集
    grpo_dataset = generate_grpo_dataset(engine)

    print("\n" + "="*70)
    print("全部完成！")
    print("="*70)
