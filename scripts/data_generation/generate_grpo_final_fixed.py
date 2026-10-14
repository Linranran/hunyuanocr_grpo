"""
✅ 最终稳定版 GRPO 生成
Hunyuan不支持真正的Batch，所以单张处理 + 图片预加载优化

特点：
1. 图片提前加载到内存，减少IO等待
2. 每张图片编码一次，4个候选共享
3. 严格三字段匹配奖励
"""
import os
import json
import re
import time
import torch
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===================== 配置 =====================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MODEL_PATH = '/path/to/model'
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 150

# 预加载图片到内存的数量（可调大，200也没问题）
PREFETCH_IMAGES = 100

# GRPO配置
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]
TARGET_FIELDS = ['统一社会信用代码', '名称', '经营者']

# ===================== 工具函数 =====================
def extract_json_safe(text):
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def field_similarity(a, b):
    """严格匹配：完全一致才算对"""
    a, b = str(a).strip(), str(b).strip()
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    return 1.0 if a == b else 0.0

def calculate_reward_3fields(pred_str, ref_str):
    """权重：统一社会信用代码40%，名称35%，经营者25%"""
    pred_json = extract_json_safe(pred_str)
    ref_json = extract_json_safe(ref_str)

    if pred_json is None:
        return 0.0
    if ref_json is None:
        return 0.2

    score_code = field_similarity(pred_json.get('统一社会信用代码', ''), ref_json.get('统一社会信用代码', ''))
    score_name = field_similarity(pred_json.get('名称', ''), ref_json.get('名称', ''))
    score_owner = field_similarity(pred_json.get('经营者', ''), ref_json.get('经营者', ''))

    return score_code * 0.40 + score_name * 0.35 + score_owner * 0.25

# ===================== 模型加载 =====================
@torch.no_grad()
def load_model_and_processor():
    print("="*70)
    print("✅ GRPO 数据集生成 (最终稳定版)")
    print("="*70)
    print(f"候选温度: {TEMPERATURES}")
    print(f"预加载图片: {PREFETCH_IMAGES} 张")
    print("="*70)

    print("\n加载模型...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="sdpa",
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    torch.backends.cudnn.benchmark = True

    device = next(model.parameters()).device
    print(f"设备: {device}")
    print(f"初始显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    return model, processor, device

# ===================== 单张生成（编码只做一次）=====================
@torch.no_grad()
def generate_single_candidates(model, processor, device, img_path, user_prompt, preloaded_images):
    """
    🔥 关键优化：图片只编码一次，4个温度共享同一个inputs
    """
    # 从预加载缓存取，或打开图片
    if img_path in preloaded_images:
        image = preloaded_images[img_path]
    else:
        image = Image.open(img_path).convert('RGB')

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 🔥 只编码一次！4个候选共享
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 4个温度生成
    candidates = []
    for temp in TEMPERATURES:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temp,
            do_sample=True,
            top_p=0.9,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        candidates.append(output_text.strip())

    return candidates

# ===================== 主函数 =====================
def main():
    model, processor, device = load_model_and_processor()

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"总样本数: {len(sft_data)}")

    # ========== 测试模式 ==========
    # sft_data = sft_data[:100]
    # print(f"⚠️ 测试模式: 只处理前 {len(sft_data)} 条")

    # ========== 第一步：预加载图片到内存（大大减少IO等待）==========
    print(f"\n📥 预加载前 {min(PREFETCH_IMAGES, len(sft_data))} 张图片到内存...")
    preloaded_images = {}
    for item in tqdm(sft_data[:PREFETCH_IMAGES], desc="预加载图片"):
        img_path = item.get('images', '')
        if isinstance(img_path, list):
            img_path = img_path[0]
        try:
            preloaded_images[img_path] = Image.open(img_path).convert('RGB')
        except:
            pass

    grpo_dataset = []
    start_time = time.time()

    print(f"\n🚀 开始生成...")
    pbar = tqdm(total=len(sft_data), desc="处理进度")

    last_save = 0
    last_speed_check = 0
    speed_window = []

    # ========== 主循环 ==========
    for idx, item in enumerate(sft_data):
        images = item.get('images', '')
        img_path = images if isinstance(images, str) else images[0]

        # 提取数据
        ref_answer = None
        user_prompt = None
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                ref_answer = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']

        if ref_answer is None:
            pbar.update(1)
            continue

        try:
            # 生成4个候选
            candidates = generate_single_candidates(model, processor, device, img_path, user_prompt, preloaded_images)

            # 计算奖励
            rewards = [calculate_reward_3fields(pred, ref_answer) for pred in candidates]

            grpo_dataset.append({
                "messages": item['messages'],
                "images": images,
                "candidates": candidates,
                "rewards": rewards,
                "reference": ref_answer
            })

        except Exception as e:
            print(f"\n❌ 处理失败 {img_path}: {e}")

        finally:
            pbar.update(1)

        # ========== 每100条保存和速度监控 ==========
        if (idx + 1) % 100 == 0:
            os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
            with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            eta = (len(sft_data) - idx - 1) / speed / 60

            vram = torch.cuda.memory_allocated() / 1024**3
            pbar.write(f"\n💾 已保存 {len(grpo_dataset)} 条 | 速度: {speed:.2f}条/秒 | 预计剩余: {eta:.1f}分钟 | 显存: {vram:.1f}GB")
            last_save = len(grpo_dataset)

    pbar.close()

    # 最终保存
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # 统计
    total_time = time.time() - start_time
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    # 奖励分布
    r_100 = sum(1 for r in all_rewards if r == 1.0)
    r_075_100 = sum(1 for r in all_rewards if 0.75 <= r < 1.0)
    r_000_075 = len(all_rewards) - r_100 - r_075_100

    print("\n" + "="*70)
    print(f"✅ 全部完成！")
    print(f"   成功样本: {len(grpo_dataset)}/{len(sft_data)}")
    print(f"   总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"   平均速度: {len(grpo_dataset)/total_time:.2f} 样本/秒")
    print(f"   峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print(f"\n📊 奖励分布:")
    print(f"   满分(三个全对): {r_100} ({r_100/len(all_rewards)*100:.1f}%)")
    print(f"   0.75-1.0 (错一个): {r_075_100} ({r_075_100/len(all_rewards)*100:.1f}%)")
    print(f"   0-0.75 (错2-3个): {r_000_075} ({r_000_075/len(all_rewards)*100:.1f}%)")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"\n   输出文件: {OUTPUT_DATASET_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()
