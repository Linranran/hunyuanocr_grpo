"""
⚡⚡⚡ 真正的Batch版本 GRPO 生成
一次性输入 BATCH_SIZE 张图片到 model.generate，GPU真正并行处理

性能提升：比多线程快 30-50%
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

# 🔥 Batch大小：4张图片同时推理（24GB卡可以调到6-8）
BATCH_SIZE = 6

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
    print("⚡ Batch版本 GRPO 生成 (GPU真正并行)")
    print("="*70)
    print(f"Batch size: {BATCH_SIZE} | 候选温度: {TEMPERATURES}")
    print(f"每张图片生成 {len(TEMPERATURES)} 个候选，每批处理 {BATCH_SIZE} 张图片")
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

# ===================== 核心：Batch生成 =====================
@torch.no_grad()
def generate_batch_candidates(model, processor, device, batch_items):
    """
    ⚡ 核心优化：一次性编码整个batch，一次性生成所有候选
    """
    results = []

    # ========== Step 1: 预加载所有图片和prompt ==========
    images = []
    prompts = []
    ref_answers = []
    fields_list = []

    for item in batch_items:
        img_path = item.get('images', '')
        if isinstance(img_path, list):
            img_path = img_path[0]

        # 提取prompt和答案
        ref_answer = None
        user_prompt = None
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                ref_answer = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']

        if ref_answer is None:
            continue

        images.append(Image.open(img_path).convert('RGB'))
        prompts.append(user_prompt)
        ref_answers.append(ref_answer)
        fields_list.append(TARGET_FIELDS)

    if not images:
        return []

    # ========== Step 2: 只编码一次！整个batch一起编码 ==========
    # 构建所有样本的messages
    all_messages = []
    for user_prompt, img_path in zip(prompts, [item.get('images', '') for item in batch_items]):
        if isinstance(img_path, list):
            img_path = img_path[0]
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        all_messages.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    # 🔥 整个batch一次性编码到GPU！只做一次！
    inputs = processor(
        text=all_messages,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # ========== Step 3: 每个温度生成一次（batch一起生成）==========
    all_candidates = [[] for _ in range(len(images))]  # N图片 × 4候选

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

        # 解码整个batch
        for i in range(len(images)):
            input_len = inputs.input_ids[i].shape[0]
            generated_ids_trimmed = generated_ids[i:i+1, input_len:]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            all_candidates[i].append(output_text.strip())

    # ========== Step 4: 批量计算奖励 ==========
    for i, item in enumerate(batch_items):
        if i >= len(ref_answers):
            continue

        candidates = all_candidates[i]
        ref_answer = ref_answers[i]
        rewards = [calculate_reward_3fields(pred, ref_answer) for pred in candidates]

        results.append({
            "messages": item['messages'],
            "images": item.get('images', ''),
            "candidates": candidates,
            "rewards": rewards,
            "reference": ref_answer
        })

    return results

# ===================== 主函数 =====================
def main():
    model, processor, device = load_model_and_processor()

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"总样本数: {len(sft_data)}")

    # ========== 测试模式 ==========
    # sft_data = sft_data[:120]
    # print(f"⚠️ 测试模式: 只处理前 {len(sft_data)} 条")

    grpo_dataset = []
    start_time = time.time()

    print(f"\n🚀 开始生成 (batch_size={BATCH_SIZE})...")
    pbar = tqdm(total=len(sft_data), desc="总进度")

    last_save = 0

    # ========== 按批处理 ==========
    for i in range(0, len(sft_data), BATCH_SIZE):
        batch = sft_data[i:i + BATCH_SIZE]

        # 处理一批
        batch_results = generate_batch_candidates(model, processor, device, batch)

        for result in batch_results:
            grpo_dataset.append(result)
            pbar.update(1)

        # 每240条保存（BATCH_SIZE的整数倍）
        if len(grpo_dataset) - last_save >= 240:
            os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
            with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

            vram = torch.cuda.memory_allocated() / 1024**3
            elapsed = time.time() - start_time
            pbar.write(f"\n💾 已保存 {len(grpo_dataset)} 条 | 显存: {vram:.1f}GB | 速度: {len(grpo_dataset)/elapsed:.2f}条/秒")
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

    print("\n" + "="*70)
    print(f"✅ 全部完成！")
    print(f"   样本数: {len(grpo_dataset)}")
    print(f"   总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"   平均速度: {len(grpo_dataset)/total_time:.2f} 样本/秒")
    print(f"   3000条预计: {(3000 * total_time / len(grpo_dataset))/60:.1f} 分钟")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"   峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print(f"   输出文件: {OUTPUT_DATASET_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()
