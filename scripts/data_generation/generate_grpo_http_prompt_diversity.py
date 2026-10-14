"""
🧠 GRPO数据集生成 - Prompt多样性方案
Temperature不生效 → 用4种不同的提问方式来制造候选多样性
"""
import os
import json
import re
import base64
import time
import requests
import concurrent.futures
from tqdm import tqdm
from difflib import SequenceMatcher

# ===================== 配置 =====================
BASE_URL = os.environ.get("MODEL_API_BASE_URL", "")
API_KEY = os.environ.get("MODEL_API_KEY", "")
MODEL = "100202"

SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

# 🔥 核心：4种不同的提问方式，来制造候选多样性
PROMPT_VARIANTS = [
    "提取图片中的: ['统一社会信用代码','名称', '经营者'] 的字段内容，并按照 JSON 格式返回。",
    "识别这张营业执照的统一社会信用代码、名称、经营者，用JSON格式输出结果。",
    "请OCR识别图片中的营业执照关键字段：统一社会信用代码、名称、经营者，返回JSON。",
    "读出图片中的三个字段：统一社会信用代码、名称、经营者，严格按照JSON格式返回。",
]

MAX_WORKERS = 6
MAX_RETRIES = 2
RETRY_BACKOFF = 2

# ===================== 工具函数 =====================
def extract_json_from_text(text):
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def field_similarity(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    return 1.0 if a == b else 0.0

def calculate_total_reward(pred_str, ref_str):
    """权重：统一社会信用代码40%，名称35%，经营者25%"""
    pred_json = extract_json_from_text(pred_str)
    ref_json = extract_json_from_text(ref_str)

    if pred_json is None:
        return 0.0
    if ref_json is None:
        return 0.2

    score_code = field_similarity(pred_json.get('统一社会信用代码', ''), ref_json.get('统一社会信用代码', ''))
    score_name = field_similarity(pred_json.get('名称', ''), ref_json.get('名称', ''))
    score_owner = field_similarity(pred_json.get('经营者', ''), ref_json.get('经营者', ''))

    return score_code * 0.40 + score_name * 0.35 + score_owner * 0.25

# ===================== HTTP调用 =====================
def call_api_with_retry(image_path, prompt, timeout=45):
    for attempt in range(MAX_RETRIES):
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            base64_string = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')

            messageList = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_string}},
                        {"type": "text", "text": prompt}
                    ],
                }
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            data = {
                "model": MODEL,
                "messages": messageList,
                "max_tokens": 200,
                "temperature": 1.0,  # 虽然不生效，但还是加上
                "do_sample": True,   # 尝试显式指定
            }

            response = requests.post(BASE_URL, headers=headers, json=data, timeout=timeout)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            else:
                print(f"❌ HTTP {response.status_code}")
                return ""

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            print(f"❌ 调用失败: {e}")
            return ""
    return ""

# ===================== 单样本处理：4种Prompt各生成一个候选 =====================
def process_single_sample(item):
    messages = item['messages']
    images = item.get('images', '')

    ref_answer = None
    for msg in messages:
        if msg['role'] == 'assistant':
            ref_answer = msg['content']
            break

    if ref_answer is None or not images:
        return None

    image_path = images if isinstance(images, str) else images[0]

    # 🔥 用4种不同的prompt生成4个候选
    candidates = []
    for prompt in PROMPT_VARIANTS:
        pred = call_api_with_retry(image_path, prompt)
        candidates.append(pred)

    # 计算奖励
    rewards = [calculate_total_reward(pred, ref_answer) for pred in candidates]

    return {
        "messages": messages,
        "images": images,
        "prompt_variants_used": PROMPT_VARIANTS,
        "candidates": candidates,
        "rewards": rewards,
        "reference": ref_answer
    }

# ===================== 主函数 =====================
def main():
    print("="*80)
    print("🧠 Prompt多样性 GRPO 数据集生成")
    print("="*80)
    print(f"候选生成方式: 用 {len(PROMPT_VARIANTS)} 种不同的提问方式")
    print(f"并发数: {MAX_WORKERS}")
    print("\n使用的Prompt变体:")
    for i, p in enumerate(PROMPT_VARIANTS, 1):
        print(f"  [{i}] {p}")
    print("="*80)

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"\n📚 总样本数: {len(sft_data)}")

    # ========== 测试模式 ==========
    # sft_data = sft_data[:20]
    # print(f"⚠️ 测试模式: 只处理前 {len(sft_data)} 条")

    grpo_dataset = []
    start_time = time.time()

    # 检查点恢复
    checkpoint_path = OUTPUT_DATASET_PATH.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_path):
        print(f"\n📂 加载检查点...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            grpo_dataset = json.load(f)
        processed_paths = {item.get('images', '') for item in grpo_dataset}
        sft_data = [item for item in sft_data if item.get('images', '') not in processed_paths]
        print(f"   已处理: {len(grpo_dataset)} 条，剩余: {len(sft_data)} 条")

    # ========== 多线程处理 ==========
    print(f"\n🚀 开始生成...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, item): item for item in sft_data}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sft_data), desc="处理进度"):
            try:
                result = future.result()
                if result and any(c.strip() for c in result['candidates']):
                    grpo_dataset.append(result)
            except Exception as e:
                print(f"\n❌ 异常: {e}")

            # 增量保存
            if len(grpo_dataset) % 50 == 0 and len(grpo_dataset) > 0:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # 最终保存
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # 清理检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # 统计
    total_time = time.time() - start_time
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    # 候选多样性统计
    diversity_stats = []
    for item in grpo_dataset:
        unique_cands = len(set(item['candidates']))
        diversity_stats.append(unique_cands)
    avg_diversity = sum(diversity_stats) / len(diversity_stats) if diversity_stats else 0

    # 奖励分布
    r_100 = sum(1 for r in all_rewards if r == 1.0)
    r_075 = sum(1 for r in all_rewards if 0.75 <= r < 1.0)
    r_other = len(all_rewards) - r_100 - r_075

    print("\n" + "="*80)
    print(f"✅ 全部完成！")
    print(f"   成功样本: {len(grpo_dataset)}")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   平均速度: {len(grpo_dataset)/total_time:.2f} 样本/秒")
    print(f"\n📊 候选多样性分析:")
    print(f"   平均唯一候选数: {avg_diversity:.2f}/4")
    print(f"\n🏆 奖励分布:")
    print(f"   满分(三个全对): {r_100} ({r_100/len(all_rewards)*100:.1f}%)")
    print(f"   错一个(0.75分): {r_075} ({r_075/len(all_rewards)*100:.1f}%)")
    print(f"   错2-3个(<0.75): {r_other} ({r_other/len(all_rewards)*100:.1f}%)")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"\n   输出文件: {OUTPUT_DATASET_PATH}")
    print("="*80)

if __name__ == '__main__':
    main()
