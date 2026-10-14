"""
🚀🚀🚀 终极加速版 GRPO 数据集生成
双层并发：外层样本并发 + 内层候选并发

理论速度：3000条 ~ 15分钟（并发数足够时）
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

# GRPO配置
NUM_CANDIDATES = 4
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]
MAX_WORKERS_SAMPLES = 8   # 外层样本并发数
MAX_WORKERS_CANDIDATES = 4 # 内层候选并发数（每个样本内部）

# 字段配置
FIELD_CONFIGS = [
    ['统一社会信用代码', '名称', '经营者'],
    ['统一社会信用代码', '名称', '类型', '经营者', '注册资本', '成立日期', '经营期限', '住址']
]

# ===================== 工具函数 =====================
def extract_json_from_text(text):
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def get_fields_from_prompt(prompt_content):
    for fields in FIELD_CONFIGS:
        for field in fields:
            if field in prompt_content:
                return fields
    return FIELD_CONFIGS[0]

def calculate_total_reward(pred_str, ref_str, fields=None):
    pred_json = extract_json_from_text(pred_str)
    ref_json = extract_json_from_text(ref_str)

    if pred_json is None or ref_json is None:
        field_reward = 0.0
    else:
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
        field_reward = correct_fields / total_fields if total_fields > 0 else 0.0

    format_reward = 1.0 if pred_json is not None else 0.0
    return field_reward * 0.8 + format_reward * 0.2

# ===================== HTTP 调用（带重试）=====================
def call_api_single(image_path, prompt, temperature, timeout=25, max_retries=2):
    """单次API调用，支持重试"""
    for attempt in range(max_retries + 1):
        try:
            # 读取图片并base64编码
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
                "temperature": temperature
            }

            response = requests.post(BASE_URL, headers=headers, json=data, timeout=timeout)

            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                time.sleep(0.5)  # 失败后延迟重试

        except Exception as e:
            if attempt < max_retries:
                time.sleep(0.5)
            continue

    return ""  # 全部重试失败

# ===================== 单个样本：4个候选并发生成 =====================
def process_single_sample(item):
    """处理一个样本，4个候选并发生成"""
    messages = item['messages']
    images = item.get('images', '')

    ref_answer = None
    user_prompt = None
    for msg in messages:
        if msg['role'] == 'assistant':
            ref_answer = msg['content']
        elif msg['role'] == 'user':
            user_prompt = msg['content']

    if ref_answer is None or not images:
        return None

    image_path = images if isinstance(images, str) else images[0]
    fields = get_fields_from_prompt(user_prompt or "")

    # ========== 4个候选并发生成！ ==========
    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_CANDIDATES) as executor:
        futures = {executor.submit(call_api_single, image_path, user_prompt, temp): temp for temp in TEMPERATURES}
        for future in concurrent.futures.as_completed(futures):
            try:
                candidates.append(future.result())
            except:
                candidates.append("")

    # 计算奖励
    rewards = [calculate_total_reward(pred, ref_answer, fields) for pred in candidates]

    return {
        "messages": messages,
        "images": images,
        "candidates": candidates,
        "rewards": rewards,
        "reference": ref_answer
    }

# ===================== 主函数 =====================
def main():
    print("="*70)
    print("🚀🚀🚀 终极版 GRPO 数据集生成 (双层并发)")
    print("="*70)
    print(f"服务地址: {BASE_URL}")
    print(f"模型: {MODEL}")
    print(f"样本并发: {MAX_WORKERS_SAMPLES} | 候选并发: {MAX_WORKERS_CANDIDATES}")
    print(f"理论最大并发: {MAX_WORKERS_SAMPLES * MAX_WORKERS_CANDIDATES} 请求")
    print("="*70)

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"\n📚 总样本数: {len(sft_data)}")

    # ========== 可选：先测试前10条 ==========
    # sft_data = sft_data[:10]
    # print(f"⚠️ 测试模式：只处理前 {len(sft_data)} 条")

    grpo_dataset = []
    start_time = time.time()
    success_count = 0

    # ========== 外层：样本级并发 ==========
    print(f"\n🚀 开始生成...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_SAMPLES) as executor:
        futures = {executor.submit(process_single_sample, item): item for item in sft_data}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sft_data), desc="进度"):
            try:
                result = future.result()
                if result:
                    grpo_dataset.append(result)
                    success_count += 1
            except Exception as e:
                print(f"\n❌ 样本失败: {e}")

            # ========== 增量保存 ==========
            if len(grpo_dataset) % 200 == 0 and len(grpo_dataset) > 0:
                os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
                with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
                    json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)
                tqdm.write(f"\n💾 已保存 {len(grpo_dataset)} 条")

    # ========== 最终保存 ==========
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # ========== 统计 ==========
    total_time = time.time() - start_time
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    print("\n" + "="*70)
    print(f"✅ 全部完成！")
    print(f"   成功: {success_count}/{len(sft_data)} 条")
    print(f"   总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"   速度: {len(sft_data)/total_time:.2f} 样本/秒 | {12*60*60/len(sft_data):.1f} 样本/小时")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"   输出: {OUTPUT_DATASET_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()
