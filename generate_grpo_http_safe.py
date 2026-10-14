"""
🛡️ 安全稳健版 HTTP GRPO 生成
增加：自动重试、指数退避、错误处理、并发数降低

针对EngineCore错误和401问题优化
"""
import os
import json
import re
import base64
import time
import random
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

# 🔥 降低并发数！避免打垮服务
MAX_WORKERS = 6

# 重试配置
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # 指数退避基础秒数

# GRPO配置
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]

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
    """严格匹配：完全一致才算对"""
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

# ===================== HTTP 调用（带重试）=====================
def call_api_with_retry(image_path, prompt, temperature, timeout=45):
    """
    🛡️ 带重试和指数退避的API调用
    """
    for attempt in range(MAX_RETRIES):
        try:
            # 读取图片并base64编码
            with open(image_path, 'rb') as f:
                image_data = f.read()
            base64_string = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')

            # 构建请求
            messageList = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_string},
                        },
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

            elif response.status_code in [429, 500, 502, 503, 504]:
                # 服务端错误，重试
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"❌ 最终失败 HTTP {response.status_code}: {response.text[:80]}")
                    return ""
            else:
                # 其他错误（如401），不重试
                print(f"❌ HTTP {response.status_code}: {response.text[:80]}")
                return ""

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            print(f"⏰ 超时失败: {os.path.basename(image_path)}")
            return ""

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            print(f"❌ 调用失败: {e}")
            return ""

    return ""

# ===================== 单样本处理 =====================
def process_single_sample(item):
    """处理一个样本，生成4个候选"""
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

    # 生成4个候选
    candidates = []
    for temp in TEMPERATURES:
        pred = call_api_with_retry(image_path, user_prompt, temp)
        candidates.append(pred)

    # 计算奖励
    rewards = [calculate_total_reward(pred, ref_answer) for pred in candidates]

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
    print("🛡️ 安全稳健版 HTTP GRPO 生成")
    print("="*70)
    print(f"并发数: {MAX_WORKERS} (降低并发，减少服务压力)")
    print(f"自动重试: {MAX_RETRIES}次 + 指数退避")
    print(f"温度参数: {TEMPERATURES}")
    print("="*70)

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"\n📚 总样本数: {len(sft_data)}")

    # ========== 测试模式 ==========
    # sft_data = sft_data[:20]
    # print(f"⚠️ 测试模式: 只处理前 {len(sft_data)} 条")

    grpo_dataset = []
    start_time = time.time()
    success_count = 0
    fail_count = 0

    # 增量保存检查点
    checkpoint_path = OUTPUT_DATASET_PATH.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_path):
        print(f"📂 发现检查点文件，加载已处理数据...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            grpo_dataset = json.load(f)
        processed_paths = {item.get('images', '') for item in grpo_dataset}
        sft_data = [item for item in sft_data if item.get('images', '') not in processed_paths]
        print(f"   已处理: {len(grpo_dataset)} 条，剩余: {len(sft_data)} 条")

    # ========== 多线程并发 ==========
    print(f"\n🚀 开始处理 (并发数={MAX_WORKERS})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, item): item for item in sft_data}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sft_data), desc="处理进度"):
            try:
                result = future.result()
                if result and any(c.strip() for c in result['candidates']):
                    grpo_dataset.append(result)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"\n❌ 异常: {e}")
                fail_count += 1

            # ========== 增量保存（每50条存检查点）==========
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

    print("\n" + "="*70)
    print(f"✅ 处理完成！")
    print(f"   成功: {success_count} | 失败: {fail_count}")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   平均速度: {len(sft_data)/total_time:.2f} 样本/秒")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"   输出文件: {OUTPUT_DATASET_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()
