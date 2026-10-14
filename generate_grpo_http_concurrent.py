"""
⚡⚡⚡ 超级加速版 GRPO 数据集生成
通过 HTTP 调用线上 HunyuanOCR 服务 + 多线程并发

预计速度: 3000条 ~ 50分钟 (并发数=16时)
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
MAX_WORKERS = 16  # 并发数，根据服务能力调整，16-32都可以

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

# ===================== 奖励函数（严格三字段匹配）=====================
def field_similarity(a, b):
    """严格匹配：完全一致才算对"""
    a, b = str(a).strip(), str(b).strip()
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    return 1.0 if a == b else 0.0

def calculate_total_reward(pred_str, ref_str, fields=None):
    """
    权重：统一社会信用代码40%，名称35%，经营者25%
    每个字段必须完全匹配才得分
    """
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

# ===================== HTTP 调用函数 =====================
def call_api(image_path, prompt, temperature, timeout=30):
    """调用一次线上API，返回生成结果"""
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
        else:
            print(f"API错误 {response.status_code}: {response.text[:100]}")
            return ""

    except Exception as e:
        print(f"调用失败: {e}")
        return ""

# ===================== 单样本处理（生成4个候选）=====================
def process_single_sample(item):
    """处理一个样本，生成4个候选并计算奖励"""
    messages = item['messages']
    images = item.get('images', '')

    # 提取数据
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

    # 串行生成4个候选（一个样本内串行，样本间并发）
    candidates = []
    for temp in TEMPERATURES:
        pred = call_api(image_path, user_prompt, temp)
        candidates.append(pred)

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
    print("⚡ HTTP并发版 GRPO 数据集生成")
    print("="*70)
    print(f"服务地址: {BASE_URL}")
    print(f"模型: {MODEL}")
    print(f"并发数: {MAX_WORKERS}")
    print(f"候选数: {NUM_CANDIDATES}")
    print(f"温度: {TEMPERATURES}")
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

    # ========== 多线程并发处理 ==========
    print(f"\n🚀 开始并发生成 (并发数={MAX_WORKERS})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, item): item for item in sft_data}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sft_data), desc="生成进度"):
            try:
                result = future.result()
                if result:
                    grpo_dataset.append(result)
            except Exception as e:
                print(f"\n❌ 样本处理失败: {e}")

            # ========== 增量保存（每100条） ==========
            if len(grpo_dataset) % 100 == 0 and len(grpo_dataset) > 0:
                os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
                with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
                    json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # ========== 最终保存 ==========
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # ========== 统计 ==========
    total_time = time.time() - start_time
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    print("\n" + "="*70)
    print(f"✅ 完成！")
    print(f"   成功样本: {len(grpo_dataset)}/{len(sft_data)}")
    print(f"   总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"   平均速度: {len(sft_data)/total_time:.2f} 样本/秒")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"   输出文件: {OUTPUT_DATASET_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()
