"""
🎯 三字段专用版 GRPO 数据集生成
针对营业执照 ['统一社会信用代码', '名称', '经营者'] 三字段优化

核心优化：
1. ✅ 奖励只看这三个字段，不检查多余字段
2. ✅ 字段缺失有惩罚（防止漏输出）
3. ✅ 相似度加权更合理
4. ✅ 多Worker充分利用24GB显存
"""
import os
import json
import re
import time
import torch
import threading
from queue import Queue
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===================== 环境变量 =====================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ===================== 配置 =====================
MODEL_PATH = '/path/to/model'
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 150

# 🔥 显存配置
NUM_WORKERS = 6  # 24GB 4090Ti 建议值

# GRPO配置
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]

# 🎯 只关注这三个字段！
TARGET_FIELDS = ['统一社会信用代码', '名称', '经营者']

# 线程安全的全局状态
result_queue = Queue()
processed_count = 0
lock = threading.Lock()

# ===================== 优化版奖励函数 =====================
def extract_json_safe(text):
    """快速提取JSON"""
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def field_similarity(a, b):
    """
    🎯 严格匹配模式：必须完全一致才算对
    完全匹配 = 1.0，否则 = 0.0
    """
    a, b = str(a).strip(), str(b).strip()

    # 空值处理
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0

    # 完全匹配才给分，差一个字符都不算对
    return 1.0 if a == b else 0.0

def calculate_reward_3fields(pred_str, ref_str):
    """
    🎯 严格匹配模式奖励函数
    权重分配:
    - 统一社会信用代码: 40%
    - 名称: 35%
    - 经营者: 25%
    每个字段：完全匹配=1分，不匹配=0分
    理论最高分 = 1.0 (三字段全对)
    """
    pred_json = extract_json_safe(pred_str)
    ref_json = extract_json_safe(ref_str)

    # 格式错误直接0分
    if pred_json is None:
        return 0.0

    # 标准答案也解析失败（极端情况），只给格式分
    if ref_json is None:
        return 0.2

    # 计算每个字段的得分
    score_code = field_similarity(pred_json.get('统一社会信用代码', ''), ref_json.get('统一社会信用代码', ''))
    score_name = field_similarity(pred_json.get('名称', ''), ref_json.get('名称', ''))
    score_owner = field_similarity(pred_json.get('经营者', ''), ref_json.get('经营者', ''))

    # 加权求和
    total_score = score_code * 0.40 + score_name * 0.35 + score_owner * 0.25

    return total_score

# ===================== Worker 线程 =====================
def worker_thread(worker_id, model, processor, device, work_queue):
    global processed_count

    while True:
        item = work_queue.get()
        if item is None:
            break

        try:
            images = item.get('images', '')
            image_path = images if isinstance(images, str) else images[0]

            # 提取数据
            ref_answer = None
            user_prompt = None
            for msg in item['messages']:
                if msg['role'] == 'assistant':
                    ref_answer = msg['content']
                elif msg['role'] == 'user':
                    user_prompt = msg['content']

            if ref_answer is None:
                continue

            # ========== 图片编码只做一次 ==========
            image = Image.open(image_path).convert('RGB')
            chat_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(device)

            # ========== 4个温度生成 ==========
            candidates = []
            with torch.no_grad():
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

            # ========== 计算奖励 ==========
            rewards = [calculate_reward_3fields(pred, ref_answer) for pred in candidates]

            result_queue.put({
                "messages": item['messages'],
                "images": images,
                "candidates": candidates,
                "rewards": rewards,
                "reference": ref_answer
            })

            with lock:
                processed_count += 1

        except Exception as e:
            print(f"\n❌ Worker {worker_id}: {e}")
            continue

        finally:
            work_queue.task_done()

# ===================== 主函数 =====================
def main():
    global processed_count

    print("="*70)
    print("🎯 三字段专用版 GRPO 数据集生成")
    print("="*70)
    print(f"目标字段: {TARGET_FIELDS}")
    print(f"权重分配: 统一社会信用代码=40%, 名称=35%, 经营者=25%")
    print(f"Worker数: {NUM_WORKERS} | 候选温度: {TEMPERATURES}")
    print("="*70)

    # 加载模型
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

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"总样本数: {len(sft_data)}")

    # ========== 测试模式（去掉注释跑100条看看）==========
    # sft_data = sft_data[:100]
    # print(f"⚠️ 测试模式: 只处理前 {len(sft_data)} 条")

    # 创建任务队列（取消maxsize避免死锁）
    work_queue = Queue()

    # 启动 Worker
    print(f"\n🚀 启动 {NUM_WORKERS} 个 Worker...")
    threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_thread, args=(i, model, processor, device, work_queue))
        t.daemon = True
        t.start()
        threads.append(t)

    # 先启动线程，再往队列放任务（不会阻塞了）
    for item in sft_data:
        work_queue.put(item)

    # 最后放退出信号（每个线程一个None）
    for _ in range(NUM_WORKERS):
        work_queue.put(None)

    # ========== 监控进度 ==========
    grpo_dataset = []
    start_time = time.time()
    pbar = tqdm(total=len(sft_data), desc="生成进度")

    last_save = 0
    while processed_count < len(sft_data):
        while not result_queue.empty():
            grpo_dataset.append(result_queue.get())
            pbar.update(1)

        # 每200条保存
        if len(grpo_dataset) - last_save >= 200:
            os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
            with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

            vram = torch.cuda.memory_allocated() / 1024**3
            elapsed = time.time() - start_time
            pbar.write(f"\n💾 已保存 {len(grpo_dataset)} 条 | 显存: {vram:.1f}GB | 速度: {len(grpo_dataset)/elapsed:.2f}条/秒")
            last_save = len(grpo_dataset)

        time.sleep(0.1)

    pbar.close()

    # 等待线程结束
    for t in threads:
        t.join()

    # 收集剩余结果
    while not result_queue.empty():
        grpo_dataset.append(result_queue.get())

    # 最终保存
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # 统计
    total_time = time.time() - start_time
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    # 奖励分布统计
    r_08_10 = sum(1 for r in all_rewards if 0.8 <= r <= 1.0)
    r_05_08 = sum(1 for r in all_rewards if 0.5 <= r < 0.8)
    r_00_05 = len(all_rewards) - r_08_10 - r_05_08

    print("\n" + "="*70)
    print(f"✅ 全部完成！")
    print(f"   样本数: {len(grpo_dataset)}")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   速度: {len(grpo_dataset)/total_time:.2f} 样本/秒")
    print(f"   峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print(f"\n📊 奖励分布:")
    print(f"   0.8-1.0 (优秀): {r_08_10} ({r_08_10/len(all_rewards)*100:.1f}%)")
    print(f"   0.5-0.8 (一般): {r_05_08} ({r_05_08/len(all_rewards)*100:.1f}%)")
    print(f"   0.0-0.5 (较差): {r_00_05} ({r_00_05/len(all_rewards)*100:.1f}%)")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"\n   输出文件: {OUTPUT_DATASET_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()
