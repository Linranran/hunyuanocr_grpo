"""
🧠🧠🧠 多Worker并发版 - 真正吃满24GB显存
一个worker占~5GB，24GB可以跑 4-6 个worker同时处理

核心思想：多线程同时调用 model.generate，让GPU一直满载
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
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ===================== 配置 =====================
MODEL_PATH = '/path/to/model'
SFT_TRAIN_DATA = '/path/to/dataset'
OUTPUT_DATASET_PATH = '/path/to/dataset'

DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 150

# 🔥 24GB显存配置：4-6个Worker最理想
NUM_WORKERS = 6  # 每个Worker ~5GB，6个 = ~30GB，实际会共享权重，所以~15-18GB

GRPO_CONFIG = {
    "num_candidates": 4,
    "temperatures": [0.3, 0.7, 1.0, 1.2]
}

FIELD_CONFIGS = [
    ['统一社会信用代码', '名称', '经营者'],
    ['统一社会信用代码', '名称', '类型', '经营者', '注册资本', '成立日期', '经营期限', '住址']
]

# ===================== 全局状态（线程安全）=====================
result_queue = Queue()
processed_count = 0
lock = threading.Lock()

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

# ===================== Worker 线程 =====================
def worker_thread(worker_id, model, processor, device, work_queue):
    """每个Worker独立处理任务"""
    global processed_count

    while True:
        item = work_queue.get()
        if item is None:  # 退出信号
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

            fields = get_fields_from_prompt(user_prompt or "")

            # ========== 加载图片 + 编码 ==========
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

            # 🔥 每个线程都会调用 model.generate，GPU上同时跑 NUM_WORKERS 个生成任务
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(device)

            # ========== 4个候选生成 ==========
            candidates = []
            for temp in GRPO_CONFIG['temperatures']:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=temp,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=processor.tokenizer.pad_token_id,
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

            # 计算奖励
            rewards = [calculate_total_reward(pred, ref_answer, fields) for pred in candidates]

            result_queue.put({
                "messages": item['messages'],
                "images": images,
                "candidates": candidates,
                "rewards": rewards,
                "reference": ref_answer
            })

            # 更新计数
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
    print("🧠🧠🧠 多Worker并发版 GRPO 生成 (吃满24GB显存)")
    print("="*70)
    print(f"模型: {MODEL_PATH}")
    print(f"Worker数: {NUM_WORKERS} | 候选数: {GRPO_CONFIG['num_candidates']}")
    print(f"预计显存: ~15-18 GB (24GB卡很宽裕)")
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

    # 🔥 关键：启用 CUDA 图优化和 cudnn benchmark
    torch.backends.cudnn.benchmark = True

    device = next(model.parameters()).device
    print(f"设备: {device}")
    print(f"初始显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # 加载数据
    with open(SFT_TRAIN_DATA, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    print(f"总样本数: {len(sft_data)}")

    # ========== 可选：测试模式 ==========
    # sft_data = sft_data[:200]
    # print(f"⚠️ 测试模式: 只处理前 {len(sft_data)} 条")

    # 创建任务队列
    work_queue = Queue(maxsize=100)
    for item in sft_data:
        work_queue.put(item)

    # 启动 Worker 线程
    print(f"\n🚀 启动 {NUM_WORKERS} 个 Worker 线程...")
    threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_thread, args=(i, model, processor, device, work_queue))
        t.daemon = True
        t.start()
        threads.append(t)
        work_queue.put(None)  # 每个线程一个退出信号

    # ========== 监控进度 + 增量保存 ==========
    grpo_dataset = []
    start_time = time.time()
    pbar = tqdm(total=len(sft_data), desc="总进度")

    last_save = 0
    while processed_count < len(sft_data):
        # 收集结果
        while not result_queue.empty():
            grpo_dataset.append(result_queue.get())
            pbar.update(1)

        # 每200条保存一次
        if len(grpo_dataset) - last_save >= 200:
            os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
            with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            elapsed = time.time() - start_time
            pbar.write(f"\n💾 已保存 {len(grpo_dataset)} 条 | 显存: {vram_used:.1f}/{vram_total:.1f} GB | 速度: {len(grpo_dataset)/elapsed:.2f} 样本/秒")
            last_save = len(grpo_dataset)

        time.sleep(0.1)

    pbar.close()

    # 等待所有线程结束
    for t in threads:
        t.join()

    # 收集剩余结果
    while not result_queue.empty():
        grpo_dataset.append(result_queue.get())

    # 最终保存
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)

    # 最终统计
    total_time = time.time() - start_time
    all_rewards = [r for item in grpo_dataset for r in item['rewards']]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    print("\n" + "="*70)
    print(f"✅ 全部完成！")
    print(f"   样本数: {len(grpo_dataset)}/{len(sft_data)}")
    print(f"   总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"   平均速度: {len(grpo_dataset)/total_time:.2f} 样本/秒")
    print(f"   3000条耗时: {(3000 * total_time / len(grpo_dataset))/60:.1f} 分钟")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"   峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print("="*70)

if __name__ == '__main__':
    main()
