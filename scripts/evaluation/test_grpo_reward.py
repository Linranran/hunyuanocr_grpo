"""
测试GRPO奖励函数
验证奖励计算是否正确
"""
import json
import re
from difflib import SequenceMatcher

def extract_json_from_text(text):
    """从文本中提取JSON内容"""
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def calculate_field_level_reward(pred_str, ref_str, fields):
    """计算字段级别的奖励分数"""
    pred_json = extract_json_from_text(pred_str)
    ref_json = extract_json_from_text(ref_str)

    if pred_json is None or ref_json is None:
        return 0.0

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

def calculate_total_reward(pred_str, ref_str, fields):
    """计算总奖励：字段准确率 * 0.8 + 格式合规性 * 0.2"""
    field_reward = calculate_field_level_reward(pred_str, ref_str, fields)
    format_reward = calculate_format_reward(pred_str)
    return field_reward * 0.8 + format_reward * 0.2


def test_reward():
    print("="*60)
    print("GRPO奖励函数测试")
    print("="*60)

    # 测试用例
    fields = ['统一社会信用代码', '名称', '经营者']

    ref_answer = '{"统一社会信用代码": "91371301MA3D0AH504", "名称": "临沂商城联财货物托运处", "经营者": "赵锦昊"}'

    test_cases = [
        ("完全匹配", '{"统一社会信用代码": "91371301MA3D0AH504", "名称": "临沂商城联财货物托运处", "经营者": "赵锦昊"}'),
        ("1个字段错误", '{"统一社会信用代码": "91371301MA3D0AH504", "名称": "临沂商城联财货物托运处", "经营者": "张三"}'),
        ("2个字段错误", '{"统一社会信用代码": "91371301MA3D0AH504", "名称": "错误名称", "经营者": "张三"}'),
        ("完全错误", '{"统一社会信用代码": "XXXXXX", "名称": "错误名称", "经营者": "张三"}'),
        ("格式错误", '这不是一个json格式'),
        ("部分匹配(相似度90%)", '{"统一社会信用代码": "91371301MA3D0AH504", "名称": "临沂商城联财货物托运部", "经营者": "赵锦昊"}'),
    ]

    print(f"\n标准答案: {ref_answer}\n")

    for case_name, pred in test_cases:
        reward = calculate_total_reward(pred, ref_answer, fields)
        field_reward = calculate_field_level_reward(pred, ref_answer, fields)
        format_reward = calculate_format_reward(pred)

        print(f"[{case_name}]")
        print(f"  字段奖励: {field_reward:.3f}, 格式奖励: {format_reward:.3f}, 总奖励: {reward:.3f}")
        print(f"  预测: {pred[:100]}...\n" if len(pred) > 100 else f"  预测: {pred}\n")

    print("="*60)
    print("测试完成！")

if __name__ == '__main__':
    test_reward()
