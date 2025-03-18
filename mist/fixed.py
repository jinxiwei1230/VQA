import json

def convert_json_to_jsonl(input_file, output_file):
    # 打开输入文件和输出文件
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        data = json.load(infile)  # 加载 JSON 数组
        for item in data:
            # 将每个 JSON 对象写入输出文件的一行
            outfile.write(json.dumps(item) + '\n')

# 调用函数修复文件
convert_json_to_jsonl(
    '/home/disk2/dachuang1-23/data/datasets/agqa/agqa_train_v2.jsonl',  # 输入文件
    '/home/disk2/dachuang1-23/data/datasets/agqa/agqa_train_v2_fixed.jsonl'  # 输出文件
)

convert_json_to_jsonl(
    '/home/disk2/dachuang1-23/data/datasets/agqa/agqa_val_v2.jsonl',
    '/home/disk2/dachuang1-23/data/datasets/agqa/agqa_val_v2_fixed.jsonl'
)

convert_json_to_jsonl(
    '/home/disk2/dachuang1-23/data/datasets/agqa/agqa_test_v2.jsonl',
    '/home/disk2/dachuang1-23/data/datasets/agqa/agqa_test_v2_fixed.jsonl'
)
