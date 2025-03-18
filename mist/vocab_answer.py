import json

# 输入文件路径
input_file = 'C:/Users/xiwei/Desktop/agqa/vocab.json'

# 输出文件路径
output_file = 'C:/Users/xiwei/Desktop/answers.json'

# 读取 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取答案（即所有的键）
answers = list(data.keys())

# 保存到新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

print(f"答案已提取并保存在文件 {output_file}")
