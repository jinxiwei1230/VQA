import os
import json
import pandas as pd
import numpy as np

# 这段代码的功能是将原始的 `train.csv` 和 `val.csv` 文件中的问答数据进行处理，并将其转换成统一的JSON格式文件（`train.json` 和 `val.json`）。
# 这个过程包括从CSV文件中读取数据、映射视频ID、并按照特定的格式进行重新组织。以下是代码的详细解释：

def convert(o):
    #convert(o)：这是一个辅助函数，用于将 `numpy.int64` 类型的整数转换为普通的Python整数类型，以确保这些数据可以被正确地序列化为JSON格式。

    if isinstance(o, np.int64):
        return int(o)
    raise TypeError(f"对象类型 {o.__class__.__name__} 无法进行 JSON 序列化")


def save_json(content, save_path):
    #save_json(content, save_path)：用于将数据保存为JSON文件，使用 `convert` 函数处理可能的 `int64` 类型。

    with open(save_path, "w") as f:
        json.dump(content, f, default=convert)


def load_jsonl(filename):
    #load_jsonl(filename)：用于加载以 `.jsonl`（JSON lines）格式存储的数据文件。
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

# 将 int64 值转换为普通的 Python 整数
path = "/home/disk2/VQA/NExT-QA/qa_annos"# 设置了原始CSV文件和映射文件的路径。
raw_train_csv = os.path.join(path, "train.csv")
raw_val_csv = os.path.join(path, "val.csv")
raw_train = pd.read_csv(raw_train_csv, delimiter=",")
raw_val = pd.read_csv(raw_val_csv, delimiter=",")
#初始化空列表 `train` 和 `val` 来存储处理后的数据。
train = []
val = []
#定义了一个关键字段列表 `key`，包括 `video_id`、`question`、`answer` 等。
key = ["video_id", "question", "a0", "a1", "a2", "a3", "a4", "answer", "qid", "type"]

#    - 循环处理训练数据：对 `raw_train` 中的每一行数据进行处理，提取关键字段，并将其存入 `train` 列表中。
#    - 循环处理验证数据：对 `raw_val` 进行类似的处理，将结果存入 `val` 列表中。
for i in range(len(raw_train)):
    data = {}
    for k in key:
        data[k] = raw_train.iloc[i][k]
    train.append(data)

for i in range(len(raw_val)):
    data = {}
    for k in key:
        data[k] = raw_val.iloc[i][k]
    val.append(data)

#使用 `json.load` 加载视频ID映射文件 `map_vid_vidorID.json`，将原始的 `video_id` 映射到新的ID。
map_json = os.path.join(path, "map_vid_vidorID.json")
vid_map = json.load(open(map_json))

# ### 6. 重新格式化问答数据
#    - **处理训练数据**：
#      - 对 `train` 列表中的每一个问答对进行处理，创建一个新的字典 `qa_dict`。
#      - 将 `video_id` 映射到新的ID，并在 `qid` 中包含 `type`、`video_id` 和 `qid`。
#      - 对每个选项 `a0` 到 `a4` 添加句号，并将问题 `question` 添加问号。
#      - 将处理后的数据存入 `new_train` 列表。
#    - **处理验证数据**：类似地处理 `val` 列表，将结果存入 `new_val` 列表。
new_train = []
new_val = []
for qa in train:
    qa_dict = {}
    qa_dict["video_id"] = vid_map[str(qa["video_id"])]
    qa_dict["num_option"] = 5
    qa_dict["qid"] = "_".join([qa["type"], str(qa["video_id"]), str(qa["qid"])])
    for i in range(5):
        qa_dict[f"a{str(i)}"] = qa[f"a{str(i)}"] + "."
    qa_dict["answer"] = qa["answer"]
    qa_dict["question"] = qa["question"] + "?"
    new_train.append(qa_dict)

for qa in val:
    qa_dict = {}
    qa_dict["video_id"] = vid_map[str(qa["video_id"])]
    qa_dict["num_option"] = 5
    qa_dict["qid"] = "_".join([qa["type"], str(qa["video_id"]), str(qa["qid"])])
    for i in range(5):
        qa_dict[f"a{str(i)}"] = qa[f"a{str(i)}"] + "."
    qa_dict["answer"] = qa["answer"]
    qa_dict["question"] = qa["question"] + "?"
    new_val.append(qa_dict)

#使用 `save_json` 函数将 `new_train` 和 `new_val` 分别保存为 `train.json` 和 `val.json` 文件。
save_json(new_train, os.path.join(path, "train.json"))
save_json(new_val, os.path.join(path, "val.json"))

# 这段代码的主要目的是将视频问答数据从原始的CSV格式转换为JSON格式，并根据映射文件重新组织和格式化数据，
# 以便后续在训练或评估过程中使用。处理过程中，特别关注了视频ID的映射、问题和答案的格式化。
