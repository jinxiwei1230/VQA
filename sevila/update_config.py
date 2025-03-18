# -*- coding: utf-8 -*-

import argparse
import re

import yaml
import os
#
# def add_id_placeholder(file_path):
#     # 读取 YAML 文件
#     try:
#         with open(file_path, 'r') as file:
#             config = yaml.safe_load(file)
#
#         # 递归查找并添加 `${id}` 占位符
#         def add_placeholder(data):
#             if isinstance(data, dict):
#                 for key, value in data.items():
#                     # 对于特定的字段，进行占位符替换
#                     if key in ['url', 'storage', 'output_dir'] and isinstance(value, str):
#                         # 检查路径中是否包含特定 ID 格式
#                         if re.search(r'/kafka_result/\d+/', value):
#                             # 将路径中的数字 ID 替换为 ${id} 占位符
#                             value = re.sub(r'(/kafka_result/)\d+/', r'\1${id}/', value)
#                             print(f"Updated {key}: {value}")  # 打印更新后的字符串
#                     # 继续递归处理
#                     data[key] = add_placeholder(value)
#             elif isinstance(data, list):  # 处理列表
#                 for index in range(len(data)):
#                     data[index] = add_placeholder(data[index])
#             elif isinstance(data, str):
#                 if "${id}" not in data:
#                     if 'output_dir' in data or 'finetuned' in data:  # 根据你的需要判断
#                         # 只在路径中包含数字 ID 的情况下替换
#                         if re.search(r'/kafka_result/\d+/', data):
#                             data = re.sub(r'(/kafka_result/)\d+/', r'\1${id}/', data)  # 替换
#                             print(f"Updated string: {data}")  # 打印更新后的字符串
#                         else:
#                             print(f"No update needed for: {data}")  # 没有更新的情况
#             return data  # 返回更新后的数据
#
#         config = add_placeholder(config)  # 确保 config 是更新后的内容
#
#         # 保存更新后的配置文件
#         with open(file_path, 'w') as file:
#             yaml.dump(config, file)
#             print("YAML file saved successfully.")  # 添加保存成功的提示
#
#     except yaml.YAMLError as e:
#         print(f"Error reading YAML file: {e}")
#     except Exception as e:
#         print(f"Error writing YAML file: {e}")

def replace_id_in_yaml(file_path, id_value):
    """
    替换 YAML 文件中路径中的数字为实际的 `id_value`。

    参数:
        file_path (str): YAML 文件路径。
        id_value (str): 替换路径中的占位符或数字的值。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # 读取 YAML 文件
    try:
        print("Reading YAML file...")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return

    # 递归函数，用于替换路径中的占位符或数字
    def replace_id(data, id_value):
        if isinstance(data, dict):  # 如果是字典
            for key, value in data.items():
                data[key] = replace_id(value, id_value)
        elif isinstance(data, list):  # 如果是列表
            for index in range(len(data)):
                data[index] = replace_id(data[index], id_value)
        elif isinstance(data, str):  # 如果是字符串
            # 匹配路径中的数字并替换为 id_value
            updated_data = re.sub(r'(/kafka_result/)\d+/', f'/kafka_result/{id_value}/', data)
            if updated_data != data:  # 如果发生替换，打印更新后的字符串
                print(f"Updated string: {updated_data}")
            return updated_data
        return data

    # 打印原始数据
    print("Original data: ", config)

    # 调用递归函数替换路径中的占位符或数字
    updated_config = replace_id(config, id_value)

    # 打印更新后的配置
    print("Updated config: ", updated_config)

    # 保存更新后的配置文件
    try:
        print("Writing updated YAML file...")
        with open(file_path, 'w') as file:
            yaml.dump(updated_config, file)
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        return

    print(f"Successfully updated {file_path} with id: {id_value}")
def main():
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&4&&&&&")
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='The ID to replace in the config file')
    parser.add_argument('--cfg-path', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--dataset-cfg-path', type=str, required=True, help='Path to the dataset YAML config file')

    args = parser.parse_args()

    # 首先添加占位符（如果必要）
    # add_id_placeholder(args.cfg_path)
    # add_id_placeholder(args.dataset_cfg_path)

    # 调用函数来替换主配置文件中的 ID
    replace_id_in_yaml(args.cfg_path, args.id)
    replace_id_in_yaml(args.dataset_cfg_path, args.id)

if __name__ == "__main__":
    main()
