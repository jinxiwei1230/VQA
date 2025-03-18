import lavis.tasks as tasks
import argparse
import json
from lavis.common.config import Config
from argparse import Namespace


# def parse_args(cfg_path, options):
#     """
#     解析传入的配置路径和选项
#     """
#
#     # 为了适应传递的参数，可以创建一个模拟的 args
#     class Args:
#         def __init__(self, cfg_path, options):
#             self.cfg_path = cfg_path
#             self.options = options
#
#     return Args(cfg_path, options)



def parse_args(cfg_path, options):
    # 解析 cfg_path 和 options，创建 argparse.Namespace
    args = argparse.Namespace(cfg_path=cfg_path, options=options)
    return args


def load_model(cfg):
    """
    根据配置加载模型、数据集和任务
    """
    task = tasks.setup_task(cfg)  # 设置任务
    # datasets = task.build_datasets(cfg)  # 构建数据集
    model = task.build_model(cfg)  # 构建模型
    return model  #, datasets, task

def load_datasets_tasks(cfg):
    """
    根据配置加载模型、数据集和任务
    """
    task = tasks.setup_task(cfg)  # 设置任务
    datasets = task.build_datasets(cfg)  # 构建数据集
    return datasets, task



def load_model_from_config(cfg_path, options):
    # 使用 argparse.Namespace 创建类似目标的 Namespace 对象
    args = argparse.Namespace(
        cfg_path=cfg_path,
        options=options
    )

    # 打印 args，确认转换是否正确
    print("0000000000000000000000000000")
    print(args)

    # 使用字典作为 args 创建 Config 实例
    cfg = Config(args)
    print(type(cfg))
    print(vars(cfg))

    # 加载模型、数据集和任务
    # model, datasets, task = load_model(cfg)
    model = load_model(cfg)

    # 返回模型、数据集和任务
    return cfg, model


def load_cfg_from_config(cfg_path, options):
    # 使用 argparse.Namespace 创建类似目标的 Namespace 对象
    args = argparse.Namespace(
        cfg_path=cfg_path,
        options=options
    )

    # 打印 args，确认转换是否正确
    print("8888888888888888888888")
    print(args)

    # 使用字典作为 args 创建 Config 实例
    cfg = Config(args)
    print(type(cfg))

    # 返回模型、数据集和任务
    return cfg

# import lavis.tasks as tasks
# import argparse
# from lavis.common.config import Config
#
# def parse_args():
#     # 解析命令行参数，获取配置文件路径
#     parser = argparse.ArgumentParser(description="Training")
#
#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#
#     args = parser.parse_args()
#     return args
#
# def load_model(cfg):
#     """
#     根据配置加载模型、数据集和任务
#     """
#     task = tasks.setup_task(cfg)  # 设置任务
#     datasets = task.build_datasets(cfg)  # 构建数据集
#     model = task.build_model(cfg)  # 构建模型
#     return model, datasets, task
#
# def main():
#     # 1. 解析命令行参数，获取配置路径
#     # args = parse_args()
#     # 2. 使用配置路径加载配置文件
#     # cfg = Config(args.cfg_path)
#     # 3. 加载模型、数据集和任务
#     # model, datasets, task = load_model(cfg)
#
#     cfg = Config(parse_args())
#     model, datasets, task = load_model(cfg)
#
#     # 这里可以继续执行后续任务，例如训练、评估等
#     # 示例：打印模型和数据集信息
#     print("Model:", model)
#     print("Datasets:", datasets)
#
# if __name__ == "__main__":
#     main()
