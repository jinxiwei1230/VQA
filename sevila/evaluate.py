# -*- coding: utf-8 -*-

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from train import setup_output_dir
import pdb


def parse_args():

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    print("************************************************")
    print(args)
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):

    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# def main():
#     # allow auto-dl completes on main process without timeout when using NCCL backend.
#     # os.environ["NCCL_BLOCKING_WAIT"] = "1"
#
#     # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
#     job_id = now()  # 通过 now() 函数获取当前时间，生成一个作业ID。
#
#     #pdb.set_trace()
#
#     cfg = Config(parse_args())  # 调用 Config(parse_args()) 加载配置文件。
#
#     setup_output_dir(cfg.run_cfg)  # 确保输出路径被设置和注册
#
#     init_distributed_mode(cfg.run_cfg)  # 初始化分布式模式
#
#     setup_seeds(cfg)  # 调用 setup_seeds(cfg) 以确保实验的可重复性。
#
#     # set after init_distributed_mode() to only log on master.
#     setup_logger()
#
#     print("############################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#     cfg.pretty_print()
#
#     print("###########################################################")
#     task = tasks.setup_task(cfg)
#     datasets = task.build_datasets(cfg)
#     model = task.build_model(cfg)
#     print("###########################################################")
#
#     runner = RunnerBase(
#         cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
#     )
#     runner.evaluate(skip_reload=True)


def main(cfg, model, datasets, task, output_dir):
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()  # 通过 now() 函数获取当前时间，生成一个作业ID。

    #pdb.set_trace()

    # cfg = Config(parse_args())  # 调用 Config(parse_args()) 加载配置文件。

    setup_output_dir(cfg.run_cfg)  # 确保输出路径被设置和注册

    init_distributed_mode(cfg.run_cfg)  # 初始化分布式模式

    setup_seeds(cfg)  # 调用 setup_seeds(cfg) 以确保实验的可重复性。

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    print("############################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    cfg.pretty_print()

    print("###########################################################")
    # task = tasks.setup_task(cfg)
    # datasets = task.build_datasets(cfg)
    # model = task.build_model(cfg)
    print("###########################################################")

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(output_dir, skip_reload=True)



# if __name__ == "__main__":
#     main()
