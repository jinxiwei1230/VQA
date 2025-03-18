"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
# os.environ["TORCH_HOME"] = "/root/autodl-tmp/.cache/torch"
# os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/.cache/huggingface/hub"
from pathlib import Path
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
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


# 用户输入的参数
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
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(cfg):
    seed = cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False  # 将 CuDNN 的性能优化模式关闭，在某些情况下可以提高稳定性
    cudnn.deterministic = True  # 将 CuDNN 设置为确定性模式，将使用确定性的算法和参数初始化


def setup_output_dir(cfg):
    lib_root = Path(registry.get_path("library_root"))

    output_dir = lib_root / cfg.output_dir  # / self.job_id
    result_dir = output_dir / "result"

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # registry.register_path("result_dir", str(result_dir))
    # registry.register_path("output_dir", str(output_dir))

    registry.force_register_path("result_dir", str(result_dir))
    registry.force_register_path("output_dir", str(output_dir))


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()  # 年月日时分构成

    # 根据配置文件和用户输入的参数合并为最终配置
    cfg = Config(parse_args())

    # 分布式模式设置
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg.run_cfg)

    # set after init_distributed_mode() to only log on master.
    setup_output_dir(cfg.run_cfg)
    setup_logger()

    # 打印run,datasets,model配置
    cfg.pretty_print()

    # 构建一个任务实例，其中设置了一些用于该任务的初始配置
    task = tasks.setup_task(cfg)
    # 调用该任务的方法构建数据集和模型
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner_cls = get_runner_class(cfg.run_cfg)
    runner = runner_cls(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
