"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from lavis.common.registry import registry

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.tasks import *
# 设置和注册一些全局配置和路径

root_dir = os.path.dirname(os.path.abspath(__file__)) # 当前文件的目录路径 lavis/
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..") # 本仓库路径
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root) # 预训练模型存储路径 ~/.cache/
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)   # 系统支持的最大整数值
registry.register("SPLIT_NAMES", ["train", "val", "test"])
