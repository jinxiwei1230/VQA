"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from omegaconf import OmegaConf
# BaseProcessor 是一个设计用于数据处理的基类，提供了一个简单的框架。它允许通过继承和配置来扩展功能。
# 具体的处理逻辑（即如何变换输入数据）可以通过子类化和覆盖 self.transform 来实现。
# from_config 和 build 方法则提供了从配置文件或参数创建处理器实例的功能。

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)
