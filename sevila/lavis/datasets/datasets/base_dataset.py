"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import pandas as pd

from typing import Iterable
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

# 提供了基本的数据集加载和处理功能，适用于所有自定义的数据集。它负责处理数据集的注释、样本数量、数据处理器设置等基础功能。
class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []

        for ann_path in ann_paths:
            if '.json' in ann_path:
                self.annotation.extend(json.load(open(ann_path, "r")))
                if 'train' in ann_path: 
                    self.data_type = 'train'
                else:
                    self.data_type = 'val'
            else:
                raise AttributeError('Undefined data type')
            
        #self.annotation = self.annotation[:100] 
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation): 
            if isinstance(ann, str):
                pass
            else:
                ann[key] = str(idx)

 # 允许将多个数据集（例如不同的数据源或不同的数据拆分）合并为一个数据集，方便在训练或评估时使用。这对于处理大型数据集或进行跨数据集训练时非常有用。
class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
