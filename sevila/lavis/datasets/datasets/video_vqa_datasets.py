"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)

# 数据加载与处理
# VideoQADataset 初始化时，加载并处理视频和对应的问答注释。视频的路径在 vis_root 中定义，问答注释在 ann_paths 中指定。
# 它通过 vis_processor 来处理视频，并通过 text_processor 来处理文本问题。
# 类标签构建 (_build_class_labels 方法)
# 该方法从一个 JSON 文件中加载 ans2label，为每个答案分配一个类别标签，目的是将答案映射为分类任务中的类别标签（即，答案分类）。
# 这个类标签字典将答案与特定的分类标签对应。
# 获取答案标签 (_get_answer_label 方法)
# 根据问题的答案，返回相应的类别标签。如果答案在定义的 class_labels 字典中，则返回对应的标签，否则返回一个默认标签（类标签字典的长度，表示新的类别）。
class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        print("VideoQADataset")
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        print("video_vqa_datasets")
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
