"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import torch

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )

# 处理 AOKVQA 训练集或验证集的图像和问题对。
# 读取图像文件，将其转换为 RGB 格式。
# 使用视觉处理器和文本处理器分别处理图像和问题。
# 计算问题的答案及其权重。
# 返回一个字典，其中包括：
# 处理后的图像。
# 处理后的问题文本。
# 问题的答案和权重。
class AOKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }

# 处理 AOKVQA 的测试或评估集，用于评估模型在视觉问答任务中的表现。
# 读取并解析包含图像路径、问题、选项、正确答案等的标注文件。
# 使用视觉和文本处理器处理图像和问题。
# 提供批量数据的收集和整理功能（collater 方法），将多个样本打包成批次以供训练或评估使用。
# 返回数据中包含图像、问题 ID、选项、正确答案、直接答案等信息。
class AOKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    # 将多个样本打包成一个批次，便于后续训练或评估。
    def collater(self, samples):
        (
            image_list,
            question_list,
            question_id_list,
            instance_id_list,
            choices_list,
            correct_choice_idx_list,
            direct_answers_list,
        ) = ([], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            choices_list.append(sample["choices"])
            correct_choice_idx_list.append(sample["correct_choice_idx"])
            direct_answers_list.append(sample["direct_answers"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "question_id": question_id_list,
            "instance_id": instance_id_list,
            "choices": choices_list,
            "correct_choice_idx": correct_choice_idx_list,
            "direct_answers": direct_answers_list,
        }


    # 获取数据集中指定索引的样本，返回图像、文本、问题ID、选项、正确答案和直接答案等。
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choices = ann["choices"]
        if "correct_choice_idx" in ann:
            correct_choice_idx = ann["correct_choice_idx"]
        else:
            correct_choice_idx = None

        if "direct_answers" in ann:
            direct_answers = ann["direct_answers"]
        else:
            direct_answers = None

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "choices": choices,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": direct_answers,
        }
