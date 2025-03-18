"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch

from lavis.datasets.datasets.base_dataset import BaseDataset

# 图像处理: 将图像数据堆叠成一个张量（torch.stack），以便批量处理。
# 问题处理: 收集所有问题文本。
# 答案和权重处理: 收集所有答案及其权重，并将权重转换为张量。
# 答案数量: 计算每个样本中的答案数量，并将其转换为张量。
class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        print("VQADataset")
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))
        # collater 方法返回一个字典，其中包含：
        # image: 处理后的图像数据（张量）
        # text_input: 问题文本
        # answer: 所有答案
        # weight: 答案的权重（张量）
        # n_answers: 每个样本的答案数量（张量）
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        print("VQAEvalDataset")
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
