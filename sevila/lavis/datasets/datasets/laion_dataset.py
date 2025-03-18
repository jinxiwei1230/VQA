"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import webdataset as wds
from lavis.datasets.datasets.base_dataset import BaseDataset

# 主要用于加载和处理 LAION 数据集中的图像-文本对。这是一个基于 WebDataset 的实现，适用于处理大型分布式数据集，如 LAION，它由多个 .tar 文件存储的样本组成。
class LaionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


if __name__ == "__main__":
    from torchvision import transforms

    def to_image_text_pair(sample):
        return sample[0], sample[1]["caption"]

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = LaionDataset(
        vis_processor=transform_train,
        text_processor=lambda x: x,
        location="/export/laion/laion2B-multi/part-00000/{00000..01743}.tar",
    )

    import torch

    loader = torch.utils.data.DataLoader(dataset.inner_dataset, batch_size=2)

    print(next(iter(loader))["text_input"])
