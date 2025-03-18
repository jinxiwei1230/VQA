"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import os

import numpy as np
import yaml
# 用于处理图像和视频数据的处理器（processor）模块，属于一个更大的视觉问答或视觉相关任务的系统。
# 该模块主要定义了一系列的处理器类，用于对图像和视频数据进行预处理，以便将其输入到模型中。
import re

import cv2
import torch
from lavis.processors import transforms_video
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.datasets.data_utils import load_video
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

MAX_INT = registry.get("MAX_INT")

class ToUint8(object): # 将张量的数据类型转换为 torch.uint8
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object): # 将张量从 (C, T, H, W) 格式转换为 (T, H, W, C) 格式。
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__
    
class BlipImageBaseProcessor(BaseProcessor): # 用于图像数据的基础处理器，包含图像归一化的功能。
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class BlipVideoBaseProcessor(BaseProcessor): # 用于视频数据的基础处理器，包含视频帧归一化的功能。

    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms

@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor): # 用于处理图像的文字描述，清理文本并限制最大单词数。
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor): # 用于处理问题文本，清理文本并限制最大单词数。
    def __init__(self, max_words=50, output_dir=None):
        self.max_words = max_words
        self.output_dir = output_dir

    def __call__(self, question):
        print("question:", question)
        processed_question = self.pre_question(question)
        print("processed_question:", processed_question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)
        output_dir = cfg.get("output_dir", None)

        return cls(max_words=max_words, output_dir=output_dir)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor): # 用于图像训练的数据处理，包括随机裁剪、水平翻转和数据增强。
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor): # 用于图像评估的数据处理，主要进行调整大小和归一化处理。
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


# 用于图像训练的数据处理，与 BlipImageTrainProcessor 类似，但具有不同的图像大小。
@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

# 用于视频训练的数据处理，随机提取视频帧，并进行图像预处理和数据增强。
@registry.register_processor("blip2_video_train")
class Blip2VideoTrainProcessor(BlipVideoBaseProcessor):
    def __init__(
        self, 
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath, clip_proposal=None):

        clip, indices, fps = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="random",
            clip_proposal=clip_proposal
        )
        return self.transform(clip), indices, fps


    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms
        )


# 用于视频评估的数据处理，均匀提取视频帧，确保一致性。
@registry.register_processor("blip_video_eval")
class BlipVideoEvalProcessor(BlipVideoBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None, n_frms=MAX_INT, output_dir=None):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.output_dir = output_dir  # 添加输出路径参数
        self.transform = transforms.Compose(
            [
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )
        self.n_frms = n_frms
        print("---------------------------self.output_dir-------------------------")
        print(self.output_dir)

    def __call__(self, vpath, clip_proposal=None, video_id=None):

        if video_id is None:
            video_id = os.path.basename(vpath).split('.')[0]

        clip, indices, fps = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="uniform",
            clip_proposal=clip_proposal
        )

        # 如果提供了输出目录，则保存视频帧
        if self.output_dir and video_id:
            data_dir = os.path.join(self.output_dir, "data")
            self.save_frames(clip, data_dir, video_id)
        else:
            print(video_id, "error!")
            print(self.output_dir)

        return self.transform(clip), indices, fps

    def save_frames(self, frames, data_dir, video_id):
        # 创建视频帧保存目录
        video_dir = os.path.join(data_dir, video_id)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        print(f"Saving frames to: {video_dir}")

        # 将 Tensor 转换为 numpy 数组并调整维度 (C, T, H, W) -> (T, H, W, C)
        frames_np = frames.permute(1, 2, 3, 0).cpu().numpy()

        for i, frame in enumerate(frames_np):
            # 将 RGB 转换为 BGR 以避免 OpenCV 保存时颜色不正确
            frame_bgr = frame[:, :, [2, 1, 0]]  # RGB -> BGR

            # 定义保存路径
            frame_path = os.path.join(str(video_dir), f"{video_id}_frm_{i}.png")

            # 保存帧并检查是否成功
            success = cv2.imwrite(frame_path, frame_bgr)
            if not success:
                print(f"错误: 无法保存帧 {i} 到 {frame_path}")
            else:
                print(f"帧 {i} 成功保存到 {frame_path}")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        output_dir = cfg.get("output_dir", None)
        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, output_dir=output_dir, n_frms=n_frms)

    # Blip2VideoTrainProcessor: 随机提取帧，用于训练，增加数据多样性。
    # BlipVideoEvalProcessor: 均匀提取帧，用于评估，确保一致性。
