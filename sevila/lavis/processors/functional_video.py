"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings

import torch
# 上面的文件是接口定义层，通过类对外提供调用接口。
# 这个文件是具体实现层，实现了这些接口背后的细节操作逻辑。
# 视频处理的工具函数

# 检查输入是否为4D的PyTorch张量（tensor），即视频片段的格式，视频张量的尺寸通常为 (C, T, H, W)
def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

# 将输入的视频片段按照给定的起始点 (i, j) 和尺寸 (h, w) 进行裁剪，返回一个经过裁剪的子区域。
def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]

# 根据目标尺寸 target_size 和插值模式 interpolation_mode 对视频片段进行缩放。
def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )

# 这个函数结合了 crop 和 resize 两步操作，将视频裁剪为指定大小，并调整为目标尺寸。
def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip

# 对视频片段进行中心裁剪。给定裁剪的尺寸 crop_size，在视频片段的中心位置进行裁剪。
def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


# 将视频片段的张量数据类型从 uint8 转换为 float，并将值归一化到 [0, 1] 之间。
# 同时将视频的维度顺序从 (T, H, W, C) 变为 (C, T, H, W)，这是大多数模型需要的输入格式。
def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "clip tensor should have data type uint8. Got %s" % str(clip.dtype)
        )
    return clip.float().permute(3, 0, 1, 2) / 255.0


# 对视频片段进行归一化处理，将每个像素值减去均值 mean 并除以标准差 std，将输入数据调整到统一的尺度。
def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

# 对视频片段进行水平翻转（水平镜像），即将每一帧从左到右反转。
def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (C, T, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)
