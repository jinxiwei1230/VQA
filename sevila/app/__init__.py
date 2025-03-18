#版权声明和许可
"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from PIL import Image #`PIL`：用于处理图像的库，这里特别使用了 `Image` 模块。
import requests       # requests`：用于发送HTTP请求以获取图像数据。

import streamlit as st# 用于构建和共享数据应用的库。
import torch

#@st.cache()`：是一个Streamlit的装饰器，用于缓存函数的输出结果，从而提高应用的性能。
# 这里用于缓存加载的图片，以避免每次重新加载。
@st.cache()
def load_demo_image():
    #这个函数通过URL从网络加载一张图片，并使用 `PIL.Image.open` 打开，然后将其转换为RGB模式的图像。
    # 返回的 `raw_image` 是一个PIL的图像对象。
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image

#用于选择在CPU或GPU上执行张量运算。如果CUDA（NVIDIA的并行计算架构）可用，则选择GPU，否则选择CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义了一个缓存根目录
cache_root = "/export/home/.cache/lavis/"

# 这段代码是一个使用了Python库 `Streamlit` 和 `PIL` (Python Imaging Library) 来加载并展示一张图片的脚本。
#它还涉及了一些设备配置的操作，用于选择计算设备（如CPU或GPU）。
# 这段代码展示了如何使用Streamlit缓存功能来高效地加载和展示图像，同时还设置了PyTorch的设备配置来支持深度学习的计算。