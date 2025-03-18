#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
sys.path.append('/home/disk2/dachuang1-23/mist/')
print(sys.path)
# 检查是否提供了 id 参数
if len(sys.argv) < 2:
    print("Error: Missing 'id' argument")
    sys.exit(1)

# 获取传递的 id 参数
id = sys.argv[1]
print(f"Received id: {id}")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tqdm
import h5py
import torch
from torch.utils.data import DataLoader
from dataloader_video import VideoCLIPDataset

import math
import urllib.request
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# In[18]:


frame_num = 32
import glob
import os

# 构建 MP4 文件路径
mp4_path = f"/home/disk2/dachuang1-23/kafka_result/{id}/*.mp4"
# 查找 MP4 文件
mp4_files = glob.glob(mp4_path)

# 如果没有找到 MP4 文件，查找 AVI 文件
if not mp4_files:
    avi_path = f"/home/disk2/dachuang1-23/kafka_result/{id}/*.avi"
    avi_files = glob.glob(avi_path)
    files_to_use = avi_files
else:
    files_to_use = mp4_files

# 确保找到符合条件的文件
if not files_to_use:
    raise FileNotFoundError(f"没有找到符合条件的文件：{mp4_path} 或 {avi_path}")
print("files_to_use:")
print(files_to_use[0])
# 创建 VideoCLIPDataset 实例，加载找到的文件
dataset = VideoCLIPDataset(None, frame_num, files_to_use[0])


# dataset = VideoCLIPDataset(None, frame_num, f"/home/disk2/dachuang1-23/kafka_result/{id}/*.mp4")  # 加载视频文件
# dataset = VideoCLIPDataset(None, frame_num, "C:/Users/xiwei/Desktop/1/*.mp4")  # 加载视频文件

print(len(dataset))

dataloader = DataLoader(
    dataset,
    batch_size=1, # 每次加载 1 个样本
    num_workers = 2,
    shuffle=False
)
data_iter = iter(dataloader)

# 指定保存帧图像的目录
frames_dir = f'/home/disk2/dachuang1-23/kafka_result/{id}/frames'

# 确保目录存在，如果没有则创建
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)
global_index = 0
for batch in tqdm.tqdm(data_iter):
    batch_size = batch['video'].shape[0]
    print("batch_size:", batch_size)

    for i in range(batch_size):
        for j in range(frame_num):
            # 获取当前帧并进行处理
            # image = (batch['video'][i][j][0].permute(1, 2, 0).numpy() * 255.).round().astype(np.uint8)
            # 确保图像数据在合法范围内并转换为 uint8 类型
            # image = np.clip(image, 0, 255).astype(np.uint8)

            image = batch['video'][i][j][0].permute(1, 2, 0).numpy()
            image = (image * 255.).round().astype(np.uint8)
            # image = np.clip(image * 255., 0, 255).round().astype(np.uint8)

            # print(image.shape)  # 检查图像的形状，应该是 (H, W, C)  (224, 224, 3)

            # 增加亮度：通过简单的亮度调整
            # image = np.clip(image + 50, 0, 255)

            # 创建图形并显示当前帧图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            ax.axis('off')  # 不显示坐标轴

            # 保存当前帧图像
            frame_save_path = os.path.join(frames_dir, f'frame_{global_index + 1}.png')
            plt.savefig(frame_save_path, bbox_inches='tight', pad_inches=0)  # 使用tight布局，去除空白边缘

            # image_pil = Image.fromarray(image)
            # image_pil.save(frame_save_path)

            plt.close()  # 关闭当前图形，释放内存

            global_index += 1

# In[19]:

# Load CLIP model.
# 指定加载的 CLIP 模型类型
print("加载 CLIP 模型")
clip_model = "ViT-B/32" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device, jit=False)
print(" CLIP 模型加载完成！")

# ## Extract Feature

# In[20]:


dataset_feats = h5py.File(f"/home/disk2/dachuang1-23/kafka_result/{id}/clip_patch_feat_all.h5", "w")

dataset_feats.create_dataset("features", (len(dataset), 32, 17, 512))
dataset_feats.create_dataset("ids", (len(dataset), ), 'S20')
# dataset_feats.close()


# In[21]:


# 从数据加载器 dataloader 中提取每个视频帧的图像特征，并将这些特征和视频 ID 保存到之前创建的 HDF5 文件（dataset_feats）中
global_index = 0
video_ids = {}
data_iter = iter(dataloader)
for batch in tqdm.tqdm(data_iter):
    batch_size = batch['video'].shape[0]
    for i in range(batch_size):
        for j in range(frame_num):
            with torch.no_grad():  
                image_features = model.encode_image(batch['video'][i][j].cuda())
            dataset_feats['features'][global_index, j] = image_features.detach().cpu().numpy()
        dataset_feats['ids'][global_index] = batch['vid'][i].encode("ascii", "ignore")  
        global_index += 1

dataset_feats.close()



# In[23]:

# batch = data_iter.next()
# batch = next(data_iter)
try:
    batch = next(data_iter)
    # 处理数据
except StopIteration:
    print("No more data, stopping.")

# In[24]:

# print("加载 CLIP 模型2")
#
# # Load CLIP model.
# clip_model = "ViT-B/32" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load(clip_model, device=device, jit=False)
# print(" CLIP 模型加载完成！")


# In[30]:


# 图像预处理、特征提取、以及图像块的可视化
def load_image(raw_img, resize=None, pil=False):
    if isinstance(raw_img, str):
        image = Image.open(raw_img).convert("RGB")
    else:
        image = Image.fromarray(raw_img).convert("RGB")
    
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

# 用于可视化图像块（patches），并可以标注特定的块
def viz_patches(x, figsize=(20, 20), patch_idx=None, topk=None, t=5, save_path = ''):

    # x: num_patches, 3, patch_size, patch_size
    n = x.shape[0]
    nrows = int(math.sqrt(n))
    _, axes = plt.subplots(nrows, nrows, figsize=figsize)

    # 如果有子图的数量小于总补丁数量，调整布局
    if len(axes.flatten()) < n:
        axes = axes.flatten()
        extra_axes = len(axes.flatten()) - n
        for _ in range(extra_axes):
            axes[-1].axis('off')

    for i, ax in enumerate(axes.flatten()):            
        im = x[i].permute(1, 2, 0).numpy()
        im = (im * 255.).round().astype(np.uint8)
        if patch_idx is not None and i == patch_idx:
            im[0:t] = (255, 0, 0)
            im[im.shape[0]-t:] = (255, 0, 0)
            im[:, 0:t] = (255, 0, 0)
            im[:, im.shape[1]-t:] = (255, 0, 0)
        if topk is not None:
            if i in topk and i != patch_idx:
                im[0:t] = (255, 255, 0)
                im[im.shape[0]-t:] = (255, 255, 0)
                im[:, 0:t] = (255, 255, 0)
                im[:, im.shape[1]-t:] = (255, 255, 0)
        ax.imshow(im)
        ax.axis("off")
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)  # 保存图像到文件
    plt.close()  # 关闭当前图形

# 将图像切分成多个小块（patches），并返回这些小块。
def patchify(image_path, resolution, patch_size, patch_stride=None):
    img_tensor = transforms.ToTensor()(load_image(image_path, resolution, True))
    if patch_stride is None:
        patch_stride = patch_size
    patches = img_tensor.unfold(
        1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
    patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    return patches  # N, 3, patch_size, patch_size


# In[31]:

# 从图像中提取图像块（patches），并进行可视化
image_resolution = 224 * 4
patch_size = 224
image_index = 0
frame_indices = [7, 15, 23, 31]  # 要展示的帧索引
output_dir1 = f'/home/disk2/dachuang1-23/kafka_result/{id}/output_patches1'  # 保存路径
os.makedirs(output_dir1, exist_ok=True)

for frame_idx in frame_indices:
    # 获取对应帧
    one_frame = batch['raw_frame'][0][frame_idx].permute(1, 2, 0).numpy()

    # 分割图像块
    raw_patches = patchify(one_frame, image_resolution, patch_size, patch_stride=None)
    print(f"Frame {frame_idx} patches: {raw_patches.shape}")

    # 保存分割块的可视化
    save_path = os.path.join(output_dir1, f'output_patches1_frame_{frame_idx}.png')
    viz_patches(raw_patches, figsize=(8, 8), save_path=save_path)

    print(f"Saved patches visualization for frame {frame_idx} to {save_path}")

# In[27]:

# plt.imshow((batch['video'][0][image_index][0].permute(1, 2, 0).numpy() * 255.).round().astype(np.uint8))
# plt.show()
frame_indices = [7, 15, 23, 31]  # 要展示的帧索引
output_dir2 = f'/home/disk2/dachuang1-23/kafka_result/{id}/output_patches2'  # 保存路径
os.makedirs(output_dir2, exist_ok=True)

for frame_idx in frame_indices:
    # 提取对应帧的原始图像
    image = (batch['video'][0][frame_idx][0].permute(1, 2, 0).numpy() * 255.).round().astype(np.uint8)

    # 可视化并保存图像
    save_path = os.path.join(output_dir2, f'output_frame_{frame_idx}.png')
    plt.imshow(image)
    plt.axis('off')  # 去掉坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭图形
    print(f"Saved frame {frame_idx} visualization to {save_path}")


# In[15]:
#@title Detect
patches = batch['video'][0][image_index][1:]
frame_indices = [7, 15, 23, 31]
output_dir3 = f'/home/disk2/dachuang1-23/kafka_result/{id}/output_patches3'
os.makedirs(output_dir3, exist_ok=True)
clip_model = "ViT-B/32" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
image_caption = 'green bottle' #@param {type:"string"}
topk =  4#@param {type:"integer"}

for frame_idx in frame_indices:
    patches = batch['video'][0][frame_idx][1:]  # 获取帧的图像块
    # 文本输入编码
    text_input = clip.tokenize([image_caption]).to(device)
    patches_pad = patches.to(device)

    with torch.no_grad():
        # 提取图像块和文本的嵌入特征
        patch_embs = model.encode_image(patches_pad)
        text_embs, _ = model.encode_text(text_input)

        # 归一化特征向量
        patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # 计算相似度
        sim = patch_embs @ text_embs.t()
        idx_max = sim.argmax().item()
        topk_idxs = torch.topk(sim.flatten(), topk)[-1].cpu().numpy().tolist()

    # 保存和展示图像块
    save_path = os.path.join(output_dir3, f'output_patches_frame_{frame_idx}.png')
    viz_patches(
        patches,
        figsize=(10, 10),
        patch_idx=idx_max,
        topk=topk_idxs,
        t=int(0.05 * patch_size),
        save_path=save_path
    )
    print(f"Saved patch visualization for frame {frame_idx} to {save_path}")

# In[13]:

#@title Detect
frame_indices = [7, 15, 23, 31]
output_dir4 = f'/home/disk2/dachuang1-23/kafka_result/{id}/output_patches4'  # 保存路径
os.makedirs(output_dir4, exist_ok=True)  # 创建保存目录（如果不存在）

# patches = patchify(one_frame, image_resolution, patch_size, patch_stride=None)

# image_caption = 'hand' #@param {type:"string"}
topk =  4#@param {type:"integer"}
text_input = clip.tokenize([image_caption]).to(device)
# patches_pad = patches.to(device)

for frame_idx in frame_indices:
    # one_frame = batch['video'][0][frame_idx].permute(1, 2, 0).numpy()  # 提取帧
    # 检查帧的形状
    frame = batch['video'][0][frame_idx][0]  # 提取当前帧

    # 如果多一维，移除第一维
    if frame.dim() == 4 and frame.shape[0] == 1:
        frame = frame.squeeze(0)

    # permute 到 (H, W, C)
    one_frame = frame.permute(1, 2, 0).numpy()  # 调整维度顺序
    one_frame = (one_frame * 255).astype(np.uint8)

    raw_patches = patchify(one_frame, image_resolution, patch_size, patch_stride=None)  # 提取图像块
    patches_pad = raw_patches.to(device)  # 转移到设备

    with torch.no_grad():
        # 提取图像块和文本的嵌入特征
        patch_embs = model.encode_image(patches_pad)
        text_embs, _ = model.encode_text(text_input)

        # 归一化特征向量
        patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # 计算相似度
        sim = patch_embs @ text_embs.t()
        idx_max = sim.argmax().item()  # 找到最匹配的补丁索引
        topk_idxs = torch.topk(sim.flatten(), topk)[-1].cpu().numpy().tolist()  # 找到最匹配的 topk 补丁

    # 保存图像块可视化结果
    save_path = os.path.join(output_dir4, f'output_patches_frame_{frame_idx}.png')
    viz_patches(
        raw_patches,
        figsize=(10, 10),
        patch_idx=idx_max,
        topk=topk_idxs,
        t=int(0.05 * patch_size),
        save_path=save_path
    )
    print(f"Saved patch visualization for frame {frame_idx} to {save_path}")

# In[ ]:




