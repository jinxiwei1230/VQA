#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: dataloader_vatexEnglish_frame.py
# @Version: version 1.0

import os
from torch.utils.data import Dataset
import numpy as np
import json
import random
import cv2
from PIL import Image
from torchvision import transforms
import lmdb
g_lmdb_frames = 30


class VATEX_multi_sentence_dataLoader(Dataset):
    def __init__(
            self,
            root,  # LMDB 数据存储路径
            language,  # 指定数据集语言（"chinese" 或 "english"）
            subset,  # 数据子集，可能的取值有 pretrain、train、val、test
            data_path,  # 数据集存放的根目录路径，包含视频资源
            tokenizer,  # 文本分词器，用于将字幕转化为模型可接受的格式
            frame_sample,  # 选择如何从视频中抽取帧（例如 uniform_random，random，uniform）
            max_words=32,  # 文本中最多的单词数
            feature_framerate=1.0,  #
            max_frames=12,  # 视频中最多的帧数
            image_resolution=224,  # 视频帧的图像分辨率
            id=id,

    ):
        # 初始化操作
        self._env = None
        self._txn = None
        self.root = root
        self.subset = subset
        self.data_path = data_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.language = language
        self.frame_sample = frame_sample
        self.resolution = image_resolution
        self.id = id
        # load the id of split list
        # 读取视频ID文件，例如train_four.txt或test.txt，从而确定当前数据集中的视频。
        assert self.subset in ["pretrain", "train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["pretrain"] = os.path.join(self.data_path, "train_four.txt")
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_four.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "test.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test.txt")  # 1-1476

        # construct ids for data loader
        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]  # video_ids 是一个包含所有视频 ID 的列表
        # print("video_ids:",video_ids)

        # load caption
        # 加载vatex_data.json文件，这个文件包含了每个视频的字幕
        caption_file = f"/home/disk2/dachuang1-23/text/kafka_result/{id}/vatex_data.json"
        # caption_file = os.path.join(self.data_path, "vatex_data.json")
        print("caption_file:", caption_file)
        captions = json.load(open(caption_file, 'r'))

        # construct pairs
        # 构造sentences_dict，每个视频对应多个字幕对（多句字幕支持），并统计视频和字幕的数量
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []  # used to tag the label when calculate the metric
        if self.language == "chinese":
            cap = "chCap"
        else:
            cap = "enCap"
        for video_id in video_ids:
            # print(video_id)
            assert video_id in captions
            for cap_txt in captions[video_id][cap]:
                # print(cap_txt)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            # self.sentences_dict[len(self.sentences_dict)] = (video_id, captions['1'][cap])
            self.cut_off_points.append(len(self.sentences_dict))

        # usd for multi-sentence retrieval
        self.multi_sentence_per_video = True # important tag for eval in multi-sentence retrieval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict) # used to cut the sentence representation
            self.video_num = len(video_ids) # used to cut the video representation
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(video_ids)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        # length of dataloader for one epoch
        self.sample_len = len(self.sentences_dict)

        # start and end token
        # 设置了一些特殊的符号（如CLS_TOKEN, SEP_TOKEN）用于文本处理。
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        # 通过transforms.Compose设置图像处理流程（调整大小、裁剪、转换为张量、归一化）。
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 80, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=1, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def __len__(self):
        """length of data loader

        Returns:
            length: length of data loader
        """
        length = self.sample_len
        return length

    def _get_text(self, caption):
        """get tokenized word feature

        Args:
            caption: caption

        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text

        """
        # tokenize word
        words = self.tokenizer.tokenize(caption)  # 分词，将字幕文本转换为词语列表。

        # 在词语列表前后加上特殊符号（ < | startoftext | >, < | endoftext | >）
        # add cls token
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token;
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # 将词语转化为词ID，并对其进行填充，确保每个文本的长度都为max_words
        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)
        # 返回处理后的文本ID、掩码（mask）和段ID（segment_ids）作为模型的输入
        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, frames):
        video_list = list()
        global g_lmdb_frames
        # global writer
        # random sample start #################################################
        # 根据 frame_sample 策略（均匀抽样、随机抽样等）选取帧
        if self.frame_sample == "uniform_random":
            # assert g_lmdb_frames % frames == 0
            video_index = list(np.arange(0, g_lmdb_frames))
            # print("video_index:{}".format(video_index))
            sample_slice = list()
            k = g_lmdb_frames // frames
            for i in np.arange(frames):
                index = random.sample(video_index[k * i:k * (i + 1)], 1)
                sample_slice.append(index[0])
        elif self.frame_sample == "random":
            # sample
            video_index = list(np.arange(0, g_lmdb_frames))
            sample_slice = random.sample(video_index, frames)
            sample_slice = sorted(sample_slice)
        else:
            sample_slice = np.linspace(0, g_lmdb_frames, frames, endpoint=False, dtype=int)
            # random sample end ##################################################

        # 从LMDB数据库中读取每一帧的数据，并通过OpenCV解码为图像
        for step, i in enumerate(sample_slice):
            video_key_new = video_key + "_%d" % i
            video_key_new = video_key_new.encode()
            video = self._txn.get(video_key_new)
            frame_buffer = np.frombuffer(video, dtype=np.uint8)
            # print("frame_buffer.shape:{}".format(frame_buffer.shape))
            frame_data = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            # print("frame_data.shape:{}".format(frame_data.shape))
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # print("[{}]frame_rgb.shape:{}".format(step, frame_rgb.shape))
            frame_img = Image.fromarray(frame_rgb).convert("RGB")
            # print("frame_img.shape:{}".format(np.array(frame_img).shape))
            # writer.add_image('original_img', frame_rgb, global_step=step, dataformats='HWC')
            frame_data = self.transform(frame_img)
            # print("[{}]frame_data.shape:{}".format(step, frame_data.shape))
            video_list.append(frame_data)
        video_data = np.stack(video_list)
        video_data = video_data.copy()
        # video_data = video_data.astype('float64')

        # 将每帧图像转换为RGB格式，并应用self.transform对图像进行预处理（包括尺寸调整、裁剪、转换为张量、归一化等）。
        video_data = video_data.reshape([self.max_frames, 3, self.resolution, self.resolution])

        return video_data


    # 数据加载的核心方法，返回给定索引（idx）的样本。它的作用是通过索引获取视频的特定数据（字幕和帧），并返回格式化的数据以供训练。
    def __getitem__(self, idx):
        """forward method
        Args:
            idx: id
        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text
            video: sampled frames
            video_mask: mask of sampled frames
        """
        if self._env is None:
            self._initEnv()
        #print(f"Sentences dictionary: {self.sentences_dict}")

        # 1.通过idx获取视频ID和对应的字幕。
        video_id, caption = self.sentences_dict[idx]
        # print(f"Caption value: {caption}, Type: {type(caption)}")
        # print(f"Video ID: {video_id}, Caption: {caption}")

        # obtain text data
        # 2.通过self._get_text(caption)获取文本数据。_get_text方法会：
        # 对字幕进行分词。
        # 添加开始（CLS_TOKEN）和结束（SEP_TOKEN）标记。
        # 将分词后的文本转化为ID，并将文本长度填充至max_words。
        # 返回处理后的文本ID、掩码（mask）和段ID（segment_ids）作为模型的输入
        pairs_text, pairs_mask, pairs_segment = self._get_text(caption)

        #obtain video data
        # 3.通过self._get_video(video_id, self.max_frames)获取视频数据。_get_video方法会：
        # 根据frame_sample设置从视频中选择帧
        # 从LMDB数据库中读取帧数据，并使用OpenCV和PIL对每一帧图像进行处理（转换为RGB格式、调整大小、归一化）。
        # 最终将选取的帧组成一个张量，形状为[max_frames, 3, resolution, resolution]，即每一帧的图像数据。
        video = self._get_video(video_id, self.max_frames)

        if self.subset == "pretrain":
            return video, self.max_frames, pairs_text, pairs_mask, pairs_text, pairs_mask
        elif self.subset == "train":
            return pairs_text, pairs_mask, video, self.max_frames, idx
        else:
            return pairs_text, pairs_mask, video, self.max_frames
