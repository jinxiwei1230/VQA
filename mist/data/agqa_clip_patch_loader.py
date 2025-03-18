import os
import sys
import json

import clip

sys.path.insert(0, '../')
from util import tokenize, transform_bb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
# from tools.object_align import align
import os.path as osp
import h5py
import random as rd
import numpy as np


# from IPython.core.debugger import Pdb
# dbg = Pdb()
# dbg.set_trace()


class VideoQADataset(Dataset):
    def __init__(
            self,
            # data_dir='/home/disk2/dachuang1-23/data/datasets/agqa/',
            data_dir=f'/home/disk2/dachuang1-23/kafka_result/',
            # data_dir=f'/home/disk2/dachuang1-23/kafka_result/',
            split='train',
            # feature_dir='/home/disk2/dachuang1-23/data/feats/agqa/',
            # feature_dir=f'/home/disk2/dachuang1-23/kafka_result/399/',
            feature_dir=f'/home/disk2/dachuang1-23/kafka_result/',
            qmax_words=20,
            amax_words=5,
            bert_tokenizer=None,
            a2id=None,
            id=id
    ):
        data_dir = os.path.dirname(data_dir)
        # 打印输入参数

        print(f"Initializing VideoQADataset for {split}")
        print(f"Data directory: {data_dir}")
        print(f"Feature directory: {feature_dir}")
        print(f"Split: {split}")

        # file_path = osp.join(data_dir, f'agqa_{split}_v2.jsonl')
        file_path = osp.join(data_dir, f'output.jsonl')

        print(f"File path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        # self.data = pd.read_json(osp.join(data_dir, f'agqa_{split}_v2.jsonl'), lines=True)
        self.data = pd.read_json(osp.join(data_dir, f'output.jsonl'), lines=True)
        print(f"Loaded {len(self.data)} samples for {split}")
        print(json.load(open(osp.join(data_dir, 'agqa_frame_size.json'))))
        self.frame_size = json.load(open(osp.join(data_dir, 'agqa_frame_size.json')))
        self.dset = 'agqa'
        self.video_feature_path = feature_dir
        self.bbox_num = 16
        self.use_frame = True
        self.use_mot = False
        self.qmax_words = qmax_words
        # vocab.json 路径已修改为固定路径
        vocab_path = '/home/disk2/dachuang1-23/data/datasets/agqa/vocab.json'
        self.a2id = json.load(open(vocab_path))
        # self.a2id = json.load(open(osp.join(data_dir, 'vocab.json')))
        self.bert_tokenizer = bert_tokenizer

        # answers_path = '/home/disk2/dachuang1-23/data/datasets/agqa/answers.json'
        # self.candidate_answer = json.load(open(osp.join(answers_path)))

        # bbox_feat_file = osp.join(self.video_feature_path, f'region_feat_n/faster_rcnn_32f20b.h5')
        # print('Load {}...'.format(bbox_feat_file))

        app_feat_file = osp.join(self.video_feature_path, f'clip_patch_feat_all.h5')
        # app_feat_file = osp.join('/home/disk2/dachuang1-23/data/feats/agqa/frame_feat', f'clip_patch_feat_all.h5')

        # print("self.video_feature_path:", self.video_feature_path)
        # app_feat_file = osp.join(self.video_feature_path, f'clip_patch_feat_all.h5')

        print('Load {}...'.format(app_feat_file))
        encoding = 'utf-8'
        self.frame_feats = {}
        with h5py.File(app_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['features']
            print(feats.shape)  # v_num, clip_num, feat_dim
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                vid = vid.decode(encoding)
                self.frame_feats[vid] = feat

    def __len__(self):
        return len(self.data)

    def get_video_feature(self, raw_vid_id, width=1, height=1):
        patch_bbox = []
        patch_size = 224
        grid_num = 4
        width, height = patch_size * grid_num, patch_size * grid_num

        for j in range(grid_num):
            for i in range(grid_num):
                patch_bbox.append([i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size])

        roi_bbox = np.tile(np.array(patch_bbox), (32, 1)).reshape(32, 16, -1)
        bbox_feat = transform_bb(roi_bbox, width, height)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        # roi_feat = self.frame_feats[:, 1:, :]
        print("type(self.frame_feats)")
        key = next(iter(self.frame_feats))  # 获取第一个键
        roi_feat = self.frame_feats[key][:, 1:, :]  # 使用该键访问值

        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)

        frame_feat = self.frame_feats[key][:, 0, :]

        frame_feat = torch.from_numpy(frame_feat).type(torch.float32)

        return region_feat, frame_feat

    # def get_video_feature(self, raw_vid_id, width=1, height=1):
    #     vid_id = raw_vid_id if raw_vid_id in self.frame_feats else raw_vid_id.removesuffix('.mp4')
    #     # generate bbox coordinates of patches
    #     patch_bbox = []
    #     patch_size = 224
    #     grid_num = 4
    #     width, height = patch_size * grid_num, patch_size * grid_num
    #
    #     for j in range(grid_num):
    #         for i in range(grid_num):
    #             patch_bbox.append([i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size])
    #
    #     roi_bbox = np.tile(np.array(patch_bbox), (32, 1)).reshape(32, 16, -1)  # [frame_num, bbox_num, -1]
    #     bbox_feat = transform_bb(roi_bbox, width, height)
    #     bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)
    #
    #     try:
    #         roi_feat = self.frame_feats[vid_id][:, 1:, :]  # [frame_num, 16, dim]
    #         roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
    #         region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)
    #     except:
    #         from IPython.core.debugger import Pdb
    #         dbg = Pdb()
    #         dbg.set_trace()
    #
    #     # vid_id = raw_vid_id if raw_vid_id in self.frame_feats else raw_vid_id.strip('.mp4')
    #
    #     frame_feat = self.frame_feats[vid_id][:, 0, :]
    #     frame_feat = torch.from_numpy(frame_feat).type(torch.float32)
    #
    #     # print('Sampled feat: {}'.format(region_feat.shape))
    #     return region_feat, frame_feat

    def __getitem__(self, index):
        cur_sample = self.data.iloc[index]
        raw_vid_id = str(cur_sample["video_id"])
        print("raw_vid_id:", raw_vid_id)
        qid = str(cur_sample['question_id'])
        # vid_id = raw_vid_id.removesuffix(".mp4")  #3.9
        # vid_id = raw_vid_id.strip(".mp4")
        if raw_vid_id.endswith(".mp4"):
            vid_id = raw_vid_id[:-4]  # 去掉后缀 ".mp4"
        else:
            vid_id = raw_vid_id

        print("vid_id:", vid_id)#3.7
        frame_size = self.frame_size[vid_id]
        width, height = frame_size['width'], frame_size['height']

        video_o, video_f = self.get_video_feature(raw_vid_id, width, height)

        vid_duration = 8  # video_f.shape[0]

        question_txt = cur_sample['question']
        question_embd = torch.tensor(
            self.bert_tokenizer.encode(
                question_txt,
                add_special_tokens=True,
                padding="longest",
                max_length=self.qmax_words,
                truncation=True,
            ),
            dtype=torch.long,
        )

        question_clip = clip.tokenize(question_txt)

        answer_txts = cur_sample["answer"]
        answer_id = self.a2id.get(answer_txts, -1)

        type = cur_sample['answer_type']
        seq_len = 0

        return {
            "video_id": raw_vid_id,
            "video": (video_o, video_f),
            # "video_f": video_f,
            "video_len": vid_duration,
            "question": question_embd,
            "question_clip": question_clip,
            "question_txt": question_txt,
            "type": type,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": answer_id,
            "seq_len": seq_len,
            "question_id": qid
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))

    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    return default_collate(batch)


#2 加载数据集
def get_videoqa_loaders(args, features, a2id, bert_tokenizer, test_mode):
    data_dir = os.path.join(args.dataset_dir, args.dataset)
    print("data_dir:", data_dir)
    if test_mode:
        print("test_mode 为 True，只需要加载测试集!\n")
        test_dataset = VideoQADataset(
            data_dir=data_dir,
            split='test',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:
        print("训练模式, 加载训练集和验证集!")
        train_dataset = VideoQADataset(
            data_dir=data_dir,
            split='train',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            id=args.id
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            drop_last=False,
            collate_fn=videoqa_collate_fn

        )
        if args.dataset.split('/')[0] in ['tgifqa', 'tgifqa2', 'msrvttmc']:
            args.val_csv_path = args.test_csv_path
        val_dataset = VideoQADataset(
            data_dir=data_dir,
            split='val',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            id=args.id

        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )
        test_loader = None

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        print(f"Train loader dataset: {train_loader.dataset}")
        print(f"Validation loader dataset: {val_loader.dataset}")

        print(f"Train batch size: {args.batch_size}, Drop last: {train_loader.drop_last}")
        print(f"Validation batch size: {args.batch_size_val}, Drop last: {val_loader.drop_last}")

        print(type(train_loader))
        print(train_loader)
        print(len(train_loader))

        print(type(val_loader))
        print(val_loader)
        print(len(val_loader))

    return train_loader, val_loader, test_loader
