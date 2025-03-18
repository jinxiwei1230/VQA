# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5
# Set your own appid/appkey.
appid = '20240716002101367'
appkey = '1lRfgIX4PCbSnrhMB4kh'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang = 'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
import os
import json
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens

from IPython.core.debugger import Pdb
dbg = Pdb()

# 读取词汇表
vocab_path = '/home/disk2/dachuang1-23/data/datasets/agqa/vocab.json'
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# 反转词汇表，创建从索引到词汇的映射
index_to_word = {index: word for word, index in vocab.items()}


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def baidu_api(query,from_lang,to_lang):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    print(json.dumps(result, indent=4, ensure_ascii=False))
    return result["trans_result"][0]['dst']


def eval(model, val_loader, a2v, args, test=False):
    print("111111111111111111111111111")
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    results = {}
    print("222222222222222222222222222222")
    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            print("333333333333333333333333333333333333")
        for i, batch in enumerate(val_loader):
            answer_id, answer, video, question, question_clip = (
                batch["answer_id"],
                batch["answer"],
                (batch["video"][0].cuda(), batch["video"][1].cuda()),
                batch["question"].cuda(),
                batch['question_clip'].cuda()
            )
            video_len = batch["video_len"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video[1].size(1)).cuda()
            count += answer_id.size(0)

            # 检查每个批次的数据是否为空
            if answer_id.size(0) == 0:
                logging.warning(f"Empty batch encountered at index {i}. Skipping batch.")
                continue

            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    # video_mask=video_mask,
                    question_clip=question_clip
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                print("答案")
                print("index_to_word:", index_to_word)
                print("topk:", topk[0])
                # 处理批次中的每个问题
                predictions = [index_to_word[idx.item()] for idx in topk[0]]
                print("predictions:", predictions)
                for bs, qid in enumerate(batch['question_id']):
                    print("@1@2@3")
                    print("bs:", bs)

                    predicted_answer = index_to_word[int(topk[bs][0])]  # 获取预测的答案文本
                    predicted_answer_trans = baidu_api(predicted_answer, from_lang, to_lang)

                    results[qid] = {'prediction': predicted_answer_trans, 'answer': int(answer_id[bs])}

                # for bs, qid in enumerate(batch['question_id']):
                #     print("@1@1@1")
                #     results[qid] = {'11prediction': int(topk.numpy()[bs,0]), '22answer':int(answer_id.numpy()[bs])}
            else:
                fusion_proj, answer_proj = model(
                    video,
                    question,
                    text_mask=question_mask,
                    # video_mask=video_mask,
                    answer=answer.cuda(),
                    question_clip=question_clip
                )
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(batch['question_id']):
                    # print("@2@2@2")
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
        print("444444444444444444444444444444444444444444")

    step = "val" if not test else "test"
    print("55555555555555555555555555555555555")
    # 添加对 count 是否为零的检查，防止除以零错误
    if count > 0:
        for k in metrics:
            v = metrics[k] / count
            logging.info(f"{step} {k}: {v:.2%}")
        acc = metrics['acc'] / count
    else:
        logging.warning(f"{step} count is zero. Skipping accuracy calculation.")
        acc = 0  # 或者可以选择设置为其他适当的默认值

    json.dump(results, open(os.path.join(args.save_dir, f"val-{acc:.5%}.json"), "w"), ensure_ascii=False)

    return metrics["acc"] / count

#3 调用训练
def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, val_loader=None, best_val_acc=None, best_epoch=None):
    print("调用：train")
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    print("for循环开始")
    for i, batch in enumerate(train_loader):
        print(f"-----batch{i}-----")
        """
        (batch_size = 8)
        video_id, 字段数据形状: 8
        video, 字段数据形状: 2
        video_len, 字段数据形状: torch.Size([8])
        question, 字段数据形状: torch.Size([8, 28])
        question_clip, 字段数据形状: torch.Size([8, 1, 77])
        question_txt, 字段数据形状: 8
        type, 字段数据形状: 8
        answer_id, 字段数据形状: torch.Size([8])
        answer_txt, 字段数据形状: 8
        answer, 字段数据形状: torch.Size([8])
        seq_len, 字段数据形状: torch.Size([8])
        question_id, 字段数据形状: 8
        """
        print("video_id:", batch["video_id"])
        # 从 batch 字典中提取出视频数据、问题数据、答案等，并将它们迁移到 GPU 上进行训练
        answer_id, answer, video, question, question_clip = (
            batch["answer_id"],
            batch["answer"],
            (batch["video"][0].cuda(), batch["video"][1].cuda()),  # 将视频数据的两个部分分别迁移到 GPU
            batch["question"].cuda(),
            batch['question_clip'].cuda()
        )
        video_len = batch["video_len"]


        # question_mask 是用于标记问题中有效单词的位置，它的作用是帮助模型忽略无效或填充的单词，从而更好地进行处理
        question_mask = (question > 0).float()
        # video_mask = (
        #     get_mask(video_len, video[1].size(1)).cuda() if args.max_feats > 0 else None
        # )

        N = answer_id.size(0)  # N 表示当前批次的样本数

        if not args.mc:
            # yes
            # 调用模型的 _compute_answer_embedding 方法,计算答案的嵌入向量或某些预处理步骤。
            model.module._compute_answer_embedding(a2v)

            # 将 video、question、text_mask、question_clip传入模型，获取预测结果。
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                question_clip=question_clip
            )
        else:
            print("------------------------MC---------------------------")
            fusion_proj, answer_proj = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                answer=answer.cuda(),
                question_clip=question_clip
            )
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        # 计算损失
        if args.dataset == "ivqa":
            print("---ivqa---")
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            # yes
            # print("---!ivqa---")

            vqa_loss = criterion(predicts, answer_id.cuda())  # 计算预测结果与真实答案之间的损失
            predicted = torch.max(predicts, dim=1).indices.cpu()
            running_acc.update((predicted == answer_id).sum().item() / N, N)  # 计算准确率，并更新running_acc

        if args.mlm_prob:
            print("-----args.mlm_prob-----")
            inputs = batch["question"]
            inputs, labels = mask_tokens(
                inputs, model.module.bert.bert_tokenizer, mlm_probability=0.15
            )
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        else:
            # yes
            # print("-----!args.mlm_prob-----")
            loss = vqa_loss

        # 如果损失为 NaN，则进行调试
        if torch.isnan(loss):
            print("当前计算出的损失 loss 是否包含 NaN:")
            print(batch['question_id'], batch['video_id'], loss)
            dbg.set_trace()

        #  梯度更新和参数优化
        optimizer.zero_grad()  # 将优化器（optimizer）的梯度清零
        loss.backward()  # 计算损失 loss 相对于网络中各个参数的梯度
        if args.clip:
            # yes
            # print("-----args.clip-----")
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()  # 优化器的参数更新操作
        scheduler.step()  # 更新学习率调度器的步进

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)  # 更新running_vqa_loss：存储和计算损失（loss）平均值的对象

        if args.mlm_prob:
            print("启用了 MLM训练模式")
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)


        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, Training MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()

        if val_loader is not None and (i + 1) % (len(train_loader) // (args.freq_display / 2)) == 0:
            # print("验证并选出最佳模型！")
            val_acc = eval(model, val_loader, a2v, args, test=False)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
            else:
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, f"model-{epoch}.pth")
                )

    return best_val_acc, best_epoch
