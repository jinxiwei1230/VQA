from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
from thop import profile
from hdfs import InsecureClient
from metrics import logging_rank
import time
import argparse
from sklearn import preprocessing
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from tensorboardX import SummaryWriter
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import BirdModel_VT, BirdPreTrainedModel, BirdModel
from modules.optimization import BertAdam
from dataloaders.dataloader import DATALOADER_DICT
from modules.until_module import get_dual_matrix
from util import parallel_apply, get_logger
from torch.cuda.amp import autocast, GradScaler

# torch.distributed.init_process_group(backend="nccl")

global logger


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.",default=False)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.",default=False)
    # test
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.",default=True)
    parser.add_argument("--do_params", action='store_true', help="text the params of the model.",default=False)
    # test
    parser.add_argument("--use_frame_fea", action='store_true', help="whether use frame feature matching text",default=True)
    # 任务类型 retrieval
    parser.add_argument('--task', type=str, default="retrieval", choices=["retrieval_VT", "retrieval"],
                        help="choose downstream task.")
    # 数据集 msrvtt
    parser.add_argument('--dataset', type=str, default="vatex", choices=["bird", "msrvtt", "vatex", "msvd"],
                        help="choose dataset.")

    parser.add_argument('--num_thread_reader', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--text_lr', type=float, default=3e-5, help='text encoder learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=100, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=1, help='Information display frequence')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--top_frames', type=int, default=3, help='')
    parser.add_argument('--frame_sample', type=str, default="random", choices=["uniform", "random", "uniform_random"],
                        help='frame sample strategy')
    parser.add_argument('--frame_sample_len', type=str, default="fix", choices=["dynamic", "fix"],
                        help='use dynamic frame length of fix frame length')
    parser.add_argument('--language', type=str, default="chinese", choices=["chinese", "english"],
                        help='language for text encoder')
    parser.add_argument('--use_temp', action='store_true', help='whether to use temporal transformer', default=True)

    parser.add_argument("--logdir", default=None, type=str, required=False, help="log dir for tensorboardX writer")
    parser.add_argument("--output_dir", default="ckpts/val", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default="/home/disk2/dachuang1-23/HMMC/ckpts/val/pytorch_model.bin.1", type=str, required=False, help="Initial model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--enable_amp', action='store_true', help="whether to use pytorch amp")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=8e-1, help='coefficient for bert branch.')

    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval and not args.do_params:
        raise ValueError("At least one of `do_train` or `do_eval` or 'do_params' must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # world_size = torch.distributed.get_world_size()
    # torch.cuda.set_device(args.local_rank)
    # args.world_size = world_size
    # rank = torch.distributed.get_rank()
    # args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    # if args.local_rank == 0:
    #     if args.logdir:
    #         args.writer = SummaryWriter(args.logdir)
    #     logger.info("Effective parameters:")
    #     for key in sorted(args.__dict__):
    #         logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    if args.logdir:
        args.writer = SummaryWriter(args.logdir)
    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


# def init_device(args, local_rank):
#     global logger
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
#
#     n_gpu = torch.cuda.device_count()
#     logger.info("device: {} n_gpu: {}".format(device, n_gpu))
#     args.n_gpu = n_gpu
#
#     if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
#         raise ValueError(
#             "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
#                 args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
#
#     return device, n_gpu

def init_device(args):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

# def init_model(args, device, n_gpu, local_rank):
#     if args.init_model:
#         model_state_dict = torch.load(args.init_model, map_location='cpu')
#     else:
#         model_state_dict = None
#
#     # Prepare model
#     cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
#     if args.task == "retrieval_VT":
#         model = BirdModel_VT.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
#                                              task_config=args)
#     elif args.task == "retrieval":
#         model = BirdModel.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
#                                           task_config=args)
#     else:
#         raise Exception('wrong task! task should in [retrieve_VT, retrieve]')
#     # args.writer.add_graph(model)
#     model.to(device)
#
#     return model

def init_model(args, device, n_gpu):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    if args.task == "retrieval_VT":
        model = BirdModel_VT.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                             task_config=args)
    elif args.task == "retrieval":
        model = BirdModel.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                          task_config=args)
    else:
        raise Exception('wrong task! task should in [retrieve_VT, retrieve]')
    # args.writer.add_graph(model)
    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "visual_encoder.visual." in n]
    decay_chinesebert_param_tp = [(n, p) for n, p in decay_param_tp if "text_encoder." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if
                             ("visual_encoder.visual." not in n) and ("text_encoder." not in n)]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "visual_encoder.visual." in n]
    no_decay_text_param_tp = [(n, p) for n, p in no_decay_param_tp if "text_encoder." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if
                                ("visual_encoder.visual." not in n) and ("text_encoder." not in n)]

    weight_decay = args.weight_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_chinesebert_param_tp], 'weight_decay': weight_decay, 'lr': args.text_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_text_param_tp], 'weight_decay': 0.0, 'lr': args.text_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    model = model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=False)
    # if args.local_rank == 0:
    #     for name, parameters in model.named_parameters():
    #         logger.info("name:{} requires_grad:{} size:{}".format(name, parameters.requires_grad, parameters.size()))
    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        if args.task == "retrieval":
            model = BirdModel.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                              task_config=args)
        elif args.task == "retrieval_VT":
            model = BirdModel_VT.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                                 task_config=args)
        else:
            model = None

        model.to(device)
    else:
        model = None
    return model


# def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, scaler, global_step, local_rank=0):
def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, scaler, global_step):

    global logger
    torch.cuda.empty_cache()  # 清空 GPU 缓存
    model.train()  # 切换到训练模式
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    load_start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        load_finish_time = time.time()
        # if global_step % log_step == 0 and local_rank == 0:
        if global_step % log_step == 0:

            logger.info("data loader time:{}".format(load_finish_time - load_start_time))
        global_step += 1
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        # print(step)
        with autocast(enabled=args.enable_amp):
            if args.task == "retrieval_VT":
                query_ids, query_mask, video_data, video_frame, title_ids, title_mask, idx = batch
                loss = model(query_ids, query_mask, video_data, video_frame, title_ids, title_mask, idx, global_step)
            elif args.task == "retrieval":
                query_ids, query_mask, video_data, video_frame, idx = batch
                loss = model(query_ids, query_mask, video_data, video_frame, idx, global_step)
            else:
                raise ValueError("wrong task type:{}".format(args.task))
            # if n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        forward_time = time.time()
        if args.enable_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += float(loss)
        backward_time = time.time()
        # if global_step % log_step == 0 and local_rank == 0:
        if global_step % log_step == 0:

            logger.info("forward_time:{},backward_time:{}".format(forward_time - load_finish_time, backward_time - forward_time))

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            if args.enable_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # if global_step % log_step == 0 and local_rank == 0:
            if global_step % log_step == 0:

                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                if args.logdir:
                    # args.writer.add_scalar('loss', loss.item(), global_step=global_step)
                    args.writer.add_scalars('lr', {"lr%d" % i: itm for i, itm in enumerate(sorted(list(set(optimizer.get_lr()))))},
                                            global_step=global_step)
                start_time = time.time()
        load_start_time = time.time()
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_query_output_list, batch_visual_output_list, batch_title_output_list,
                       batch_frame_output_list):
    # 只取第一个文本（query_output）
    print("len(batch_query_output_list):", len(batch_query_output_list))
    print(batch_query_output_list)
    print("len(batch_visual_output_list):", len(batch_visual_output_list))
    print(batch_visual_output_list)
    print("len(batch_title_output_list):", len(batch_title_output_list))
    # print(batch_title_output_list)
    print("len(batch_frame_output_list):", len(batch_frame_output_list))
    # print(batch_frame_output_list)

    query_output = batch_query_output_list[0][0]  # 选取第一个文本
    print("query_output.shape:", query_output.shape)
    # print("query_output:", query_output)

    sim_matrix = []  # 存储文本和视频的相似度矩阵
    sim_matrix_title = []  # 存储文本和标题的相似度矩阵
    sim_matrix_frame = []  # 存储文本和帧的相似度矩阵

    # for idx1, query_output in enumerate(batch_query_output_list):
    #     # print(idx1)
    #     each_row = []
    #     title_each_row = []
    #     frame_each_row = []
    #     for idx2, (visual_output, title_output, frame_output) in enumerate(zip(batch_visual_output_list,
    #                                                                            batch_title_output_list,
    #                                                                            batch_frame_output_list)):
    #         # print(idx2)
    #         b1b2_logits = model.loose_similarity(query_output, visual_output)
    #         title_logits = model.loose_similarity(query_output, title_output)
    #         frame_logits = model.loose_similarity(query_output, frame_output)
    #
    #         frame_logits = torch.topk(frame_logits, k=model.top_frames, dim=2)[0]
    #         frame_logits = torch.mean(frame_logits, dim=2)
    #
    #         b1b2_logits = b1b2_logits.cpu().detach().numpy()
    #         title_logits = title_logits.cpu().detach().numpy()
    #         frame_logits = frame_logits.cpu().detach().numpy()
    #         each_row.append(b1b2_logits)
    #         title_each_row.append(title_logits)
    #         frame_each_row.append(frame_logits)
    #         # logger.info("b1b2_logits:{}".format(b1b2_logits.shape))
    #         # logger.info("frame_logits:{}".format(frame_logits.shape))
    #
    #     each_row = np.concatenate(tuple(each_row), axis=-1)
    #     # logger.info("each_row:{}".format(each_row.shape))
    #     title_each_row = np.concatenate(tuple(title_each_row), axis=-1)
    #     # frame_each_row = np.concatenate(tuple(frame_each_row), axis=-1)
    #     frame_each_row = np.concatenate(tuple(frame_each_row), axis=1)
    #     # logger.info("frame_each_row:{}".format(frame_each_row.shape))
    #     # sim_matrix.append(preprocessing.scale(each_row, axis=1))
    #     sim_matrix.append(each_row)
    #     sim_matrix_title.append(title_each_row)
    #     sim_matrix_frame.append(frame_each_row)


    # 遍历所有视频
    for idx2, (visual_output, title_output, frame_output) in enumerate(zip(batch_visual_output_list,
                                                                           batch_title_output_list,
                                                                           batch_frame_output_list)):
        # 计算文本和视频之间的相似度
        b1b2_logits = model.loose_similarity(query_output, visual_output)
        title_logits = model.loose_similarity(query_output, title_output)
        frame_logits = model.loose_similarity(query_output, frame_output)

        print("b1b2_logits shape:", b1b2_logits.shape)
        print("title_logits shape:", title_logits.shape)
        print("frame_logits shape:", frame_logits.shape)

        # 计算帧的相似度，选取前 k 个帧
        frame_logits = torch.topk(frame_logits, k=model.top_frames, dim=2)[0]
        frame_logits = torch.mean(frame_logits, dim=2)

        # 将结果转换为 numpy 数组
        b1b2_logits = b1b2_logits.cpu().detach().numpy()
        title_logits = title_logits.cpu().detach().numpy()
        frame_logits = frame_logits.cpu().detach().numpy()

        # 填充到相同的大小
        max_size = 100  # 你希望填充到的批次大小
        if b1b2_logits.shape[1] < max_size:
            b1b2_logits = np.pad(b1b2_logits, ((0, 0), (0, max_size - b1b2_logits.shape[1])), mode='constant',
                                 constant_values=0)
        if title_logits.shape[1] < max_size:
            title_logits = np.pad(title_logits, ((0, 0), (0, max_size - title_logits.shape[1])), mode='constant',
                                  constant_values=0)
        if frame_logits.shape[1] < max_size:
            frame_logits = np.pad(frame_logits, ((0, 0), (0, max_size - frame_logits.shape[1])), mode='constant',
                                  constant_values=0)

        print("b1b2_logits shape:", b1b2_logits.shape)
        print("title_logits shape:", title_logits.shape)
        print("frame_logits shape:", frame_logits.shape)
        # 将相似度结果加入对应的列表
        sim_matrix.append(b1b2_logits)  # 文本与视频的相似度
        sim_matrix_title.append(title_logits)  # 文本与标题的相似度
        sim_matrix_frame.append(frame_logits)  # 文本与帧的相似度
        print("----------------------------------------------")

    # 将每个视频与文本的相似度拼接成矩阵，结果应该是 n x 1
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)  # n x 1 的相似度矩阵
    sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)  # n x 1 的标题相似度矩阵
    sim_matrix_frame = np.concatenate(tuple(sim_matrix_frame), axis=0)  # n x 1 的帧相似度矩阵
    print("sim_matrix:", sim_matrix.shape)
    print("sim_matrix_title:", sim_matrix_title.shape)
    print("sim_matrix_frame:", sim_matrix_frame.shape)

    # logger.info("sim_matrix:{}".format(sim_matrix))

    # 只遍历所有的视频和标题，计算与第一个文本的相似度
    # for idx2, (visual_output, title_output, frame_output) in enumerate(zip(batch_visual_output_list,
    #                                                                        batch_title_output_list,
    #                                                                        batch_frame_output_list)):
    #     # 计算第一个文本与每个视频的相似度
    #     b1b2_logits = model.loose_similarity(query_output, visual_output)
    #     title_logits = model.loose_similarity(query_output, title_output)
    #     frame_logits = model.loose_similarity(query_output, frame_output)
    #     frame_logits = torch.topk(frame_logits, k=model.top_frames, dim=2)[0]
    #     frame_logits = torch.mean(frame_logits, dim=2)
    #     # 将结果转为 numpy 数组
    #     b1b2_logits = b1b2_logits.cpu().detach().numpy()
    #     title_logits = title_logits.cpu().detach().numpy()
    #     frame_logits = frame_logits.cpu().detach().numpy()
    #
    #     # 将每个相似度结果添加到对应的列表中
    #     sim_matrix.append(b1b2_logits)
    #     sim_matrix_title.append(title_logits)
    #     sim_matrix_frame.append(frame_logits)

    # sim_matrix = np.concatenate(tuple(sim_matrix), axis=1)
    # sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=1)
    # sim_matrix_frame = np.concatenate(tuple(sim_matrix_frame), axis=1)
    # # 打印形状
    # print("sim_matrix shape:", np.array(sim_matrix).shape)
    # print("sim_matrix_title shape:", np.array(sim_matrix_title).shape)
    # print("sim_matrix_frame shape:", np.array(sim_matrix_frame).shape)
    return sim_matrix, sim_matrix_title, sim_matrix_frame


def eval_epoch(args, model, test_dataloader, device, n_gpu, id, userid):
    # ------1.初始化模型和设备
    torch.cuda.empty_cache()  # 清空 GPU 缓存，以确保不会因为内存不足导致问题
    if hasattr(model, 'module'):
        model = model.module.to(device)  # 如果是多卡训练，`model.module` 包含实际模型
    else:
        model = model.to(device)  # 单卡训练，直接将模型转移到指定设备

    # -------2.设置模型为评估模式
    model.eval()
    logger.info("args.task:{}".format(args.task))

    # if multi_sentence_ == True: compute the similarity with multi-sentences retrieval
    # 判断test_dataloader是否支持每个视频有多个句子的情况。如果是，multi_sentence_被设为True，并准备相应的数据切割点（如cut_off_points_，sentence_num_等）。
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points  # used to tag the label when calculate the metric
        sentence_num_ = test_dataloader.dataset.sentence_num  # used to cut the sentence representation
        video_num_ = test_dataloader.dataset.video_num  # used to cut the video representation
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]
    logger.info("multi_sentence_:{}".format(multi_sentence_))


    with torch.no_grad():  # 禁用梯度计算，节省内存
        # 存储各个批次的特征输出
        batch_query_output_list, batch_visual_output_list = [], []
        batch_title_output_list = []
        batch_frame_output_list = []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------

        print("----- 1.cache the features -----")
        for bid, batch in enumerate(test_dataloader):
            print("type(batch):", type(batch))  # list
            print("len(batch):", len(batch))   # 4
            print("----------------------------")
            # 将数据加载到GPU上。根据任务类型（如retrieval_VT或retrieval），拆分批次数据。
            batch = tuple(t.to(device) for t in batch)
            if args.task == "retrieval_VT":
                query_ids, query_mask, video, video_frame, title_ids, title_mask = batch
            elif args.task == "retrieval":
                query_ids, query_mask, video, video_frame = batch
            else:
                raise ValueError("wrong task type:{}".format(args.task))

            print("bid:{}/{}".format(bid, len(test_dataloader)), end="\r")

            # 对文本进行编码，视频特征也进行编码并缓存。
            # 如果是多句子检索，则提取每个视频相关的帧并计算其视觉特征。
            if multi_sentence_:
                print("多句子检索,multi_sentence_:{}".format(multi_sentence_))
                # multi-sentences retrieval means: one frame clip has two or more descriptions.
                b, *_t = video.shape
                # logger.info("query_ids.shape:{}".format(query_ids.shape))
                # logger.info("video.shape:{}".format(video.shape))
                query_output = model.text_encoder(query_ids, query_mask)
                batch_query_output_list.append(query_output)
                title_output = torch.zeros_like(query_output)
                batch_title_output_list.append(title_output)
                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if s_ <= itm < e_]

                if len(filter_inds) > 0:
                    video = video[filter_inds, ...]
                    visual_output, frame_output = model.visual_encoder(video, video_frame)
                    # frame_output = torch.mean(frame_output, dim=1)
                    batch_visual_output_list.append(visual_output)
                    batch_frame_output_list.append(frame_output)
                total_video_num += b
            # 对于单一文本检索任务，直接提取文本和视频的特征。
            else:
                print("单一文本检索任务")
                query_output = model.text_encoder(query_ids, query_mask)
                visual_output, frame_output = model.visual_encoder(video, video_frame)
                # frame_output = torch.mean(frame_output, dim=1)
                if args.task == "retrieval_VT":
                    title_output = model.text_encoder(title_ids, title_mask)
                    logger.info("title_output.shape:{}".format(title_output.shape))
                elif args.task == "retrieval":# "retrieval" 任务下，模型不需要用标题编码，而是用一个零向量作为标题的表示
                    title_output = torch.zeros_like(query_output)# 标题的输出 title_output 会被设为与查询输出 (query_output) 相同形状的全零张量
                else:
                    raise ValueError("wrong task type:{}".format(args.task))

                # logger.info("query_output.shape:{}".format(query_output.shape))
                # logger.info("weight_VTM:{},weight_FTM:{},exp:{}".format(model.weight_VTM, model.weight_FTM,
                #                                                         model.text_encoder.logit_scale.exp()))
                logger.info("visual_output.shape:{}".format(visual_output.shape))
                logger.info("frame_output.shape:{}".format(frame_output.shape))
                logger.info("query_output.shape:{}".format(query_output.shape))

                batch_query_output_list.append(query_output)
                batch_visual_output_list.append(visual_output)
                batch_title_output_list.append(title_output)
                batch_frame_output_list.append(frame_output)

        print("batch_query_output_list:", len(batch_query_output_list))
        print("batch_visual_output_list:", len(batch_visual_output_list))
        print("batch_title_output_list:", len(batch_title_output_list))
        print("batch_frame_output_list", len(batch_frame_output_list))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        print("----- 2.calculate the similarity -----")
        logger.info("n_gpu:{}".format(n_gpu))
        # logger.info("model.weight_sum:{}".format(model.weight_sum))
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_t_output_splits = []
            batch_v_output_splits = []
            batch_title_output_splits = []
            batch_frame_output_splits = []
            bacth_len = len(batch_query_output_list)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_t_output_splits.append(batch_query_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                    batch_title_output_splits.append(batch_title_output_list)
                    batch_frame_output_splits.append(batch_frame_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))

                    devc_batch_list = [b.to(devc) for b in batch_query_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_title_output_list]
                    batch_title_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_frame_output_list]
                    batch_frame_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_t_output_splits[dev_id], batch_v_output_splits[dev_id],
                                      batch_title_output_splits[dev_id], batch_frame_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs_tuple = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            sim_matrix_title = []
            sim_matrix_frame = []
            for idx in range(len(parallel_outputs_tuple)):
                parallel_outputs, parallel_outputs_title, parallel_outputs_frame = parallel_outputs_tuple[idx]
                sim_matrix += parallel_outputs
                sim_matrix_title += parallel_outputs_title
                sim_matrix_frame += parallel_outputs_frame
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)
            sim_matrix_frame = np.concatenate(tuple(sim_matrix_frame), axis=0)
        else:
            sim_matrix_tuple = _run_on_single_gpu(model, batch_query_output_list, batch_visual_output_list,
                                                  batch_title_output_list, batch_frame_output_list)
            sim_matrix, sim_matrix_title, sim_matrix_frame = sim_matrix_tuple
            print("sim_matrix:", sim_matrix.shape)
            print("sim_matrix_title:", sim_matrix_title.shape)
            print("sim_matrix_frame:", sim_matrix_frame.shape)

            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)
            sim_matrix_frame = np.concatenate(tuple(sim_matrix_frame), axis=0)
            print("sim_matrix:", sim_matrix.shape)
            print("sim_matrix_title:", sim_matrix_title.shape)
            print("sim_matrix_frame:", sim_matrix_frame.shape)

            # batch_visual_output_list = torch.cat(batch_visual_output_list, dim=0)
            # batch_frame_output_list = torch.cat(batch_frame_output_list, dim=0)
            # batch_visual_output_list = batch_visual_output_list.cpu().detach().numpy()
            # batch_frame_output_list = batch_frame_output_list.cpu().detach().numpy()
            # np.save("/ai/swxdisk/data/vatex/features/Chinese_batch_visual_output_list", batch_visual_output_list)
            # np.save("/ai/swxdisk/data/vatex/features/Chinese_batch_frame_output_list", batch_frame_output_list)
            # np.save("/ai/swxdisk/data/vatex/features/English_batch_visual_output_list", batch_visual_output_list)
            # np.save("/ai/swxdisk/data/vatex/features/English_batch_frame_output_list", batch_frame_output_list)

        # logger.info("sim_matrix:{}".format(sim_matrix.shape))
        # logger.info("sim_matrix_frame:{}".format(sim_matrix_frame.shape))
        # np.save("/ai/swxdisk/data/msrvtt/visualize/sim_matrix", sim_matrix)
        # np.save("/ai/swxdisk/data/msrvtt/visualize/sim_matrix_frame_top2", sim_matrix_frame)
        # sim_matrix_frame = np.topk(sim_matrix_frame, k=model.top_frames, dim=2)[0]
        # sim_matrix_frame = np.mean(sim_matrix_frame, dim=2)
        if args.use_frame_fea:
            sim_matrix += sim_matrix_frame

        if args.task == "retrieval_VT":
            # logger.info("sim_matrix_title:{}".format(sim_matrix_title))
            weight_title = model.weight_title
            sim_matrix += weight_title * sim_matrix_title
            # sim_matrix = weight_title * sim_matrix_title

    logger.info("sim matrix size:  {}".format(np.array(sim_matrix).shape))
    # sim_matrix = get_dual_matrix(sim_matrix)
    # print('相似度矩阵', sim_matrix)
    # print('1',np.argmax(sim_matrix[0]))
    tv_metrics = logging_rank(sim_matrix, multi_sentence_, cut_off_points_, logger)

    # 假设 sim_matrix 是相似度矩阵
    sim_matrix = np.array(sim_matrix)  # 确保 sim_matrix 是一个 numpy 数组
    # 读取视频ID文件
    with open('/home/disk2/dachuang1-23/text/test.txt', 'r') as file:
        video_ids = [line.strip() for line in file.readlines()]
    # 获取第一个文本与所有视频的相似度
    first_text_similarities = sim_matrix  # sim_matrix[0] 是第一个文本与所有视频之间的相似度
    print("first_text_similarities: ")
    print(first_text_similarities)
    print("(argsort)first_text_similarities: ")
    print(np.argsort(first_text_similarities))
    # 获取相似度最高的5个视频的索引
    top_k_indices = np.argsort(first_text_similarities)[-5:][::-1]  # 排序并取前3个最大值
    print("top_k_indices:", top_k_indices)
    # 获取与第一个文本相似度最高的5个视频ID
    top_k_indices = [video_ids[i] for i in top_k_indices]
    print("与第一个文本相似度最高的5个视频ID:", top_k_indices)
    # 将索引保存到文件
    output_dir = f'/home/disk2/dachuang1-23/text/kafka_result/{id}'
    file_name = 'top_k_video_indices.txt'  # 确保每个文件都有后缀名
    output_file = os.path.join(output_dir, file_name)
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, 'w') as f:
        for idx in top_k_indices:
            index = int(idx)
            index = str(index)
            f.write(f"{index}\n")  # 每个索引写入一行
    print("Top k video indices have been saved to 'top_k_video_indices.txt'.")

    # 读取并获取第一个值
    with open(output_file, 'r') as f:
        best_value = f.readline().strip()  # 读取第一行，并去掉多余的空白符（如换行符）
        print(f"The first value in the file is: {best_value}")
    #/bdap/students/public/four_avi
    # name = str(best_value) + ".avi"
    name = str(best_value) + ".mp4"
    print("name:", name)
    try:
        # copy_video_file_in_hdfs( '/bdap/students/public/four_avi', f'/bdap/students/{userid}/videos', name)
        copy_video_file_in_hdfs( '/bdap/students/public/four_mp4', f'/bdap/students/{userid}/videos', name)
        print("文件复制成功！")
    except Exception as e:
        print(f"Exception occurred: {e}")
        print("复制文件出错！")

    return tv_metrics

def copy_video_file_in_hdfs(hdfs_dir1, hdfs_dir2, file_name):
    try:
        hdfs_client = InsecureClient("http://10.92.64.241:14000", user='yanch')

        # 构建源文件和目标文件路径
        source_file_path = os.path.join(hdfs_dir1, file_name)
        dest_file_path = os.path.join(hdfs_dir2, file_name)

        # 检查源文件是否存在
        try:
            hdfs_client.status(source_file_path)  # 如果文件不存在，会抛出异常
        except Exception as e:
            print(f"源文件 {source_file_path} 在 HDFS 中不存在: {e}")
            return

        # 检查目标文件是否存在
        try:
            hdfs_client.status(dest_file_path)  # 如果目标文件已存在，会返回文件状态
            # 如果目标文件存在，打印信息并跳过复制
            print(f"目标文件 {dest_file_path} 已经存在，跳过复制。")
        except Exception as e:
            # 如果目标文件不存在，则开始复制
            print(f"目标文件 {dest_file_path} 不存在，开始复制文件。")
            with hdfs_client.read(source_file_path) as reader:
                with hdfs_client.write(dest_file_path, overwrite=True) as writer:
                    writer.write(reader.read())  # 将源文件内容写入目标路径
            print(f"成功将 {file_name} 从 {hdfs_dir1} 复制到 {hdfs_dir2}")

    except Exception as e:
        print(f"Failed to copy file from {hdfs_dir1} to {hdfs_dir2}: {e}")


def main(id, userid):
    # ---1.PyTorch版本和CUDA信息打印
    print("PyTorch版本:",torch.__version__)  # PyTorch版本
    print("PyTorch编译时使用的CUDA版本",torch.version.cuda)  # PyTorch编译时使用的CUDA版本

    print("id", id)
    global logger
    # ---2.参数设置和设备初始化
    args = get_args()  # 从命令行或配置文件中获取程序运行参数
    args = set_seed_logger(args)  # 置随机种子以确保结果可重复性，并初始化日志记录器
    # device, n_gpu = init_device(args, args.local_rank)  # 根据参数选择运行设备（CPU/GPU），并返回设备信息
    device, n_gpu = init_device(args)  # 根据参数选择运行设备（CPU/GPU），并返回设备信息

    # ---3.Tokenizer和模型初始化
    # get text pretrained path
    pretrained_text = "/home/disk2/DATA/MSRVTT/hfl/chinese-roberta-wwm-ext"
    args.pretrained_text = pretrained_text
    if args.language == "chinese":  # 如果语言是中文，使用 BertTokenizer；否则，使用 ClipTokenizer
        tokenizer = BertTokenizer.from_pretrained(pretrained_text)
    else:
        tokenizer = ClipTokenizer()

    # model = init_model(args, device, n_gpu, args.local_rank)  # 模型通过 init_model 函数初始化，具体参数如设备、GPU 数量和分布式训练的本地 rank 都会被传入

    model = init_model(args, device, n_gpu)  # 模型通过 init_model 函数初始化，具体参数如设备、GPU 数量和分布式训练的本地 rank 都会被传入

    # ---4.冻结层（freeze testing）代码块（注释掉）
    ## ####################################
    # freeze testing
    # 冻结模型的部分层，防止某些层的参数在训练过程中更新。这里的代码被注释掉了，意味着目前模型的所有层都参与训练。
    ## ####################################
    '''
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "visual_encoder") and args.freeze_layer_num > -1:
        for name, param in model.visual_encoder.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue  # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False
    '''
    # ---5.数据加载
    assert args.dataset in DATALOADER_DICT  # 检查 args.dataset 是否在数据加载器字典中。
    test_dataloader, test_length = DATALOADER_DICT[args.dataset]["test"](args, tokenizer, id)  # 使用相应的 test 方法加载测试数据
    print("数据加载完成！")
    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    # ---6.训练模式
    print("args.do_train:",args.do_train)
    print("args.do_eval:",args.do_eval)
    print("args.do_params:",args.do_params)
    if args.do_train:  # 如果开启训练模式，还会加载训练数据，执行训练和评估。
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.dataset]["train"](args, tokenizer)

        # 优化器与调度器
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        # logger.info("train_dataloader len = {}".format(len(train_dataloader)))
        # logger.info("gradient_accumulation_steps = {}".format(args.gradient_accumulation_steps))
        coef_lr = args.coef_lr
        # optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
        #                                              args.local_rank, coef_lr=coef_lr)
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
                                                     coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        if args.enable_amp:
            scaler = GradScaler()
        else:
            scaler = None

        # ---7.训练过程
        for epoch in range(args.epochs):  # 训练过程
            train_sampler.set_epoch(epoch)
            # tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
            #                                  scheduler, scaler, global_step, local_rank=args.local_rank)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, scaler, global_step)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                # for name, param in model.named_parameters():
                # args.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                # writer.add_histogram(name + '/grad', param.requires_grad_().clone().cpu().data.numpy(), epoch)
                if epoch % 1 == 0:
                    ## Uncomment if want to save checkpoint
                    output_model_file = save_model(epoch, args, model, type_name="")
                    # if epoch == 100:
                    # ---8.评估和模型保存
                    metrics = eval_epoch(args, model, test_dataloader, device, n_gpu)
                    if args.logdir:  #模型评估与保存
                        args.writer.add_scalars('metrics', {'R1': metrics["R1"], 'R5': metrics["R5"],
                                                            'R10': metrics["R10"]}, global_step=epoch)
                    if best_score < metrics["R1"]:
                        best_score = metrics["R1"]
                        best_output_model_file = output_model_file
                    logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
    # ---9.评估模式
    elif args.do_eval:  # 仅进行模型评估
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu, id, userid)

    # ---10.参数统计与性能分析
    elif args.do_params:  #  参数统计与性能分析
        logger.info("do_params begin!")
        # total = sum([param.nelement() for param in model.parameters()])
        total = sum(p.numel() for p in model.parameters())
        logger.info("Number of parameter: %.2fM" % (total / 1e6))
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            query_ids, query_mask, pos_video_data, pos_title_ids, pos_title_mask, = batch
            flops, params = profile(model, (query_ids, query_mask, pos_video_data, pos_title_ids, pos_title_mask,))
            print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
            break
    # ---11.资源释放
    if args.local_rank == 0 and args.logdir:
        args.writer.close()

#
# if __name__ == "__main__":
#     main()
