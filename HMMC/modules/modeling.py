from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from abc import ABC
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from diffdist import functional
from transformers import AutoConfig, AutoModel, BertTokenizer
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.until_module import PreTrainedModel, AllGather, CrossEn, Dual_CrossEn
from modules.module_cross import TextEncoder, VisualEncoder, CrossConfig, BertLMPredictionHead
# from modules.module_vilbert import co_attention_model, BertLMPredictionHead, BertConfig
from modules.module_clip import CLIP, convert_weights, build_model

logger = logging.getLogger(__name__)
allgather = AllGather.apply


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, *inputs, **kwargs)

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class BirdPreTrainedModel(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.rank = task_config.local_rank
        self.mlm_probability = cross_config.mlm_probability
        self.top_frames = task_config.top_frames
        # self.weight_sum = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
        self.weight_FAM = cross_config.weight_FAM
        self.weight_VTM = cross_config.weight_VTM
        self.weight_FTM = cross_config.weight_FTM
        self.weight_MLM = cross_config.weight_MLM
        self.contrast_momentum = task_config.contrast_momentum
        self.contrast_temperature = task_config.contrast_temperature
        self.contrast_num_negative = task_config.contrast_num_negative
        ################## chinese text Encoder
        if self.task_config.language == "chinese":
            self.tokenizer = BertTokenizer.from_pretrained(self.task_config.pretrained_text)
        else:
            self.tokenizer = ClipTokenizer()
        if self.rank == 0:
            logger.info("voacb_size:{}".format(self.tokenizer.vocab_size))
        t_config = AutoConfig.from_pretrained(self.task_config.pretrained_text)
        self.text_encoder = TextEncoder(self.task_config, cross_config)
        self.text_encoder_k = TextEncoder(self.task_config, cross_config)
        self.t_projector = MLP(num_layers=cross_config.proj_num_layers)
        self.t_projector_k = MLP(num_layers=cross_config.proj_num_layers)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.t_projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.t_projector_k)
        # for MLM
        t_config.hidden_size = cross_config.temporal_hidden_size
        t_config.vocab_size = self.tokenizer.vocab_size
        self.cls = BertLMPredictionHead(t_config)
        ################## visual_encoder
        self.visual_encoder = VisualEncoder(self.task_config, cross_config)
        self.visual_encoder_k = VisualEncoder(self.task_config, cross_config)
        self.v_projector = MLP(num_layers=cross_config.proj_num_layers)
        self.v_projector_k = MLP(num_layers=cross_config.proj_num_layers)
        self.v_predictor = MLP(num_layers=cross_config.pred_num_layers)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.v_projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.v_projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.v_predictor)
        ################# momemtun mdoel pairs
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_k],
                            [self.text_encoder, self.text_encoder_k],
                            [self.v_projector, self.v_projector_k],
                            [self.t_projector, self.t_projector_k],
                            ]
        self.copy_params()
        ################## create queue
        self.register_buffer("queue_v_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_frame_proj_ng", torch.randn(cross_config.temporal_hidden_size,
                                                                self.contrast_num_negative * self.task_config.max_frames))
        self.register_buffer("queue_frame_cross_ng", torch.randn(cross_config.temporal_hidden_size,
                                                                 self.contrast_num_negative * self.task_config.max_frames))
        self.register_buffer("queue_title_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_tag_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.queue_v_cross_ng = F.normalize(self.queue_v_cross_ng, dim=0)
        self.queue_frame_proj_ng = F.normalize(self.queue_frame_proj_ng, dim=0)
        self.queue_frame_cross_ng = F.normalize(self.queue_frame_cross_ng, dim=0)
        self.queue_title_cross_ng = F.normalize(self.queue_title_cross_ng, dim=0)
        self.queue_tag_cross_ng = F.normalize(self.queue_tag_cross_ng, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()


        # self.apply(self.init_weights)

    def get_mlm_loss(self, input_ids, input_mask):
        to_mask_input_ids = input_ids.clone()
        input_labels = to_mask_input_ids.clone()
        input_probability_matrix = torch.full(input_labels.shape, self.mlm_probability)
        masked_input_ids, input_labels = self.mask(to_mask_input_ids, self.tokenizer.vocab_size,
                                                   input_mask.device, targets=input_labels,
                                                   probability_matrix=input_probability_matrix)
        masked_input_output = self.text_encoder(masked_input_ids, input_mask, return_hidden=True)
        mlm_input_loss = self.calculate_mlm_loss(masked_input_output, input_labels)
        return mlm_input_loss

    def calculate_mlm_loss(self, sequence_output_mlm, labels):

        mlm_scores = self.cls(sequence_output_mlm)
        # logger.info("sequence_output_mlm.shape:{}".format(sequence_output_mlm.shape))
        # logger.info("mlm_scores.shape:{}".format(mlm_scores.shape))
        # logger.info("labels.shape:{}".format(labels.shape))
        mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.tokenizer.vocab_size),
                                   labels.view(-1), ignore_index=-100)
        return mlm_loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        # logger.info("masked_indices:{}".format(masked_indices))
        # logger.info("masked_indices.shape:{}".format(masked_indices.shape))
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def loose_similarity(self, sequence_output, visual_output):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        visual_output = visual_output.squeeze()
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze()
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.text_encoder.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        # if self.rank == 0:
        #     logger.info("logit_scale:{},dtype:{}".format(logit_scale, logit_scale.dtype))
        #     logger.info("sequence_output.shape:{}".format(sequence_output.shape))
        #     logger.info("visual_output.shape:{}".format(visual_output.shape))
        print("len(visual_output.shape):", len(visual_output.shape))
        print("len(sequence_output.shape):", len(sequence_output.shape))
        if len(sequence_output.shape) == 1:
            # 扩展 sequence_output 使其成为 2D 张量，形状变为 (1, feature_dim)
            sequence_output = sequence_output.unsqueeze(0)

        if len(visual_output.shape) == 2:
            retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        else:
            visual_temp = visual_output.permute(0, 2, 1)
            retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_temp)
            retrieve_logits = retrieve_logits.permute(1, 0, 2)
        # 打印相似度矩阵
        # print("Retrieve logits (Similarity Matrix):", retrieve_logits)

        return retrieve_logits

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_k in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_k.data.copy_(param.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_k in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_k.data = param_k.data * self.contrast_momentum + param.data * (1. - self.contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_fea_k, tag_fea_k, title_fea_k, frame_fea_k, frame_proj_k):

        # gather keys before updating queue
        # [bs,hidden]
        v_fea_k = dist_collect(v_fea_k).squeeze()
        v_fea_k = F.normalize(v_fea_k, dim=1)
        tag_fea_k = dist_collect(tag_fea_k).squeeze()
        tag_fea_k = F.normalize(tag_fea_k, dim=1)
        title_fea_k = dist_collect(title_fea_k).squeeze()
        title_fea_k = F.normalize(title_fea_k, dim=1)
        # [bs,frame,hidden]
        frame_fea_k = dist_collect(frame_fea_k).squeeze()
        frame_fea_k = F.normalize(frame_fea_k, dim=2)
        frame_proj_k = dist_collect(frame_proj_k).squeeze()
        frame_proj_k = F.normalize(frame_proj_k, dim=2)

        batch_size = v_fea_k.size(0)
        frame_num = frame_fea_k.size(1)
        frame_fea_k = frame_fea_k.view(-1, frame_fea_k.size(-1))
        frame_proj_k = frame_proj_k.view(-1, frame_proj_k.size(-1))

        ptr = int(self.queue_ptr)
        # if self.rank == 0:
        #     logger.info(
        #         "begin>>>>: ptr:{},batch_size:{},frame_num:{},queue_size:{}".format(ptr, batch_size, frame_num, self.contrast_num_negative))
        #     logger.info("v1_self_k.shape:{},tag_cross_k.shape:{},frame_proj_k.shape:{}".format(v_fea_k.shape, tag_fea_k.shape, frame_proj_k.shape))

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_v_cross_ng[:, ptr:ptr + batch_size] = v_fea_k.T
        self.queue_tag_cross_ng[:, ptr:ptr + batch_size] = tag_fea_k.T
        self.queue_title_cross_ng[:, ptr:ptr + batch_size] = title_fea_k.T

        self.queue_frame_proj_ng[:, ptr * frame_num:(ptr + batch_size) * frame_num] = frame_proj_k.T
        self.queue_frame_cross_ng[:, ptr * frame_num:(ptr + batch_size) * frame_num] = frame_fea_k.T
        # move pointer
        ptr = (ptr + batch_size) % self.contrast_num_negative

        # if self.rank == 0:
        #     logger.info("end>>>>: ptr:{}".format(ptr))
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        q = q.squeeze()
        q = F.normalize(q, dim=1)
        k = k.squeeze()
        k = F.normalize(k, dim=1)

        bs = q.size(0)
        # logger.info("q.dtype:{},k.dtype:{}".format(q.dtype, k.dtype))
        # positive logits: Nx1
        # >>>>>>got error in apex:amp level=01!!!!!!!!!
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos = torch.matmul(q, k.T)
        l_pos = torch.diag(l_pos).reshape([bs, -1])
        # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        l_neg = torch.matmul(q, queue.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # if self.rank == 0:
        #     logger.info("logits.shape:{}".format(logits.shape))
        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)

    def frame_self_loss(self, frame_fea, frame_fea_k, queue_frame_ng):
        loss = 0.

        for i in range(frame_fea.size(1) - 1):
            frame_loss = self.contrastive_loss(frame_fea[:, i, :], frame_fea_k[:, i+1, :], queue_frame_ng) \
                        + self.contrastive_loss(frame_fea[:, i+1, :], frame_fea_k[:, i, :], queue_frame_ng)
            loss += frame_loss
        loss = loss / (frame_fea.size(1) - 1)
        return loss

    def frame_cross_loss(self, frame_fea, frame_fea_k, queue_frame_ng, text_fea, text_fea_k, queue_text_ng):
        loss = 0.
        for i in range(frame_fea.size(1)):
            frame_loss = self.contrastive_loss(text_fea, frame_fea_k[:, i, :], queue_frame_ng) + \
                         self.contrastive_loss(frame_fea[:, i, :], text_fea_k, queue_text_ng)
            loss += frame_loss
        loss = loss / frame_fea.size(1)
        return loss

    def forward(self, video_data, video_frame, tag_ids, tag_mask, title_ids, title_mask, global_step):
        tag_ids = tag_ids.view(-1, tag_ids.shape[-1])
        tag_mask = tag_mask.view(-1, tag_mask.shape[-1])
        title_ids = title_ids.view(-1, title_ids.shape[-1])
        title_mask = title_mask.view(-1, title_mask.shape[-1])
        # bs x frames x 3 x H x W
        video = torch.as_tensor(video_data)

        if self.rank == 0 and global_step % self.task_config.n_display == 0:
            logger.info("video1.shape:{}, dtype:{}, device:{}".format(video.shape, video.dtype, video.device))

        if self.training:
            # loss = 0.0
            v_fea, frame_fea = self.visual_encoder(video, video_frame)
            if self.task_config.dataset == "bird":
                tag_fea = self.text_encoder(tag_ids, tag_mask)
            title_fea = self.text_encoder(title_ids, title_mask)

            # for video self supervised learning
            # [bs,hidden_size]
            bs, frame, hidden = frame_fea.shape
            frame_fea = frame_fea.view(-1, hidden)
            frame_proj = self.v_projector(frame_fea)
            frame_pred = self.v_predictor(frame_proj)
            frame_fea = frame_fea.view(bs, frame, hidden)
            frame_proj = frame_proj.view(bs, frame, hidden)
            frame_pred = frame_pred.view(bs, frame, hidden)
            if self.rank == 0 and global_step % self.task_config.n_display == 0:
                logger.info("v_fea.shape:{},device:{}".format(v_fea.shape, v_fea.device))
                logger.info("frame_fea.shape:{},device:{}".format(frame_fea.shape, frame_fea.device))
                logger.info("frame_proj.shape:{},device:{}".format(frame_proj.shape, frame_proj.device))
                logger.info("title_fea.shape:{}".format(title_fea.shape))
                logger.info("queue_v_cross_ng.shape:{}".format(self.queue_v_cross_ng.shape))
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update()  # update the key encoder

                tag_fea_k = self.text_encoder_k(tag_ids, tag_mask)
                title_fea_k = self.text_encoder_k(title_ids, title_mask)
                #
                v_fea_k, frame_fea_k = self.visual_encoder_k(video, video_frame)
                frame_fea_k = frame_fea_k.view(-1, hidden)
                frame_proj_k = self.v_projector_k(frame_fea_k)
                frame_fea_k = frame_fea_k.view(bs, frame, hidden)
                frame_proj_k = frame_proj_k.view(bs, frame, hidden)

            # compute loss
            if self.rank == 0 and global_step % self.task_config.n_display == 0:
                logger.info(
                    "dtype: v_fea:{},v_fea_k:{},title_fea:{}".format(v_fea.dtype, v_fea_k.dtype, title_fea.dtype))
            # single video modality: video queue loss
            loss_FAM = self.frame_self_loss(frame_pred, frame_proj_k, self.queue_frame_proj_ng)
            # cross modality: cross queue loss
            v_title_queue_loss = self.contrastive_loss(v_fea, title_fea_k, self.queue_title_cross_ng) \
                                 + self.contrastive_loss(title_fea, v_fea_k, self.queue_v_cross_ng)
            if self.task_config.dataset == "bird":
                v_tag_queue_loss = self.contrastive_loss(v_fea, tag_fea_k, self.queue_tag_cross_ng) \
                                   + self.contrastive_loss(tag_fea, v_fea_k, self.queue_v_cross_ng)
                loss_VTM = (v_tag_queue_loss + v_title_queue_loss) / 2
            else:
                loss_VTM = v_title_queue_loss

            loss_FTM = 0.
            if self.task_config.use_frame_fea:
                frame_title_loss = self.frame_cross_loss(frame_fea, frame_fea_k, self.queue_frame_cross_ng, title_fea,
                                                         title_fea_k, self.queue_title_cross_ng)
                if self.task_config.dataset == "bird":
                    frame_tag_loss = self.frame_cross_loss(frame_fea, frame_fea_k, self.queue_frame_cross_ng, tag_fea,
                                                           tag_fea_k, self.queue_tag_cross_ng)
                    loss_FTM += (frame_tag_loss + frame_title_loss) / 2
                else:
                    loss_FTM = frame_title_loss

            # single text modality: text queue loss
            # t_queue_loss = self.contrastive_loss(title_fea, tag_fea_k, self.queue_tag_cross_ng) \
            #                + self.contrastive_loss(tag_fea, title_fea_k, self.queue_v_cross_ng)

            # dequeue_and_enqueue
            self._dequeue_and_enqueue(v_fea_k, tag_fea_k, title_fea_k, frame_fea_k, frame_proj_k)

            # mlm loss

            mlm_title_loss = self.get_mlm_loss(title_ids, title_mask)
            if self.task_config.dataset == "bird":
                mlm_tag_loss = self.get_mlm_loss(tag_ids, tag_mask)
                loss_MLM = (mlm_tag_loss + mlm_title_loss) / 2
            else:
                loss_MLM = mlm_title_loss

            # total loss
            loss = self.weight_FAM * loss_FAM + self.weight_VTM * loss_VTM + self.weight_FTM * loss_FTM + self.weight_MLM * loss_MLM
            if self.rank == 0:
                if global_step % self.task_config.n_display == 0:
                    logger.info("loss:{},loss_FAM:{},loss_VTM:{},loss_FTM:{},loss_MLM:{}"
                                "".format(loss, loss_FAM, loss_VTM, loss_FTM, loss_MLM))
                if self.task_config.logdir:
                    loss_item = {"loss": float(loss), "loss_FAM": float(loss_FAM), "loss_VTM": float(loss_VTM),
                                 "loss_FTM": float(loss_FTM), "loss_MLM": float(loss_MLM)}
                    self.task_config.writer.add_scalars('loss', loss_item, global_step=global_step)
                    # self.task_config.writer.add_scalar('loss', video_cross_loss, global_step=global_step)
            return loss
        else:
            return None

###################### this model is just for compare momentum queue and inbatch negative ##########
# class BirdPreTrainedModel(CLIP4ClipPreTrainedModel):
#     def __init__(self, cross_config, task_config):
#         super(BirdPreTrainedModel, self).__init__(cross_config)
#         self.task_config = task_config
#         self.rank = task_config.local_rank
#         self.mlm_probability = cross_config.mlm_probability
#         self.top_frames = task_config.top_frames
#         # self.weight_sum = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
#         self.weight_FAM = cross_config.weight_FAM
#         self.weight_VTM = cross_config.weight_VTM
#         self.weight_FTM = cross_config.weight_FTM
#         self.weight_MLM = cross_config.weight_MLM
#         self.contrast_momentum = task_config.contrast_momentum
#         self.contrast_temperature = task_config.contrast_temperature
#         self.contrast_num_negative = task_config.contrast_num_negative
#         ################## chinese text Encoder
#         if self.task_config.language == "chinese":
#             self.tokenizer = BertTokenizer.from_pretrained(self.task_config.pretrained_text)
#         else:
#             self.tokenizer = ClipTokenizer()
#         if self.rank == 0:
#             logger.info("voacb_size:{}".format(self.tokenizer.vocab_size))
#         t_config = AutoConfig.from_pretrained(self.task_config.pretrained_text)
#         self.text_encoder = TextEncoder(self.task_config, cross_config)
#
#         # for MLM
#         t_config.hidden_size = cross_config.temporal_hidden_size
#         t_config.vocab_size = self.tokenizer.vocab_size
#         self.cls = BertLMPredictionHead(t_config)
#         ################## visual_encoder
#         self.visual_encoder = VisualEncoder(self.task_config, cross_config)
#
#         ################## loss function
#         self.loss_fct = CrossEn()
#         self.loss_fct_dual = Dual_CrossEn()
#
#
#         # self.apply(self.init_weights)
#
#     def get_mlm_loss(self, input_ids, input_mask):
#         to_mask_input_ids = input_ids.clone()
#         input_labels = to_mask_input_ids.clone()
#         input_probability_matrix = torch.full(input_labels.shape, self.mlm_probability)
#         masked_input_ids, input_labels = self.mask(to_mask_input_ids, self.tokenizer.vocab_size,
#                                                    input_mask.device, targets=input_labels,
#                                                    probability_matrix=input_probability_matrix)
#         masked_input_output = self.text_encoder(masked_input_ids, input_mask, return_hidden=True)
#         mlm_input_loss = self.calculate_mlm_loss(masked_input_output, input_labels)
#         return mlm_input_loss
#
#     def calculate_mlm_loss(self, sequence_output_mlm, labels):
#
#         mlm_scores = self.cls(sequence_output_mlm)
#         # logger.info("sequence_output_mlm.shape:{}".format(sequence_output_mlm.shape))
#         # logger.info("mlm_scores.shape:{}".format(mlm_scores.shape))
#         # logger.info("labels.shape:{}".format(labels.shape))
#         mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.tokenizer.vocab_size),
#                                    labels.view(-1), ignore_index=-100)
#         return mlm_loss
#
#     def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
#         if masked_indices is None:
#             masked_indices = torch.bernoulli(probability_matrix).bool()
#
#         masked_indices[input_ids == self.tokenizer.pad_token_id] = False
#         masked_indices[input_ids == self.tokenizer.cls_token_id] = False
#         # logger.info("masked_indices:{}".format(masked_indices))
#         # logger.info("masked_indices.shape:{}".format(masked_indices.shape))
#         if targets is not None:
#             targets[~masked_indices] = -100  # We only compute loss on masked tokens
#
#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
#         input_ids[indices_replaced] = self.tokenizer.mask_token_id
#
#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
#         input_ids[indices_random] = random_words[indices_random]
#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#
#         if targets is not None:
#             return input_ids, targets
#         else:
#             return input_ids
#
#     def loose_similarity(self, sequence_output, visual_output):
#         sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
#
#         visual_output = visual_output.squeeze()
#         visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
#
#         sequence_output = sequence_output.squeeze()
#         sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
#
#         logit_scale = self.text_encoder.logit_scale.exp()
#         logit_scale.data = torch.clamp(logit_scale.data, max=100)
#         # if self.rank == 0:
#         #     logger.info("logit_scale:{},dtype:{}".format(logit_scale, logit_scale.dtype))
#         #     logger.info("sequence_output.shape:{}".format(sequence_output.shape))
#         #     logger.info("visual_output.shape:{}".format(visual_output.shape))
#         if len(visual_output.shape) == 2:
#             retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
#         else:
#             visual_temp = visual_output.permute(0, 2, 1)
#             retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_temp)
#             retrieve_logits = retrieve_logits.permute(1, 0, 2)
#
#         return retrieve_logits
#
#     def forward(self, video_data, video_frame, tag_ids, tag_mask, title_ids, title_mask, global_step):
#         tag_ids = tag_ids.view(-1, tag_ids.shape[-1])
#         tag_mask = tag_mask.view(-1, tag_mask.shape[-1])
#         title_ids = title_ids.view(-1, title_ids.shape[-1])
#         title_mask = title_mask.view(-1, title_mask.shape[-1])
#         # bs x frames x 3 x H x W
#         video = torch.as_tensor(video_data)
#
#         if self.rank == 0 and global_step % self.task_config.n_display == 0:
#             logger.info("video1.shape:{}, dtype:{}, device:{}".format(video.shape, video.dtype, video.device))
#
#         if self.training:
#             # loss = 0.0
#             v_fea, frame_fea = self.visual_encoder(video, video_frame)
#             if self.task_config.dataset == "bird":
#                 tag_fea = self.text_encoder(tag_ids, tag_mask)
#             title_fea = self.text_encoder(title_ids, title_mask)
#
#             # for video self supervised learning
#             # [bs,hidden_size]
#             # bs, frame, hidden = frame_fea.shape
#             # frame_fea = frame_fea.view(-1, hidden)
#             v_fea = dist_collect(v_fea).squeeze(1)
#             title_fea = dist_collect(title_fea).squeeze(1)
#             frame_fea = dist_collect(frame_fea).squeeze(1)
#             tag_fea = dist_collect(tag_fea).squeeze(1)
#             if self.rank == 0 and global_step % self.task_config.n_display == 0:
#                 logger.info("v_fea.shape:{},device:{}".format(v_fea.shape, v_fea.device))
#                 logger.info("frame_fea.shape:{},device:{}".format(frame_fea.shape, frame_fea.device))
#                 # logger.info("frame_proj.shape:{},device:{}".format(frame_proj.shape, frame_proj.device))
#                 logger.info("title_fea.shape:{}".format(title_fea.shape))
#
#             # compute loss
#             if self.rank == 0 and global_step % self.task_config.n_display == 0:
#                 logger.info("dtype: v_fea:{},title_fea:{}".format(v_fea.dtype, title_fea.dtype))
#             # single video modality: video queue loss
#             loss_FAM = 0.
#             # cross modality: cross queue loss
#             sim_matrix = self.loose_similarity(title_fea, v_fea)
#             v_title_queue_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
#
#             if self.task_config.dataset == "bird":
#                 sim_matrix = self.loose_similarity(tag_fea, v_fea)
#                 v_tag_queue_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
#                 loss_VTM = (v_tag_queue_loss + v_title_queue_loss) / 2
#             else:
#                 loss_VTM = v_title_queue_loss
#
#             loss_FTM = 0.
#             if self.task_config.use_frame_fea:
#                 frame_title_loss = 0.
#                 for i in range(frame_fea.size(1)):
#                     sim_matrix = self.loose_similarity(title_fea, frame_fea[:, i, :])
#                     temp_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
#                     frame_title_loss += temp_loss
#                 frame_title_loss = frame_title_loss / frame_fea.size(1)
#                 if self.task_config.dataset == "bird":
#                     frame_tag_loss = 0.
#                     for i in range(frame_fea.size(1)):
#                         sim_matrix = self.loose_similarity(tag_fea, frame_fea[:, i, :])
#                         temp_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
#                         frame_tag_loss += temp_loss
#                     frame_tag_loss = frame_tag_loss / frame_fea.size(1)
#                     loss_FTM += (frame_tag_loss + frame_title_loss) / 2
#                 else:
#                     loss_FTM = frame_title_loss
#
#             # single text modality: text queue loss
#             # t_queue_loss = self.contrastive_loss(title_fea, tag_fea_k, self.queue_tag_cross_ng) \
#             #                + self.contrastive_loss(tag_fea, title_fea_k, self.queue_v_cross_ng)
#
#             # dequeue_and_enqueue
#             # self._dequeue_and_enqueue(v_fea_k, tag_fea_k, title_fea_k, frame_fea_k, frame_proj_k)
#
#             # mlm loss
#
#             # mlm_title_loss = self.get_mlm_loss(title_ids, title_mask)
#             # if self.task_config.dataset == "bird":
#             #     mlm_tag_loss = self.get_mlm_loss(tag_ids, tag_mask)
#             #     loss_MLM = (mlm_tag_loss + mlm_title_loss) / 2
#             # else:
#             #     loss_MLM = mlm_title_loss
#             loss_MLM = 0.
#             # total loss
#             loss = self.weight_FAM * loss_FAM + self.weight_VTM * loss_VTM + self.weight_FTM * loss_FTM + self.weight_MLM * loss_MLM
#             if self.rank == 0:
#                 if global_step % self.task_config.n_display == 0:
#                     logger.info("loss:{},loss_FAM:{},loss_VTM:{},loss_FTM:{},loss_MLM:{}"
#                                 "".format(loss, loss_FAM, loss_VTM, loss_FTM, loss_MLM))
#                 if self.task_config.logdir:
#                     loss_item = {"loss": float(loss), "loss_FAM": float(loss_FAM), "loss_VTM": float(loss_VTM),
#                                  "loss_FTM": float(loss_FTM), "loss_MLM": float(loss_MLM)}
#                     self.task_config.writer.add_scalars('loss', loss_item, global_step=global_step)
#                     # self.task_config.writer.add_scalar('loss', video_cross_loss, global_step=global_step)
#             return loss
#         else:
#             return None


class BirdModel(BirdPreTrainedModel):
    def __init__(self, cross_config, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.rank = task_config.local_rank
        # self.weight_sim = torch.nn.Parameter(torch.tensor([0.9], dtype=torch.float32), requires_grad=True)
        self.weight_VTM_finetune = cross_config.weight_VTM_finetune
        self.weight_FTM_finetune = cross_config.weight_FTM_finetune
        self.top_frames = task_config.top_frames
        ################## text Encoder
        self.text_encoder = TextEncoder(self.task_config, cross_config)
        ################## visual_encoder
        self.visual_encoder = VisualEncoder(self.task_config, cross_config)
        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

    def frame_loss(self, query_output, frame_output):
        frame_num = frame_output.size(1)
        loss = 0.
        for i in range(frame_num):
            frame_single = frame_output[:, i, :].squeeze()
            sim_matrix = self.loose_similarity(query_output, frame_single)
            sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            loss += sim_loss / frame_num
        # logger.info("frame_output.shape:{},dtype:{}".format(frame_output.shape, frame_output.dtype))
        # logger.info("query_output.shape:{},dtype:{}".format(query_output.shape, frame_output.dtype))
        # sim_matrix = self.loose_similarity(query_output, frame_output)
        # sim_matrix = torch.topk(sim_matrix, k=self.top_frames, dim=2)[0]
        # sim_matrix = torch.mean(sim_matrix, dim=2)
        # sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
        # loss += sim_loss
        return loss

    def forward(self, query_ids, query_mask, video_data, video_frame, idx, global_step):
        query_ids = query_ids.view(-1, query_ids.shape[-1])
        query_mask = query_mask.view(-1, query_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video_data)
        # if self.rank == 0:
        #     logger.info("video.shape:{}, dtype:{}".format(video.shape, video.dtype))
        if self.training:
            loss = 0.0
            query_output = self.text_encoder(query_ids, query_mask)
            visual_output, frame_output = self.visual_encoder(video, video_frame)
            # if self.rank == 0:
            #     logger.info("query_output.shape:{},dtype:{}".format(query_output.shape, query_output.dtype))
            #     logger.info("visual_output.shape:{},dtype:{}".format(visual_output.shape, visual_output.dtype))
            #     logger.info("frame_output.shape:{},dtype:{}".format(frame_output.shape, frame_output.dtype))

            visual_output = dist_collect(visual_output).squeeze(1)
            query_output = dist_collect(query_output).squeeze(1)
            frame_output = dist_collect(frame_output).squeeze(1)
            # print(visual_output.shape)
            # print(query_output.shape)
            # print(frame_output.shape)
            # frame loss
            if self.task_config.use_frame_fea:
                frame_loss = self.frame_loss(query_output, frame_output)
                loss += self.weight_FTM_finetune * frame_loss
            # video loss
            sim_matrix = self.loose_similarity(query_output, visual_output)
            sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            loss += self.weight_VTM_finetune * sim_loss
            # loss += sim_loss

            if self.task_config.local_rank == 0:
                if global_step % self.task_config.n_display == 0:
                    logger.info(
                        "loss:{},frame_loss:{},sim_loss:{},type:{},sim_matrix.shape:{}".format(loss, loss - sim_loss,
                                                                                sim_loss, sim_loss.dtype, sim_matrix.shape))

                if self.task_config.logdir:
                    self.task_config.writer.add_scalar('loss', float(loss), global_step=global_step)
            return loss
        else:
            return None


class BirdModel_VT(BirdPreTrainedModel):
    def __init__(self, cross_config, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.rank = task_config.local_rank
        # self.weight_sim = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float32), requires_grad=True)
        # self.weight_frame = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32), requires_grad=True)
        self.weight_VTM = cross_config.weight_VTM_finetune
        self.weight_FTM = cross_config.weight_FTM_finetune
        ################## text Encoder
        self.text_encoder = TextEncoder(self.task_config, cross_config)
        ################## visual_encoder
        self.visual_encoder = VisualEncoder(self.task_config, cross_config)
        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

    def forward(self, query_ids, query_mask, video_data, video_frame, title_ids, title_mask, idx, global_step):
        query_ids = query_ids.view(-1, query_ids.shape[-1])
        query_mask = query_mask.view(-1, query_mask.shape[-1])
        title_ids = title_ids.view(-1, title_ids.shape[-1])
        title_mask = title_mask.view(-1, title_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video_data)
        if self.training:
            loss = 0.0
            query_output = self.text_encoder(query_ids, query_mask)
            title_output = self.text_encoder(title_ids, title_mask)
            visual_output, frame_output = self.visual_encoder(video, video_frame)

            visual_output = dist_collect(visual_output).squeeze(1)
            query_output = dist_collect(query_output).squeeze(1)
            title_output = dist_collect(title_output).squeeze(1)

            # frame loss
            # if self.task_config.use_frame_fea:
            #     frame_loss = self.frame_loss(query_output, frame_output)
            #     # loss += self.weight_frame * frame_loss
            #
            # # video loss
            # sim_matrix = self.loose_similarity(query_output, visual_output)
            # sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            # loss += self.weight_sim * sim_loss
            sim_loss = 0

            # title loss
            sim_matrix_title = self.loose_similarity(query_output, title_output)
            sim_loss_title = self.loss_fct(sim_matrix_title) + self.loss_fct(sim_matrix_title.T)
            loss += self.weight_title * sim_loss_title

            if self.task_config.local_rank == 0:
                if global_step % self.task_config.n_display == 0:
                    logger.info("sim_loss:{},sim_loss_title:{}".format(sim_loss, sim_loss_title))
                if self.task_config.logdir:
                    loss_item = {"loss": float(loss), "sim_loss": float(sim_loss),
                                 "sim_loss_title": float(sim_loss_title)}
                    # self.task_config.writer.add_scalars('loss', loss_item, global_step=global_step)
                    self.task_config.writer.add_scalar('loss', float(loss), global_step=global_step)
            return loss
        else:
            return None


class MLP(nn.Module):
    def __init__(self, in_dim=512, inner_dim=4096, out_dim=512, num_layers=2):
        super(MLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x
