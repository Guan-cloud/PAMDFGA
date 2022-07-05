# -*- coding:utf-8 -*-
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/7 15:16
# @Author :
# @File : alignment.py
# @Function:(1)获取每个batch的attention
#           (2)对attention 进行聚类
#           (3)提取attentoin的模式
#           (4)将attention模式与实例级别的相似性进行对齐
#           (5)对attention patterns进行分解，确保不同的pattern表示不相关

import torch
import torch.nn.functional as F


def attention_diversity(attention, tau=2.0):
    '''
    :param attention: [batch_size, head_num, seq_len]
    :return:
    '''
    # 先对特征进行归一化处理
    attention = F.normalize(attention, p=2, dim=-1)
    coef_mat = torch.bmm(attention, torch.transpose(attention, 2, 1))  # [batch_size, head_num, head_num] # [4 144 144]
    coef_mat.div_(tau)
    a = torch.arange(coef_mat.size(1), device=coef_mat.device)
    a_batch = a.repeat(coef_mat.size(0), 1) #[batch_size, head_num]
    coef_mat = coef_mat.reshape(-1, coef_mat.size(-1))  # [batch_size*head_num, head_num]
    a_batch = a_batch.reshape(-1)  # [batch_size*head_num]
    ad_loss = F.cross_entropy(coef_mat, a_batch)
    return ad_loss





