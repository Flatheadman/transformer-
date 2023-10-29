import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


'''
自注意力算法的数学原理
    至于为什么query和key的矩阵乘法可以算出注意力评分？参照语雀笔记：
    https://www.yuque.com/kevin-haytk/ghaoho/bsc2bs4smtg9wo2i

'''
def attention(query, key, value, mask=None, dropout=None):
    # 首先取query最后一个维度的大小。一般情况下等同于词向量的维度
    d_query_last = query.size(-1)
    # 初始化前提是假设 query=key=value=position_encoding_res
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_query_last)

    # 判断是否使用掩码张量
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作，获取注意力评分张量
    atten_scores = F.softmax(scores, dim=-1)

    # dropout操作
    if dropout is not None:
        atten_scores = dropout(atten_scores)

    # 计算结合value后的注意力张量
    atten_res = torch.matmul(atten_scores, value)

    return atten_res, atten_scores