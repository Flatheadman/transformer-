import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class Embeddings(nn.Module):
    # d_model: 词向量维度；vocab：词表长度
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 根据指定维度初始化一个embedding的查找表
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    # x: 传入本层的张量数据，一般是词汇表的index数字张量或者one-hot向量张量
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
