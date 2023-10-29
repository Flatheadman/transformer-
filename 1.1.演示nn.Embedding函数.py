import torch
import torch.nn as nn

from torch.autograd import Variable
import math


# 构建一个长度为10，维度为3的词向量查找表
embedding = nn.Embedding(10, 3)
print(embedding)
# 构建2个长度为4的输入序列组
input_seq = torch.LongTensor([
    [5, 6, 8, 2],
    [1, 4, 6, 3]
])
# 将序列组转化为词向量。每个词都被转化为了一个3维向量。
words_vec = embedding(input_seq)
print(words_vec)
print('end')

