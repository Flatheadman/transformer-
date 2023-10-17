import torch
import torch.nn as nn
from torch.autograd import Variable
import math

max_len = 50
d_model = 4
# 1.初始化一个位置编码矩阵，它是一个0阵，维度为max_len x d_model（50x3）
position_encode = torch.zeros(max_len, d_model)
print(position_encode)

# 2.初始化一个绝对位置矩阵。绝对位置用词汇在句子中的索引表示
abs_position = torch.arange(0, max_len)
# print(abs_position, abs_position.shape)
abs_position = abs_position.unsqueeze(1)
# print(abs_position, abs_position.shape)


'''
3.对绝对位置编码，转化为和词向量维度相同的向量
怎样将绝对位置编码和原本的词向量融合。
最简单就是直接相加。由于每个词向量是3维的，所以需要先把绝对位置索引转化为3维向量。需要一个1xd_model的变换矩阵
该变换矩阵最好能将自然数索引缩放为足够小的数字，这样有助于sygmoid等激活函数获得有效值
'''
# div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))
half_d_model = torch.arange(0, d_model, 2)
print(half_d_model)
scale_param = math.log(10000)/d_model
print(scale_param)
tmp = half_d_model*-scale_param
print(tmp)
div_term = torch.exp(tmp)
print(div_term)

print(abs_position.shape, div_term.shape)
position_encode[:, 0::2] = torch.sin(abs_position * div_term)
print(position_encode)
position_encode[:, 1::2] = torch.cos(abs_position * div_term)
print(position_encode)

# 4.将二维张量扩充为3维张量，以匹配序列组形式的输入数据
position_encode = position_encode.unsqueeze(0)
print(position_encode.shape)

print('end')
