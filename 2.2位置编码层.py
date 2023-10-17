import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class PositionEncoding(nn.Module):
    # d_model: 词向量维度； dropout: 置零比率；max_len: 句子最大长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoding, self).__init__()

        # 实例化nn中预定义的dropout层，获得self.dropout
        self.dropout = nn.Dropout(p=dropout)

        # 1.初始化一个位置编码矩阵，它是一个0阵，维度为max_len x d_model（50x3）
        position_encode = torch.zeros(max_len, d_model)
        # 2.初始化一个绝对位置矩阵。绝对位置用词汇在句子中的索引表示
        abs_position = torch.arange(0, max_len)
        abs_position = abs_position.unsqueeze(1)

        '''
        3.对绝对位置编码，转化为和词向量维度相同的向量
        怎样将绝对位置编码和原本的词向量融合。
        最简单就是直接相加。由于每个词向量是3维的，所以需要先把绝对位置索引转化为3维向量。需要一个1xd_model的变换矩阵
        该变换矩阵最好能将自然数索引缩放为足够小的数字，这样有助于sygmoid等激活函数获得有效值
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))
        position_encode[:, 0::2] = torch.sin(abs_position * div_term)
        position_encode[:, 1::2] = torch.cos(abs_position * div_term)

        # 4.将二维张量扩充为3维张量，以匹配序列组形式的输入数据
        position_encode = position_encode.unsqueeze(0)

        # 5.注册buffer
        self.register_buffer('position_encode', position_encode)

    def forward(self, input_data):
        # 由于位置编码的max_len是过饱和的，实际应用中只需要截取实际序列的长度即可
        input_data = input_data + Variable(self.position_encode[:, :input_data.size(1)], requires_grad=False)
        return self.dropout(input_data)

d_model = 512
dropout = 0.1
max_len = 60

