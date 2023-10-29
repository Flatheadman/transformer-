import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy


# 1.词向量编码
class Embeddings(nn.Module):
    # d_model: 词向量维度；vocab：词表长度
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 根据指定维度初始化一个embedding的查找表
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    # x: 传入本层的张量数据，一般是词汇表的index数字张量或者one-hot向量张量
    def forward(self, input_data):
        return self.lut(input_data) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
input_data = Variable(torch.LongTensor([
    [100, 2, 421, 587],
    [491, 998, 1, 221]
]))
emb_word_table = Embeddings(d_model, vocab)
emb_word_res = emb_word_table(input_data)
print('emb_word_res: ', emb_word_res, emb_word_res.shape)


# 2.位置编码
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


dropout = 0.1
max_len = 60
position_encoding = PositionEncoding(d_model, dropout, max_len)
position_encoding_res = position_encoding(emb_word_res)
print('position_encoding_res: ', position_encoding_res, position_encoding_res.shape)


# 3.单头自注意力算法
def attention(query, key, value, mask=None, dropout=None):
    # 首先取query最后一个维度的大小。一般情况下等同于词向量的维度
    d_query_last = query.size(-1)
    # 初始化前提是假设 query=key=value=position_encoding_res
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_query_last)
    print('scores:', scores, scores.shape)

    # 判断是否使用掩码张量
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        print('masked_scores:', scores, scores.shape)
    # 对scores的最后一维进行softmax操作，获取注意力评分张量
    atten_scores = F.softmax(scores, dim=-1)
    print('atten_scores:', atten_scores, atten_scores.shape)

    # dropout操作
    if dropout is not None:
        atten_scores = dropout(atten_scores)
        print('dropout atten scores:', atten_scores, atten_scores.shape)
    # 计算结合value后的注意力张量
    atten_res = torch.matmul(atten_scores, value)
    print('atten res:', atten_res, atten_res.shape)
    return atten_res, atten_scores

# query = key = value = position_encoding_res
# atten_res, atten_scores = attention(query, key, value)

query = key = value = position_encoding_res
mask = Variable(torch.zeros(2, 4, 4))
atten_res, atten_scores = attention(query, key, value, mask=mask)


# 4.多头自注意力算法
# 深度复制多个相同的torch module. 深度复制不是简单的指针复制
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % head_num == 0
        # 每个头分到的词向量维度
        self.head_dim = embedding_dim // head_num
        self.head_num = head_num
        self.linears = clones(nn.linear(embedding_dim, embedding_dim), 4)
        self.atten = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size[0]

        zip_data = zip(self.linears, (query, key, value))
        print('zip_data:', zip_data, zip_data.shape)

        modeled_data = [model(x) for model, x in zip_data]
        print('modeled_data:', modeled_data, modeled_data.shape)

        query, key, value = \
            [md.view(batch_size, -1, self.head_num, self.head_dim) for md in modeled_data]
        print('query:', query, query.shape)
        print('key:', key, key.shape)
        print('value:', value, value.shape)

        x, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 这个attention函数有能力处理四维张量吗？
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.head_dim)
        return self.linears[-1](x)


head_num = 8
query = key = value = position_encoding_res
print('position_encoding_res:', position_encoding_res, position_encoding_res.shape)
mask = Variable(torch.zeros(8, 4, 4))
print('mask', mask, mask.shape)
mha = MultiHeadedAttention(head_num, d_model, dropout)
mha_res = mha(query, key, value, mask)
print('mha_res: ', mha_res, mha_res.shape)
