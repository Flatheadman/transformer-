import torch.nn as nn
import copy


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
        query, key, value = \
            [model(x).view(batch_size, -1, self.head_num, self.head_dim) for model, x in zip(self.linears, (query, key, value))]
        x, self.atten = attension(query, key, value, mask=mask, dropout=self.dropout)

        # 这个attention函数有能力处理四维张量吗？
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.head_dim)
        return self.linears[-1](x)

