'''
掩码张量是一个元素值只能取1或0的张量，代表是否遮掩。
注意力的计算理论上会涉及到一个句子的所有部分。但实际中我们在生成句子中的一个词汇时，只希望模型参考当前位置之前的词汇，那么后面的词汇就需要使用掩码遮掩起来；
并且每生成一个词汇，掩码的位置就需要往后移动一位
'''

import numpy as np
import torch
from torch.autograd import Variable


# np.triu函数表示将主对角线以下的元素置零。-1表示主对角线下移一行，1表示主对角线上移一行
def triu_show(pos):
    print(np.triu([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], pos))

triu_show(0)
triu_show(-1)
triu_show(1)


# 生成向后遮掩的掩码张量
def subsequent_mask(size):
    # 指定掩码张量的形状
    mask_shape = [1, size, size]
    # 用1元素填充张量
    mask = np.ones(mask_shape)
    # 用triu函数斜切张量
    mask = np.triu(mask, 1)
    # 调整数据类型
    mask = mask.astype('uint8')
    # 将张量进行0-1反转
    mask = 1 - mask
    # 将np张量格式转化为torch张量
    mask = torch.from_numpy(mask)
    print(mask)
    return mask

subsequent_mask(5)

# 掩盖效果演示
input = Variable(torch.randn(5, 5))
print("input:", input)
mask = Variable(torch.zeros(5, 5))
print('mask:', mask)
masked_input = input.masked_fill(mask == 0, -1e9)
print('masked input', masked_input)
