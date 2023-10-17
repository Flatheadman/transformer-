import torch

# 创建一个 2x3 的张量
tensor1 = torch.tensor([
    [1],
    [4]
])

print(tensor1)

# 创建一个 2x3 的张量
tensor2 = torch.tensor([5,6,7])

print(tensor2)

res = tensor1 * tensor2
print(res, 123)
