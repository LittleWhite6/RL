from __future__ import print_function
import torch

x = torch.empty(5, 3)    #创建一个没有初始化的矩阵
print(x)


x = torch.rand(5, 3) #创建一个随机初始化矩阵
print(x)


x = torch.zeros(5, 3, dtype=torch.long) #构造一个填满0且数据类型为long的矩阵
print(x)


x = torch.tensor([5.5, 3])  #直接从数据构造张量
print(x)


x = x.new_ones(5, 3, dtype=torch.double)    #new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)  #重载dtype!
print(x)


print(x.size()) #获取张量的形状：


