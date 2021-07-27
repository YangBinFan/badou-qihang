# 批归一化层   batch Normalization
# 在torch里面有一个实现
import numpy as np
import torch
import math
import torch.nn as nn


#
# class batchnorm(nn.Module):
#     def __init__(self, init1 = 10):
#         super(batchnorm, self).__init__()
#         self.bn = nn.BatchNorm1d(init1)
#     def forward(self, x):
#         return self.bn(x)

class Diymodel:
    def __init__(self,weight, bias):
        self.weight = weight
        self.bias = bias
    def forward(self, x):
        x = x.numpy()
        n,m = x.shape
        y_ = []
        for i in range(m):
            mu = sum([x[j][i] for j in range(n)]) / n
            val =  sum([(x[j][i]-mu) ** 2 for j in range(n)]) / n
            y_.append([self.weight[i] * (x[j][i] - mu) / math.sqrt(val) + self.bias[i] for j in range(n)])
        return np.array(y_).T

class Diymodel1:
    def __init__(self,weight, bias):
        self.weight = weight
        self.bias = bias
    def forward(self, x):
        x = x.numpy()
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        y = (x - mean) / std * self.weight + self.bias    # np.around(y, 4)
        return y



bn = torch.nn.BatchNorm1d(10)
# print(bn.state_dict())
weight,bias = bn.state_dict()['weight'].numpy(), bn.state_dict()['bias'].numpy()
# 随机选择一个输入
x = torch.randn(2,10)
print(x)
print(bn(x), "bn的结果")

# Diymodel1:
bn1 = Diymodel(weight, bias)
output1 = bn1.forward(x)
print(output1, "Diy1的结果")


# diymodel2
bn2 = Diymodel1(weight, bias)
output2 = bn2.forward(x)
print(output2, "Diy2的结果")

