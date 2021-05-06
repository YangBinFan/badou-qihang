import os
import sys
import torch
import torch.nn as nn
import numpy as np

class NP_BN():
    def __init__(self, weight, bias, eps = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(axis = 0)
        x_var = np.mean(np.square(x - x_mean), axis=0)
        u = (x - x_mean) / np.sqrt(x_var + self.eps)
        y_pred = self.weight * u + self.bias
        return y_pred




if __name__ == '__main__':
    BN = nn.BatchNorm1d(4)
    BN_state = BN.state_dict()
    print(BN_state)
    weight = BN_state["weight"].numpy()
    bias = BN_state["bias"].numpy()
    x = torch.randn(2, 4)
    print(BN(x))
    diy_bn = NP_BN(weight, bias)
    print(diy_bn.forward(x.numpy()))
