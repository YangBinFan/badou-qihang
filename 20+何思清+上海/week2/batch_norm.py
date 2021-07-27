import numpy as np
import torch
from torch import nn


def batch_norm1d(feature,eps=1e-5):
    mean=np.mean(feature,axis=0)
    std=np.std(feature,axis=0)
    result=(feature-mean)/(std+eps)
    # print(result)
    return result

# num_fea=len(feature)
# a=nn.BatchNorm1d(num_features=)

if __name__ == "__main__":
    feature=np.random.randn(4,8)
    result=batch_norm1d(feature)
    a=nn.BatchNorm1d(num_features=8) #,momentum=0,affine=False
    weight=a.state_dict()["weight"].numpy()
    bias=a.state_dict()["bias"].numpy()
    print(a(torch.Tensor(feature)))
    print(result*weight+bias)