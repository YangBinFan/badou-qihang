import os
import sys
import numpy as np

def cos_similarity(x, y):
    molecular = sum([item[0] * item[1] for item in zip(x, y)])
    denominator = np.sqrt(sum([item[0] * item[1] for item in zip(x, x)])) * np.sqrt(sum([item[0] * item[1] for item in zip(y, y)]))
    return  molecular / denominator


if __name__ == '__main__':
    x = [1, 2]
    y = [1, 2]
    res = cos_similarity(x, y)
    print(res)

# def cosine_similarity(x, y, norm=False):
#     """ 计算两个向量x和y的余弦相似度 """
#     assert len(x) == len(y), "len(x) != len(y)"
#     zero_list = [0] * len(x)
#     if x == zero_list or y == zero_list:
#         return float(1) if x == y else float(0)
#
#     # method 1
#     res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
#     cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
#     print(cos)
#
# if __name__ == '__main__':
#     x = [1, 2]
#     y = [1, 2]
#     res = cosine_similarity(x, y)

