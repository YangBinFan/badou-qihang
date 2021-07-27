import numpy as np

# 计算模长
np.random.seed(2222)   # 产生随机种子
A = np.random.randint(0, 20, 10)
B = np.random.randint(0, 20, 10)
print("A:", A)
print("B:", B)
# A: [ 1 17  9  6  8  0  4  0 16 17]
# B: [ 6 16 18  4  2 15  9  5  3  2]

# 计算模长
A_ = np.sum(np.sqrt(A ** 2))
B_ = np.sum(np.sqrt(A ** 2))

# 求余弦值 cos(A,B) = A * B / |A|.|B|
cos_AB = (np.sum(A * B)) / (A_ * B_)
print(cos_AB)