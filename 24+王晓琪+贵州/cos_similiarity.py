import numpy as np

'''
余弦相似度计算
'''

def cosine_distance(A, B):
    return np.sum(A * B) / np.sqrt(np.sum(np.square(A)) * np.sum(np.square(B)))

#两个一致的向量，夹角为0度，余弦值为1
A = np.array([1,2])
B = np.array([1,2])
print(cosine_distance(A, B))

#两个垂直的向量，夹角为90度，余弦值为0
A = np.array([1,0])
B = np.array([0,1])
print(cosine_distance(A, B))

#向量夹角45度，余弦值(根号2)/2
A = np.array([1,1])
B = np.array([0,1])
print(cosine_distance(A, B))

#任意相同维度的向量都可以计算余弦相似度
A = np.array([1,2,3,4,5])
B = np.array([5,4,3,2,1])
print(cosine_distance(A, B))