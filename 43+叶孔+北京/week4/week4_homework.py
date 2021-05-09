#!/usr/bin/env python3
#coding: utf-8

import numpy as np
import pandas as pd
import random
import sys
import time,json,jieba
from collections import Counter
from tkinter import _flatten

'''

1.距离：采用jaccard距离
2.中心点的更新：
        1.将每个簇topn词频作为中心，但可能难以收敛，这样造成有些句子和任何一个簇相似度为0，
3.停止条件，上一个中心点和下一个中心点的jaccard相似度大于0.5
'''

def delete_stop_word(cut_word,stop_word):
    data = []
    for i in cut_word :
        if i == ' ':
            continue
        if i not in stop_word:
            data.append(i)
    return data


def load_data(file):
    data = []
    stop_word =[]
    for stopword in open("stop_words", encoding="utf8").readlines():
        stop_word.append(stopword.strip())


    with open(file, encoding="utf8") as f:
        for line in f:
            title = jieba.lcut(json.loads(line.strip())["title"])
            # print(title)
            title = delete_stop_word(title,stop_word)
            # print(title)
            data.append(title)

    return data



class KMeansClusterer:
    # ndarray 为2维数组，句子词数*句子个数
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)
        self.epoch = 0

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray: # 遍历每个句子
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)): # 遍历中心点
                distance = self.__distance(item, self.points[i])
                print(distance)
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item]
        new_center = []
        # 更新中心点
        for item in result:
            new_center.append(self.__center(item))
        # 中心点未改变，说明达到稳态，结束递归
        flag =True
        for i,j in zip(self.points,new_center):
            # print(i,j)
            if len(i)!= 0 and len(j)!=0:
                print(len(set(i) & set(j)),len(set(i).union(set(j))))
                if len(set(i) & set(j)) / len(set(i).union(set(j))) < 0.5:
                    flag = False

        if flag !=False:
            return result

        self.points = np.array(new_center)
        print(self.epoch)
        self.epoch +=1
        return self.cluster()

    def __center(self, data_ite,choose_all=False):
        '''
        生成的中心应该能计算distance,生成的应是一组字符串：可以采用的
        1.利用最高词频的top100作为中心（可能无法收敛）
        '''
        #
        if choose_all ==False:
            data = []
            for word in data_ite:
                data.append(word)
            # print(data)
            data = list(_flatten(data))
            data_len = len(data)
            # print(data_len)
            sort_word = Counter(data).most_common(500)
            out = []
            for i in sort_word:
                out.append(i[0])
            return out
        else:
            data = []
            for word in data_ite:
                data.append(word)
            data = list(set(_flatten(data)))
            return data


    def __distance(self, p1, p2):  # jaccard距离
        '''计算两句子间距
        '''
        return len(set(p1) & set(p2)) / len(set(p1).union(set(p2)))

    # 初始化句子
    def __pick_start_point(self, ndarray, cluster_num):

        if cluster_num < 0 or cluster_num > len(ndarray):
            raise Exception("簇数设置有误")

        # 随机点的下标
        indexes = random.sample(np.arange(0, len(ndarray), step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index])
        return np.array(points)


'''
基于sklearn实现
'''

# from sklearn.cluster import KMeans


file = "tag_news.json"
sentences = load_data(file)
print(len(sentences))
kmeans = KMeansClusterer(ndarray=sentences, cluster_num=5).cluster()
# print(kmeans)
#
# kmeans.predict([[0, 0], [12, 3]])
#
# print(kmeans.cluster_centers_)