import numpy as np
import json
import os
from collections import defaultdict


class Kmeanscluster:
    def __init__(self,cluster_num,data):
        self.num_cluster=cluster_num
        self.data=data
    def __init_center(self,data):
        return np.random.choice(data,self.num_cluster,replace=False)
    def get_center(self,kind_dict):
        # new_center=[0]*self.num_cluster
        new_center=[]
        # print(set(list(kind_dict.values())))
        for i in range(self.num_cluster):
            kind_i=[_ for _ in kind_dict if kind_dict[_]==i]
            if len(kind_i)==0:
                new_center.append("")
                continue
            # print(len(kind_i))
            dis_sum_list=[sum([self.dis_computer(i,j) for i in kind_i]) for j in kind_i]
            # print(dis_sum_list,dis_sum_list.index(min(dis_sum_list)))
            new_center.append(kind_i[dis_sum_list.index(min(dis_sum_list))])
        return new_center
    def get_kind_dict(self,center):
        kind_dict = defaultdict(int)
        for _, item in enumerate(self.data):
            dis_list = [self.dis_computer(item, _) for _ in center]
            kind_dict[item] = dis_list.index(min(dis_list))
        return kind_dict
    def update_center(self):
        init_center=[_ for _ in self.__init_center(self.data)]
        # new_center = list(init_center)
        kind_dict=self.get_kind_dict(init_center)
        new_center=self.get_center(kind_dict)
        while new_center!=init_center:
            init_center=new_center
            kind_dict=self.get_kind_dict(new_center)
            new_center=self.get_center(kind_dict)
        for i in range(self.num_cluster):
            kind_i=[_ for _ in kind_dict if kind_dict[_]==i]
            print(kind_i)
        return kind_dict
    def dis_computer(self,item1,item2):
        dis=1-len(set(item1).intersection(set(item2)))/len(set(item1).union(set(item2)))
        return dis

def json_load(json_file):
    json_list=[]
    with open(json_file,"r",encoding="utf-8") as jf:
        # line=jf.readline()
        for line in jf:
            json_dict=json.loads(line)
            json_list.append(json_dict)
            # line=jf.readline()
    return json_list

if __name__ == "__main__":
    json_data=json_load(json_file=r"H:\badou_learning\八斗ai启航班\第四周\PPT\词向量和文本向量\tag_news.json")
    # pass
    kmeans=Kmeanscluster(cluster_num=50,data=[_["title"] for _ in json_data])
    kmeans.update_center()