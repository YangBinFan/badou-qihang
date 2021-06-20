from gensim.summarization import bm25
import jieba
import json
import numpy as np
from ast import literal_eval

class QA_model():
    def __init__(self, train_f,valid_f,data_f,schema_f):

        with open('./../data/schema.json','r',encoding='utf8')as fp:
            self.schema_f = json.load(fp)

    def train_model(self,data):
        self.train_data,self.train_id = data_load(data)
        self.bm25Model = bm25.BM25(self.train_data)

    def predict(self,data,id):
        for i,j in zip(data,id):
            score = self.bm25Model.get_scores(i)
            index = np.argmax(np.array(score))
            print("当前句子是：",i)
            print("得分最高的分数：",score[index],"###得分最高的句子：",self.train_data[index])
            print("###真实target：",j,"###对应schema_id：",self.schema_f[j])
# def data_load(self,train_f,valid_f,data_f,schema_f):
    #     self.train_data = []
    #     with open(train_f,"r") as f:
    #         for row in f:
    #             jieba.cut("row")
    #     self.valid_data =
    #     self.data_data =
    #     self.schema_data =


def data_load(file):
    with open(file,encoding="utf-8") as f:
        d_data,d_id,index = [],[],0
        for row in f:
            data_cur = json.loads(row.strip())
            for i in data_cur["questions"]:
                d_data.append(jieba.lcut(i))
                d_id.append(data_cur["target"])
    return d_data,d_id
def valid_data_load(file):
    with open(file,encoding="utf-8") as f:
        d_data,d_id = [],[]
        for row in f:
            data_cur = literal_eval(row)
            d_data.append(jieba.lcut(data_cur[0]))
            d_id.append(data_cur[-1])
    return d_data,d_id

if __name__ == '__main__':
    train_f,valid_f,data_f,schema_f = "./../data/train.json","./../data/valid.json","./../data/data.json","./../data/schema.json"
    model = QA_model(train_f,valid_f,data_f,schema_f)
    model.train_model("./../data/train.json")
    valid_data,valid_id = valid_data_load(valid_f)
    model.predict(valid_data,valid_id)

# 部分输出：
# 当前句子是： ['其他', '业务']
# 得分最高的分数： 5.076249324469579 ###得分最高的句子： ['办理', '业务']
# ###真实target： 宽泛业务问题 ###对应schema_id： 2
# 当前句子是： ['手机', '信息']
# 得分最高的分数： 4.989474458470932 ###得分最高的句子： ['怎么', '发', '信息', '查', '电话费']
# ###真实target： 宽泛业务问题 ###对应schema_id： 2