import json
import jieba
import numpy as np
from gensim.summarization import bm25

class QASystem:
    def __init__(self, know_base_path):
        self.load_know_base(know_base_path)
        self.train()
        print("模型训练完毕，可以开始问答！")

    def load_know_base(self, know_base_path):
        self.corpus = []
        self.index_to_target = {}
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.corpus.append([])
                for question in questions:
                    self.corpus[index] += jieba.lcut(question)
                self.index_to_target[index] = target
        return

    def train(self):
        self.model = bm25.BM25(self.corpus)
        return

    def query(self, question):
        words = jieba.lcut(question)
        scores = self.model.get_scores(words)
        index = np.argmax(scores)
        return self.index_to_target[index]

if __name__ == '__main__':
    qas = QASystem("data/train.json")
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")
