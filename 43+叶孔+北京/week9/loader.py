# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import  DataLoader


"""
数据加载
"""


class Dataset:
    def __init__(self, config,):
        self.config = config
        self.vocab = build_vocab(config["vocab_path"])
        self.corpus_path = config["corpus_path"]
        self.max_length = config["max_length"]
        self.load()
        self.config["vocab_size"] = len(self.vocab)
    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                if len(self.data) > 10000:
                    break

    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence

#基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

#建立数据集
def build_dataset(config,):
    dataset = Dataset(config) #diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"]) #torch
    return data_loader


if __name__ == "__main__":
    from config import Config
    dg = Dataset(Config)


