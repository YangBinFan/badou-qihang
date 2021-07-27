# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import build_dataset,build_vocab


import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    cuda_flag = True  # True
    data_loader = build_dataset(config)  #建立数据集
    model = TorchModel(config)   #建立模型
    optim = choose_optimizer(config, model)    #建立优化器
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #训练开始
    for epoch in range(config["epoch_num"]):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            if cuda_flag:
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    #保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(build_vocab(config["vocab"]), ensure_ascii=False, indent=2))
    writer.close()
    return



if __name__ == "__main__":
    # main()
    # print(jieba.lcut("今天天气不错我们去春游吧"))
    # print(sequence_to_label("今天天气不错我们去春游吧"))
    # print(sentence_to_sequence("今天天气不错我们去春游吧"))
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]

    # main(Config)
    evaluator = Evaluator(logger)
    evaluator.eval(Config, input_strings,"./model/model.pth")