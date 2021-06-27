# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model",
    "corpus_path": "data/corpus.txt",#语料文件路径
    "vocab_path":"data/chars.txt",#字表文件路径
    "max_length": 20 ,# 样本最大长度
    "hidden_size": 100,     # 隐含层维度
    "epoch_num": 10,        # 训练轮数
    "batch_size": 20,    # 每次训练样本个数
    "optimizer": "adam",  # 优化器
    "learning_rate": 1e-3,  # 学习率
    "char_dim": 50,  # 每个字的维度
    "num_rnn_layers": 3    # rnn层数
}




