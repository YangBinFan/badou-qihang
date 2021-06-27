# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from torch.optim import Adam, SGD




class TorchModel(nn.Module):
    def __init__(self, config,):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(config["vocab_size"] + 1, config["char_dim"]) #shape=(vocab_size, dim)
        self.rnn_layer = nn.RNN(input_size=config["char_dim"],
                                hidden_size=config["hidden_size"],
                                batch_first=True,
                                bidirectional=False,
                                num_layers=config["num_rnn_layers"],
                                nonlinearity="relu",
                                dropout=0.1)
        self.classify = nn.Linear(config["hidden_size"], 2)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  #output shape:(batch_size, sen_len, hidden_size)
        y_pred = self.classify(x)   #input shape:(batch_size, sen_len, 2)
        if y is not None:
            return self.loss_func(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
