
import torch
from loader import build_vocab,sentence_to_sequence
from model import TorchModel
class Evaluator:
    def __init__(self,logger):
        self.logger = logger

    def eval(self, config,input_strings,model_path):
        #配置保持和训练时一致

        vocab = build_vocab(config["vocab_path"])       #建立字表
        model = TorchModel(config)   #建立模型
        model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
        model.eval()
        for input_string in input_strings:
            #逐条预测
            x = sentence_to_sequence(input_string, vocab)
            with torch.no_grad():
                result = model.forward(torch.LongTensor([x]))[0]
                result = torch.argmax(result, dim=-1)  #预测出的01序列
                # 在预测为1的地方切分，将切分后文本打印出来
                for index, p in enumerate(result):
                    if p == 1:
                        print(input_string[index], end=" ")
                    else:
                        print(input_string[index], end="")
                print()