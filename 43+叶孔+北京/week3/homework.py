# -*- coding:utf-8 -*-

dict = { "经常":0.1,
         "经":0.05,
         "有":0.1,
         "常":0.001,
         "有意见":0.1,
         "歧":0.2,
         "意见":0.2,
         "分歧":0.2,
         "见":0.05,
         "意":0.05,
         "见分歧":0.05,
         "分":0.1

}
sentence1 = "经常有意见分歧"
max_len = len(sentence1)
result = []
# current代表分词的句子
# sentence 代表未划分的句子
# i代表当前节点
def cut_sentence_point(last,current_i,sentence,score):
    current = ""
    if current_i >= max_len:
        print(last, score)
        return
    for i,stri in enumerate(sentence):
        current = current+stri
        if current in dict:
            cut_sentence_point(last + " " + current,current_i+i+1,sentence[i+1:],score +dict[current])

cut_sentence_point("",0,sentence1,0.0)



