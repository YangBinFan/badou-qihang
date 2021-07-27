'''
作业：根据词典，输出一段文本所有可能的切割方式
'''


#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

sentence = "经常有意见分歧"


def recur(n,res = []):
    if n == 1:
        return [[0], [1]]
    else:
        return [_+[0] for _ in recur(n-1)] +[_+[1] for _ in recur(n-1)]

def get_split(case,sentense):
    case=case+[1]
    split_point=[0]+[i+1 for i,_ in enumerate(case) if _==1]
    result=[]
    for i in range(len(split_point)-1):
        result.append(sentense[split_point[i]:split_point[i+1]])
    return result

def word_splitcase(freq_dict,sentense):

    split_case=recur(len(sentense)-1)
    all_result=[]
    for case in split_case:
        case_list=get_split(case,sentense)
        indict_bull=True
        for split_item in case_list:
            if split_item not in freq_dict:
                indict_bull=False
        if indict_bull:
            all_result.append(case_list)
    return all_result


"""
预期输出
[['经常', '有意见', '分歧'], 
 ['经常', '有意见', '分', '歧'],
 ['经常', '有', '意见', '分歧'], 
 ['经常', '有', '意见', '分', '歧'], 
 ['经常', '有', '意', '见分歧'], 
 ['经常', '有', '意', '见', '分歧'], 
 ['经常', '有', '意', '见', '分', '歧'], 
 ['经', '常', '有意见', '分歧'], 
 ['经', '常', '有意见', '分', '歧'], 
 ['经', '常', '有', '意见', '分歧'], 
 ['经', '常', '有', '意见', '分', '歧'], 
 ['经', '常', '有', '意', '见分歧'], 
 ['经', '常', '有', '意', '见', '分歧'], 
 ['经', '常', '有', '意', '见', '分', '歧']]
"""

if __name__ == "__main__":
    # aa=recur(n=3)
    word_splitcase(freq_dict=Dict, sentense=sentence)
    pass
    # word_splitcase(freq_dict=Dict, sentense=sentence)
