# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:51:34 2019

For data loading、save and load model.

"""

from sklearn.utils import shuffle
import pickle

def read_TREC():
    data = {}
    
    def read(mode):
        x, y = [], []
        
        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])
        
        # 这里的shuffle是对应的元素一起做的吗？会把对应的x, y 搞乱吗？
        x, y = shuffle(x, y)
        
        if mode == "train":
            # // 取整除 - 返回商的整数部分（向下取整）
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            # 这里为什么是这样？如果mode是test的话，那岂不是所有的数据作为测试集？当我没说 = =， train和test需要传的参数
            data["test_x"], data["test_y"] = x, y
            
    read("train")
    read("test")
    
    return data

def read_MR():
    data = {}
    x, y = [], []
    
    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-10]
            x.append(line.split())
            y.append(0)
            
    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9
    
    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx, test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]
    
    return data

def save_model(model, params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")

def load_model(params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    
    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")
        
        return model
    except:
        print(f"No available model such as {path}.")
        exit()