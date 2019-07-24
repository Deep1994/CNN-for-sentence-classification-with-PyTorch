# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 20:04:42 2019

"""

import argparse
import utils
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from model import CNN
import torch.optim as optim
import torch.nn as nn
from sklearn.utils import shuffle
from torch.autograd import Variable
import copy
import torch

def train(data, params):
    if params["MODEL"] !="rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        
        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
                
        # one for UNK and one for zero padding 
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
        
    model = CNN(**params).cuda(params["GPU"])
    
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    # 注意这里的损失函数已经帮我们做了softmax了
    criterion = nn.CrossEntropyLoss()
    
    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
            
            
            # [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent)
            # 前面是此表最后一个词的索引，倒数第二个是UNK，倒数第一个是0(for padding),其实就是在做padding
            # 下面两步其实是向量化 + padding + 取batch 操作
            batch_x = [[data["word_to_idx"][w] for w in sent] +   
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent)) 
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
            
            # 注意, 一定转变成tensor再输入
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])
            
            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()
            
        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)
        
        # pre_dev_acc存储了前一个epoch模型在dev上的准确率
        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc
            
        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)
            
    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model
    

def test(data, model, params, mode="test"):
    model.eval()
    
    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]
    
    # 为什么这里要加一个UNK的处理方式(else params["VOCAB_SIZE"]，params["VOCAB_SIZE"]代表UNK的在词表里的索引)
    # 但是上面的训练过程中没有对训练集做同样处理，为什么？
    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] + 
    [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent)) for sent in x]
    
    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    
    return acc
    
    
def main():
    parser = argparse.ArgumentParser(description="******CNN for text classification******")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
    # python run.py save_model属性值是False
    # python run.py --save_model save_model属性值是True, 注意这里没有给save_model赋值，就是说你想保存模型了就加上--save_model
    parser.add_argument("--save_model", default=False, action="store_true", help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action="store_true", help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate" )
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")
    
    options = parser.parse_args()
    data = getattr(utils, f"read_{options.dataset}")()
    
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    
    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu 
    }
    
    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)
    
    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda(params["GPU"])
        
        test_acc = test(data, model, params)
        print("test acc:", test_acc)
        
if __name__ == "__main__":
    main()