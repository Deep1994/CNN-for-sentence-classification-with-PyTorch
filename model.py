# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 20:06:21 2019

The implementation of paper <Convolutional Neural Networks for Sentence Classification> by Yoon Kim.
The paper is available at: https://arxiv.org/abs/1408.5882
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Question:
    1. **kwargs的作用
    2. __init__的作用
    3. super()的作用
    4. self是指？
    5. FILTERS 和 FILTER_NUM 的区别
    答： FILTERS 和 FILTER_NUM都是列表，分别存放着比如[3, 4, 5]和[100, 100, 100]
    6. setattr(object, name, value), setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是存在的。
    7. torch.cat()按行还是按列拼接
"""
class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        
        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1
        
        assert (len(self.FILTERS) == len(self.FILTER_NUM))
        
        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2
        
        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], kernel_size=self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)
            
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        
    def get_conv(self, i):
        return getattr(self, f'conv_{i}')
    
    def forward(self, inp):
        # 将(batch_size, MAX_SENT_LEN, WORD_DIM)reshape成1通道，即(batch_size, 1, WORD_DIM * self.MAX_SENT_LEN)
        # 方便cnn的输入
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1) # 1表示按列拼接
        # pooling的步长是正好是整个feature map，因此变成了一维   
        conv_results = [
                F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1) 
                .view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        
        x = torch.cat(conv_results, 1)
        # training: apply dropout if is ``True``. Default: ``True``
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        
        return x