import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import logging
import pandas as pd
import re
import numpy as np

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)

logger = logging.getLogger(__name__)

def test1():
    # [seq_len, batch_size, hidden_size]
    h = torch.rand(10, 3, 5)
    print(h.shape)
    x = torch.tanh(h)
    print(x.shape)
    s = h.sum(dim=0)
    print(s.shape)
    z = s * x
    print(z)

    S = 0
    for time in range(h.size(0)):
        S += h[time] * x
    print(S == z)


def test2():
    x = torch.arange(24).reshape(2, 4, 3)
    print(x)
    print(x.sum(dim=1))
    print(x.sum(dim=0))


def test3():
    x = torch.tensor([[1, 2, 3],
                      [0, 1, 4],
                      [1, 1, 1]], dtype=torch.float)
    y = torch.ones(3, 3)
    z = torch.arange(9).reshape(3, 3).float()
    print(x*z + y*z)
    print((x+y)*z)
    print((x*z+y*z) == (x+y)*z)

# def test4():
#     logging.info("ok")

def test5():
    df = pd.read_csv('data/1.txt', header=None, sep='|||')
    # print(df.count())
    print(df.head())

def test6():
    with open("data/char_corpus.txt", encoding='utf-8', errors='ignore') as fin:
        with open('data/char_tokens.txt', 'a', encoding='utf-8') as fout:
            for line in fin:
                line = re.sub(r'\s+', '', line)
                fout.write(' '.join(list(line)))
                fout.write('\n')

def test7():
    logger.info("hello world")

def test8():
    x = np.random.randint(1, 50, 10)
    y = list(range(10))
    print(x, '\n', y)
    z = list(zip(x, y))  # 打包
    z.sort(key=lambda t: t[0], reverse=True)  # 指定取待排序元素的哪一项进行排序
    print(z)
    a, b = zip(*z)  # 与zip相反，zip*可理解为解压
    print(a, '\n', b)

    nums = ['flower', 'flow', 'flight']
    for i in zip(*nums):
        print(i)

class Instance(object):
    def __init__(self, wds, tag):
        self.wds = wds
        self.tag = tag
    def __str__(self):
        return ''.join(self.wds) + '|' + str(self.tag)

def test9():
    insts = [
        Instance(['我', '是', '天大人'], 1),
        Instance(['可恶'], 0),
        Instance(['它', '走了'], 0),
        Instance(['我', '爱', '我', '的', '祖国'], 1)
    ]

    # sorted(insts, key=lambda s: s.tag)
    insts.sort(key=lambda s: len(s.wds), reverse=True)

    for i in insts:
        print(i.wds, i.tag)


def test10():
    inputs = torch.randn(3, 64, 8)
    print(inputs.shape)  # [3, 64, 8]
    inputs.transpose_(1, 2)
    print(inputs.shape)
    apool = nn.AdaptiveMaxPool1d(output_size=1)
    output = apool(inputs)
    print(output.shape)  # [3, 64, 1]


def test11():
    # x = torch.arange(24).float().reshape((2, 3, 4))
    # print(x)
    # print(F.softmax(x[0][0] * x[0]))

    # for b in range(2):
    #     for s in range(3):
    #         t = x[b][s] * x[b]
    #         print(t, F.softmax(t, dim=1))

    y = torch.arange(12).float().reshape(3, 4)
    print(y)
    print(F.softmax(y[0] * y))


def test12():
    x = torch.arange(24).float().reshape(2, 4, 3)
    print(x)
    print(x[-1], x[-2])
    print(torch.cat((x[-1], x[-2]), dim=1))


def test13():
    x = torch.arange(12).float().reshape(3, 4)
    print(x)
    print(F.softmax(x, dim=0))
    print(F.softmax(x, dim=1))
