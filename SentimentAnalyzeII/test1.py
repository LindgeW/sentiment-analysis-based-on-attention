import torch
from collections import Counter
import logging
import pandas as pd


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


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
def test4():
    logging.info("ok")

def test5():
    df = pd.read_csv('data/1.txt', header=None, sep='|||')
    # print(df.count())
    print(df.head())