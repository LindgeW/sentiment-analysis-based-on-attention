import sys
sys.path.append(["../../", "../", "./"])
import torch
from text_utils import TextUtils
import numpy as np
from collections import Counter
from vocab import Vocab


class Instance(object):
    def __init__(self, words, tag):
        self.words = words  # 单词序列（经过分词）
        self.tag = tag      # 标签

    def __str__(self):
        return str(self.tag) + '\t' + ' '.join(self.words)


# 预测：对传入的数据列表进行预处理
def preprocess_data(input_list):
    insts = []
    for line in input_list:
        line = TextUtils.remove_blank(line)
        token_seq = TextUtils.tokenize(line)
        insts.append(Instance(token_seq, tag=-1))
    return insts


# 以Instance对象的形式保持数据和标签
def load_data_instance(file):
    insts = []
    with open(file, 'r', encoding='utf-8', errors='ignore') as fin:
        for example in fin:
            lbl, sent = example.split('|||')
            sent = sent.strip().split()
            lbl = int(lbl.strip())
            insts.append(Instance(sent, lbl))
    np.random.shuffle(insts)  # 初步随机化
    return insts


# 根据训练集数据创建Vocab
def create_vocab(corpus_path, lexicon_path=None):
    wds_counter = Counter()
    tags_counter = Counter()
    insts = load_data_instance(corpus_path)
    # 统计语料中词频和标签数量
    for inst in insts:
        for wd in inst.words:
            wds_counter[wd] += 1
        tags_counter[inst.tag] += 1

    return Vocab(wds_counter, tags_counter, lexicon_path)


# 产生batch
def get_batch(data, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(data)

    nb_batch = int(np.ceil(len(data) / float(batch_size)))
    for i in range(nb_batch):
        batch_data = data[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)
        yield batch_data


# 将原始数据转换成Tensor
def batch_data_variable(batch_data, vocab):
    batch_size = len(batch_data)
    max_len = max([len(inst.words) for inst in batch_data])

    seq_lens = []
    corpus_idxs = torch.zeros(batch_size, max_len, dtype=torch.long)
    wd2vec_idxs = torch.zeros(batch_size, max_len, dtype=torch.long)
    att_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    tags = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len)

    for i, inst in enumerate(batch_data):
        seq_len = len(inst.words)
        seq_lens.append(seq_len)
        corpus_idxs[i, :seq_len] = torch.LongTensor(vocab.corpus_wd2idx(inst.words))
        wd2vec_idxs[i, :seq_len] = torch.LongTensor(vocab.word2index(inst.words))
        tags[i] = vocab.tag2index(inst.tag)
        att_ids[i, :seq_len] = torch.LongTensor(vocab.lexicon_vec(inst.words))
        mask[i, :seq_len].fill_(1.)
    return corpus_idxs, wd2vec_idxs, tags, att_ids, mask
