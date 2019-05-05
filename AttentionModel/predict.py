import sys
sys.path.append(["../../", "../", "./"])
import torch
import torch.nn.functional as F
import data
import config.HyperConfig as Config
import pickle
torch.manual_seed(3347)


def highlight_word(wd, **color):
    return '<span style="background-color: rgb(%d, %d, %d)">%s</span>' % (color['r'], color['g'], color['b'], wd)


def highlight_seq(wd_seq, idxs, weights, pred_no):
    marked_seq = []
    for i, wd in enumerate(wd_seq):
        if i in idxs:
            w = int(255 * (1 - weights[i]))
            color = {}
            if pred_no == 0:
                color = {'r': w, 'g': w, 'b': 255}
            elif pred_no == 1:
                color = {'r': w, 'g': 255, 'b': w}
            elif pred_no == 2:
                color = {'r': 255, 'g': w, 'b': w}

            wd = highlight_word(wd, **color)

        marked_seq.append(wd)

    return ''.join(marked_seq)


def load_vocab(load_vocab_path):
    with open(load_vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    return vocab


def predict(pred_data, vocab, config):
    # classifier = Attention_LSTM_SA.Attention_LSTM_SA()
    # classifier.load_state_dict(torch.load(config.load_model_path))  # 加载模型参数

    # 加载模型
    classifier = torch.load(config.load_model_path)

    classifier.eval()  # self.training = False

    insts = data.preprocess_data(pred_data)  # 单词序列

    corpus_idxs, wd2vec_idxs, _, _, seq_lens, origin_indices = data.batch_data_variable(insts, vocab)

    out, weights = classifier(corpus_idxs, wd2vec_idxs, seq_lens)
    pred = torch.argmax(F.softmax(out, dim=1), dim=1)
    weights = weights[origin_indices]
    pred = pred[origin_indices]
    print(pred)

    topn = 5
    results = []
    # top_weights, top_idxs = torch.topk(weights, topn, dim=1)
    # for ws, idxs, wd_seq, pred_no in zip(top_weights, top_idxs, wd_seqs, pred.numpy()):
    #     hs = highlight_seq(wd_seq, idxs.data.numpy(), ws.data.numpy(), pred_no)
    #     print(hs)
    #     results.append(hs)

    for inst, ws, pred_no in zip(insts, weights, pred):
        _, top_idxs = torch.topk(ws, topn)
        hs = highlight_seq(inst.words, top_idxs.numpy(), ws.data.numpy(), pred_no)
        print(hs)
        results.append(hs)

    return results


if __name__ == '__main__':

    X = [
        '鞋子很快就收到了，质量过关，价格不贵，性价比 很高，鞋子穿得也比较舒服，尺码也标准就按平时 自己穿得码数买就行了！',
        '质量很好 版型也很好 码子很标准 穿上很有档次 卖家服务超级好 很满意的一次网上购物',
        '发过来的鞋子跟图片不是同一款 没有图片上的好看 鞋子的鞋面跟鞋带都不一样 只有鞋底一样 太坑了 并且物流不是一般的慢',
        '鞋子感觉一般吧，穿上不是特别舒服，这个价钱中规中矩吧。'
    ]

    config = Config.Config('config/hyper_param.cfg')
    vocab = load_vocab(config.load_vocab_path)
    predict(X, vocab, config)

