import sys
sys.path.append(["../../", "../", "./"])
import torch
import copy
import torch.nn.functional as F
import AttentionModel.data as data
import AttentionModel.config.HyperConfig as Config
torch.manual_seed(3347)


# def highlight_word(wd, w):
#     return '<span style="background-color: rgb(255, %d, %d)">%s</span>' % (round(255*(1-w)), round(255*(1-w)), wd)

def highlight_word(wd, color):
    return '<span style="background-color: rgb(%d, %d, %d)">%s</span>' % (color[0], color[1], color[2], wd)


def highlight_seq(wd_seq, idxs, weights, pred_no):
    print(wd_seq)
    marked_seq = []
    for i, wd in enumerate(wd_seq):
        if i in idxs:
            w = int(255 * (1 - weights[i]))
            color = ()
            if pred_no == 0:
                color = (w, w, 255)
            elif pred_no == 1:
                color = (w, 255, w)
            elif pred_no == 2:
                color = (255, w, w)

            wd = highlight_word(wd, color)

        marked_seq.append(wd)

    return ''.join(marked_seq)


# def highlight_seq(wd_seq, idxs, weights, pred_no):
#     if not isinstance(wd_seq, np.ndarray):
#         wd_seq = np.asarray(wd_seq)
#
#     idxs = [i for i in idxs if i in range(len(wd_seq))]
#
#     # ws = map(lambda x: int(255*(1-x)), weights)
#     # mark_seq = np.choose(idxs, wd_seq)
#     mark_seq = wd_seq[idxs]
#
#     marked_seq = []
#     for w, wd in zip(weights, mark_seq):
#         w = int(255 * (1 - w))
#         color = ()
#         if pred_no == 0:
#             color = (w, w, 255)
#         elif pred_no == 1:
#             color = (w, 255, w)
#         elif pred_no == 2:
#             color = (255, w, w)
#
#         marked_wd = highlight_word(wd, color)
#
#         print('marked: ', marked_wd)
#         marked_seq.append(marked_wd)
#
#     print(marked_seq)
#     wd_seq[idxs] = marked_seq
#     print(wd_seq)
#
#     return ''.join(wd_seq)


def predict(pred_data, config):
    # att_lstm = Attention_LSTM_SA.Attention_LSTM_SA()
    # att_lstm.load_state_dict(torch.load(config.load_model_path))  # 加载模型参数

    # 加载模型
    att_lstm = torch.load(config.load_model_path)
    att_lstm.eval()  # self.training = False

    _, wd_to_idx = data.get_embedding_weights(config)
    insts = data.preprocess_data(pred_data)  # 单词序列
    wd_seqs = copy.deepcopy(insts)  # 深拷贝：完全拷贝了父对象及其子对象，两者是完全独立的
    id_seqs = data.data_to_index(insts, wd_to_idx)
    padded_data, seq_lens, origin_indices = data.pad_batch(id_seqs, config.max_len)
    print(seq_lens)
    input_data, _, _ = data.batch_data_variable(padded_data)

    out, weights = att_lstm(input_data, seq_lens)
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

    for inst, ws, pred_no in zip(wd_seqs, weights, pred.numpy()):
        _, top_idxs = torch.topk(ws, topn)
        hs = highlight_seq(inst.words, top_idxs.numpy(), ws.data.numpy(), pred_no)
        print(hs)
        results.append(hs)

    return results


if __name__ == '__main__':
    config = Config.Config('config/hyper_param.cfg')

    X = [
        '鞋子很快就收到了，质量过关，价格不贵，性价比 很高，鞋子穿得也比较舒服，尺码也标准就按平时 自己穿得码数买就行了！',
        '质量很好 版型也很好 码子很标准 穿上很有档次 卖家服务超级好 很满意的一次网上购物',
        '发过来的鞋子跟图片不是同一款 没有图片上的好看 鞋子的鞋面跟鞋带都不一样 只有鞋底一样 太坑了 并且物流不是一般的慢',
        '鞋子感觉一般吧，穿上不是特别舒服，这个价钱中规中矩吧。'
    ]
    predict(X, config)

