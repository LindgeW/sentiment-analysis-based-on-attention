import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rnn_encoder import RNNEncoder
'''
batch_first = False
LSTM输入: input, (h_0, c_0)
    - input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.lstm.pack_padded_sequence(input, lengths, batch_first=False[source])
    - h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
    - c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

LSTM输出 output, (h_n, c_n)
    - output (seq_len, batch, hidden_size * num_directions): 保存lstm最后一层的输出的Tensor。 如果输入是torch.nn.utils.lstm.PackedSequence，那么输出也是torch.nn.utils.lstm.PackedSequence。
    - h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的隐状态。
    - c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的细胞状态。


batch_first = True
LSTM输入: input, (h_0, c_0)
    - input (batch, seq_len, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.lstm.pack_padded_sequence(input, lengths, batch_first=False[source])
    - h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
    - c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

LSTM输出 output, (h_n, c_n)
    - output (batch, seq_len, hidden_size * num_directions): 保存lstm最后一层的输出的Tensor。 如果输入是torch.nn.utils.lstm.PackedSequence，那么输出也是torch.nn.utils.lstm.PackedSequence。
    - h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的隐状态。
    - c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的细胞状态。
'''


# att = W*tanh(V*X)
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.correlation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),       # nn.ReLU()
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, encoder_output):  # [batch_size, seq_len, hidden_size]
        a = self.correlation(encoder_output)
        # print(a.shape)  # [batch_size, seq_len, 1]
        weights = F.softmax(a.squeeze(-1), dim=1)  # squeeze:去掉a中指定的维数为1的维度
        # print(weights.shape)  # [batch_size, seq_len]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1] -> [batch_size, hidden_size]
        out = (encoder_output * weights.unsqueeze(-1)).sum(dim=1)
        # print(out.shape)  # [batch_size, hidden_size]
        # batch, seq_len
        out = torch.tanh(out)
        return out, weights


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    # att(query, keys, values) = softmax(sim(query, keys)) * values
    def forward(self, query, keys, values):
        # query: 最终隐层状态，keys、values：lstm层输出且keys == values
        # query: B × Q
        # keys: B × T × K
        # values: B × T × V
        scale = np.sqrt(query.shape[1])  # 调节因子
        # # 设Q=K(=hidden_size), [B, 1, Q] * [B, K, T] -> [B, 1, T]
        # att_weights = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        # soft_att_weights = F.softmax(att_weights.mul_(scale), dim=2)
        # # [B, 1, T] * [B, T, V] -> [B, 1, V] -> [B, V]
        # att_out = torch.bmm(att_weights, values).squeeze(1)

        # [B, T, K] * [B, Q, 1] -> [B, T, 1]
        att_weights = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        soft_att_weights = F.softmax(att_weights.mul_(scale), dim=1)  # [B, T]
        # [B, V, T] * [B, T, 1] -> [B, V, 1]
        att_out = torch.bmm(values.transpose(1, 2), soft_att_weights).squeeze(2)

        return att_out, soft_att_weights


class SentimentModel(nn.Module):
    def __init__(self, vocab, config, embedding_weights):
        super(SentimentModel, self).__init__()
        self.config = config

        embedding_dim = embedding_weights.shape[1]  # vocab_size * embedding_dim
        embed_init = torch.zeros((vocab.corpus_vocab_size, embedding_dim), dtype=torch.float32)
        self.corpus_embeddings = nn.Embedding(num_embeddings=vocab.corpus_vocab_size,
                                              embedding_dim=embedding_dim)
        self.corpus_embeddings.weight.data.copy_(embed_init)
        # self.corpus_embeddings.weight = nn.Parameter(embed_init)
        self.corpus_embeddings.weight.requires_grad = True  # 默认

        self.wd2vec_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))
        self.wd2vec_embeddings.weight.requires_grad = False

        self.bidirectional = True
        self.nb_directions = 2 if self.bidirectional else 1

        self.rnn_encoder = RNNEncoder(input_size=embedding_dim,  # 输入的特征维度
                                      hidden_size=self.config.hidden_size,  # 隐层状态的特征维度
                                      num_layers=self.config.nb_layers,  # LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果
                                      dropout=self.config.drop_rate,  # 除了最后一层外，其它层的输出都会套上一个dropout层
                                      bidirectional=self.bidirectional,  # 是否为双向LSTM
                                      batch_first=True)  # [batch_size, seq, feature]

        # self.self_attention = SelfAttention(self.nb_directions * self.config.hidden_size)
        self.self_attention = SelfAttention(self.config.hidden_size)
        self.dropout_embed = nn.Dropout(self.config.drop_embed_rate)
        self.dropout = nn.Dropout(self.config.drop_rate)
        # self.out = nn.Linear(self.nb_directions * self.config.hidden_size, vocab.tag_size)
        self.out = nn.Linear(self.config.hidden_size, vocab.tag_size)

    # Att = ht * h
    def _attention(self, encoder_output, hidden_n):
        # encoder_output: [batch_size, seq_len, hidden_size]
        # hidden_n: [batch_size, hidden_size]

        # [batch_size, seq_len, hidden_size] * [batch_size, hidden_size, 1]
        # -> [batch_size, seq_len, 1] -> [batch_size, seq_len]
        att_weights = torch.bmm(encoder_output, hidden_n.unsqueeze(2)).squeeze(2)
        soft_att_weights = F.softmax(att_weights, dim=1)  # [batch_size, seq_len]
        # [batch_size, hidden_size, seq_len] * [batch_size, seq_len, 1]
        # -> [batch_size, hidden_size, 1] -> [batch_size, hidden_size]
        out = torch.bmm(encoder_output.transpose(1, 2), soft_att_weights.unsqueeze(2)).squeeze(2)

        return out, soft_att_weights

    def forward(self, inputs, wd2vec_inputs, mask):  # (h0_state, c0_state)
        corpus_embed = self.corpus_embeddings(inputs)
        wd2vec_embed = self.wd2vec_embeddings(wd2vec_inputs)
        embed = corpus_embed + wd2vec_embed

        if self.training:  # 训练过程中采用Dropout，预测时False
            embed = self.dropout_embed(embed)

        # r_out: (batch, max_seq_len, hidden_size * num_directions)
        r_out, hidden = self.rnn_encoder(embed, mask)  # None 表示0初始化

        if self.bidirectional:  # 前向输出和反向输出相加
            r_out = r_out[:, :, :self.config.hidden_size] + r_out[:, :, self.config.hidden_size:]
            # fw, bw = r_out.split(2, dim=2)
            # r_out = fw + bw

        # r_out = torch.cat((r_out[-1], r_out[-2]), dim=1)

        out, weights = self.self_attention(r_out)

        if self.training:  # 训练过程中采用Dropout，预测时False
            out = self.dropout(out)

        out = self.out(out)

        # out = F.log_softmax(self.out(x), dim=1)

        return out, weights

