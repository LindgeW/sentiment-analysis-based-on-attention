import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

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


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.correlation = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),       # nn.Tanh()
            nn.Linear(64, 1)
        )

    def forward(self, encoder_output):  # [batch_size, seq_len, hidden_size]
        a = self.correlation(encoder_output)
        # print(a.shape)  # [batch_size, seq_len, 1]
        weights = F.softmax(a.squeeze(-1), dim=1)  # 去掉a中指定的维数为1的维度
        # print(weights.shape)  # [batch_size, seq_len]
        out = (encoder_output * weights.unsqueeze(-1)).sum(dim=1)
        # print(out.shape)  # [batch_size, hidden_size]
        return out, weights


class Attention_LSTM(nn.Module):
    def __init__(self, vocab, config, embedding_weights, bidirectional=True):
        super(Attention_LSTM, self).__init__()
        self.config = config

        embedding_dim = embedding_weights.shape[1]
        embed_init = torch.zeros((vocab.corpus_vocab_size, embedding_dim), dtype=torch.float32)
        self.corpus_embeddings = nn.Embedding(num_embeddings=vocab.corpus_vocab_size, embedding_dim=embedding_dim)
        self.corpus_embeddings.weight.data.copy_(embed_init)
        self.corpus_embeddings.weight.requires_grad = True
        # self.corpus_embeddings.weight = nn.Parameter(embed_init)

        self.wd2vec_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))
        self.wd2vec_embeddings.weight.requires_grad = False
        # print(self.wd2vec_embeddings.weight.shape)  # [22667, 300]

        self.bidirectional = bidirectional
        self.nb_directions = 2 if bidirectional else 1
        self.lstm_dropout = self.config.drop_rate if self.config.nb_layers > 1 else 0

        self.lstm = nn.LSTM(input_size=self.config.embedding_size,  # 输入的特征维度
                            hidden_size=self.config.hidden_size,  # 隐层状态的特征维度
                            num_layers=self.config.nb_layers,  # LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果
                            dropout=self.lstm_dropout,  # 除了最后一层外，其它层的输出都会套上一个dropout层
                            bidirectional=self.bidirectional,  # 是否为双向LSTM
                            batch_first=True)  # [batch_size, seq, feature]

        # self.self_attention = SelfAttention(self.nb_directions * self.config.hidden_size)
        self.self_attention = SelfAttention(self.config.hidden_size)
        self.dropout_embed = nn.Dropout(self.config.drop_embed_rate)
        self.dropout = nn.Dropout(self.config.drop_rate)
        # self.out = nn.Linear(self.nb_directions * self.config.hidden_size, self.config.nb_class)
        self.out = nn.Linear(self.config.hidden_size, vocab.tag_size)

    def init_hidden(self, batch_size=64):
        torch.manual_seed(3347)
        h_0 = torch.randn((self.config.nb_layers * self.nb_directions, batch_size, self.config.hidden_size))
        c_0 = torch.randn((self.config.nb_layers * self.nb_directions, batch_size, self.config.hidden_size))
        return h_0, c_0

    def forward(self, inputs, wd2vec_inputs, seq_lens):  # (h0_state, c0_state)
        batch_size = inputs.shape[0]

        init_hidden = self.init_hidden(batch_size)
        # print(inputs.shape)  # [64, 100]

        corpus_embed = self.corpus_embeddings(inputs)
        wd2vec_embed = self.wd2vec_embeddings(wd2vec_inputs)
        # print(wd2vec.shape)  # [64, 100, 300]
        embed = corpus_embed + wd2vec_embed

        if self.training:  # 训练过程中采用Dropout，预测时False
            embed = self.dropout_embed(embed)

        # 使用pack_padded_sequence来确保LSTM模型不会处理用于填充的元素
        packed_embed = nn_utils.rnn.pack_padded_sequence(embed, seq_lens, batch_first=True)
        # 保存着lstm最后一层的输出特征和最后一个时刻隐状态
        r_out, hidden = self.lstm(packed_embed, init_hidden)  # None 表示0初始化
        r_out, _ = nn_utils.rnn.pad_packed_sequence(r_out, batch_first=True)

        if self.bidirectional:
            r_out = r_out[:, :, :self.config.hidden_size] + r_out[:, :, self.config.hidden_size:]

        out, weights = self.self_attention(r_out)
        # print(out.shape)  # [64, 128]
        # print(weights.shape)  # [64, 100]

        if self.training:  # 训练过程中采用Dropout，预测时False
            out = self.dropout(out)
        out = self.out(out)
        # print(out.shape)    # [64, 3]
        # out = F.log_softmax(self.out(x), dim=1)

        return out, weights

