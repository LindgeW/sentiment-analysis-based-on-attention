import torch
import torch.nn as nn

'''
LSTMCell
输入: input, (h_0, c_0)
    input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable
    h_0 (batch, hidden_size): 保存着batch中每个元素的初始化隐状态的Tensor
    c_0 (batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出：h_1, c_1
    h_1 (batch, hidden_size): 下一个时刻的隐状态。
    c_1 (batch, hidden_size): 下一个时刻的细胞状态。


LSTM
输入: input, (h_0, c_0)
    input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])
    h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
    c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出: output, (h_n, c_n)
    output (seq_len, batch, hidden_size * num_directions): 保存RNN最后一层的输出的Tensor。 如果输入是torch.nn.utils.rnn.PackedSequence，那么输出也是torch.nn.utils.rnn.PackedSequence。
    h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的隐状态。
    c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的细胞状态。
'''


class RNNEncoder(nn.Module):
    def __init__(self, input_size=0, hidden_size=0, num_layers=1, batch_first=False, bidirectional=False, dropout=0.0, rnn_type='lstm'):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_directions = 2 if self.bidirectional else 1

        self._rnn_types = ['RNN', 'LSTM', 'GRU']
        self.rnn_type = rnn_type.upper()
        assert self.rnn_type in self._rnn_types
        # 获取torch.nn对象中相应的的构造函数
        self._rnn_cell = getattr(nn, self.rnn_type+'Cell')  # getattr获取对象的属性或者方法

        # ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module
        # 当添加 nn.ModuleList作为nn.Module对象的一个成员时（即当我们添加模块到我们的网络时），
        # 所有nn.ModuleList内部的nn.Module的parameter也被添加作为我们的网络的parameter
        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList()
        for layer_i in range(self.num_layers):
            layer_input_size = self.input_size if layer_i == 0 else self.num_directions * self.hidden_size
            self.fw_cells.append(self._rnn_cell(input_size=layer_input_size, hidden_size=self.hidden_size))
            if self.bidirectional:
                self.bw_cells.append(self._rnn_cell(input_size=layer_input_size, hidden_size=self.hidden_size))

        # self.cell = nn.LSTMCell(
        # 	input_size=self.input_size,   # 输入的特征维度
        # 	hidden_size=self.hidden_size  # 隐层的维度
        # )

    def init_hidden(self, batch_size, retain=True, device=torch.device('cpu')):
        if retain:  # 是否保证每轮迭代都初始化隐层
            torch.manual_seed(3357)
        # hidden = torch.randn(batch_size, self.hidden_size, device=device)
        # hidden = torch.rand(batch_size, self.hidden_size, device=device)
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        if self.rnn_type == 'LSTM':
            hidden = (hidden, hidden)
        return hidden

    def _forward_mask(self, cell, inputs, mask, init_hidden, drop_mask=None):
        out_fw = []
        seq_len = inputs.size(0)  # seq_len * batch_size * input_size
        hx_fw = init_hidden
        # print('cell: ', next(cell.parameters()).is_cuda)
        for xi in range(seq_len):
            hidden = cell(input=inputs[xi], hx=hx_fw)
            if self.rnn_type == 'LSTM':
                h_next, c_next = hidden
                h_next = h_next * mask[xi] + init_hidden[0] * (1 - mask[xi])
                c_next = c_next * mask[xi] + init_hidden[1] * (1 - mask[xi])
                out_fw.append(h_next)
                if drop_mask is not None:  # 循环层使用dropout
                    h_next = h_next * drop_mask
                hx_next = (h_next, c_next)
            else:
                h_next = hidden
                h_next = h_next * mask[xi] + init_hidden * (1 - mask[xi])
                out_fw.append(h_next)
                if drop_mask is not None:  # 循环层使用dropout
                    h_next = h_next * drop_mask
                hx_next = h_next

            hx_fw = hx_next

        out_fw = torch.stack(tuple(out_fw), dim=0)
        return out_fw, hx_fw

    def _backward_mask(self, cell, inputs, mask, init_hidden, drop_mask=None):
        out_bw = []
        seq_len = inputs.size(0)
        hx_bw = init_hidden

        for xi in reversed(range(seq_len)):
            hidden = cell(input=inputs[xi], hx=hx_bw)
            if self.rnn_type == 'LSTM':
                h_next, c_next = hidden
                h_next = h_next * mask[xi] + init_hidden[0] * (1 - mask[xi])
                c_next = c_next * mask[xi] + init_hidden[1] * (1 - mask[xi])
                out_bw.append(h_next)

                if drop_mask is not None:  # 循环层使用dropout
                    h_next = h_next * drop_mask

                hx_next = (h_next, c_next)
            else:
                h_next = hidden
                h_next = h_next * mask[xi] + init_hidden * (1 - mask[xi])
                out_bw.append(h_next)

                if drop_mask is not None:  # 循环层使用dropout
                    h_next = h_next * drop_mask

                hx_next = h_next

            hx_bw = hx_next

        out_bw.reverse()
        out_bw = torch.stack(tuple(out_bw), dim=0)
        return out_bw, hx_bw

    def forward(self, inputs, mask, init_hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
            mask = mask.transpose(0, 1)

        mask = mask.float().unsqueeze(-1).expand((-1, -1, self.hidden_size))

        batch_size = inputs.size(1)
        if init_hidden is None:
            init_hidden = self.init_hidden(batch_size, device=inputs.device)
            # init_hidden = inputs.data.new(batch_size, self.hidden_size).zero_()
            # if self.rnn_type == 'LSTM':
            # 	init_hidden = (init_hidden, init_hidden)

        hx = init_hidden

        hn, cn = [], []
        for layer in range(self.num_layers):
            input_drop_mask, hidden_drop_mask = None, None
            seq_len, batch_size, input_size = inputs.size()
            if self.training:
                # print('use dropout...')
                if layer != 0:
                    input_drop_mask = torch.zeros(batch_size, input_size, device=inputs.device).fill_(1 - self.dropout)
                    # 在相同的设备上创建一个和inputs数据类型相同的tensor
                    # input_drop_mask = inputs.data.new(batch_size, input_size).fill_(1 - self.dropout)
                    input_drop_mask = torch.bernoulli(input_drop_mask)
                    input_drop_mask = torch.div(input_drop_mask, (1 - self.dropout))
                    input_drop_mask = input_drop_mask.unsqueeze(-1).expand((-1, -1, seq_len)).permute((2, 0, 1))
                    inputs = inputs * input_drop_mask

                hidden_drop_mask = torch.zeros(batch_size, self.hidden_size, device=inputs.device).fill_(1 - self.dropout)
                # hidden_drop_mask = inputs.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout)
                hidden_drop_mask = torch.bernoulli(hidden_drop_mask)     # 以输入值为概率p输出1，(1-p)输出0
                hidden_drop_mask = torch.div(hidden_drop_mask, (1 - self.dropout))  # 保证训练和预测时期望值一致

            # print('data is in cuda: ', inputs.device, mask.device, hx.device, hidden_drop_mask.device)
            out_fw, (hn_f, cn_f) = self._forward_mask(cell=self.fw_cells[layer], inputs=inputs, mask=mask, init_hidden=hx, drop_mask=hidden_drop_mask)
            # print(out_fw.shape, hn_f.shape, cn_f.shape)

            out_bw, hn_b, cn_b = None, None, None
            if self.bidirectional:
                out_bw, (hn_b, cn_b) = self._backward_mask(cell=self.bw_cells[layer], inputs=inputs, mask=mask, init_hidden=hx, drop_mask=hidden_drop_mask)
            # print(out_bw.shape, hn_b.shape, cn_b.shape)
            hn.append(torch.cat((hn_f, hn_b), dim=1) if self.bidirectional else hn_f)
            cn.append(torch.cat((cn_f, cn_b), dim=1) if self.bidirectional else cn_f)
            inputs = torch.cat((out_fw, out_bw), dim=2) if self.bidirectional else out_fw
            # print('input shape:', inputs.shape)   # (6, 3, 10)

        hn = torch.stack(tuple(hn), dim=0)
        cn = torch.stack(tuple(cn), dim=0)

        output = inputs.transpose(0, 1) if self.batch_first else inputs

        return output, (hn, cn)
